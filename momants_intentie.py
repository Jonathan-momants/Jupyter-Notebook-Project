"""Train en gebruik een SetFit-classifier voor Momants-gespreksintenties."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

from datasets import Dataset
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments

from momants_sentiment import laad_momants_csv


BASIS_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TRAININGSDATA_PAD = Path("intentie_training.csv")
LOKAAL_MODEL_PAD = Path("model/momants-intentie")
ZEKERHEIDSDREMPEL = 0.60

INTENTIECATEGORIEEN = [
    "Informatie opvragen",
    "Probleem of Incident oplossen",
    "Transactie / Mutatie uitvoeren",
    "Actievere Navigatiehulp",
    "Systeem bedienen",
    "Noodgeval melden",
]

UITVOERKOLOMMEN = [
    "conversation_id",
    "intentie",
    "zekerheid",
    "eerst_gedetecteerd_op",
]

KALE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)


def _valideer_trainingsdata(data: pd.DataFrame) -> pd.DataFrame:
    """Controleer het trainingsschema en de zes toegestane intentielabels."""
    vereiste_kolommen = {"tekst", "intentie"}
    ontbrekend = vereiste_kolommen - set(data.columns)
    if ontbrekend:
        raise ValueError(
            "De trainings-CSV mist kolommen: "
            f"{', '.join(sorted(ontbrekend))}."
        )

    schoon = data[["tekst", "intentie"]].copy()
    if schoon.isna().any().any():
        raise ValueError(
            "De trainingsdata bevat ontbrekende tekst- of intentiewaarden."
        )
    schoon["tekst"] = schoon["tekst"].astype(str).str.strip()
    schoon["intentie"] = schoon["intentie"].astype(str).str.strip()
    if schoon["tekst"].eq("").any() or schoon["intentie"].eq("").any():
        raise ValueError(
            "De trainingsdata bevat lege tekst- of intentiewaarden."
        )

    onbekend = set(schoon["intentie"]) - set(INTENTIECATEGORIEEN)
    if onbekend:
        raise ValueError(
            "Onbekende intentielabels in de trainingsdata: "
            f"{', '.join(sorted(onbekend))}."
        )

    ontbrekende_labels = set(INTENTIECATEGORIEEN) - set(schoon["intentie"])
    if ontbrekende_labels:
        raise ValueError(
            "Trainingsvoorbeelden ontbreken voor: "
            f"{', '.join(sorted(ontbrekende_labels))}."
        )
    return schoon


def train_model(
    trainingsdata_pad: str | Path = TRAININGSDATA_PAD,
    model_pad: str | Path = LOKAAL_MODEL_PAD,
) -> Path:
    """Train SetFit op handgeschreven voorbeelden en sla het model lokaal op."""
    bron = Path(trainingsdata_pad).expanduser()
    if not bron.is_file():
        raise FileNotFoundError(f"Trainingsbestand niet gevonden: {bron}")

    training = _valideer_trainingsdata(pd.read_csv(bron))
    label_naar_index = {
        label: index for index, label in enumerate(INTENTIECATEGORIEEN)
    }
    dataset = Dataset.from_dict(
        {
            "text": training["tekst"].tolist(),
            "label": training["intentie"].map(label_naar_index).tolist(),
        }
    )

    model = SetFitModel.from_pretrained(
        BASIS_MODEL_ID,
        labels=INTENTIECATEGORIEEN,
    )
    argumenten = TrainingArguments(
        output_dir="model/checkpoints-intentie",
        batch_size=16,
        num_epochs=(1, 16),
        report_to="none",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=argumenten,
        train_dataset=dataset,
    )
    trainer.train()

    doel = Path(model_pad).expanduser()
    doel.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(doel)
    return doel


def _is_bruikbaar_bezoekersbericht(rij: pd.Series) -> bool:
    if bool(rij["from_agent"]) or pd.isna(rij["text"]):
        return False
    tekst = str(rij["text"]).strip()
    return bool(tekst) and KALE_URL.fullmatch(tekst) is None


def selecteer_bezoekersberichten(data: pd.DataFrame) -> pd.DataFrame:
    """Selecteer uitsluitend bruikbare bezoekersberichten in tijdsvolgorde."""
    selectie = data.loc[
        data.apply(_is_bruikbaar_bezoekersbericht, axis=1)
    ].copy()
    selectie["text"] = selectie["text"].astype(str).str.strip()
    return selectie.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)


def _modellabels(model: SetFitModel) -> list[str]:
    labels = [str(label) for label in model.labels]
    if set(labels) != set(INTENTIECATEGORIEEN):
        raise ValueError(
            "Het lokale model bevat niet exact de zes vereiste intentielabels."
        )
    return labels


def classificeer_berichten(
    bezoekersberichten: pd.DataFrame,
    model_pad: str | Path = LOKAAL_MODEL_PAD,
    batchgrootte: int = 32,
    model: SetFitModel | None = None,
) -> pd.DataFrame:
    """Classificeer ieder bruikbaar bezoekersbericht afzonderlijk."""
    if bezoekersberichten.empty:
        leeg = bezoekersberichten.copy()
        leeg["intentie"] = pd.Series(dtype="object")
        leeg["zekerheid"] = pd.Series(dtype="float64")
        return leeg

    if model is None:
        pad = Path(model_pad).expanduser()
        if not pad.is_dir():
            raise FileNotFoundError(
                f"Getraind intentiemodel niet gevonden in {pad}. "
                "Draai eerst: python momants_intentie.py --train"
            )
        model = SetFitModel.from_pretrained(pad, local_files_only=True)

    labels = _modellabels(model)
    kansen = model.predict_proba(
        bezoekersberichten["text"].tolist(),
        batch_size=batchgrootte,
        as_numpy=True,
        show_progress_bar=True,
    )
    beste_indices = kansen.argmax(axis=1)
    beste_scores = kansen.max(axis=1)

    resultaat = bezoekersberichten.copy()
    resultaat["intentie"] = [labels[index] for index in beste_indices]
    resultaat["zekerheid"] = beste_scores
    return resultaat


def maak_intentieoverzicht(
    geclassificeerd: pd.DataFrame,
    zekerheidsdrempel: float = ZEKERHEIDSDREMPEL,
) -> pd.DataFrame:
    """Bewaar de eerste detectie per unieke gesprek-intentie-combinatie."""
    if geclassificeerd.empty:
        return pd.DataFrame(columns=UITVOERKOLOMMEN)

    boven_drempel = geclassificeerd.loc[
        geclassificeerd["zekerheid"].gt(zekerheidsdrempel)
    ].copy()
    if boven_drempel.empty:
        return pd.DataFrame(columns=UITVOERKOLOMMEN)

    eerste = (
        boven_drempel.sort_values(
            ["conversation_id", "created_at"], kind="stable"
        )
        .drop_duplicates(["conversation_id", "intentie"], keep="first")
        .rename(columns={"created_at": "eerst_gedetecteerd_op"})
    )
    eerste["eerst_gedetecteerd_op"] = eerste[
        "eerst_gedetecteerd_op"
    ].apply(lambda waarde: waarde.isoformat())
    eerste["zekerheid"] = eerste["zekerheid"].round(4)
    return eerste[UITVOERKOLOMMEN].reset_index(drop=True)


def verwerk_csv(
    csv_pad: str | Path,
    uitvoermap: str | Path = "resultaten",
    model_pad: str | Path = LOKAAL_MODEL_PAD,
    batchgrootte: int = 32,
) -> pd.DataFrame:
    """Classificeer een Momants-export en schrijf de gesprek-intenties."""
    data = laad_momants_csv(csv_pad)
    bezoekersberichten = selecteer_bezoekersberichten(data)
    geclassificeerd = classificeer_berichten(
        bezoekersberichten,
        model_pad=model_pad,
        batchgrootte=batchgrootte,
    )
    overzicht = maak_intentieoverzicht(geclassificeerd)

    doelmap = Path(uitvoermap).expanduser()
    doelmap.mkdir(parents=True, exist_ok=True)
    overzicht.to_csv(doelmap / "intenties_per_gesprek.csv", index=False)
    return overzicht


def _maak_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train of gebruik de Momants-intentieclassifier."
    )
    parser.add_argument(
        "csv_pad",
        type=Path,
        nargs="?",
        help="Pad naar de lokale Momants CSV-export.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train het model eenmalig op intentie_training.csv.",
    )
    parser.add_argument(
        "--trainingsdata",
        type=Path,
        default=TRAININGSDATA_PAD,
        help="CSV met de kolommen tekst en intentie.",
    )
    parser.add_argument(
        "--model-pad",
        type=Path,
        default=LOKAAL_MODEL_PAD,
        help="Map waarin het getrainde model staat.",
    )
    parser.add_argument(
        "--uitvoermap",
        type=Path,
        default=Path("resultaten"),
        help="Map voor intenties_per_gesprek.csv.",
    )
    parser.add_argument(
        "--batchgrootte",
        type=int,
        default=32,
        help="Aantal bezoekersberichten per modelbatch.",
    )
    parser.add_argument(
        "--alleen-controleren",
        action="store_true",
        help="Controleer alleen het inladen, zonder het model te laden.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    argumenten = _maak_parser().parse_args(argv)

    if argumenten.train:
        model_pad = train_model(
            trainingsdata_pad=argumenten.trainingsdata,
            model_pad=argumenten.model_pad,
        )
        print(f"Intentiemodel opgeslagen in: {model_pad.resolve()}")
        return 0

    if argumenten.csv_pad is None:
        raise SystemExit("Geef een CSV-pad op, of gebruik --train.")

    data = laad_momants_csv(argumenten.csv_pad)
    bezoekers = selecteer_bezoekersberichten(data)
    if argumenten.alleen_controleren:
        print(f"Berichtrijen ingelezen: {len(data)}")
        print(f"Bruikbare bezoekersberichten: {len(bezoekers)}")
        print(f"Gesprekken: {bezoekers['conversation_id'].nunique()}")
        return 0

    geclassificeerd = classificeer_berichten(
        bezoekers,
        model_pad=argumenten.model_pad,
        batchgrootte=argumenten.batchgrootte,
    )
    overzicht = maak_intentieoverzicht(geclassificeerd)
    argumenten.uitvoermap.mkdir(parents=True, exist_ok=True)
    uitvoerpad = argumenten.uitvoermap / "intenties_per_gesprek.csv"
    overzicht.to_csv(uitvoerpad, index=False)
    print(f"Gesprek-intenties: {len(overzicht)}")
    print(f"Uitvoer geschreven naar: {uitvoerpad.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())