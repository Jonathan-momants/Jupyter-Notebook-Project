"""Bepaal per Momants-gesprek of vragen van bezoekers zijn beantwoord."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from momants_sentiment import laad_momants_csv


MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
HYPOTHESE = "Dit antwoordt op de vraag van de bezoeker."
ENTAILMENT_DREMPEL = 0.50
# Tijdelijk: zet op False om weer uitsluitend de privacy-arme samenvatting te schrijven.
VOEG_GESPREKSTEKST_TOE = True

VRAAGSTARTEN = (
    "wie",
    "wat",
    "waar",
    "wanneer",
    "hoe",
    "waarom",
    "welke",
    "mag",
    "kan",
    "is er",
    "zijn er",
)

BASIS_UITVOERKOLOMMEN = [
    "conversation_id",
    "aantal_vragen",
    "aantal_beantwoord",
    "percentage_beantwoord",
    "eindoordeel",
    "uitleg",
]
UITVOERKOLOMMEN = [
    *BASIS_UITVOERKOLOMMEN,
    *(["gesprekstekst"] if VOEG_GESPREKSTEKST_TOE else []),
]

KALE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
VRAAGSTARTPATROON = re.compile(
    r"^(?:" + "|".join(re.escape(woord) for woord in VRAAGSTARTEN) + r")\b",
    flags=re.IGNORECASE,
)


def is_vraag(tekst: object) -> bool:
    """Herken een niet-lege vraag aan een vraagteken of een vraagstart."""
    if pd.isna(tekst):
        return False
    schoon = str(tekst).strip()
    if not schoon or KALE_URL.fullmatch(schoon):
        return False
    return "?" in schoon or VRAAGSTARTPATROON.match(schoon) is not None


def detecteer_vragen(data: pd.DataFrame) -> pd.DataFrame:
    """Selecteer bruikbare bezoekersvragen in chronologische volgorde."""
    bezoekers = data.loc[data["from_agent"].eq(False)].copy()
    bezoekers = bezoekers.loc[bezoekers["text"].apply(is_vraag)].copy()
    bezoekers["text"] = bezoekers["text"].astype(str).str.strip()
    return bezoekers.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)


def koppel_vragen_aan_antwoorden(data: pd.DataFrame) -> pd.DataFrame:
    """Koppel iedere bezoekersvraag aan het eerstvolgende agentbericht."""
    kolommen = [
        "conversation_id",
        "vraag_text",
        "antwoord_text",
        "heeft_agentantwoord",
    ]
    paren: list[dict[str, object]] = []

    gesorteerd = data.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    )
    for conversation_id, gesprek in gesorteerd.groupby(
        "conversation_id", sort=False
    ):
        regels = list(gesprek.itertuples(index=False))
        for positie, bericht in enumerate(regels):
            if bool(bericht.from_agent) or not is_vraag(bericht.text):
                continue

            antwoord_text: str | None = None
            for volgend in regels[positie + 1 :]:
                if volgend.created_at <= bericht.created_at:
                    continue
                if bool(volgend.from_agent):
                    if pd.notna(volgend.text) and str(volgend.text).strip():
                        antwoord_text = str(volgend.text).strip()
                    break

            paren.append(
                {
                    "conversation_id": conversation_id,
                    "vraag_text": str(bericht.text).strip(),
                    "antwoord_text": antwoord_text,
                    "heeft_agentantwoord": antwoord_text is not None,
                }
            )

    return pd.DataFrame(paren, columns=kolommen)


def _labelindices(model: AutoModelForSequenceClassification) -> dict[str, int]:
    """Vind de drie NLI-labelindices in de modelconfiguratie."""
    labels = {
        str(label).strip().lower(): int(index)
        for index, label in model.config.id2label.items()
    }
    ontbrekend = {"entailment", "neutral", "contradiction"} - set(labels)
    if ontbrekend:
        raise ValueError(
            "De modelconfiguratie mist NLI-labels: "
            f"{', '.join(sorted(ontbrekend))}."
        )
    return labels


def classificeer_vraag_antwoordparen(
    paren: pd.DataFrame,
    batchgrootte: int = 32,
    tokenizer: AutoTokenizer | None = None,
    model: AutoModelForSequenceClassification | None = None,
) -> pd.DataFrame:
    """Bereken NLI-scores voor paren die daadwerkelijk een agentantwoord hebben."""
    resultaat = paren.copy()
    resultaat["beantwoord"] = False
    resultaat["entailment_score"] = pd.Series(index=resultaat.index, dtype="float64")

    te_classificeren = resultaat.index[resultaat["heeft_agentantwoord"]]
    if te_classificeren.empty:
        return resultaat

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(MODEL_ID)
    model = model or AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    indices = _labelindices(model)

    for start in range(0, len(te_classificeren), batchgrootte):
        batchindices = te_classificeren[start : start + batchgrootte]
        antwoorden = resultaat.loc[batchindices, "antwoord_text"].tolist()
        hypotheses = [HYPOTHESE] * len(antwoorden)
        invoer = tokenizer(
            antwoorden,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            kansen = torch.softmax(model(**invoer).logits, dim=-1).cpu()

        entailment = kansen[:, indices["entailment"]]
        neutral = kansen[:, indices["neutral"]]
        contradiction = kansen[:, indices["contradiction"]]
        beantwoord = (
            entailment.ge(ENTAILMENT_DREMPEL)
            & entailment.gt(neutral)
            & entailment.gt(contradiction)
        )
        resultaat.loc[batchindices, "entailment_score"] = entailment.tolist()
        resultaat.loc[batchindices, "beantwoord"] = beantwoord.tolist()

    resultaat["entailment_score"] = resultaat["entailment_score"].round(4)
    return resultaat


def _eindoordeel(aantal_vragen: int, aantal_beantwoord: int) -> str:
    if aantal_vragen == 0:
        return "Geen vragen gevonden"
    if aantal_beantwoord == aantal_vragen:
        return "Beantwoord"
    if aantal_beantwoord == 0:
        return "Niet beantwoord"
    return "Deels beantwoord"


def _maak_gespreksteksten(data: pd.DataFrame) -> pd.Series:
    """Combineer tijdelijk bezoeker- en agenttekst per gesprek in tijdsvolgorde."""
    gesorteerd = data.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).copy()
    heeft_tekst = gesorteerd["text"].notna() & gesorteerd["text"].astype(str).str.strip().ne("")
    gesorteerd = gesorteerd.loc[heeft_tekst].copy()
    gesorteerd["regel"] = (
        gesorteerd["from_agent"]
        .map({True: "Agent", False: "Bezoeker"})
        .fillna("Onbekend")
        + ": "
        + gesorteerd["text"].astype(str).str.strip()
    )
    return gesorteerd.groupby("conversation_id", sort=False)["regel"].agg("\n".join)


def maak_gespreksoverzicht(
    data: pd.DataFrame,
    batchgrootte: int = 32,
    tokenizer: AutoTokenizer | None = None,
    model: AutoModelForSequenceClassification | None = None,
) -> pd.DataFrame:
    """Maak één antwoordstatus per gesprek, inclusief gesprekken zonder vragen."""
    gesprekken = pd.Index(
        data["conversation_id"].drop_duplicates(), name="conversation_id"
    )
    paren = koppel_vragen_aan_antwoorden(data)
    beoordeeld = classificeer_vraag_antwoordparen(
        paren,
        batchgrootte=batchgrootte,
        tokenizer=tokenizer,
        model=model,
    )

    aantallen = beoordeeld.groupby("conversation_id", sort=False).agg(
        aantal_vragen=("conversation_id", "size"),
        aantal_beantwoord=("beantwoord", "sum"),
    )
    overzicht = aantallen.reindex(gesprekken, fill_value=0).reset_index()
    overzicht["aantal_vragen"] = overzicht["aantal_vragen"].astype(int)
    overzicht["aantal_beantwoord"] = overzicht["aantal_beantwoord"].astype(int)
    overzicht["percentage_beantwoord"] = (
        overzicht["aantal_beantwoord"]
        .div(overzicht["aantal_vragen"].where(overzicht["aantal_vragen"].ne(0)))
        .mul(100)
        .fillna(0)
        .round(1)
    )
    overzicht["eindoordeel"] = overzicht.apply(
        lambda rij: _eindoordeel(
            int(rij["aantal_vragen"]), int(rij["aantal_beantwoord"])
        ),
        axis=1,
    )
    overzicht["uitleg"] = overzicht.apply(
        lambda rij: (
            "Er zijn geen vragen van de bezoeker gevonden."
            if rij["aantal_vragen"] == 0
            else (
                f"{int(rij['aantal_beantwoord'])} van de "
                f"{int(rij['aantal_vragen'])} "
                f"{'vraag is' if rij['aantal_vragen'] == 1 else 'vragen zijn'} "
                "beantwoord."
            )
        ),
        axis=1,
    )
    if VOEG_GESPREKSTEKST_TOE:
        gespreksteksten = _maak_gespreksteksten(data)
        overzicht["gesprekstekst"] = (
            overzicht["conversation_id"].map(gespreksteksten).fillna("")
        )
    return overzicht[UITVOERKOLOMMEN]


def verwerk_csv(
    csv_pad: str | Path,
    uitvoermap: str | Path = "resultaten",
    batchgrootte: int = 32,
) -> pd.DataFrame:
    """Verwerk een lokale export en schrijf één veilige gesprekstabel."""
    data = laad_momants_csv(csv_pad)
    overzicht = maak_gespreksoverzicht(data, batchgrootte=batchgrootte)
    map_pad = Path(uitvoermap).expanduser()
    map_pad.mkdir(parents=True, exist_ok=True)
    overzicht.to_csv(map_pad / "antwoordcheck_per_gesprek.csv", index=False)
    return overzicht


def _maak_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controleer per Momants-gesprek of bezoekersvragen zijn beantwoord."
    )
    parser.add_argument("csv_pad", type=Path, help="Pad naar de Momants CSV-export.")
    parser.add_argument(
        "--uitvoermap",
        type=Path,
        default=Path("resultaten"),
        help="Map voor antwoordcheck_per_gesprek.csv.",
    )
    parser.add_argument(
        "--batchgrootte",
        type=int,
        default=32,
        help="Aantal vraag-antwoordparen per modelbatch.",
    )
    parser.add_argument(
        "--alleen-controleren",
        action="store_true",
        help="Controleer inladen en vraagdetectie zonder het model te starten.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    argumenten = _maak_parser().parse_args(argv)
    data = laad_momants_csv(argumenten.csv_pad)

    if argumenten.alleen_controleren:
        vragen = detecteer_vragen(data)
        print(f"Berichtrijen ingelezen: {len(data)}")
        print(f"Vragen gevonden: {len(vragen)}")
        print(f"Gesprekken met vragen: {vragen['conversation_id'].nunique()}")
        return 0

    overzicht = maak_gespreksoverzicht(
        data,
        batchgrootte=argumenten.batchgrootte,
    )
    argumenten.uitvoermap.mkdir(parents=True, exist_ok=True)
    uitvoerpad = argumenten.uitvoermap / "antwoordcheck_per_gesprek.csv"
    overzicht.to_csv(uitvoerpad, index=False)
    print(f"Gesprekresultaten: {len(overzicht)}")
    print(f"Uitvoer geschreven naar: {uitvoerpad.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())