"""Verwerk een Momants CSV-export tot sentiment per bericht en per gesprek."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
from typing import Iterable

import pandas as pd
from transformers import pipeline


MODEL_ID = "tabularisai/multilingual-sentiment-analysis"

LABEL_MAPPING = {
    "Very Positive": "Positief",
    "Positive": "Positief",
    "Neutral": "Neutraal (taakgericht)",
    "Negative": "Negatief (gefrustreerd)",
    "Very Negative": "Boos (paniek)",
}

VEILIGE_KOLOMMEN = [
    "created_at",
    "text",
    "from_agent",
    "message_type",
    "conversation_id",
    "agent_id",
]

VERPLICHTE_KOLOMMEN = {
    "created_at",
    "text",
    "from_agent",
    "conversation_id",
}

# Posities in het headerloze Momants-exportformaat uit het aangeleverde voorbeeld.
# De privacygevoelige velden chat_sender en raw_json worden bewust niet geselecteerd.
HEADERLOZE_POSITIES = {
    0: "created_at",
    3: "text",
    8: "from_agent",
    10: "message_type",
    18: "conversation_id",
    19: "agent_id",
}

KOLOM_ALIASES = {
    "createdat": "created_at",
    "created": "created_at",
    "timestamp": "created_at",
    "text": "text",
    "message": "text",
    "message_text": "text",
    "fromagent": "from_agent",
    "is_agent": "from_agent",
    "messagetype": "message_type",
    "type": "message_type",
    "conversationid": "conversation_id",
    "conversation": "conversation_id",
    "agentid": "agent_id",
}

KALE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)


def _normaliseer_kolomnaam(naam: object) -> str:
    """Maak een kolomnaam geschikt voor vergelijking met bekende namen."""
    tekst = str(naam).strip().lower().replace("-", "_").replace(" ", "_")
    compact = tekst.replace("_", "")
    return KOLOM_ALIASES.get(tekst, KOLOM_ALIASES.get(compact, tekst))


def _heeft_momants_header(csv_pad: Path) -> bool:
    """Controleer of de eerste CSV-regel herkenbare Momants-kolomnamen bevat."""
    kolommen = pd.read_csv(csv_pad, nrows=0).columns
    genormaliseerd = {_normaliseer_kolomnaam(kolom) for kolom in kolommen}
    return VERPLICHTE_KOLOMMEN.issubset(genormaliseerd)


def _lees_met_header(csv_pad: Path) -> pd.DataFrame:
    """Lees alleen veilige velden uit een CSV met kolomkoppen."""
    originele_kolommen = pd.read_csv(csv_pad, nrows=0).columns
    mapping = {
        kolom: _normaliseer_kolomnaam(kolom)
        for kolom in originele_kolommen
        if _normaliseer_kolomnaam(kolom) in VEILIGE_KOLOMMEN
    }
    data = pd.read_csv(
        csv_pad,
        usecols=list(mapping),
        on_bad_lines="skip",
    )
    return data.rename(columns=mapping)


def _lees_zonder_header(csv_pad: Path) -> pd.DataFrame:
    """Lees veilige kolomposities uit het bekende 22-veldenformaat."""
    csv.field_size_limit(sys.maxsize)
    veilige_records: list[dict[str, object]] = []
    hoogste_positie = max(HEADERLOZE_POSITIES)

    with csv_pad.open(newline="", encoding="utf-8-sig") as csv_bestand:
        for record in csv.reader(csv_bestand):
            if len(record) <= hoogste_positie:
                continue
            veilige_records.append(
                {
                    kolomnaam: record[positie]
                    for positie, kolomnaam in HEADERLOZE_POSITIES.items()
                }
            )

    return pd.DataFrame(veilige_records, columns=VEILIGE_KOLOMMEN)


def laad_momants_csv(csv_pad: str | Path) -> pd.DataFrame:
    """Laad een Momants-export zonder privacygevoelige kolommen te bewaren."""
    pad = Path(csv_pad).expanduser()
    if not pad.is_file():
        raise FileNotFoundError(f"CSV-bestand niet gevonden: {pad}")

    data = _lees_met_header(pad) if _heeft_momants_header(pad) else _lees_zonder_header(pad)

    ontbrekend = VERPLICHTE_KOLOMMEN - set(data.columns)
    if ontbrekend:
        raise ValueError(
            "De CSV mist verplichte Momants-velden: "
            f"{', '.join(sorted(ontbrekend))}."
        )

    for optionele_kolom in {"message_type", "agent_id"} - set(data.columns):
        data[optionele_kolom] = pd.NA

    data = data[VEILIGE_KOLOMMEN].copy()
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=True)
    data["from_agent"] = (
        data["from_agent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    # Onvolledige regels uit een geknipte export worden niet verwerkt.
    data = data.dropna(subset=["created_at", "from_agent", "conversation_id"])
    data["conversation_id"] = data["conversation_id"].astype(str).str.strip()
    data = data.loc[data["conversation_id"].ne("")].copy()

    return data.sort_values(["conversation_id", "created_at"]).reset_index(drop=True)


def _is_bruikbare_bezoekerstekst(rij: pd.Series) -> bool:
    """Selecteer vrije bezoekerstekst; sla bottekst, leegte en kale URL's over."""
    if bool(rij["from_agent"]) or pd.isna(rij["text"]):
        return False
    tekst = str(rij["text"]).strip()
    return bool(tekst) and KALE_URL.fullmatch(tekst) is None


def selecteer_bezoekersberichten(data: pd.DataFrame) -> pd.DataFrame:
    """Houd chronologisch gesorteerde, bruikbare bezoekersberichten over."""
    selectie = data.loc[data.apply(_is_bruikbare_bezoekerstekst, axis=1)].copy()
    selectie["text"] = selectie["text"].astype(str).str.strip()
    return selectie.sort_values(["conversation_id", "created_at"]).reset_index(drop=True)


def _vertaal_modeluitkomst(uitkomst: dict[str, object]) -> tuple[str, float]:
    """Vertaal één TabularisAI-resultaat naar een Momants-label."""
    modellabel = str(uitkomst["label"]).strip()
    if modellabel not in LABEL_MAPPING:
        raise ValueError(
            f"Onbekend modellabel {modellabel!r}; werk LABEL_MAPPING bij."
        )
    return LABEL_MAPPING[modellabel], float(uitkomst["score"])


def classificeer_berichten(
    bezoekersberichten: pd.DataFrame,
    batchgrootte: int = 32,
) -> pd.DataFrame:
    """Classificeer alle bezoekersberichten in batches met TabularisAI."""
    if bezoekersberichten.empty:
        raise ValueError("De CSV bevat geen bruikbare bezoekersberichten.")

    sentiment_model = pipeline(
        task="text-classification",
        model=MODEL_ID,
    )
    teksten = bezoekersberichten["text"].tolist()
    modeluitkomsten = sentiment_model(
        teksten,
        truncation=True,
        batch_size=batchgrootte,
    )

    vertaald = [_vertaal_modeluitkomst(uitkomst) for uitkomst in modeluitkomsten]
    resultaat = bezoekersberichten.copy()
    resultaat["sentiment_bericht"] = [item[0] for item in vertaald]
    resultaat["zekerheid_bericht"] = [round(item[1], 4) for item in vertaald]
    return resultaat


def maak_gespreksoverzicht(berichtresultaten: pd.DataFrame) -> pd.DataFrame:
    """Vat ieder gesprek samen op basis van het laatste bezoekersbericht."""
    gegroepeerd = berichtresultaten.groupby("conversation_id", sort=False)

    overzicht = gegroepeerd.agg(
        eerste_bericht_at=("created_at", "min"),
        laatste_bericht_at=("created_at", "max"),
        aantal_bezoekersberichten=("text", "size"),
    )

    # Het laatste bericht benadert de stemming waarmee de bezoeker vertrekt.
    laatste = (
        berichtresultaten.sort_values(["conversation_id", "created_at"])
        .groupby("conversation_id", as_index=False)
        .tail(1)
        .set_index("conversation_id")
    )
    overzicht["sentiment_gesprek"] = laatste["sentiment_bericht"]
    overzicht["zekerheid"] = laatste["zekerheid_bericht"]
    overzicht = overzicht.reset_index()
    overzicht["uitleg"] = overzicht.apply(
        lambda rij: (
            f"Het laatste van {rij['aantal_bezoekersberichten']} bruikbare "
            f"bezoekersberichten is {rij['sentiment_gesprek']} "
            f"({rij['zekerheid']:.0%} zekerheid)."
        ),
        axis=1,
    )
    return overzicht


def _berichtuitvoer(
    berichtresultaten: pd.DataFrame,
    tekst_opnemen: bool,
) -> pd.DataFrame:
    """Maak de veilige berichtentabel; tekst is standaard uitgesloten."""
    kolommen = [
        "conversation_id",
        "created_at",
        "message_type",
        "agent_id",
        "sentiment_bericht",
        "zekerheid_bericht",
    ]
    if tekst_opnemen:
        kolommen.insert(2, "text")
    return berichtresultaten[kolommen].copy()


def verwerk_csv(
    csv_pad: str | Path,
    uitvoermap: str | Path = "resultaten",
    batchgrootte: int = 32,
    tekst_opnemen: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Voer de volledige verwerking uit en schrijf twee resultaatbestanden."""
    data = laad_momants_csv(csv_pad)
    bezoekersberichten = selecteer_bezoekersberichten(data)
    berichtresultaten = classificeer_berichten(bezoekersberichten, batchgrootte)
    gespreksoverzicht = maak_gespreksoverzicht(berichtresultaten)

    map_pad = Path(uitvoermap).expanduser()
    map_pad.mkdir(parents=True, exist_ok=True)

    veilige_berichten = _berichtuitvoer(berichtresultaten, tekst_opnemen)
    veilige_berichten.to_csv(map_pad / "sentiment_per_bericht.csv", index=False)
    gespreksoverzicht.to_csv(map_pad / "sentiment_per_gesprek.csv", index=False)

    return veilige_berichten, gespreksoverzicht


def _maak_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extraheer sentiment uit een Momants CSV-export.",
    )
    parser.add_argument("csv_pad", type=Path, help="Pad naar de Momants CSV-export.")
    parser.add_argument(
        "--uitvoermap",
        type=Path,
        default=Path("resultaten"),
        help="Map voor de twee resultaat-CSV's (standaard: resultaten).",
    )
    parser.add_argument(
        "--batchgrootte",
        type=int,
        default=32,
        help="Aantal berichten per modelbatch (standaard: 32).",
    )
    parser.add_argument(
        "--tekst-opnemen",
        action="store_true",
        help="Neem berichttekst op in de berichtuitvoer; standaard blijft die weg.",
    )
    parser.add_argument(
        "--alleen-controleren",
        action="store_true",
        help="Controleer inladen en groepering zonder het model te starten.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    argumenten = _maak_parser().parse_args(argv)

    if argumenten.alleen_controleren:
        data = laad_momants_csv(argumenten.csv_pad)
        bezoekers = selecteer_bezoekersberichten(data)
        print(f"Berichtrijen ingelezen: {len(data)}")
        print(f"Bruikbare bezoekersberichten: {len(bezoekers)}")
        print(f"Gesprekken: {bezoekers['conversation_id'].nunique()}")
        return 0

    berichten, gesprekken = verwerk_csv(
        csv_pad=argumenten.csv_pad,
        uitvoermap=argumenten.uitvoermap,
        batchgrootte=argumenten.batchgrootte,
        tekst_opnemen=argumenten.tekst_opnemen,
    )
    print(f"Berichtresultaten: {len(berichten)}")
    print(f"Gesprekresultaten: {len(gesprekken)}")
    print(f"Uitvoer geschreven naar: {argumenten.uitvoermap.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())