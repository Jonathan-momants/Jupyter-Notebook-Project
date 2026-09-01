"""Verwerk een Momants CSV-export tot start- en eindsentiment per gesprek."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
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

UITVOERKOLOMMEN = [
    "conversation_id",
    "aantal_bezoekersberichten",
    "sentiment_start",
    "zekerheid_start",
    "sentiment_eind",
    "zekerheid_eind",
    "uitleg",
]

VERPLICHTE_KOLOMMEN = {
    "created_at",
    "text",
    "from_agent",
    "conversation_id",
}

# Posities van uitsluitend veilige velden in het headerloze 22-veldenformaat.
HEADERLOZE_POSITIES = {
    0: "created_at",
    3: "text",
    8: "from_agent",
    10: "message_type",
    17: "agent_id",
    19: "conversation_id",
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
    """Lees uitsluitend veilige kolomposities uit het 22-veldenformaat."""
    posities = list(HEADERLOZE_POSITIES)
    data = pd.read_csv(
        csv_pad,
        header=None,
        usecols=posities,
        on_bad_lines="skip",
        engine="python",
    )
    return data.rename(columns=HEADERLOZE_POSITIES)


def laad_momants_csv(bron: str | Path) -> pd.DataFrame:
    """Laad een lokale Momants-export en laat alleen veilige kolommen door."""
    if isinstance(bron, str) and re.match(r"^https?://", bron, re.IGNORECASE):
        raise ValueError(
            "Endpoint-URL's worden nog niet ondersteund; geef een lokaal CSV-pad op."
        )

    pad = Path(bron).expanduser()
    if not pad.is_file():
        raise FileNotFoundError(f"CSV-bestand niet gevonden: {pad}")

    data = (
        _lees_met_header(pad)
        if _heeft_momants_header(pad)
        else _lees_zonder_header(pad)
    )

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
        resultaat = bezoekersberichten.copy()
        resultaat["sentiment_bericht"] = pd.Series(dtype="object")
        resultaat["zekerheid_bericht"] = pd.Series(dtype="float64")
        return resultaat

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


def _selecteer_grensberichten(
    bezoekersberichten: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """Selecteer unieke eerste/laatste rijen, zodat één bericht één modelcall krijgt."""
    gesorteerd = bezoekersberichten.sort_values(
        ["conversation_id", "created_at"]
    )
    groepen = gesorteerd.groupby("conversation_id", sort=False)
    eerste_indices = groepen.head(1).index
    laatste_indices = groepen.tail(1).index
    grensindices = pd.Index(
        list(dict.fromkeys([*eerste_indices, *laatste_indices]))
    )
    return gesorteerd.loc[grensindices].copy(), eerste_indices, laatste_indices


def maak_gespreksoverzicht(
    bezoekersberichten: pd.DataFrame,
    batchgrootte: int = 32,
) -> pd.DataFrame:
    """Bepaal per gesprek sentiment op het eerste én laatste bezoekersbericht."""
    if bezoekersberichten.empty:
        return pd.DataFrame(columns=UITVOERKOLOMMEN)

    grensberichten, eerste_indices, laatste_indices = _selecteer_grensberichten(
        bezoekersberichten
    )
    geclassificeerd = classificeer_berichten(grensberichten, batchgrootte)

    aantallen = (
        bezoekersberichten.groupby("conversation_id", sort=False)
        .size()
        .rename("aantal_bezoekersberichten")
    )
    eerste = geclassificeerd.loc[eerste_indices].set_index("conversation_id")
    laatste = geclassificeerd.loc[laatste_indices].set_index("conversation_id")

    overzicht = aantallen.to_frame()
    overzicht["sentiment_start"] = eerste["sentiment_bericht"]
    overzicht["zekerheid_start"] = eerste["zekerheid_bericht"]
    overzicht["sentiment_eind"] = laatste["sentiment_bericht"]
    overzicht["zekerheid_eind"] = laatste["zekerheid_bericht"]
    overzicht = overzicht.reset_index()
    overzicht["uitleg"] = overzicht.apply(
        lambda rij: (
            f"Het gesprek begint {rij['sentiment_start'].lower()} en eindigt "
            f"{rij['sentiment_eind'].lower()}, op basis van "
            f"{rij['aantal_bezoekersberichten']} "
            f"{'bruikbaar bezoekersbericht' if rij['aantal_bezoekersberichten'] == 1 else 'bruikbare bezoekersberichten'}."
        ),
        axis=1,
    )
    return overzicht[UITVOERKOLOMMEN]


def verwerk_csv(
    csv_pad: str | Path,
    uitvoermap: str | Path = "resultaten",
    batchgrootte: int = 32,
) -> pd.DataFrame:
    """Voer de verwerking uit en schrijf één veilige gesprekstabel."""
    gestart_op = datetime.now().astimezone()
    data = laad_momants_csv(csv_pad)
    bezoekersberichten = selecteer_bezoekersberichten(data)
    gespreksoverzicht = maak_gespreksoverzicht(
        bezoekersberichten,
        batchgrootte=batchgrootte,
    )

    map_pad = Path(uitvoermap).expanduser()
    map_pad.mkdir(parents=True, exist_ok=True)
    tijdstempel = gestart_op.strftime("%Y%m%d_%H%M%S_%f")
    uitvoerpad = map_pad / f"sentiment_per_gesprek_{tijdstempel}.csv"
    gespreksoverzicht.to_csv(uitvoerpad, index=False)
    gespreksoverzicht.attrs["uitvoerpad"] = uitvoerpad.resolve()

    return gespreksoverzicht


def _maak_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extraheer sentiment uit een Momants CSV-export.",
    )
    parser.add_argument("csv_pad", type=Path, help="Pad naar de Momants CSV-export.")
    parser.add_argument(
        "--uitvoermap",
        type=Path,
        default=Path("resultaten"),
        help="Map voor sentiment_per_gesprek.csv (standaard: resultaten).",
    )
    parser.add_argument(
        "--batchgrootte",
        type=int,
        default=32,
        help="Aantal berichten per modelbatch (standaard: 32).",
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

    gesprekken = verwerk_csv(
        csv_pad=argumenten.csv_pad,
        uitvoermap=argumenten.uitvoermap,
        batchgrootte=argumenten.batchgrootte,
    )
    print(f"Gesprekresultaten: {len(gesprekken)}")
    print(f"Uitvoer geschreven naar: {gesprekken.attrs['uitvoerpad']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())