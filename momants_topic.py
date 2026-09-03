"""Classify Momants visitor messages into festival topics and subtopics."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from momants_sentiment import load_momants_csv


MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CSV_PAD = Path("data/tests/conversations_100.csv")
TOPICS_SEED_PATH = Path("data/seeds/topics_en_v2.csv")
EVENT_ID = "decibel_2026"

MAIN_TOPICS = [
    "Access",
    "Transport",
    "Venue",
    "Payments",
    "Program",
]
NONE_LABEL = "None"
NONE_PROTOTYPES = [
    "A thank you or a compliment, with no question in it",
    "A greeting or a goodbye, with no question in it",
    "A short confirmation that the answer was understood, such as ok, clear, fine",
    "Just a bare web address with no question",
    "A complaint that the previous answer did not help, without naming a new subject",
]
SEED_COLUMNS = [
    "id",
    "event_id",
    "main_topic",
    "subtopic",
    "active",
    "description",
]
OUTPUT_COLUMNS = [
    "conversation_id",
    "main_topic",
    "subtopic",
    "similarity",
    "first_detected_at",
]
DEBUG_OUTPUT_COLUMNS = [
    "conversation_id",
    "created_at",
    "text",
    "main_topic",
    "subtopic",
    "similarity",
]
BARE_URL = re.compile(
    r"^\s*(?:https?://|www\.)\S+\s*$",
    flags=re.IGNORECASE,
)


def laad_momants_csv(bron: str | Path) -> pd.DataFrame:
    """Load only the six permitted fields through the shared safe loader."""
    return load_momants_csv(bron)


def _parse_active(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(
        f"Invalid active value {value!r}; use true or false in the topic seed."
    )


def laad_onderwerpen(
    seed_pad: str | Path = TOPICS_SEED_PATH,
    event_id: str = EVENT_ID,
) -> pd.DataFrame:
    """Load and validate active topic labels for one manually selected event."""
    pad = Path(seed_pad).expanduser()
    if not pad.is_file():
        raise FileNotFoundError(f"Topic seed file not found: {pad}")
    if not str(event_id).strip():
        raise ValueError("EVENT_ID may not be empty.")

    kolommen = pd.read_csv(pad, nrows=0).columns.tolist()
    ontbrekend = set(SEED_COLUMNS) - set(kolommen)
    if ontbrekend:
        raise ValueError(
            "The topic seed is missing columns: "
            f"{', '.join(sorted(ontbrekend))}."
        )

    onderwerpen = pd.read_csv(
        pad,
        usecols=SEED_COLUMNS,
        keep_default_na=False,
    )
    onderwerpen["active"] = onderwerpen["active"].map(_parse_active)
    onderwerpen = onderwerpen.loc[
        onderwerpen["event_id"].astype(str).str.strip().eq(str(event_id).strip())
        & onderwerpen["active"]
    ].copy()
    if onderwerpen.empty:
        raise ValueError(f"No active topic labels found for event_id {event_id!r}.")

    for kolom in ["main_topic", "subtopic", "description"]:
        onderwerpen[kolom] = onderwerpen[kolom].astype(str).str.strip()
        if onderwerpen[kolom].eq("").any():
            raise ValueError(f"The topic seed contains an empty {kolom} value.")

    onbekende_hoofdonderwerpen = set(onderwerpen["main_topic"]) - set(MAIN_TOPICS)
    if onbekende_hoofdonderwerpen:
        raise ValueError(
            "Unknown main topics for this event: "
            f"{', '.join(sorted(onbekende_hoofdonderwerpen))}."
        )

    dubbelen = onderwerpen.duplicated(["main_topic", "subtopic"], keep=False)
    if dubbelen.any():
        combinaties = onderwerpen.loc[
            dubbelen, ["main_topic", "subtopic"]
        ].drop_duplicates()
        weergegeven = ", ".join(
            f"{rij.main_topic}/{rij.subtopic}"
            for rij in combinaties.itertuples(index=False)
        )
        raise ValueError(f"Duplicate active topic combinations: {weergegeven}.")

    return onderwerpen[SEED_COLUMNS].reset_index(drop=True)


def _is_bruikbaar_bezoekersbericht(rij: pd.Series) -> bool:
    if bool(rij["from_agent"]) or pd.isna(rij["text"]):
        return False
    tekst = str(rij["text"]).strip()
    return bool(tekst)


def selecteer_bezoekersberichten(data: pd.DataFrame) -> pd.DataFrame:
    """Select usable visitor messages in chronological order."""
    selectie = data.loc[
        data.apply(_is_bruikbaar_bezoekersbericht, axis=1)
    ].copy()
    selectie["text"] = selectie["text"].astype(str).str.strip()
    return selectie.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)


def _maak_embeddingmodel() -> SentenceTransformer:
    """Load the multilingual embedding model once for one processing run."""
    return SentenceTransformer(MODEL_ID)


def _encodeer(
    model: Any,
    teksten: list[str],
    batchgrootte: int,
) -> np.ndarray:
    """Create normalized embeddings so their dot product is cosine similarity."""
    embeddings = model.encode(
        teksten,
        batch_size=batchgrootte,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(teksten):
        raise ValueError("The embedding model returned an invalid result.")
    return matrix


def classificeer_berichten(
    bezoekersberichten: pd.DataFrame,
    onderwerpen: pd.DataFrame,
    batchgrootte: int = 16,
    model: Any | None = None,
) -> pd.DataFrame:
    """Choose the closest subtopic, then derive its main topic from the seed."""
    if batchgrootte < 1:
        raise ValueError("batchgrootte must be at least 1.")
    if bezoekersberichten.empty:
        leeg = bezoekersberichten.copy()
        for kolom, dtype in [
            ("main_topic", "object"),
            ("subtopic", "object"),
            ("similarity", "float64"),
        ]:
            leeg[kolom] = pd.Series(dtype=dtype)
        return leeg

    model = model or _maak_embeddingmodel()
    werk = bezoekersberichten.copy()
    werk["main_topic"] = NONE_LABEL
    werk["subtopic"] = NONE_LABEL
    werk["similarity"] = np.nan

    url_masker = werk["text"].str.match(BARE_URL)
    te_embedden = werk.index[~url_masker]
    if len(te_embedden) == 0:
        return werk

    onderwerp_embeddings = _encodeer(
        model,
        onderwerpen["description"].tolist(),
        batchgrootte,
    )
    none_embeddings = _encodeer(model, NONE_PROTOTYPES, batchgrootte)
    bericht_embeddings = _encodeer(
        model,
        werk.loc[te_embedden, "text"].tolist(),
        batchgrootte,
    )
    onderwerp_scores = bericht_embeddings @ onderwerp_embeddings.T
    none_scores = bericht_embeddings @ none_embeddings.T

    beste_onderwerp_indices = onderwerp_scores.argmax(axis=1)
    beste_onderwerp_scores = onderwerp_scores.max(axis=1)
    beste_none_scores = none_scores.max(axis=1)

    for positie, dataframe_index in enumerate(te_embedden):
        if beste_none_scores[positie] > beste_onderwerp_scores[positie]:
            werk.at[dataframe_index, "similarity"] = float(
                beste_none_scores[positie]
            )
            continue

        onderwerp = onderwerpen.iloc[int(beste_onderwerp_indices[positie])]
        werk.at[dataframe_index, "main_topic"] = onderwerp["main_topic"]
        werk.at[dataframe_index, "subtopic"] = onderwerp["subtopic"]
        # This is cosine similarity, not a probability or calibrated confidence.
        werk.at[dataframe_index, "similarity"] = float(
            beste_onderwerp_scores[positie]
        )

    return werk


def maak_onderwerpoverzicht(geclassificeerd: pd.DataFrame) -> pd.DataFrame:
    """Keep the first detection of each unique conversation-topic combination."""
    if geclassificeerd.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    echte_onderwerpen = geclassificeerd.loc[
        geclassificeerd["main_topic"].ne(NONE_LABEL)
    ].copy()
    if echte_onderwerpen.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    eerste = (
        echte_onderwerpen.sort_values(
            ["conversation_id", "created_at"], kind="stable"
        )
        .drop_duplicates(
            ["conversation_id", "main_topic", "subtopic"], keep="first"
        )
        .rename(columns={"created_at": "first_detected_at"})
    )
    eerste["first_detected_at"] = eerste["first_detected_at"].apply(
        lambda waarde: waarde.isoformat()
    )
    eerste["similarity"] = eerste["similarity"].round(4)
    return eerste[OUTPUT_COLUMNS].reset_index(drop=True)


def process_csv(
    csv_path: str | Path = CSV_PAD,
    seed_path: str | Path = TOPICS_SEED_PATH,
    event_id: str = EVENT_ID,
    output_directory: str | Path = "results",
    batch_size: int = 16,
    model: Any | None = None,
) -> pd.DataFrame:
    """Write privacy-safe conversation-topic and message-level validation tables."""
    gestart_op = datetime.now().astimezone()
    data = laad_momants_csv(csv_path)
    onderwerpen = laad_onderwerpen(seed_path, event_id)
    bezoekers = selecteer_bezoekersberichten(data)
    geclassificeerd = classificeer_berichten(
        bezoekers,
        onderwerpen,
        batchgrootte=batch_size,
        model=model,
    )
    overzicht = maak_onderwerpoverzicht(geclassificeerd)

    doelmap = Path(output_directory).expanduser()
    doelmap.mkdir(parents=True, exist_ok=True)
    tijdstempel = gestart_op.strftime("%Y%m%d_%H%M%S_%f")
    uitvoerpad = doelmap / f"topics_per_conversation_{tijdstempel}.csv"
    debugpad = doelmap / f"topics_per_message_{tijdstempel}.csv"
    overzicht.to_csv(uitvoerpad, index=False)
    debugbestand = geclassificeerd[DEBUG_OUTPUT_COLUMNS].copy()
    debugbestand["created_at"] = debugbestand["created_at"].apply(
        lambda waarde: waarde.isoformat()
    )
    debugbestand.to_csv(debugpad, index=False)
    overzicht.attrs["output_path"] = uitvoerpad.resolve()
    overzicht.attrs["debug_output_path"] = debugpad.resolve()
    return overzicht


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify Momants conversations by festival topic."
    )
    parser.add_argument("csv_path", type=Path, help="Local Momants CSV export.")
    parser.add_argument(
        "--event-id",
        default=EVENT_ID,
        help=f"Event ID to select in the topic seed (default: {EVENT_ID}).",
    )
    parser.add_argument(
        "--topics-seed",
        type=Path,
        default=TOPICS_SEED_PATH,
        help="CSV containing event-specific topic labels.",
    )
    parser.add_argument(
        "--output-directory",
        "--uitvoermap",
        dest="output_directory",
        type=Path,
        default=Path("results"),
        help="Directory for topics_per_conversation_<timestamp>.csv.",
    )
    parser.add_argument(
        "--batch-size",
        "--batchgrootte",
        dest="batch_size",
        type=int,
        default=16,
        help="Number of visitor messages per model batch.",
    )
    parser.add_argument(
        "--check-only",
        "--alleen-controleren",
        dest="check_only",
        action="store_true",
        help="Validate Momants loading and event labels without loading the model.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = laad_momants_csv(args.csv_path)
    onderwerpen = laad_onderwerpen(args.topics_seed, args.event_id)
    bezoekers = selecteer_bezoekersberichten(data)

    if args.check_only:
        print(f"Message rows read: {len(data)}")
        print(f"Usable visitor messages: {len(bezoekers)}")
        print(f"Conversations: {bezoekers['conversation_id'].nunique()}")
        print(f"Event ID: {args.event_id}")
        print(f"Active topic labels: {len(onderwerpen)}")
        for hoofdonderwerp in MAIN_TOPICS:
            aantal = int(onderwerpen["main_topic"].eq(hoofdonderwerp).sum())
            print(f"- {hoofdonderwerp}: {aantal} subtopics")
        return 0

    overzicht = process_csv(
        csv_path=args.csv_path,
        seed_path=args.topics_seed,
        event_id=args.event_id,
        output_directory=args.output_directory,
        batch_size=args.batch_size,
    )
    print(f"Conversation-topic combinations: {len(overzicht)}")
    print(f"Output written to: {overzicht.attrs['output_path']}")
    print(f"Validation debug output written to: {overzicht.attrs['debug_output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())