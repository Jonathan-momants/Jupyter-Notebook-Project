"""Classify Momants visitor messages into festival topics and subtopics."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
from transformers import pipeline

from momants_sentiment import load_momants_csv


MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
CSV_PAD = Path("attached_assets/test_gesprekken_100.csv")
TOPICS_SEED_PATH = Path("momants_topics_seed_en.csv")
EVENT_ID = "decibel_2026"

MAIN_TOPICS = [
    "Access",
    "Transport",
    "Venue",
    "Payments",
    "Program",
    "Feedback",
]
MAIN_TOPIC_DESCRIPTIONS = {
    "Access": (
        "Getting in or having a valid ticket, wristband, or reservation to attend"
    ),
    "Transport": (
        "Getting to the venue by car, bike, public transport, or shuttle, "
        "and parking"
    ),
    "Venue": (
        "The physical site itself: camping, cabins, lockers, charging points, "
        "shops, the map, or on-site emergencies"
    ),
    "Payments": (
        "Cashless balance, refunds, deposits, or paying for food and drinks"
    ),
    "Program": (
        "The schedule of acts and activities, or general festival information"
    ),
    "Feedback": (
        "A complaint or negative experience about noise, facilities, or staff"
    ),
}
NONE_LABEL = "None"
NONE_DESCRIPTION = (
    "This message is small talk, a greeting, or not about any specific "
    "festival topic (e.g. thanks, chit-chat)"
)
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
    "confidence",
    "first_detected_at",
]
BARE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)


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
    return bool(tekst) and BARE_URL.fullmatch(tekst) is None


def selecteer_bezoekersberichten(data: pd.DataFrame) -> pd.DataFrame:
    """Select usable visitor messages in chronological order."""
    selectie = data.loc[
        data.apply(_is_bruikbaar_bezoekersbericht, axis=1)
    ].copy()
    selectie["text"] = selectie["text"].astype(str).str.strip()
    return selectie.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)


def _maak_classifier() -> Any:
    """Create the requested standard Transformers zero-shot pipeline."""
    return pipeline("zero-shot-classification", model=MODEL_ID)


def _normaliseer_resultaten(
    resultaten: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [resultaten] if isinstance(resultaten, dict) else resultaten


def _beste_label(resultaat: dict[str, Any]) -> tuple[str, float]:
    labels = resultaat.get("labels", [])
    scores = resultaat.get("scores", [])
    if not labels or not scores or len(labels) != len(scores):
        raise ValueError("The zero-shot model returned an invalid result.")
    return str(labels[0]), float(scores[0])


def classificeer_berichten(
    bezoekersberichten: pd.DataFrame,
    onderwerpen: pd.DataFrame,
    batchgrootte: int = 16,
    classifier: Any | None = None,
) -> pd.DataFrame:
    """Classify each visitor message hierarchically in two zero-shot steps."""
    if batchgrootte < 1:
        raise ValueError("batchgrootte must be at least 1.")
    if bezoekersberichten.empty:
        leeg = bezoekersberichten.copy()
        for kolom, dtype in [
            ("main_topic", "object"),
            ("subtopic", "object"),
            ("confidence", "float64"),
        ]:
            leeg[kolom] = pd.Series(dtype=dtype)
        return leeg

    classifier = classifier or _maak_classifier()
    none_candidate = f"{NONE_LABEL}: {NONE_DESCRIPTION}"
    hoofd_kandidaat_naar_label = {
        f"{topic}: {MAIN_TOPIC_DESCRIPTIONS[topic]}": topic
        for topic in MAIN_TOPICS
    }
    hoofd_candidates = list(hoofd_kandidaat_naar_label) + [none_candidate]
    hoofd_resultaten = _normaliseer_resultaten(
        classifier(
            bezoekersberichten["text"].tolist(),
            candidate_labels=hoofd_candidates,
            multi_label=False,
            batch_size=batchgrootte,
        )
    )
    if len(hoofd_resultaten) != len(bezoekersberichten):
        raise ValueError("The model returned the wrong number of main-topic results.")

    werk = bezoekersberichten.copy()
    hoofdlabels: list[str] = []
    hoofdscores: list[float] = []
    for resultaat in hoofd_resultaten:
        label, score = _beste_label(resultaat)
        if label == none_candidate:
            hoofdlabels.append(NONE_LABEL)
        elif label in hoofd_kandidaat_naar_label:
            hoofdlabels.append(hoofd_kandidaat_naar_label[label])
        else:
            raise ValueError(
                f"The model returned an unknown main-topic label: {label!r}."
            )
        hoofdscores.append(score)
    werk["main_topic"] = hoofdlabels
    werk["_main_confidence"] = hoofdscores
    werk = werk.loc[werk["main_topic"].ne(NONE_LABEL)].copy()

    if werk.empty:
        werk["subtopic"] = pd.Series(dtype="object")
        werk["confidence"] = pd.Series(dtype="float64")
        return werk.drop(columns=["_main_confidence"])

    delen: list[pd.DataFrame] = []
    for hoofdonderwerp, groep in werk.groupby("main_topic", sort=False):
        labels = onderwerpen.loc[
            onderwerpen["main_topic"].eq(hoofdonderwerp),
            ["subtopic", "description"],
        ]
        other_candidate = (
            f"Other: General {hoofdonderwerp.lower()} question that doesn't "
            "fit a more specific subtopic"
        )
        kandidaat_naar_subtopic: dict[str, str] = {other_candidate: "Other"}
        for rij in labels.itertuples(index=False):
            if rij.subtopic == "Other":
                continue
            kandidaat = f"{rij.subtopic}: {rij.description}"
            if kandidaat in kandidaat_naar_subtopic:
                raise ValueError(
                    f"Duplicate subtopic candidate text for {hoofdonderwerp}: "
                    f"{kandidaat!r}."
                )
            kandidaat_naar_subtopic[kandidaat] = rij.subtopic

        sub_resultaten = _normaliseer_resultaten(
            classifier(
                groep["text"].tolist(),
                candidate_labels=list(kandidaat_naar_subtopic),
                multi_label=False,
                batch_size=batchgrootte,
            )
        )
        if len(sub_resultaten) != len(groep):
            raise ValueError(
                f"The model returned the wrong number of subtopic results for "
                f"{hoofdonderwerp}."
            )

        groep = groep.copy()
        subtopics: list[str] = []
        gecombineerde_scores: list[float] = []
        for (_, rij), resultaat in zip(groep.iterrows(), sub_resultaten):
            kandidaat, subscore = _beste_label(resultaat)
            if kandidaat not in kandidaat_naar_subtopic:
                raise ValueError(f"The model returned an unknown label: {kandidaat!r}.")
            subtopics.append(kandidaat_naar_subtopic[kandidaat])
            gecombineerde_scores.append(float(rij["_main_confidence"]) * subscore)
        groep["subtopic"] = subtopics
        groep["confidence"] = gecombineerde_scores
        delen.append(groep)

    return (
        pd.concat(delen)
        .sort_index(kind="stable")
        .drop(columns=["_main_confidence"])
    )


def maak_onderwerpoverzicht(geclassificeerd: pd.DataFrame) -> pd.DataFrame:
    """Keep the first detection of each unique conversation-topic combination."""
    if geclassificeerd.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if geclassificeerd["main_topic"].eq(NONE_LABEL).any():
        raise ValueError("None may not appear in topic output.")

    eerste = (
        geclassificeerd.sort_values(
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
    eerste["confidence"] = eerste["confidence"].round(4)
    return eerste[OUTPUT_COLUMNS].reset_index(drop=True)


def process_csv(
    csv_path: str | Path = CSV_PAD,
    seed_path: str | Path = TOPICS_SEED_PATH,
    event_id: str = EVENT_ID,
    output_directory: str | Path = "results",
    batch_size: int = 16,
    classifier: Any | None = None,
) -> pd.DataFrame:
    """Classify topics and write one privacy-safe conversation-topic table."""
    gestart_op = datetime.now().astimezone()
    data = laad_momants_csv(csv_path)
    onderwerpen = laad_onderwerpen(seed_path, event_id)
    bezoekers = selecteer_bezoekersberichten(data)
    geclassificeerd = classificeer_berichten(
        bezoekers,
        onderwerpen,
        batchgrootte=batch_size,
        classifier=classifier,
    )
    overzicht = maak_onderwerpoverzicht(geclassificeerd)

    doelmap = Path(output_directory).expanduser()
    doelmap.mkdir(parents=True, exist_ok=True)
    tijdstempel = gestart_op.strftime("%Y%m%d_%H%M%S_%f")
    uitvoerpad = doelmap / f"topics_per_conversation_{tijdstempel}.csv"
    overzicht.to_csv(uitvoerpad, index=False)
    overzicht.attrs["output_path"] = uitvoerpad.resolve()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())