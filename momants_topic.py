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

from momants_conversation_filter import select_visitor_messages
from momants_sentiment import load_momants_csv


MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CSV_PATH = Path("data/tests/conversations_100.csv")
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


def load_topics(
    seed_path: str | Path = TOPICS_SEED_PATH,
    event_id: str = EVENT_ID,
) -> pd.DataFrame:
    """Load and validate active topic labels for one manually selected event."""
    path = Path(seed_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Topic seed file not found: {path}")
    if not str(event_id).strip():
        raise ValueError("EVENT_ID may not be empty.")

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = set(SEED_COLUMNS) - set(columns)
    if missing:
        raise ValueError(
            "The topic seed is missing columns: "
            f"{', '.join(sorted(missing))}."
        )

    topics = pd.read_csv(
        path,
        usecols=SEED_COLUMNS,
        keep_default_na=False,
    )
    topics["active"] = topics["active"].map(_parse_active)
    topics = topics.loc[
        topics["event_id"].astype(str).str.strip().eq(str(event_id).strip())
        & topics["active"]
    ].copy()
    if topics.empty:
        raise ValueError(f"No active topic labels found for event_id {event_id!r}.")

    for column in ["main_topic", "subtopic", "description"]:
        topics[column] = topics[column].astype(str).str.strip()
        if topics[column].eq("").any():
            raise ValueError(f"The topic seed contains an empty {column} value.")

    unknown_main_topics = set(topics["main_topic"]) - set(MAIN_TOPICS)
    if unknown_main_topics:
        raise ValueError(
            "Unknown main topics for this event: "
            f"{', '.join(sorted(unknown_main_topics))}."
        )

    duplicates = topics.duplicated(["main_topic", "subtopic"], keep=False)
    if duplicates.any():
        combinations = topics.loc[
            duplicates, ["main_topic", "subtopic"]
        ].drop_duplicates()
        displayed = ", ".join(
            f"{row.main_topic}/{row.subtopic}"
            for row in combinations.itertuples(index=False)
        )
        raise ValueError(f"Duplicate active topic combinations: {displayed}.")

    return topics[SEED_COLUMNS].reset_index(drop=True)


def _build_embedding_model() -> SentenceTransformer:
    """Load the multilingual embedding model once for one processing run."""
    return SentenceTransformer(MODEL_ID)


def _encode(
    model: Any,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Create normalized embeddings so their dot product is cosine similarity."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError("The embedding model returned an invalid result.")
    return matrix


def classify_messages(
    visitor_messages: pd.DataFrame,
    topics: pd.DataFrame,
    batch_size: int = 16,
    model: Any | None = None,
) -> pd.DataFrame:
    """Choose the closest subtopic, then derive its main topic from the seed."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if visitor_messages.empty:
        empty = visitor_messages.copy()
        for column, dtype in [
            ("main_topic", "object"),
            ("subtopic", "object"),
            ("similarity", "float64"),
        ]:
            empty[column] = pd.Series(dtype=dtype)
        return empty

    model = model or _build_embedding_model()
    work = visitor_messages.copy()
    work["main_topic"] = NONE_LABEL
    work["subtopic"] = NONE_LABEL
    work["similarity"] = np.nan

    url_mask = work["text"].str.match(BARE_URL)
    indices_to_embed = work.index[~url_mask]
    if len(indices_to_embed) == 0:
        return work

    topic_embeddings = _encode(
        model,
        topics["description"].tolist(),
        batch_size,
    )
    none_embeddings = _encode(model, NONE_PROTOTYPES, batch_size)
    message_embeddings = _encode(
        model,
        work.loc[indices_to_embed, "text"].tolist(),
        batch_size,
    )
    topic_scores = message_embeddings @ topic_embeddings.T
    none_scores = message_embeddings @ none_embeddings.T

    best_topic_indices = topic_scores.argmax(axis=1)
    best_topic_scores = topic_scores.max(axis=1)
    best_none_scores = none_scores.max(axis=1)

    for position, dataframe_index in enumerate(indices_to_embed):
        if best_none_scores[position] > best_topic_scores[position]:
            work.at[dataframe_index, "similarity"] = float(
                best_none_scores[position]
            )
            continue

        topic = topics.iloc[int(best_topic_indices[position])]
        work.at[dataframe_index, "main_topic"] = topic["main_topic"]
        work.at[dataframe_index, "subtopic"] = topic["subtopic"]
        # This is cosine similarity, not a probability or calibrated confidence.
        work.at[dataframe_index, "similarity"] = float(
            best_topic_scores[position]
        )

    return work


def build_topic_overview(classified: pd.DataFrame) -> pd.DataFrame:
    """Keep the first detection of each unique conversation-topic combination."""
    if classified.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    real_topics = classified.loc[
        classified["main_topic"].ne(NONE_LABEL)
    ].copy()
    if real_topics.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Each row represents a conversation-topic, not one conversation. Use
    # conversation_id.nunique() when reporting how many conversations covered a topic.
    first = (
        real_topics.sort_values(
            ["conversation_id", "created_at"], kind="stable"
        )
        .drop_duplicates(
            ["conversation_id", "main_topic", "subtopic"], keep="first"
        )
        .rename(columns={"created_at": "first_detected_at"})
    )
    first["first_detected_at"] = first["first_detected_at"].apply(
        lambda value: value.isoformat()
    )
    first["similarity"] = first["similarity"].round(4)
    return first[OUTPUT_COLUMNS].reset_index(drop=True)


def process_csv(
    csv_path: str | Path = CSV_PATH,
    seed_path: str | Path = TOPICS_SEED_PATH,
    event_id: str = EVENT_ID,
    output_directory: str | Path = "results",
    batch_size: int = 16,
    model: Any | None = None,
) -> pd.DataFrame:
    """Write privacy-safe conversation-topic and message-level validation tables."""
    started_at = datetime.now().astimezone()
    data = load_momants_csv(csv_path)
    topics = load_topics(seed_path, event_id)
    visitors = select_visitor_messages(data)
    classified = classify_messages(
        visitors,
        topics,
        batch_size=batch_size,
        model=model,
    )
    overview = build_topic_overview(classified)

    output_directory_path = Path(output_directory).expanduser()
    output_directory_path.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_directory_path / f"topics_per_conversation_{timestamp}.csv"
    debug_path = output_directory_path / f"topics_per_message_{timestamp}.csv"
    low_similarity_path = (
        output_directory_path / f"lowest_similarity_assignments_{timestamp}.csv"
    )
    overview.to_csv(output_path, index=False)
    debug_output = classified[DEBUG_OUTPUT_COLUMNS].copy()
    debug_output["created_at"] = debug_output["created_at"].apply(
        lambda value: value.isoformat()
    )
    debug_output.to_csv(debug_path, index=False)

    assignments = classified.loc[
        classified["main_topic"].ne(NONE_LABEL),
        ["text", "main_topic", "subtopic", "similarity"],
    ]
    lowest_count = max(1, int(np.ceil(len(assignments) * 0.10)))
    unique_assignments = (
        assignments.sort_values("similarity", kind="stable")
        .drop_duplicates("text", keep="first")
    )
    lowest_similarity = unique_assignments.head(lowest_count)
    lowest_similarity.to_csv(low_similarity_path, index=False)

    overview.attrs["output_path"] = output_path.resolve()
    overview.attrs["debug_output_path"] = debug_path.resolve()
    overview.attrs["low_similarity_output_path"] = low_similarity_path.resolve()
    overview.attrs["unreadable_from_agent_count"] = visitors.attrs.get(
        "unreadable_from_agent_count", 0
    )
    return overview


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
        type=Path,
        default=Path("results"),
        help="Directory for topics_per_conversation_<timestamp>.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of visitor messages per model batch.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate Momants loading and event labels without loading the model.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.check_only:
        data = load_momants_csv(args.csv_path)
        topics = load_topics(args.topics_seed, args.event_id)
        visitors = select_visitor_messages(data)
        print(f"Message rows read: {len(data)}")
        print(f"Usable visitor messages: {len(visitors)}")
        print(f"Conversations: {visitors['conversation_id'].nunique()}")
        print(f"Event ID: {args.event_id}")
        print(f"Active topic labels: {len(topics)}")
        for main_topic in MAIN_TOPICS:
            count = int(topics["main_topic"].eq(main_topic).sum())
            print(f"- {main_topic}: {count} subtopics")
        return 0

    overview = process_csv(
        csv_path=args.csv_path,
        seed_path=args.topics_seed,
        event_id=args.event_id,
        output_directory=args.output_directory,
        batch_size=args.batch_size,
    )
    print(f"Conversation-topic combinations: {len(overview)}")
    print(f"Output written to: {overview.attrs['output_path']}")
    print(f"Validation debug output written to: {overview.attrs['debug_output_path']}")
    print(
        "Lowest-similarity assignments written to: "
        f"{overview.attrs['low_similarity_output_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())