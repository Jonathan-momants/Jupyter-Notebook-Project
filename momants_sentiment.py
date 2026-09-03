"""Process a Momants CSV export into starting and ending sentiment per conversation."""

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
    "Very Positive": "Positive",
    "Positive": "Positive",
    "Neutral": "Neutral (task-focused)",
    "Negative": "Negative (frustrated)",
    "Very Negative": "Angry (panic)",
}

SAFE_COLUMNS = [
    "created_at",
    "text",
    "from_agent",
    "message_type",
    "conversation_id",
    "agent_id",
]

OUTPUT_COLUMNS = [
    "conversation_id",
    "customer_message_count",
    "starting_sentiment",
    "starting_confidence",
    "ending_sentiment",
    "ending_confidence",
    "explanation",
]

REQUIRED_COLUMNS = {
    "created_at",
    "text",
    "from_agent",
    "conversation_id",
}

# Positions of safe fields only in the headerless 22-field format.
HEADERLESS_POSITIONS = {
    0: "created_at",
    3: "text",
    8: "from_agent",
    10: "message_type",
    17: "agent_id",
    19: "conversation_id",
}

COLUMN_ALIASES = {
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

BARE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)


def _normalize_column_name(name: object) -> str:
    """Normalize a column name for comparison with known names."""
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    compact = text.replace("_", "")
    return COLUMN_ALIASES.get(text, COLUMN_ALIASES.get(compact, text))


def _has_momants_header(csv_path: Path) -> bool:
    """Check whether the first CSV row contains recognizable Momants column names."""
    columns = pd.read_csv(csv_path, nrows=0).columns
    normalized = {_normalize_column_name(column) for column in columns}
    return REQUIRED_COLUMNS.issubset(normalized)


def _read_with_header(csv_path: Path) -> pd.DataFrame:
    """Read only safe fields from a CSV with column headers."""
    original_columns = pd.read_csv(csv_path, nrows=0).columns
    column_mapping = {
        column: _normalize_column_name(column)
        for column in original_columns
        if _normalize_column_name(column) in SAFE_COLUMNS
    }
    dataframe = pd.read_csv(
        csv_path,
        usecols=list(column_mapping),
        on_bad_lines="skip",
    )
    return dataframe.rename(columns=column_mapping)


def _read_without_header(csv_path: Path) -> pd.DataFrame:
    """Read only safe column positions from the 22-field format."""
    positions = list(HEADERLESS_POSITIONS)
    dataframe = pd.read_csv(
        csv_path,
        header=None,
        usecols=positions,
        on_bad_lines="skip",
        engine="python",
    )
    return dataframe.rename(columns=HEADERLESS_POSITIONS)


def load_momants_csv(source: str | Path) -> pd.DataFrame:
    """Load a local Momants export and retain only safe columns."""
    if isinstance(source, str) and re.match(r"^https?://", source, re.IGNORECASE):
        raise ValueError(
            "Endpoint URLs are not yet supported; provide a local CSV path."
        )

    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    dataframe = (
        _read_with_header(path)
        if _has_momants_header(path)
        else _read_without_header(path)
    )

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing required Momants fields: "
            f"{', '.join(sorted(missing_columns))}."
        )

    for optional_column in {"message_type", "agent_id"} - set(dataframe.columns):
        dataframe[optional_column] = pd.NA

    dataframe = dataframe[SAFE_COLUMNS].copy()
    dataframe["created_at"] = pd.to_datetime(dataframe["created_at"], errors="coerce", utc=True)
    dataframe["from_agent"] = (
        dataframe["from_agent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    # Keep unreadable from_agent values visible so each analysis can report them.
    dataframe = dataframe.dropna(subset=["created_at", "conversation_id"])
    dataframe["conversation_id"] = dataframe["conversation_id"].astype(str).str.strip()
    dataframe = dataframe.loc[dataframe["conversation_id"].ne("")].copy()

    return dataframe.sort_values(["conversation_id", "created_at"]).reset_index(drop=True)


def _is_usable_customer_text(row: pd.Series) -> bool:
    """Select free customer text; skip agent text, blank values, and bare URLs."""
    if bool(row["from_agent"]) or pd.isna(row["text"]):
        return False
    text = str(row["text"]).strip()
    return bool(text) and BARE_URL.fullmatch(text) is None


def select_customer_messages(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep chronologically sorted, usable customer messages."""
    selection = dataframe.loc[dataframe.apply(_is_usable_customer_text, axis=1)].copy()
    selection["text"] = selection["text"].astype(str).str.strip()
    return selection.sort_values(["conversation_id", "created_at"]).reset_index(drop=True)


def _translate_model_result(result: dict[str, object]) -> tuple[str, float]:
    """Translate one TabularisAI result to a Momants label."""
    model_label = str(result["label"]).strip()
    if model_label not in LABEL_MAPPING:
        raise ValueError(
            f"Unknown model label {model_label!r}; update LABEL_MAPPING."
        )
    return LABEL_MAPPING[model_label], float(result["score"])


def classify_messages(
    customer_messages: pd.DataFrame,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Classify all customer messages in batches with TabularisAI."""
    if customer_messages.empty:
        result = customer_messages.copy()
        result["message_sentiment"] = pd.Series(dtype="object")
        result["message_confidence"] = pd.Series(dtype="float64")
        return result

    sentiment_model = pipeline(
        task="text-classification",
        model=MODEL_ID,
    )
    texts = customer_messages["text"].tolist()
    model_results = sentiment_model(
        texts,
        truncation=True,
        batch_size=batch_size,
    )

    translated = [_translate_model_result(item) for item in model_results]
    result = customer_messages.copy()
    result["message_sentiment"] = [item[0] for item in translated]
    result["message_confidence"] = [round(item[1], 4) for item in translated]
    return result


def _select_boundary_messages(
    customer_messages: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """Select unique first/last rows, so each message receives one model call."""
    sorted_messages = customer_messages.sort_values(
        ["conversation_id", "created_at"]
    )
    groups = sorted_messages.groupby("conversation_id", sort=False)
    first_indices = groups.head(1).index
    last_indices = groups.tail(1).index
    boundary_indices = pd.Index(
        list(dict.fromkeys([*first_indices, *last_indices]))
    )
    return sorted_messages.loc[boundary_indices].copy(), first_indices, last_indices


def create_conversation_summary(
    customer_messages: pd.DataFrame,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Determine sentiment for the first and last customer message per conversation."""
    if customer_messages.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    boundary_messages, first_indices, last_indices = _select_boundary_messages(
        customer_messages
    )
    classified = classify_messages(boundary_messages, batch_size)

    counts = (
        customer_messages.groupby("conversation_id", sort=False)
        .size()
        .rename("customer_message_count")
    )
    first = classified.loc[first_indices].set_index("conversation_id")
    last = classified.loc[last_indices].set_index("conversation_id")

    summary = counts.to_frame()
    summary["starting_sentiment"] = first["message_sentiment"]
    summary["starting_confidence"] = first["message_confidence"]
    summary["ending_sentiment"] = last["message_sentiment"]
    summary["ending_confidence"] = last["message_confidence"]
    summary = summary.reset_index()
    summary["explanation"] = summary.apply(
        lambda row: (
            f"The conversation starts {row['starting_sentiment'].lower()} and ends "
            f"{row['ending_sentiment'].lower()}, based on "
            f"{row['customer_message_count']} "
            f"{'usable customer message' if row['customer_message_count'] == 1 else 'usable customer messages'}."
        ),
        axis=1,
    )
    return summary[OUTPUT_COLUMNS]


def process_csv(
    csv_path: str | Path,
    output_directory: str | Path = "results",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run processing and write one safe conversation table."""
    started_at = datetime.now().astimezone()
    dataframe = load_momants_csv(csv_path)
    customer_messages = select_customer_messages(dataframe)
    conversation_summary = create_conversation_summary(
        customer_messages,
        batch_size=batch_size,
    )

    output_path = Path(output_directory).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    result_path = output_path / f"sentiment_per_conversation_{timestamp}.csv"
    conversation_summary.to_csv(result_path, index=False)
    conversation_summary.attrs["output_path"] = result_path.resolve()

    return conversation_summary


# Backward-compatible programmatic aliases for the original public API.
laad_momants_csv = load_momants_csv
selecteer_bezoekersberichten = select_customer_messages
classificeer_berichten = classify_messages
maak_gespreksoverzicht = create_conversation_summary


def verwerk_csv(
    csv_pad: str | Path,
    uitvoermap: str | Path = "results",
    batchgrootte: int = 32,
) -> pd.DataFrame:
    """Backward-compatible wrapper for :func:`process_csv`."""
    return process_csv(csv_pad, uitvoermap, batchgrootte)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract sentiment from a Momants CSV export.",
    )
    parser.add_argument("csv_path", type=Path, help="Path to the Momants CSV export.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results"),
        help="Directory for sentiment_per_conversation_<timestamp>.csv (default: results).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of messages per model batch (default: 32).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check loading and grouping without starting the model.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _create_parser().parse_args(argv)

    if arguments.check_only:
        dataframe = load_momants_csv(arguments.csv_path)
        customer_messages = select_customer_messages(dataframe)
        print(f"Message rows loaded: {len(dataframe)}")
        print(f"Usable customer messages: {len(customer_messages)}")
        print(f"Conversations: {customer_messages['conversation_id'].nunique()}")
        return 0

    conversations = process_csv(
        csv_path=arguments.csv_path,
        output_directory=arguments.output_directory,
        batch_size=arguments.batch_size,
    )
    print(f"Conversation results: {len(conversations)}")
    print(f"Output written to: {conversations.attrs['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())