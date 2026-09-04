"""Run all five independent Momants processors and combine conversation results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

import momants_answer_check
import momants_intent
import momants_language
import momants_sentiment
import momants_topic


DEFAULT_INPUT = Path("notebooks/decibel_50_conversations.csv")
DEFAULT_OUTPUT = Path("results/decibel_50_all_processors.csv")
DEFAULT_PROCESSOR_OUTPUT = Path("results/decibel_50_processors")
LIST_SEPARATOR = " ||| "


def _aggregate_intents(intents: pd.DataFrame) -> pd.DataFrame:
    columns = ["conversation_id", "intent_count", "intents"]
    if intents.empty:
        return pd.DataFrame(columns=columns)
    work = intents.copy()
    work["_display"] = work.apply(
        lambda row: (
            f"{row['intent']} "
            f"[confidence={float(row['confidence']):.4f}; "
            f"first_detected_at={row['first_detected_at']}]"
        ),
        axis=1,
    )
    result = (
        work.groupby("conversation_id", sort=False)
        .agg(
            intent_count=("intent", "size"),
            intents=("_display", lambda values: LIST_SEPARATOR.join(values)),
        )
        .reset_index()
    )
    return result[columns]


def _aggregate_topics(topics: pd.DataFrame) -> pd.DataFrame:
    columns = ["conversation_id", "topic_count", "topics"]
    if topics.empty:
        return pd.DataFrame(columns=columns)
    work = topics.copy()
    work["_display"] = work.apply(
        lambda row: (
            f"{row['main_topic']} > {row['subtopic']} "
            f"[similarity={float(row['similarity']):.4f}; "
            f"first_detected_at={row['first_detected_at']}]"
        ),
        axis=1,
    )
    result = (
        work.groupby("conversation_id", sort=False)
        .agg(
            topic_count=("subtopic", "size"),
            topics=("_display", lambda values: LIST_SEPARATOR.join(values)),
        )
        .reset_index()
    )
    return result[columns]


def _prefix_except_id(data: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return data.rename(
        columns={
            column: f"{prefix}{column}"
            for column in data.columns
            if column != "conversation_id"
        }
    )


def run_all_processors(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    processor_output_directory: str | Path = DEFAULT_PROCESSOR_OUTPUT,
) -> pd.DataFrame:
    """Run every processor on one source and write one combined conversation CSV."""
    source = Path(input_path)
    destination = Path(output_path)
    processor_directory = Path(processor_output_directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    processor_directory.mkdir(parents=True, exist_ok=True)

    safe_data = momants_sentiment.load_momants_csv(source)
    conversation_ids = safe_data[["conversation_id"]].drop_duplicates(
        ignore_index=True
    )

    sentiment = momants_sentiment.process_csv(
        source,
        output_directory=processor_directory,
    )
    answer_check = momants_answer_check.process_csv(
        source,
        output_dir=processor_directory,
    )
    intents = momants_intent.process_csv(
        source,
        output_directory=processor_directory,
    )
    topics = momants_topic.process_csv(
        csv_path=source,
        output_directory=processor_directory,
    )
    language = momants_language.process_csv(
        source,
        output_directory=processor_directory,
    )

    combined = conversation_ids.copy()
    processor_tables = [
        _prefix_except_id(sentiment, "sentiment_"),
        _prefix_except_id(answer_check, "answer_check_"),
        _prefix_except_id(_aggregate_intents(intents), "intent_"),
        _prefix_except_id(_aggregate_topics(topics), "topic_"),
        _prefix_except_id(language, "language_"),
    ]
    for table in processor_tables:
        combined = combined.merge(
            table,
            on="conversation_id",
            how="left",
            validate="one_to_one",
        )

    for count_column in ("intent_intent_count", "topic_topic_count"):
        combined[count_column] = combined[count_column].fillna(0).astype(int)
    for list_column in ("intent_intents", "topic_topics"):
        combined[list_column] = combined[list_column].fillna("")

    combined = combined.sort_values(
        "conversation_id",
        kind="stable",
    ).reset_index(drop=True)
    combined.to_csv(destination, index=False)
    combined.attrs["output_path"] = destination.resolve()
    combined.attrs["source_conversation_count"] = len(conversation_ids)
    return combined


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all Momants processors and combine their conversation output."
    )
    parser.add_argument("input_path", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--processor-output",
        type=Path,
        default=DEFAULT_PROCESSOR_OUTPUT,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    combined = run_all_processors(
        args.input_path,
        args.output,
        args.processor_output,
    )
    print(f"Conversations combined: {len(combined)}")
    print(f"Columns: {len(combined.columns)}")
    print(f"Combined output: {combined.attrs['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())