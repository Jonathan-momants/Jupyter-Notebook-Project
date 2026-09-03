"""Export a blind labeling key and separate answer-check predictions.

Label criterion: KAN DE BEZOEKER HIERNA VERDER?

answered = yes
    Een concreet antwoord, OF een bruikbare doorverwijzing met een plaats of
    actie ("ga naar de Lost&Found-balie bij ingang Noord").
answered = no
    Geen antwoord, een ontwijking, of een vage doorverwijzing zonder plaats of
    actie ("neem contact op met de organisatie").
answered = unclear
    Niet eerlijk te beoordelen met de gegeven context.

Een bruikbare doorverwijzing telt dus als beantwoord.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from momants_answer_check import (
    classify_question_answer_pairs,
    pair_questions_with_answers,
)
from momants_sentiment import load_momants_csv


DEFAULT_SOURCE = Path("data/tests/conversations_100.csv")
DEFAULT_LABEL_OUTPUT = Path("data/tests/answer_check_to_label.csv")
DEFAULT_PREDICTION_OUTPUT = Path("data/tests/answer_check_predictions.csv")
NEXT_VARIANT_SEPARATOR = "\n\n---\n\n"


def normalize_pair_text(value: object) -> str:
    """Normalize Unicode, whitespace, and case solely for deduplication."""
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _pair_instances_with_next_reply(
    data: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Split the current episode's LLM replies into first and following replies."""
    result = pairs.copy()
    result["agent_reply_next"] = result["llm_responses"].apply(
        lambda replies: NEXT_VARIANT_SEPARATOR.join(replies[1:])
    )
    has_llm_reply = result["llm_responses"].apply(bool)
    result.loc[has_llm_reply, "answer_text"] = result.loc[
        has_llm_reply, "llm_responses"
    ].apply(lambda replies: replies[0])
    return result


def _unique_nonempty(values: pd.Series) -> list[str]:
    """Return unique non-empty text variants in stable first-seen order."""
    variants: list[str] = []
    normalized_seen: set[str] = set()
    for value in values:
        if pd.isna(value) or not str(value).strip():
            continue
        text = str(value).strip()
        normalized = normalize_pair_text(text)
        if normalized not in normalized_seen:
            normalized_seen.add(normalized)
            variants.append(text)
    return variants


def build_exports(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build the blind key and detached predictions from the same pair IDs."""
    pairs = pair_questions_with_answers(data)
    instances = _pair_instances_with_next_reply(data, pairs)
    classified = classify_question_answer_pairs(pairs)

    instances["_question_key"] = instances["question_text"].map(normalize_pair_text)
    instances["_reply_key"] = instances["answer_text"].map(normalize_pair_text)
    classified["_question_key"] = classified["question_text"].map(normalize_pair_text)
    classified["_reply_key"] = classified["answer_text"].map(normalize_pair_text)

    prediction_counts = classified.groupby(
        ["_question_key", "_reply_key"], sort=False, dropna=False
    )["answered"].nunique()
    if prediction_counts.gt(1).any():
        raise RuntimeError("Duplicate normalized pairs received different predictions.")

    grouped = instances.groupby(
        ["_question_key", "_reply_key"], sort=False, dropna=False
    )
    rows: list[dict[str, object]] = []
    for pair_id, (_, group) in enumerate(grouped, start=1):
        next_variants = _unique_nonempty(group["agent_reply_next"])
        rows.append(
            {
                "pair_id": pair_id,
                "question": group.iloc[0]["question_text"],
                "agent_reply": (
                    ""
                    if pd.isna(group.iloc[0]["answer_text"])
                    else str(group.iloc[0]["answer_text"])
                ),
                "agent_reply_next": NEXT_VARIANT_SEPARATOR.join(next_variants),
                "occurrences": len(group),
                "answered": "",
            }
        )

    labels = pd.DataFrame(
        rows,
        columns=[
            "pair_id",
            "question",
            "agent_reply",
            "agent_reply_next",
            "occurrences",
            "answered",
        ],
    )

    prediction_by_key = (
        classified.groupby(
            ["_question_key", "_reply_key"], sort=False, dropna=False
        )["answered"]
        .first()
        .reset_index()
    )
    prediction_by_key["prediction"] = prediction_by_key["answered"].map(
        {True: "yes", False: "no"}
    )
    pair_ids = instances[
        ["_question_key", "_reply_key"]
    ].drop_duplicates(ignore_index=True)
    pair_ids.insert(0, "pair_id", range(1, len(pair_ids) + 1))
    predictions = pair_ids.merge(
        prediction_by_key[
            ["_question_key", "_reply_key", "prediction"]
        ],
        on=["_question_key", "_reply_key"],
        how="left",
        validate="one_to_one",
    )[["pair_id", "prediction"]]

    return labels, predictions, len(pairs)


def export_files(
    source: str | Path = DEFAULT_SOURCE,
    label_output: str | Path = DEFAULT_LABEL_OUTPUT,
    prediction_output: str | Path = DEFAULT_PREDICTION_OUTPUT,
) -> tuple[int, int, int]:
    """Write both files and return only non-sensitive export counts."""
    data = load_momants_csv(source)
    labels, predictions, detected_questions = build_exports(data)

    label_path = Path(label_output)
    prediction_path = Path(prediction_output)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(label_path, index=False)
    predictions.to_csv(prediction_path, index=False)

    pairs_with_next = int(labels["agent_reply_next"].ne("").sum())
    return detected_questions, len(labels), pairs_with_next


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a blind answer-check labeling key."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_OUTPUT)
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_PREDICTION_OUTPUT
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    detected, unique, with_next = export_files(
        args.source, args.labels, args.predictions
    )
    print(f"Detected questions: {detected}")
    print(f"Unique normalized pairs: {unique}")
    print(f"Pairs with agent_reply_next: {with_next}")
    print(f"Label file: {args.labels}")
    print(f"Prediction file: {args.predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())