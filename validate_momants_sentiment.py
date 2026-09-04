"""Validate the sentiment classifier against the held-out Claude-labelled set.

The 200 conversations in attached_assets/conversations_200_1788514819128.csv were
labelled independently
and share no message text with data/training/sentiment_training.csv, so they are a
genuine held-out test. Two baselines are printed alongside the model, because
accuracy alone is misleading on this data: roughly 88% of boundary messages are
Neutral, so a classifier that always answers Neutral already scores 88% while
detecting nothing. The numbers that matter are precision and recall on the two
signal classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

import momants_sentiment


CONVERSATIONS_PATH = Path("attached_assets/conversations_200_1788514819128.csv")
LABELS_PATH = Path("data/tests/claude_200_labeled.csv")
NEUTRAL = momants_sentiment.NEUTRAL_LABEL
SENTIMENT_CLASSES = momants_sentiment.SENTIMENT_CATEGORIES


def _precision_recall(
    predicted: pd.Series,
    actual: pd.Series,
    label: str,
) -> dict[str, float]:
    true_positive = int((predicted.eq(label) & actual.eq(label)).sum())
    false_positive = int((predicted.eq(label) & actual.ne(label)).sum())
    false_negative = int((predicted.ne(label) & actual.eq(label)).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else float("nan")
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else float("nan")
    )
    return {
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def validate(
    conversations_path: str | Path = CONVERSATIONS_PATH,
    labels_path: str | Path = LABELS_PATH,
) -> dict[str, float]:
    missing = [
        str(path)
        for path in (Path(conversations_path), Path(labels_path))
        if not Path(path).is_file()
    ]
    if missing:
        raise SystemExit(
            "Evaluation data not found: "
            + ", ".join(missing)
            + ".\nConversation exports and answer keys are deliberately not tracked "
            "in this repository (see commit 814376a). Supply them locally and pass "
            "--conversations / --labels, or place them at the default paths."
        )
    data = momants_sentiment.load_momants_csv(conversations_path)
    customer_messages = momants_sentiment.select_customer_messages(data)
    summary = momants_sentiment.create_conversation_summary(customer_messages)

    labels = pd.read_csv(labels_path)
    labels = labels.loc[labels["has_question"].eq(True)]
    evaluation = labels.merge(summary, on="conversation_id", how="left")

    starting_accuracy = float(
        evaluation["starting_sentiment"].eq(evaluation["sentiment_start"]).mean() * 100
    )
    ending_accuracy = float(
        evaluation["ending_sentiment"].eq(evaluation["sentiment_end"]).mean() * 100
    )
    neutral_baseline = float(
        (
            evaluation["sentiment_start"].eq(NEUTRAL).mean()
            + evaluation["sentiment_end"].eq(NEUTRAL).mean()
        )
        / 2
        * 100
    )

    print("SENTIMENT VALIDATION (held-out Claude-labelled conversations)")
    print(f"conversations with a question: {len(evaluation)}")
    print(f"starting_sentiment accuracy: {starting_accuracy:.1f}%")
    print(f"ending_sentiment accuracy:   {ending_accuracy:.1f}%")
    print(f"always-Neutral baseline:     {neutral_baseline:.1f}%")
    print(
        "\nAccuracy is not the goal: the baseline above detects nothing. "
        "Read the signal classes."
    )

    predicted = pd.concat(
        [evaluation["starting_sentiment"], evaluation["ending_sentiment"]],
        ignore_index=True,
    )
    actual = pd.concat(
        [evaluation["sentiment_start"], evaluation["sentiment_end"]],
        ignore_index=True,
    )
    print("\nSIGNAL CLASSES (start and end labels pooled)")
    metrics: dict[str, float] = {
        "starting_accuracy": starting_accuracy,
        "ending_accuracy": ending_accuracy,
        "neutral_baseline": neutral_baseline,
    }
    for label in SENTIMENT_CLASSES:
        if label == NEUTRAL:
            continue
        scores = _precision_recall(predicted, actual, label)
        threshold = momants_sentiment.DECISION_THRESHOLDS.get(label)
        print(
            f"- {label} (threshold {threshold}): "
            f"precision={scores['precision']:.2f} recall={scores['recall']:.2f} "
            f"(tp={scores['true_positive']} fp={scores['false_positive']} "
            f"fn={scores['false_negative']})"
        )
        metrics[f"{label}_precision"] = scores["precision"]
        metrics[f"{label}_recall"] = scores["recall"]

    print("\nCONFUSION MATRIX (rows=Claude, columns=model; start and end pooled)")
    matrix = pd.crosstab(actual, predicted).reindex(
        index=SENTIMENT_CLASSES, columns=SENTIMENT_CLASSES, fill_value=0
    )
    print(matrix.to_string())

    without_trend = int((~evaluation["has_trend"].astype(bool)).sum())
    print(
        f"\nConversations where start and end are the same message: "
        f"{without_trend} of {len(evaluation)}. These have has_trend=False and must "
        "be excluded from any sentiment-movement chart."
    )

    disagreements = evaluation.loc[
        evaluation["starting_sentiment"].ne(evaluation["sentiment_start"])
        | evaluation["ending_sentiment"].ne(evaluation["sentiment_end"]),
        ["gesprek_nr", "sentiment_start", "starting_sentiment", "sentiment_end", "ending_sentiment"],
    ]
    print(f"\nDISAGREEMENTS ({len(disagreements)})")
    print("None" if disagreements.empty else disagreements.to_string(index=False))
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the sentiment classifier against held-out labels."
    )
    parser.add_argument("--conversations", type=Path, default=CONVERSATIONS_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    validate(arguments.conversations, arguments.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
