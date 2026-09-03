"""Measure four pretrained sentiment models without training or changing production."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoConfig, pipeline


ANSWER_KEY_PATH = Path("data/tests/answer_key_sentiment_v2.csv")
MOMANTS_CLASSES = [
    "Positive",
    "Neutral (task-oriented)",
    "Negative (frustrated)",
]
LANGUAGES = ["nl", "en", "de"]
EMERGENCY_TEXTS = [
    "Er ligt iemand bewusteloos bij podium 2, we hebben nu hulp nodig!",
    "Mijn dochter is zoek, ik kan haar nergens meer vinden op het terrein.",
    "Er breekt brand uit bij de foodtrucks, help snel!",
    "There's a medical emergency near the main stage, someone collapsed!",
]

MODEL_MAPPINGS = {
    "tabularisai/multilingual-sentiment-analysis": {
        "Very Negative": "Negative (frustrated)",
        "Negative": "Negative (frustrated)",
        "Neutral": "Neutral (task-oriented)",
        "Positive": "Positive",
        "Very Positive": "Positive",
    },
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": {
        "negative": "Negative (frustrated)",
        "neutral": "Neutral (task-oriented)",
        "positive": "Positive",
    },
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student": {
        "negative": "Negative (frustrated)",
        "neutral": "Neutral (task-oriented)",
        "positive": "Positive",
    },
    "nlptown/bert-base-multilingual-uncased-sentiment": {
        "1 star": "Negative (frustrated)",
        "2 stars": "Negative (frustrated)",
        "3 stars": "Neutral (task-oriented)",
        "4 stars": "Positive",
        "5 stars": "Positive",
    },
}


def load_answer_key(path: str | Path = ANSWER_KEY_PATH) -> pd.DataFrame:
    """Load and strictly validate the supplied 195-message answer key."""
    key = pd.read_csv(
        path,
        usecols=["text", "sentiment", "language"],
        keep_default_na=False,
    )
    if len(key) != 195 or key["text"].nunique() != 195:
        raise ValueError("The sentiment v2 key must contain 195 unique texts.")
    expected_sentiments = {
        "Neutral (task-oriented)": 154,
        "Negative (frustrated)": 28,
        "Positive": 13,
    }
    if key["sentiment"].value_counts().to_dict() != expected_sentiments:
        raise ValueError(f"Unexpected sentiment counts; expected {expected_sentiments}.")
    expected_languages = {"nl": 151, "en": 34, "de": 9, "und": 1}
    if key["language"].value_counts().to_dict() != expected_languages:
        raise ValueError(f"Unexpected language counts; expected {expected_languages}.")
    missing_emergencies = set(EMERGENCY_TEXTS) - set(key["text"])
    if missing_emergencies:
        raise ValueError("The answer key is missing one or more emergency checks.")
    return key


def _recall(
    evaluation: pd.DataFrame,
    sentiment: str,
) -> float:
    selected = evaluation.loc[evaluation["sentiment"].eq(sentiment)]
    return float(selected["predicted_sentiment"].eq(sentiment).mean() * 100)


def _metrics(evaluation: pd.DataFrame) -> dict[str, object]:
    correct = evaluation["sentiment"].eq(evaluation["predicted_sentiment"])
    recalls = {
        sentiment: _recall(evaluation, sentiment)
        for sentiment in MOMANTS_CLASSES
        if evaluation["sentiment"].eq(sentiment).any()
    }
    return {
        "accuracy": float(correct.mean() * 100),
        "macro_recall": sum(recalls.values()) / len(recalls),
        "recalls": recalls,
    }


def _print_mapping(model_id: str, raw_labels: list[str], observed: set[str]) -> None:
    mapping = MODEL_MAPPINGS[model_id]
    print(f"Configured raw labels: {raw_labels}")
    print(f"Observed returned raw labels: {sorted(observed)}")
    print("Mapping to Momants classes:")
    for raw_label in raw_labels:
        print(f"- {raw_label!r} -> {mapping[raw_label]!r}")


def evaluate_model(
    model_id: str,
    key: pd.DataFrame,
    batch_size: int = 32,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one pretrained model and evaluate all messages directly."""
    config = AutoConfig.from_pretrained(model_id)
    raw_labels = [str(config.id2label[index]) for index in sorted(config.id2label)]
    mapping = MODEL_MAPPINGS[model_id]
    if set(raw_labels) != set(mapping):
        raise ValueError(
            f"Raw labels for {model_id} differ from the declared mapping: {raw_labels}"
        )

    load_started = perf_counter()
    classifier = pipeline("text-classification", model=model_id, device=-1)
    load_seconds = perf_counter() - load_started
    inference_started = perf_counter()
    raw_predictions = classifier(
        key["text"].tolist(),
        truncation=True,
        batch_size=batch_size,
    )
    inference_seconds = perf_counter() - inference_started

    evaluation = key.copy()
    evaluation["raw_label"] = [str(item["label"]) for item in raw_predictions]
    evaluation["raw_score"] = [float(item["score"]) for item in raw_predictions]
    unknown = set(evaluation["raw_label"]) - set(mapping)
    if unknown:
        raise ValueError(f"Unmapped labels returned by {model_id}: {sorted(unknown)}")
    evaluation["predicted_sentiment"] = evaluation["raw_label"].map(mapping)

    metrics = _metrics(evaluation)
    metrics["milliseconds_per_message"] = inference_seconds / len(evaluation) * 1000
    metrics["load_seconds"] = load_seconds

    print(f"\n{'=' * 88}\nMODEL: {model_id}\n{'=' * 88}")
    _print_mapping(model_id, raw_labels, set(evaluation["raw_label"]))
    print(f"\nAccuracy: {metrics['accuracy']:.1f}%")
    print(f"Macro-recall over three classes: {metrics['macro_recall']:.1f}%")
    print("Recall per class:")
    for sentiment in MOMANTS_CLASSES:
        recall = metrics["recalls"][sentiment]
        count = int(evaluation["sentiment"].eq(sentiment).sum())
        print(f"- {sentiment}: {recall:.1f}% (n={count})")
    print(f"Inference time per message: {metrics['milliseconds_per_message']:.2f} ms")
    print(f"Model/tokenizer load time: {load_seconds:.2f} s")

    matrix = pd.crosstab(
        evaluation["sentiment"], evaluation["predicted_sentiment"]
    ).reindex(index=MOMANTS_CLASSES, columns=MOMANTS_CLASSES, fill_value=0)
    print("\nCONFUSION MATRIX (rows=true, columns=predicted)")
    print(matrix.to_string())

    print("\nBY TRUE LANGUAGE")
    language_metrics: dict[str, dict[str, object]] = {}
    for language in LANGUAGES:
        selected = evaluation.loc[evaluation["language"].eq(language)].copy()
        selected_metrics = _metrics(selected)
        language_metrics[language] = selected_metrics
        present_classes = [
            label for label in MOMANTS_CLASSES if selected["sentiment"].eq(label).any()
        ]
        print(
            f"- {language}: accuracy {selected_metrics['accuracy']:.1f}%, "
            f"macro-recall {selected_metrics['macro_recall']:.1f}% "
            f"over {len(present_classes)} present classes (n={len(selected)})"
        )
    metrics["by_language"] = language_metrics

    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evaluation, metrics


def compare(
    answer_key_path: str | Path = ANSWER_KEY_PATH,
    batch_size: int = 32,
) -> dict[str, dict[str, object]]:
    """Run the four independent model measurements and print comparison tables."""
    key = load_answer_key(answer_key_path)
    baseline = float(key["sentiment"].eq("Neutral (task-oriented)").mean() * 100)
    print(f"Naive always-Neutral baseline: {baseline:.1f}%")
    print(
        "German caveat: n=9 is too small for a reliable percentage, "
        "but sufficient to reveal total failure."
    )

    metrics_by_model: dict[str, dict[str, object]] = {}
    emergency_table = key.loc[
        key["text"].isin(EMERGENCY_TEXTS),
        ["text", "sentiment", "language"],
    ].copy()
    emergency_table = emergency_table.set_index("text").loc[EMERGENCY_TEXTS].reset_index()

    for model_id in MODEL_MAPPINGS:
        evaluation, metrics = evaluate_model(model_id, key, batch_size=batch_size)
        metrics_by_model[model_id] = metrics
        predictions = evaluation.set_index("text")["predicted_sentiment"]
        emergency_table[model_id] = emergency_table["text"].map(predictions)

    print(f"\n{'=' * 88}\nFOUR EMERGENCY MESSAGES SIDE BY SIDE\n{'=' * 88}")
    with pd.option_context("display.max_colwidth", None, "display.width", 300):
        print(emergency_table.to_string(index=False))

    print(f"\n{'=' * 88}\nSUMMARY — NO WINNER SELECTED\n{'=' * 88}")
    rows = []
    for model_id, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_id,
                "accuracy": round(float(metrics["accuracy"]), 1),
                "macro_recall": round(float(metrics["macro_recall"]), 1),
                "ms_per_message": round(
                    float(metrics["milliseconds_per_message"]), 2
                ),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    return metrics_by_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare four pretrained sentiment models without training."
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=ANSWER_KEY_PATH,
        help="CSV with text, sentiment, certainty, and language.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    compare(args.answer_key, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())