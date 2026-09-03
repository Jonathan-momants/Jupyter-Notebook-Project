"""Direct validation of the per-message Momants sentiment classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import momants_sentiment


ANSWER_KEY_PATH = Path("data/tests/answer_key_sentiment.csv")
SENTIMENT_CLASSES = [
    "Neutral (task-oriented)",
    "Negative (frustrated)",
    "Angry (panic)",
    "Positive",
]


def validate(answer_key_path: str | Path = ANSWER_KEY_PATH) -> dict[str, float]:
    key = pd.read_csv(
        answer_key_path,
        usecols=["text", "sentiment"],
        keep_default_na=False,
    )
    unknown = set(key["sentiment"]) - set(SENTIMENT_CLASSES)
    if unknown:
        raise ValueError(f"Unknown sentiment labels: {', '.join(sorted(unknown))}")

    predicted = momants_sentiment.classify_messages(key[["text"]].copy())
    evaluation = key.rename(columns={"sentiment": "true_sentiment"}).copy()
    evaluation["predicted_sentiment"] = (
        predicted["message_sentiment"]
        .replace({"Neutral (task-focused)": "Neutral (task-oriented)"})
        .tolist()
    )
    present_classes = [
        label for label in SENTIMENT_CLASSES if evaluation["true_sentiment"].eq(label).any()
    ]
    absent_classes = [label for label in SENTIMENT_CLASSES if label not in present_classes]
    correct = evaluation["true_sentiment"].eq(evaluation["predicted_sentiment"])
    accuracy = float(correct.mean() * 100)
    recalls = [
        float(
            evaluation.loc[evaluation["true_sentiment"].eq(label), "predicted_sentiment"]
            .eq(label)
            .mean()
            * 100
        )
        for label in present_classes
    ]
    macro_recall = sum(recalls) / len(recalls)

    print("SENTIMENT VALIDATION")
    print(f"sentiment_accuracy over {len(evaluation)} messages: {accuracy:.1f}%")
    print(
        f"macro_recall over {len(present_classes)} present classes: "
        f"{macro_recall:.1f}%"
    )
    if absent_classes:
        print("Not present and untested: " + ", ".join(absent_classes))

    matrix = pd.crosstab(
        evaluation["true_sentiment"], evaluation["predicted_sentiment"]
    ).reindex(index=SENTIMENT_CLASSES, columns=SENTIMENT_CLASSES, fill_value=0)
    print("\nCONFUSION MATRIX (rows=true, columns=predicted)")
    print(matrix.to_string())

    errors = evaluation.loc[
        ~correct, ["text", "true_sentiment", "predicted_sentiment"]
    ]
    print(f"\nMISCLASSIFIED MESSAGES ({len(errors)})")
    print("None" if errors.empty else errors.to_string(index=False))
    return {"sentiment_accuracy": accuracy, "macro_recall": macro_recall}


if __name__ == "__main__":
    validate()