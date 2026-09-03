"""Direct validation of the per-message Momants intent classifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import momants_intent


ANSWER_KEY_PATH = Path("data/tests/answer_key_v2.csv")
NONE_LABEL = "None"


def validate(answer_key_path: str | Path = ANSWER_KEY_PATH) -> dict[str, float]:
    key = pd.read_csv(answer_key_path, usecols=["text", "intent"], keep_default_na=False)
    key["intent"] = key["intent"].replace("", NONE_LABEL)
    unknown = set(key["intent"]) - set(momants_intent.INTENT_CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown intent labels: {', '.join(sorted(unknown))}")

    predicted = momants_intent.classify_messages(key[["text"]].copy())
    evaluation = key.rename(columns={"intent": "true_intent"}).copy()
    evaluation["predicted_intent"] = predicted["intent"].tolist()

    real = evaluation["true_intent"].ne(NONE_LABEL)
    present_categories = [
        label
        for label in momants_intent.INTENT_CATEGORIES
        if label != NONE_LABEL and evaluation.loc[real, "true_intent"].eq(label).any()
    ]
    absent_categories = [
        label
        for label in momants_intent.INTENT_CATEGORIES
        if label != NONE_LABEL and label not in present_categories
    ]
    accuracy = float(
        evaluation.loc[real, "predicted_intent"]
        .eq(evaluation.loc[real, "true_intent"])
        .mean()
        * 100
    )
    recalls = [
        float(
            evaluation.loc[evaluation["true_intent"].eq(label), "predicted_intent"]
            .eq(label)
            .mean()
            * 100
        )
        for label in present_categories
    ]
    macro_recall = sum(recalls) / len(recalls)
    none_rejected = float(
        evaluation.loc[~real, "predicted_intent"].eq(NONE_LABEL).mean() * 100
    )

    print("INTENT VALIDATION")
    print(f"intent_accuracy over {int(real.sum())} real messages: {accuracy:.1f}%")
    print(
        f"macro_recall over {len(present_categories)} present categories: "
        f"{macro_recall:.1f}%"
    )
    print(f"none_rejected over {int((~real).sum())} smalltalk messages: {none_rejected:.1f}%")
    if absent_categories:
        print("Not present and excluded from macro-recall: " + ", ".join(absent_categories))

    labels = momants_intent.INTENT_CATEGORIES
    matrix = pd.crosstab(
        evaluation["true_intent"], evaluation["predicted_intent"]
    ).reindex(index=labels, columns=labels, fill_value=0)
    print("\nCONFUSION MATRIX (rows=true, columns=predicted)")
    print(matrix.to_string())

    errors = evaluation.loc[
        evaluation["true_intent"].ne(evaluation["predicted_intent"]),
        ["text", "true_intent", "predicted_intent"],
    ]
    print(f"\nMISCLASSIFIED MESSAGES ({len(errors)})")
    print("None" if errors.empty else errors.to_string(index=False))
    return {
        "intent_accuracy": accuracy,
        "macro_recall": macro_recall,
        "none_rejected": none_rejected,
    }


if __name__ == "__main__":
    validate()