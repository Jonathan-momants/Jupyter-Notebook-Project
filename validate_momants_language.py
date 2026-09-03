"""Direct validation for the three-language Momants Lingua classifier."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

import momants_language


ANSWER_KEY_PATH = Path("data/tests/answer_key_language.csv")
LABELS = ["nl", "en", "de", "und"]


def load_answer_key(path: str | Path = ANSWER_KEY_PATH) -> pd.DataFrame:
    key = pd.read_csv(path, usecols=["text", "language"], keep_default_na=False)
    if len(key) != 199 or key["text"].nunique() != 199:
        raise ValueError("The language answer key must contain 199 unique texts.")
    expected = {"nl": 151, "en": 34, "de": 9, "und": 5}
    if key["language"].value_counts().to_dict() != expected:
        raise ValueError(f"Unexpected language distribution; expected {expected}.")
    return key


def classify_answer_key(
    key: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    build_started = perf_counter()
    detector = momants_language.build_detector()
    build_seconds = perf_counter() - build_started
    classify_started = perf_counter()
    predicted = momants_language.classify_messages(key[["text"]], detector=detector)
    elapsed = perf_counter() - classify_started
    evaluation = key.rename(columns={"language": "true_language"}).copy()
    evaluation["predicted_language"] = predicted["language"].tolist()
    evaluation["confidence"] = predicted["confidence"].tolist()
    return evaluation, build_seconds, elapsed / len(key) * 1000


def validate(path: str | Path = ANSWER_KEY_PATH) -> dict[str, float]:
    key = load_answer_key(path)
    evaluation, build_seconds, milliseconds = classify_answer_key(key)
    real = evaluation["true_language"].ne("und")
    correct = evaluation["true_language"].eq(evaluation["predicted_language"])
    accuracy = float(correct.loc[real].mean() * 100)
    recalls = {
        language: float(
            correct.loc[evaluation["true_language"].eq(language)].mean() * 100
        )
        for language in ["nl", "en", "de"]
    }
    macro_recall = sum(recalls.values()) / len(recalls)
    und_correct = float(correct.loc[~real].mean() * 100)
    baseline = float(
        evaluation.loc[real, "true_language"].value_counts().max() / real.sum() * 100
    )

    print("LANGUAGE VALIDATION")
    print(f"language_accuracy over {int(real.sum())} real-language texts: {accuracy:.1f}%")
    print(f"macro_recall over nl, en, de: {macro_recall:.1f}%")
    for language, recall in recalls.items():
        count = int(evaluation["true_language"].eq(language).sum())
        print(f"- {language}: {recall:.1f}% (n={count})")
    print(f"und_correct over {int((~real).sum())} texts: {und_correct:.1f}%")
    print(f"naive Dutch baseline: {baseline:.1f}%")
    print(f"average time per message: {milliseconds:.2f} ms")
    print(f"detector build time: {build_seconds:.4f} s")
    print("Caveat: German has only n=9; enough to detect total failure, not a stable rate.")

    matrix = pd.crosstab(
        evaluation["true_language"], evaluation["predicted_language"]
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)
    print("\nCONFUSION MATRIX (rows=true, columns=predicted)")
    print(matrix.to_string())

    lengths = evaluation["text"].astype(str).str.len()
    groups = [
        ("1-11", lengths.between(1, 11)),
        ("12-30", lengths.between(12, 30)),
        ("31+", lengths.ge(31)),
    ]
    print("\nACCURACY BY TEXT LENGTH")
    for label, mask in groups:
        print(f"- {label}: {correct.loc[mask].mean() * 100:.1f}% (n={int(mask.sum())})")

    errors = evaluation.loc[
        ~correct,
        ["text", "true_language", "predicted_language", "confidence"],
    ]
    print(f"\nMISCLASSIFIED TEXTS ({len(errors)})")
    print("None" if errors.empty else errors.to_string(index=False))
    return {
        "language_accuracy": accuracy,
        "macro_recall": macro_recall,
        "und_correct": und_correct,
        "naive_baseline": baseline,
        "milliseconds_per_message": milliseconds,
        "detector_build_seconds": build_seconds,
    }


if __name__ == "__main__":
    validate()