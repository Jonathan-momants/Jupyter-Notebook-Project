"""Golden-output regression checks for all five Momants analysis systems."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

import momants_answer_check
import momants_intent
import momants_language
import momants_sentiment
import momants_topic


INPUT_PATH = Path("data/tests/conversations_100.csv")
GOLDEN_DIRECTORY = Path("data/golden")
SORT_COLUMNS = {
    "sentiment": ["conversation_id"],
    "answer_check": ["conversation_id"],
    "intent": ["conversation_id", "intent", "first_detected_at"],
    "topic": ["conversation_id", "main_topic", "subtopic", "first_detected_at"],
    "language": ["conversation_id"],
}


def _canonicalize(name: str, output: pd.DataFrame) -> pd.DataFrame:
    result = output.copy()
    return result.sort_values(SORT_COLUMNS[name], kind="stable").reset_index(drop=True)


def _run_all() -> dict[str, pd.DataFrame]:
    data = momants_sentiment.load_momants_csv(INPUT_PATH)

    customer_messages = momants_sentiment.select_customer_messages(data)
    sentiment = momants_sentiment.create_conversation_summary(customer_messages)

    answer_check = momants_answer_check.create_conversation_summary(data)

    intent_messages = momants_intent.select_visitor_messages(data)
    classified_intents = momants_intent.classify_messages(intent_messages)
    intent = momants_intent.create_intent_summary(classified_intents)

    topics = momants_topic.load_topics()
    topic_messages = momants_topic.select_visitor_messages(data)
    classified_topics = momants_topic.classify_messages(topic_messages, topics)
    topic = momants_topic.build_topic_overview(classified_topics)

    language_messages = momants_language.select_visitor_messages(data)
    classified_languages = momants_language.classify_messages(language_messages)
    language = momants_language.create_conversation_summary(classified_languages)

    return {
        "sentiment": _canonicalize("sentiment", sentiment),
        "answer_check": _canonicalize("answer_check", answer_check),
        "intent": _canonicalize("intent", intent),
        "topic": _canonicalize("topic", topic),
        "language": _canonicalize("language", language),
    }


def _first_differences(
    expected: pd.DataFrame, actual: pd.DataFrame, limit: int = 10
) -> pd.DataFrame:
    width = max(len(expected), len(actual))
    expected_rows = expected.reindex(range(width))
    actual_rows = actual.reindex(range(width))
    different = ~(
        expected_rows.fillna("<NA>").astype(str)
        .eq(actual_rows.fillna("<NA>").astype(str))
        .all(axis=1)
    )
    indices = different[different].index[:limit]
    rows: list[dict[str, object]] = []
    for index in indices:
        rows.append(
            {
                "row": int(index),
                "expected": expected_rows.loc[index].to_dict(),
                "actual": actual_rows.loc[index].to_dict(),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    outputs = _run_all()
    GOLDEN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    changed = False

    for name, actual in outputs.items():
        golden_path = GOLDEN_DIRECTORY / f"{name}_golden.csv"
        if not golden_path.exists():
            print(f"{name}: MISSING — creating {golden_path}")
            actual.to_csv(golden_path, index=False, lineterminator="\n")
            continue

        expected = pd.read_csv(golden_path, keep_default_na=False)
        with tempfile.TemporaryDirectory() as directory:
            actual_path = Path(directory) / f"{name}.csv"
            actual.to_csv(actual_path, index=False, lineterminator="\n")
            actual_roundtrip = pd.read_csv(actual_path, keep_default_na=False)

        if expected.equals(actual_roundtrip):
            print(f"{name}: OK")
            continue

        changed = True
        print(f"{name}: CHANGED")
        print(_first_differences(expected, actual_roundtrip).to_string(index=False))

    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())