"""Break down existing topic, intent, and sentiment validation accuracy by true language."""

from __future__ import annotations

import pandas as pd

import momants_intent
import momants_sentiment
import momants_topic
import validate_momants_language


def _language_lookup() -> pd.DataFrame:
    return validate_momants_language.load_answer_key()[["text", "language"]]


def _print_accuracy(name: str, evaluation: pd.DataFrame, correct_column: str) -> None:
    print(f"\n{name.upper()} ACCURACY BY TRUE LANGUAGE")
    for language in validate_momants_language.LABELS:
        selected = evaluation.loc[evaluation["language"].eq(language)]
        if selected.empty:
            print(f"- {language}: n=0 (not present)")
        else:
            print(
                f"- {language}: {selected[correct_column].mean() * 100:.1f}% "
                f"(n={len(selected)})"
            )


def topic_breakdown(languages: pd.DataFrame) -> pd.DataFrame:
    key = pd.read_csv(
        "data/tests/answer_key_extra158.csv", keep_default_na=False
    )
    topics = momants_topic.load_topics()
    predicted = momants_topic.classify_messages(key[["text"]].copy(), topics)
    evaluation = key[["text", "main_topic"]].rename(
        columns={"main_topic": "true_main_topic"}
    )
    evaluation["predicted_main_topic"] = predicted["main_topic"].tolist()
    evaluation = evaluation.merge(languages, on="text", how="left", validate="one_to_one")
    evaluation["correct"] = evaluation["true_main_topic"].eq(
        evaluation["predicted_main_topic"]
    )
    _print_accuracy("topic", evaluation, "correct")
    return evaluation


def intent_breakdown(languages: pd.DataFrame) -> pd.DataFrame:
    key = pd.read_csv(
        "data/tests/answer_key_v2.csv",
        usecols=["text", "intent"],
        keep_default_na=False,
    )
    key["intent"] = key["intent"].replace("", "None")
    predicted = momants_intent.classify_messages(key[["text"]].copy())
    evaluation = key.rename(columns={"intent": "true_intent"})
    evaluation["predicted_intent"] = predicted["intent"].tolist()
    evaluation = evaluation.merge(languages, on="text", how="left", validate="one_to_one")
    evaluation["correct"] = evaluation["true_intent"].eq(
        evaluation["predicted_intent"]
    )
    _print_accuracy("intent", evaluation, "correct")
    return evaluation


def sentiment_breakdown(languages: pd.DataFrame) -> pd.DataFrame:
    key = pd.read_csv(
        "data/tests/answer_key_sentiment.csv",
        usecols=["text", "sentiment"],
        keep_default_na=False,
    )
    predicted = momants_sentiment.classify_messages(key[["text"]].copy())
    evaluation = key.rename(columns={"sentiment": "true_sentiment"})
    evaluation["predicted_sentiment"] = predicted["message_sentiment"].replace(
        {"Neutral (task-focused)": "Neutral (task-oriented)"}
    ).tolist()
    evaluation = evaluation.merge(languages, on="text", how="left", validate="one_to_one")
    evaluation["correct"] = evaluation["true_sentiment"].eq(
        evaluation["predicted_sentiment"]
    )
    _print_accuracy("sentiment", evaluation, "correct")
    return evaluation


def main() -> None:
    languages = _language_lookup()
    topic_breakdown(languages)
    intent_breakdown(languages)
    sentiment_breakdown(languages)


if __name__ == "__main__":
    main()