"""Determine whether visitor questions were answered in each Momants conversation."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from momants_sentiment import load_momants_csv


MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
HYPOTHESIS = "This answers the visitor's question."
ENTAILMENT_THRESHOLD = 0.50
# Temporary: set to False to write only the privacy-conscious summary again.
INCLUDE_CONVERSATION_TEXT = True

DUTCH_QUESTION_STARTS = (
    "wie",
    "wat",
    "waar",
    "wanneer",
    "hoe",
    "waarom",
    "welke",
    "mag",
    "kan",
    "is er",
    "zijn er",
)

BASE_OUTPUT_COLUMNS = [
    "conversation_id",
    "question_count",
    "answered_count",
    "answered_percentage",
    "status",
    "explanation",
]
OUTPUT_COLUMNS = [
    *BASE_OUTPUT_COLUMNS,
    *(["conversation_text"] if INCLUDE_CONVERSATION_TEXT else []),
]

BARE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
QUESTION_START_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(word) for word in DUTCH_QUESTION_STARTS) + r")\b",
    flags=re.IGNORECASE,
)


def is_question(text: object) -> bool:
    """Recognize a non-empty question by a question mark or Dutch question start."""
    if pd.isna(text):
        return False
    cleaned = str(text).strip()
    if not cleaned or BARE_URL.fullmatch(cleaned):
        return False
    return "?" in cleaned or QUESTION_START_PATTERN.match(cleaned) is not None


def detect_questions(data: pd.DataFrame) -> pd.DataFrame:
    """Select usable visitor questions in chronological order."""
    visitors = data.loc[data["from_agent"].eq(False)].copy()
    visitors = visitors.loc[visitors["text"].apply(is_question)].copy()
    visitors["text"] = visitors["text"].astype(str).str.strip()
    return visitors.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)


def pair_questions_with_answers(data: pd.DataFrame) -> pd.DataFrame:
    """Pair each visitor question with the following agent message."""
    columns = [
        "conversation_id",
        "question_text",
        "answer_text",
        "has_agent_answer",
    ]
    pairs: list[dict[str, object]] = []

    sorted_data = data.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    )
    for conversation_id, conversation in sorted_data.groupby(
        "conversation_id", sort=False
    ):
        messages = list(conversation.itertuples(index=False))
        for position, message in enumerate(messages):
            if bool(message.from_agent) or not is_question(message.text):
                continue

            answer_text: str | None = None
            for following_message in messages[position + 1 :]:
                if following_message.created_at <= message.created_at:
                    continue
                if bool(following_message.from_agent):
                    if pd.notna(following_message.text) and str(following_message.text).strip():
                        answer_text = str(following_message.text).strip()
                    break

            pairs.append(
                {
                    "conversation_id": conversation_id,
                    "question_text": str(message.text).strip(),
                    "answer_text": answer_text,
                    "has_agent_answer": answer_text is not None,
                }
            )

    return pd.DataFrame(pairs, columns=columns)


def _label_indices(model: AutoModelForSequenceClassification) -> dict[str, int]:
    """Find the three NLI label indices in the model configuration."""
    labels = {
        str(label).strip().lower(): int(index)
        for index, label in model.config.id2label.items()
    }
    missing = {"entailment", "neutral", "contradiction"} - set(labels)
    if missing:
        raise ValueError(
            "The model configuration is missing NLI labels: "
            f"{', '.join(sorted(missing))}."
        )
    return labels


def classify_question_answer_pairs(
    pairs: pd.DataFrame,
    batch_size: int = 32,
    tokenizer: AutoTokenizer | None = None,
    model: AutoModelForSequenceClassification | None = None,
) -> pd.DataFrame:
    """Calculate NLI scores for pairs that have an agent answer."""
    result = pairs.copy()
    result["answered"] = False
    result["entailment_score"] = pd.Series(index=result.index, dtype="float64")

    to_classify = result.index[result["has_agent_answer"]]
    if to_classify.empty:
        return result

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(MODEL_ID)
    model = model or AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    indices = _label_indices(model)

    for start in range(0, len(to_classify), batch_size):
        batch_indices = to_classify[start : start + batch_size]
        answers = result.loc[batch_indices, "answer_text"].tolist()
        hypotheses = [HYPOTHESIS] * len(answers)
        inputs = tokenizer(
            answers,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()

        entailment = probabilities[:, indices["entailment"]]
        neutral = probabilities[:, indices["neutral"]]
        contradiction = probabilities[:, indices["contradiction"]]
        answered = (
            entailment.ge(ENTAILMENT_THRESHOLD)
            & entailment.gt(neutral)
            & entailment.gt(contradiction)
        )
        result.loc[batch_indices, "entailment_score"] = entailment.tolist()
        result.loc[batch_indices, "answered"] = answered.tolist()

    result["entailment_score"] = result["entailment_score"].round(4)
    return result


def _final_status(question_count: int, answered_count: int) -> str:
    if question_count == 0:
        return "No questions found"
    if answered_count == question_count:
        return "Answered"
    if answered_count == 0:
        return "Not answered"
    return "Partially answered"


def _build_conversation_texts(data: pd.DataFrame) -> pd.Series:
    """Temporarily combine visitor and agent text per conversation in time order."""
    sorted_data = data.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).copy()
    has_text = sorted_data["text"].notna() & sorted_data["text"].astype(str).str.strip().ne("")
    sorted_data = sorted_data.loc[has_text].copy()
    sorted_data["line"] = (
        sorted_data["from_agent"]
        .map({True: "Agent", False: "Visitor"})
        .fillna("Unknown")
        + ": "
        + sorted_data["text"].astype(str).str.strip()
    )
    return sorted_data.groupby("conversation_id", sort=False)["line"].agg("\n".join)


def create_conversation_summary(
    data: pd.DataFrame,
    batch_size: int = 32,
    tokenizer: AutoTokenizer | None = None,
    model: AutoModelForSequenceClassification | None = None,
) -> pd.DataFrame:
    """Create one answer status per conversation, including conversations without questions."""
    conversations = pd.Index(
        data["conversation_id"].drop_duplicates(), name="conversation_id"
    )
    pairs = pair_questions_with_answers(data)
    assessed = classify_question_answer_pairs(
        pairs,
        batch_size=batch_size,
        tokenizer=tokenizer,
        model=model,
    )

    counts = assessed.groupby("conversation_id", sort=False).agg(
        question_count=("conversation_id", "size"),
        answered_count=("answered", "sum"),
    )
    summary = counts.reindex(conversations, fill_value=0).reset_index()
    summary["question_count"] = summary["question_count"].astype(int)
    summary["answered_count"] = summary["answered_count"].astype(int)
    summary["answered_percentage"] = (
        summary["answered_count"]
        .div(summary["question_count"].where(summary["question_count"].ne(0)))
        .mul(100)
        .fillna(0)
        .round(1)
    )
    summary["status"] = summary.apply(
        lambda row: _final_status(
            int(row["question_count"]), int(row["answered_count"])
        ),
        axis=1,
    )
    summary["explanation"] = summary.apply(
        lambda row: (
            "No visitor questions were found."
            if row["question_count"] == 0
            else (
                f"{int(row['answered_count'])} of "
                f"{int(row['question_count'])} "
                f"{'question was' if row['question_count'] == 1 else 'questions were'} "
                "answered."
            )
        ),
        axis=1,
    )
    if INCLUDE_CONVERSATION_TEXT:
        conversation_texts = _build_conversation_texts(data)
        summary["conversation_text"] = (
            summary["conversation_id"].map(conversation_texts).fillna("")
        )
    return summary[OUTPUT_COLUMNS]


def process_csv(
    csv_path: str | Path,
    output_dir: str | Path = "results",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Process a local export and write one safe conversation table."""
    started_at = datetime.now().astimezone()
    data = load_momants_csv(csv_path)
    summary = create_conversation_summary(data, batch_size=batch_size)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    csv_output_path = output_path / f"answer_check_per_conversation_{timestamp}.csv"
    summary.to_csv(csv_output_path, index=False)
    summary.attrs["output_path"] = csv_output_path.resolve()
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether visitor questions were answered in each Momants conversation."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the Momants CSV export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for answer_check_per_conversation_<timestamp>.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of question-answer pairs per model batch.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check loading and question detection without starting the model.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)

    if arguments.check_only:
        data = load_momants_csv(arguments.csv_path)
        questions = detect_questions(data)
        print(f"Message rows loaded: {len(data)}")
        print(f"Questions found: {len(questions)}")
        print(f"Conversations with questions: {questions['conversation_id'].nunique()}")
        return 0

    summary = process_csv(
        csv_path=arguments.csv_path,
        output_dir=arguments.output_dir,
        batch_size=arguments.batch_size,
    )
    print(f"Conversation results: {len(summary)}")
    print(f"Output written to: {summary.attrs['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())