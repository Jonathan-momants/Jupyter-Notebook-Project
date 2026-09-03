"""Detect Dutch, English, or German in privacy-safe Momants visitor messages."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

from lingua import Language, LanguageDetector, LanguageDetectorBuilder
import pandas as pd

from momants_sentiment import BARE_URL, load_momants_csv


LANGUAGE_CODES = {
    Language.DUTCH: "nl",
    Language.ENGLISH: "en",
    Language.GERMAN: "de",
}
LANGUAGES = ["nl", "en", "de", "und"]
MINIMUM_TEXT_LENGTH = 4
MINIMUM_CONFIDENCE = 0.55
MULTILINGUAL_SHARE = 0.25
OUTPUT_COLUMNS = ["conversation_id", "language", "confidence", "is_multilingual"]
MESSAGE_OUTPUT_COLUMNS = [
    "conversation_id",
    "created_at",
    "text",
    "language",
    "confidence",
]


def build_detector() -> LanguageDetector:
    """Build the deliberately restricted three-language Lingua detector."""
    return LanguageDetectorBuilder.from_languages(
        Language.DUTCH,
        Language.ENGLISH,
        Language.GERMAN,
    ).build()


def select_visitor_messages(data: pd.DataFrame) -> pd.DataFrame:
    """Keep non-empty visitor messages, including URLs for explicit und output."""
    from_agent = data["from_agent"]
    readable = from_agent.eq(True) | from_agent.eq(False)
    unreadable_count = int((~readable).sum())
    print(f"Rows skipped with unreadable from_agent: {unreadable_count}")
    text = data["text"].fillna("").astype(str).str.strip()
    mask = from_agent.eq(False) & text.ne("")
    selected = data.loc[mask].copy()
    selected["text"] = text.loc[mask]
    return selected.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)


def classify_messages(
    visitor_messages: pd.DataFrame,
    detector: LanguageDetector | None = None,
) -> pd.DataFrame:
    """Detect language per message and return und rather than a weak guess."""
    result = visitor_messages.copy()
    result["language"] = pd.Series("und", index=result.index, dtype="object")
    result["confidence"] = pd.Series(0.0, index=result.index, dtype="float64")
    if result.empty:
        return result

    detector = detector or build_detector()
    for index, value in result["text"].items():
        text = str(value).strip()
        if BARE_URL.fullmatch(text) or len(text) < MINIMUM_TEXT_LENGTH:
            continue
        confidence_values = detector.compute_language_confidence_values(text)
        if not confidence_values:
            continue
        best = confidence_values[0]
        confidence = float(best.value)
        result.at[index, "confidence"] = round(confidence, 4)
        if confidence >= MINIMUM_CONFIDENCE:
            result.at[index, "language"] = LANGUAGE_CODES[best.language]
    return result


def create_conversation_summary(classified_messages: pd.DataFrame) -> pd.DataFrame:
    """Choose the primary conversation language using determined character weights."""
    if classified_messages.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rows: list[dict[str, object]] = []
    for conversation_id, conversation in classified_messages.groupby(
        "conversation_id", sort=False
    ):
        determined = conversation.loc[conversation["language"].ne("und")].copy()
        if determined.empty:
            rows.append(
                {
                    "conversation_id": conversation_id,
                    "language": "und",
                    "confidence": 0.0,
                    "is_multilingual": False,
                }
            )
            continue

        determined["character_count"] = determined["text"].astype(str).str.len()
        weights = (
            determined.groupby("language", sort=False)["character_count"]
            .sum()
            .sort_values(ascending=False, kind="stable")
        )
        total = int(weights.sum())
        primary_language = str(weights.index[0])
        primary_share = float(weights.iloc[0] / total)
        second_share = float(weights.iloc[1] / total) if len(weights) > 1 else 0.0
        rows.append(
            {
                "conversation_id": conversation_id,
                "language": primary_language,
                "confidence": round(primary_share, 4),
                "is_multilingual": second_share > MULTILINGUAL_SHARE,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def print_customer_statistics(conversations: pd.DataFrame) -> None:
    """Print customer-facing conversation-language percentages and multilingual count."""
    print("Conversation language distribution:")
    total = len(conversations)
    counts = conversations["language"].value_counts() if total else pd.Series(dtype=int)
    for language in LANGUAGES:
        count = int(counts.get(language, 0))
        percentage = count / total * 100 if total else 0.0
        print(f"- {language}: {percentage:.1f}% ({count}/{total})")
    multilingual = int(conversations["is_multilingual"].sum()) if total else 0
    print(f"Multilingual conversations: {multilingual}")


def process_csv(
    csv_path: str | Path,
    output_directory: str | Path = "results",
) -> pd.DataFrame:
    """Detect visitor-message languages and write one row per conversation."""
    started_at = datetime.now().astimezone()
    data = load_momants_csv(csv_path)
    visitors = select_visitor_messages(data)
    classified = classify_messages(visitors)
    conversations = create_conversation_summary(classified)
    print_customer_statistics(conversations)

    destination = Path(output_directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    output_path = destination / f"language_per_conversation_{timestamp}.csv"
    message_output_path = destination / f"language_per_message_{timestamp}.csv"
    conversations.to_csv(output_path, index=False)
    message_output = classified[MESSAGE_OUTPUT_COLUMNS].copy()
    message_output["created_at"] = message_output["created_at"].apply(
        lambda value: value.isoformat()
    )
    message_output.to_csv(message_output_path, index=False)
    conversations.attrs["output_path"] = output_path.resolve()
    conversations.attrs["message_output_path"] = message_output_path.resolve()
    return conversations


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect Momants visitor languages.")
    parser.add_argument("csv_path", type=Path, help="Local Momants CSV export.")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results"),
        help="Directory for language_per_conversation_<timestamp>.csv.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    conversations = process_csv(args.csv_path, args.output_directory)
    print(f"Conversation results: {len(conversations)}")
    print(f"Output written to: {conversations.attrs['output_path']}")
    print(f"Message output written to: {conversations.attrs['message_output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())