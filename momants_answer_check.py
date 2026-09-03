"""Bepaal per Momants-gesprek of vragen van bezoekers zijn beantwoord."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from momants_sentiment import load_momants_csv


CROSS_ENCODER_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RELEVANCE_THRESHOLD = -3.0

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

OUTPUT_COLUMNS = [
    "conversation_id",
    "aantal_vragen",
    "aantal_beantwoord",
    "percentage_beantwoord",
    "eindoordeel",
    "uitleg",
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
    readable_from_agent = data["from_agent"].eq(True) | data["from_agent"].eq(False)
    unreadable_count = int((~readable_from_agent).sum())
    print(f"Rows skipped with unreadable from_agent: {unreadable_count}")
    visitors = data.loc[data["from_agent"].eq(False)].copy()
    visitors = visitors.loc[visitors["text"].apply(is_question)].copy()
    visitors["text"] = visitors["text"].astype(str).str.strip()
    return visitors.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)


def pair_questions_with_answers(data: pd.DataFrame) -> pd.DataFrame:
    """Pair each visitor question with the following agent message."""
    readable_from_agent = data["from_agent"].eq(True) | data["from_agent"].eq(False)
    unreadable_count = int((~readable_from_agent).sum())
    print(f"Rows skipped with unreadable from_agent: {unreadable_count}")
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
            if message.from_agent is not False or not is_question(message.text):
                continue

            answer_text: str | None = None
            for following_message in messages[position + 1 :]:
                if following_message.created_at <= message.created_at:
                    continue
                if following_message.from_agent is True:
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


def classify_question_answer_pairs(
    pairs: pd.DataFrame,
    batch_size: int = 32,
    model: CrossEncoder | None = None,
) -> pd.DataFrame:
    """Bereken relevantiescores voor paren met een agentantwoord."""
    result = pairs.copy()
    result["answered"] = False
    result["relevance_score"] = pd.Series(index=result.index, dtype="float64")

    to_classify = result.index[result["has_agent_answer"]]
    if to_classify.empty:
        return result

    model = model or CrossEncoder(CROSS_ENCODER_MODEL_ID)
    question_answer_pairs = list(
        zip(
            result.loc[to_classify, "question_text"].astype(str),
            result.loc[to_classify, "answer_text"].astype(str),
        )
    )
    scores = np.asarray(
        model.predict(question_answer_pairs, batch_size=batch_size),
        dtype=float,
    ).reshape(-1)
    result.loc[to_classify, "relevance_score"] = scores
    result.loc[to_classify, "answered"] = scores >= RELEVANCE_THRESHOLD
    result["relevance_score"] = result["relevance_score"].round(4)
    return result


def _final_status(question_count: int, answered_count: int) -> str:
    if question_count == 0:
        return "Geen vragen gevonden"
    if answered_count == question_count:
        return "Beantwoord"
    if answered_count == 0:
        return "Niet beantwoord"
    return "Deels beantwoord"


def create_conversation_summary(
    data: pd.DataFrame,
    batch_size: int = 32,
    model: CrossEncoder | None = None,
) -> pd.DataFrame:
    """Maak één antwoordstatus per gesprek, inclusief gesprekken zonder vragen."""
    conversations = pd.Index(
        data["conversation_id"].drop_duplicates(), name="conversation_id"
    )
    pairs = pair_questions_with_answers(data)
    assessed = classify_question_answer_pairs(
        pairs,
        batch_size=batch_size,
        model=model,
    )

    counts = assessed.groupby("conversation_id", sort=False).agg(
        aantal_vragen=("conversation_id", "size"),
        aantal_beantwoord=("answered", "sum"),
    )
    summary = counts.reindex(conversations, fill_value=0).reset_index()
    summary["aantal_vragen"] = summary["aantal_vragen"].astype(int)
    summary["aantal_beantwoord"] = summary["aantal_beantwoord"].astype(int)
    summary["percentage_beantwoord"] = (
        summary["aantal_beantwoord"]
        .div(summary["aantal_vragen"].where(summary["aantal_vragen"].ne(0)))
        .mul(100)
        .fillna(0)
        .round(1)
    )
    summary["eindoordeel"] = summary.apply(
        lambda row: _final_status(
            int(row["aantal_vragen"]), int(row["aantal_beantwoord"])
        ),
        axis=1,
    )
    summary["uitleg"] = summary.apply(
        lambda row: (
            "Er zijn geen vragen van de bezoeker gevonden."
            if row["aantal_vragen"] == 0
            else (
                f"{int(row['aantal_beantwoord'])} van de "
                f"{int(row['aantal_vragen'])} "
                f"{'vraag is' if row['aantal_vragen'] == 1 else 'vragen zijn'} "
                "beantwoord."
            )
        ),
        axis=1,
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
    csv_output_path = output_path / f"antwoordcheck_per_gesprek_{timestamp}.csv"
    summary.to_csv(csv_output_path, index=False)
    summary.attrs["output_path"] = csv_output_path.resolve()
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controleer per Momants-gesprek of bezoekersvragen zijn beantwoord."
    )
    parser.add_argument("csv_path", type=Path, help="Pad naar de Momants CSV-export.")
    parser.add_argument(
        "--uitvoermap",
        dest="output_dir",
        type=Path,
        default=Path("results"),
        help="Map voor antwoordcheck_per_gesprek_<tijdstempel>.csv.",
    )
    parser.add_argument(
        "--batchgrootte",
        dest="batch_size",
        type=int,
        default=32,
        help="Aantal vraag-antwoordparen per modelbatch.",
    )
    parser.add_argument(
        "--alleen-controleren",
        dest="check_only",
        action="store_true",
        help="Controleer inladen en vraagdetectie zonder het model te starten.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)

    if arguments.check_only:
        data = load_momants_csv(arguments.csv_path)
        questions = detect_questions(data)
        print(f"Berichtrijen ingelezen: {len(data)}")
        print(f"Vragen gevonden: {len(questions)}")
        print(f"Gesprekken met vragen: {questions['conversation_id'].nunique()}")
        return 0

    summary = process_csv(
        csv_path=arguments.csv_path,
        output_dir=arguments.output_dir,
        batch_size=arguments.batch_size,
    )
    print(f"Gesprekresultaten: {len(summary)}")
    print(f"Uitvoer geschreven naar: {summary.attrs['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())