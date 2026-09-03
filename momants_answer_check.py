"""Bepaal per Momants-gesprek of informatiebehoeften zijn beantwoord."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from momants_intent import classify_messages, select_visitor_messages
from momants_sentiment import load_momants_csv


CROSS_ENCODER_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RELEVANCE_THRESHOLD = -3.0

OUTPUT_COLUMNS = [
    "conversation_id",
    "aantal_vragen",
    "aantal_beantwoord",
    "percentage_beantwoord",
    "telt_mee",
    "eindoordeel",
    "uitleg",
]

# Exact button payloads are customer-specific and must be reviewed for each client.
VISITOR_BUTTON_PAYLOADS = frozenset(
    {
        "yes! (opt in)",
        "see saturday's recap",
        "see sunday's recap",
        "stop messaging",
        "opt-in again",
    }
)
ANSWER_MESSAGE_TYPES = frozenset({"LLM_RESPONSE", "REPLY_TAKEOVER"})

FALLBACK_PATTERNS = (
    (
        "nl_geen_informatie",
        re.compile(
            r"\b(?:ik\s+heb(?:\s+hier)?|heb\s+ik)\s+(?:helaas\s+)?geen\s+"
            r"(?:(?:specifieke|exacte|gedetailleerde)\s+)?"
            r"(?:informatie|locatie|details|gegevens|openingstijden|tijden|update)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "nl_niet_in_kennis",
        re.compile(
            r"\bstaat\s+niet\s+in\s+mijn\s+(?:informatie|kennis)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "nl_geen_specifieke_locatie",
        re.compile(
            r"\bgeen\s+specifieke\s+locatie\s+in\s+mijn\s+informatie\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "nl_kan_niet_precies",
        re.compile(
            r"\b(?:kan\s+ik\s+(?:je\s+)?|ik\s+kan\s+(?:je\s+)?)"
            r"(?:helaas\s+)?niet\s+(?:precies\s+)?"
            r"(?:vertellen|vinden|zeggen)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "nl_kan_zo_snel_geen",
        re.compile(
            r"\b(?:kan\s+ik|ik\s+kan)\s+zo\s+snel\s+geen\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "nl_weet_ik_niet",
        re.compile(r"\bweet\s+ik\s+(?:helaas\s+)?niet\b", flags=re.IGNORECASE),
    ),
    (
        "en_i_do_not_have",
        re.compile(
            r"\bi\s+(?:don't|do\s+not)\s+have\s+(?!to\b)"
            r"(?:the\s+)?(?:(?:exact|specific|any)\s+)?"
            r"(?:information|details|data|times|update|departure\s+times)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "en_no_specific_information",
        re.compile(
            r"\b(?:looks\s+like\s+)?i\s+don't\s+have\s+specific\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "it_no_specific_information",
        re.compile(r"\bnon\s+ho\s+informazioni\s+specifiche\b", flags=re.IGNORECASE),
    ),
)

PROMISE_PATTERNS = (
    (
        "nl_ik_ga_kijken",
        re.compile(
            r"\bik\s+ga\s+(?:(?:meteen|even)\s+)?(?:voor\s+je\s+)?"
            r"(?:kijken|zoeken)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "nl_zoek_ik_op",
        re.compile(r"\bzoek\s+ik\s+(?:even\s+)?op\b", flags=re.IGNORECASE),
    ),
    (
        "nl_kom_erop_terug",
        re.compile(
            r"\bkom\s+er(?:op)?\s+(?:bij\s+je\s+)?terug\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "en_let_me_check",
        re.compile(r"\b(?:let\s+me\s+check|i['’]ll\s+check)\b", flags=re.IGNORECASE),
    ),
)

EXTERNAL_CHANNEL_PATTERN = re.compile(
    r"\b(?:mail|e-mail|email|bel|call|whatsapp|servicenummer|telefoonnummer|"
    r"contact\s+op(?:nemen)?|neem\s+contact\s+op|website)\b",
    flags=re.IGNORECASE,
)
ON_SITE_ACTION_PATTERN = re.compile(
    r"\b(?:lost\s*(?:&|and)\s*found|lockerdesk|hoofdingang|festivalterrein|"
    r"balie|desk|punt|ingang|ter\s+plaatse|on[\s-]?site)\b",
    flags=re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _normalized_message_type(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _with_message_order(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)
    ordered["_message_order"] = range(len(ordered))
    return ordered


def detect_questions(
    data: pd.DataFrame,
    batch_size: int = 32,
    intent_model: object | None = None,
) -> pd.DataFrame:
    """Use the existing intent model to select visitor information needs."""
    ordered = _with_message_order(data)
    visitors = select_visitor_messages(ordered)
    normalized_text = visitors["text"].astype(str).str.strip().str.casefold()
    visitors = visitors.loc[
        ~normalized_text.isin(VISITOR_BUTTON_PAYLOADS)
    ].reset_index(drop=True)
    classified = classify_messages(
        visitors,
        batchgrootte=batch_size,
        model=intent_model,
    )
    return classified.loc[classified["intent"].ne("None")].reset_index(drop=True)


def pair_questions_with_answers(
    data: pd.DataFrame,
    questions: pd.DataFrame | None = None,
    batch_size: int = 32,
    intent_model: object | None = None,
) -> pd.DataFrame:
    """Build one episode per information need using only valid agent replies."""
    ordered = _with_message_order(data)
    if questions is None:
        questions = detect_questions(
            ordered,
            batch_size=batch_size,
            intent_model=intent_model,
        )
    columns = [
        "conversation_id",
        "question_text",
        "answer_text",
        "has_agent_answer",
        "has_reply_takeover",
        "llm_responses",
    ]
    pairs: list[dict[str, object]] = []
    for conversation_id, conversation_questions in questions.groupby(
        "conversation_id", sort=False
    ):
        conversation_messages = ordered.loc[
            ordered["conversation_id"].eq(conversation_id)
        ]
        question_rows = conversation_questions.sort_values(
            ["created_at", "_message_order"], kind="stable"
        ).to_dict(
            orient="records"
        )
        for question_index, question in enumerate(question_rows):
            next_order = (
                question_rows[question_index + 1]["_message_order"]
                if question_index + 1 < len(question_rows)
                else float("inf")
            )
            episode = conversation_messages.loc[
                conversation_messages["_message_order"].gt(question["_message_order"])
                & conversation_messages["_message_order"].lt(next_order)
                & conversation_messages["created_at"].gt(question["created_at"])
                & conversation_messages["from_agent"].eq(True)
            ].copy()
            episode["_normalized_type"] = episode["message_type"].map(
                _normalized_message_type
            )
            episode = episode.loc[
                episode["_normalized_type"].isin(ANSWER_MESSAGE_TYPES)
            ]
            replies = [
                str(value).strip()
                for value in episode["text"]
                if pd.notna(value) and str(value).strip()
            ]
            llm_responses = [
                str(text).strip()
                for text, message_type in zip(
                    episode["text"], episode["_normalized_type"]
                )
                if message_type == "LLM_RESPONSE"
                and pd.notna(text)
                and str(text).strip()
            ]
            has_takeover = episode["_normalized_type"].eq("REPLY_TAKEOVER").any()
            pairs.append(
                {
                    "conversation_id": conversation_id,
                    "question_text": str(question["text"]).strip(),
                    "answer_text": " ||| ".join(replies) if replies else None,
                    "has_agent_answer": bool(replies) or bool(has_takeover),
                    "has_reply_takeover": bool(has_takeover),
                    "llm_responses": llm_responses,
                }
            )

    return pd.DataFrame(pairs, columns=columns)


def _matching_pattern_names(
    text: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[str]:
    return [name for name, pattern in patterns if pattern.search(text)]


def _has_fallback(responses: list[str]) -> bool:
    return any(
        _matching_pattern_names(response, FALLBACK_PATTERNS)
        for response in responses
    )


def _is_promise_without_answer(responses: list[str]) -> bool:
    """Treat promises as unanswered only when no other substantive sentence exists."""
    if not responses:
        return False
    saw_promise = False
    saw_substantive_non_promise = False
    for response in responses:
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_SPLIT.split(response)
            if sentence.strip()
        ]
        for sentence in sentences:
            if _matching_pattern_names(sentence, PROMISE_PATTERNS):
                saw_promise = True
            elif len(re.findall(r"\w+", sentence)) >= 3:
                saw_substantive_non_promise = True
    return saw_promise and not saw_substantive_non_promise


def _is_external_channel_only(responses: list[str]) -> bool:
    combined = " ".join(responses)
    return bool(
        EXTERNAL_CHANNEL_PATTERN.search(combined)
        and not ON_SITE_ACTION_PATTERN.search(combined)
    )


def classify_question_answer_pairs(
    pairs: pd.DataFrame,
    batch_size: int = 32,
    model: CrossEncoder | None = None,
) -> pd.DataFrame:
    """Apply the ordered answer rules and cross-encoder knowledge check."""
    result = pairs.copy()
    result["answered"] = False
    result["relevance_score"] = pd.Series(index=result.index, dtype="float64")
    result["answer_rule"] = "geen_geldig_agentantwoord"

    has_takeover = result["has_reply_takeover"].eq(True)
    result.loc[has_takeover, "answered"] = True
    result.loc[has_takeover, "answer_rule"] = "reply_takeover"

    has_llm_response = result["llm_responses"].apply(bool)
    to_classify = result.index[~has_takeover & has_llm_response]
    if not to_classify.empty:
        model = model or CrossEncoder(CROSS_ENCODER_MODEL_ID)
        combined_llm_answers = result.loc[to_classify, "llm_responses"].apply(
            lambda responses: " ||| ".join(responses)
        )
        question_answer_pairs = list(
            zip(
                result.loc[to_classify, "question_text"].astype(str),
                combined_llm_answers,
            )
        )
        scores = np.asarray(
            model.predict(question_answer_pairs, batch_size=batch_size),
            dtype=float,
        ).reshape(-1)
        result.loc[to_classify, "relevance_score"] = scores
        result.loc[to_classify, "answered"] = scores >= RELEVANCE_THRESHOLD
        result.loc[to_classify, "answer_rule"] = "llm_response"

    fallback = ~has_takeover & result["llm_responses"].apply(_has_fallback)
    result.loc[fallback, "answered"] = False
    result.loc[fallback, "answer_rule"] = "fallback"

    external_channel = (
        ~has_takeover
        & ~fallback
        & result["llm_responses"].apply(_is_external_channel_only)
    )
    result.loc[external_channel, "answered"] = False
    result.loc[external_channel, "answer_rule"] = "extern_kanaal"

    promise_only = (
        ~has_takeover
        & ~fallback
        & ~external_channel
        & result["llm_responses"].apply(_is_promise_without_answer)
    )
    result.loc[promise_only, "answered"] = False
    result.loc[promise_only, "answer_rule"] = "belofte_zonder_antwoord"
    result["relevance_score"] = result["relevance_score"].round(4)
    return result


def collect_pattern_sentences(data: pd.DataFrame) -> pd.DataFrame:
    """Collect and count LLM_RESPONSE sentences matching fallback/promise rules."""
    agent = data.loc[data["from_agent"].eq(True)].copy()
    agent["_normalized_type"] = agent["message_type"].map(_normalized_message_type)
    llm_responses = agent.loc[
        agent["_normalized_type"].eq("LLM_RESPONSE"), "text"
    ]
    matched_rows: list[dict[str, str]] = []
    pattern_groups = (
        ("fallback", FALLBACK_PATTERNS),
        ("belofte", PROMISE_PATTERNS),
    )
    for value in llm_responses:
        if pd.isna(value) or not str(value).strip():
            continue
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_SPLIT.split(str(value))
            if sentence.strip()
        ]
        for sentence in sentences:
            for pattern_type, patterns in pattern_groups:
                for pattern_name in _matching_pattern_names(sentence, patterns):
                    matched_rows.append(
                        {
                            "zin": sentence,
                            "patroon_dat_matchte": pattern_name,
                            "type": pattern_type,
                        }
                    )

    columns = [
        "zin",
        "patroon_dat_matchte",
        "type",
        "aantal_keer_voorgekomen",
    ]
    if not matched_rows:
        return pd.DataFrame(columns=columns)
    matched = pd.DataFrame(matched_rows)
    return (
        matched.groupby(
            ["zin", "patroon_dat_matchte", "type"],
            sort=False,
            dropna=False,
        )
        .size()
        .rename("aantal_keer_voorgekomen")
        .reset_index()[columns]
    )


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
    intent_model: object | None = None,
) -> pd.DataFrame:
    """Maak één antwoordstatus per gesprek, inclusief gesprekken zonder vragen."""
    conversations = pd.Index(
        data["conversation_id"].drop_duplicates(), name="conversation_id"
    )
    questions = detect_questions(
        data,
        batch_size=batch_size,
        intent_model=intent_model,
    )
    pairs = pair_questions_with_answers(
        data,
        questions=questions,
        batch_size=batch_size,
        intent_model=intent_model,
    )
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
        .round(1)
    )
    summary["telt_mee"] = summary["aantal_vragen"].gt(0)
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
    pattern_sentences = collect_pattern_sentences(data)
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    csv_output_path = output_path / f"antwoordcheck_per_gesprek_{timestamp}.csv"
    pattern_output_path = output_path / "fallback_zinnen.csv"
    summary.to_csv(csv_output_path, index=False)
    pattern_sentences.to_csv(pattern_output_path, index=False)
    summary.attrs["output_path"] = csv_output_path.resolve()
    summary.attrs["pattern_output_path"] = pattern_output_path.resolve()
    summary.attrs["pattern_row_count"] = len(pattern_sentences)
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