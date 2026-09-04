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
    r"(?:\b(?:mail(?:en)?\s+naar|e-?mailadres|servicenummer|klantenservice|"
    r"neem\s+contact\s+op|contact\s+us|reach\s+out\s+to|"
    r"website\s+in\s+de\s+gaten\s+houden)\b|"
    r"(?<!\w)[\w.+-]+@[\w.-]+\.[a-z]{2,}(?!\w))",
    flags=re.IGNORECASE,
)
ON_SITE_ACTION_PATTERN = re.compile(
    r"\b(?:lost\s*(?:&|and)\s*found|informatiepunt|info\s+point|ticketpunt|"
    r"ticket\s+desk|ticketpunkt|lockerdesk|kluisjespunt|cashless-balie|"
    r"statiegeldpunt|ehbo|first\s+aid|crewlid|crew\s+member|balie|desk|"
    r"hoofdingang|main\s+entrance|haupteingang|ingang\s+noord|ingang\s+zuid|"
    r"entrance\s+north|entrance\s+south|bel\s+112|bel\s+de\s+beveiliging)\b",
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
    classified = _classify_visitor_messages(
        data,
        batch_size=batch_size,
        intent_model=intent_model,
    )
    return classified.loc[classified["intent"].ne("None")].reset_index(drop=True)


def _classify_visitor_messages(
    data: pd.DataFrame,
    batch_size: int = 32,
    intent_model: object | None = None,
) -> pd.DataFrame:
    """Classify all usable non-button visitor messages once for episode grouping."""
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
    return classified.reset_index(drop=True)


def _agent_asks_followup(text: object) -> bool:
    if pd.isna(text):
        return False
    cleaned = str(text).strip().casefold()
    return bool(
        cleaned.endswith("?")
        or any(
            phrase in cleaned
            for phrase in (
                "wil je ook weten",
                "kan ik je ook",
                "zal ik",
                "would you like",
                "shall i",
            )
        )
    )


def pair_questions_with_answers(
    data: pd.DataFrame,
    questions: pd.DataFrame | None = None,
    classified_visitors: pd.DataFrame | None = None,
    batch_size: int = 32,
    intent_model: object | None = None,
) -> pd.DataFrame:
    """Group visitor followups and valid agent replies into question episodes."""
    ordered = _with_message_order(data)
    if classified_visitors is None:
        classified_visitors = _classify_visitor_messages(
            ordered,
            batch_size=batch_size,
            intent_model=intent_model,
        )
    columns = [
        "episode_id",
        "conversation_id",
        "question_text",
        "answer_text",
        "has_agent_answer",
        "has_reply_takeover",
        "llm_responses",
        "visitor_message_count",
    ]
    episodes: list[dict[str, object]] = []
    visitor_by_order = {
        int(row["_message_order"]): row
        for row in classified_visitors.to_dict(orient="records")
    }

    for conversation_id, conversation_messages in ordered.groupby(
        "conversation_id", sort=False
    ):
        current: dict[str, object] | None = None
        last_valid_agent_asked_followup = False

        def finish_current() -> None:
            nonlocal current
            if current is not None:
                episodes.append(current)
                current = None

        for message in conversation_messages.to_dict(orient="records"):
            message_order = int(message["_message_order"])
            visitor = visitor_by_order.get(message_order)
            if visitor is not None:
                intent = str(visitor["intent"])
                if current is None:
                    if intent != "None":
                        current = {
                            "episode_id": len(episodes) + 1,
                            "conversation_id": conversation_id,
                            "_intent": intent,
                            "_visitor_messages": [str(visitor["text"]).strip()],
                            "_replies": [],
                            "has_reply_takeover": False,
                            "llm_responses": [],
                        }
                else:
                    has_agent_answer = bool(current["_replies"]) or bool(
                        current["has_reply_takeover"]
                    )
                    is_followup = (
                        intent == "None"
                        or last_valid_agent_asked_followup
                        or (
                            has_agent_answer
                            and intent == str(current["_intent"])
                        )
                    )
                    if is_followup:
                        current["_visitor_messages"].append(
                            str(visitor["text"]).strip()
                        )
                    else:
                        finish_current()
                        current = {
                            "episode_id": len(episodes) + 1,
                            "conversation_id": conversation_id,
                            "_intent": intent,
                            "_visitor_messages": [str(visitor["text"]).strip()],
                            "_replies": [],
                            "has_reply_takeover": False,
                            "llm_responses": [],
                        }
                last_valid_agent_asked_followup = False
                continue

            if message["from_agent"] is not True or current is None:
                continue
            message_type = _normalized_message_type(message["message_type"])
            if message_type not in ANSWER_MESSAGE_TYPES:
                continue
            text = (
                str(message["text"]).strip()
                if pd.notna(message["text"]) and str(message["text"]).strip()
                else ""
            )
            if text:
                current["_replies"].append(text)
            if message_type == "REPLY_TAKEOVER":
                current["has_reply_takeover"] = True
            elif message_type == "LLM_RESPONSE" and text:
                current["llm_responses"].append(text)
            last_valid_agent_asked_followup = _agent_asks_followup(text)

        finish_current()

    rows: list[dict[str, object]] = []
    for episode_id, episode in enumerate(episodes, start=1):
        replies = list(episode.pop("_replies"))
        visitor_messages = list(episode.pop("_visitor_messages"))
        episode.pop("_intent")
        rows.append(
            {
                **episode,
                "episode_id": episode_id,
                "question_text": " ||| ".join(visitor_messages),
                "answer_text": " ||| ".join(replies) if replies else None,
                "has_agent_answer": bool(replies)
                or bool(episode["has_reply_takeover"]),
                "visitor_message_count": len(visitor_messages),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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
    return bool(EXTERNAL_CHANNEL_PATTERN.search(combined))


def _has_on_site_action(responses: list[str]) -> bool:
    return bool(ON_SITE_ACTION_PATTERN.search(" ".join(responses)))


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
    result["has_fallback_or_promise"] = False

    has_takeover = result["has_reply_takeover"].eq(True)
    result.loc[has_takeover, "answered"] = True
    result.loc[has_takeover, "answer_rule"] = "reply_takeover"

    response_rows: list[dict[str, object]] = []
    for pair_index in result.index[~has_takeover]:
        for response in result.at[pair_index, "llm_responses"]:
            response_rows.append(
                {
                    "pair_index": pair_index,
                    "question_text": str(result.at[pair_index, "question_text"]),
                    "response": response,
                }
            )

    if response_rows:
        model = model or CrossEncoder(CROSS_ENCODER_MODEL_ID)
        question_answer_pairs = list(
            (row["question_text"], row["response"]) for row in response_rows
        )
        scores = np.asarray(
            model.predict(question_answer_pairs, batch_size=batch_size),
            dtype=float,
        ).reshape(-1)
        assessments: dict[int, list[dict[str, object]]] = {}
        for row, score in zip(response_rows, scores):
            response = str(row["response"])
            fallback = _has_fallback([response])
            promise = _is_promise_without_answer([response])
            on_site_action = _has_on_site_action([response])
            external_channel = _is_external_channel_only([response])
            response_answered = bool(
                not fallback
                and not promise
                and (
                    on_site_action
                    or (
                        not external_channel
                        and score >= RELEVANCE_THRESHOLD
                    )
                )
            )
            assessments.setdefault(int(row["pair_index"]), []).append(
                {
                    "score": float(score),
                    "answered": response_answered,
                    "fallback": fallback,
                    "promise": promise,
                    "on_site_action": on_site_action,
                    "external_channel": external_channel,
                }
            )

        for pair_index, response_assessments in assessments.items():
            result.at[pair_index, "relevance_score"] = max(
                assessment["score"] for assessment in response_assessments
            )
            result.at[pair_index, "has_fallback_or_promise"] = any(
                assessment["fallback"] or assessment["promise"]
                for assessment in response_assessments
            )
            if any(
                assessment["answered"] for assessment in response_assessments
            ):
                result.at[pair_index, "answered"] = True
                result.at[pair_index, "answer_rule"] = (
                    "on_site_doorverwijzing"
                    if any(
                        assessment["answered"]
                        and assessment["on_site_action"]
                        for assessment in response_assessments
                    )
                    else "llm_response"
                )
            elif any(
                assessment["fallback"] for assessment in response_assessments
            ):
                result.at[pair_index, "answer_rule"] = "fallback"
            elif any(
                assessment["promise"] for assessment in response_assessments
            ):
                result.at[pair_index, "answer_rule"] = "belofte_zonder_antwoord"
            elif any(
                assessment["external_channel"]
                for assessment in response_assessments
            ):
                result.at[pair_index, "answer_rule"] = "extern_kanaal"
            else:
                result.at[pair_index, "answer_rule"] = "llm_response_afgewezen"

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
    classified_visitors = _classify_visitor_messages(
        data,
        batch_size=batch_size,
        intent_model=intent_model,
    )
    pairs = pair_questions_with_answers(
        data,
        classified_visitors=classified_visitors,
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
    output = summary[OUTPUT_COLUMNS].copy()
    message_counts = pairs["visitor_message_count"]
    output.attrs["episode_count"] = len(pairs)
    output.attrs["episode_message_distribution"] = {
        "1": int(message_counts.eq(1).sum()),
        "2": int(message_counts.eq(2).sum()),
        "3": int(message_counts.eq(3).sum()),
        "4+": int(message_counts.ge(4).sum()),
    }
    output.attrs["unanswered_without_fallback_or_promise"] = int(
        (~assessed["answered"] & ~assessed["has_fallback_or_promise"]).sum()
    )
    output.attrs["episodes_without_agent_answer"] = int(
        (~pairs["has_agent_answer"]).sum()
    )
    output.attrs["answer_rule_counts"] = {
        str(rule): int(count)
        for rule, count in assessed["answer_rule"].value_counts().items()
    }
    return output


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