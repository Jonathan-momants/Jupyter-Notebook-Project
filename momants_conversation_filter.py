"""Shared message selection rules for all Momants analysis axes."""

from __future__ import annotations

import re

import pandas as pd


# Exact button payloads are customer-specific and must be reviewed per client.
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
BARE_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)


def normalize_message_type(value: object) -> str:
    """Normalize message_type without treating missing values as answers."""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def is_visitor_button_payload(value: object) -> bool:
    """Return whether text is an exact, customer-specific button payload."""
    if pd.isna(value):
        return False
    return str(value).strip().casefold() in VISITOR_BUTTON_PAYLOADS


def select_visitor_messages(
    data: pd.DataFrame,
    *,
    include_bare_urls: bool = False,
) -> pd.DataFrame:
    """Select usable visitor text consistently across all analysis axes."""
    from_agent = data["from_agent"]
    readable = from_agent.eq(True) | from_agent.eq(False)
    unreadable_count = int((~readable).sum())
    print(f"Rows skipped with unreadable from_agent: {unreadable_count}")

    text = data["text"].fillna("").astype(str).str.strip()
    button_payload = text.str.casefold().isin(VISITOR_BUTTON_PAYLOADS)
    mask = from_agent.eq(False) & text.ne("") & ~button_payload
    if not include_bare_urls:
        mask &= ~text.str.fullmatch(BARE_URL)

    selected = data.loc[mask].copy()
    selected["text"] = text.loc[mask]
    selected.attrs["unreadable_from_agent_count"] = unreadable_count
    return selected.sort_values(
        ["conversation_id", "created_at"], kind="stable"
    ).reset_index(drop=True)