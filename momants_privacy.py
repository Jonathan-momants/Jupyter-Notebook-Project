"""Privacy helpers for free-text fields in Momants exports."""

from __future__ import annotations

import re


# Names are intentionally not masked: reliable name detection requires NER and can
# destroy valid questions. IBANs, ticket numbers, and order numbers are also outside
# the current scope by explicit design.
EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])"
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
    r"(?![\w.-])",
    flags=re.IGNORECASE,
)

PHONE_PATTERN = re.compile(
    r"(?<![\w])"
    r"(?:"
    r"(?:\+|00)(?:31|32|49)(?:[\s().-]*\d){8,11}"
    r"|"
    r"0[1-9](?:[\s.-]*\d){8}"
    r")"
    r"(?!\d)",
)


def mask_pii(text: str) -> tuple[str, int]:
    """Mask email addresses and Dutch/Belgian/German phone numbers."""
    masked, email_count = EMAIL_PATTERN.subn("[EMAIL]", text)
    masked, phone_count = PHONE_PATTERN.subn("[PHONE]", masked)
    return masked, email_count + phone_count