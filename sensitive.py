"""
Sensitive Information Detection & Masking
-------------------------------------------
Messages contain sensitive data written OPENLY in plain text
(e.g. "my card number is 112348865966", "your OTP is 4821").
This module finds those patterns with regex, classifies risk,
and produces a masked version. Nothing here calls an external
service — everything runs locally on the message string.
"""
import re

# Order matters: more specific patterns first so a card number
# isn't accidentally also flagged as a generic "PIN".
PATTERNS = [
    {
        "type": "card_number",
        "risk": "high",
        # keyword nearby + 12-19 digit run, with optional space/dash separators between groups
        "regex": re.compile(
            r"(?:card\s*(?:number|no\.?|#)?\s*(?:is|:)?\s*)(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}(?:[\s\-]?\d{1,4})?)",
            re.IGNORECASE,
        ),
        "requires_keyword": ["card"],
        "recommended_action": "do_not_store",
    },
    {
        "type": "card_number",
        "risk": "high",
        # fallback: contiguous 12-19 digit run (no spaces)
        "regex": re.compile(
            r"(?:card\s*(?:number|no\.?|#)?\s*(?:is|:)?\s*)?\b(\d{12,19})\b",
            re.IGNORECASE,
        ),
        "requires_keyword": ["card"],
        "recommended_action": "do_not_store",
    },
    {
        "type": "bank_account_number",
        "risk": "high",
        "regex": re.compile(
            r"\b(?:account|a/?c)\s*(?:no\.?|number|#)?\s*(?:is|:)?\s*(\d{8,18})\b",
            re.IGNORECASE,
        ),
        "requires_keyword": None,  # keyword already baked into regex
        "recommended_action": "do_not_store",
    },
    {
        "type": "one_time_password",
        "risk": "high",
        "regex": re.compile(
            r"\b(?:otp|one[\s-]?time\s?password)\s*(?:is|:)?\s*(\d{4,8})\b",
            re.IGNORECASE,
        ),
        "requires_keyword": None,
        "recommended_action": "do_not_store",
    },
    {
        "type": "pin",
        "risk": "high",
        "regex": re.compile(
            r"\bpin\s*(?:code|number)?\s*(?:is|:)?\s*(\d{4,6})\b",
            re.IGNORECASE,
        ),
        "requires_keyword": None,
        "recommended_action": "do_not_store",
    },
    {
        "type": "password",
        "risk": "high",
        "regex": re.compile(
            r"\bpass(?:word|code)\s*(?:is|:)?\s*([^\s,.;]{3,})",
            re.IGNORECASE,
        ),
        "requires_keyword": None,
        "recommended_action": "do_not_store",
    },
    {
        "type": "auth_token",
        "risk": "high",
        "regex": re.compile(
            r"\btoken\s*(?:is|:)?\s*([A-Za-z0-9\-_.]{6,})",
            re.IGNORECASE,
        ),
        "requires_keyword": None,
        "recommended_action": "do_not_store",
    },
    {
        "type": "recovery_code",
        "risk": "high",
        "regex": re.compile(
            r"\b(?:recovery|access|backup)\s*code\s*(?:is|:)?\s*([A-Za-z0-9\-]{6,})",
            re.IGNORECASE,
        ),
        "requires_keyword": None,
        "recommended_action": "do_not_store",
    },
    {
        "type": "phone_number",
        "risk": "medium",
        "regex": re.compile(r"\b(\+?\d{1,3}[\s-]?)?\d{10}\b"),
        "requires_keyword": None,
        "recommended_action": "ask_for_confirmation",
    },
    {
        "type": "email_address",
        "risk": "medium",
        "regex": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "requires_keyword": None,
        "recommended_action": "ask_for_confirmation",
    },
    {
        "type": "private_address",
        "risk": "medium",
        "regex": re.compile(
            r"\b(?:i\s+live\s+(?:at|near)|my\s+(?:home\s+)?address\s+is)\s+([^.!?]{5,60})",
            re.IGNORECASE,
        ),
        "requires_keyword": None,
        "recommended_action": "ask_for_confirmation",
    },
    {
        # Third-person/delivery phrasing ("Deliver the device to 22 Green
        # Park Road, Chennai") -- the pattern above only catches first-person
        # disclosure ("I live at..."). Anchored on a leading street number so
        # "deliver this to John" (no address) doesn't false-positive.
        "type": "private_address",
        "risk": "medium",
        "regex": re.compile(
            r"\b(?:deliver|ship|send)\b.{0,40}?\bto\s+(\d+[^.!?]{4,55})",
            re.IGNORECASE,
        ),
        "requires_keyword": None,
        "recommended_action": "ask_for_confirmation",
    },
]


def _mask_span(text: str, start: int, end: int) -> str:
    """Replace text[start:end] with asterisks, keep the rest intact."""
    return text[:start] + ("*" * (end - start)) + text[end:]


def detect_sensitive(message_id: str, message: str):
    """
    Scan one message for sensitive info.
    Returns a list of finding dicts (usually 0 or 1, occasionally more
    if a message contains multiple distinct sensitive items).
    """
    findings = []
    masked_text = message
    claimed_spans = []  # digit/value spans already matched by a higher-priority pattern

    def overlaps(span, others):
        return any(not (span[1] <= o[0] or span[0] >= o[1]) for o in others)

    for pat in PATTERNS:
        match = pat["regex"].search(message)
        if not match:
            continue
        if pat["requires_keyword"] and not any(
            kw in message.lower() for kw in pat["requires_keyword"]
        ):
            continue

        span = match.span(1) if match.lastindex else match.span()

        # skip if this span overlaps something already claimed (e.g. a phone-number
        # regex matching a substring of an already-found card/account number)
        if overlaps(span, claimed_spans):
            continue
        claimed_spans.append(span)

        # mask just the captured sensitive value, not the whole message
        masked_text = _mask_span(masked_text, span[0], span[1])

        findings.append(
            {
                "message_id": message_id,
                "sensitivity_type": pat["type"],
                "risk": pat["risk"],
                "masked_text": None,  # filled in after all patterns applied
                "recommended_action": pat["recommended_action"],
            }
        )

    # apply final masked_text to every finding for this message
    for f in findings:
        f["masked_text"] = masked_text

    return findings


if __name__ == "__main__":
    # quick self-test with FABRICATED examples only (not real data)
    tests = [
        ("MSG_TEST_01", "Hey Ram, your OTP is 4821 for the login."),
        ("MSG_TEST_02", "my card number is 4111222233334444 please charge it"),
        ("MSG_TEST_03", "Let's catch up for coffee tomorrow"),
        ("MSG_TEST_04", "account number is 000123456789, transfer today"),
        ("MSG_TEST_05", "my pin is 4521 don't share it"),
    ]
    for mid, msg in tests:
        result = detect_sensitive(mid, msg)
        print(mid, "->", result if result else "no sensitive info found")
