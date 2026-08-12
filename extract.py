"""
Task / Event Extraction
--------------------------
Rule: never guess. If a date, time, or person can't be confidently
resolved, the field is stored as null — not a best guess.

Design: sub-class taxonomy (mirrors Part 1). Each sub-class knows its
type (event/task), default priority, and trigger keywords. The first
matching sub-class (events checked before tasks) wins.

Date/time resolution uses the message's own timestamp for relative
phrases ("tomorrow", "next week") — computation from known data, not a
guess. Bare hours like "at 9" with no AM/PM stay null.
"""
import re
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Sub-class taxonomy
# ---------------------------------------------------------------------------
EXTRACTION_SUBCLASSES = [
    # -- events ---------------------------------------------------------------
    {
        "sub": "scheduled_meeting",
        "type": "event",
        "priority": "medium",
        # use whole-word variants to avoid "meeting" matching inside "meetings now"
        "keywords": [
            "the meeting", "a meeting", "team meeting", "board meeting",
            "sprint meeting", "standup", "stand-up", "conference call",
            "webinar", "seminar", "workshop", "the session", "zoom call",
        ],
    },
    {
        "sub": "calendar_event",
        "type": "event",
        "priority": "medium",
        "keywords": [
            "calendar update", "calendar", "appointment", "dinner",
            "lunch", "brunch", "orientation", "hackathon", "party",
        ],
    },
    {
        "sub": "catchup_or_interview",
        "type": "event",
        "priority": "medium",
        "keywords": [
            "catch-up", "catch up", "let's meet", "can we meet",
            "are you available", "interview", "sync call",
        ],
    },
    {
        "sub": "event_announcement",
        "type": "event",
        "priority": "medium",
        "keywords": [
            "join us", "join the", "join me", "please join",
            "conference", "event starts", "kick-off", "kickoff",
        ],
    },
    # -- tasks ----------------------------------------------------------------
    {
        "sub": "submission_task",
        "type": "task",
        "priority": "medium",
        "keywords": [
            "submit", "upload", "send", "fill", "sign", "pay",
        ],
    },
    {
        "sub": "review_task",
        "type": "task",
        "priority": "medium",
        "keywords": [
            "review", "approve", "check", "confirm", "verify",
        ],
    },
    {
        "sub": "deadline_task",
        "type": "task",
        "priority": "high",
        "keywords": [
            "deadline", "due", "don't forget", "asap", "urgent", "critical",
        ],
    },
    {
        "sub": "follow_up_task",
        "type": "task",
        "priority": "medium",
        "keywords": [
            "reply", "respond", "revert", "kindly", "need you to",
            "please submit", "please review", "please send", "please reply",
            "please confirm", "please update", "please complete",
            "complete", "prepare", "finish",
        ],
    },
]

# Events are checked before tasks so a message with both (e.g. "Please join
# the meeting") is typed as an event, not a task.
_EVENT_SUBCLASSES = [s for s in EXTRACTION_SUBCLASSES if s["type"] == "event"]
_TASK_SUBCLASSES  = [s for s in EXTRACTION_SUBCLASSES if s["type"] == "task"]

PRIORITY_HIGH_SIGNALS = ["asap", "urgent", "immediately", "important", "critical"]

# Strip common filler prefixes before generating the title.
_FILLER_PREFIX = re.compile(
    r"^(for today[:\s]*|fyi[:\s]*|just (checking|saying|a heads-up)[^\w]*"
    r"|one more thing[:\s]*|please note[:\s]*|important[:\s]*"
    r"|can you help[^\w]*|hi[,\s]+|hey[,\s]+|note[:\s]*|reminder[:\s]*)+",
    re.IGNORECASE,
)

# matches "9 AM", "9AM", "9:30 am" etc — with explicit am/pm
TIME_WITH_AMPM = re.compile(r"\b(\d{1,2}(?::\d{2})?)\s*([APap][Mm])\b")
# matches 24-hour format: 13:00, 09:30 etc — unambiguous, no guessing needed
TIME_24H = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")

CAPITALIZED_NAME = re.compile(r"\b([A-Z][a-z]{2,})\b")
# words that look capitalised but are never person names in this dataset
COMMON_SENTENCE_STARTERS = {
    "The", "Please", "Hi", "Hey", "Dear", "This", "Your", "You", "We",
    "Our", "It", "Kindly", "Let", "For", "Just", "Note", "All", "One",
    "Don", "Can", "Will", "Do", "Use", "Buy", "Join", "See", "Get",
    "Conference", "Meeting", "Room", "Hall", "Floor", "Gate", "Park",
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "January", "February", "March", "April", "June", "July",
    "August", "September", "October", "November", "December",
    # dataset-specific non-name capitalized words
    "Calendar", "Reminder", "Update", "Important", "Alert", "Notice",
    "Training", "Portal", "Office", "Server", "Network", "Building",
    "Shuttle", "Cafeteria", "Parking", "Webinar", "Flash", "Special",
    "Premium", "Subscribe", "Upgrade", "Exclusive",
    "City", "Library", "Clinic", "Auditorium", "Campus",
    "Task", "Event", "Project", "Report", "Document", "File",
    "Remember", "Check", "Review", "Submit", "Complete", "Send",
    "Client", "Team", "Sprint", "Quarter", "Deadline", "Sale",
}


def _resolve_date(message: str, msg_date: datetime):
    # Explicit dates are more reliable than relative phrases, so check them
    # FIRST. A message can contain both ("...tomorrow... deadline is
    # 2026-08-20") and the explicit one is what should be trusted.
    iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", message)
    if iso:
        return iso.group(1)
    dmy = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", message)
    if dmy:
        d, m, y = dmy.groups()
        try:
            return datetime(int(y), int(m), int(d)).strftime("%Y-%m-%d")
        except ValueError:
            return None

    text = message.lower()
    if "tomorrow" in text:
        return (msg_date + timedelta(days=1)).strftime("%Y-%m-%d")
    if "today" in text:
        return msg_date.strftime("%Y-%m-%d")
    if "next week" in text:
        return (msg_date + timedelta(days=7)).strftime("%Y-%m-%d")
    # named weekday: resolve to next occurrence from msg_date
    _DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(_DAYS):
        if day in text:
            diff = (i - msg_date.weekday()) % 7 or 7
            return (msg_date + timedelta(days=diff)).strftime("%Y-%m-%d")
    return None


def _resolve_time(message: str):
    m = TIME_WITH_AMPM.search(message)
    if m:
        return f"{m.group(1)} {m.group(2).upper()}"
    m = TIME_24H.search(message)
    if m:
        return m.group(0)  # 24h is unambiguous — no guessing
    return None  # bare "at 9" with no AM/PM and no colon -> deliberately unresolved


def _resolve_person(message: str, sender: str):
    # Exclude position-0 words (sentence-start capitalisation, not names).
    # Also skip words immediately followed by apostrophe ("Don't", "It's").
    for m in CAPITALIZED_NAME.finditer(message):
        word = m.group(1)
        if m.start() == 0:
            continue
        end = m.end()
        if end < len(message) and message[end] == "'":
            continue  # contraction — not a name
        if word in COMMON_SENTENCE_STARTERS or word == sender:
            continue
        return word
    return None  # conservative: don't default to sender if unclear


def _clean_title(message: str) -> str:
    title = _FILLER_PREFIX.sub("", message).strip()
    if len(title) <= 80:
        return title
    # break at last word boundary before 80 chars
    cut = title[:80].rsplit(" ", 1)[0]
    return cut + "…"


def _match_subclass(text: str):
    """Return matched sub-class dict or None. Events take priority over tasks.
    Uses word-boundary matching to avoid substring false positives (e.g.
    'check' must not fire on 'checking', 'sign' must not fire on 'signing').
    """
    def _hits(keywords):
        for kw in keywords:
            # multi-word phrases: plain substring is fine (boundaries implied)
            if " " in kw:
                if kw in text:
                    return True
            else:
                if re.search(r"\b" + re.escape(kw) + r"\b", text):
                    return True
        return False

    for sc in _EVENT_SUBCLASSES:
        if _hits(sc["keywords"]):
            return sc
    for sc in _TASK_SUBCLASSES:
        if _hits(sc["keywords"]):
            return sc
    return None


def extract_item(message_id: str, message: str, msg_date: datetime, sender: str):
    text = message.lower()
    matched = _match_subclass(text)
    if matched is None:
        return None

    # High-urgency signals can override the sub-class default priority.
    priority = "high" if any(p in text for p in PRIORITY_HIGH_SIGNALS) else matched["priority"]

    prefix = "TASK" if matched["type"] == "task" else "EVENT"
    num = message_id.split("_")[-1].lstrip("0") or "0"
    return {
        "item_id": f"{prefix}_{num}",
        "type": matched["type"],
        "sub_class": matched["sub"],
        "title": _clean_title(message),
        "description": message,
        "deadline": _resolve_date(message, msg_date),
        "time": _resolve_time(message),
        "person": _resolve_person(message, sender),
        "priority": priority,
        "source_message_id": message_id,
    }


if __name__ == "__main__":
    base = datetime(2026, 8, 10)
    tests = [
        ("MSG_T1", "Please submit the report tomorrow, this is urgent", "Ram"),
        ("MSG_T2", "Let's meet at 9 AM for the sprint call", "Priya"),
        ("MSG_T3", "Call with Arjun at 12, don't forget", "Ram"),
        ("MSG_T4", "Just saying hi, how are you", "Ram"),
    ]
    for mid, msg, sender in tests:
        print(extract_item(mid, msg, base, sender))
