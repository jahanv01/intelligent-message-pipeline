"""
Priority and Action Engine (L2 Part 1)
------------------------------------------
Assigns critical/high/medium/low to every actionable message and keeps
that assignment current as later messages change the deadline, status,
or urgency of the same underlying task/event.

Design:
  - Every extracted item (extract.py) is registered into a SubjectRegistry
    (linking.py) the first time its subject is seen -- that registration
    is the "original" item_id for that subject, for the rest of the run.
  - A status-changing message (completed/cancelled/rescheduled/deadline
    moved/conflicting-deadline -- linking.detect_status_update) that
    resolves to an already-registered subject updates THAT subject's
    state and produces a priority decision keyed by *this* message's
    own message_id but the *original* item_id -- matching the spec's own
    example (MSG_1042 / TASK_084). It does not spawn a second item, even
    if extract.py's plain keyword matcher also happened to fire on the
    same message (e.g. "please confirm whether you started to review the
    privacy checklist" contains "confirm", which independently matches
    extract.py's review_task keyword) -- the more specific template match
    wins for priority attribution.
  - A message with no resolvable update template and no item is not
    actionable and gets no decision.

Every signal that contributes to a score is named and returned, per the
assignment rule against single-keyword or random priority assignment.
"""
from __future__ import annotations

from datetime import datetime

from extract import PRIORITY_HIGH_SIGNALS
from linking import SubjectRegistry, detect_status_update, resolve_when

# ---------------------------------------------------------------------------
# Tunable weights -- documented verbatim in README "How priority is calculated"
# ---------------------------------------------------------------------------
SIGNAL_WEIGHTS = {
    "response_required": 1,
    "overdue": 3,
    "deadline_today": 3,
    "deadline_within_2_days": 2,
    "deadline_within_7_days": 1,
    "deadline_future": 0,
    "no_deadline": 0,
    "urgent_language": 2,
    "deadline_moved_earlier": 2,
    "deadline_moved_later": -1,
    "conflicting_deadline": 1,
    "sensitive_content": 1,
    "priority_sender": 0.5,
    "restated_subject": -0.5,
    "reopened": 0.5,
}

TASK_TYPE_BASE = {
    "deadline_task": 1, "submission_task": 1, "review_task": 1,
    "follow_up_task": 1, "new_task_announcement": 1,
}
EVENT_TYPE_BASE = {
    "scheduled_meeting": 0.5, "calendar_event": 0.5, "catchup_or_interview": 0.5,
    "event_announcement": 0.5, "scheduled_session_announcement": 0.5,
}
RESPONSE_REQUIRED_SUBCLASSES = {"review_task", "follow_up_task", "catchup_or_interview"}
PRIORITY_SENDER_ROLES = {"project lead", "hr team", "mentor", "operations"}

LEVEL_THRESHOLDS = [(6, "critical"), (4, "high"), (2, "medium")]  # else "low"
CONFIDENCE_BASE = 0.55

_SIGNAL_TEXT = {
    "overdue": "the deadline has already passed",
    "deadline_today": "the deadline is today",
    "deadline_within_2_days": "the deadline is within the next two days",
    "deadline_within_7_days": "the deadline is later this week",
    "deadline_future": "the deadline is more than a week away",
    "no_deadline": "no explicit deadline is stated",
    "urgent_language": "the message uses explicit urgent language",
    "deadline_moved_earlier": "a later message moved the deadline earlier",
    "deadline_moved_later": "a later message pushed the deadline out",
    "conflicting_deadline": "two messages state different deadlines for the same item",
    "sensitive_content": "the message also contains sensitive information",
    "priority_sender": "the sender holds a priority role",
    "response_required": "a response or confirmation is expected",
    "restated_subject": "this only restates an already-tracked request",
    "reopened": "a new deadline/schedule change contradicts a prior completed/cancelled status",
    "ambiguous_status": "the update uses uncertain language ('might'/'may'/'probably')",
}
_TERMINAL_TEXT = {
    "completed": "the item was marked completed, so it no longer needs action",
    "cancelled": "the item was marked cancelled, so it no longer needs action",
}


def _deadline_bucket(deadline_str, reference_dt: datetime):
    if not deadline_str:
        return "no_deadline"
    try:
        d = datetime.strptime(deadline_str, "%Y-%m-%d").date()
    except ValueError:
        return "no_deadline"
    delta = (d - reference_dt.date()).days
    if delta < 0:
        return "overdue"
    if delta == 0:
        return "deadline_today"
    if delta <= 2:
        return "deadline_within_2_days"
    if delta <= 7:
        return "deadline_within_7_days"
    return "deadline_future"


def _bucket_level(score: float) -> str:
    for threshold, level in LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "low"


def _build_reason(signals, sender):
    ranked = [s for s in signals if SIGNAL_WEIGHTS.get(s, 0) != 0]
    ranked.sort(key=lambda s: abs(SIGNAL_WEIGHTS.get(s, 0)), reverse=True)
    parts = [_SIGNAL_TEXT[s] for s in ranked[:2] if s in _SIGNAL_TEXT]
    if not parts:
        parts = ["no strong urgency signals were found"]
    sentence = " and ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def _score(state, sender, message_text, sensitive_hit, reference_dt, is_restatement):
    status = state.get("status", "pending")
    if status in _TERMINAL_TEXT:
        return "low", 0.9, [status], _TERMINAL_TEXT[status][0].upper() + _TERMINAL_TEXT[status][1:] + "."

    signals = []
    score = 0.0
    sub_class = state.get("sub_class")

    if state.get("reopened"):
        score += SIGNAL_WEIGHTS["reopened"]
        signals.append("reopened")

    if sub_class in RESPONSE_REQUIRED_SUBCLASSES or state.get("last_update_type") == "status_check":
        score += SIGNAL_WEIGHTS["response_required"]
        signals.append("response_required")

    score += TASK_TYPE_BASE.get(sub_class, EVENT_TYPE_BASE.get(sub_class, 0))

    bucket = _deadline_bucket(state.get("deadline"), reference_dt)
    score += SIGNAL_WEIGHTS[bucket]
    signals.append(bucket)

    urgent_now = state.get("urgent", False) or any(
        kw in message_text.lower() for kw in PRIORITY_HIGH_SIGNALS
    )
    if urgent_now:
        score += SIGNAL_WEIGHTS["urgent_language"]
        signals.append("urgent_language")

    direction = state.get("deadline_direction")
    if direction == "earlier":
        score += SIGNAL_WEIGHTS["deadline_moved_earlier"]
        signals.append("deadline_moved_earlier")
    elif direction == "later":
        score += SIGNAL_WEIGHTS["deadline_moved_later"]
        signals.append("deadline_moved_later")

    if state.get("last_conflicting"):
        score += SIGNAL_WEIGHTS["conflicting_deadline"]
        signals.append("conflicting_deadline")

    if sensitive_hit:
        score += SIGNAL_WEIGHTS["sensitive_content"]
        signals.append("sensitive_content")

    if sender and sender.strip().lower() in PRIORITY_SENDER_ROLES:
        score += SIGNAL_WEIGHTS["priority_sender"]
        signals.append("priority_sender")

    if is_restatement:
        score += SIGNAL_WEIGHTS["restated_subject"]
        signals.append("restated_subject")

    ambiguous = state.get("last_update_type") == "ambiguous"
    if ambiguous:
        signals.append("ambiguous_status")

    level = _bucket_level(score)

    confidence = CONFIDENCE_BASE
    if state.get("deadline"):
        confidence += 0.15
    if state.get("last_update_type") in {"completed", "cancelled", "deadline_changed", "rescheduled"}:
        confidence += 0.10
    if sender and sender.strip().lower() in PRIORITY_SENDER_ROLES:
        confidence += 0.10
    if ambiguous:
        confidence -= 0.25
    if state.get("last_conflicting"):
        confidence -= 0.15
    confidence = max(0.3, min(round(confidence, 2), 0.97))

    return level, confidence, signals, _build_reason(signals, sender)


def compute_priorities(messages, classifications, extracted_items, sensitive_findings, reference_dt=None):
    """
    messages: chronological list of {message_id, timestamp(datetime), sender, message}
    classifications: list of dicts from classify_message (unused directly today,
        reserved so category can be folded in without changing the call signature)
    extracted_items: list of dicts from extract_item
    sensitive_findings: list of dicts from detect_sensitive
    reference_dt: datetime treated as "now" for overdue/proximity. Defaults to
        the last message's own timestamp -- this is a historical/fictional
        dataset, so real wall-clock time would misjudge every deadline.
    """
    if reference_dt is None:
        reference_dt = max(m["timestamp"] for m in messages) if messages else datetime.now()

    items_by_msg = {i["source_message_id"]: i for i in extracted_items}
    sensitive_by_msg = {}
    for f in sensitive_findings:
        sensitive_by_msg.setdefault(f["message_id"], []).append(f)

    registry = SubjectRegistry()
    decisions = []

    def _base_state(it):
        return {"sub_class": it["sub_class"], "type": it["type"],
                "deadline": it["deadline"], "time": it["time"], "status": "pending"}

    def _mint_ref_id(message_id):
        # A status-changing/status-check message can be the FIRST mention of a
        # subject that extract.py never turned into a formal item (its
        # phrasing doesn't hit any extraction keyword). Rather than silently
        # dropping every later update about that subject, mint a lightweight
        # reference id from this message so the chain still links up -- kept
        # visually distinct (REF_ prefix) from real TASK_/EVENT_ ids so it's
        # obvious in the output which subjects have no formal item behind them.
        return f"REF_{message_id}"

    for m in messages:
        mid = m["message_id"]
        item = items_by_msg.get(mid)
        update = detect_status_update(m["message"])

        resolved_item_id = None
        is_restatement = False

        found_item_id = registry.find(update["subject_norm"]) if update and update["subject_norm"] else None

        if update:
            if found_item_id:
                registry.apply_update(found_item_id, update, m["timestamp"])
                resolved_item_id = found_item_id
                is_restatement = item is not None  # item also exists but we deliberately don't use it
            elif item:
                registry.register(item["item_id"], item["description"], _base_state(item))
                registry.apply_update(item["item_id"], update, m["timestamp"])
                resolved_item_id = item["item_id"]
            elif update["subject_norm"]:
                ref_id = _mint_ref_id(mid)
                registry.register(ref_id, m["message"], {
                    "sub_class": None, "type": "reference", "deadline": None, "time": None, "status": "pending",
                })
                registry.apply_update(ref_id, update, m["timestamp"])
                resolved_item_id = ref_id
            # else: no subject could be resolved at all -- deliberately not
            # linked or invented (see README "Assumptions and limitations").
        elif item:
            registry.register(item["item_id"], item["description"], _base_state(item))
            resolved_item_id = item["item_id"]

        if resolved_item_id is None:
            continue

        state = registry.state_for(resolved_item_id)
        sensitive_hit = bool(sensitive_by_msg.get(mid))
        level, confidence, signals, reason = _score(
            state, m.get("sender"), m["message"], sensitive_hit, reference_dt, is_restatement,
        )
        decisions.append({
            "message_id": mid,
            "item_id": resolved_item_id,
            "priority": level,
            "reason": reason,
            "signals": signals,
            "confidence": confidence,
        })

    return decisions
