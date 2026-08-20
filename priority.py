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

import re

import numpy as np

from extract import PRIORITY_HIGH_SIGNALS
from classify import embed_texts
from linking import SubjectRegistry, detect_status_update, resolve_when

# ---------------------------------------------------------------------------
# Semantic urgency -- catches paraphrases the fixed PRIORITY_HIGH_SIGNALS
# keyword list misses ("please expedite this", "time-sensitive, need this
# today"). Reuses classify.py's already-loaded fastembed model (see
# classify.embed_texts) rather than loading a second copy. Only consulted
# when the keyword check found nothing -- it's a fallback that catches more
# phrasings, not a second vote that double-counts the same concept.
# ---------------------------------------------------------------------------
_URGENT_ANCHORS = [
    "This is extremely urgent and needs immediate attention.",
    "Please treat this as a top priority right now.",
    "We need this resolved without any delay.",
    "This cannot wait -- please act on it today.",
    "Dropping everything else, this takes precedence over all other work.",
    "This is time-critical and must be actioned before end of day.",
    "Escalating this -- it needs attention right now, not later.",
    "High priority: this must be handled before anything else.",
]
_ROUTINE_ANCHORS = [
    "This is a routine update, no rush at all.",
    "Whenever you get a chance, no immediate action needed.",
    "This can wait until next week, no hurry.",
    "Just a general FYI, nothing time-sensitive here.",
    "Take your time with this, there is no deadline pressure.",
    "This is low priority and can be scheduled for later.",
    "No particular urgency here, just sharing for awareness.",
    "Feel free to get to this whenever convenient.",
]
# Empirically chosen (see tests/test_priority.py) -- margin: urgent-anchor
# similarity minus routine-anchor similarity. Real margins on held-out
# paraphrases top out around 0.19; 0.55 (an initial guess) never fired at
# all. 0.12 sits above every observed routine false-positive margin (<=0.065)
# while still catching most held-out urgent paraphrases -- conservative on
# purpose, since a noisy signal here would itself be the "random priority"
# behavior the assignment warns against.
SEMANTIC_URGENCY_THRESHOLD = 0.12

_urgent_matrix = None   # (n_anchors, dim), normalised
_routine_matrix = None
_semantic_urgency_available = None  # None = not yet attempted, True/False after first try


def _ensure_urgency_anchors():
    global _urgent_matrix, _routine_matrix, _semantic_urgency_available
    if _semantic_urgency_available is not None:
        return
    urgent_vecs = embed_texts(_URGENT_ANCHORS)
    if urgent_vecs is None:
        _semantic_urgency_available = False
        return
    routine_vecs = embed_texts(_ROUTINE_ANCHORS)

    def _normalise(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, 1e-10, None)

    _urgent_matrix = _normalise(urgent_vecs)
    _routine_matrix = _normalise(routine_vecs)
    _semantic_urgency_available = True


# Hedge/uncertainty language ("could be", "might", "probably"...) inverts
# urgency regardless of topical similarity -- MiniLM's embedding does not
# reliably capture that distinction (verified empirically: "The review could
# be Friday afternoon" scored as urgent by margin alone). A deterministic
# pre-filter catches what the embedding can't, the same reasoning sensitive.py
# already applies to fixed-format secrets: use the right tool for the job.
_HEDGE_RE = re.compile(
    r"\b(could be|might|may|possibly|perhaps|probably|maybe|not sure|uncertain)\b", re.I,
)


def _semantic_urgency_hit(message: str) -> bool:
    """True if `message` reads as semantically urgent -- similarity to the
    urgent anchors clears the routine anchors by SEMANTIC_URGENCY_THRESHOLD.
    False (never guessed True) if the embedding model isn't installed, or if
    the message hedges (uncertainty language always overrides a topical
    similarity match)."""
    if _HEDGE_RE.search(message):
        return False
    _ensure_urgency_anchors()
    if not _semantic_urgency_available:
        return False
    vec = embed_texts([message])[0]
    vec = vec / max(np.linalg.norm(vec), 1e-10)
    urgent_sim = float(np.max(_urgent_matrix.dot(vec)))
    routine_sim = float(np.max(_routine_matrix.dot(vec)))
    return (urgent_sim - routine_sim) >= SEMANTIC_URGENCY_THRESHOLD

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
    "semantic_urgency": 1.5,
    "deadline_moved_earlier": 2,
    "deadline_moved_later": -1,
    "conflicting_deadline": 1,
    "sensitive_content": 1,
    "priority_sender": 0.5,
    "restated_subject": -0.5,
    "reopened": 0.5,
    "category_action_required": 0.5,
}

# Categories classify.py would never consider actionable. If a decision
# still gets produced for a message in one of these (e.g. extract.py's
# keyword match fired on incidental wording in a promotional/general
# message), that's the two independent signals disagreeing -- flagged via
# confidence, never used to silently overrule either one.
NON_ACTIONABLE_CATEGORIES = {"promotional", "general_information", "personal_information"}

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
    "semantic_urgency": "the message reads as urgent in meaning, even without an urgent keyword",
    "deadline_moved_earlier": "a later message moved the deadline earlier",
    "deadline_moved_later": "a later message pushed the deadline out",
    "conflicting_deadline": "two messages state different deadlines for the same item",
    "sensitive_content": "the message also contains sensitive information",
    "priority_sender": "the sender holds a priority role",
    "response_required": "a response or confirmation is expected",
    "restated_subject": "this only restates an already-tracked request",
    "reopened": "a new deadline/schedule change contradicts a prior completed/cancelled status",
    "category_action_required": "the message was independently classified as action-required",
    "category_mismatch": "the message's own classification doesn't look actionable, despite a task/event being tracked here",
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


def _score(state, sender, message_text, sensitive_hit, reference_dt, is_restatement, category):
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
    elif _semantic_urgency_hit(message_text):
        score += SIGNAL_WEIGHTS["semantic_urgency"]
        signals.append("semantic_urgency")

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

    category_mismatch = category in NON_ACTIONABLE_CATEGORIES
    if category == "action_required":
        score += SIGNAL_WEIGHTS["category_action_required"]
        signals.append("category_action_required")
    elif category_mismatch:
        signals.append("category_mismatch")

    level = _bucket_level(score)

    confidence = CONFIDENCE_BASE
    if state.get("deadline"):
        confidence += 0.15
    if state.get("last_update_type") in {"completed", "cancelled", "deadline_changed", "rescheduled"}:
        confidence += 0.10
    if sender and sender.strip().lower() in PRIORITY_SENDER_ROLES:
        confidence += 0.10
    if category == "action_required":
        confidence += 0.05
    if "semantic_urgency" in signals:
        confidence += 0.05  # inferred, not literal -- smaller bump than an explicit keyword match
    if ambiguous:
        confidence -= 0.25
    if state.get("last_conflicting"):
        confidence -= 0.15
    if category_mismatch:
        confidence -= 0.20
    confidence = max(0.3, min(round(confidence, 2), 0.97))

    return level, confidence, signals, _build_reason(signals, sender)


def compute_priorities(messages, classifications, extracted_items, sensitive_findings, reference_dt=None):
    """
    messages: chronological list of {message_id, timestamp(datetime), sender, message}
    classifications: list of dicts from classify_message -- used as an
        independent cross-check signal (category_action_required /
        category_mismatch), separate from extract.py's own sub_class
    extracted_items: list of dicts from extract_item
    sensitive_findings: list of dicts from detect_sensitive
    reference_dt: datetime treated as "now" for overdue/proximity. Defaults to
        the last message's own timestamp -- this is a historical/fictional
        dataset, so real wall-clock time would misjudge every deadline.
    """
    if reference_dt is None:
        reference_dt = max(m["timestamp"] for m in messages) if messages else datetime.now()

    items_by_msg = {i["source_message_id"]: i for i in extracted_items}
    category_by_msg = {c["message_id"]: c["category"] for c in classifications}
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
        category = category_by_msg.get(mid)
        level, confidence, signals, reason = _score(
            state, m.get("sender"), m["message"], sensitive_hit, reference_dt, is_restatement, category,
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
