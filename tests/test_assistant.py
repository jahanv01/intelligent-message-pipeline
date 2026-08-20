import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from classify import classify_message
from extract import extract_item
from sensitive import detect_sensitive
from priority import compute_priorities
from grouping import compute_groups, annotate_superseded
from privacy import route_all
from assistant import build_knowledge_base, answer_query


def _pipeline(rows):
    """rows: list of (message_id, timestamp, sender, message) -> full KB, exactly like main.py builds it."""
    messages, classifications, items, sens = [], [], [], []
    for mid, ts, sender, text in rows:
        messages.append({"message_id": mid, "timestamp": ts, "sender": sender, "message": text})
        cls, hits = classify_message(mid, text)
        classifications.append(cls)
        sens.extend(hits)
        item = extract_item(mid, text, ts, sender)
        if item:
            items.append(item)
    priorities = compute_priorities(messages, classifications, items, sens)
    groups = compute_groups(messages, items)
    annotate_superseded(items, groups)
    routing = route_all(messages, sens)
    kb = build_knowledge_base(messages, classifications, items, sens, priorities, groups, routing)
    return kb


REPORT_CHAIN = [
    ("MSG_1", datetime(2026, 9, 1), "Ram", "Please submit the quarterly report by 2026-09-10"),
    ("MSG_2", datetime(2026, 9, 3), "Priya", "Following up on submit the quarterly report; is it in progress?"),
    ("MSG_3", datetime(2026, 9, 9), "Priya", "Update: submit the quarterly report has been completed successfully."),
]


def test_subject_search_finds_group_by_explicit_id():
    kb = _pipeline(REPORT_CHAIN)
    ans = answer_query(kb, "Show all messages related to MSG_1")
    assert set(ans["supporting_message_ids"]) == {"MSG_1", "MSG_2", "MSG_3"}
    assert ans["reason"].startswith("Matched via explicit id")
    assert ans["relevance_scores"] == [1.0]


def test_subject_search_finds_group_by_free_text():
    kb = _pipeline(REPORT_CHAIN)
    ans = answer_query(kb, "What's the latest on the quarterly report?")
    assert "MSG_3" in ans["supporting_message_ids"]
    assert ans["relevance_scores"][0] > 0


def test_subject_search_insufficient_evidence_for_unrelated_topic():
    kb = _pipeline(REPORT_CHAIN)
    ans = answer_query(kb, "Was the compliance form approved by the finance director?")
    assert ans["supporting_message_ids"] == []
    assert ans["confidence"] == 0.0
    assert "insufficient" in ans["reason"].lower() or "no explicit id" in ans["reason"].lower()


def test_latest_status_reports_completed():
    kb = _pipeline(REPORT_CHAIN)
    ans = answer_query(kb, "What is the latest status of MSG_1?")
    assert "Completed" in ans["answer"]
    assert ans["group_id"] is not None


def test_completed_or_cancelled_intent():
    kb = _pipeline(REPORT_CHAIN)
    ans = answer_query(kb, "Which tasks have been completed?")
    assert "MSG_3" in ans["supporting_message_ids"]


def test_blocked_messages_intent():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Your OTP is 4821 for the login."),
        ("MSG_2", datetime(2026, 9, 2), "Ram", "Let's catch up for coffee tomorrow."),
    ]
    kb = _pipeline(rows)
    ans = answer_query(kb, "Which messages must be blocked from external processing?")
    assert ans["supporting_message_ids"] == ["MSG_1"]
    assert ans["confidence"] > 0.9


def test_requires_confirmation_intent():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "My home address is 42 Lake View Road, Chennai."),
        ("MSG_2", datetime(2026, 9, 2), "Ram", "Let's catch up for coffee tomorrow."),
    ]
    kb = _pipeline(rows)
    ans = answer_query(kb, "Which message requires confirmation before processing?")
    assert ans["supporting_message_ids"] == ["MSG_1"]


def test_why_critical_returns_the_actual_priority_reason():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please review the audit report by 2026-09-20"),
        ("MSG_2", datetime(2026, 9, 19), "Ram",
         "The deadline to review the audit report is now 2026-09-19, earlier than "
         "previously planned. Treat this as urgent."),
    ]
    kb = _pipeline(rows)
    ans = answer_query(kb, "Why was MSG_2 marked as critical?")
    assert ans["answer"] != ""
    assert "urgent" in ans["answer"].lower() or "deadline" in ans["answer"].lower()
    assert ans["supporting_message_ids"] == ["MSG_2"]


def test_rescheduled_intent():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Mentor", "A new architecture-review session is scheduled for 2026-09-10 at 09:00."),
        ("MSG_2", datetime(2026, 9, 3), "Mentor", "The architecture-review has been moved to 2026-09-15 at 10:00. Please use the new schedule."),
    ]
    kb = _pipeline(rows)
    ans = answer_query(kb, "What meetings were rescheduled?")
    assert "MSG_1" in ans["supporting_message_ids"] and "MSG_2" in ans["supporting_message_ids"]
    assert "2026-09-15" in ans["answer"]


def test_conflicting_deadlines_intent():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please review the audit report by 2026-09-20"),
        ("MSG_2", datetime(2026, 9, 5), "Ram",
         "Please note that review the audit report is due on 2026-09-25, although "
         "the earlier message listed another date."),
    ]
    kb = _pipeline(rows)
    ans = answer_query(kb, "Are there any conflicting messages about the same event?")
    assert "MSG_2" in ans["supporting_message_ids"]


def test_no_answer_leaks_raw_sensitive_text():
    """A message with a sensitive finding must never have its raw text
    surfaced anywhere in an answer, even indirectly via masked_or_raw()."""
    rows = [("MSG_1", datetime(2026, 9, 1), "Ram", "Your OTP is 552134 for the vendor portal.")]
    kb = _pipeline(rows)
    safe_text = kb.masked_or_raw("MSG_1")
    assert "552134" not in safe_text
