import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract import extract_item
from sensitive import detect_sensitive
from priority import compute_priorities


def _build(rows):
    """rows: list of (message_id, timestamp, sender, message) -> (messages, items, sens)."""
    messages, items, sens = [], [], []
    for mid, ts, sender, text in rows:
        messages.append({"message_id": mid, "timestamp": ts, "sender": sender, "message": text})
        item = extract_item(mid, text, ts, sender)
        if item:
            items.append(item)
        sens.extend(detect_sensitive(mid, text))
    return messages, items, sens


def _by_id(decisions, message_id):
    return next(d for d in decisions if d["message_id"] == message_id)


def test_new_task_with_near_deadline_is_medium():
    rows = [("MSG_1", datetime(2026, 9, 5), "Ram", "Please submit the report by 2026-09-10")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_1")
    assert d["priority"] == "medium"
    assert "deadline_within_7_days" in d["signals"]


def test_deadline_today_plus_urgent_is_critical():
    rows = [("MSG_1", datetime(2026, 9, 10), "Ram",
              "Don't forget, the deadline is today, this is urgent.")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_1")
    assert d["priority"] == "critical"
    assert "deadline_today" in d["signals"]
    assert "urgent_language" in d["signals"]


def test_overdue_task_is_bumped_to_high():
    rows = [("MSG_1", datetime(2026, 9, 15), "Ram",
              "Don't forget, the report is due on 2026-09-10.")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_1")
    assert d["priority"] == "high"
    assert "overdue" in d["signals"]


def test_completed_update_forces_low_with_high_confidence():
    rows = [
        ("MSG_1", datetime(2026, 9, 5), "Ram", "Please submit the report by 2026-09-10"),
        ("MSG_2", datetime(2026, 9, 12), "Ram", "Update: submit the report has been completed successfully."),
    ]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_2")
    assert d["priority"] == "low"
    assert d["confidence"] >= 0.85
    assert d["item_id"] == items[0]["item_id"]  # attached to the ORIGINAL item, not a new one
    assert "completed" in d["signals"]


def test_cancelled_update_forces_low():
    rows = [
        ("MSG_1", datetime(2026, 9, 5), "Ram", "Please submit the report by 2026-09-10"),
        ("MSG_2", datetime(2026, 9, 12), "Ram", "You can cancel submit the report; it is no longer required."),
    ]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_2")
    assert d["priority"] == "low"
    assert "cancelled" in d["signals"]


def test_deadline_moved_earlier_plus_urgent_is_critical_and_keeps_original_item_id():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please review the test cases by 2026-09-20"),
        ("MSG_2", datetime(2026, 9, 12), "Ram",
         "The deadline to review the test cases is now 2026-09-12, earlier than "
         "previously planned. Treat this as urgent."),
    ]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_2")
    assert d["priority"] == "critical"
    assert d["item_id"] == items[0]["item_id"]
    assert "deadline_moved_earlier" in d["signals"]
    assert "urgent_language" in d["signals"]


def test_conflicting_deadline_lowers_confidence():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please review the test cases by 2026-09-20"),
        ("MSG_2", datetime(2026, 9, 5), "Ram",
         "Please note that review the test cases is due on 2026-09-25, although "
         "the earlier message listed another date."),
    ]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_2")
    assert "conflicting_deadline" in d["signals"]
    assert d["confidence"] <= 0.65


def test_ambiguous_status_lowers_confidence_without_inventing_certainty():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please prepare the quarterly summary report by 2026-09-15"),
        ("MSG_2", datetime(2026, 9, 10), "Ram",
         "The quarterly summary report might already be finished, but I cannot confirm it."),
    ]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_2")
    assert "ambiguous_status" in d["signals"]
    assert d["confidence"] < 0.5


def test_fully_ambiguous_message_with_no_known_subject_is_not_linked():
    """No fabricated link: a generic ambiguous statement with no task-taxonomy
    keyword and no resolvable subject must not produce a priority decision."""
    rows = [("MSG_1", datetime(2026, 9, 1), "Kabir", "Maya said someone probably handled the task.")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    assert decisions == []


def test_sensitive_content_adds_signal_and_score():
    rows = [("MSG_1", datetime(2026, 9, 5), "Ram",
              "Please review the access token before 2026-09-10; token is tok_abc123456")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_1")
    assert "sensitive_content" in d["signals"]


def test_earlier_decision_in_a_chain_is_not_overwritten_by_the_items_final_state():
    """Regression: a message's priority decision must reflect state AS OF
    THAT MESSAGE, not the item's FINAL state after every later message has
    also been processed. Found via manual output audit -- three different
    real messages on the same item were producing byte-identical decisions
    because resolve_messages() (used by both priority.py and grouping.py)
    finishes its whole chronological pass before priority.py ever scores
    anything, and registry.state_for() returns the SAME mutable dict for
    every message on that item -- so scoring after the fact silently
    flattens the entire history to one value."""
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please review the report by 2026-09-20"),
        ("MSG_2", datetime(2026, 9, 5), "Ram", "Please check the latest status of review the report."),
        ("MSG_3", datetime(2026, 9, 10), "Ram",
         "The deadline to review the report is now 2026-09-10, earlier than "
         "previously planned. Treat this as urgent."),
    ]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d1 = _by_id(decisions, "MSG_1")
    d2 = _by_id(decisions, "MSG_2")
    d3 = _by_id(decisions, "MSG_3")
    # MSG_2 is a plain status-check BEFORE the deadline was ever moved --
    # it must NOT show urgent_language or deadline_moved_earlier, even
    # though MSG_3 (which comes after it) introduces both.
    assert "urgent_language" not in d2["signals"]
    assert "deadline_moved_earlier" not in d2["signals"]
    assert d2["reason"] != d3["reason"]
    assert "urgent_language" in d3["signals"]
    assert "deadline_moved_earlier" in d3["signals"]
    assert d3["priority"] == "critical"


def test_priority_sender_role_adds_signal():
    rows = [("MSG_1", datetime(2026, 9, 5), "Project Lead", "New task: measure memory usage by 2026-09-20")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_1")
    assert "priority_sender" in d["signals"]


def test_message_category_action_required_reinforces_signal():
    rows = [("MSG_1", datetime(2026, 9, 5), "Ram", "Please submit the report by 2026-09-10")]
    messages, items, sens = _build(rows)
    cls = [{"message_id": "MSG_1", "category": "action_required", "confidence": 0.9, "reason": "test"}]
    decisions = compute_priorities(messages, cls, items, sens)
    d = _by_id(decisions, "MSG_1")
    assert "category_action_required" in d["signals"]


def test_semantic_urgency_catches_paraphrase_without_keyword():
    """'expedite'/'cannot wait' aren't in PRIORITY_HIGH_SIGNALS -- only the
    embedding fallback can catch this one."""
    rows = [("MSG_1", datetime(2026, 9, 5), "Ram",
              "Please submit the vendor invoice, it cannot wait any longer.")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_1")
    assert "semantic_urgency" in d["signals"]
    assert "urgent_language" not in d["signals"]


def test_semantic_urgency_does_not_fire_on_routine_message():
    rows = [("MSG_1", datetime(2026, 9, 5), "Ram",
              "Please review the document sometime this week, no rush at all.")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_1")
    assert "semantic_urgency" not in d["signals"]
    assert "urgent_language" not in d["signals"]


def test_explicit_keyword_urgency_skips_semantic_check():
    """When the keyword hits, the semantic fallback shouldn't also fire and
    double-count the same concept under two signal names."""
    rows = [("MSG_1", datetime(2026, 9, 5), "Ram", "Please review the report, this is urgent.")]
    messages, items, sens = _build(rows)
    decisions = compute_priorities(messages, [], items, sens)
    d = _by_id(decisions, "MSG_1")
    assert "urgent_language" in d["signals"]
    assert "semantic_urgency" not in d["signals"]


def test_message_category_mismatch_lowers_confidence():
    """If classify.py independently thinks a message isn't actionable
    (promotional/general/personal) but something still tracked it as a
    task/event, that disagreement must show up as reduced confidence --
    never silently resolved in either direction."""
    rows = [("MSG_1", datetime(2026, 9, 5), "Ram", "Please submit the report by 2026-09-10")]
    messages, items, sens = _build(rows)
    matching = compute_priorities(
        messages, [{"message_id": "MSG_1", "category": "action_required", "confidence": 0.9, "reason": "t"}],
        items, sens,
    )
    mismatched = compute_priorities(
        messages, [{"message_id": "MSG_1", "category": "promotional", "confidence": 0.9, "reason": "t"}],
        items, sens,
    )
    d_match = _by_id(matching, "MSG_1")
    d_mismatch = _by_id(mismatched, "MSG_1")
    assert "category_mismatch" in d_mismatch["signals"]
    assert d_mismatch["confidence"] < d_match["confidence"]
