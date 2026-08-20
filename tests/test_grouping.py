import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract import extract_item
from grouping import compute_groups


def _build(rows):
    messages, items = [], []
    for mid, ts, sender, text in rows:
        messages.append({"message_id": mid, "timestamp": ts, "sender": sender, "message": text})
        item = extract_item(mid, text, ts, sender)
        if item:
            items.append(item)
    return messages, items


def test_full_lifecycle_chain_forms_one_group_matching_spec_shape():
    """Mirrors the spec's own example: create -> follow-up -> status-check
    -> completed, all about the same subject."""
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please submit the report by 2026-09-10"),
        ("MSG_2", datetime(2026, 9, 3), "Priya", "Following up on submit the report; is it in progress?"),
        ("MSG_3", datetime(2026, 9, 5), "Ram", "Please check the latest status of submit the report."),
        ("MSG_4", datetime(2026, 9, 9), "Priya", "Update: submit the report has been completed successfully."),
    ]
    messages, items = _build(rows)
    groups = compute_groups(messages, items)
    assert len(groups) == 1
    g = groups[0]
    for key in ("group_id", "title", "related_message_ids", "related_item_ids",
                "summary", "status", "latest_deadline", "confidence"):
        assert key in g
    assert g["related_message_ids"] == ["MSG_1", "MSG_2", "MSG_3", "MSG_4"]
    assert g["related_item_ids"] == [items[0]["item_id"]]
    assert g["status"] == "Completed"
    assert g["latest_deadline"] == "2026-09-10"
    assert 0.0 <= g["confidence"] <= 1.0


def test_single_mention_item_produces_no_group():
    rows = [("MSG_1", datetime(2026, 9, 1), "Ram", "Please submit the report by 2026-09-10")]
    messages, items = _build(rows)
    groups = compute_groups(messages, items)
    assert groups == []


def test_status_cancelled():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please submit the report by 2026-09-10"),
        ("MSG_2", datetime(2026, 9, 3), "Ram", "You can cancel submit the report; it is no longer required."),
    ]
    messages, items = _build(rows)
    groups = compute_groups(messages, items)
    assert groups[0]["status"] == "Cancelled"


def test_status_rescheduled_updates_latest_deadline():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Mentor", "A new architecture-review session is scheduled for 2026-09-10 at 09:00."),
        ("MSG_2", datetime(2026, 9, 3), "Mentor", "The architecture-review has been moved to 2026-09-15 at 10:00. Please use the new schedule."),
    ]
    messages, items = _build(rows)
    groups = compute_groups(messages, items)
    g = groups[0]
    assert g["status"] == "Rescheduled"
    assert g["latest_deadline"] == "2026-09-15"


def test_status_in_progress_when_pending_with_followup():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please review the report by 2026-09-10"),
        ("MSG_2", datetime(2026, 9, 3), "Ram", "Any progress on the item concerning review the report?"),
    ]
    messages, items = _build(rows)
    groups = compute_groups(messages, items)
    assert groups[0]["status"] == "In progress"


def test_status_unclear_on_ambiguous_update():
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please prepare the quarterly summary report by 2026-09-15"),
        ("MSG_2", datetime(2026, 9, 10), "Ram",
         "The quarterly summary report might already be finished, but I cannot confirm it."),
    ]
    messages, items = _build(rows)
    groups = compute_groups(messages, items)
    assert groups[0]["status"] == "Unclear"


def test_title_derived_for_ref_only_group_with_no_formal_item():
    """First mention of a subject can be a template match with no item of
    its own (extract.py's keywords never fire on it) -- group should still
    form, titled from that first message."""
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Following up on renew the library book; is it in progress?"),
        ("MSG_2", datetime(2026, 9, 3), "Ram", "Please check the latest status of renew the library book."),
    ]
    messages, items = _build(rows)
    groups = compute_groups(messages, items)
    assert len(groups) == 1
    assert groups[0]["related_item_ids"][0].startswith("REF_")
    assert "renew the library book" in groups[0]["title"].lower()


def test_default_config_never_merges_different_subjects_sharing_a_template():
    """Regression guard: two DIFFERENT tasks that happen to use the exact
    same phrasing template must stay in separate groups. This is the
    specific failure mode the assignment prohibits ('should not be grouped
    only because they contain one common word') -- with the semantic
    fallback off by default, only a literal shared action phrase can merge
    two subjects, so 'interview slot' and 'delivery slot' must not collide."""
    rows = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please confirm the interview slot by 2026-09-05"),
        ("MSG_2", datetime(2026, 9, 2), "Ram", "Please confirm whether you started to confirm the interview slot."),
        ("MSG_3", datetime(2026, 9, 3), "Priya", "Please confirm the delivery slot by 2026-09-05"),
        ("MSG_4", datetime(2026, 9, 4), "Priya", "Please confirm whether you started to confirm the delivery slot."),
    ]
    messages, items = _build(rows)
    groups = compute_groups(messages, items)
    assert len(groups) == 2
    all_related = [set(g["related_message_ids"]) for g in groups]
    assert {"MSG_1", "MSG_2"} in all_related
    assert {"MSG_3", "MSG_4"} in all_related


def test_confidence_lower_for_ref_only_group_than_formal_item_group():
    rows_formal = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Please submit the report by 2026-09-10"),
        ("MSG_2", datetime(2026, 9, 3), "Ram", "Update: submit the report has been completed successfully."),
    ]
    rows_ref = [
        ("MSG_1", datetime(2026, 9, 1), "Ram", "Following up on renew the library book; is it in progress?"),
        ("MSG_2", datetime(2026, 9, 3), "Ram", "Please check the latest status of renew the library book."),
    ]
    m1, i1 = _build(rows_formal)
    m2, i2 = _build(rows_ref)
    g1 = compute_groups(m1, i1)[0]
    g2 = compute_groups(m2, i2)[0]
    assert g2["confidence"] < g1["confidence"]
