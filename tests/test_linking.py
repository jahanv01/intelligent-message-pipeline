import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from linking import detect_status_update, normalize_subject, resolve_when, SubjectRegistry

BASE_DATE = datetime(2026, 9, 28)


def test_detect_completed():
    u = detect_status_update("Update: prepare the demo video has been completed successfully.")
    assert u["update_type"] == "completed"
    assert u["subject_norm"] == "prepare the demo video"


def test_detect_cancelled():
    u = detect_status_update("You can cancel call the service centre; it is no longer required.")
    assert u["update_type"] == "cancelled"
    assert u["subject_norm"] == "call the service centre"


def test_detect_deadline_changed_earlier_and_urgent():
    u = detect_status_update(
        "The deadline to finish the test cases is now 2026-09-29, earlier than "
        "previously planned. Treat this as urgent."
    )
    assert u["update_type"] == "deadline_changed"
    assert u["subject_norm"] == "finish the test cases"
    assert u["when_raw"] == "2026-09-29"
    assert u["urgent"] is True


def test_detect_rescheduled_event():
    u = detect_status_update(
        "The family dinner has been moved to 2026-09-29 at 10:00. Please use the new schedule."
    )
    assert u["update_type"] == "rescheduled"
    assert u["subject_norm"] == "family dinner"


def test_detect_status_check():
    u = detect_status_update("Following up on review the model results; is it in progress?")
    assert u["update_type"] == "status_check"
    assert u["subject_norm"] == "review the model results"


def test_leading_noise_stripped():
    """'Follow-up: Additional update:' style prefixes must not block a template match."""
    u = detect_status_update(
        "Follow-up: update: prepare the demo video has been completed successfully."
    )
    assert u["update_type"] == "completed"
    assert u["subject_norm"] == "prepare the demo video"


def test_non_actionable_message_returns_none():
    assert detect_status_update("Just saying hi, how are you") is None


def test_too_short_subject_not_linkable():
    """A subject phrase shorter than the safety floor must not be searched."""
    u = detect_status_update("You can cancel it; it is no longer required.")
    assert u["subject_norm"] is None


def test_resolve_when_explicit_date_and_time():
    deadline, time_ = resolve_when("2026-09-29 at 10:00", BASE_DATE)
    assert deadline == "2026-09-29"
    assert time_ == "10:00"


def test_resolve_when_relative_date_and_ampm_time():
    deadline, time_ = resolve_when("tomorrow at 10 AM", BASE_DATE)
    assert deadline == "2026-09-29"
    assert time_ == "10 AM"


def test_registry_finds_earliest_containing_item():
    reg = SubjectRegistry()
    reg.register("EVENT_1", "Calendar update: family dinner, 2026-09-19 at 10:00, the library.", {})
    reg.register("EVENT_9", "Reminder about the family dinner tonight.", {})
    assert reg.find("family dinner") == "EVENT_1"


def test_registry_find_none_when_unregistered():
    reg = SubjectRegistry()
    assert reg.find("sprint planning") is None
