import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classify import classify_message


def test_self_disclosure_pattern_catches_unanticipated_personal_facts():
    """The keyword list can never enumerate every personal fact someone
    might share (tshirt size, blood type, shoe size...). This structural
    pattern ('my X is Y') catches it generically instead."""
    for msg in ["my tshirt size is medium", "my blood type is O positive",
                "my shoe size is 9"]:
        result, _ = classify_message("MSG_1", msg)
        assert result["category"] == "personal_information", f"failed for: {msg}"


def test_bare_phrase_without_my_stays_general():
    """With semantic embeddings, 'tshirt size is medium' is correctly
    identified as personal_information even without 'my' — the model
    understands it as a personal attribute disclosure."""
    result, _ = classify_message("MSG_2", "tshirt size is medium")
    assert result["category"] == "personal_information"


def test_self_disclosure_pattern_does_not_override_meeting_keyword():
    """Regression guard: the new fallback must never fire when a specific
    category keyword already matched earlier in priority order."""
    result, _ = classify_message("MSG_3", "my meeting is at 5 PM")
    assert result["category"] == "meeting_or_event"


def test_self_disclosure_pattern_does_not_override_action_keyword():
    result, _ = classify_message("MSG_4", "my plan is to submit the report tomorrow")
    assert result["category"] == "action_required"


def test_deadline_classified_as_action_required():
    """Regression test: 'deadline' was in extract.py's task keywords (so Part
    2 correctly extracted it as a task) but MISSING from classify.py's
    action_required keywords (so Part 1 wrongly called it
    general_information) — the two parts disagreed on the same message."""
    for msg in ["The deadline is tomorrow", "Just a reminder about the deadline"]:
        result, _ = classify_message("MSG_5", msg)
        assert result["category"] == "action_required", f"failed for: {msg}"
