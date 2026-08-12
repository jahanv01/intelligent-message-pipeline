import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from classify import classify_message


def test_action_required_classified():
    result, _ = classify_message("MSG_1", "Please submit the report by tomorrow, kindly revert asap")
    assert result["category"] == "action_required"
    assert result["confidence"] > 0.5
    assert result["reason"]  # must always have a non-empty reason


def test_meeting_classified():
    result, _ = classify_message("MSG_2", "Let's meet at 9 AM for the sprint call")
    assert result["category"] == "meeting_or_event"


def test_promotional_classified():
    result, _ = classify_message("MSG_3", "Flat 50% off on all items, shop now!")
    assert result["category"] == "promotional"


def test_general_fallback_when_no_keywords_match():
    result, _ = classify_message("MSG_4", "Just checking in, how's your day going?")
    assert result["category"] == "general_information"
    assert 0.0 < result["confidence"] <= 0.95


def test_sensitive_takes_priority_over_action_required():
    """A message can look like an action item AND contain sensitive data —
    sensitive detection must win so it never gets missed."""
    result, sens_hits = classify_message("MSG_5", "Please confirm your OTP is 1234 asap")
    assert result["category"] == "sensitive_information"
    assert len(sens_hits) == 1


def test_confidence_never_exceeds_095():
    result, _ = classify_message("MSG_6", "please submit kindly asap review approve sign")
    assert result["confidence"] <= 0.95
