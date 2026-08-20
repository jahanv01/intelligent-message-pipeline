"""
Tests use only FABRICATED example strings, never real assignment data.
Run: pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sensitive import detect_sensitive


def test_otp_detected_and_masked():
    result = detect_sensitive("MSG_1", "Hey your OTP is 4821 for the login.")
    assert len(result) == 1
    assert result[0]["sensitivity_type"] == "one_time_password"
    assert result[0]["risk"] == "high"
    assert "4821" not in result[0]["masked_text"]
    assert result[0]["recommended_action"] == "do_not_store"


def test_card_number_detected_and_masked():
    result = detect_sensitive("MSG_2", "my card number is 4111222233334444 charge it")
    assert len(result) == 1
    assert result[0]["sensitivity_type"] == "card_number"
    assert "4111222233334444" not in result[0]["masked_text"]


def test_no_false_positive_on_clean_message():
    result = detect_sensitive("MSG_3", "Let's catch up for coffee tomorrow")
    assert result == []


def test_no_double_flag_on_overlapping_patterns():
    """Regression test: an account number was previously ALSO flagged as a
    phone number because the digit patterns overlapped. Must not recur."""
    result = detect_sensitive("MSG_4", "account number is 000123456789, transfer today")
    types = [r["sensitivity_type"] for r in result]
    assert types.count("bank_account_number") == 1
    assert "phone_number" not in types


def test_pin_detected():
    result = detect_sensitive("MSG_5", "my pin is 4521 don't share it")
    assert len(result) == 1
    assert result[0]["sensitivity_type"] == "pin"


def test_delivery_phrasing_address_detected():
    """'Deliver X to ADDRESS' is third-person/instructional, not the
    first-person 'I live at...' the original pattern only covered."""
    result = detect_sensitive("MSG_7", "Deliver the demo device to 22 Green Park Road, Chennai.")
    assert len(result) == 1
    assert result[0]["sensitivity_type"] == "private_address"
    assert result[0]["recommended_action"] == "ask_for_confirmation"
    assert "22 Green Park Road" not in result[0]["masked_text"]


def test_deliver_to_person_without_address_not_flagged():
    result = detect_sensitive("MSG_8", "Please deliver this update to John before the meeting.")
    assert result == []


def test_masked_text_preserves_message_structure():
    result = detect_sensitive("MSG_6", "Your OTP is 9999 today")
    masked = result[0]["masked_text"]
    assert masked.startswith("Your OTP is ")
    assert masked.endswith(" today")
    assert "9999" not in masked
