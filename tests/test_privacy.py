import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from sensitive import detect_sensitive
from privacy import route_message, route_all, ROUTE_BLOCKED, ROUTE_CONFIRM, ROUTE_LOCAL


def test_high_risk_credential_is_blocked():
    hits = detect_sensitive("MSG_1", "Your OTP is 4821 for the login.")
    r = route_message("MSG_1", hits)
    assert r["route"] == ROUTE_BLOCKED
    assert r["risk"] == "high"


def test_medium_risk_pii_requires_confirmation():
    hits = detect_sensitive("MSG_1", "My home address is 42 Lake View Road, Chennai.")
    r = route_message("MSG_1", hits)
    assert r["route"] == ROUTE_CONFIRM
    assert r["risk"] == "medium"


def test_clean_message_processed_locally():
    hits = detect_sensitive("MSG_1", "Let's catch up for coffee tomorrow.")
    r = route_message("MSG_1", hits)
    assert r["route"] == ROUTE_LOCAL
    assert r["risk"] == "none"


def test_mixed_findings_blocked_wins_over_confirm():
    """A message with BOTH a card number (blocked) and a phone number
    (confirm) must route to the stricter tier, never the looser one."""
    hits = detect_sensitive("MSG_1", "My card number is 4111222233334444, call me at 9876543210.")
    r = route_message("MSG_1", hits)
    assert r["route"] == ROUTE_BLOCKED


def test_route_all_preserves_message_order():
    messages = [
        {"message_id": "MSG_1", "timestamp": datetime(2026, 9, 1), "sender": "Ram", "message": "Your OTP is 4821."},
        {"message_id": "MSG_2", "timestamp": datetime(2026, 9, 2), "sender": "Ram", "message": "Hi there!"},
    ]
    findings = detect_sensitive("MSG_1", messages[0]["message"])
    routes = route_all(messages, findings)
    assert [r["message_id"] for r in routes] == ["MSG_1", "MSG_2"]
    assert routes[0]["route"] == ROUTE_BLOCKED
    assert routes[1]["route"] == ROUTE_LOCAL
