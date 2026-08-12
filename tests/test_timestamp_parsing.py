import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import parse_msg_date


def test_timestamp_format_no_day_month_swap():
    """Regression test: dayfirst=True previously swapped day/month on the
    real dataset's YYYY-MM-DD format whenever both were <=12 — e.g.
    2026-09-10 (Sep 10) was silently corrupted into 2026-10-09 (Oct 9).
    This broke chronological ordering across the whole dataset."""
    result = parse_msg_date("2026-09-10 03:31:00")
    assert result.month == 9
    assert result.day == 10


def test_chronological_order_preserved_for_ambiguous_dates():
    earlier = parse_msg_date("2026-09-10 03:31:00")
    later = parse_msg_date("2026-09-24 04:13:00")
    assert earlier < later


def test_date_only_format_still_parses():
    result = parse_msg_date("2026-08-15")
    assert result.year == 2026
    assert result.month == 8
    assert result.day == 15
