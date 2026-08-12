import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract import extract_item

BASE_DATE = datetime(2026, 8, 10)


def test_task_with_relative_date_resolved():
    item = extract_item("MSG_1", "Please submit the report tomorrow, this is urgent", BASE_DATE, "Ram")
    assert item["type"] == "task"
    assert item["deadline"] == "2026-08-11"
    assert item["priority"] == "high"


def test_time_requires_explicit_ampm():
    item = extract_item("MSG_2", "Let's meet at 9 AM for the sprint call", BASE_DATE, "Priya")
    assert item["time"] == "9 AM"


def test_bare_time_without_ampm_left_null():
    """Must NOT guess AM or PM when it isn't stated — assignment rule."""
    item = extract_item("MSG_3", "Call with Arjun at 12, don't forget", BASE_DATE, "Ram")
    assert item["time"] is None


def test_person_extracted_correctly_not_first_word():
    """Regression test: 'Call' (first word, capitalized by sentence position)
    was previously misread as a person's name instead of 'Arjun'."""
    item = extract_item("MSG_4", "Call with Arjun at 12, don't forget", BASE_DATE, "Ram")
    assert item["person"] == "Arjun"


def test_explicit_date_wins_over_relative_phrase():
    """Regression test: a message with both 'tomorrow' and an explicit date
    previously used the vaguer relative phrase instead of the explicit date."""
    item = extract_item(
        "MSG_5",
        "Meeting with Arjun scheduled at 12 tomorrow, deadline is 2026-08-20",
        BASE_DATE, "Karan",
    )
    assert item["deadline"] == "2026-08-20"


def test_no_extraction_for_non_task_non_event_message():
    item = extract_item("MSG_6", "Just saying hi, how are you", BASE_DATE, "Ram")
    assert item is None


def test_unresolved_date_stays_null_not_guessed():
    item = extract_item("MSG_7", "Please review the document", BASE_DATE, "Ram")
    assert item["deadline"] is None
