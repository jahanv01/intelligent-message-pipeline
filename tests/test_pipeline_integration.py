import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from main import run


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [
        ("MSG_0001", "01/08/2026", "9:00", "Ram", "Please submit the report tomorrow"),
        ("MSG_0002", "01/08/2026", "10:00", "Priya", "Meet at 11 AM tomorrow"),
        ("MSG_0003", "02/08/2026", "8:00", "Ram", "Your OTP is 5521"),
    ]
    df = pd.DataFrame(rows, columns=["message_id", "timestamp", "time", "sender", "message"])
    input_path = tmp_path / "test_input.xlsx"
    df.to_excel(input_path, engine="openpyxl", index=False)

    result = run(str(input_path), None)

    assert len(result["classifications"]) == 3
    assert len(result["sensitive_findings"]) == 1
    assert result["row_errors"] == []
    assert (tmp_path / "results" / "output_classifications.json").exists()
    assert (tmp_path / "logs" / "run.log").exists()


def test_pipeline_survives_a_broken_row(tmp_path, monkeypatch):
    """One malformed row must not crash the whole batch — it should be
    logged to output_row_errors.json and everything else still processes."""
    monkeypatch.chdir(tmp_path)
    rows = [
        ("MSG_0001", "01/08/2026", "9:00", "Ram", "Please submit the report tomorrow"),
        ("MSG_0002", "01/08/2026", "10:00", "Priya", None),  # broken: missing message text
        ("MSG_0003", "02/08/2026", "8:00", "Ram", "Your OTP is 5521"),
    ]
    df = pd.DataFrame(rows, columns=["message_id", "timestamp", "time", "sender", "message"])
    input_path = tmp_path / "test_input.xlsx"
    df.to_excel(input_path, engine="openpyxl", index=False)

    result = run(str(input_path), None)

    # the two good rows must still have been processed despite the broken one
    good_ids = {c["message_id"] for c in result["classifications"]}
    assert "MSG_0001" in good_ids
    assert "MSG_0003" in good_ids


def test_run_log_never_contains_raw_sensitive_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [("MSG_0001", "01/08/2026", "9:00", "Ram", "Your OTP is 778899 today")]
    df = pd.DataFrame(rows, columns=["message_id", "timestamp", "time", "sender", "message"])
    input_path = tmp_path / "test_input.xlsx"
    df.to_excel(input_path, engine="openpyxl", index=False)

    run(str(input_path), None, verbose=True)  # even in verbose mode

    run_log = (tmp_path / "logs" / "run.log").read_text()
    assert "778899" not in run_log


def test_mandatory_ids_check_flags_missing_ids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [("MSG_0001", "01/08/2026", "9:00", "Ram", "Hello there")]
    df = pd.DataFrame(rows, columns=["message_id", "timestamp", "time", "sender", "message"])
    input_path = tmp_path / "test_input.xlsx"
    df.to_excel(input_path, engine="openpyxl", index=False)

    mand_df = pd.DataFrame({"message_id": ["MSG_0001", "MSG_9999"]})  # MSG_9999 doesn't exist
    mand_path = tmp_path / "mandatory.csv"
    mand_df.to_csv(mand_path, index=False)

    run(str(input_path), str(mand_path))

    run_log = (tmp_path / "logs" / "run.log").read_text()
    assert "MSG_9999" in run_log  # missing mandatory ID must show up in the log
