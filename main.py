"""
Main Pipeline
---------------
Reads messages.xls (or .xlsx/.csv), runs classification, extraction,
and sensitive-info detection on every row IN CHRONOLOGICAL ORDER, and
writes three output JSON files.

Everything runs 100% locally. No message content is sent anywhere.

USAGE:
    python main.py --input messages.xls --mandatory mandatory_demo_ids.csv
    python main.py --input messages.xls --verbose      # for local debugging

Logs are written to logs/run.log (safe, IDs+categories only — fine to
include in your repo) and logs/debug.log (may contain raw message text
when --verbose is used — LOCAL USE ONLY, do not commit or share this file).

Expected columns in the input file: message_id, timestamp, time, sender, message
  - timestamp column format: DD/MM/YYYY (per this dataset)
  - dates mentioned INSIDE message text: YYYY-MM-DD (handled in extract.py)
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from classify import classify_message, wait_for_model_ready
from extract import extract_item
from priority import compute_priorities
from grouping import compute_groups, annotate_superseded
from privacy import route_all


def setup_logging(verbose: bool = False) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)
    fh = logging.FileHandler("logs/run.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

DATE_FORMAT_HINT = "DMY"


def load_messages(path: str, logger) -> pd.DataFrame:
    logger.info(f"Loading input file: {path}")
    try:
        if path.lower().endswith(".xls"):
            df = pd.read_excel(path, engine="xlrd")
        elif path.lower().endswith(".xlsx"):
            df = pd.read_excel(path, engine="openpyxl")
        else:
            df = pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to load input file '{path}': {e}")
        raise

    required = {"message_id", "timestamp", "sender", "message"}
    missing = required - set(c.lower() for c in df.columns)
    if missing:
        logger.warning(f"Expected columns missing: {missing}. Found columns: {list(df.columns)}")
    else:
        logger.info(f"All required columns present. {len(df)} rows loaded.")

    return df


def parse_msg_date(raw_ts, logger=None) -> datetime:
    """Best-effort parse of the message's own timestamp for relative-date math.

    The real dataset's timestamp column is YYYY-MM-DD HH:MM:SS (confirmed by
    inspecting actual values — NOT DD/MM/YYYY as originally assumed). Using
    dayfirst=True against this format silently swapped day/month whenever
    both were <=12 (e.g. 2026-09-10 became 2026-10-09) — a real data
    corruption bug, not a cosmetic warning. Parsing the known format
    explicitly removes all ambiguity instead of guessing.
    """
    if pd.isna(raw_ts):
        return datetime.now()
    if isinstance(raw_ts, datetime):
        return raw_ts
    s = str(raw_ts)
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        pass
    # last resort fallback for any other format actually encountered in the data
    try:
        parsed = pd.to_datetime(s, dayfirst=False).to_pydatetime()
        if logger:
            logger.warning(f"Timestamp '{s}' didn't match expected YYYY-MM-DD format, "
                            f"used fallback parser -> {parsed}. Verify this is correct.")
        return parsed
    except Exception as e:
        if logger:
            logger.warning(f"Could not parse timestamp '{s}', defaulting to now(). Error: {e}")
        return datetime.now()


def run(input_path: str, mandatory_path: Optional[str], verbose: bool = False, extra_inputs=None):
    logger = setup_logging(verbose=verbose)
    start = time.time()
    logger.info("=== Pipeline run started ===")

    # classify.py loads its embedding model in a background thread so
    # app.py's web server can open its port immediately (see classify.py).
    # This batch pipeline produces the actual submitted output files, so it
    # must wait for full semantic classification here -- unlike a live web
    # request, there's no user waiting on a fast first response to justify
    # the keyword-fallback tradeoff.
    if not wait_for_model_ready(timeout=60):
        logger.warning("Embedding model did not finish loading within 60s -- "
                        "classification will fall back to keyword matching.")

    df = load_messages(input_path, logger)
    for extra_path in (extra_inputs or []):
        # L2 batches (e.g. l2_messages.csv, then l2_demo_messages.csv) are
        # appended and re-sorted below -- "process L2 after L1" is guaranteed
        # by the chronological sort, not by argument order.
        df = pd.concat([df, load_messages(extra_path, logger)], ignore_index=True)

    if "timestamp" in [c.lower() for c in df.columns]:
        ts_col = [c for c in df.columns if c.lower() == "timestamp"][0]
        df["_sort_key"] = df[ts_col].apply(lambda x: parse_msg_date(x, logger))
        df = df.sort_values("_sort_key")
        logger.info("Sorted messages into chronological order.")
    else:
        logger.warning("No 'timestamp' column found — cannot guarantee chronological order.")

    classifications = []
    extracted_items = []
    sensitive_findings = []
    row_errors = []  # (message_id, error) — so failures are diagnosable, not silent
    messages = []    # chronological {message_id, timestamp, sender, message} for priority.py

    for idx, row in df.iterrows():
        mid = str(row.get("message_id", row.get("Message ID", f"UNKNOWN_ROW_{idx}")))
        try:
            message = str(row.get("message", row.get("Message", "")))
            sender = str(row.get("sender", row.get("Sender", "")))
            raw_ts = row.get("timestamp", row.get("Timestamp", None))
            msg_date = parse_msg_date(raw_ts, logger)
            messages.append({"message_id": mid, "timestamp": msg_date, "sender": sender, "message": message})

            cls_result, sens_hits = classify_message(mid, message)
            classifications.append(cls_result)
            sensitive_findings.extend(sens_hits)

            # DEBUG only — may contain raw message text, never goes to console/run.log
            logger.debug(f"{mid} | category={cls_result['category']} | raw_text={message!r}")
            if sens_hits:
                logger.info(f"{mid} | sensitive info detected: "
                            f"{[h['sensitivity_type'] for h in sens_hits]} (masked in output)")

            item = extract_item(mid, message, msg_date, sender)
            if item:
                extracted_items.append(item)

        except Exception as e:
            # Isolate the failure to this one row — do not let one bad row
            # kill a 900-message batch. Log it so it's diagnosable.
            logger.error(f"Row failed: message_id={mid} | error={e}")
            row_errors.append({"message_id": mid, "error": str(e)})

    # reference "now" for deadline proximity/overdue = last processed message's
    # own timestamp, not wall-clock time — this is a historical/fictional
    # dataset, so datetime.now() would misjudge every deadline (see README).
    reference_dt = max((m["timestamp"] for m in messages), default=None)
    priorities = compute_priorities(messages, classifications, extracted_items, sensitive_findings, reference_dt)
    groups = compute_groups(messages, extracted_items)
    annotate_superseded(extracted_items, groups)  # adds "superseded_by" -- see README
    routing = route_all(messages, sensitive_findings)

    os.makedirs("results", exist_ok=True)
    with open("results/output_classifications.json", "w") as f:
        json.dump(classifications, f, indent=2)
    with open("results/output_extracted_items.json", "w") as f:
        json.dump(extracted_items, f, indent=2)
    with open("results/output_sensitive_findings.json", "w") as f:
        json.dump(sensitive_findings, f, indent=2)
    with open("results/output_priorities.json", "w") as f:
        json.dump(priorities, f, indent=2)
    with open("results/output_groups.json", "w") as f:
        json.dump(groups, f, indent=2)
    with open("results/output_privacy_routing.json", "w") as f:
        json.dump(routing, f, indent=2)
    if row_errors:
        with open("results/output_row_errors.json", "w") as f:
            json.dump(row_errors, f, indent=2)

    elapsed = time.time() - start
    logger.info(f"Processed {len(df)} messages in {elapsed:.2f}s.")
    logger.info(f"Classifications written: {len(classifications)} -> results/output_classifications.json")
    logger.info(f"Extracted tasks/events:  {len(extracted_items)} -> results/output_extracted_items.json")
    logger.info(f"Sensitive findings:      {len(sensitive_findings)} -> results/output_sensitive_findings.json")
    logger.info(f"Priority decisions:      {len(priorities)} -> results/output_priorities.json "
                f"(reference time: {reference_dt})")
    logger.info(f"Related-message groups:  {len(groups)} -> results/output_groups.json")
    blocked_n = sum(1 for r in routing if r["route"] == "blocked")
    confirm_n = sum(1 for r in routing if r["route"] == "requires_confirmation")
    logger.info(f"Privacy routing:         {len(routing)} -> results/output_privacy_routing.json "
                f"({blocked_n} blocked, {confirm_n} require confirmation)")
    if row_errors:
        logger.warning(f"{len(row_errors)} row(s) failed — see results/output_row_errors.json and logs/run.log")

    if mandatory_path:
        try:
            mand_df = pd.read_csv(mandatory_path) if mandatory_path.endswith(".csv") else pd.read_excel(mandatory_path)
            mand_ids = set(mand_df.iloc[:, 0].astype(str))
            found_ids = {c["message_id"] for c in classifications}
            missing_mandatory = mand_ids - found_ids
            if missing_mandatory:
                logger.warning(f"Mandatory IDs NOT found in dataset: {missing_mandatory}")
            else:
                logger.info(f"All {len(mand_ids)} mandatory demo IDs found and processed. ✓")
        except Exception as e:
            logger.error(f"Failed to check mandatory IDs file '{mandatory_path}': {e}")

    logger.info("=== Pipeline run finished ===")
    return {
        "messages": messages,
        "classifications": classifications,
        "extracted_items": extracted_items,
        "sensitive_findings": sensitive_findings,
        "priorities": priorities,
        "groups": groups,
        "routing": routing,
        "row_errors": row_errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to messages file (.xls/.xlsx/.csv)")
    parser.add_argument("--extra-input", nargs="*", default=[],
                         help="additional message file(s) to process after --input, e.g. the L2 "
                              "batch and/or the L2 demo batch. Always re-sorted chronologically "
                              "with --input, so pass them in any order.")
    parser.add_argument("--mandatory", required=False, help="path to mandatory_demo_ids file")
    parser.add_argument("--verbose", action="store_true",
                         help="enable DEBUG logging to logs/debug.log (may contain raw message text — local use only)")
    args = parser.parse_args()
    run(args.input, args.mandatory, verbose=args.verbose, extra_inputs=args.extra_input)
