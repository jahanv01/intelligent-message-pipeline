"""
Part 1 Demo Test — Message Classification
Run: python3 submission_tests/part1_classification.py
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from classify import classify_message, wait_for_model_ready

DATASET = "dataset/messages.csv"
MANDATORY = "dataset/mandatory_demo_ids.csv"
RESULTS_FILE = "results/part1_mandatory_classifications.json"


def run():
    os.makedirs("results", exist_ok=True)
    # classify.py loads its model in a background thread (see classify.py) --
    # this script calls classify_message() directly rather than through
    # main.py, so it must wait for full semantic classification itself.
    wait_for_model_ready(timeout=60)

    msgs = pd.read_csv(DATASET)
    ids = pd.read_csv(MANDATORY)["message_id"].tolist()

    # preserve order from mandatory_demo_ids.csv
    id_to_row = msgs.set_index("message_id")["message"].to_dict()
    missing = [i for i in ids if i not in id_to_row]
    if missing:
        print(f"[WARN] IDs not found in dataset: {missing}")

    found = [i for i in ids if i in id_to_row]
    print(f"Part 1 — Classification Demo ({len(found)}/{len(ids)} mandatory IDs)\n")
    print(f"{'ID':<12} {'CATEGORY':<25} {'CONF':<6} REASON")
    print("─" * 105)

    results = []
    for msg_id in found:
        message = id_to_row[msg_id]
        result, _ = classify_message(msg_id, message)
        results.append(result)
        print(f"{msg_id:<12} {result['category']:<25} {result['confidence']:<6} {result['reason']}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")

    os.makedirs("sample_outputs", exist_ok=True)  # committed copy -- results/ is git-ignored
    with open("sample_outputs/sample_classifications.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nMessage text (for reference):")
    print("─" * 105)
    for msg_id in found:
        print(f"{msg_id}: {id_to_row[msg_id]}")

    # category coverage check
    categories_seen = {r["category"] for r in results}
    all_six = {
        "action_required", "meeting_or_event", "personal_information",
        "general_information", "promotional", "sensitive_information",
    }
    print("\nCategory coverage across mandatory IDs:")
    for cat in sorted(all_six):
        tick = "✓" if cat in categories_seen else "✗ MISSING"
        count = sum(1 for r in results if r["category"] == cat)
        print(f"  {tick}  {cat} ({count})")


if __name__ == "__main__":
    run()
