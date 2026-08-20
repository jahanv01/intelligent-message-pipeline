"""
Part 3 Demo Test — Sensitive Information Detection
Run: python3 submission_tests/part3_sensitive.py
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sensitive import detect_sensitive

DATASET   = "dataset/messages.csv"
MANDATORY = "dataset/mandatory_demo_ids.csv"
FULL_RESULTS = "results/output_sensitive_findings.json"
RESULTS_FILE = "results/part3_mandatory_findings.json"

RISK_LABEL = {"high": "🔴 high", "medium": "🟡 medium", "low": "🟢 low"}


def run():
    os.makedirs("results", exist_ok=True)

    msgs = pd.read_csv(DATASET)
    ids  = pd.read_csv(MANDATORY)["message_id"].tolist()
    rows = msgs[msgs["message_id"].isin(ids)].set_index("message_id")

    print("Part 3 — Sensitive Information Detection Demo")
    print(f"Mandatory IDs: {len(ids)}\n")
    print(f"{'ID':<12} {'TYPE':<22} {'RISK':<8} MASKED TEXT")
    print("─" * 100)

    findings = []
    clean_ids = []

    for msg_id in ids:
        if msg_id not in rows.index:
            print(f"{msg_id:<12} [ID not found in dataset]")
            continue
        hits = detect_sensitive(msg_id, rows.loc[msg_id, "message"])
        if hits:
            for h in hits:
                findings.append(h)
                risk = h["risk"]
                print(f"{msg_id:<12} {h['sensitivity_type']:<22} {risk:<8} {h['masked_text']}")
        else:
            clean_ids.append(msg_id)
            print(f"{msg_id:<12} {'—':<22} {'clean':<8}")

    print(f"\nDetected: {len(findings)}   Clean: {len(clean_ids)}")

    # -- risk breakdown -------------------------------------------------------
    print("\nRisk breakdown (mandatory IDs):")
    for level in ["high", "medium", "low"]:
        count = sum(1 for f in findings if f["risk"] == level)
        if count:
            print(f"  {level}: {count}")

    # -- full dataset summary (reads previously generated output) -------------
    if os.path.exists(FULL_RESULTS):
        with open(FULL_RESULTS) as f:
            all_findings = json.load(f)
        print(f"\nFull dataset summary ({len(all_findings)} findings across 900 messages):")
        type_counts: dict = {}
        for item in all_findings:
            type_counts[item["sensitivity_type"]] = type_counts.get(item["sensitivity_type"], 0) + 1
        for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {stype:<25} {count}")

        # one example of each type from the full dataset
        print("\nOne masked example per sensitivity type (from full dataset):")
        print("─" * 100)
        seen_types: set = set()
        for item in all_findings:
            if item["sensitivity_type"] not in seen_types:
                seen_types.add(item["sensitivity_type"])
                print(f"  [{item['sensitivity_type']}] {item['masked_text']}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")

    os.makedirs("sample_outputs", exist_ok=True)  # committed copy -- results/ is git-ignored.
    # Safe to commit: masked_text never contains a raw sensitive value (see sensitive.py).
    with open("sample_outputs/sample_sensitive_findings.json", "w") as f:
        json.dump(findings, f, indent=2)

    print("\nFull message text (for reference):")
    print("─" * 100)
    for msg_id in ids:
        if msg_id in rows.index:
            print(f"{msg_id}: {rows.loc[msg_id, 'message']}")


if __name__ == "__main__":
    run()
