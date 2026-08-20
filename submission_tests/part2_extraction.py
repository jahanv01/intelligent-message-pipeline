"""
Part 2 Demo Test — Task and Event Extraction
Run: python3 submission_tests/part2_extraction.py
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from extract import extract_item
from main import parse_msg_date

DATASET       = "dataset/messages.csv"
MANDATORY     = "dataset/mandatory_demo_ids.csv"
FULL_RESULTS  = "results/output_extracted_items.json"
RESULTS_FILE  = "results/part2_mandatory_extractions.json"


def run():
    os.makedirs("results", exist_ok=True)

    msgs = pd.read_csv(DATASET)
    ids = pd.read_csv(MANDATORY)["message_id"].tolist()

    id_map = msgs.set_index("message_id")
    found = [i for i in ids if i in id_map.index]

    print(f"Part 2 — Task & Event Extraction Demo ({len(found)}/{len(ids)} mandatory IDs)\n")
    print(f"{'ID':<12} {'TYPE:SUB_CLASS':<35} {'DEADLINE':<12} {'TIME':<8} {'PERSON':<12} TITLE")
    print("─" * 115)

    results = []
    not_extracted = []
    for msg_id in found:
        row = id_map.loc[msg_id]
        msg_date = parse_msg_date(row.get("timestamp"))
        item = extract_item(msg_id, row["message"], msg_date, row.get("sender", ""))
        if item:
            results.append(item)
            label = f"{item['type']}:{item['sub_class']}"
            print(
                f"{msg_id:<12} {label:<35} "
                f"{str(item['deadline']):<12} {str(item['time']):<8} "
                f"{str(item['person']):<12} {item['title'][:45]}"
            )
        else:
            not_extracted.append(msg_id)
            print(f"{msg_id:<12} {'NOT EXTRACTED (not a task/event)'}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")

    os.makedirs("sample_outputs", exist_ok=True)  # committed copy -- results/ is git-ignored
    with open("sample_outputs/sample_extractions.json", "w") as f:
        json.dump(results, f, indent=2)

    tasks = [r for r in results if r["type"] == "task"]
    events = [r for r in results if r["type"] == "event"]
    print(f"\nSummary: {len(tasks)} tasks  |  {len(events)} events  |  {len(not_extracted)} not extracted (not a task/event)")

    # sub-class breakdown
    print("\nSub-class breakdown:")
    sub_counts: dict = {}
    for item in results:
        sub_counts[item["sub_class"]] = sub_counts.get(item["sub_class"], 0) + 1
    for sub, count in sorted(sub_counts.items()):
        print(f"  {sub}: {count}")

    # null-field example (required by assignment point 7)
    uncertain = [i for i in results if i["deadline"] is None or i["time"] is None]
    if uncertain:
        ex = uncertain[0]
        print(f"\nExample with missing/unresolved fields ({ex['source_message_id']}):")
        print(f"  title    = {ex['title']}")
        print(f"  deadline = {ex['deadline']}  ← null: no date/deadline found in message")
        print(f"  time     = {ex['time']}  ← null: no unambiguous time found")

    # video requirement 6: show 3+ events from the full dataset
    if os.path.exists(FULL_RESULTS):
        import json as _json
        with open(FULL_RESULTS) as f:
            all_items = _json.load(f)
        all_events = [i for i in all_items if i["type"] == "event"]
        all_tasks  = [i for i in all_items if i["type"] == "task"]
        print(f"\nFull dataset: {len(all_tasks)} tasks / {len(all_events)} events across 900 messages")
        print("\nSample meetings/events (video requirement — 3 minimum):")
        print("─" * 115)
        for e in all_events[:5]:
            print(f"  {e['source_message_id']}: [{e['sub_class']}] deadline={e['deadline']} time={e['time']} | {e['title'][:60]}")

    print("\nFull message text (for reference):")
    print("─" * 115)
    for msg_id in found:
        row = id_map.loc[msg_id]
        print(f"{msg_id}: {row['message']}")


if __name__ == "__main__":
    run()
