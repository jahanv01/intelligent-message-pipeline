"""
L2 Part 1 Demo Test — Priority and Action Engine
Run: python3 submission_tests/part1b_priority_l2.py

Runs the FULL pipeline (L1 -> L2 -> L2 demo batch, chronologically) exactly
as main.py does -- priority decisions for the demo batch depend on state
built up while processing everything before it -- then prints/saves just the
decisions for the L2 demo message IDs (DEMO_001..DEMO_024) for the video.
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import run

L1_DATASET = "dataset/messages.csv"
L2_DATASET = "l2_candidate_dataset/l2_messages.csv"
L2_DEMO_DATASET = "l2_candidate_dataset/l2_demo_messages.csv"
MANDATORY = "dataset/mandatory_demo_ids.csv"
RESULTS_FILE = "results/part1b_mandatory_priorities.json"


def run_demo():
    os.makedirs("results", exist_ok=True)

    result = run(L1_DATASET, MANDATORY, extra_inputs=[L2_DATASET, L2_DEMO_DATASET])
    priorities = result["priorities"]

    demo_decisions = [p for p in priorities if p["message_id"].startswith("DEMO_")]
    demo_decisions.sort(key=lambda p: p["message_id"])

    print(f"\nL2 Part 1 — Priority Demo ({len(demo_decisions)} decisions for the L2 demo batch)\n")
    print(f"{'MESSAGE':<10} {'ITEM':<10} {'PRIORITY':<10} {'CONF':<6} REASON")
    print("─" * 110)
    for d in demo_decisions:
        print(f"{d['message_id']:<10} {d['item_id']:<10} {d['priority']:<10} {d['confidence']:<6} {d['reason']}")
        print(f"{'':<10} signals: {d['signals']}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(demo_decisions, f, indent=2)
    print(f"\nSaved -> {RESULTS_FILE}")

    covered = {p["message_id"] for p in demo_decisions}
    all_demo_ids = {f"DEMO_{i:03d}" for i in range(1, 25)}
    not_actionable = sorted(all_demo_ids - covered)
    if not_actionable:
        print(f"\nDemo IDs with no priority decision (not actionable, or an "
              f"intentionally unresolved 'insufficient evidence' case — see README):")
        for mid in not_actionable:
            print(f"  {mid}")


if __name__ == "__main__":
    run_demo()
