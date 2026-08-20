"""
L2 Part 3 Demo Test — Semantic Search & Intelligent Assistant
Run: python3 submission_tests/part3_assistant_l2.py

Runs the FULL pipeline (L1 -> L2 -> L2 demo batch) exactly as main.py does,
builds the assistant's knowledge base from the real output, then answers
every mandatory query in l2_candidate_dataset/l2_demo_queries.csv.
"""
import sys
import os
import json
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import run
from assistant import build_knowledge_base, answer_query

L1_DATASET = "dataset/messages.csv"
L2_DATASET = "l2_candidate_dataset/l2_messages.csv"
L2_DEMO_DATASET = "l2_candidate_dataset/l2_demo_messages.csv"
MANDATORY = "dataset/mandatory_demo_ids.csv"
QUERIES = "l2_candidate_dataset/l2_demo_queries.csv"
RESULTS_FILE = "results/part3_mandatory_answers.json"


def run_demo():
    os.makedirs("results", exist_ok=True)

    result = run(L1_DATASET, MANDATORY, extra_inputs=[L2_DATASET, L2_DEMO_DATASET])
    kb = build_knowledge_base(
        result["messages"], result["classifications"], result["extracted_items"],
        result["sensitive_findings"], result["priorities"], result["groups"], result["routing"],
    )

    with open(QUERIES, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    answers = []
    print(f"\nL2 Part 3 — Assistant Demo ({len(rows)} mandatory queries)\n")
    print("─" * 110)
    for row in rows:
        qid, query = row["query_id"], row["query"]
        ans = answer_query(kb, query)
        answers.append({"query_id": qid, **ans})
        print(f"{qid}: {query}")
        print(f"  answer:     {ans['answer']}")
        print(f"  supporting: {ans['supporting_message_ids']}")
        print(f"  related:    {ans['related_item_ids']}  group: {ans['group_id']}")
        print(f"  scores:     {ans['relevance_scores']}  confidence: {ans['confidence']}")
        print(f"  reason:     {ans['reason']}")
        print("─" * 110)

    with open(RESULTS_FILE, "w") as f:
        json.dump(answers, f, indent=2)
    print(f"\nSaved -> {RESULTS_FILE}")


if __name__ == "__main__":
    run_demo()
