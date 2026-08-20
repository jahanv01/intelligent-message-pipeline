"""
Export mandatory-scoped sample outputs for the submission.
Run: python3 submission_tests/export_sample_outputs.py

`results/` (the full run over the real 900+180+24 message dataset) is
git-ignored on purpose -- publishing the full output would mean publishing
near-verbatim excerpts of most of the supplied dataset (group titles,
task descriptions), which the assignment explicitly prohibits ("Do not
include the original L1 or L2 datasets in your public GitHub repository").

But the assignment ALSO requires submitting a priority output file, a
related-message group output file, and a privacy-routing output file as
deliverables. This script resolves that by writing small, SAFE, mandatory-
ID-scoped copies (matching the same pattern the existing L1 submission_tests
scripts already use for classifications/extractions/sensitive findings) into
sample_outputs/, which IS committed. Every file here is scoped to the 15
mandatory L1 message IDs and/or the 24-message L2 demo batch -- the exact
messages the assignment says must appear in the video -- not the full
dataset.
"""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from main import run

L1_DATASET = "dataset/messages.csv"
L2_DATASET = "l2_candidate_dataset/l2_messages.csv"
L2_DEMO_DATASET = "l2_candidate_dataset/l2_demo_messages.csv"
MANDATORY = "dataset/mandatory_demo_ids.csv"
OUT_DIR = "sample_outputs"


def _is_scoped(message_id: str, mandatory_ids: set) -> bool:
    return message_id in mandatory_ids or message_id.startswith("DEMO_")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mandatory_ids = set(pd.read_csv(MANDATORY).iloc[:, 0].astype(str))

    result = run(L1_DATASET, MANDATORY, extra_inputs=[L2_DATASET, L2_DEMO_DATASET])

    # --- Priority (L2 Part 1) -------------------------------------------
    priorities = [p for p in result["priorities"] if _is_scoped(p["message_id"], mandatory_ids)]
    with open(f"{OUT_DIR}/sample_priorities.json", "w") as f:
        json.dump(priorities, f, indent=2)

    # --- Related-message groups (L2 Part 2) ------------------------------
    groups = [g for g in result["groups"]
              if any(_is_scoped(mid, mandatory_ids) for mid in g["related_message_ids"])]
    with open(f"{OUT_DIR}/sample_groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    # --- Privacy routing (L2 Part 3) -- no raw message text in this file,
    # so it's safe to include broadly, but scoped anyway for consistency
    # with the other files here.
    routing = [r for r in result["routing"] if _is_scoped(r["message_id"], mandatory_ids)]
    with open(f"{OUT_DIR}/sample_privacy_routing.json", "w") as f:
        json.dump(routing, f, indent=2)

    print(f"Wrote {len(priorities)} priority decisions, {len(groups)} groups, "
          f"{len(routing)} routing decisions -> {OUT_DIR}/")
    print("(Assistant answers for the 8 mandatory queries: see submission_tests/part3_assistant_l2.py "
          "-> sample_outputs/sample_assistant_answers.json)")


if __name__ == "__main__":
    main()
