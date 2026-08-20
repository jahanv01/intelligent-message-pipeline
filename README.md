# Message Intelligence Pipeline

Fully local pipeline that classifies messages, extracts tasks/events, and detects + masks sensitive information. No external API is called — all inference runs on-device.

---

## Minimum requirements

| Requirement | Version |
|---|---|
| Python | 3.8 or higher |
| pip | 21+ |
| RAM | 2 GB free (model loads into memory) |
| Disk | ~200 MB (fastembed ONNX model cache) |
| OS | Linux / macOS / Windows (WSL recommended on Windows) |

A virtual environment (`venv`) is required — do not install dependencies system-wide.

---

## Setup from scratch

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd intelligent-message-pipeline
```

**2. Add the datasets** (not included in the repo — assignment rule)

Create a `dataset/` folder for L1 and an `l2_candidate_dataset/` folder for L2, and place the supplied CSVs there:
```
intelligent-message-pipeline/
├── dataset/
│   ├── messages.csv               ← 900-row L1 dataset supplied with the assignment
│   └── mandatory_demo_ids.csv     ← 15 mandatory L1 demo IDs supplied with the assignment
└── l2_candidate_dataset/
    ├── l2_messages.csv            ← 180-row L2 dataset (MSG_0901..MSG_1080)
    ├── l2_demo_messages.csv       ← 24-row L2 demo batch (DEMO_001..DEMO_024)
    └── l2_demo_queries.csv        ← mandatory L2 demo queries
```
Both folders are git-ignored and will never be committed. `l2_candidate_dataset/` is only needed if you're running the L2 commands below — the L1-only commands work with just `dataset/`.

**3. Create a virtual environment**

On Ubuntu/Debian, install the venv package first if missing:
```bash
sudo apt install python3-venv -y   # or python3.14-venv if using Python 3.14
```
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**4. Install all dependencies**
```bash
pip install -r requirements.txt
```
> First run downloads the `all-MiniLM-L6-v2` ONNX model (~90 MB) via `fastembed` and caches it in `~/.cache/fastembed/`. Every run after that is instant.

---

## Live demo

Hosted on Render (free tier):
**[https://intelligent-message-pipeline.onrender.com](https://intelligent-message-pipeline.onrender.com)**

> **Note on free-tier limitations:** Render's free instance uses 1 shared CPU and spins down after 15 minutes of inactivity — the first request after sleep takes ~30–60 seconds to wake up. Once running, the **Run Demo** button takes ~10 seconds (model inference on a single slow CPU core). Uploading a large CSV may trigger an out-of-memory kill — the free tier has 512 MB RAM shared between the model and the data. This deployment is for **demo purposes only** and is not production-ready.
>
> Hugging Face Spaces was the preferred hosting platform (zero cold-start, faster CPU), but as of 2026 the free tier no longer includes Gradio/Docker compute — a PRO subscription is required.

---

## Run the Gradio web UI locally

```bash
source .venv/bin/activate
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

- **Demo tab** — click *Run Demo* to process 8 built-in fabricated messages (no dataset needed)
- **Upload tab** — upload your own CSV with columns `message_id`, `timestamp`, `sender`, `message` (e.g. `l2_candidate_dataset/l2_demo_messages.csv`)

Both tabs show four tables: classification, extraction, sensitive findings, and the L2 **Priority** table (message/item id, priority, confidence, signals, reason).

> Do not upload the real assignment dataset to a public Space.

---

## Run the full pipeline (L1, or L1 + L2)

```bash
# L1 only (900 messages)
python main.py --input dataset/messages.csv --mandatory dataset/mandatory_demo_ids.csv

# L1 + L2 + the L2 demo batch, chronologically, in one run
python main.py --input dataset/messages.csv \
    --extra-input l2_candidate_dataset/l2_messages.csv l2_candidate_dataset/l2_demo_messages.csv \
    --mandatory dataset/mandatory_demo_ids.csv
```

`--extra-input` accepts any number of additional files; everything is concatenated and re-sorted by timestamp before processing, so "L2 after L1, chronological order" holds regardless of argument order. This writes four output files to `results/`:

```
results/
  output_classifications.json      ← Part 1: every message classified
  output_extracted_items.json      ← Part 2: extracted tasks and events
  output_sensitive_findings.json   ← Part 3: sensitive findings (values masked)
  output_priorities.json           ← L2 Part 1: priority decision per actionable message
```

The `results/` folder is git-ignored — outputs are never committed to the repo.

---

## Run the demo scripts (mandatory IDs only)

These scripts are for the video demonstration. They read from the dataset and print formatted results for the 15 mandatory message IDs.

```bash
# Part 1 — classification results + category coverage
python3 submission_tests/part1_classification.py

# Part 2 — task/event extraction results
python3 submission_tests/part2_extraction.py

# Part 3 — sensitive information detection + masking
python3 submission_tests/part3_sensitive.py

# L2 Part 1 — priority decisions for the L2 demo batch (runs the full L1+L2+demo pipeline)
python3 submission_tests/part1b_priority_l2.py
```

Each script saves its output to `results/` as a JSON file.

---

## Run the test suite

```bash
pip install -r requirements-dev.txt   # only needed once
pytest tests/ -v
```

54 tests cover classification, extraction, sensitive detection, pipeline integration, timestamp parsing, message linking, and priority scoring. All tests use fabricated example messages — the real dataset is never used in tests.

---

## Project structure

```
intelligent-message-pipeline/
├── app.py                   ← Gradio web UI (demo + CSV upload)
├── classify.py              ← Part 1: semantic sub-class classifier
├── extract.py               ← Part 2: task/event extractor
├── sensitive.py             ← Part 3: sensitive info detector + masker
├── linking.py               ← L2: status-update templates + subject registry (shared by priority.py)
├── priority.py              ← L2 Part 1: priority and action engine
├── main.py                  ← pipeline entry point (run this)
├── requirements.txt
├── requirements-dev.txt
├── dataset/                 ← git-ignored — place messages.csv and mandatory_demo_ids.csv here
├── l2_candidate_dataset/     ← git-ignored — place l2_messages.csv and l2_demo_messages.csv here
├── results/                 ← git-ignored — all output JSON files written here
├── submission_tests/        ← demo scripts for the video
│   ├── part1_classification.py
│   ├── part1b_priority_l2.py
│   ├── part2_extraction.py
│   └── part3_sensitive.py
└── tests/                   ← pytest test suite (54 tests)
```

---

## How classification works (Part 1)

Instead of keyword lists, the classifier uses a sub-class taxonomy with **17 narrow sub-classes**. Each sub-class has 4–6 human-written anchor sentences. `fastembed` (ONNX, CPU-only, no PyTorch) embeds the message and all anchors using `all-MiniLM-L6-v2`, then picks the closest sub-class by cosine similarity. The sub-class maps up to the main category deterministically.

Sensitive information is still detected via regex (OTP / card / PIN / token / recovery code) and overrides the model result — regex is the right tool for fixed-format secrets.

```
message
  ├─► regex sensitive detector ──► sensitive_information (hard override)
  └─► fastembed (ONNX)
           ↓ cosine similarity vs all sub-class anchors
      best sub-class → main category
      confidence = normalised similarity score
```

**Sub-class taxonomy**

| Category | Sub-classes |
|---|---|
| `meeting_or_event` | `scheduled_meeting`, `calendar_event`, `catchup_or_interview`, `event_announcement` |
| `action_required` | `task_assignment`, `deadline_reminder`, `approval_or_review` |
| `personal_information` | `location_disclosure`, `preference_disclosure`, `profile_or_contact`, `health_information` |
| `general_information` | `casual_or_social`, `system_or_infra_update`, `logistics_info`, `status_update` |
| `promotional` | `discount_or_sale`, `subscription_or_plan` |
| `sensitive_information` | regex-only |

See `classify.py` for the full anchor sentences and similarity logic.

---

## How extraction works (Part 2)

Messages are matched against an extraction sub-class taxonomy (8 sub-classes: `scheduled_meeting`, `calendar_event`, `catchup_or_interview`, `event_announcement`, `submission_task`, `review_task`, `deadline_task`, `follow_up_task`). Word-boundary matching prevents false substring hits (e.g. `check` does not match `checking`).

- **Date** — explicit ISO date in the text wins; falls back to relative phrases (`tomorrow`, `next week`, named weekday) resolved against the message's own timestamp.
- **Time** — only extracted when AM/PM is explicit or 24-hour format is used. Bare `"at 9"` is left `null` — not guessed.
- **Person** — first capitalised name in the message, excluding sentence-start words, days, months, and dataset-specific non-name words.

See `extract.py`.

---

## How sensitive detection works (Part 3)

Regex patterns for: OTP, PIN, password, card number (with and without spaces/dashes), bank account number, auth token, recovery code, phone number, email address, physical address. Each pattern requires nearby context keywords where needed to reduce false positives. Overlapping matches are deduplicated. The matched value is replaced with `***` in `masked_text` — the raw value never appears in any output file or log.

See `sensitive.py`.

---

## L2 extension — how it builds on L1

L2 does not replace any L1 module — `classify.py`, `extract.py`, and `sensitive.py` still run unchanged (`extract.py` gained two extra sub-classes, see below) and produce the same three L1 output files. L2 adds two new modules on top:

```
messages (L1, then L2, then the L2 demo batch — chronological throughout)
        │
        ▼
  extract_item()  ──────► items (extract.py, + 2 new sub-classes: "New task:
                           ... by DATE" and "A new ... is scheduled for
                           DATE at TIME" — L2 phrasing the L1 keyword list
                           didn't cover)
        │
        ▼
  linking.py   — regex templates recognise L2's status-update phrasing
   ├─ detect_status_update(message)  → completed / cancelled / rescheduled /
   │                                    deadline_changed / conflicting_deadline /
   │                                    status_check / ambiguous
   └─ SubjectRegistry.find(phrase)   → earliest earlier item whose own
                                        description CONTAINS that phrase
        │
        ▼
  priority.py  — one chronological pass, scores every actionable message
        │
        ▼
  results/output_priorities.json
```

Run order is enforced by `main.py --extra-input` re-sorting on timestamp (not by argument order), and by `linking.py`'s registry only ever looking *backward* in time — a message can only update a subject that was already seen.

### How related messages are linked (without a separate "grouping" pass yet)

L1's original titles are free-text sentences a human could have phrased any way. L2's status-update messages, by contrast, are template-generated and always **restate the task/event's own action phrase verbatim** ("Update: **review the privacy checklist** has been completed successfully."). So instead of canonicalising both sides into a symmetric key (fragile), `linking.py` does a one-sided match: pull the clean phrase out of the *update* message with a regex template, then find the **earliest** previously-registered item whose own description contains that phrase as a substring. Phrases shorter than 6 characters are never searched (avoids matching on generic words like "it"). This is the same primitive L2 Part 2 (related-message grouping) will build its groups on — it is not duplicated there.

If a status-changing message (e.g. "Update: X has been completed") is the *first* mention of X — extract.py's keyword list never turned it into a formal item — it still gets linked: a lightweight `REF_<message_id>` reference is registered from that first mention so later messages about the same X can still be found. This is visibly distinct from a real `TASK_/EVENT_` id in the output, on purpose — it's a transparent flag that the underlying item extraction has a gap, not a silent guess.

### How priority is calculated and updated

`priority.py` scores every actionable message on a small, fully-documented weighted signal table (`SIGNAL_WEIGHTS` in `priority.py`):

| Signal | Weight | Meaning |
|---|---|---|
| `overdue` / `deadline_today` | +3 | deadline vs. reference time |
| `deadline_within_2_days` | +2 | |
| `deadline_within_7_days` | +1 | |
| `deadline_future` / `no_deadline` | 0 | |
| `urgent_language` | +2 | asap/urgent/critical/immediately, or "treat this as urgent" |
| `deadline_moved_earlier` | +2 | a later message pulled the deadline in |
| `deadline_moved_later` | −1 | a later message pushed the deadline out |
| `conflicting_deadline` | +1 (and −0.15 confidence) | two messages state different deadlines |
| `response_required` | +1 | sub-class is review/follow-up/catch-up, or the message is a status check |
| `sensitive_content` | +1 | this message_id also appears in the sensitive findings |
| `priority_sender` | +0.5 | sender is Project Lead / HR Team / Mentor / Operations |
| `restated_subject` | −0.5 | the message merely re-mentions an already-tracked subject |
| `reopened` | +0.5 | a new deadline/reschedule contradicts a prior completed/cancelled status |
| task/event base | +0.5 to +1 | sub-class weight (deadline/review/submission/follow-up tasks weigh more than a plain calendar event) |

Weights sum to a score, bucketed to a level: **critical** ≥ 6, **high** ≥ 4, **medium** ≥ 2, else **low**. `completed`/`cancelled` are hard overrides straight to `low` (confidence 0.9) — *unless* a later message about the same subject moves its deadline, reschedules it, or reports a conflicting deadline, in which case the item is treated as reopened (a fresh, more specific signal contradicts a stale status) and scored normally again. Confidence starts at 0.55 and is adjusted by how much corroborating evidence is present (explicit deadline, explicit status keyword, known sender role) minus penalties for `ambiguous_status` (uncertain language: "might"/"may"/"probably" — never changes the *level*, only how sure we are of it) and `conflicting_deadline`.

The `reason` string is generated from the top 1–2 highest-weight signals that actually fired for that decision — never a static template.

**Updating an existing decision:** a later message about the same subject does not edit an earlier JSON record in place — `results/output_priorities.json` is an append-only log, one entry per actionable message, in chronological order. The *current* priority for an item is simply its most recent entry (`item_id` is stable across all of a subject's entries). This mirrors the assignment's own example (`MSG_1042` / `TASK_084`): the update message gets its own `message_id`, but the `item_id` it's scored against is the original.

If a message can't be confidently tied to a task/event — a generic ambiguous statement like `"This may no longer be urgent."` with no task-taxonomy keyword and no matching earlier subject — no decision is produced for it at all. No priority level is invented without evidence.

### Reference time, not wall-clock time

`overdue`/`deadline_today`/proximity buckets are computed against the **timestamp of the last processed message in the batch**, not `datetime.now()`. This dataset's dates are fictional (September–October 2026); using real wall-clock time would misjudge every deadline as "far future" today. This mirrors the existing L1 rule in `extract.py._resolve_date` (relative phrases like "tomorrow" are resolved against the message's own timestamp, never wall-clock).

---

## Assumptions and limitations

- Messages are expected to be in English.
- The fastembed ONNX model may misclassify ambiguous messages where a single sentence contains signals from multiple categories (e.g. `"I might prefer evening meetings now"` — preference vs. event).
- Person extraction relies on capitalisation; lowercase names in casual text are missed.
- Sensitive detection uses regex — novel secret formats not covered by the patterns will be missed.
- `extract.py`'s keyword taxonomy still misses some phrasings (e.g. a plain "The X has been moved to DATE" reschedule where X was never independently extracted as an item by any keyword match). `priority.py`'s `REF_` fallback catches most of these by linking off the update message itself, but a subject that is *never* mentioned in an update-style template and *never* hits an extraction keyword has no way to be tracked — this is a real gap in extraction coverage, not a priority-engine guess.
- Item ids are derived from the full `message_id` (e.g. `TASK_MSG_0042`, not a bare running number) specifically so that combining files with independent numbering (the L1 file, the L2 file, and the separate L2 demo file, which all restart their own numeric-looking suffixes) can never silently collide two unrelated items into one.
- Priority weights are hand-set constants (`priority.py`), not learned from labelled data — there is no ground-truth priority dataset to fit against. They are documented above so every decision is explainable and reproducible, not a black box.

---

## AI-tool usage disclosure

- **L1:** GitHub Copilot (VS Code) was used to assist with code suggestions, debugging, and refactoring during development.
- **L2:** Claude Code was used to help design and implement the priority/linking engine (`linking.py`, `priority.py`), extend `extract.py`/`main.py`/`app.py`, and write the accompanying tests, working from the assignment brief and the existing L1 codebase.
- All code was reviewed, understood, and verified by the author before submission — including manually tracing the signal weights and linking logic against the actual dataset (see "How priority is calculated and updated" above).

