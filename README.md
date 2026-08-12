# Message Intelligence Pipeline

Fully local pipeline that classifies messages, extracts tasks/events, and detects + masks sensitive information. No external API is called — all inference runs on-device.

---

## Minimum requirements

| Requirement | Version |
|---|---|
| Python | 3.8 or higher |
| pip | 21+ |
| RAM | 2 GB free (model loads into memory) |
| Disk | ~500 MB (PyTorch CPU build + model cache) |
| OS | Linux / macOS / Windows (WSL recommended on Windows) |

A virtual environment (`venv`) is required — do not install dependencies system-wide.

---

## Setup from scratch

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd intelligent-message-pipeline
```

**2. Add the dataset** (not included in the repo — assignment rule)

Create a `dataset/` folder inside the project root and place both CSV files there:
```
intelligent-message-pipeline/
└── dataset/
    ├── messages.csv              ← 900-row dataset supplied with the assignment
    └── mandatory_demo_ids.csv   ← 15 mandatory demo IDs supplied with the assignment
```
The `dataset/` folder is git-ignored and will never be committed.

**3. Create a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**4. Install PyTorch (CPU-only, avoids the large GPU build)**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**5. Install all dependencies**
```bash
pip install -r requirements.txt
```
> First run downloads the `all-MiniLM-L6-v2` model (~90 MB) and caches it in `~/.cache/huggingface/`. Every run after that is instant.

---

## Live demo

Hosted on Render (free tier):
**[https://intelligent-message-pipeline.onrender.com](https://intelligent-message-pipeline.onrender.com)**

> **Note on free-tier limitations:** Render's free instance uses 1 shared CPU and spins down after 15 minutes of inactivity — the first request after sleep takes ~30–60 seconds to wake up. Inference is also slower than on dedicated hardware.
>
> Hugging Face Spaces was the preferred hosting platform (zero cold-start, faster CPU), but as of 2026 the free tier no longer includes Gradio/Docker compute — a PRO subscription is required.

---

## Run the Gradio web UI locally

```bash
source .venv/bin/activate
python3 app.py
```

Open `http://127.0.0.1:7860` in your browser.

- **Demo tab** — click *Run Demo* to process 8 built-in fabricated messages (no dataset needed)
- **Upload tab** — upload your own CSV with columns `message_id`, `timestamp`, `sender`, `message`

> Do not upload the real assignment dataset to a public Space.

---

## Run the full pipeline (all 900 messages)

```bash
python3 main.py --input dataset/messages.csv --mandatory dataset/mandatory_demo_ids.csv
```

This processes every message in chronological order and writes three output files to `results/`:

```
results/
  output_classifications.json      ← Part 1: every message classified
  output_extracted_items.json      ← Part 2: extracted tasks and events
  output_sensitive_findings.json   ← Part 3: sensitive findings (values masked)
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
```

Each script saves its output to `results/` as a JSON file.

---

## Run the test suite

```bash
pip install -r requirements-dev.txt   # only needed once
pytest tests/ -v
```

31 tests cover classification, extraction, sensitive detection, pipeline integration, and timestamp parsing. All tests use fabricated example messages — the real dataset is never used in tests.

---

## Project structure

```
intelligent-message-pipeline/
├── app.py                   ← Gradio web UI (demo + CSV upload)
├── classify.py              ← Part 1: semantic sub-class classifier
├── extract.py               ← Part 2: task/event extractor
├── sensitive.py             ← Part 3: sensitive info detector + masker
├── main.py                  ← pipeline entry point (run this)
├── requirements.txt
├── requirements-dev.txt
├── dataset/                 ← git-ignored — place messages.csv and mandatory_demo_ids.csv here
├── results/                 ← git-ignored — all output JSON files written here
├── submission_tests/        ← demo scripts for the video
│   ├── part1_classification.py
│   ├── part2_extraction.py
│   └── part3_sensitive.py
└── tests/                   ← pytest test suite (31 tests)
```

---

## How classification works (Part 1)

Instead of keyword lists, the classifier uses a sub-class taxonomy with **17 narrow sub-classes**. Each sub-class has 4–6 human-written anchor sentences. A local sentence-transformer model (`all-MiniLM-L6-v2`, ~90 MB, CPU) embeds the message and all anchors, then picks the closest sub-class by cosine similarity. The sub-class maps up to the main category deterministically.

Sensitive information is still detected via regex (OTP / card / PIN / token / recovery code) and overrides the model result — regex is the right tool for fixed-format secrets.

```
message
  ├─► regex sensitive detector ──► sensitive_information (hard override)
  └─► sentence-transformer
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

## Assumptions and limitations

- Messages are expected to be in English.
- The sentence-transformer model may misclassify ambiguous messages where a single sentence contains signals from multiple categories (e.g. `"I might prefer evening meetings now"` — preference vs. event).
- Person extraction relies on capitalisation; lowercase names in casual text are missed.
- Sensitive detection uses regex — novel secret formats not covered by the patterns will be missed.

---

## AI-tool usage disclosure

GitHub Copilot (VS Code) was used to assist with code suggestions, debugging, and refactoring during development. All code was reviewed, understood, and verified by the author before submission.

