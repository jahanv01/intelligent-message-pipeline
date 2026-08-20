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
  output_groups.json               ← L2 Part 2: related-message groups
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

Part 2 (related-message groups) has no separate mandatory-ID script — a group only exists once 2+ messages are linked, and its most useful demo view is the whole `results/output_groups.json` produced by the full pipeline run above.

Each script saves its output to `results/` as a JSON file.

---

## Run the test suite

```bash
pip install -r requirements-dev.txt   # only needed once
pytest tests/ -v
```

71 tests cover classification, extraction, sensitive detection, pipeline integration, timestamp parsing, message linking, priority scoring, and related-message grouping. All tests use fabricated example messages — the real dataset is never used in tests.

---

## Project structure

```
intelligent-message-pipeline/
├── app.py                   ← Gradio web UI (demo + CSV upload)
├── classify.py              ← Part 1: semantic sub-class classifier
├── extract.py               ← Part 2: task/event extractor
├── sensitive.py             ← Part 3: sensitive info detector + masker
├── linking.py               ← L2: status-update templates + subject registry + shared resolver
├── priority.py              ← L2 Part 1: priority and action engine
├── grouping.py              ← L2 Part 2: related-message grouping
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
└── tests/                   ← pytest test suite (71 tests)
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

L2 does not replace any L1 module — `classify.py`, `extract.py`, and `sensitive.py` still run unchanged (`extract.py` gained two extra sub-classes, see below) and produce the same three L1 output files. L2 adds three new modules on top:

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
  linking.py   — regex templates recognise L2's status-update phrasing,
                 ONE shared chronological resolver for Part 1 AND Part 2
   ├─ detect_status_update(message)  → completed / cancelled / rescheduled /
   │                                    deadline_changed / conflicting_deadline /
   │                                    status_check / ambiguous
   ├─ SubjectRegistry.find(phrase)   → earliest earlier item whose own
   │                                    description CONTAINS that phrase
   └─ resolve_messages(...)          → registry + per-message resolution,
                                        called by BOTH modules below so they
                                        can never disagree about groupings
        │                    │
        ▼                    ▼
  priority.py          grouping.py
  (L2 Part 1)           (L2 Part 2)
  scores every          turns each tracked subject into a group: title,
  actionable message     related message ids, chronology-derived summary,
        │                status, latest deadline, confidence
        ▼                    ▼
  output_priorities.json    output_groups.json
```

Run order is enforced by `main.py --extra-input` re-sorting on timestamp (not by argument order), and by `linking.py`'s registry only ever looking *backward* in time — a message can only update a subject that was already seen.

### How related messages are linked (without a separate "grouping" pass yet)

L1's original titles are free-text sentences a human could have phrased any way. L2's status-update messages, by contrast, are template-generated and always **restate the task/event's own action phrase verbatim** ("Update: **review the privacy checklist** has been completed successfully."). So instead of canonicalising both sides into a symmetric key (fragile), `linking.py` does a one-sided match: pull the clean phrase out of the *update* message with a regex template, then find the **earliest** previously-registered item whose own description contains that phrase as a substring. Phrases shorter than 6 characters are never searched (avoids matching on generic words like "it"). This is the same primitive L2 Part 2 (related-message grouping, below) builds its groups on via `linking.resolve_messages()` — it is not duplicated there.

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
| `category_action_required` | +0.5 (and +0.05 confidence) | classify.py's independent semantic category agrees this is actionable |
| `category_mismatch` | 0 (and −0.20 confidence) | classify.py's category is promotional/general/personal — disagrees this is actionable at all |
| `semantic_urgency` | +1.5 (and +0.05 confidence) | embedding similarity reads as urgent even without an urgent keyword — only checked when the keyword check found nothing |
| task/event base | +0.5 to +1 | sub-class weight (deadline/review/submission/follow-up tasks weigh more than a plain calendar event) |

Weights sum to a score, bucketed to a level: **critical** ≥ 6, **high** ≥ 4, **medium** ≥ 2, else **low**. `completed`/`cancelled` are hard overrides straight to `low` (confidence 0.9) — *unless* a later message about the same subject moves its deadline, reschedules it, or reports a conflicting deadline, in which case the item is treated as reopened (a fresh, more specific signal contradicts a stale status) and scored normally again. Confidence starts at 0.55 and is adjusted by how much corroborating evidence is present (explicit deadline, explicit status keyword, known sender role) minus penalties for `ambiguous_status` (uncertain language: "might"/"may"/"probably" — never changes the *level*, only how sure we are of it) and `conflicting_deadline`.

The `reason` string is generated from the top 1–2 highest-weight signals that actually fired for that decision — never a static template.

**Updating an existing decision:** a later message about the same subject does not edit an earlier JSON record in place — `results/output_priorities.json` is an append-only log, one entry per actionable message, in chronological order. The *current* priority for an item is simply its most recent entry (`item_id` is stable across all of a subject's entries). This mirrors the assignment's own example (`MSG_1042` / `TASK_084`): the update message gets its own `message_id`, but the `item_id` it's scored against is the original.

If a message can't be confidently tied to a task/event — a generic ambiguous statement like `"This may no longer be urgent."` with no task-taxonomy keyword and no matching earlier subject — no decision is produced for it at all. No priority level is invented without evidence.

### Semantic urgency (embeddings, used deliberately in exactly one place)

`urgent_language` from a fixed keyword list (`asap`/`urgent`/`immediately`/...) misses paraphrases like "please expedite this" or "it cannot wait." For that one narrow case, `priority.py` reuses `classify.py`'s already-loaded fastembed model (`classify.embed_texts` — no second model is loaded, which matters on the 512MB Render free tier) to compare the message against small "urgent" vs. "routine" anchor sets, the same anchor-similarity idea `classify.py` already uses for categories. It only runs when the keyword check found nothing, so it's a fallback that widens coverage, not a second vote that double-counts the same concept.

This was tuned empirically, not guessed — an initial threshold of 0.55 turned out to never fire at all (real margins top out around 0.19); testing against held-out paraphrases (not the anchor sentences themselves) settled on `0.12`. That same testing also caught a real false positive: *"The review could be Friday afternoon"* scored as urgent purely on topical similarity, even though "could be" is tentative language, the opposite of urgent. Embeddings alone don't reliably capture that inversion, so a deterministic hedge-word guard (`could be`/`might`/`may`/`probably`/...) runs first and disables the semantic check outright — the same "right tool for the job" reasoning `sensitive.py` already uses for fixed-format secrets, applied here to uncertainty language.

Even after that fix, `semantic_urgency` firing on a `status_check` message (e.g. "please confirm whether you started to...") about an already-overdue task can be the deciding factor that pushes `high` to `critical`, since `response_required` already contributes for the same status-check phrasing — some overlap between the two signals is a judgment call, not a bug fixed here. Fires on 5 of 566 real decisions (0.9%) — deliberately conservative.

### Reference time, not wall-clock time

`overdue`/`deadline_today`/proximity buckets are computed against the **timestamp of the last processed message in the batch**, not `datetime.now()`. This dataset's dates are fictional (September–October 2026); using real wall-clock time would misjudge every deadline as "far future" today. This mirrors the existing L1 rule in `extract.py._resolve_date` (relative phrases like "tomorrow" are resolved against the message's own timestamp, never wall-clock).

---

## How related messages are identified (L2 Part 2)

Part 2 does not re-derive "which messages are about the same subject" — that would risk two engines quietly disagreeing. It calls `linking.resolve_messages()`, the exact same chronological resolver `priority.py` uses, refactored out of `priority.py` into `linking.py` specifically so both modules share one source of truth.

```
linking.resolve_messages(messages, extracted_items)
        │  (shared with priority.py — one resolution, not two)
        ▼
  SubjectRegistry            now also tracks, per subject:
                                mentions   -- every message_id linked to it, in order
                                history    -- {message_id, event} for every mention
                                first_seen_order -- registration position (backward-only lookups)
        │
        ▼
  grouping.compute_groups()
   ├─ status:   maps internal state -> Pending / In progress / Completed /
   │            Rescheduled / Cancelled / Unclear (Unclear = last event was
   │            an ambiguous mention; In progress = still pending but with
   │            2+ mentions -- i.e. actively being followed up on)
   ├─ title:    the item's own extract.py title, or (for REF_-only subjects
   │            with no formal item) extract.py's own _clean_title() applied
   │            to the message that first established the subject
   ├─ summary:  generated from `history`'s event sequence ("created, then
   │            followed up on (3x), marked completed") -- not a static
   │            template per group, a fresh sentence built from that
   │            group's actual chronology every time
   └─ confidence: higher for real TASK_/EVENT_ items than REF_-only ones,
                +  for more corroborating mentions, − for any ambiguous or
                   conflicting-deadline event seen along the way
        │
        ▼
  results/output_groups.json   (only subjects with 2+ related messages --
                                 a lone, never-followed-up message isn't a
                                 "grouping", Part 3 can reference it directly)
```

**"Must consider message meaning, not just a common word":** the template layer itself is what does this — a match requires the full distinguishing action phrase to appear (e.g. "review the security audit report"), not one shared word. Verified directly with a regression test (`test_default_config_never_merges_different_subjects_sharing_a_template`): two different tasks phrased with the identical template ("confirm the **interview** slot" vs. "confirm the **delivery** slot") stay in separate groups.

**An embedding-based fallback was built and rejected, on purpose.** For messages that match no template at all but are clearly about a known subject in meaning (the spec's own example, "Reminder — the report is due tomorrow," wouldn't hit any fixed template), `grouping.py` has a `_link_semantic_mentions()` function using the same fastembed model, gated **off by default**. It was tested against held-out same-subject and different-subject message pairs before being trusted, the same discipline used for Part 1's semantic urgency signal — and it failed that test: adversarial pairs sharing a phrasing template but referring to different subjects scored 0.40–0.65 similarity, and 3 of 5 of those scored *higher* than genuine same-subject pairs (0.32–0.55). MiniLM's embedding is picking up sentence structure more than the specific entity discussed, and no threshold cleanly separates the two distributions. Shipping it would produce exactly the false grouping the assignment prohibits, so it stays disabled — a real example of "we do not build features just because the tool exists," proven with numbers, not asserted.

### Self-QA pass on the real dataset — two real bugs found and fixed by reading the actual output

Before calling Part 2 done, every one of the 32 groups produced from the real 1104-message run was read message-by-message (not spot-checked), and every one of the 490 items that *didn't* end up in a group was scanned for near-duplicate text against every other standalone item, looking specifically for the two failure modes that matter most here: messages wrongly grouped together, and messages that should have been grouped but weren't. Two real, non-cosmetic issues came out of that pass:

1. **16.1% of extracted items (84 of 521) were silent duplicates, and `related_item_ids` was hiding it.** `extract.py` runs independently per message — if a follow-up message's own wording happens to hit an extraction keyword (e.g. "please **confirm** whether you started to review the privacy checklist" contains "confirm"), it gets its *own* item_id even though `linking.py` correctly resolves that same message to the *original* subject for grouping and priority purposes. The result: `output_extracted_items.json` on its own looks like it has more distinct tasks than actually exist, and a consumer reading it in isolation (exactly what Part 3 is asked to do — "L1 message classifications, extracted tasks and events" as one of its knowledge sources) would silently overcount. **Fix:** `linking.SubjectRegistry` now tracks `superseded_item_ids` per subject; `related_item_ids` surfaces *all* of them (canonical first — matching the spec's own plural "Related task or event IDs" naming, which a single-id-per-group design was under-serving), and a new `annotate_superseded()` step adds a `superseded_by` field directly onto `output_extracted_items.json` so the gap is closed at the source file too, not only in `output_groups.json`. Covered by 2 new tests.
2. **A later ambiguous mention after a `Rescheduled`/`Completed`/`Cancelled` status was being silently ignored.** `_status_for()` checked terminal statuses *before* checking whether the most recent event was an ambiguous one, so a group that was rescheduled and then genuinely thrown into doubt by a later "we may move this again, I'll confirm" message kept reporting `Rescheduled` — asserting a confidence the evidence no longer supported. Found via the manual read-through (`GROUP_006`, the internship orientation thread, ending on `DEMO_017`'s ambiguous mention). **Fix:** the ambiguous check now runs first, unconditionally — the most recent word on a subject gets the final say over status, consistent with how `priority.py` already treats `ambiguous_status` as a confidence penalty rather than a fact. Covered by 1 new regression test.

**What the audit did *not* find, which matters as much as what it did:** scanning all 490 standalone items for near-duplicate phrasing (e.g. two "Are you available for the technical interview...?" messages at different times) surfaced dozens of superficially similar candidates — and correctly grouped none of them, because none actually reference each other (no "moved to"/"is now due" language connecting them — they're independent synthetic messages that happen to share a template). That's the deterministic design working as intended, not a gap: grouping only two messages because they resemble each other in surface form is exactly the failure mode the assignment prohibits, and the audit is the evidence that it isn't happening.

---

## Assumptions and limitations

- Messages are expected to be in English.
- The fastembed ONNX model may misclassify ambiguous messages where a single sentence contains signals from multiple categories (e.g. `"I might prefer evening meetings now"` — preference vs. event).
- Person extraction relies on capitalisation; lowercase names in casual text are missed.
- Sensitive detection uses regex — novel secret formats not covered by the patterns will be missed.
- `extract.py`'s keyword taxonomy still misses some phrasings (e.g. a plain "The X has been moved to DATE" reschedule where X was never independently extracted as an item by any keyword match). `priority.py`'s `REF_` fallback catches most of these by linking off the update message itself, but a subject that is *never* mentioned in an update-style template and *never* hits an extraction keyword has no way to be tracked — this is a real gap in extraction coverage, not a priority-engine guess.
- Item ids are derived from the full `message_id` (e.g. `TASK_MSG_0042`, not a bare running number) specifically so that combining files with independent numbering (the L1 file, the L2 file, and the separate L2 demo file, which all restart their own numeric-looking suffixes) can never silently collide two unrelated items into one.
- Priority weights are hand-set constants (`priority.py`), not learned from labelled data — there is no ground-truth priority dataset to fit against. They are documented above so every decision is explainable and reproducible, not a black box.
- `high` accounts for ~54% of all priority decisions over the full L1+L2+demo run (308/566), and `overdue` is the single biggest driver — because `reference_dt` is pinned to the *last* message in the whole batch (2026-10-05), and most L1 tasks from September never received a follow-up or completion message anywhere in the dataset. Evaluated from that single end-of-batch snapshot, those tasks genuinely are overdue — this is the correct "what's the state right now" answer (and what Part 3's assistant needs), not an inflated score. It does mean the aggregate distribution reflects "old backlog with no resolution," not scoring being loose.
- Grouping only forms a group once a message either creates a formal item or matches one of `linking.py`'s regex templates — a subject that's only ever discussed in free-form prose with no template match and no shared item-creating keyword will never be grouped (the semantic fallback that could catch some of these was tested and deliberately disabled — see above). Over the real dataset this produces 32 groups covering the templated backlog, which is the dataset's actual structure; a less rigidly-templated real-world inbox would need a better meaning-based fallback than raw cosine similarity (named-entity extraction, most likely) before that gap could close safely.
- A group's `status` reflects the state as of the LAST status-changing event for that subject, which is not always the same as the state implied by the LAST message in the group — a trailing `status_check` ("still needs attention") after a `cancelled` message correctly leaves the group `Cancelled`, since asking about something doesn't change its status. Conversely, a `deadline_changed`/`rescheduled`/`conflicting_deadline` message arriving after a `completed`/`cancelled` claim is treated as more specific, more recent evidence and reopens the item (see Part 1's `reopened` signal) — this is intentional, not a stale read.

---

## AI-tool usage disclosure

- **L1:** GitHub Copilot (VS Code) was used to assist with code suggestions, debugging, and refactoring during development.
- **L2:** Claude Code was used to help design and implement the priority/linking engine (`linking.py`, `priority.py`), extend `extract.py`/`main.py`/`app.py`, and write the accompanying tests, working from the assignment brief and the existing L1 codebase.
- All code was reviewed, understood, and verified by the author before submission — including manually tracing the signal weights and linking logic against the actual dataset (see "How priority is calculated and updated" above).

