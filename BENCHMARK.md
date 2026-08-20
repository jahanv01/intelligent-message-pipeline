# Benchmark Comparison Report

## What was optimized

The embedding runtime behind `classify.py`'s semantic classifier (used for message category and, since L2, `priority.py`'s semantic urgency signal and `assistant.py`'s subject search — all three reuse the same loaded model via `classify.embed_texts`). The change was made in this repo's own history, commit [`3cde402`](.) *"fix: replace torch+sentence-transformers with fastembed to fit 512MB Render free tier"* (2026-08-13), well before this L2 work started. Both versions load the **same model weights** (`all-MiniLM-L6-v2`) — this is a runtime/export-format swap (PyTorch inference → ONNX inference via `fastembed`), not a model change.

## How this was measured

Every number below was either measured directly on the testing machine used for this submission, or is a real artifact (a downloaded package file, an installed directory) inspected on that same machine on 2026-08-20 — none of it is copied from documentation or estimated.

**Testing device:** Intel Core Ultra 7 255U (14 logical CPUs), 15 GB RAM, WSL2/Ubuntu, Python 3.14.4. CPU-only throughout — no GPU used or available, matching the Render free-tier deployment target.

**Method for each row:**
- *Wheel sizes* — `pip download --no-deps torch sentence-transformers fastembed` into a scratch directory (download only, nothing installed) and inspected the resulting file sizes directly.
- *Installed footprint* — this dev `.venv` had an orphaned `torch==2.13.0+cpu` install left over from before the 2026-08-13 migration (`pip show torch` confirmed `Required-by: (none)` and a full `grep` of every `.py` file in the repo confirmed zero remaining imports of `torch`/`sentence_transformers`) — real, verifiable evidence of the "before" footprint sitting in the same environment as the "after" one. Measured with `du -sh`, then removed (`pip uninstall torch sentence-transformers sympy`) to get a clean "after" figure in the exact same venv, and the full test suite (90/90) re-run afterward to confirm nothing depended on it.
- *Cached model* — the actual ONNX model file `fastembed` downloads and caches on first run, measured with `du -sh`.
- *Throughput* — `classify_message()` timed directly over 200 real L1 messages (`time.perf_counter()`, one warm-up call excluded from the average, model already loaded — this measures steady-state inference, not cold start).
- *Full pipeline* — `main.py`'s own logged wall-clock time processing all 1104 messages (L1 + L2 + L2 demo batch) end-to-end: classification, extraction, sensitive detection, priority scoring, grouping, and privacy routing.

## Results

| Metric | Before (torch + sentence-transformers) | After (fastembed, ONNX) |
|---|---|---|
| `torch` wheel download size | 526.6 MB | — (not a dependency) |
| `sentence-transformers` wheel download size | 739.6 KB (pulls in `transformers`, `huggingface_hub`, and `torch` itself as further dependencies — the number above is only the top-level package) | — |
| `fastembed` wheel download size | — | 116.6 KB |
| Installed footprint of the embedding stack alone | `torch` alone: **757 MB** installed (`sentence-transformers` was never even installed in this venv — `torch` alone already exceeds the entire 512 MB Render free-tier RAM budget on disk, before a single tensor is loaded) | `fastembed` + `onnxruntime` + `tokenizers` + `huggingface_hub` + `hf_xet`: **~89 MB** installed |
| Whole `.venv` (app + all dependencies) | 1.3 GB (measured directly, before cleanup) | 450 MB (measured directly, after cleanup — same venv, same app code, both numbers real) |
| Cached model file (`all-MiniLM-L6-v2`) | 87 MB (same file either way — same weights) | 87 MB |
| Classification throughput | *(not re-measured — see Assumptions below)* | 14.2 ms/message (70.6 messages/sec), 200 real L1 messages, steady state |
| Full pipeline (1104 messages, all 6 pipeline stages) | *(not re-measured)* | 29.67 s wall clock |

## Result quality

Same model weights both ways, so classification behavior is not expected to differ meaningfully — this is a runtime swap, not a model swap. **This was not independently re-verified in this session**: doing so honestly would mean actually installing `torch`+`sentence-transformers` (526+ MB) and running the full 900-message L1 batch through the old code path, which wasn't done here — stated plainly rather than assumed. What *is* verified: fastembed's ONNX export is exporting the identical published `all-MiniLM-L6-v2` checkpoint (same model id passed to both `SentenceTransformer(...)` and `TextEmbedding(...)` — see the actual diff in commit `3cde402`), and the 31-test L1 suite (classification, extraction, sensitive detection) plus this L2 work's 59 additional tests all pass against the fastembed path, giving behavioral coverage even without a formal side-by-side accuracy diff.

## Assumptions and limitations of this benchmark

- The "before" installed-footprint number (757 MB for `torch` alone) is real and measured, but it's an *orphaned leftover* in this dev venv, not a fresh install — a brand-new `pip install torch sentence-transformers` would also pull in `transformers` and its own dependency tree on top of that 757 MB, meaning the true old footprint was almost certainly larger; this report only claims the piece it could actually verify.
- Throughput/pipeline timing has no "before" comparison point in this report, for the same honesty reason: re-running the full old stack wasn't done. The Render free-tier note already in the main README (`~10s` for the 8-message demo, `~30-60s` cold start) is the only prior first-hand timing data available for the old-vs-new UX, and predates this benchmark.
- All numbers are single-run measurements on one machine, not averaged over multiple runs — reported as directional evidence for a 512 MB-constrained deployment decision, not as a rigorously controlled experiment.
