"""
Gradio demo for Hugging Face Spaces.
Supports two modes:
  1. Upload a CSV (messages.csv format) — processes it live
  2. Demo mode — runs on 8 built-in fabricated sample messages
"""
import os, json
from datetime import datetime
import pandas as pd
import gradio as gr

from classify import classify_message
from extract import extract_item
from sensitive import detect_sensitive
from priority import compute_priorities
from grouping import compute_groups, annotate_superseded
from privacy import route_all
from assistant import build_knowledge_base, answer_query


def _parse_ts(raw_ts) -> datetime:
    if pd.isna(raw_ts):
        return datetime.now()
    if isinstance(raw_ts, datetime):
        return raw_ts
    s = str(raw_ts)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return pd.to_datetime(s, dayfirst=False).to_pydatetime()
    except Exception:
        return datetime.now()

# ---------------------------------------------------------------------------
# Fabricated sample messages — safe to show publicly, NOT from the real dataset
# ---------------------------------------------------------------------------
SAMPLE_ROWS = [
    ("MSG_S01", "2026-09-01 09:00:00", "Arjun",  "Please submit the quarterly report by Friday. This is urgent."),
    ("MSG_S02", "2026-09-01 09:30:00", "Meera",  "Calendar update: team standup on 2026-09-05 at 10:00 in Room 3."),
    ("MSG_S03", "2026-09-01 10:00:00", "Kabir",  "My home address is 12 Park Street, Chennai. Please update records."),
    ("MSG_S04", "2026-09-01 10:30:00", "Priya",  "Flash sale tonight — 50% off all laptops. Use code SAVE50."),
    ("MSG_S05", "2026-09-01 11:00:00", "Ishaan", "The office Wi-Fi will be down for maintenance from 8 to 10 PM."),
    ("MSG_S06", "2026-09-01 11:30:00", "Arjun",  "Your OTP is 482910. It expires in 10 minutes. Do not share."),
    ("MSG_S07", "2026-09-01 12:00:00", "Meera",  "Can you review the privacy checklist before 2026-09-09?"),
    ("MSG_S08", "2026-09-01 12:30:00", "Kabir",  "Just a quick hello — hope your week is going well!"),
]


def _run_pipeline(df: pd.DataFrame):
    classifications, extracted, sensitive_findings, messages = [], [], [], []

    for _, row in df.iterrows():
        mid     = str(row.get("message_id", ""))
        message = str(row.get("message", ""))
        sender  = str(row.get("sender", ""))
        ts      = row.get("timestamp", "")
        msg_date = _parse_ts(ts)
        messages.append({"message_id": mid, "timestamp": msg_date, "sender": sender, "message": message})

        result, sens_hits = classify_message(mid, message)
        classifications.append(result)
        sensitive_findings.extend(sens_hits)

        item = extract_item(mid, message, msg_date, sender)
        if item:
            extracted.append(item)

    priorities = compute_priorities(messages, classifications, extracted, sensitive_findings)
    groups = compute_groups(messages, extracted)
    annotate_superseded(extracted, groups)
    routing = route_all(messages, sensitive_findings)
    # NOTE: the knowledge base is NOT built here. It re-derives a
    # min_group_size=1 grouping internally (one entry per item, not just
    # the 2+-message ones) -- for an uploaded CSV of any size that's a much
    # bigger object than anything this app used to hold. Building it here
    # and caching it in gr.State would keep it retained in server memory
    # for the whole browser session, on a 512MB host. Instead we cache
    # only these plain lists (no extra cost -- already computed for the
    # tables below) and rebuild the knowledge base fresh, on demand, only
    # when a question is actually asked -- see run_ask().
    bundle = {
        "messages": messages, "classifications": classifications, "extracted": extracted,
        "sensitive_findings": sensitive_findings, "priorities": priorities,
        "groups": groups, "routing": routing,
    }
    return classifications, extracted, sensitive_findings, priorities, groups, routing, bundle


def _cls_table(classifications):
    rows = [[c["message_id"], c["category"], c["confidence"], c["reason"][:80]]
            for c in classifications]
    return pd.DataFrame(rows, columns=["ID", "Category", "Conf", "Reason"])


def _ext_table(extracted):
    if not extracted:
        return pd.DataFrame(columns=["ID", "Type", "Sub-class", "Deadline", "Time", "Person", "Priority", "Title"])
    rows = [[i["source_message_id"], i["type"], i["sub_class"],
             i["deadline"], i["time"], i["person"], i["priority"], i["title"][:60]]
            for i in extracted]
    return pd.DataFrame(rows, columns=["ID", "Type", "Sub-class", "Deadline", "Time", "Person", "Priority", "Title"])


def _sens_table(findings):
    if not findings:
        return pd.DataFrame(columns=["ID", "Type", "Risk", "Recommended Action", "Masked Text"])
    rows = [[f["message_id"], f["sensitivity_type"], f["risk"],
             f["recommended_action"], f["masked_text"][:90]]
            for f in findings]
    return pd.DataFrame(rows, columns=["ID", "Type", "Risk", "Recommended Action", "Masked Text"])


def _pri_table(priorities):
    if not priorities:
        return pd.DataFrame(columns=["Message ID", "Item ID", "Priority", "Confidence", "Signals", "Reason"])
    rows = [[p["message_id"], p["item_id"], p["priority"], p["confidence"],
             ", ".join(p["signals"]), p["reason"]]
            for p in priorities]
    return pd.DataFrame(rows, columns=["Message ID", "Item ID", "Priority", "Confidence", "Signals", "Reason"])


def _route_table(routing):
    cols = ["Message ID", "Route", "Risk", "Reason"]
    if not routing:
        return pd.DataFrame(columns=cols)
    rows = [[r["message_id"], r["route"], r["risk"], r["reason"]] for r in routing]
    return pd.DataFrame(rows, columns=cols)


def _grp_table(groups):
    cols = ["Group ID", "Title", "Status", "Latest Deadline", "Confidence", "Related Messages", "Summary"]
    if not groups:
        return pd.DataFrame(columns=cols)
    rows = [[g["group_id"], g["title"][:50], g["status"], g["latest_deadline"], g["confidence"],
             ", ".join(g["related_message_ids"]), g["summary"]]
            for g in groups]
    return pd.DataFrame(rows, columns=cols)


def run_demo(_):
    df = pd.DataFrame(SAMPLE_ROWS, columns=["message_id", "timestamp", "sender", "message"])
    cls, ext, sens, pri, grp, routing, bundle = _run_pipeline(df)
    summary = (
        f"**Demo mode** — {len(cls)} fabricated sample messages processed\n\n"
        f"- Classified: {len(cls)}  |  Tasks/events extracted: {len(ext)}  |  "
        f"Sensitive findings: {len(sens)}  |  Priority decisions: {len(pri)}  |  Groups: {len(grp)}\n\n"
        f"> These 8 messages are independent fabricated examples, so no related-message "
        f"groups are expected here — try the Upload tab with a real multi-message CSV to see grouping in action.\n\n"
        f"> Ask a question in the **Ask the Assistant** tab now — it will answer using this batch."
    )
    return (summary, _cls_table(cls), _ext_table(ext), _sens_table(sens), _pri_table(pri),
            _grp_table(grp), _route_table(routing), bundle)


def run_upload(file):
    if file is None:
        return "Please upload a CSV file.", None, None, None, None, None, None, None
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return f"Could not read file: {e}", None, None, None, None, None, None, None

    required = {"message_id", "timestamp", "sender", "message"}
    missing = required - set(c.lower() for c in df.columns)
    if missing:
        return (f"Missing columns: {missing}. Required: message_id, timestamp, sender, message",
                None, None, None, None, None, None, None)

    df = df.sort_values("timestamp").reset_index(drop=True)
    cls, ext, sens, pri, grp, routing, bundle = _run_pipeline(df)

    summary = (
        f"**Processed {len(df)} messages**\n\n"
        f"- Classified: {len(cls)}  |  Tasks/events extracted: {len(ext)}  |  "
        f"Sensitive findings: {len(sens)}  |  Priority decisions: {len(pri)}  |  Groups: {len(grp)}\n\n"
        f"> Sensitive values are masked — raw values are never displayed.\n\n"
        f"> Ask a question in the **Ask the Assistant** tab now — it will answer using this batch."
    )
    return (summary, _cls_table(cls), _ext_table(ext), _sens_table(sens), _pri_table(pri),
            _grp_table(grp), _route_table(routing), bundle)


def run_ask(bundle, query):
    if bundle is None:
        return "Run the Demo or Upload tab first, then come back and ask a question.", None
    if not query or not query.strip():
        return "Type a question first.", None
    # Built fresh per question, not cached -- see the note in _run_pipeline().
    kb = build_knowledge_base(
        bundle["messages"], bundle["classifications"], bundle["extracted"],
        bundle["sensitive_findings"], bundle["priorities"], bundle["groups"], bundle["routing"],
    )
    ans = answer_query(kb, query)
    md = (
        f"**Answer:** {ans['answer']}\n\n"
        f"**Confidence:** {ans['confidence']}\n\n"
        f"**Reason:** {ans['reason']}"
    )
    detail = pd.DataFrame([{
        "supporting_message_ids": ", ".join(ans["supporting_message_ids"]),
        "related_item_ids": ", ".join(ans["related_item_ids"]),
        "group_id": ans["group_id"] or "",
        "relevance_scores": ", ".join(f"{s:.2f}" for s in ans["relevance_scores"]),
    }])
    return md, detail


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Message Intelligence Pipeline") as demo:
    gr.Markdown(
        "# Message Intelligence Pipeline\n"
        "Classifies messages · Extracts tasks/events · Detects & masks sensitive info\n\n"
        "**Fully local — no external API calls.**"
    )

    demo_bundle_state = gr.State(value=None)
    upload_bundle_state = gr.State(value=None)

    with gr.Tab("Demo (fabricated samples)"):
        gr.Markdown("Click **Run Demo** to process 8 built-in fabricated messages covering all 6 categories.")
        demo_btn = gr.Button("Run Demo", variant="primary")
        demo_summary = gr.Markdown()
        demo_cls   = gr.Dataframe(label="Part 1 — Classification")
        demo_ext   = gr.Dataframe(label="Part 2 — Extraction")
        demo_sens  = gr.Dataframe(label="Part 3 — Sensitive Findings")
        demo_pri   = gr.Dataframe(label="L2 Part 1 — Priority")
        demo_grp   = gr.Dataframe(label="L2 Part 2 — Related-Message Groups")
        demo_route = gr.Dataframe(label="L2 Part 3 — Privacy Routing")
        demo_btn.click(run_demo, inputs=demo_btn,
                        outputs=[demo_summary, demo_cls, demo_ext, demo_sens, demo_pri, demo_grp,
                                 demo_route, demo_bundle_state])

    with gr.Tab("Upload your CSV"):
        gr.Markdown(
            "Upload a CSV with columns: `message_id`, `timestamp`, `sender`, `message`.\n\n"
            "> Do not upload the real assignment dataset to a public Space."
        )
        file_input = gr.File(label="Upload messages.csv", file_types=[".csv"])
        upload_btn = gr.Button("Process", variant="primary")
        upload_summary = gr.Markdown()
        upload_cls   = gr.Dataframe(label="Part 1 — Classification")
        upload_ext   = gr.Dataframe(label="Part 2 — Extraction")
        upload_sens  = gr.Dataframe(label="Part 3 — Sensitive Findings")
        upload_pri   = gr.Dataframe(label="L2 Part 1 — Priority")
        upload_grp   = gr.Dataframe(label="L2 Part 2 — Related-Message Groups")
        upload_route = gr.Dataframe(label="L2 Part 3 — Privacy Routing")
        upload_btn.click(run_upload, inputs=file_input,
                          outputs=[upload_summary, upload_cls, upload_ext, upload_sens, upload_pri, upload_grp,
                                   upload_route, upload_bundle_state])

    with gr.Tab("Ask the Assistant"):
        gr.Markdown(
            "Run **Demo** or **Upload** first, then ask a question about that batch — "
            "e.g. *\"Which tasks are still pending?\"*, *\"What meetings were rescheduled?\"*, "
            "*\"Which messages require confirmation?\"*.\n\n"
            "> Answers only use retrieved evidence from this batch — if nothing matches confidently, "
            "the assistant says so instead of guessing."
        )
        which_batch = gr.Radio(["Demo batch", "Uploaded batch"], value="Demo batch", label="Answer using")
        query_box = gr.Textbox(label="Your question", placeholder="What tasks should I complete today?")
        ask_btn = gr.Button("Ask", variant="primary")
        ask_answer = gr.Markdown()
        ask_detail = gr.Dataframe(label="Evidence")

        def _run_ask(which, query, demo_bundle, upload_bundle):
            bundle = demo_bundle if which == "Demo batch" else upload_bundle
            return run_ask(bundle, query)

        ask_btn.click(_run_ask, inputs=[which_batch, query_box, demo_bundle_state, upload_bundle_state],
                      outputs=[ask_answer, ask_detail])

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), ssr_mode=False)
