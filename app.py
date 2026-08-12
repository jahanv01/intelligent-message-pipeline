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
    classifications, extracted, sensitive_findings = [], [], []

    for _, row in df.iterrows():
        mid     = str(row.get("message_id", ""))
        message = str(row.get("message", ""))
        sender  = str(row.get("sender", ""))
        ts      = row.get("timestamp", "")

        result, sens_hits = classify_message(mid, message)
        classifications.append(result)
        sensitive_findings.extend(sens_hits)

        item = extract_item(mid, message, _parse_ts(ts), sender)
        if item:
            extracted.append(item)

    return classifications, extracted, sensitive_findings


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


def run_demo(_):
    df = pd.DataFrame(SAMPLE_ROWS, columns=["message_id", "timestamp", "sender", "message"])
    cls, ext, sens = _run_pipeline(df)
    summary = (
        f"**Demo mode** — {len(cls)} fabricated sample messages processed\n\n"
        f"- Classified: {len(cls)}  |  Tasks/events extracted: {len(ext)}  |  Sensitive findings: {len(sens)}"
    )
    return summary, _cls_table(cls), _ext_table(ext), _sens_table(sens)


def run_upload(file):
    if file is None:
        return "Please upload a CSV file.", None, None, None
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"Could not read file: {e}", None, None, None

    required = {"message_id", "timestamp", "sender", "message"}
    missing = required - set(c.lower() for c in df.columns)
    if missing:
        return f"Missing columns: {missing}. Required: message_id, timestamp, sender, message", None, None, None

    df = df.sort_values("timestamp").reset_index(drop=True)
    cls, ext, sens = _run_pipeline(df)

    summary = (
        f"**Processed {len(df)} messages**\n\n"
        f"- Classified: {len(cls)}  |  Tasks/events extracted: {len(ext)}  |  Sensitive findings: {len(sens)}\n\n"
        f"> Sensitive values are masked — raw values are never displayed."
    )
    return summary, _cls_table(cls), _ext_table(ext), _sens_table(sens)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Message Intelligence Pipeline") as demo:
    gr.Markdown(
        "# Message Intelligence Pipeline\n"
        "Classifies messages · Extracts tasks/events · Detects & masks sensitive info\n\n"
        "**Fully local — no external API calls.**"
    )

    with gr.Tab("Demo (fabricated samples)"):
        gr.Markdown("Click **Run Demo** to process 8 built-in fabricated messages covering all 6 categories.")
        demo_btn = gr.Button("Run Demo", variant="primary")
        demo_summary = gr.Markdown()
        demo_cls  = gr.Dataframe(label="Part 1 — Classification")
        demo_ext  = gr.Dataframe(label="Part 2 — Extraction")
        demo_sens = gr.Dataframe(label="Part 3 — Sensitive Findings")
        demo_btn.click(run_demo, inputs=demo_btn, outputs=[demo_summary, demo_cls, demo_ext, demo_sens])

    with gr.Tab("Upload your CSV"):
        gr.Markdown(
            "Upload a CSV with columns: `message_id`, `timestamp`, `sender`, `message`.\n\n"
            "> Do not upload the real assignment dataset to a public Space."
        )
        file_input = gr.File(label="Upload messages.csv", file_types=[".csv"])
        upload_btn = gr.Button("Process", variant="primary")
        upload_summary = gr.Markdown()
        upload_cls  = gr.Dataframe(label="Part 1 — Classification")
        upload_ext  = gr.Dataframe(label="Part 2 — Extraction")
        upload_sens = gr.Dataframe(label="Part 3 — Sensitive Findings")
        upload_btn.click(run_upload, inputs=file_input, outputs=[upload_summary, upload_cls, upload_ext, upload_sens])

if __name__ == "__main__":
    demo.launch()
