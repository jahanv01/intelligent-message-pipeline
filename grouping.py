"""
Related-Message Grouping (L2 Part 2)
----------------------------------------
Identifies messages that refer to the same task/meeting/event/request, and
produces one group per subject with a title, combined summary, current
status, latest deadline, and confidence score.

Design: this does NOT re-implement subject matching. It calls the exact
same `linking.resolve_messages()` chronological resolver that priority.py
uses, then turns each SubjectRegistry entry (already carrying every
mention, in order, plus a full event history) into a group. Two engines
independently re-deriving "which messages are about the same thing" could
quietly disagree; one shared resolver can't.

"Message meaning" is satisfied primarily by the template layer itself: a
match requires the FULL distinguishing action phrase to literally appear
(e.g. "review the security audit report"), not one shared word -- this is
already a much stronger meaning-preserving signal than keyword overlap, and
is exactly what keeps two same-template-different-subject messages (e.g.
"confirm the interview slot" vs. "confirm the delivery slot") from merging.

An embedding-based semantic fallback (`_link_semantic_mentions`, DISABLED
by default -- `use_semantic_fallback=False`) was built and empirically
tested for messages that don't match any template at all (the spec's own
illustrative example, "Reminder -- the report is due tomorrow," wouldn't
match a fixed template). It was rejected: tested against held-out same-
subject and different-subject pairs, adversarial pairs sharing a template
but referring to DIFFERENT subjects scored 0.40-0.65, and 3 of 5 of those
scored HIGHER than genuine same-subject pairs (0.32-0.55). MiniLM's
sentence embedding is picking up phrasing/structure similarity more than
the specific entity being discussed here, and no threshold separates the
two distributions cleanly. Shipping it would produce exactly the false
grouping the assignment explicitly warns against, so it stays off by
default -- see README "Assumptions and limitations" for the numbers and
what a fix would need (named-entity extraction, not raw cosine similarity).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from classify import embed_texts
from extract import _clean_title
from linking import resolve_messages

MIN_GROUP_SIZE = 2  # a lone, never-followed-up message isn't a "grouping" --
                     # Part 3 can still reference it directly via its item_id

_STATUS_MAP_TERMINAL = {"completed": "Completed", "cancelled": "Cancelled", "rescheduled": "Rescheduled"}

_EVENT_PHRASES = {
    "status_check": "followed up on",
    "deadline_changed": "had its deadline changed",
    "conflicting_deadline": "flagged with a conflicting deadline",
    "rescheduled": "rescheduled",
    "completed": "marked completed",
    "cancelled": "marked cancelled",
    "ambiguous": "mentioned again with uncertain status",
    "mentioned": "referenced elsewhere in similar terms",
}

# Semantic-fallback linking: DISABLED by default (see module docstring --
# empirically rejected, no threshold separates same-subject from different-
# subject-same-template pairs). 0.62 is the value that was tested; it still
# lets through adversarial pairs at 0.65 while missing genuine matches at
# 0.55, so treat it as a documented dead end, not a tuned setting.
SEMANTIC_GROUP_THRESHOLD = 0.62
_MIN_MESSAGE_LEN = 12  # very short messages ("Yes." "Sure.") are never searched


def _status_for(state) -> str:
    internal = state.get("status", "pending")
    if internal in _STATUS_MAP_TERMINAL:
        return _STATUS_MAP_TERMINAL[internal]
    # internal == "pending"
    if state.get("last_update_type") == "ambiguous":
        return "Unclear"
    if len(state.get("mentions", [])) > 1:
        return "In progress"
    return "Pending"


def _title_for(item_id, state, items_by_id) -> str:
    item = items_by_id.get(item_id)
    if item:
        return item["title"]
    # REF_ entry -- no formal item, derive a title the same way extract.py
    # would have, from the message that first established the subject.
    desc = state.get("_origin_description", "")
    return _clean_title(desc) if desc else item_id


def _summarize(item_type: str, history: list) -> str:
    if not history:
        return "No activity recorded."
    rest = [h["event"] for h in history[1:]]
    if not rest:
        return "Created, with no follow-up activity recorded after it."
    counts = Counter(rest)
    seen_order = []
    for e in rest:
        if e not in seen_order:
            seen_order.append(e)
    clauses = []
    for e in seen_order:
        phrase = _EVENT_PHRASES.get(e, e)
        n = counts[e]
        clauses.append(f"{phrase} ({n}x)" if n > 1 else phrase)
    noun = "task" if item_type == "task" else "event" if item_type == "event" else "item"
    return f"This {noun} was created, then " + ", ".join(clauses) + "."


def _confidence_for(item_id: str, state: dict) -> float:
    base = 0.65 if item_id.startswith("REF_") else 0.8
    mentions = state.get("mentions", [])
    base += min(0.03 * max(len(mentions) - 1, 0), 0.15)
    history_events = [h["event"] for h in state.get("history", [])]
    if "mentioned" in history_events:
        base -= 0.10 * history_events.count("mentioned")
    if state.get("last_update_type") == "ambiguous":
        base -= 0.15
    if any(h["event"] == "conflicting_deadline" for h in state.get("history", [])):
        base -= 0.10
    return max(0.3, min(round(base, 2), 0.97))


def _link_semantic_mentions(messages, registry, resolutions):
    """For messages the deterministic pass left fully unresolved, try a
    meaning-based match against subjects already known BY THAT POINT IN
    TIME. Batches all embeddings in two calls (all unresolved messages, all
    candidate descriptions) rather than one call per message -- cheap even
    over the full dataset."""
    unresolved_idx = [i for i, r in enumerate(resolutions) if r["item_id"] is None
                       and len(messages[i]["message"]) >= _MIN_MESSAGE_LEN]
    item_ids = registry.all_item_ids()
    if not unresolved_idx or not item_ids:
        return

    msg_vecs = embed_texts([messages[i]["message"] for i in unresolved_idx])
    desc_vecs = embed_texts([registry.description_for(iid) for iid in item_ids])
    if msg_vecs is None or desc_vecs is None:
        return  # fastembed unavailable -- skip the fallback, never guess

    def _normalise(mat):
        n = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(n, 1e-10, None)

    msg_vecs = _normalise(msg_vecs)
    desc_vecs = _normalise(desc_vecs)
    sims = msg_vecs.dot(desc_vecs.T)  # (n_unresolved, n_items)

    item_positions = [registry.state_for(iid).get("first_seen_order", -1) for iid in item_ids]

    for row, msg_idx in enumerate(unresolved_idx):
        position = msg_idx  # resolutions is in the same order as messages
        row_sims = sims[row].copy()
        for col, seen_at in enumerate(item_positions):
            if seen_at >= position:  # backward-only, same rule as everything else
                row_sims[col] = -1.0
        best_col = int(np.argmax(row_sims))
        if row_sims[best_col] < SEMANTIC_GROUP_THRESHOLD:
            continue
        item_id = item_ids[best_col]
        mid = messages[msg_idx]["message_id"]
        registry.add_mention(item_id, mid, event="mentioned")
        resolutions[msg_idx]["item_id"] = item_id
        resolutions[msg_idx]["link_type"] = "semantic"


def compute_groups(messages, extracted_items, min_group_size: int = MIN_GROUP_SIZE, use_semantic_fallback: bool = False):
    """
    messages: chronological list of {message_id, timestamp, sender, message}
    extracted_items: list of dicts from extract_item
    Returns a list of group dicts:
        {group_id, title, related_message_ids, related_item_ids, summary,
         status, latest_deadline, confidence}
    """
    registry, resolutions = resolve_messages(messages, extracted_items)
    items_by_id = {i["item_id"]: i for i in extracted_items}

    # Stash the REF_ origin message text for title derivation (register()
    # already lowercases descriptions for containment search; keep the
    # original-case text separately, cheaply, only for REF_ entries).
    for res in resolutions:
        if res["item_id"] and res["item_id"].startswith("REF_") and res["link_type"] == "origin":
            state = registry.state_for(res["item_id"])
            msg = next(m for m in messages if m["message_id"] == res["message_id"])
            state["_origin_description"] = msg["message"]

    if use_semantic_fallback:
        _link_semantic_mentions(messages, registry, resolutions)

    groups = []
    seq = 0
    for item_id in registry.all_item_ids():
        state = registry.state_for(item_id)
        mentions = state.get("mentions", [])
        if len(mentions) < min_group_size:
            continue
        seq += 1
        item_type = state.get("type")
        groups.append({
            "group_id": f"GROUP_{seq:03d}",
            "title": _title_for(item_id, state, items_by_id),
            "related_message_ids": list(mentions),
            "related_item_ids": [item_id],
            "summary": _summarize(item_type, state.get("history", [])),
            "status": _status_for(state),
            "latest_deadline": state.get("deadline"),
            "confidence": _confidence_for(item_id, state),
        })

    return groups
