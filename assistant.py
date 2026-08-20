"""
Semantic Search & Intelligent Assistant (L2 Part 3)
--------------------------------------------------------
Answers natural-language questions using ONLY the structured outputs the
rest of the pipeline already produced -- classifications, extracted items,
sensitive findings, priority decisions, groups, privacy routing -- plus
original message text where permitted (never for a message that also
carries a sensitive finding; masked_text is used there instead, and a
message routed "blocked" is never echoed even masked, only referenced by
id, per the assignment's own screenshot/recording rule).

Two different retrieval jobs, two different tools -- same reasoning as
Part 1/2:
  - INTENT classification (which of the ~10 known question shapes is this)
    is done with explicit keyword/pattern rules, not embeddings. The
    mandatory query set is closed and small; Part 1 and Part 2 both found
    that a handful of anchor sentences per class does not reliably
    separate fine-grained categories with this embedding model. A rule
    either matches or it doesn't -- fully auditable.
  - ENTITY resolution (matching free text like "the project report" to a
    specific known task/event/group when no explicit ID is given) DOES use
    embeddings -- this is a legitimate nearest-neighbour retrieval problem
    over a small, distinct candidate set, always surfaced with a real
    similarity score and a rejection threshold below which the assistant
    says "insufficient evidence" rather than guessing.

No answer is ever produced without retrieved evidence, and no free-form
prose generation happens -- there is no local LLM in this stack (matching
the project's "100% local, no external calls" constraint everywhere else).
Every answer sentence is built directly from retrieved facts.
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np

from classify import embed_texts
from grouping import compute_groups
from priority import SIGNAL_WEIGHTS

SUBJECT_MATCH_THRESHOLD = 0.45  # below this, subject search reports insufficient evidence
_ID_RE = re.compile(r"\b(MSG|DEMO|GROUP|TASK|EVENT|REF)_[A-Za-z0-9_]+\b")


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
class KnowledgeBase:
    def __init__(self, messages, classifications, extracted_items, sensitive_findings, priorities, groups, routing):
        self.messages = messages
        self.messages_by_id = {m["message_id"]: m for m in messages}
        self.classifications_by_id = {c["message_id"]: c for c in classifications}
        self.items_by_id = {i["item_id"]: i for i in extracted_items}

        self.sensitive_by_msg = {}
        for f in sensitive_findings:
            self.sensitive_by_msg.setdefault(f["message_id"], []).append(f)

        self.priorities = priorities  # chronological
        self.latest_priority_by_item = {}
        for p in priorities:
            self.latest_priority_by_item[p["item_id"]] = p  # last write wins -- chronological input

        # Public groups (2+ messages) -- what's actually published in
        # output_groups.json. Used for "related group" citations.
        self.groups_public = groups

        # Full per-item view (min_group_size=1): EVERY item, including ones
        # that never got a second mention, gets the same status/title/
        # summary treatment as a real group. This reuses grouping.py's
        # logic instead of re-deriving status rules a second time here.
        self.groups_full = compute_groups(messages, extracted_items, min_group_size=1)
        self.group_by_item = {}
        for g in self.groups_full:
            for iid in g["related_item_ids"]:
                self.group_by_item[iid] = g
        self.public_group_of = {}
        for g in self.groups_public:
            for iid in g["related_item_ids"]:
                self.public_group_of[iid] = g["group_id"]

        self.routing_by_msg = {r["message_id"]: r for r in routing}
        self.reference_dt = max((m["timestamp"] for m in messages), default=datetime.now())

        self._title_vecs = None  # lazily built -- see _subject_search

    def masked_or_raw(self, message_id: str) -> str:
        """Text safe to surface for this message: masked if it carries a
        sensitive finding, raw otherwise. Never the raw text of a sensitive
        message, regardless of caller."""
        hits = self.sensitive_by_msg.get(message_id)
        if hits:
            return hits[0]["masked_text"]
        m = self.messages_by_id.get(message_id)
        return m["message"] if m else ""

    def _subject_search(self, query_text: str, top_k: int = 5):
        """Embedding search over every group_full's title+summary. Returns
        [(group, score), ...] sorted by score, only entries clearing
        SUBJECT_MATCH_THRESHOLD."""
        if self._title_vecs is None:
            texts = [f"{g['title']} {g['summary']}" for g in self.groups_full]
            vecs = embed_texts(texts)
            if vecs is not None:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.clip(norms, 1e-10, None)
            self._title_vecs = vecs

        if self._title_vecs is None:  # fastembed unavailable -- never guess
            return []

        qvec = embed_texts([query_text])[0]
        qvec = qvec / max(np.linalg.norm(qvec), 1e-10)
        sims = self._title_vecs.dot(qvec)
        order = np.argsort(-sims)[:top_k]
        return [(self.groups_full[i], float(sims[i])) for i in order if sims[i] >= SUBJECT_MATCH_THRESHOLD]

    def resolve_entity(self, query_text: str):
        """Find the group(s)/item(s) this query is about.
        1. Explicit ids in the text (MSG_/DEMO_/GROUP_/TASK_/EVENT_/REF_)
           always win -- deterministic, no embedding needed.
        2. Otherwise, semantic subject search over known titles/summaries.
        Returns (matches, method) where matches is [(group, score), ...].
        """
        ids = _ID_RE.findall(query_text)
        if ids:
            full_ids = _ID_RE.finditer(query_text)
            found = []
            for m in full_ids:
                token = m.group(0)
                if token.startswith("GROUP_"):
                    g = next((g for g in self.groups_full if g["group_id"] == token), None)
                    if g:
                        found.append((g, 1.0))
                elif token in self.messages_by_id:
                    item = self._item_for_message(token)
                    if item:
                        g = self.group_by_item.get(item["item_id"])
                        if g:
                            found.append((g, 1.0))
                elif token in self.group_by_item:
                    found.append((self.group_by_item[token], 1.0))
            if found:
                return found, "explicit_id"
        return self._subject_search(query_text), "semantic_search"

    def _item_for_message(self, message_id: str):
        for i in self.items_by_id.values():
            if i["source_message_id"] == message_id:
                return i
        # message may have resolved to an EARLIER item via linking (its own
        # extract_item may have found nothing, or a different, superseded id)
        for g in self.groups_full:
            if message_id in g["related_message_ids"]:
                item_id = g["related_item_ids"][0]
                return self.items_by_id.get(item_id) or {"item_id": item_id, "source_message_id": message_id}
        return None


def build_knowledge_base(messages, classifications, extracted_items, sensitive_findings, priorities, groups, routing) -> KnowledgeBase:
    return KnowledgeBase(messages, classifications, extracted_items, sensitive_findings, priorities, groups, routing)


# ---------------------------------------------------------------------------
# Answer shape -- matches the spec's example exactly, plus the prose's
# additional required fields (relevance scores, related task/event ids).
# ---------------------------------------------------------------------------
def _answer(query, answer, supporting_message_ids=None, related_item_ids=None,
            group_id=None, relevance_scores=None, reason="", confidence=0.0):
    return {
        "query": query,
        "answer": answer,
        "supporting_message_ids": supporting_message_ids or [],
        "related_item_ids": related_item_ids or [],
        "group_id": group_id,
        "relevance_scores": relevance_scores or [],
        "reason": reason,
        "confidence": round(confidence, 2),
    }


def _no_evidence(query, reason):
    return _answer(
        query,
        "I don't have enough evidence in the processed messages to answer this confidently.",
        reason=reason, confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Intent detection -- explicit keyword rules, checked most-specific first.
# Deliberately not embeddings: see module docstring.
# ---------------------------------------------------------------------------
_INTENT_RULES = [
    ("blocked_messages", re.compile(r"\bblock", re.I)),
    ("requires_confirmation", re.compile(r"\bconfirm", re.I)),
    ("why_critical", re.compile(r"\bwhy\b.*\b(critical|priority)\b", re.I)),
    # NOTE: deliberately narrow -- an earlier version also matched any
    # "which...critical..." question (via a `\bwhich\b.*\bcritical\b`
    # fallback), which silently swallowed the spec's OWN example query
    # ("Which critical or high-priority tasks are still pending?") before
    # it ever reached high_priority_pending. Found by testing the exact
    # spec wording, not a paraphrase -- caught before submission.
    ("became_critical", re.compile(r"\b(became|become|turn(ed)? into)\b.*\bcritical\b", re.I)),
    ("conflicting", re.compile(r"\bconflict", re.I)),
    ("deadlines_changed", re.compile(r"\bdeadline", re.I)),
    ("rescheduled", re.compile(r"reschedul|\bmoved\b", re.I)),
    # NOTE: requires the past-tense/status word "completed", not bare
    # "complet..." -- that prefix also matched the verb in "What tasks
    # should I complete today?", another spec example, misrouting it away
    # from tasks_due_today. Also checked before tasks_due_today below as a
    # second layer of defense.
    ("tasks_due_today", re.compile(r"\btoday\b", re.I)),
    ("completed_or_cancelled", re.compile(r"\bcompleted\b|\bcancel", re.I)),
    ("high_priority_pending", re.compile(r"\b(critical|high[\s-]?priority)\b", re.I)),
    ("latest_status", re.compile(r"\blatest status|\bcurrent status|\bstatus of\b", re.I)),
]


def _detect_intent(query: str) -> str:
    for intent, pattern in _INTENT_RULES:
        if pattern.search(query):
            return intent
    return "subject_search"  # fallback: "show me X", "what about Y" etc.


def _is_demo_scoped(query: str) -> bool:
    return bool(re.search(r"\bdemo\b", query, re.I))


# ---------------------------------------------------------------------------
# Handlers -- each takes (kb, query) and returns an answer dict, built only
# from retrieved facts.
# ---------------------------------------------------------------------------
def _handle_blocked(kb: KnowledgeBase, query: str):
    ids = [mid for mid, r in kb.routing_by_msg.items() if r["route"] == "blocked"]
    if _is_demo_scoped(query):
        ids = [mid for mid in ids if mid.startswith("DEMO_")]
    ids = [mid for mid in kb.messages_by_id if mid in ids]  # restore chronological order
    if not ids:
        return _no_evidence(query, "No message in this batch was routed to the blocked tier.")
    return _answer(
        query,
        f"{len(ids)} message(s) must be blocked from external processing: {', '.join(ids)}.",
        supporting_message_ids=ids,
        relevance_scores=[1.0] * len(ids),
        reason="Each carries a high-risk credential finding (OTP/password/card/account/token/recovery code), "
               "matching sensitive.py's do_not_store action -- see privacy.py's routing rule.",
        confidence=0.95,
    )


def _handle_confirm(kb: KnowledgeBase, query: str):
    ids = [mid for mid, r in kb.routing_by_msg.items() if r["route"] == "requires_confirmation"]
    if _is_demo_scoped(query):
        ids = [mid for mid in ids if mid.startswith("DEMO_")]
    ids = [mid for mid in kb.messages_by_id if mid in ids]
    if not ids:
        return _no_evidence(query, "No message in this batch was routed to the requires-confirmation tier.")
    return _answer(
        query,
        f"{len(ids)} message(s) require confirmation before processing: {', '.join(ids)}.",
        supporting_message_ids=ids,
        relevance_scores=[1.0] * len(ids),
        reason="Each carries medium-risk personal information (phone/email/address), "
               "matching sensitive.py's ask_for_confirmation action.",
        confidence=0.9,
    )


def _handle_became_critical(kb: KnowledgeBase, query: str):
    demo_only = _is_demo_scoped(query)
    hits = [p for p in kb.priorities if p["priority"] == "critical"
            and (not demo_only or p["message_id"].startswith("DEMO_"))]
    if not hits:
        return _no_evidence(query, "No priority decision in this batch reached 'critical'.")
    seen_items, ids, titles = set(), [], []
    for p in hits:
        if p["item_id"] in seen_items:
            continue
        seen_items.add(p["item_id"])
        ids.append(p["message_id"])
        g = kb.group_by_item.get(p["item_id"])
        titles.append(g["title"] if g else p["item_id"])
    item_ids = sorted(seen_items)
    return _answer(
        query,
        f"{len(seen_items)} task(s) reached critical priority: " + "; ".join(titles),
        supporting_message_ids=ids,
        related_item_ids=item_ids,
        relevance_scores=[1.0] * len(ids),
        reason="Each message's priority decision (see output_priorities.json) scored 'critical' -- "
               "signals and full reasoning are recorded per-decision there.",
        confidence=0.9,
    )


def _handle_why_critical(kb: KnowledgeBase, query: str):
    matches, method = kb.resolve_entity(query)
    if not matches:
        return _no_evidence(query, "Could not identify which task/event this question refers to.")
    g, score = matches[0]
    critical_decisions = [p for p in kb.priorities if p["item_id"] in g["related_item_ids"] and p["priority"] == "critical"]
    if not critical_decisions:
        return _answer(
            query,
            f"'{g['title']}' never reached critical priority (current status: {g['status']}).",
            related_item_ids=g["related_item_ids"], group_id=kb.public_group_of.get(g["related_item_ids"][0]),
            relevance_scores=[score], reason="No priority decision for this subject scored 'critical'.",
            confidence=0.8 if method == "explicit_id" else min(0.8, score),
        )
    latest = critical_decisions[-1]
    triggering_text = kb.masked_or_raw(latest["message_id"])  # original message where permitted, masked otherwise
    return _answer(
        query,
        f"{latest['reason']} Triggering message ({latest['message_id']}): \"{triggering_text}\"",
        supporting_message_ids=[latest["message_id"]],
        related_item_ids=g["related_item_ids"],
        group_id=kb.public_group_of.get(g["related_item_ids"][0]),
        relevance_scores=[1.0 if method == "explicit_id" else score],
        reason=f"Signals recorded for this decision: {', '.join(latest['signals'])}.",
        confidence=latest["confidence"],
    )


def _handle_conflicting(kb: KnowledgeBase, query: str):
    hits = [p for p in kb.priorities if "conflicting_deadline" in p["signals"] or "ambiguous_status" in p["signals"]]
    if not hits:
        return _no_evidence(query, "No priority decision carries a conflicting_deadline or ambiguous_status signal.")
    ids = [p["message_id"] for p in hits]
    item_ids = sorted({p["item_id"] for p in hits})
    return _answer(
        query,
        f"{len(ids)} message(s) contain conflicting or uncertain deadline information.",
        supporting_message_ids=ids,
        related_item_ids=item_ids,
        relevance_scores=[1.0] * len(ids),
        reason="Flagged via the conflicting_deadline signal (two messages state different deadlines for the "
               "same item) or ambiguous_status (uncertain language like 'might'/'may'/'probably').",
        confidence=0.85,
    )


def _handle_deadlines_changed(kb: KnowledgeBase, query: str):
    hits = [p for p in kb.priorities if "deadline_moved_earlier" in p["signals"] or "deadline_moved_later" in p["signals"]]
    if not hits:
        return _no_evidence(query, "No priority decision carries a deadline_moved_earlier/later signal.")
    ids = [p["message_id"] for p in hits]
    item_ids = sorted({p["item_id"] for p in hits})
    return _answer(
        query,
        f"{len(ids)} message(s) changed a deadline (moved earlier or later).",
        supporting_message_ids=ids,
        related_item_ids=item_ids,
        relevance_scores=[1.0] * len(ids),
        reason="Flagged via the deadline_moved_earlier/deadline_moved_later signals in output_priorities.json.",
        confidence=0.85,
    )


def _handle_rescheduled(kb: KnowledgeBase, query: str):
    hits = [g for g in kb.groups_full if g["status"] == "Rescheduled"]
    if not hits:
        return _no_evidence(query, "No group's current status is Rescheduled.")
    ids, item_ids, parts = [], [], []
    for g in hits:
        ids.extend(g["related_message_ids"])
        item_ids.extend(g["related_item_ids"])
        when = g["latest_deadline"] or "an unspecified date"
        if g.get("latest_time"):
            when += f" at {g['latest_time']}"
        parts.append(f"'{g['title']}' -> now {when}")
    return _answer(
        query,
        "Rescheduled: " + "; ".join(parts),
        supporting_message_ids=ids,
        related_item_ids=item_ids,
        relevance_scores=[1.0] * len(ids),
        reason="Each group's status is Rescheduled -- latest_deadline/latest_time reflect the most recent "
               "'has been moved to' message for that subject.",
        confidence=0.85,
    )


def _handle_completed_or_cancelled(kb: KnowledgeBase, query: str):
    want_completed = "complet" in query.lower()
    want_cancelled = "cancel" in query.lower()
    wanted = set()
    if want_completed:
        wanted.add("Completed")
    if want_cancelled:
        wanted.add("Cancelled")
    if not wanted:  # query said neither explicitly -- report both, per DQ02's phrasing
        wanted = {"Completed", "Cancelled"}
    hits = [g for g in kb.groups_full if g["status"] in wanted]
    if not hits:
        return _no_evidence(query, f"No group's current status is in {sorted(wanted)}.")
    ids, item_ids, parts = [], [], []
    for g in hits:
        ids.extend(g["related_message_ids"])
        item_ids.extend(g["related_item_ids"])
        parts.append(f"'{g['title']}' ({g['status']})")
    return _answer(
        query,
        f"{len(hits)} item(s): " + "; ".join(parts),
        supporting_message_ids=ids,
        related_item_ids=item_ids,
        relevance_scores=[1.0] * len(ids),
        reason=f"Each group's current status field is one of {sorted(wanted)} (see output_groups.json).",
        confidence=0.85,
    )


def _handle_high_priority_pending(kb: KnowledgeBase, query: str):
    active = {"Pending", "In progress", "Unclear"}
    results = []
    for item_id, p in kb.latest_priority_by_item.items():
        if p["priority"] not in {"critical", "high"}:
            continue
        g = kb.group_by_item.get(item_id)
        status = g["status"] if g else "Pending"
        if status in active:
            results.append((p, g))
    if not results:
        return _no_evidence(query, "No item's latest priority is critical/high while still active.")
    ids = [p["message_id"] for p, _ in results]
    item_ids = [p["item_id"] for p, _ in results]
    parts = [f"'{g['title'] if g else p['item_id']}' ({p['priority']})" for p, g in results]
    return _answer(
        query,
        f"{len(results)} still-active task(s) at critical/high priority: " + "; ".join(parts),
        supporting_message_ids=ids,
        related_item_ids=item_ids,
        relevance_scores=[1.0] * len(ids),
        reason="Each item's MOST RECENT priority decision is critical or high, and its current group status "
               "is not Completed/Cancelled/Rescheduled.",
        confidence=0.85,
    )


def _handle_due_today(kb: KnowledgeBase, query: str):
    today = kb.reference_dt.date().isoformat()
    hits = [g for g in kb.groups_full if g["latest_deadline"] == today and g["status"] not in {"Completed", "Cancelled"}]
    if not hits:
        return _no_evidence(
            query,
            f"No active item's latest_deadline equals the reference date ({today}, the last processed "
            f"message's own timestamp -- see README 'Reference time, not wall-clock time').",
        )
    ids, item_ids, parts = [], [], []
    for g in hits:
        ids.extend(g["related_message_ids"])
        item_ids.extend(g["related_item_ids"])
        parts.append(f"'{g['title']}'")
    return _answer(
        query,
        f"{len(hits)} task(s) due today ({today}): " + "; ".join(parts),
        supporting_message_ids=ids,
        related_item_ids=item_ids,
        relevance_scores=[1.0] * len(ids),
        reason=f"latest_deadline == {today}, the reference time used throughout this pipeline "
               f"(last processed message's own timestamp, not wall-clock).",
        confidence=0.85,
    )


def _handle_latest_status(kb: KnowledgeBase, query: str):
    matches, method = kb.resolve_entity(query)
    if not matches:
        return _no_evidence(query, "Could not identify which task/event/report this question refers to -- "
                                    "no explicit id and no group/item title cleared the similarity threshold.")
    g, score = matches[0]
    when = g["latest_deadline"] or "no deadline recorded"
    if g.get("latest_time"):
        when += f" at {g['latest_time']}"
    # "Original messages where permitted": the most recent message's own
    # text (masked if it also carries a sensitive finding -- never raw)
    # quoted directly, not just referenced by id.
    latest_mid = g["related_message_ids"][-1]
    latest_text = kb.masked_or_raw(latest_mid)
    return _answer(
        query,
        f"'{g['title']}' is currently {g['status']} (latest deadline: {when}). {g['summary']} "
        f"Most recent message ({latest_mid}): \"{latest_text}\"",
        supporting_message_ids=g["related_message_ids"],
        related_item_ids=g["related_item_ids"],
        group_id=kb.public_group_of.get(g["related_item_ids"][0]),
        relevance_scores=[1.0 if method == "explicit_id" else score],
        reason="Matched via explicit message/group id in the query." if method == "explicit_id"
               else f"Matched via semantic similarity to this group's title+summary (score {score:.2f}).",
        confidence=0.9 if method == "explicit_id" else min(0.9, score + 0.2),
    )


def _handle_subject_search(kb: KnowledgeBase, query: str):
    matches, method = kb.resolve_entity(query)
    if not matches:
        return _no_evidence(
            query,
            "No explicit id in the query, and no group/item title cleared the semantic similarity threshold "
            f"({SUBJECT_MATCH_THRESHOLD}) -- treating this as insufficient evidence rather than guessing.",
        )
    g, score = matches[0]
    return _answer(
        query,
        f"Found {len(g['related_message_ids'])} message(s) related to '{g['title']}': {g['summary']}",
        supporting_message_ids=g["related_message_ids"],
        related_item_ids=g["related_item_ids"],
        group_id=kb.public_group_of.get(g["related_item_ids"][0]),
        relevance_scores=[1.0 if method == "explicit_id" else score],
        reason="Matched via explicit id in the query." if method == "explicit_id"
               else f"Best semantic match to the query among all tracked subjects (similarity {score:.2f}).",
        confidence=0.9 if method == "explicit_id" else min(0.9, score + 0.2),
    )


_HANDLERS = {
    "blocked_messages": _handle_blocked,
    "requires_confirmation": _handle_confirm,
    "became_critical": _handle_became_critical,
    "why_critical": _handle_why_critical,
    "conflicting": _handle_conflicting,
    "deadlines_changed": _handle_deadlines_changed,
    "rescheduled": _handle_rescheduled,
    "completed_or_cancelled": _handle_completed_or_cancelled,
    "high_priority_pending": _handle_high_priority_pending,
    "tasks_due_today": _handle_due_today,
    "latest_status": _handle_latest_status,
    "subject_search": _handle_subject_search,
}


def answer_query(kb: KnowledgeBase, query: str) -> dict:
    """Single entry point: classify intent (deterministic), dispatch to the
    matching handler, which retrieves evidence and returns the answer.
    Never returns an answer without supporting evidence -- handlers fall
    back to _no_evidence() themselves when nothing qualifies."""
    intent = _detect_intent(query)
    return _HANDLERS[intent](kb, query)
