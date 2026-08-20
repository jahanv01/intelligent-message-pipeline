"""
Message Linking
------------------
Shared primitive for L2 Part 1 (priority.py) and, later, L2 Part 2
(related-message grouping): given a message, decide (a) does it describe
a *status change* to an already-known task/event, and if so (b) which
one.

Design: the L2/L1 datasets are template-generated. Status-change messages
("Update: X has been completed successfully.", "The deadline to X is now
DATE...", "You can cancel X; it is no longer required.") always restate
the task/event's own action phrase verbatim as X. So instead of trying to
canonicalise both sides into a symmetric key (fragile -- the *original*
message that created the item can be phrased however a human phrased it),
we do a one-sided match: pull the clean phrase X out of the update
message with a regex template, then find the EARLIEST previously-seen
item whose own description/title CONTAINS X as a substring. That is a
reliable join because X is, by construction, copied from the original.

Never invents a link: if no template matches, or the matched phrase is
too short/generic to search safely, or no earlier item contains it,
callers get None back and must not guess.
"""
from __future__ import annotations

import re
from datetime import datetime
from extract import _resolve_date, _resolve_time

# Leading noise that can stack in front of any template ("Follow-up:",
# "Additional update:") -- stripped repeatedly before template matching.
_LEADING_NOISE = re.compile(r"^(?:follow-up|additional update)\s*:\s*", re.IGNORECASE)


def _strip_leading_noise(text: str) -> str:
    while True:
        new = _LEADING_NOISE.sub("", text, count=1).strip()
        if new == text:
            return text
        text = new


# Each template: (update_type, compiled regex). Regex must expose named
# groups: "subject" (required) and optionally "when" (raw date/time text,
# resolved later via extract.py's own date/time parsers -- never
# reimplemented here).
_TEMPLATES = [
    ("completed", re.compile(
        r"^(?:update|confirmed):?\s*(?P<subject>.+?)\s+has been completed successfully\.?\s*$", re.I)),
    ("completed", re.compile(
        r"^confirmed:\s*(?P<subject>.+?)\s+has been completed\.?\s*$", re.I)),
    ("cancelled", re.compile(
        r"^you can cancel\s+(?P<subject>.+?);\s*it is no longer required\.?\s*$", re.I)),
    ("cancelled", re.compile(
        r"^cancel\s+(?P<subject>.+?);\s*it is no longer needed\.?\s*$", re.I)),
    ("cancelled", re.compile(
        r"^the\s+(?P<subject>.+?)\s+has been cancelled\.?\s*$", re.I)),
    ("deadline_changed", re.compile(
        r"^the deadline to\s+(?P<subject>.+?)\s+is now\s+(?P<when>.+?)"
        r"(?:,\s*earlier than previously planned)?\.\s*"
        r"(?:treat this as urgent\.?|this is urgent\.?)?\s*$", re.I)),
    ("deadline_changed", re.compile(
        r"^the deadline for\s+(?P<subject>.+?)\s+has been extended to\s+(?P<when>.+?)\.\s*$", re.I)),
    ("conflicting_deadline", re.compile(
        r"^please note that\s+(?P<subject>.+?)\s+is due on\s+(?P<when>.+?),\s*"
        r"although the earlier message listed another date\.?\s*$", re.I)),
    ("conflicting_deadline", re.compile(
        r"^one message says .+?,\s*but the latest instruction says\s+"
        r"(?P<subject>.+?)\s+is due on\s+(?P<when>.+?)\.\s*$", re.I)),
    ("rescheduled", re.compile(
        r"^the\s+(?P<subject>.+?)\s+has (?:been )?moved to\s+(?P<when>.+?)\.\s*"
        r"(?:please use the new schedule\.?)?\s*$", re.I)),
    ("rescheduled", re.compile(
        r"^the date for\s+(?P<subject>.+?)\s+stays the same,\s*but the time is now\s+"
        r"(?P<when>.+?)\.\s*$", re.I)),
    ("status_check", re.compile(
        r"^can you share an update on\s+(?P<subject>.+?)\?\s*$", re.I)),
    ("status_check", re.compile(
        r"^following up on\s+(?P<subject>.+?);\s*is it in progress\?\s*$", re.I)),
    ("status_check", re.compile(
        r"^please confirm whether you started to\s+(?P<subject>.+?)\.?\s*$", re.I)),
    ("status_check", re.compile(
        r"^any progress on the item concerning\s+(?P<subject>.+?)\?\s*$", re.I)),
    ("status_check", re.compile(
        r"^please check the latest status of\s+(?P<subject>.+?)\.?\s*$", re.I)),
    ("status_check", re.compile(
        r"^the work we discussed about\s+(?P<subject>.+?)\s+still needs attention\.?\s*$", re.I)),
    ("status_check", re.compile(
        r"^has the\s+(?P<subject>.+?)\s+item been handled yet\??\s*$", re.I)),
    ("status_check", re.compile(
        r"^i am referring to our earlier request about\s+(?P<subject>.+?)\.?\s*$", re.I)),
    ("status_check", re.compile(
        r"^this is another status request about\s+(?P<subject>.+?),\s*not a new task\.?\s*$", re.I)),
    ("ambiguous", re.compile(
        r"^(?P<subject>.+?)\s+might already be (?:done|finished)\b.*$", re.I)),
    ("ambiguous", re.compile(
        r"^we may move\s+(?P<subject>.+?);.*$", re.I)),
    ("ambiguous", re.compile(
        r"^the deadline could be .+$", re.I)),  # no reliable subject -- left unresolved by design
    ("ambiguous", re.compile(
        r"^the deadline may be .+$", re.I)),  # e.g. "...may be Monday, or it may be Wednesday..."
    ("ambiguous", re.compile(
        r"^.+ said someone probably handled (?P<subject>the task)\.?\s*$", re.I)),
    ("ambiguous", re.compile(
        r"^this may no longer be urgent\.?\s*$", re.I)),  # no reliable subject
]

# Explicit urgency language on the update message itself.
_URGENT_RE = re.compile(r"\b(urgent|asap|immediately|treat this as urgent|this is urgent)\b", re.I)

_MIN_SUBJECT_LEN = 6  # shorter/generic phrases are not searched -- avoids false links


def normalize_subject(phrase: str) -> str:
    """lowercase, strip leading article, strip trailing punctuation/whitespace."""
    p = phrase.strip().lower()
    p = re.sub(r"^(the|a|an)\s+", "", p)
    p = re.sub(r"[.,;:!?]+$", "", p).strip()
    return p


def detect_status_update(message: str) -> dict | None:
    """Match `message` against known status-change/status-check templates.

    Returns None if nothing matches. Otherwise returns:
        {update_type, subject_raw, subject_norm, when_raw, urgent}
    `subject_norm` is None if the matched phrase is too short/generic to
    safely search for (caller must not attempt a link in that case).
    """
    text = _strip_leading_noise(message.strip())
    for update_type, regex in _TEMPLATES:
        m = regex.match(text)
        if not m:
            continue
        groups = m.groupdict()
        subject_raw = groups.get("subject")
        when_raw = groups.get("when")
        subject_norm = None
        if subject_raw:
            norm = normalize_subject(subject_raw)
            if len(norm) >= _MIN_SUBJECT_LEN:
                subject_norm = norm
        return {
            "update_type": update_type,
            "subject_raw": subject_raw,
            "subject_norm": subject_norm,
            "when_raw": when_raw,
            "urgent": bool(_URGENT_RE.search(message)),
        }
    return None


def resolve_when(when_raw: str, msg_date: datetime):
    """Resolve a captured date/time phrase using extract.py's own explicit-only
    parsers (never guesses) -- returns (deadline_iso_or_None, time_or_None)."""
    if not when_raw:
        return None, None
    return _resolve_date(when_raw, msg_date), _resolve_time(when_raw)


class SubjectRegistry:
    """Chronological registry of items, searchable by substring containment.

    `find()` always returns the EARLIEST-registered item whose description
    contains the query phrase -- that's the "original" task/event.

    Also tracks, per item_id: `mentions` (every message_id linked to it, in
    order -- Part 2's related_message_ids) and `history` (a
    {message_id, event} list -- Part 2's narrative summary source), and
    `first_seen_order` (registration position -- lets Part 2's semantic
    fallback only ever look backward in time, same rule as everything else).
    """

    def __init__(self):
        self._order = []            # item_ids in registration order
        self._descriptions = {}     # item_id -> lowercased description
        self._state = {}            # item_id -> mutable state dict

    def register(self, item_id: str, description: str, base_state: dict,
                 origin_message_id: str = None, position: int = None):
        if item_id in self._descriptions:
            return
        self._order.append(item_id)
        self._descriptions[item_id] = description.lower()
        state = dict(base_state)
        state["mentions"] = [origin_message_id] if origin_message_id else []
        state["history"] = [{"message_id": origin_message_id, "event": "created"}] if origin_message_id else []
        state["first_seen_order"] = position if position is not None else len(self._order) - 1
        self._state[item_id] = state

    def find(self, subject_norm: str):
        if not subject_norm:
            return None
        for item_id in self._order:
            if subject_norm in self._descriptions[item_id]:
                return item_id
        return None

    def all_item_ids(self):
        return list(self._order)

    def description_for(self, item_id: str) -> str:
        return self._descriptions.get(item_id, "")

    def state_for(self, item_id: str) -> dict:
        return self._state.get(item_id, {})

    def add_mention(self, item_id: str, message_id: str, event: str = "mentioned"):
        """Record a message as related to item_id WITHOUT applying any state
        change -- used for semantic-fallback links (Part 2), where we know a
        message is likely about this subject but not what kind of update (if
        any) it represents, so status/deadline must not move."""
        state = self._state.setdefault(item_id, {})
        state.setdefault("mentions", []).append(message_id)
        state.setdefault("history", []).append({"message_id": message_id, "event": event})

    def apply_update(self, item_id: str, update: dict, msg_date: datetime, message_id: str = None):
        state = self._state.setdefault(item_id, {})
        utype = update["update_type"]
        if message_id:
            state.setdefault("mentions", []).append(message_id)
            state.setdefault("history", []).append({"message_id": message_id, "event": utype})
        state["reopened"] = False
        # A new deadline/reschedule/conflict is more specific and more recent
        # than a prior "completed"/"cancelled" claim -- it contradicts that
        # claim, so treat the item as active again rather than trusting stale
        # status over fresh information.
        if utype in {"rescheduled", "deadline_changed", "conflicting_deadline"} and \
                state.get("status") in {"completed", "cancelled"}:
            state["status"] = "pending"
            state["reopened"] = True
        if utype == "completed":
            state["status"] = "completed"
        elif utype == "cancelled":
            state["status"] = "cancelled"
        elif utype == "rescheduled":
            state["status"] = "rescheduled"
            new_deadline, new_time = resolve_when(update["when_raw"], msg_date)
            if new_deadline:
                state["deadline"] = new_deadline
            if new_time:
                state["time"] = new_time
        elif utype == "deadline_changed":
            new_deadline, new_time = resolve_when(update["when_raw"], msg_date)
            old_deadline = state.get("deadline")
            state["deadline_direction"] = None
            if new_deadline and old_deadline:
                state["deadline_direction"] = "earlier" if new_deadline < old_deadline else "later"
            if new_deadline:
                state["deadline"] = new_deadline
            if new_time:
                state["time"] = new_time
            state["urgent"] = update["urgent"]
        elif utype == "conflicting_deadline":
            new_deadline, _ = resolve_when(update["when_raw"], msg_date)
            if new_deadline:
                state["deadline"] = new_deadline
            state["last_conflicting"] = True
        if utype != "conflicting_deadline":
            state["last_conflicting"] = False
        state["last_update_type"] = utype
        state["mention_count"] = state.get("mention_count", 0) + 1
        return state


def _base_state(item):
    return {"sub_class": item["sub_class"], "type": item["type"],
            "deadline": item["deadline"], "time": item["time"], "status": "pending"}


def _mint_ref_id(message_id):
    # A status-changing/status-check message can be the FIRST mention of a
    # subject that extract.py never turned into a formal item (its phrasing
    # doesn't hit any extraction keyword). Rather than silently dropping
    # every later update about that subject, mint a lightweight reference id
    # from this message so the chain still links up -- kept visually
    # distinct (REF_ prefix) from real TASK_/EVENT_ ids so it's obvious in
    # the output which subjects have no formal item behind them.
    return f"REF_{message_id}"


def resolve_messages(messages, extracted_items):
    """
    Single chronological pass shared by priority.py and grouping.py: for
    every message, decide which item_id (if any) it belongs to, registering
    new subjects and applying template-matched updates to a SubjectRegistry
    as it goes. Both modules must see the SAME resolution -- if they each
    ran their own version of this loop, they could quietly disagree about
    which messages belong to the same subject.

    Returns (registry, resolutions) where resolutions is a list of dicts,
    one per message, in the same order as `messages`:
        {message_id, item_id_or_None, is_restatement, update_or_None,
         link_type}
    link_type is "origin" (this message created the item), "template" (this
    message matched a status-change/status-check template and resolved to
    an existing subject), or None (unresolved by this deterministic pass --
    priority.py stops here; grouping.py may still try a semantic fallback).
    """
    items_by_msg = {i["source_message_id"]: i for i in extracted_items}
    registry = SubjectRegistry()
    resolutions = []

    for position, m in enumerate(messages):
        mid = m["message_id"]
        item = items_by_msg.get(mid)
        update = detect_status_update(m["message"])

        resolved_item_id = None
        is_restatement = False
        link_type = None

        found_item_id = registry.find(update["subject_norm"]) if update and update["subject_norm"] else None

        if update:
            if found_item_id:
                registry.apply_update(found_item_id, update, m["timestamp"], message_id=mid)
                resolved_item_id = found_item_id
                is_restatement = item is not None  # item also exists but we deliberately don't use it
                link_type = "template"
            elif item:
                # NOTE: no origin_message_id here -- apply_update() below
                # records this same message as the mention. Passing it to
                # both would double-record mid in state["mentions"].
                registry.register(item["item_id"], item["description"], _base_state(item),
                                   position=position)
                registry.apply_update(item["item_id"], update, m["timestamp"], message_id=mid)
                resolved_item_id = item["item_id"]
                link_type = "origin"
            elif update["subject_norm"]:
                ref_id = _mint_ref_id(mid)
                registry.register(ref_id, m["message"], {
                    "sub_class": None, "type": "reference", "deadline": None, "time": None, "status": "pending",
                }, position=position)
                registry.apply_update(ref_id, update, m["timestamp"], message_id=mid)
                resolved_item_id = ref_id
                link_type = "origin"
            # else: no subject could be resolved at all -- deliberately not
            # linked or invented (see README "Assumptions and limitations").
        elif item:
            registry.register(item["item_id"], item["description"], _base_state(item),
                               origin_message_id=mid, position=position)
            resolved_item_id = item["item_id"]
            link_type = "origin"

        resolutions.append({
            "message_id": mid,
            "item_id": resolved_item_id,
            "is_restatement": is_restatement,
            "update": update,
            "link_type": link_type,
        })

    return registry, resolutions
