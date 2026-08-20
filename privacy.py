"""
Privacy-Aware Routing (L2 Part 3)
--------------------------------------
Routes every message to exactly one of three tiers. No new detection
logic here -- this is a routing DECISION layered directly on top of
sensitive.py's already-tested regex findings, not a second sensitivity
detector.

  blocked                -- high-risk credentials/secrets (OTP, password,
                             card number, bank account, auth token,
                             recovery code). Must never be sent externally
                             or stored -- matches sensitive.py's own
                             "do_not_store" recommended_action.
  requires_confirmation   -- medium-risk PII (phone, email, physical
                             address). Can be processed, but a human should
                             confirm before any external action --
                             matches sensitive.py's "ask_for_confirmation".
  local                   -- no sensitive finding at all. Processed
                             locally with no restriction.

If a single message trips multiple findings, the HIGHEST-risk route wins
(blocked > requires_confirmation) -- never average or pick the first match.
"""

ROUTE_BLOCKED = "blocked"
ROUTE_CONFIRM = "requires_confirmation"
ROUTE_LOCAL = "local"

_ACTION_TO_ROUTE = {
    "do_not_store": ROUTE_BLOCKED,
    "ask_for_confirmation": ROUTE_CONFIRM,
}


def route_message(message_id: str, hits: list) -> dict:
    """
    hits: the sensitive_findings entries for THIS message_id (from
    sensitive.detect_sensitive / results/output_sensitive_findings.json).
    Returns {message_id, route, risk, types, reason}.
    """
    if not hits:
        return {
            "message_id": message_id,
            "route": ROUTE_LOCAL,
            "risk": "none",
            "types": [],
            "reason": "No sensitive information detected in this message -- safe to process locally.",
        }

    routes = [_ACTION_TO_ROUTE.get(h["recommended_action"], ROUTE_CONFIRM) for h in hits]
    types = [h["sensitivity_type"] for h in hits]

    if ROUTE_BLOCKED in routes:
        blocked_types = [h["sensitivity_type"] for h in hits if _ACTION_TO_ROUTE.get(h["recommended_action"]) == ROUTE_BLOCKED]
        return {
            "message_id": message_id,
            "route": ROUTE_BLOCKED,
            "risk": "high",
            "types": types,
            "reason": f"Contains high-risk credential(s) ({', '.join(blocked_types)}) -- "
                      f"must not be sent externally or stored, per sensitive.py's do_not_store action.",
        }

    confirm_types = [h["sensitivity_type"] for h in hits if _ACTION_TO_ROUTE.get(h["recommended_action"]) == ROUTE_CONFIRM]
    return {
        "message_id": message_id,
        "route": ROUTE_CONFIRM,
        "risk": "medium",
        "types": types,
        "reason": f"Contains personal information ({', '.join(confirm_types)}) -- "
                  f"requires user confirmation before any external processing.",
    }


def route_all(messages: list, sensitive_findings: list) -> list:
    """
    messages: chronological list of {message_id, timestamp, sender, message}
    sensitive_findings: list of dicts from detect_sensitive, across the batch
    Returns one routing decision per message, in the same chronological order.
    """
    hits_by_msg = {}
    for f in sensitive_findings:
        hits_by_msg.setdefault(f["message_id"], []).append(f)

    return [route_message(m["message_id"], hits_by_msg.get(m["message_id"], [])) for m in messages]
