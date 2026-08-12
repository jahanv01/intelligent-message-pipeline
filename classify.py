"""
Message Classification
------------------------
Architecture: sub-class -> main category via sentence embeddings.

Instead of keyword lists, we define narrow sub-classes
(e.g. "scheduled_meeting", "task_assignment", "preference_disclosure").
Each sub-class has representative anchor sentences.
A sentence-transformer model embeds the message and every anchor;
cosine similarity picks the closest sub-class, which rolls up to the
main category deterministically.

Why sub-classes:
  - Categories like meeting_or_event and action_required overlap in
    phrasing. Sub-classes are narrow enough to distinguish reliably.
  - An unseen phrasing still lands correctly if it means the same thing.

Sensitive info is still found via regex (the right tool for card
numbers and OTPs, not semantic similarity).

Requires: sentence-transformers  (pip install sentence-transformers)
Falls back to keyword heuristics if the library is not installed.
"""
import re
import numpy as np
from sensitive import detect_sensitive

# ---------------------------------------------------------------------------
# Sub-class taxonomy  (sub-class -> main category)
# ---------------------------------------------------------------------------
SUBCLASSES = [
    # -- meeting_or_event ----------------------------------------------------
    {
        "sub": "scheduled_meeting",
        "category": "meeting_or_event",
        "anchors": [
            "The team standup is on Monday at 10 AM in conference room 3.",
            "Client discussion is scheduled for September 12 at 11:00 in Meeting Room A.",
            "Please join the AI workshop on 2026-09-18 at 9:00 at Conference Room 2.",
            "The sprint review is set for Friday at 3 PM in the office.",
            "Board meeting confirmed for next Tuesday at 2 PM in the main hall.",
        ],
    },
    {
        "sub": "calendar_reminder",
        "category": "meeting_or_event",
        "anchors": [
            "Calendar update: family dinner on September 19 at 10:00 at the library.",
            "Reminder: doctor appointment happens on 2026-09-06 at 9:00 in the city clinic.",
            "Calendar update: team stand-up, 2026-09-04 at 11:00, the college auditorium.",
            "Just a heads-up: mentor catch-up is on September 16 at 11:00.",
            "FYI: your dentist appointment is tomorrow at 4 PM.",
        ],
    },
    {
        "sub": "availability_check",
        "category": "meeting_or_event",
        "anchors": [
            "Are you available for the college seminar at 14:00 on September 15?",
            "Can we meet sometime next week to discuss the project?",
            "Is Thursday afternoon good for a quick sync call?",
            "Let me know if you are free for a call on Friday.",
            "Just checking, are you available for the webinar at 3 PM?",
        ],
    },
    {
        "sub": "event_announcement",
        "category": "meeting_or_event",
        "anchors": [
            "The event registration desk opens at 9 AM tomorrow.",
            "The internship orientation is on September 18 at 1 PM.",
            "The product launch event is next Thursday in the auditorium.",
            "Annual tech fest starts this weekend at the campus ground.",
            "The hackathon kick-off is at 10 AM on Saturday.",
        ],
    },

    # -- action_required -----------------------------------------------------
    {
        "sub": "task_assignment",
        "category": "action_required",
        "anchors": [
            "Please submit your report by Friday.",
            "Can you update the project tracker before 2026-09-04?",
            "I need you to review the model results by tomorrow.",
            "Kindly upload the signed document before end of day.",
            "Please reply to the client email by September 7.",
            "Don't forget to call the service centre by Thursday.",
        ],
    },
    {
        "sub": "deadline_reminder",
        "category": "action_required",
        "anchors": [
            "Don't forget to pay the electricity bill, deadline is September 9.",
            "Complete the onboarding form, it is due on September 10.",
            "The assignment submission deadline is tomorrow.",
            "Reminder: the Python exercise must be completed by September 9.",
            "Send the expense receipt, it is due on September 7.",
            "Please confirm the interview slot by September 5.",
        ],
    },
    {
        "sub": "approval_or_review",
        "category": "action_required",
        "anchors": [
            "Kindly approve the budget proposal at your earliest.",
            "Please review the privacy checklist before September 9.",
            "Can you review the file before the meeting?",
            "The document needs your sign-off before it goes to the client.",
            "Waiting for your approval on the leave request.",
        ],
    },

    # -- personal_information ------------------------------------------------
    {
        "sub": "location_disclosure",
        "category": "personal_information",
        "anchors": [
            "My home address is 42 Lake View Road, Chennai.",
            "I live near the central library on the east side.",
            "For my profile: I reside at 10 Green Park Avenue.",
            "My delivery address is Flat 3B, Tower 2, Sunrise Apartments.",
            "I live at 7 Marine Drive, Mumbai.",
        ],
    },
    {
        "sub": "preference_disclosure",
        "category": "personal_information",
        "anchors": [
            "Just so you know, I am vegetarian.",
            "Personal note: my T-shirt size is medium.",
            "For my profile: I prefer evening meetings.",
            "I usually study after dinner, mornings don't work for me.",
            "My favourite programming language is Python.",
            "I drink coffee without sugar.",
        ],
    },
    {
        "sub": "profile_or_contact",
        "category": "personal_information",
        "anchors": [
            "My emergency contact is my brother.",
            "For my profile: I prefer receiving updates by email.",
            "My contact number is listed on the portal.",
            "Please note my date of birth for the records.",
            "For the records, my name is Aarav Sharma.",
        ],
    },
    {
        "sub": "health_information",
        "category": "personal_information",
        "anchors": [
            "My recent test result says vitamin D deficiency.",
            "The doctor confirmed I have mild anaemia.",
            "My blood pressure report came back normal.",
            "Health update: I was diagnosed with seasonal allergies.",
            "My medical report is attached for your reference.",
        ],
    },

    {
        "sub": "casual_or_social",
        "category": "general_information",
        "anchors": [
            "Just checking in, hope you're doing well.",
            "How's everything going on your end?",
            "Just wanted to say hi and see how things are.",
            "Hope your week is going well!",
            "Just a quick hello — let me know if you need anything.",
        ],
    },
    {
        "sub": "system_or_infra_update",
        "category": "general_information",
        "anchors": [
            "The office Wi-Fi will be under maintenance tonight.",
            "The server will be restarted at midnight for updates.",
            "The building entrance has moved temporarily to the south gate.",
            "The portal will be down for maintenance from 2 to 4 AM.",
            "Network services will be unavailable on Sunday morning.",
        ],
    },
    {
        "sub": "logistics_info",
        "category": "general_information",
        "anchors": [
            "The shuttle leaves every thirty minutes from gate B.",
            "The cafeteria will close early today at 6 PM.",
            "Parking on the second floor is unavailable this week.",
            "The training material is now on the portal.",
            "The new office address is effective from October 1.",
        ],
    },
    {
        "sub": "status_update",
        "category": "general_information",
        "anchors": [
            "The project folder was reorganized on the shared drive.",
            "The webinar recording is now available on the portal.",
            "The laptop battery is fully charged and ready.",
            "The report may be needed tomorrow for the review.",
            "The file might be required sometime next week.",
            "The document could possibly be needed later today.",
            "Maya asked whether the demo was ready.",
            "The review could happen sometime Friday afternoon.",
            "The work may be useful at some point in the future.",
        ],
    },

    # -- promotional ---------------------------------------------------------
    {
        "sub": "discount_or_sale",
        "category": "promotional",
        "anchors": [
            "Special festival discount on clothing. Use code SAVE17 for 20% off.",
            "Flash sale on laptops starts at 6 PM today.",
            "Buy now and get flat 50% off on all items. Limited time offer.",
            "End-of-season sale: up to 70% off on selected brands.",
            "Your food-delivery coupon expires tonight. Use code SAVE42.",
        ],
    },
    {
        "sub": "subscription_or_plan",
        "category": "promotional",
        "anchors": [
            "Join our premium plan for exclusive benefits. Use code SAVE49.",
            "Subscribe today and get one month free.",
            "Upgrade to the pro tier and unlock all features.",
            "Exclusive member offer: free shipping on your next order.",
            "Sign up now and enjoy a 30-day free trial.",
        ],
    },
]

# ---------------------------------------------------------------------------
# Only hard credentials elevate to sensitive_information in Part 1.
# Medium-risk PII stays personal_information but is still flagged in Part 3.
# ---------------------------------------------------------------------------
HIGH_RISK_TYPES = {
    "card_number", "bank_account_number", "one_time_password",
    "pin", "password", "auth_token", "recovery_code",
}

DATE_TIME_SIGNAL = re.compile(
    r"\d{4}-\d{2}-\d{2}.{0,15}\d{1,2}:\d{2}|"
    r"\d{1,2}:\d{2}.{0,15}\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Embedding engine -- loaded once at import time
# ---------------------------------------------------------------------------
_model = None
_anchor_matrix = None   # shape: (n_subclasses, dim), rows pre-normalised
_subclass_list = None


def _load_model():
    global _model, _anchor_matrix, _subclass_list
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _subclass_list = SUBCLASSES
        rows = []
        for sc in _subclass_list:
            vecs = _model.encode(sc["anchors"], convert_to_numpy=True)
            rows.append(vecs.mean(axis=0))
        mat = np.array(rows)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        _anchor_matrix = mat / np.clip(norms, 1e-10, None)
    except ImportError:
        _model = None


_load_model()


def _cosine_classify(message: str):
    """Return (sub_class_dict, similarity_score)."""
    vec = _model.encode([message], convert_to_numpy=True)[0]
    norm = np.linalg.norm(vec)
    vec = vec / max(norm, 1e-10)
    sims = _anchor_matrix.dot(vec)
    best = int(np.argmax(sims))
    return _subclass_list[best], float(sims[best])


# ---------------------------------------------------------------------------
# Keyword fallback (only when sentence-transformers is not installed)
# ---------------------------------------------------------------------------
_KW_MAP = {
    "meeting_or_event": [
        "meeting", "schedule", "conference", "webinar", "appointment",
        "join the", "join us", "zoom", "calendar update", "seminar",
        "catch up", "interview", "event", "reminder:",
    ],
    "action_required": [
        "please submit", "need you to", "kindly", "reply to",
        "review the", "complete", "deadline", "due", "submit",
        "update the", "please update", "approve",
    ],
    "personal_information": [
        "home address", "i live at", "i live near", "my name is",
        "contact number", "date of birth", "personal note", "for my profile",
    ],
    "promotional": [
        "discount", "% off", "sale", "coupon", "use code",
        "buy now", "free trial", "subscribe", "exclusive offer",
    ],
}
_KW_ORDER = ["meeting_or_event", "action_required", "personal_information", "promotional"]
_SELF_DISC = re.compile(
    r"\bmy\s+[a-z][a-z\-]*(\s+[a-z][a-z\-]*){0,2}\s+(is|are)\s+", re.IGNORECASE
)


def _keyword_classify(message: str):
    text = message.lower()
    for cat in _KW_ORDER:
        hits = [kw for kw in _KW_MAP[cat] if kw in text]
        if hits:
            return cat, f"keyword: {', '.join(hits[:3])}", 0.60
    if DATE_TIME_SIGNAL.search(message):
        return "meeting_or_event", "date+time pattern", 0.65
    if _SELF_DISC.search(message):
        return "personal_information", "self-disclosure pattern", 0.55
    return "general_information", "no specific signal found", 0.40


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_message(message_id: str, message: str):
    # Sensitive detection always runs -- feeds Part 3 output unconditionally.
    sensitive_hits = detect_sensitive(message_id, message)
    high_risk_hits = [h for h in sensitive_hits if h["sensitivity_type"] in HIGH_RISK_TYPES]

    # Hard override: credentials/payment data -> sensitive_information.
    if high_risk_hits:
        types = [h["sensitivity_type"] for h in high_risk_hits]
        return {
            "message_id": message_id,
            "category": "sensitive_information",
            "confidence": 0.97,
            "reason": f"Credential detected: {', '.join(types)}",
        }, sensitive_hits

    if _model is not None:
        sub, score = _cosine_classify(message)
        category = sub["category"]
        # MiniLM similarities typically 0.2-0.9; map to 0.50-0.95
        confidence = round(min(0.50 + score * 0.50, 0.95), 2)
        reason = f"Sub-class '{sub['sub']}' (similarity {score:.2f})"
    else:
        category, reason, confidence = _keyword_classify(message)

    # Medium-risk PII detected but model returned general_information -> promote.
    if category == "general_information" and sensitive_hits:
        pii = [h["sensitivity_type"] for h in sensitive_hits]
        category = "personal_information"
        reason = f"PII detected ({', '.join(pii)}); promoted from general_information"
        confidence = round(min(confidence + 0.10, 0.90), 2)

    return {
        "message_id": message_id,
        "category": category,
        "confidence": confidence,
        "reason": reason,
    }, sensitive_hits


if __name__ == "__main__":
    tests = [
        ("MSG_T1", "Hey Ram, your OTP is 4821 for the login."),
        ("MSG_T2", "Please join the team standup on Monday at 10 AM in Room 3."),
        ("MSG_T3", "Can you submit the quarterly report by Friday?"),
        ("MSG_T4", "Flash sale on laptops tonight. Use code SAVE23."),
        ("MSG_T5", "The office Wi-Fi will be down for maintenance tonight."),
        ("MSG_T6", "For my profile: I am vegetarian and prefer evenings."),
        ("MSG_T7", "My home address is 42 Lake View Road, Chennai."),
        ("MSG_T8", "I usually study after dinner, mornings don't work for me."),
    ]
    for mid, msg in tests:
        result, _ = classify_message(mid, msg)
        print(f"{mid}: [{result['category']}] conf={result['confidence']} -- {result['reason']}")
