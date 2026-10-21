"""Call-attempt state machine.

A call is in exactly one state; only the transitions listed here are legal. Illegal or duplicate
transitions (e.g. a repeated Twilio webhook, or 'ringing' arriving after 'completed') are ignored,
which makes status handling idempotent."""
from datetime import datetime, timezone

from ..models import CallAttempt, Event

TRANSITIONS: dict[str, set[str]] = {
    "scheduled": {"dialing", "in_progress", "cancelled"},  # text channel goes straight to in_progress
    "dialing": {"ringing", "in_progress", "no_answer", "failed"},
    "ringing": {"in_progress", "no_answer", "failed"},
    "in_progress": {"completed", "abandoned", "failed"},
    "completed": set(),
    "no_answer": set(),
    "failed": set(),
    "abandoned": set(),
    "cancelled": set(),
}
TERMINAL = {s for s, nxt in TRANSITIONS.items() if not nxt}


def transition(session, call: CallAttempt, new_status: str, reason: str = "") -> bool:
    """Move the call to `new_status` if legal. Returns True if the state changed."""
    old = call.status
    if new_status == old or new_status not in TRANSITIONS.get(old, set()):
        session.add(Event(call_id=call.id, type="state.ignored", payload={"from": old, "to": new_status}))
        return False
    call.status = new_status
    now = datetime.now(timezone.utc)
    if new_status == "dialing":
        call.dialed_at = now
    if new_status == "in_progress":
        call.started_at = now
    if new_status in TERMINAL:
        call.ended_at = now
    session.add(Event(call_id=call.id, type="state.changed", payload={"from": old, "to": new_status, "reason": reason}))
    return True
