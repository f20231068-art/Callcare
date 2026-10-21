"""Tools: the ONLY code that writes facts to the database during a call.

Each tool is a plain function called by the graph's `act` node after validation. (Contrast with
'LLM function calling', where the model decides to call these. Here code decides; the model only
supplies interpreted values. That makes behavior deterministic and testable.)

Every tool writes (a) its domain row and (b) an append-only audit event."""
from datetime import datetime, timezone

from .. import tracing
from ..db import session_scope
from ..models import (
    Alert, Callback, Event, MealLog, MealProfile, MedicineLog, ReminderAck, Routine, CallAttempt,
)


def add_event(session, call_id: int | None, type_: str, payload: dict):
    session.add(Event(call_id=call_id, type=type_, payload=payload))


def record_event(call_id: int | None, type_: str, payload: dict):
    with session_scope() as s:
        add_event(s, call_id, type_, payload)


@tracing.traced("tool:log_meal", as_type="tool")
def log_meal(ctx, meal_type: str, status: str, items: list[str], evidence: str, confidence: float):
    with session_scope() as s:
        s.add(MealLog(parent_id=ctx["parent_id"], call_id=ctx["call_id"], meal_type=meal_type, status=status,
                      items=items, source_utterance=evidence, confidence=confidence))
        add_event(s, ctx["call_id"], "tool.log_meal",
                  {"meal_type": meal_type, "status": status, "items": items, "evidence": evidence})


@tracing.traced("tool:log_medicine", as_type="tool")
def log_medicine(ctx, medicine_id: int, status: str, evidence: str, confidence: float):
    with session_scope() as s:
        s.add(MedicineLog(parent_id=ctx["parent_id"], call_id=ctx["call_id"], medicine_id=medicine_id,
                          status=status, source_utterance=evidence, confidence=confidence))
        add_event(s, ctx["call_id"], "tool.log_medicine",
                  {"medicine_id": medicine_id, "status": status, "evidence": evidence})


@tracing.traced("tool:set_meal_time", as_type="tool")
def set_meal_time(ctx, meal_type: str, usual_time: str, evidence: str, source: str = "stated"):
    with session_scope() as s:
        row = s.get(MealProfile, (ctx["parent_id"], meal_type))
        if row and row.source == "caregiver" and source != "caregiver":
            # the caretaker fixed this time: what the parent says is recorded as a note, the schedule is untouched
            add_event(s, ctx["call_id"], "tool.set_meal_time", {"meal_type": meal_type, "usual_time": usual_time, "evidence": evidence,
                                                                "source": source, "ignored": "caretaker fixed this time"})
            return
        if row:
            row.usual_time, row.source, row.updated_at = usual_time, source, datetime.now(timezone.utc)
        else:
            s.add(MealProfile(parent_id=ctx["parent_id"], meal_type=meal_type, usual_time=usual_time, source=source))
        add_event(s, ctx["call_id"], "tool.set_meal_time",
                  {"meal_type": meal_type, "usual_time": usual_time, "evidence": evidence, "source": source})


@tracing.traced("tool:ack_reminder", as_type="tool")
def ack_reminder(ctx, reminder_id: int, acknowledged: bool, evidence: str):
    with session_scope() as s:
        s.add(ReminderAck(reminder_id=reminder_id, call_id=ctx["call_id"], acknowledged=acknowledged,
                          source_utterance=evidence))
        add_event(s, ctx["call_id"], "tool.ack_reminder", {"reminder_id": reminder_id, "acknowledged": acknowledged})


@tracing.traced("tool:schedule_callback", as_type="tool")
def schedule_callback(ctx, minutes: int, reason: str):
    with session_scope() as s:
        call = s.get(CallAttempt, ctx["call_id"])
        s.add(Callback(parent_id=ctx["parent_id"], call_id=ctx["call_id"], routine_id=call.routine_id,
                       delay_minutes=minutes, reason=reason))
        add_event(s, ctx["call_id"], "tool.schedule_callback", {"minutes": minutes, "reason": reason})


@tracing.traced("tool:relay_message", as_type="tool")
def relay_message(ctx, text: str, severity: str = "info"):
    """Pass a message from the parent to the caregiver (it travels as an alert: it is shown in the summary and notification)."""
    with session_scope() as s:
        s.add(Alert(parent_id=ctx["parent_id"], call_id=ctx["call_id"], severity=severity,
                    reason=f'message for {ctx["caregiver_name"]}: "{text[:200]}"', source_utterance=text))
        add_event(s, ctx["call_id"], "tool.relay_message", {"text": text})


@tracing.traced("tool:raise_alert", as_type="tool")
def raise_alert(ctx, severity: str, reason: str, evidence: str):
    with session_scope() as s:
        s.add(Alert(parent_id=ctx["parent_id"], call_id=ctx["call_id"], severity=severity, reason=reason,
                    source_utterance=evidence))
        add_event(s, ctx["call_id"], "tool.raise_alert", {"severity": severity, "reason": reason, "evidence": evidence})
