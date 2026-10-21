"""Context assembly: everything the agent is allowed to know about THIS call, gathered once at dial time.

Principle: minimum necessary. Only data relevant to this routine goes in (privacy, cost, accuracy).
The context is stored as an audit event so you can later answer 'what did the agent know?'."""
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field
from sqlalchemy import select

from ..config import get_settings
from .routines import effective_meal_times
from ..models import (
    CallAttempt, Caregiver, MealLog, MealProfile, Medicine, ParentCaregiver, Reminder, Routine,
)
from ..models import Parent


class CallContext(BaseModel):
    call_id: int
    parent_id: int
    parent_name: str
    parent_preferred_name: str
    language: str
    timezone: str
    local_now: str
    caregiver_name: str
    routine_key: str
    routine_type: str
    routine_config: dict = Field(default_factory=dict)
    medicine: dict | None = None
    reminder: dict | None = None
    meal_times: dict = Field(default_factory=dict)  # {"lunch": "13:30"} ; missing = unknown
    todays_logs: list[str] = Field(default_factory=list)
    recent_flags: list[str] = Field(default_factory=list)
    medicine_lead_minutes: int = 15
    call_kind: str = "manual"  # scheduled | retry | callback | manual
    attempt_no: int = 1
    prev_outcome: str = ""  # how the previous attempt ended: cut_off, never_connected, callback_requested...
    resume: dict | None = None  # where a cut-off call stopped (steps, idx, learned, pending_meal)
    prompt_version: str = "v1"
    prev_attempt_id: int | None = None  # the attempt this one follows (cut-off ring-back, call-back)


def _local_now(tz: str) -> str:
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo(tz))
    except Exception:
        now = datetime.now(timezone.utc)
    return now.strftime("%A %d %B %Y, %H:%M")


def build_context(session, call_id: int) -> CallContext:
    call = session.get(CallAttempt, call_id)
    parent = session.get(Parent, call.parent_id)
    routine = session.get(Routine, call.routine_id)
    caregiver = session.scalars(
        select(Caregiver).join(ParentCaregiver, ParentCaregiver.caregiver_id == Caregiver.id)
        .where(ParentCaregiver.parent_id == parent.id)
    ).first()

    medicine = reminder = None
    cfg = routine.config or {}
    if routine.type == "medicine_reminder":
        m = session.get(Medicine, cfg["medicine_id"])
        medicine = {"id": m.id, "name": m.name, "dose_note": m.dose_note, "timing": m.timing,
                    "meal_anchor": m.meal_anchor, "offset_minutes": m.offset_minutes}
    if routine.type == "defined_reminder":
        r = session.get(Reminder, cfg["reminder_id"])
        reminder = {"id": r.id, "message": r.message, "ack_required": r.ack_required}

    from .routines import DEFAULT_MEAL_TIMES

    meal_times = effective_meal_times(session, parent.id)  # defaults, plus only what the caretaker fixed

    since = datetime.now(timezone.utc) - timedelta(days=1)
    todays = [
        f"{l.meal_type}: {l.status}" for l in session.scalars(
            select(MealLog).where(MealLog.parent_id == parent.id, MealLog.created_at >= since))
    ]
    skipped_recent = session.scalars(
        select(MealLog).where(MealLog.parent_id == parent.id, MealLog.status == "skipped",
                              MealLog.created_at >= datetime.now(timezone.utc) - timedelta(days=3))
    ).all()
    flags = [f"skipped {l.meal_type} recently" for l in skipped_recent]

    prev_outcome, resume = "", None
    if call.prev_attempt_id:
        prev = session.get(CallAttempt, call.prev_attempt_id)
        prev_outcome = (prev.outcome or "") if prev else ""
        if prev and prev_outcome in ("cut_off", "hung_up_on_purpose"):
            from ..models import Event

            fin = session.scalars(select(Event).where(Event.call_id == prev.id, Event.type == "call.finished")
                                  .order_by(Event.id.desc())).first()
            resume = fin.payload if fin else None

    s = get_settings()
    return CallContext(prev_attempt_id=call.prev_attempt_id, call_kind=call.kind, attempt_no=call.attempt_no, prev_outcome=prev_outcome, resume=resume,
        call_id=call.id, parent_id=parent.id, parent_name=parent.name,
        parent_preferred_name=parent.preferred_name, language=parent.language, timezone=parent.timezone,
        local_now=_local_now(parent.timezone),
        caregiver_name=caregiver.name if caregiver else "your family",
        routine_key=routine.key, routine_type=routine.type, routine_config=cfg,
        medicine=medicine, reminder=reminder, meal_times=meal_times,
        todays_logs=todays, recent_flags=flags,
        medicine_lead_minutes=s.medicine_lead_minutes, prompt_version=s.prompt_version,
    )
