"""What happens after sign-up: add a parent, fix their meal times, create the routines, put the first calls on the calendar.
Everything the dashboard shows or changes goes through here, so the API stays thin and the logic is testable without HTTP."""
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .accounts import AuthError, normalize_phone
from .agent import tools
from .agent.routines import CARETAKER, DEFAULT_MEAL_TIMES, MEALS, effective_meal_times
from .calls.keys import occurrence_key
from .calls.planner import fire_time, plan_day
from .calls.service import LIVE_STATES, in_quiet_hours, schedule_attempt
from .calls.state import TERMINAL
from .config import get_settings
from .db import session_scope
from .models import (Alert, CallAttempt, Callback, Event, MealLog, MealProfile, Medicine, MedicineLog, Parent, ParentCaregiver,
                      Reminder, Routine, TranscriptTurn, as_utc)

HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
TIMINGS = ("before_food", "after_food", "with_food")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "medicine"


def onboard_parent(caregiver_id: int, data: dict) -> dict:
    """Create the parent, meal times (set by the caregiver, so the schedule obeys them), medicines and routines, then plan today's calls."""
    s_ = get_settings()
    phone = normalize_phone(data.get("phone", ""))
    name = (data.get("parent_name") or "").strip()
    if not name:
        raise AuthError("Enter your parent's name.")
    if not data.get("consent"):
        raise AuthError("Please confirm that your parent knows about and agrees to these calls.")
    tz = data.get("timezone") or s_.default_timezone
    try:
        ZoneInfo(tz)
    except Exception:  # noqa: BLE001
        raise AuthError(f"Unknown time zone {tz!r}.")
    meals = {**DEFAULT_MEAL_TIMES, **{k: v for k, v in (data.get("meals") or {}).items() if v}}
    for meal, t in meals.items():
        if meal not in MEALS or not HHMM.match(t):
            raise AuthError(f"Meal time for {meal} must look like 13:30.")
    meds = data.get("medicines") or []
    for m in meds:
        if not (m.get("name") or "").strip():
            raise AuthError("Every medicine needs a name.")
        if m.get("timing", "after_food") not in TIMINGS or m.get("meal_anchor", "lunch") not in meals:
            raise AuthError(f"{m['name']}: pick before/after/with food and a meal that has a time.")
    with session_scope() as s:
        if s.scalars(select(Parent).join(ParentCaregiver, ParentCaregiver.parent_id == Parent.id)
                     .where(ParentCaregiver.caregiver_id == caregiver_id, Parent.phone == phone)).first():
            raise AuthError("You already added a parent with this phone number.")
        parent = Parent(name=name, preferred_name=(data.get("preferred_name") or name.split()[0]).strip(), phone=phone,
                        language=data.get("language") or "en", timezone=tz, consent_given_at=datetime.now(timezone.utc))
        s.add(parent)
        s.flush()
        s.add(ParentCaregiver(parent_id=parent.id, caregiver_id=caregiver_id, relationship_=data.get("relationship") or "child"))
        for meal, t in meals.items():
            s.add(MealProfile(parent_id=parent.id, meal_type=meal, usual_time=t, source=CARETAKER))
        for meal in meals:
            s.add(Routine(parent_id=parent.id, key=f"{meal}_checkin", type="meal_checkin", config={"meal_type": meal}))
        for m in meds:
            timing = m.get("timing", "after_food")
            med = Medicine(parent_id=parent.id, name=m["name"].strip(), dose_note=(m.get("dose_note") or "").strip(), timing=timing,
                           meal_anchor=m.get("meal_anchor", "lunch"), offset_minutes=int(m.get("offset_minutes", 30 if timing == "after_food" else 0)))
            s.add(med)
            s.flush()
            s.add(Routine(parent_id=parent.id, key=f"{_slug(med.name)}_{med.id}_reminder", type="medicine_reminder", config={"medicine_id": med.id}))
        s.flush()
        pid = parent.id
        tools.add_event(s, None, "onboarding.parent_created", {"parent_id": pid, "caregiver_id": caregiver_id, "meals": meals, "medicines": len(meds)})
    created = plan_day(parent_id=pid)
    with session_scope() as s:
        tools.add_event(s, None, "onboarding.planned", {"parent_id": pid, "calls_created": created})
    return {"parent_id": pid, "calls_planned": len(created)}


def _owned_parent(s, caregiver_id: int, parent_id: int) -> Parent:
    p = s.scalars(select(Parent).join(ParentCaregiver, ParentCaregiver.parent_id == Parent.id)
                  .where(Parent.id == parent_id, ParentCaregiver.caregiver_id == caregiver_id)).first()
    if p is None:
        raise AuthError("No such parent on your account.")
    return p


def add_reminder(caregiver_id: int, parent_id: int, message: str, in_minutes: float | None = None, at: str | None = None) -> dict:
    """A one-off reminder call, either N minutes from now (down to the second, so 'in the next minute' means that) or at a local time."""
    message = (message or "").strip()
    if not message:
        raise AuthError("Say what the reminder is about.")
    with session_scope() as s:
        p = _owned_parent(s, caregiver_id, parent_id)
        tz = ZoneInfo(p.timezone)
        if in_minutes is not None:
            when = datetime.now(timezone.utc) + timedelta(minutes=float(in_minutes))
        elif at:
            try:
                when = datetime.strptime(at, "%Y-%m-%d %H:%M").replace(tzinfo=tz).astimezone(timezone.utc)
            except ValueError:
                raise AuthError("Time must look like 2026-10-21 16:30.")
        else:
            raise AuthError("Give either minutes from now or a time.")
        if when < datetime.now(timezone.utc) - timedelta(seconds=5):
            raise AuthError("That time is in the past.")
        rem = Reminder(parent_id=p.id, message=message[:300], due_at=when)
        s.add(rem)
        s.flush()
        routine = Routine(parent_id=p.id, key=f"reminder_{rem.id}", type="defined_reminder", config={"reminder_id": rem.id})
        s.add(routine)
        s.flush()
        local = when.astimezone(tz)
        call, _ = schedule_attempt(s, parent_id=p.id, routine_id=routine.id, attempt_no=1, kind="scheduled", when=when,
                                   occurrence_key=occurrence_key(p.id, routine.key, local.date().isoformat()))
        tools.add_event(s, call.id, "dashboard.reminder_added", {"reminder_id": rem.id, "message": rem.message, "at": when.isoformat(),
                                                                  "by_caregiver": caregiver_id, "calling_enabled": p.calling_enabled})
        return {"reminder_id": rem.id, "call_id": call.id, "at_utc": when.isoformat(), "at_local": f"{local:%Y-%m-%d %H:%M:%S}",
                "calling_enabled": p.calling_enabled, "in_quiet_hours": in_quiet_hours(when, p.timezone)}


def set_calling(caregiver_id: int, parent_id: int, enabled: bool) -> dict:
    from .calls.outcome import cancel_pending

    with session_scope() as s:
        p = _owned_parent(s, caregiver_id, parent_id)
        p.calling_enabled = enabled
        cancelled = 0 if enabled else cancel_pending(s, p.id, "caregiver paused calls")
        tools.add_event(s, None, "dashboard.calling_enabled" if enabled else "dashboard.calling_paused",
                        {"parent_id": p.id, "cancelled_waiting_calls": cancelled})
    if enabled:
        plan_day(parent_id=parent_id)
    return {"calling_enabled": enabled}


def _iso(dt):
    return as_utc(dt).isoformat() if dt else None


def snapshot(caregiver_id: int, parent_id: int | None = None) -> dict:
    """Everything the dashboard shows, in one read: parents, routines with their next call, the calendar, the live call and the activity feed."""
    now = datetime.now(timezone.utc)
    with session_scope() as s:
        parents = s.scalars(select(Parent).join(ParentCaregiver, ParentCaregiver.parent_id == Parent.id)
                            .where(ParentCaregiver.caregiver_id == caregiver_id).order_by(Parent.id)).all()
        out = {"server_time": now.isoformat(), "parents": [{"id": p.id, "name": p.preferred_name} for p in parents], "parent": None}
        if not parents:
            return out
        p = next((x for x in parents if x.id == parent_id), parents[0])
        tz = ZoneInfo(p.timezone)
        profile = effective_meal_times(s, p.id)
        calls = s.scalars(select(CallAttempt).where(CallAttempt.parent_id == p.id).order_by(CallAttempt.id.desc()).limit(60)).all()
        waiting = [c for c in calls if c.status == "scheduled"]
        routines = []
        for r in s.scalars(select(Routine).where(Routine.parent_id == p.id).order_by(Routine.id)):
            nxt = min((c for c in waiting if c.routine_id == r.id), key=lambda c: as_utc(c.scheduled_at), default=None)
            expected = None
            for d in (now.astimezone(tz).date(), now.astimezone(tz).date() + timedelta(days=1)):
                f = fire_time(s, r, profile, d, tz)
                if f and f.astimezone(timezone.utc) > now:
                    expected = f.astimezone(timezone.utc)
                    break
            detail = ""
            if r.type == "medicine_reminder":
                m = s.get(Medicine, r.config.get("medicine_id"))
                detail = f"{m.name} {m.dose_note}, {m.timing.replace('_', ' ')} {m.meal_anchor}" if m else ""
            elif r.type == "defined_reminder":
                rem = s.get(Reminder, r.config.get("reminder_id"))
                detail = rem.message if rem else ""
            routines.append({"id": r.id, "key": r.key, "type": r.type, "detail": detail,
                             "next_call_at": _iso(nxt.scheduled_at if nxt else expected), "next_is_booked": nxt is not None})

        def call_row(c):
            return {"id": c.id, "routine": c.routine.key, "kind": c.kind, "attempt_no": c.attempt_no, "status": c.status,
                    "scheduled_at": _iso(c.scheduled_at), "outcome": c.outcome, "outcome_reason": c.outcome_reason, "end_reason": c.end_reason,
                    "started_at": _iso(c.started_at), "ended_at": _iso(c.ended_at), "twilio_sid": c.twilio_sid, "key": c.idempotency_key}

        live = next((c for c in calls if c.status in LIVE_STATES), None)
        focus = live or next((c for c in calls if c.status in TERMINAL and c.status != "cancelled"), None)
        focus_out = None
        if focus:
            turns = s.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == focus.id).order_by(TranscriptTurn.turn_no)).all()
            focus_out = {**call_row(focus), "live": live is not None,
                         "transcript": [{"n": t.turn_no, "speaker": t.speaker, "text": t.text} for t in turns]}
        call_ids = [c.id for c in calls]
        events = s.scalars(select(Event).where(Event.call_id.in_(call_ids)).order_by(Event.id.desc()).limit(80)).all() if call_ids else []
        facts = []
        for m in s.scalars(select(MealLog).where(MealLog.parent_id == p.id).order_by(MealLog.id.desc()).limit(5)):
            facts.append({"at": _iso(m.created_at), "text": f"meal {m.meal_type}: {m.status} {', '.join(m.items or [])}".strip(), "quote": m.source_utterance})
        for m in s.scalars(select(MedicineLog).where(MedicineLog.parent_id == p.id).order_by(MedicineLog.id.desc()).limit(5)):
            facts.append({"at": _iso(m.created_at), "text": f"medicine #{m.medicine_id}: {m.status}", "quote": m.source_utterance})
        for a in s.scalars(select(Alert).where(Alert.parent_id == p.id).order_by(Alert.id.desc()).limit(5)):
            facts.append({"at": _iso(a.created_at), "text": f"alert ({a.severity}): {a.reason}", "quote": a.source_utterance})
        facts.sort(key=lambda f: f["at"] or "", reverse=True)
        phone = p.phone
        out["parent"] = {
            "id": p.id, "name": p.name, "preferred_name": p.preferred_name, "phone_masked": phone[:3] + "•" * max(0, len(phone) - 6) + phone[-3:],
            "timezone": p.timezone, "local_time": f"{now.astimezone(tz):%H:%M:%S}", "calling_enabled": p.calling_enabled,
            "meal_times": profile, "quiet_hours": f"{get_settings().quiet_start}-{get_settings().quiet_end}",
            "routines": routines,
            "upcoming": [call_row(c) for c in sorted((c for c in calls if c.status == "scheduled" or c.status in LIVE_STATES),
                                                      key=lambda c: as_utc(c.scheduled_at))],
            "recent": [call_row(c) for c in calls if c.status in TERMINAL][:8],
            "focus": focus_out,
            "callbacks": [{"id": cb.id, "call_id": cb.call_id, "delay_minutes": cb.delay_minutes, "reason": cb.reason, "status": cb.status,
                           "created_at": _iso(cb.created_at)}
                          for cb in s.scalars(select(Callback).where(Callback.parent_id == p.id).order_by(Callback.id.desc()).limit(5))],
            "facts": facts[:8],
            "events": [{"id": e.id, "at": _iso(e.created_at), "call_id": e.call_id, "type": e.type, "payload": e.payload} for e in events],
        }
        return out
