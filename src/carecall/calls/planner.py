"""Turns what was stored at onboarding into calls on the calendar.

Stored at onboarding                    becomes
  parent + timezone + phone             WHO and WHEN (local time -> UTC)
  meal_profile (stated or default)      meal times: breakfast 08:30, lunch 12:30, dinner 19:30 unless the parent said otherwise
  routines + medicines + reminders      WHAT to call about, and at what time:
      meal_checkin     meal time + 30 min               ("have you eaten?")
      medicine before  meal time - 15 min               (take it before food)
      medicine after   meal time + offset (e.g. 30)     (take it after food)
      defined_reminder its due time                     (alarm-like)
      meal_discovery   once, right after onboarding     (learn the meal times)

`plan_day` is idempotent: each planned call has a computed occurrence key (parent, routine, local date), so running it every hour,
or twice by mistake, creates each call once."""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select

from ..agent.routines import DEFAULT_MEAL_TIMES, effective_meal_times  # noqa: F401
from ..config import get_settings
from ..db import session_scope
from ..models import MealProfile, Medicine, Parent, Reminder, Routine
from .keys import occurrence_key
from .service import schedule_attempt

CHECKIN_AFTER_MIN = 30  # ask "have you eaten?" this long after the usual meal time
GRACE = timedelta(minutes=10)  # a call this far in the past is still worth making; older means that window was missed


def _t(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


def fire_time(session, routine: Routine, profile: dict[str, str], day: date, tz: ZoneInfo) -> datetime | None:
    """Local time to call about this routine on `day`, or None if it does not apply today."""
    cfg = routine.config or {}
    combine = lambda t, delta=0: datetime.combine(day, t, tzinfo=tz) + timedelta(minutes=delta)  # noqa: E731
    if routine.type == "meal_checkin":
        meal = profile.get(cfg["meal_type"])
        return combine(_t(meal), CHECKIN_AFTER_MIN) if meal else None  # no time known (e.g. snack): nothing to anchor to
    if routine.type == "medicine_reminder":
        med = session.get(Medicine, cfg["medicine_id"])
        if not med or not med.active:
            return None
        anchor = profile.get(med.meal_anchor or "")
        if not anchor:
            return None
        if med.timing == "before_food":
            return combine(_t(anchor), -get_settings().medicine_lead_minutes)
        if med.timing == "after_food":
            return combine(_t(anchor), med.offset_minutes)
        return combine(_t(anchor))
    if routine.type == "defined_reminder":
        rem = session.get(Reminder, cfg["reminder_id"])
        if rem and rem.due_at:
            due = rem.due_at.astimezone(tz)
            return due if due.date() == day else None
    return None  # meal_discovery is scheduled once at onboarding, not daily


def plan_day(local_date: date | None = None, now: datetime | None = None, parent_id: int | None = None) -> list[int]:
    """Create tomorrow's/today's calls. Returns the ids created this run (empty on a repeat run)."""
    now = now or datetime.now(timezone.utc)
    created: list[int] = []
    with session_scope() as s:
        q = select(Parent).where(Parent.calling_enabled.is_(True))
        if parent_id:
            q = q.where(Parent.id == parent_id)
        for parent in s.scalars(q):
            tz = ZoneInfo(parent.timezone)
            day = local_date or now.astimezone(tz).date()  # "today" is the PARENT's today
            profile = effective_meal_times(s, parent.id)
            for routine in s.scalars(select(Routine).where(Routine.parent_id == parent.id)):
                local = fire_time(s, routine, profile, day, tz)
                if local is None:
                    continue
                when = local.astimezone(timezone.utc)
                if when < now - GRACE:
                    continue  # that moment already passed: do not call at 4 pm about a 1 pm lunch
                call, new = schedule_attempt(s, parent_id=parent.id, routine_id=routine.id, attempt_no=1, kind="scheduled", when=when,
                                             occurrence_key=occurrence_key(parent.id, routine.key, day.isoformat()))
                if new:
                    created.append(call.id)
    return created


def plan_next_occurrence(session, call, now: datetime | None = None) -> int | None:
    """After an occurrence ends (done OR given up), put the same routine on the calendar for the next local day. Idempotent."""
    routine = session.get(Routine, call.routine_id)
    if routine.type not in ("meal_checkin", "medicine_reminder"):  # one-off reminders and discovery do not repeat
        return None
    parts = (call.occurrence_key or "").split(":")
    if len(parts) < 4:
        return None
    parent = session.get(Parent, call.parent_id)
    tz = ZoneInfo(parent.timezone)
    day = date.fromisoformat(parts[2]) + timedelta(days=1)
    profile = effective_meal_times(session, parent.id)
    local = fire_time(session, routine, profile, day, tz)
    if local is None:
        return None
    nxt, new = schedule_attempt(session, parent_id=parent.id, routine_id=routine.id, attempt_no=1, kind="scheduled",
                                when=local.astimezone(timezone.utc), occurrence_key=occurrence_key(parent.id, routine.key, day.isoformat()),
                                channel=call.channel)
    return nxt.id if new else None


def plan_onboarding(parent_id: int, now: datetime | None = None) -> int | None:
    """Right after onboarding: one call to learn the real meal times (until then the defaults are used)."""
    now = now or datetime.now(timezone.utc)
    with session_scope() as s:
        routine = s.scalars(select(Routine).where(Routine.parent_id == parent_id, Routine.type == "meal_discovery")).first()
        if routine is None:
            return None
        call, new = schedule_attempt(s, parent_id=parent_id, routine_id=routine.id, attempt_no=1, kind="scheduled",
                                     when=now + timedelta(minutes=1), occurrence_key=occurrence_key(parent_id, routine.key, "onboarding"))
        return call.id if new else None
