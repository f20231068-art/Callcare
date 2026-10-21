"""Creating and scheduling call attempts.

create_call()       an attempt NOW (CLI, API, tests). An optional idempotency key makes "start this call" safe to repeat.
schedule_attempt()  an attempt LATER with a COMPUTED key (calls/keys.py): the planner, retries and callbacks all use it.
has_live_call()     the active-call guard: is this parent on a call right now? (an idempotency key cannot answer that)."""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..db import session_scope
from ..models import CallAttempt, Parent, Routine
from .keys import attempt_key

LIVE_STATES = ("dialing", "ringing", "in_progress")


def has_live_call(session, parent_id: int, exclude_id: int | None = None) -> CallAttempt | None:
    """The parent's current live call, if any. Two overlapping calls to one person must never happen."""
    q = select(CallAttempt).where(CallAttempt.parent_id == parent_id, CallAttempt.status.in_(LIVE_STATES))
    if exclude_id is not None:
        q = q.where(CallAttempt.id != exclude_id)
    return session.scalars(q).first()


def create_call(routine_key: str, parent_id: int | None = None, channel: str = "text",
                idempotency_key: str | None = None, occurrence_key: str | None = None, kind: str = "manual") -> tuple[int, bool]:
    """Returns (call_id, created). `created` is False when the idempotency key already existed."""
    with session_scope() as s:
        if idempotency_key:
            existing = s.scalars(select(CallAttempt).where(CallAttempt.idempotency_key == idempotency_key)).first()
            if existing:
                return existing.id, False
        parent_id = parent_id or s.scalars(select(Parent.id)).first()
        routine = s.scalars(select(Routine).where(Routine.parent_id == parent_id, Routine.key == routine_key)).first()
        if routine is None:
            raise ValueError(f"unknown routine {routine_key!r} for parent {parent_id}")
        call = CallAttempt(parent_id=parent_id, routine_id=routine.id, channel=channel, idempotency_key=idempotency_key,
                           occurrence_key=occurrence_key, kind=kind)
        try:
            # The row is added INSIDE a SAVEPOINT. If a concurrent request got the same key first, the insert fails, the savepoint
            # rolls back AND the row is discarded from the session (adding it outside would leave it pending, to be inserted
            # again at commit and crash). Real Postgres concurrency found this; SQLite never could.
            with s.begin_nested():
                s.add(call)
                s.flush()
        except IntegrityError:  # lost a race with a concurrent identical request
            existing = s.scalars(select(CallAttempt).where(CallAttempt.idempotency_key == idempotency_key)).first()
            return existing.id, False
        return call.id, True


def schedule_attempt(session, *, parent_id: int, routine_id: int, occurrence_key: str, attempt_no: int, kind: str,
                     when: datetime, prev_attempt_id: int | None = None, channel: str = "twilio") -> tuple[CallAttempt, bool]:
    """Insert a `scheduled` attempt whose key is computed from (occurrence, number, kind). Returns (attempt, created).
    Calling this twice for the same follow-up returns the first one: that is the whole point."""
    key = attempt_key(occurrence_key, attempt_no, kind)
    found = session.scalars(select(CallAttempt).where(CallAttempt.idempotency_key == key)).first()
    if found:
        return found, False
    call = CallAttempt(parent_id=parent_id, routine_id=routine_id, status="scheduled", scheduled_at=when, channel=channel,
                       occurrence_key=occurrence_key, kind=kind, attempt_no=attempt_no, prev_attempt_id=prev_attempt_id,
                       idempotency_key=key)
    try:
        with session.begin_nested():  # add inside the savepoint: see create_call
            session.add(call)
            session.flush()
    except IntegrityError:
        return session.scalars(select(CallAttempt).where(CallAttempt.idempotency_key == key)).one(), False
    return call, True


def _hhmm(v: str) -> time:
    h, m = v.split(":")
    return time(int(h), int(m))


def in_quiet_hours(when_utc: datetime, tz: str) -> bool:
    s = get_settings()
    t = when_utc.astimezone(ZoneInfo(tz)).time()
    start, end = _hhmm(s.quiet_start), _hhmm(s.quiet_end)
    return t >= start or t < end  # the window crosses midnight


def shift_out_of_quiet_hours(when_utc: datetime, tz: str) -> datetime:
    """Move a scheduled time out of the parent's night: 22:10 -> 07:00 the next morning, 05:30 -> 07:00 today."""
    if not in_quiet_hours(when_utc, tz):
        return when_utc
    s = get_settings()
    local = when_utc.astimezone(ZoneInfo(tz))
    end = datetime.combine(local.date(), _hhmm(s.quiet_end), tzinfo=ZoneInfo(tz))
    if local.time() >= _hhmm(s.quiet_start):
        end += timedelta(days=1)
    return end.astimezone(timezone.utc)
