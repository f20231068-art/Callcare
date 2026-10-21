"""The thing that actually fires calls. The database IS the queue: a call is a `call_attempts` row with status='scheduled' and a
`scheduled_at`. Nothing is held in memory, so a restart loses nothing and several workers can share the work.

    planner (daily)  ──►  rows: scheduled, scheduled_at=…   ◄──  outcome engine (retries, call backs, after every call)
                                     │
        dispatch_once()  every ~10 s │  SELECT … WHERE status='scheduled' AND scheduled_at<=now  FOR UPDATE SKIP LOCKED
                                     ▼
              for each due row:  parent paused? → cancel        (opt-out)
                                 parent on a call? → postpone 5 min  (active-call guard)
                                 otherwise → status 'dialing' (committed FIRST), then ask Twilio to ring
                                     │
        reap_stale()  every tick     ▼  calls stuck in dialing/ringing/in_progress → failed → outcome engine (so a lost webhook
                                        can never lock a parent out)

FOR UPDATE SKIP LOCKED (Postgres) is what lets many workers run this loop at once: each locks the rows it takes and skips rows
another worker already holds, so one scheduled call is dialed once. (SQLite ignores the clause; it has only one writer anyway.)"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import select

from ..agent import tools
from ..config import get_settings
from ..db import session_scope
from ..models import CallAttempt, Parent
from .service import has_live_call
from .state import transition

log = logging.getLogger(__name__)
POSTPONE_S = get_settings().postpone_live_call_s


def dispatch_once(dialer: Callable[[int], str], now: datetime | None = None, limit: int = 20) -> list[dict]:
    """Claim due attempts and dial them. `dialer(call_id)` places the real call (Twilio) and returns its sid."""
    now = now or datetime.now(timezone.utc)
    claimed: list[int] = []
    with session_scope() as s:
        due = s.scalars(select(CallAttempt).where(CallAttempt.status == "scheduled", CallAttempt.scheduled_at <= now,
                                                  CallAttempt.channel == "twilio").order_by(CallAttempt.scheduled_at)
                        .limit(limit).with_for_update(skip_locked=True)).all()
        for call in due:
            parent = s.get(Parent, call.parent_id)
            if not parent.calling_enabled:
                transition(s, call, "cancelled", "calls switched off for this parent")
                continue
            live = has_live_call(s, parent.id, exclude_id=call.id)
            if live is not None:  # one person, one call at a time
                log.info("call %s postponed %ss: parent is on call %s", call.id, POSTPONE_S, live.id)
                call.scheduled_at = now + timedelta(seconds=POSTPONE_S)
                tools.add_event(s, call.id, "dispatch.postponed", {"because_live_call": live.id, "until": call.scheduled_at.isoformat()})
                continue
            transition(s, call, "dialing", "dispatcher")  # committed BEFORE dialing: no other worker can take it now
            call.dialed_at = now
            claimed.append(call.id)
    results = []
    for cid in claimed:
        try:
            log.info("dialing call %s", cid)
            sid = dialer(cid)
            log.info("call %s placed with Twilio, sid %s", cid, sid)
            tools.record_event(cid, "dispatch.placed", {"twilio_sid": sid})
            results.append({"call_id": cid, "twilio_sid": sid})
        except Exception as e:  # noqa: BLE001 - could not even place the call: treat as a technical failure and let the outcome engine retry
            log.warning("dialing call %s failed: %s", cid, e)
            tools.record_event(cid, "dispatch.dial_failed", {"error": str(e)[:300]})
            with session_scope() as s:
                transition(s, s.get(CallAttempt, cid), "failed", f"dial error: {str(e)[:120]}")
            from .outcome import apply

            apply(cid)
            results.append({"call_id": cid, "error": str(e)[:120]})
    return results


def reap_stale(now: datetime | None = None) -> list[int]:
    """Calls that never reported back. A lost webhook must not leave a parent 'on a call' forever."""
    s_ = get_settings()
    now = now or datetime.now(timezone.utc)
    reaped: list[int] = []
    with session_scope() as s:
        for c in s.scalars(select(CallAttempt).where(CallAttempt.status.in_(("dialing", "ringing")),
                                                     CallAttempt.dialed_at < now - timedelta(seconds=s_.stale_dial_s))):
            if transition(s, c, "failed", "stale: no status after dialing"):
                reaped.append(c.id)
        for c in s.scalars(select(CallAttempt).where(CallAttempt.status == "in_progress",
                                                     CallAttempt.started_at < now - timedelta(seconds=s_.stale_call_s))):
            if transition(s, c, "failed", "stale: in progress far longer than any real call"):
                reaped.append(c.id)
    from .outcome import apply

    for cid in reaped:
        apply(cid)
    return reaped


async def scheduler_loop(dialer: Callable[[int], str], stop: asyncio.Event | None = None):
    """Run forever (or until `stop` is set): plan the day, reap, dispatch. Blocking DB work runs in a thread."""
    from .planner import plan_day

    s_ = get_settings()
    last_plan_hour = None
    while not (stop and stop.is_set()):
        try:
            hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
            if hour != last_plan_hour:  # every hour: cheap, idempotent, and catches parents in every timezone
                made = await asyncio.to_thread(plan_day)
                if made:
                    log.info("planner created calls %s", made)
                last_plan_hour = hour
            reaped = await asyncio.to_thread(reap_stale)
            if reaped:
                log.warning("reaped stale calls %s", reaped)
            await asyncio.to_thread(dispatch_once, dialer)
        except Exception:  # noqa: BLE001 - the loop must survive anything
            log.exception("scheduler tick failed")
        await asyncio.sleep(s_.scheduler_interval_s)
