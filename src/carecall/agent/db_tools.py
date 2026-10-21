"""Database tools: how the agent (or the outcome engine) READS and CHANGES the schedule, profile and calendar.

Why a registry instead of loose functions
  * One place that says what exists, what each tool needs, and whether it writes  -> the model can be shown the list
    (`openai_tool_specs`) and code can call the same tools by name (`call_tool`).
  * One door in: `call_tool` checks the arguments, forces the parent id from the call context (a model can never name
    another parent), enforces who may use it (scope), runs the tool, and writes an audit event + trace span.
  * Tools return plain dicts {"ok": bool, ...}; they never raise into the conversation. A bad call is an answer
    ("ok": false, "error": ...), so the loop can retry or give up.

Idempotency: every write that creates a call uses a computed key (parent:routine:local-date:slot), so the model repeating
itself, or a retried request, never produces a second call. Updates are "set to X", never "add X".

Scopes: "agent" = safe inside a live call; "system" = the outcome engine / planner; "admin" = a human operator only."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .. import tracing
from ..calls.keys import occurrence_key
from ..calls.notify import notify_caregivers
from ..calls.service import schedule_attempt, shift_out_of_quiet_hours
from ..calls.state import transition
from ..db import session_scope
from ..models import (Callback, CallAttempt, MealLog, MealProfile, Medicine, MedicineLog, Parent, Reminder, Routine, as_utc)
from . import tools as write_tools

import re as _re

MEALS = ("breakfast", "lunch", "snack", "dinner")
OPEN_STATES = ("scheduled",)


@dataclass
class Tool:
    name: str
    description: str
    params: dict  # {arg: (type, required, description)}
    fn: Callable
    writes: bool
    scope: str = "agent"  # agent | system | admin
    result_matters: bool = True  # the agent must SEE the result before it speaks (bookings, lookups); False = fire and forget (logging)
    dry: bool = False  # supports dry_run: validate and describe what it WOULD do, write nothing (used for confirm-back)


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, params: dict, writes: bool, scope: str = "agent", dry: bool = False, result_matters: bool = True):
    def deco(fn):
        REGISTRY[name] = Tool(name, description, params, fn, writes, scope, result_matters, dry)
        return fn
    return deco


_TYPES = {"str": str, "int": int, "bool": bool, "list": list}
_JSON = {"str": "string", "int": "integer", "bool": "boolean", "list": "array"}
_SCOPES = {"agent": {"agent"}, "system": {"agent", "system"}, "admin": {"agent", "system", "admin"}}


def openai_tool_specs(scope: str = "agent") -> list[dict]:
    """The tools a model at this scope is allowed to see, in the OpenAI function-calling format."""
    out = []
    for t in REGISTRY.values():
        if t.scope not in _SCOPES[scope]:
            continue
        props = {k: {"type": _JSON[ty], "description": d} for k, (ty, _, d) in t.params.items()}
        req = [k for k, (_, r, _) in t.params.items() if r]
        out.append({"type": "function", "function": {"name": t.name, "description": t.description,
                                                     "parameters": {"type": "object", "properties": props, "required": req,
                                                                    "additionalProperties": False}}})
    return out


def call_tool(name: str, args: dict | None, ctx: dict, scope: str = "agent", dry_run: bool = False) -> dict:
    """The single entry point. `ctx` must carry parent_id (and call_id when inside a call)."""
    args = dict(args or {})
    t = REGISTRY.get(name)
    if t is None:
        return {"ok": False, "error": f"unknown tool {name!r}"}
    if t.scope not in _SCOPES[scope]:
        return {"ok": False, "error": f"tool {name!r} is not allowed here"}
    if "parent_id" in args:  # a model must not be able to aim a tool at someone else
        return {"ok": False, "error": "parent_id is not an argument; it comes from the call"}
    if dry_run and not t.dry:
        return {"ok": False, "error": f"tool {name!r} has no preview"}
    for k in args:
        if k not in t.params:
            return {"ok": False, "error": f"unexpected argument {k!r}"}
    for k, (ty, required, _) in t.params.items():
        if k not in args or args[k] is None:
            if required:
                return {"ok": False, "error": f"missing argument {k!r}"}
            args.pop(k, None)
        elif not isinstance(args[k], _TYPES[ty]) or (ty == "int" and isinstance(args[k], bool)):
            return {"ok": False, "error": f"argument {k!r} must be {ty}"}
    try:
        res = tracing.traced(f"db_tool:{name}", as_type="tool")(t.fn)(ctx, **args, **({"_dry": True} if dry_run else {}))  # shows up as a tool span in the call's trace
    except ToolError as e:
        res = {"ok": False, "error": str(e), **e.extra}
    except Exception as e:  # never let a tool crash a live call
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if t.writes and not dry_run:
        write_tools.record_event(ctx.get("call_id"), f"db_tool.{name}", {"args": args, "ok": res.get("ok"), "result": _short(res)})
    return res


class ToolError(Exception):
    """A request the tool refuses (bad value, not allowed). Becomes {"ok": false, "error": ...}; `extra` carries a reason code and
    alternatives so the agent can explain the refusal and offer something that WOULD work."""

    def __init__(self, msg: str, **extra):
        super().__init__(msg)
        self.extra = extra


def _short(res: dict) -> dict:
    return {k: (v if not isinstance(v, (list, dict)) else f"<{len(v)} items>") for k, v in res.items()}


def live_tool_prompt() -> str:
    """The part of the interpreter prompt that tells the model which requests it may raise and EXACTLY what their arguments look like.
    Generated from the registry, so the prompt and the tools cannot drift apart."""
    t = REGISTRY["add_reminder"]
    args = ", ".join(f'"{k}": {ty}' + ("" if req else " (optional)") + f"  // {d}" for k, (ty, req, d) in t.params.items())
    return ("## Requests the parent may make (tools)\n"
            'If the parent clearly asks for something to be put on the calendar, add it to "requests" (a list; usually empty):\n'
            f'  {{"tool": "{t.name}", "args": {{{args}}}, "evidence": "<their exact words>"}}\n'
            f'  - {t.description} Use it for "remind me about my doctor on Friday at 4".\n'
            '  - Work out the date and time from "now_local" in the input. Never guess a time they did not say; if the day or time is missing, leave "requests" empty.\n'
            '  - Only for the PARENT\'s own appointment or thing to remember ("remind ME about..."). Never for a message to the family, a callback, or something you inferred.\n'
            '  - When the reply holds ONLY such a request (no answer to the question), set intent "request".\n'
            '  - "Call me back at 4" / "call me in 20 minutes" is NOT a request: use intent "callback" with callback_minutes, or callback_at "HH:MM" (24h) for a clock time.\n'
            '    If they ALSO answered the question ("yes I ate, call me after my nap around 4"), keep intent "answer" with the answer fields AND set callback_at / callback_minutes.\n')


def tools_text(names: tuple[str, ...]) -> str:
    """The tool list as text for a model prompt: name(args): what it does. Generated from the registry so they cannot drift."""
    out = []
    for n in names:
        t = REGISTRY[n]
        args = ", ".join(f"{k}: {ty}{'' if req else '?'}" for k, (ty, req, _) in t.params.items())
        notes = "; ".join(f"{k} = {d}" for k, (_, _, d) in t.params.items() if d)
        out.append(f"- {n}({args}): {t.description}" + (f" [{notes}]" if notes else "")
                   + ("" if t.result_matters else " (you do not need to wait for the result)"))
    return "\n".join(out)


def _parent(s, ctx) -> Parent:
    p = s.get(Parent, ctx["parent_id"])
    if p is None:
        raise ToolError("unknown parent")
    return p


def _when(parent: Parent, in_minutes: int | None, at: str | None) -> datetime:
    """`at` is parent-local 'YYYY-MM-DD HH:MM' (or ISO with offset); `in_minutes` is relative to now."""
    if (in_minutes is None) == (at is None):
        raise ToolError("give exactly one of in_minutes or at")
    if at is not None and len(at.strip()) <= 5 and ":" in at:  # "16:15" = the next time the clock shows that, parent-local
        try:
            hh, mm = at.strip().split(":")
            tz = ZoneInfo(parent.timezone)
            now_l = datetime.now(tz)
            cand = now_l.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
            return (cand if cand > now_l else cand + timedelta(days=1)).astimezone(timezone.utc)
        except ValueError:
            raise ToolError("at must look like 16:15 or 2026-10-20 16:30")
    if in_minutes is not None:
        if not 0 <= in_minutes <= 60 * 24 * 14:
            raise ToolError("in_minutes must be between 0 and 14 days")
        return datetime.now(timezone.utc) + timedelta(minutes=in_minutes)
    try:
        dt = datetime.fromisoformat(at.replace(" ", "T"))
    except ValueError:
        raise ToolError("at must look like 2026-10-20 16:30")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(parent.timezone))
    return dt.astimezone(timezone.utc)


# ------------------------------------------------------------------ reads
@tool("get_parent_profile", "Who the parent is: name, timezone, whether calls are on, meal times, active medicines, caregiver names.",
      {}, writes=False)
def get_parent_profile(ctx):
    from ..models import Caregiver, ParentCaregiver
    from .routines import effective_meal_times

    with session_scope() as s:
        p = _parent(s, ctx)
        meals = effective_meal_times(s, p.id)
        meds = [{"id": m.id, "name": m.name, "timing": m.timing, "meal": m.meal_anchor}
                for m in s.scalars(select(Medicine).where(Medicine.parent_id == p.id, Medicine.active.is_(True)))]
        cgs = [c.name for c in s.scalars(select(Caregiver).join(ParentCaregiver, ParentCaregiver.caregiver_id == Caregiver.id)
                                         .where(ParentCaregiver.parent_id == p.id))]
        return {"ok": True, "name": p.preferred_name, "timezone": p.timezone, "calling_enabled": p.calling_enabled,
                "meal_times": meals, "medicines": meds, "caregivers": cgs}


@tool("get_today_status", "What has been logged today (parent's local day): meals and medicines, with their status.", {}, writes=False)
def get_today_status(ctx):
    with session_scope() as s:
        p = _parent(s, ctx)
        tz = ZoneInfo(p.timezone)
        start = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        meals = [{"meal": m.meal_type, "status": m.status} for m in s.scalars(
            select(MealLog).where(MealLog.parent_id == p.id, MealLog.created_at >= start).order_by(MealLog.id))]
        meds = [{"medicine_id": m.medicine_id, "status": m.status} for m in s.scalars(
            select(MedicineLog).where(MedicineLog.parent_id == p.id, MedicineLog.created_at >= start).order_by(MedicineLog.id))]
        return {"ok": True, "meals": meals, "medicines": meds}


@tool("get_upcoming_calls", "The parent's calendar: calls still waiting to be made, soonest first.",
      {"limit": ("int", False, "how many to return (default 10)")}, writes=False)
def get_upcoming_calls(ctx, limit: int = 10):
    with session_scope() as s:
        rows = s.scalars(select(CallAttempt).where(CallAttempt.parent_id == ctx["parent_id"], CallAttempt.status.in_(OPEN_STATES))
                         .order_by(CallAttempt.scheduled_at).limit(max(1, min(limit, 50)))).all()
        return {"ok": True, "calls": [{"call_id": c.id, "routine": c.routine.key, "at": as_utc(c.scheduled_at).isoformat(),
                                       "kind": c.kind, "attempt_no": c.attempt_no} for c in rows]}


# ------------------------------------------------------------------ writes: the calendar
@tool("schedule_next_call", "Put a call on the calendar for one of the parent's routines (e.g. 'lunch_checkin'), either in N minutes or at a "
      "parent-local time. Asking twice for the same routine and time does nothing new.",
      {"routine_key": ("str", True, "e.g. lunch_checkin, metformin_reminder"), "in_minutes": ("int", False, "minutes from now"),
       "at": ("str", False, "parent-local 'YYYY-MM-DD HH:MM'"), "reason": ("str", False, "why, for the audit trail")},
      writes=True)
def schedule_next_call(ctx, routine_key: str, in_minutes: int | None = None, at: str | None = None, reason: str = ""):
    with session_scope() as s:
        p = _parent(s, ctx)
        if not p.calling_enabled:
            raise ToolError("calls are switched off for this parent")
        routine = s.scalars(select(Routine).where(Routine.parent_id == p.id, Routine.key == routine_key)).first()
        if routine is None:
            raise ToolError(f"no routine called {routine_key!r}")
        when = shift_out_of_quiet_hours(_when(p, in_minutes, at), p.timezone)
        local = when.astimezone(ZoneInfo(p.timezone))
        occ = occurrence_key(p.id, routine.key, local.date().isoformat(), slot=f"t{local:%H%M}")  # same time again -> same key
        call, new = schedule_attempt(s, parent_id=p.id, routine_id=routine.id, occurrence_key=occ, attempt_no=1, kind="manual",
                                     when=when, prev_attempt_id=ctx.get("call_id"))
        return {"ok": True, "call_id": call.id, "at": as_utc(call.scheduled_at).isoformat(), "already_existed": not new, "reason": reason}


def _own_open_call(s, ctx, call_id: int) -> CallAttempt:
    c = s.get(CallAttempt, call_id)
    if c is None or c.parent_id != ctx["parent_id"]:
        raise ToolError("no such call for this parent")
    if c.status != "scheduled":
        raise ToolError(f"call is {c.status}, only waiting calls can be changed")
    return c


@tool("reschedule_call", "Move a waiting call to a new time.",
      {"call_id": ("int", True, "id from get_upcoming_calls"), "in_minutes": ("int", False, "minutes from now"),
       "at": ("str", False, "parent-local 'YYYY-MM-DD HH:MM'")}, writes=True)
def reschedule_call(ctx, call_id: int, in_minutes: int | None = None, at: str | None = None):
    with session_scope() as s:
        p = _parent(s, ctx)
        c = _own_open_call(s, ctx, call_id)
        c.scheduled_at = shift_out_of_quiet_hours(_when(p, in_minutes, at), p.timezone)
        return {"ok": True, "call_id": c.id, "at": as_utc(c.scheduled_at).isoformat()}


@tool("cancel_call", "Cancel a waiting call.", {"call_id": ("int", True, "id from get_upcoming_calls"), "reason": ("str", False, "why")},
      writes=True)
def cancel_call(ctx, call_id: int, reason: str = ""):
    with session_scope() as s:
        c = _own_open_call(s, ctx, call_id)
        transition(s, c, "cancelled", reason or "cancelled by tool")
        return {"ok": True, "call_id": c.id}


@tool("finish_occurrence", "The job for this reminder is done: cancel any waiting retries of it and put the NEXT day's call on the calendar.",
      {}, writes=True, scope="system")
def finish_occurrence(ctx):
    from ..calls.planner import plan_next_occurrence

    with session_scope() as s:
        call = s.get(CallAttempt, ctx["call_id"])
        if call is None or call.parent_id != ctx["parent_id"]:
            raise ToolError("no such call")
        n = 0
        for c in s.scalars(select(CallAttempt).where(CallAttempt.occurrence_key == call.occurrence_key, CallAttempt.status == "scheduled",
                                                      CallAttempt.id != call.id)):
            transition(s, c, "cancelled", "occurrence finished")
            n += 1
        nxt = plan_next_occurrence(s, call)
        return {"ok": True, "cancelled_waiting": n, "next_call_id": nxt}


@tool("add_reminder", "Create a one-off reminder (e.g. a doctor's appointment) and schedule the call for it.",
      {"message": ("str", True, "what to remind about, in plain words"), "at": ("str", True, "parent-local 'YYYY-MM-DD HH:MM'")},
      writes=True, dry=True)
def add_reminder(ctx, message: str, at: str, _dry: bool = False):
    with session_scope() as s:
        p = _parent(s, ctx)
        when = _when(p, None, at)
        if when < datetime.now(timezone.utc):
            raise ToolError("that time is in the past")
        if not p.calling_enabled:
            raise ToolError("calls are switched off for this parent")
        if _dry:
            from .routines import spoken_time
            loc = when.astimezone(ZoneInfo(p.timezone))
            return {"ok": True, "preview": True, "summary": f"{loc:%A} {loc.day} {loc:%B} {spoken_time(f'{loc:%H:%M}')}",
                    "message": message[:300]}
        dup = s.scalars(select(Reminder).where(Reminder.parent_id == p.id, Reminder.message == message[:300], Reminder.due_at == when)).first()
        if dup:  # same reminder asked twice
            return {"ok": True, "reminder_id": dup.id, "already_existed": True}
        rem = Reminder(parent_id=p.id, message=message[:300], due_at=when)
        s.add(rem)
        s.flush()
        routine = Routine(parent_id=p.id, key=f"reminder_{rem.id}", type="defined_reminder", config={"reminder_id": rem.id})
        s.add(routine)
        s.flush()
        local = when.astimezone(ZoneInfo(p.timezone))
        call, _ = schedule_attempt(s, parent_id=p.id, routine_id=routine.id, attempt_no=1, kind="scheduled", when=when,
                                   occurrence_key=occurrence_key(p.id, routine.key, local.date().isoformat()))
        return {"ok": True, "reminder_id": rem.id, "call_id": call.id, "already_existed": False}


@tool("book_callback", "Book a call back to THIS parent about the reminder in progress, in N minutes or at a parent-local time (e.g. '16:15'). "
      "Booking again during the same call moves the existing booking. Returns the time actually booked (quiet hours can move it).",
      {"in_minutes": ("int", False, "minutes from now"), "at": ("str", False, "parent-local 'HH:MM' or 'YYYY-MM-DD HH:MM'"),
       "reason": ("str", False, "why")}, writes=True)
def book_callback(ctx, in_minutes: int | None = None, at: str | None = None, reason: str = ""):
    from ..calls.outcome import _next_routine_time
    from ..config import get_settings
    from .routines import spoken_time

    cap = get_settings().max_attempts_per_occurrence
    with session_scope() as s:
        p = _parent(s, ctx)
        if not p.calling_enabled:
            raise ToolError("calls are switched off for this parent", reason_code="calls_off")
        call = s.get(CallAttempt, ctx.get("call_id"))
        if call is None or call.parent_id != p.id:
            raise ToolError("not inside a call")
        asked = _when(p, in_minutes, at)
        when = shift_out_of_quiet_hours(asked, p.timezone)
        if call.attempt_no >= cap + 2:
            raise ToolError("already called back several times for this reminder", reason_code="too_many_callbacks")
        upcoming = _next_routine_time(s, call)
        if upcoming is not None and when >= upcoming:
            tz = ZoneInfo(p.timezone)
            soonest = datetime.now(timezone.utc) + timedelta(minutes=2)
            latest_dt = upcoming - timedelta(minutes=10)
            night = abs((when - asked).total_seconds()) > 60  # the asked time fell in quiet hours and was pushed to the morning
            if night:  # the real reason is the night, not the reminder: only a time before quiet hours begins would work
                q_start = get_settings().quiet_start
                latest_ok = shift_out_of_quiet_hours(latest_dt, p.timezone) == latest_dt and latest_dt > soonest
                raise ToolError(f"no calls between {q_start} and {get_settings().quiet_end}", reason_code="quiet_hours",
                                quiet_hours_start_local=q_start, quiet_hours_end_local=get_settings().quiet_end,
                                asked_local=f"{asked.astimezone(tz):%H:%M}",
                                **({"latest_possible_local": f"{latest_dt.astimezone(tz):%H:%M}"} if latest_ok else {}))
            raise ToolError("the next reminder is due before that time", reason_code="next_reminder_first",
                            next_reminder_at_local=f"{upcoming.astimezone(tz):%H:%M}", latest_possible_local=f"{latest_dt.astimezone(tz):%H:%M}",
                            asked_local=f"{asked.astimezone(tz):%H:%M}")
        existing = s.scalars(select(CallAttempt).where(CallAttempt.prev_attempt_id == call.id, CallAttempt.kind == "callback",
                                                        CallAttempt.status == "scheduled")).first()
        if existing is not None:  # "actually, make it later": move the booking, never double-book
            existing.scheduled_at, row, reused = when, existing, True
        else:
            row, _ = schedule_attempt(s, parent_id=p.id, routine_id=call.routine_id, occurrence_key=call.occurrence_key or f"adhoc:{call.id}",
                                      attempt_no=call.attempt_no + 1, kind="callback", when=when, prev_attempt_id=call.id, channel=call.channel)
            reused = False
        loc = when.astimezone(ZoneInfo(p.timezone))
        now_l = datetime.now(ZoneInfo(p.timezone))
        mins = max(0, round((when - datetime.now(timezone.utc)).total_seconds() / 60))
        cb = s.scalars(select(Callback).where(Callback.call_id == call.id, Callback.status == "pending")).first()
        if cb is not None:  # changed their mind: update the same promise, never a second one
            cb.delay_minutes, cb.reason = mins, (reason or cb.reason)
        else:
            s.add(Callback(parent_id=p.id, call_id=call.id, routine_id=call.routine_id, delay_minutes=mins, reason=reason or "call back"))
        return {"ok": True, "call_id": row.id, "key": row.idempotency_key, "at_utc": as_utc(when).isoformat(), "in_minutes": mins,
                "at_local": f"{loc:%H:%M}", "spoken": spoken_time(f"{loc:%H:%M}"), "tomorrow": loc.date() != now_l.date(),
                "shifted": abs((when - asked).total_seconds()) > 60, "by_clock": at is not None, "reused": reused, "reason": reason}


# ------------------------------------------------------------------ writes: facts the agent learned in the conversation
def _norm(s: str) -> str:
    return _re.sub(r"\s+", " ", _re.sub(r"[^\w\s:']", " ", (s or "").lower())).strip()


def _quoted(ctx: dict, evidence: str) -> bool:
    """Data integrity, not interpretation: a fact may only be logged with a quote of what the parent really said in THIS call
    (pieces joined by "..." are checked piece by piece). The model decides what the words mean; this stops invented quotes."""
    from ..models import TranscriptTurn

    pieces = [_norm(x) for x in _re.split(r"\.{3}|\u2026", evidence or "") if _norm(x)]
    if not pieces:
        return False
    with session_scope() as s:
        said = [_norm(t.text) for t in s.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == ctx.get("call_id"),
                                                                              TranscriptTurn.speaker == "parent"))]
    return all(any(p in turn for turn in said) for p in pieces)


def _need_quote(ctx: dict, evidence: str, allow_empty: bool = False):
    if not evidence:
        if allow_empty:
            return
        raise ToolError("evidence is required: quote the parent's exact words", reason_code="evidence_missing")
    if not _quoted(ctx, evidence):
        raise ToolError("evidence must be words the parent actually said in this call", reason_code="evidence_not_found")


@tool("log_meal", "Record whether the parent has had a meal. status: eaten | skipped | not_yet | unclear. items = the foods they named "
      "(empty if none). evidence = their exact words.",
      {"meal_type": ("str", True, "breakfast, lunch, snack or dinner"), "status": ("str", True, "eaten, skipped, not_yet or unclear"),
       "items": ("list", False, "foods they named, e.g. [\"dal\", \"rice\"]"), "evidence": ("str", False, "their exact words")},
      writes=True, result_matters=False)
def log_meal(ctx, meal_type: str, status: str, items: list | None = None, evidence: str = ""):
    if meal_type not in MEALS:
        raise ToolError(f"meal_type must be one of {MEALS}")
    if status not in ("eaten", "skipped", "not_yet", "unclear"):
        raise ToolError("status must be eaten, skipped, not_yet or unclear")
    _need_quote(ctx, evidence, allow_empty=status == "unclear")
    items = [str(i).strip() for i in (items or []) if str(i).strip()]
    with session_scope() as s:
        last = s.scalars(select(MealLog).where(MealLog.call_id == ctx["call_id"], MealLog.meal_type == meal_type)
                         .order_by(MealLog.id.desc())).first()
        if last is not None and last.status == status and list(last.items or []) == items:
            return {"ok": True, "already_logged": True}
    write_tools.log_meal(ctx, meal_type, status, items, evidence, 0.9)
    return {"ok": True}


@tool("log_medicine", "Record whether the parent has taken a medicine. status: taken | not_taken | unclear. evidence = their exact words.",
      {"medicine_id": ("int", True, "the medicine's id from the call details"), "status": ("str", True, "taken, not_taken or unclear"),
       "evidence": ("str", False, "their exact words")}, writes=True, result_matters=False)
def log_medicine(ctx, medicine_id: int, status: str, evidence: str = ""):
    if status not in ("taken", "not_taken", "unclear"):
        raise ToolError("status must be taken, not_taken or unclear")
    _need_quote(ctx, evidence, allow_empty=status == "unclear")
    with session_scope() as s:
        med = s.get(Medicine, medicine_id)
        if med is None or med.parent_id != ctx["parent_id"]:
            raise ToolError("no such medicine for this parent")
        last = s.scalars(select(MedicineLog).where(MedicineLog.call_id == ctx["call_id"], MedicineLog.medicine_id == medicine_id)
                         .order_by(MedicineLog.id.desc())).first()
        if last is not None and last.status == status:
            return {"ok": True, "already_logged": True}
    write_tools.log_medicine(ctx, medicine_id, status, evidence, 0.9)
    return {"ok": True}


@tool("ack_reminder", "Record whether the parent acknowledged (heard and understood) a reminder you read to them.",
      {"reminder_id": ("int", True, "the reminder's id from the call details"), "acknowledged": ("bool", True, "true if they confirmed"),
       "evidence": ("str", False, "their exact words")}, writes=True, result_matters=False)
def ack_reminder(ctx, reminder_id: int, acknowledged: bool, evidence: str = ""):
    _need_quote(ctx, evidence, allow_empty=not acknowledged)
    from ..models import ReminderAck

    with session_scope() as s:
        last = s.scalars(select(ReminderAck).where(ReminderAck.call_id == ctx["call_id"], ReminderAck.reminder_id == reminder_id)
                         .order_by(ReminderAck.id.desc())).first()
        if last is not None and last.acknowledged == acknowledged:
            return {"ok": True, "already_logged": True}
    write_tools.ack_reminder(ctx, reminder_id, acknowledged, evidence)
    return {"ok": True}


@tool("raise_alert", "Tell the family something they should know. severity: emergency (urgent danger, call ends) | warning (health or "
      "mood worry, a missed medicine) | info (logistics, a request you could not meet). One alert per matter.",
      {"severity": ("str", True, "emergency, warning or info"), "reason": ("str", True, "one plain sentence for the family"),
       "evidence": ("str", False, "their exact words")}, writes=True, result_matters=False)
def raise_alert(ctx, severity: str, reason: str, evidence: str = ""):
    if severity not in ("emergency", "warning", "info"):
        raise ToolError("severity must be emergency, warning or info")
    if evidence and not _quoted(ctx, evidence):
        evidence = ""  # a safety message to the family is never refused over a quote; unverified supporting words are just dropped
    from ..models import Alert

    with session_scope() as s:
        dup = s.scalars(select(Alert).where(Alert.call_id == ctx["call_id"], Alert.severity == severity, Alert.reason == reason[:300])).first()
        if dup is not None:
            return {"ok": True, "already_raised": True}
    write_tools.raise_alert(ctx, severity, reason[:300], evidence)
    return {"ok": True}


@tool("relay_message", "Pass a message from the parent to the family, in the parent's own words.",
      {"text": ("str", True, "what the parent wants the family to know")}, writes=True, result_matters=False)
def relay_message(ctx, text: str):
    write_tools.relay_message(ctx, text[:300])
    return {"ok": True}


# ------------------------------------------------------------------ writes: profile and switches
@tool("update_meal_time", "Fix a meal time for this parent (the CARETAKER's decision; calls follow it). Not available to the agent.",
      {"meal": ("str", True, "breakfast, lunch, snack or dinner"), "time": ("str", True, "HH:MM, 24-hour"),
       "evidence": ("str", False, "why / who asked")}, writes=True, scope="admin")
def update_meal_time(ctx, meal: str, time: str, evidence: str = ""):
    if meal not in MEALS:
        raise ToolError(f"meal must be one of {MEALS}")
    try:
        h, m = time.split(":")
        assert 0 <= int(h) < 24 and 0 <= int(m) < 60 and len(m) == 2
    except Exception:
        raise ToolError("time must be HH:MM")
    write_tools.set_meal_time(ctx, meal, f"{int(h):02d}:{m}", evidence or "set by the caretaker", source="caregiver")
    return {"ok": True, "meal": meal, "time": f"{int(h):02d}:{m}"}


@tool("pause_calls", "Switch calls off for this parent and cancel everything waiting (e.g. they asked us to stop).",
      {"reason": ("str", True, "why")}, writes=True, scope="system")
def pause_calls(ctx, reason: str):
    from ..calls.outcome import cancel_pending

    with session_scope() as s:
        p = _parent(s, ctx)
        p.calling_enabled = False
        n = cancel_pending(s, p.id, reason)
        return {"ok": True, "cancelled": n}


@tool("resume_calls", "Switch calls back on and plan today's calls. Only a human operator may do this.", {}, writes=True, scope="admin")
def resume_calls(ctx):
    from ..calls.planner import plan_day

    with session_scope() as s:
        _parent(s, ctx).calling_enabled = True
    return {"ok": True, "planned": len(plan_day(parent_id=ctx["parent_id"]))}


@tool("notify_caregiver", "Send a message to the parent's family (son/daughter).",
      {"message": ("str", True, "the text"), "severity": ("str", False, "info or alert")}, writes=True)
def notify_caregiver(ctx, message: str, severity: str = "info"):
    if severity not in ("info", "alert"):
        raise ToolError("severity must be info or alert")
    with session_scope() as s:
        n = notify_caregivers(s, ctx["parent_id"], ctx.get("call_id"), message[:500], "alert" if severity == "alert" else "summary")
        return {"ok": True, "notified": n}
