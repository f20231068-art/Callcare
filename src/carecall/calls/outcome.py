"""After ANY call ends: what was it, and what happens next?

    signals (call status, how the conversation ended, who spoke last, was the parent mid-word when the line dropped...)
        -> classify -> OUTCOME -> follow-up call (or none) + a message to the family (or none)

Outcomes
  completed            all done                              -> nothing more today (the planner already has the next occurrence)
  callback_requested   the parent said "call me in 20 min"    -> call back then          (key ...:cb<n>)
  cut_off              the line dropped mid-conversation      -> call back in ~90 s      (key ...:a<n>), max 2, resumes where it stopped
  hung_up_on_purpose   "I have to go, bye" before it was done -> try once more in 30 min
  never_connected      no answer / busy                       -> retry after 10, 30, 90 min, then tell the family
  technical_failure    Twilio / our side failed               -> retry after 5, 15 min, then tell the family
  emergency            the agent escalated                    -> NO retry; the family was alerted in the call
  off_topic_limit      three off-topic requests               -> no retry; the family was told
  opted_out            "stop calling me"                      -> calling is switched off for this parent, pending calls cancelled
  in_flight            still dialing / ringing / in progress  -> not decided yet

Deciding "cut off or hung up on purpose" is the one fuzzy judgement. Rules decide the clear cases (the parent was mid-speech when the
line dropped; the agent's question was left hanging; or the last words were a goodbye). Only when the signals conflict is a
model asked, and if it is unavailable we choose cut_off: ringing back too soon is a smaller mistake than leaving a parent whose
line dropped uncalled.

Idempotent: the outcome is stored on the call and the follow-up has a computed key (calls/keys.py), so running this twice
(a crash, a repeated webhook) schedules exactly one follow-up."""
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from .. import tracing
from ..agent import tools
from ..config import get_settings
from ..db import session_scope
from ..llm.base import LLMClient, complete_json
from ..models import Alert, Callback, CallAttempt, Event, Parent, TranscriptTurn, as_utc
from .service import schedule_attempt, shift_out_of_quiet_hours
from .state import transition

OPT_OUT = re.compile(r"\b(stop (calling|ringing|phoning)|don'?t (call|ring|phone) me|do not (call|ring|phone)|never (call|ring) me|"
                     r"remove my number|take me off|no more calls)\b", re.I)
GOODBYE = re.compile(r"\b(bye|goodbye|have to go|got to go|gotta go|need to go|call (me )?later|not now|talk later|busy now|"
                     r"i am busy|i'm busy)\b", re.I)
TRAILING_WORD = re.compile(r"\b(and|but|so|because|then|the|a|an|to|of|my|i|is|was|with|for|or|that|which|um|uh)\W*$", re.I)


@dataclass
class FollowUp:
    kind: str  # retry | callback
    delay_s: int
    reason: str
    quiet_hours: bool = True  # False for "ring back now" cases: the parent was just on the phone


@dataclass
class Outcome:
    kind: str
    reason: str
    follow_up: FollowUp | None = None
    notify: str | None = None  # a message for the family (we gave up, or they need to know)
    notify_severity: str = "warning"
    pause_calls: bool = False
    judged_by: str = "rules"


# ---------------------------------------------------------------- signals
def gather(session, call: CallAttempt) -> dict:
    fin = session.scalars(select(Event).where(Event.call_id == call.id, Event.type == "call.finished")
                          .order_by(Event.id.desc())).first()
    turns = session.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == call.id).order_by(TranscriptTurn.turn_no)).all()
    pending_cb = session.scalars(select(Callback).where(Callback.call_id == call.id, Callback.status == "pending")).all()
    return {
        "status": call.status, "end_reason": call.end_reason, "attempt_no": call.attempt_no, "fin": (fin.payload if fin else {}),
        "turns": [(t.speaker, t.text) for t in turns], "callbacks": [(c.id, c.delay_minutes) for c in pending_cb],
        "connected": call.started_at is not None,
        "duration_s": int((as_utc(call.ended_at) - as_utc(call.started_at)).total_seconds()) if call.started_at and call.ended_at else None,
    }


# ---------------------------------------------------------------- the fuzzy judgement
def cut_off_evidence(sig: dict) -> tuple[bool, bool]:
    """(evidence the line dropped mid-conversation, evidence the parent chose to end it)."""
    fin, turns = sig["fin"], sig["turns"]
    parent = [t for s, t in turns if s == "parent"]
    last_parent = parent[-1] if parent else ""
    cut = bool(fin.get("parent_speaking_at_hangup")) or bool(fin.get("pending_stt_text"))
    if turns and turns[-1][0] == "agent" and turns[-1][1].strip().endswith("?") and not fin.get("done"):
        cut = True  # we asked, and the answer never came
    if last_parent and TRAILING_WORD.search(last_parent):
        cut = True  # "...and then I" is not a finished sentence
    if not parent and sig["connected"] and (sig["duration_s"] or 0) < 20:
        cut = True  # picked up and dropped straight away
    purposeful = bool(last_parent and GOODBYE.search(last_parent))
    return cut, purposeful


JUDGE_SYSTEM = """A phone assistant was talking to an older person when the call ended before the check-in was finished.
Decide whether the line DROPPED / the person was CUT OFF by accident, or whether they ENDED it ON PURPOSE.
Output one JSON object: {"verdict": "cut_off" | "on_purpose", "confidence": 0..1, "reason": "one sentence"}.
Clues for cut_off: the person was mid-sentence, the last words trail off, no goodbye, a question left unanswered.
Clues for on_purpose: a goodbye, "I have to go", "not now", irritation, a clear refusal to continue."""


def judge_hangup(client: LLMClient | None, sig: dict) -> tuple[str, str, str]:
    """Ask the model only when the rules conflict. Returns (verdict, reason, judged_by)."""
    if client is None:
        return "cut_off", "signals conflict and no model available: ring back (the safer mistake)", "rules"
    last = sig["turns"][-6:]
    user = json.dumps({"last_turns": [{"speaker": s, "text": t} for s, t in last], "parent_speaking_when_line_dropped":
                       sig["fin"].get("parent_speaking_at_hangup"), "unfinished_sentence_heard": sig["fin"].get("pending_stt_text"),
                       "seconds_connected": sig["duration_s"], "steps_done": sig["fin"].get("steps_resolved"),
                       "steps_total": sig["fin"].get("steps_total")}, ensure_ascii=False)
    try:
        data = complete_json(client, JUDGE_SYSTEM, user, purpose="hangup_judge")
        verdict = "on_purpose" if data.get("verdict") == "on_purpose" and float(data.get("confidence", 0)) >= 0.6 else "cut_off"
        return verdict, str(data.get("reason", ""))[:200], f"llm:{client.model}"
    except Exception as e:  # noqa: BLE001 - a failed judge must not lose the follow-up
        return "cut_off", f"model judge failed ({type(e).__name__}); ringing back", "rules"


# ---------------------------------------------------------------- classification
def _delays(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def classify(session, call: CallAttempt, client: LLMClient | None = None) -> Outcome:
    s = get_settings()
    if call.status in ("scheduled", "dialing", "ringing", "in_progress"):
        return Outcome("in_flight", f"call is {call.status}")
    sig = gather(session, call)
    n = call.attempt_no
    cap = s.max_attempts_per_occurrence
    who = session.get(Parent, call.parent_id).preferred_name

    def give_up(msg: str) -> str:
        return msg.format(name=who, n=n)

    if call.status == "no_answer":
        d = _delays(s.unreachable_delays_min)
        if n <= len(d) and n < cap:
            return Outcome("never_connected", f"no answer on attempt {n}", FollowUp("retry", d[n - 1] * 60, "no answer"))
        return Outcome("never_connected", f"no answer after {n} attempts", notify=give_up("We could not reach {name} after {n} tries today."))
    if call.status == "failed":
        d = _delays(s.technical_delays_min)
        if n <= len(d) and n < cap:
            return Outcome("technical_failure", f"call failed on attempt {n}", FollowUp("retry", d[n - 1] * 60, "technical failure"))
        return Outcome("technical_failure", f"call kept failing ({n} attempts)", notify=give_up("The check-in call to {name} kept failing ({n} tries)."))
    if call.status == "cancelled":
        return Outcome("cancelled", "cancelled before dialing")

    parent_text = " ".join(t for sp, t in sig["turns"] if sp == "parent")
    if call.status in ("completed", "abandoned") and OPT_OUT.search(parent_text):
        return Outcome("opted_out", "the parent asked us to stop calling", pause_calls=True,
                       notify=give_up("{name} asked us to stop calling. Calls are switched off until you turn them back on."))

    if call.status == "completed":
        er = call.end_reason
        if er == "emergency":
            return Outcome("emergency", "the agent escalated an emergency (family already alerted); no retry")
        if er == "off_topic_limit":
            return Outcome("off_topic_limit", "ended after repeated off-topic requests; family told in the call")
        if er == "callback" and sig["callbacks"]:
            mins = min(m for _, m in sig["callbacks"])
            return Outcome("callback_requested", f"parent/agent arranged a call back in {mins} min",
                           FollowUp("callback", mins * 60, "call back as arranged"))
        gaps = sig["fin"].get("unresolved", [])
        return Outcome("completed", "all steps handled" + (f" (unresolved: {', '.join(gaps)})" if gaps else ""))

    # abandoned: the conversation stopped before it finished
    cut, purposeful = cut_off_evidence(sig)
    judged_by = "rules"
    if cut and not purposeful:
        verdict, why = "cut_off", "line dropped mid-conversation"
    elif purposeful and not cut:
        verdict, why = "on_purpose", "the parent said goodbye"
    else:
        verdict, why, judged_by = judge_hangup(client, sig)
    if verdict == "cut_off":
        if n <= s.max_cutoff_retries and n < cap:
            return Outcome("cut_off", why, FollowUp("retry", s.cutoff_retry_delay_s, "ring back after a cut-off call", quiet_hours=False),
                           judged_by=judged_by)
        return Outcome("cut_off", f"{why}; cut off {n} times", notify=give_up("The calls to {name} keep getting cut off ({n} tries). You may want to check on them."),
                       judged_by=judged_by)
    if n == 1 and n < cap:
        return Outcome("hung_up_on_purpose", why, FollowUp("retry", s.hangup_retry_delay_min * 60, "try again later"), judged_by=judged_by)
    return Outcome("hung_up_on_purpose", f"{why}; ended the call again", notify=give_up("{name} ended the check-in call early again."),
                   notify_severity="info", judged_by=judged_by)


# ---------------------------------------------------------------- acting on the outcome
def apply(call_id: int, client: LLMClient | None = None, now: datetime | None = None) -> Outcome:
    """Decide and act. Safe to call twice."""
    now = now or datetime.now(timezone.utc)
    from . import postcall_agent

    facts = None
    with session_scope() as s:  # phase 1 (read only): what do the rules say, and is a model worth asking?
        call = s.get(CallAttempt, call_id)
        if call.outcome:  # already decided: do not decide or schedule again
            return Outcome(call.outcome, call.outcome_reason)
        o = classify(s, call, client)
        if o.kind == "in_flight":
            return o
        if postcall_agent.eligible(call, o, client):
            facts = postcall_agent.build_facts(s, call, o, now)
    if facts:  # the agent's tools open their own sessions, so this must run outside phase 1
        o = postcall_agent.decide(facts, o, call_id, client)
    with session_scope() as s:  # phase 2: record the decision and act on it (the only place the schedule is written)
        call = s.get(CallAttempt, call_id)
        if call.outcome:  # someone else decided while we were thinking
            return Outcome(call.outcome, call.outcome_reason)
        call.outcome, call.outcome_reason = o.kind, o.reason
        tools.add_event(s, call_id, "outcome.decided", {"outcome": o.kind, "reason": o.reason, "judged_by": o.judged_by,
                                                        "follow_up": o.follow_up.__dict__ if o.follow_up else None})
        tracing.update_span(metadata={"outcome": o.kind})
        parent = s.get(Parent, call.parent_id)
        if o.pause_calls:
            parent.calling_enabled = False
            cancel_pending(s, parent.id, f"opted out via call {call_id}")
        live_fu = s.scalars(select(CallAttempt).where(CallAttempt.prev_attempt_id == call_id, CallAttempt.status == "scheduled")).first()
        if live_fu is not None:  # the agent already booked it during the call (and told the parent): keep that, add nothing
            tools.add_event(s, call_id, "followup.already_scheduled", {"next_call_id": live_fu.id, "key": live_fu.idempotency_key,
                                                                       "booked": "live, during the call", "kind": live_fu.kind})
            for cb_id, _ in gather(s, call)["callbacks"]:
                s.get(Callback, cb_id).status = "scheduled"
            o.follow_up = None
        if o.follow_up and parent.calling_enabled:
            fu = o.follow_up
            when = now + timedelta(seconds=fu.delay_s)
            if fu.quiet_hours:
                when = shift_out_of_quiet_hours(when, parent.timezone)
            upcoming = _next_routine_time(s, call)
            if fu.kind == "retry" and upcoming is not None and when >= upcoming:
                # "Don't keep reminding about lunch until dinner": the next thing is due before this retry would be, so move on.
                tools.add_event(s, call_id, "followup.skipped", {"reason": "next routine is due first", "retry_at": when.isoformat(),
                                                                 "next_routine_at": upcoming.isoformat()})
                o.follow_up = None
                if not o.notify:
                    who = parent.preferred_name
                    notify_caregivers_later = f"We could not complete the {call.routine.key.replace('_', ' ')} with {who}; moving on to the next reminder."
                    o.notify, o.notify_severity = notify_caregivers_later, "info"
        if o.follow_up and parent.calling_enabled:
            fu = o.follow_up
            nxt, created = schedule_attempt(s, parent_id=call.parent_id, routine_id=call.routine_id,
                                            occurrence_key=call.occurrence_key or f"adhoc:{call.id}", attempt_no=call.attempt_no + 1,
                                            kind=fu.kind, when=when, prev_attempt_id=call.id, channel=call.channel)
            tools.add_event(s, call_id, "followup.scheduled" if created else "followup.already_scheduled",
                            {"next_call_id": nxt.id, "key": nxt.idempotency_key, "at": when.isoformat(), "kind": fu.kind})
            for cb_id, _ in gather(s, call)["callbacks"]:
                s.get(Callback, cb_id).status = "scheduled"
        if o.follow_up is None and live_fu is None and parent.calling_enabled and o.kind not in ("cancelled", "opted_out"):
            from .planner import plan_next_occurrence  # this occurrence is finished (done or given up): put tomorrow's on the calendar

            nxt_id = plan_next_occurrence(s, call, now)
            if nxt_id:
                tools.add_event(s, call_id, "next_occurrence.scheduled", {"call_id": nxt_id})
        if o.notify:
            from .notify import notify_caregivers
            notify_caregivers(s, call.parent_id, call_id, o.notify, "alert" if o.notify_severity == "warning" else "summary")
    return o


def _next_routine_time(session, call: CallAttempt):
    """When the parent's next DIFFERENT planned call is due (None if nothing else is planned today)."""
    q = (select(CallAttempt.scheduled_at).where(CallAttempt.parent_id == call.parent_id, CallAttempt.status == "scheduled",
                                                 CallAttempt.kind == "scheduled", CallAttempt.occurrence_key != call.occurrence_key,
                                                 CallAttempt.scheduled_at > call.scheduled_at).order_by(CallAttempt.scheduled_at).limit(1))
    t = session.scalars(q).first()
    return as_utc(t) if t else None


def cancel_pending(session, parent_id: int, reason: str) -> int:
    """Cancel every not-yet-dialed attempt for a parent (opt-out, pause)."""
    n = 0
    for c in session.scalars(select(CallAttempt).where(CallAttempt.parent_id == parent_id, CallAttempt.status == "scheduled")):
        transition(session, c, "cancelled", reason)
        n += 1
    return n
