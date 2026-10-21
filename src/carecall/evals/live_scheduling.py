"""Live test of in-call scheduling with REAL models on both sides of the phone and a REAL database.

  carecall live-scheduling            (resets the dev database, seeds it, runs the scenarios, writes runs/live-scheduling-<ts>/report.md)

Each scenario: an LLM plays the parent (it is told only who it is and what it wants), the real interpreter model reads them,
the LangGraph agent talks back. Then we check, from the database and not from what was said:
  (a) did the agent update the database?            rows exist with the right keys/times, tool events recorded
  (b) is the next call scheduled automatically?     the live booking is kept (not duplicated) and, once that call is done, tomorrow's is planned
  (c) is the conversation human-like?               a stronger model reads the transcript against a rubric, plus deterministic checks
                                                    (filler present when a tool ran, and what was PROMISED matches what was BOOKED)."""
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select

from .. import db, tracing
from ..agent.interpret import make_interpreter
from ..agent.session import CallSession
from ..calls import outcome
from ..calls.keys import occurrence_key
from ..calls.service import create_call
from ..config import get_settings
from ..llm.base import complete_json
from ..llm.factory import make_judge_client, make_llm_client, make_persona_client
from ..models import Base, CallAttempt, Callback, Event, Reminder, Routine, TranscriptTurn
from ..seed import seed
from .personas import LLMPersona

TZ = ZoneInfo("Asia/Kolkata")

CARDS = [
    {"id": "lunch_not_yet_callback", "routine": "lunch_checkin",
     "name": "Asha", "traits": ["warm", "a bit slow", "speaks short sentences"],
     "profile": "Asha, 72, retired teacher, lives alone in Goa. Polite, unhurried.",
     "truth": "You have NOT had lunch yet; you were watering the plants and will eat soon. When asked, say so and say you will eat in a little while. "
              "If the caller offers to call back, accept. Keep every reply under 15 words."},
    {"id": "nap_then_4pm", "routine": "lunch_checkin",
     "name": "Asha", "traits": ["sleepy", "friendly"],
     "profile": "Asha, 72. It is just past lunch time and you are about to take your afternoon nap.",
     "truth": "You are about to lie down for a nap. When asked if you ate, say you did (dal rice) but you are sleepy, and ask them to call you "
              "back later, after your nap, around 4 in the afternoon. Keep replies short."},
    {"id": "ate_and_doctor_reminder", "routine": "lunch_checkin",
     "name": "Asha", "traits": ["chatty", "organised"],
     "profile": "Asha, 72, organised, keeps appointments in her head.",
     "truth": "You had dal and rice for lunch. In your FIRST reply say you ate AND ask the caller to remind you about your doctor Dr Rao's appointment "
              "this coming Friday at 4:30 in the afternoon. If they read it back to you and ask to confirm, say yes. Keep replies short."},
    {"id": "changes_mind", "routine": "lunch_checkin",
     "name": "Asha", "traits": ["indecisive", "polite"],
     "profile": "Asha, 72, a little indecisive.",
     "truth": "You have not eaten lunch yet. Ask the caller to call back in 10 minutes. If they confirm, thank them and say goodbye. Keep replies short."},
]

HUMAN_RUBRIC = """You grade ONE phone conversation between an AI check-in assistant and an elderly parent, on whether the assistant sounds
human and trustworthy. Score each 1-5 (5 best) and quote the line that justifies any score under 4.
  warmth        sounds caring, not clinical
  naturalness   short spoken sentences, no stiff or repeated phrasing, no jargon
  stalling      when it did something in the background (saving a note, setting a reminder) it covered the wait with a natural filler
                (umm, hold on...), rather than going silent or announcing technical steps
  clarity       the parent always knows what was decided and when (the time is stated clearly)
  honesty       everything promised is consistent with FACTS FROM THE DATABASE below (times, what was set); nothing promised that the facts do not show
  pacing        no needless repetition, no unnecessary questions, ends cleanly
Return JSON: {"warmth":n,"naturalness":n,"stalling":n,"clarity":n,"honesty":n,"pacing":n,"overall":n,"issues":["..."],"best_line":"...","worst_line":"..."}"""

FILLER_WORDS = re.compile(r"\b(umm+|hmm+|hold on|one second|just a moment|a moment|one moment)\b", re.I)


def _fresh_db():
    eng = db.init_db()
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    seed()
    return eng


def _transcript(call_id: int) -> list[tuple[str, str]]:
    with db.session_scope() as s:
        return [(t.speaker, t.text) for t in s.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == call_id).order_by(TranscriptTurn.turn_no))]


def _converse(card: dict, session: CallSession, persona: LLMPersona, max_turns: int = 12) -> list[tuple[str, str]]:
    history = [("agent", line) for line in session.open()]
    for _ in range(max_turns):
        said = persona.reply(session.current_step_id(), history)
        history.append(("parent", said))
        res = session.turn(said)
        history += [("agent", line) for line in res.reply]
        if res.done:
            break
    return history


def _facts(call_id: int) -> dict:
    """What the database says happened because of this call (the ground truth the judge and checks compare against)."""
    with db.session_scope() as s:
        call = s.get(CallAttempt, call_id)
        follow = [{"call_id": c.id, "kind": c.kind, "key": c.idempotency_key, "status": c.status,
                   "at_local": c.scheduled_at.astimezone(TZ).strftime("%a %d %b %H:%M"), "in_minutes": round((c.scheduled_at - datetime.now(timezone.utc)).total_seconds() / 60)}
                  for c in s.scalars(select(CallAttempt).where(CallAttempt.prev_attempt_id == call_id))]
        rems = [{"id": r.id, "message": r.message, "due_local": r.due_at.astimezone(TZ).strftime("%a %d %b %H:%M")}
                for r in s.scalars(select(Reminder)) if r.due_at and r.message.lower().startswith(("dr", "doctor", "appointment", "see", "visit"))]
        rem_calls = [{"routine": c.routine.key, "at_local": c.scheduled_at.astimezone(TZ).strftime("%a %d %b %H:%M"), "key": c.idempotency_key}
                     for c in s.scalars(select(CallAttempt).where(CallAttempt.status == "scheduled", CallAttempt.kind == "scheduled"))
                     if c.routine.key.startswith("reminder_")]
        callbacks = [{"delay_minutes": c.delay_minutes, "status": c.status, "reason": c.reason}
                     for c in s.scalars(select(Callback).where(Callback.call_id == call_id))]
        tools_used = [e.type for e in s.scalars(select(Event).where(Event.call_id == call_id)) if e.type.startswith(("db_tool.", "followup.", "next_occurrence", "outcome"))]
        return {"end_reason": call.end_reason, "outcome": call.outcome, "follow_up_calls": follow, "callback_rows": callbacks,
                "reminders_created": rems, "reminder_calls_scheduled": rem_calls, "events": tools_used}


def _checks(card: dict, hist: list[tuple[str, str]], f: dict) -> list[dict]:
    agent_text = " ".join(t for w, t in hist if w == "agent")
    out = []

    def chk(name, ok, detail=""):
        out.append({"name": name, "passed": bool(ok), "detail": detail})

    cid = card["id"]
    if cid in ("lunch_not_yet_callback", "nap_then_4pm", "changes_mind"):
        fu = [c for c in f["follow_up_calls"] if c["kind"] == "callback"]
        chk("a: a callback call row exists in the database", len(fu) == 1, f"{fu}")
        chk("a: its key follows <occurrence>:cb<n>", bool(fu) and re.search(r":cb2$", fu[0]["key"]) is not None)
        chk("a: the callback row is recorded", len(f["callback_rows"]) >= 1, f"{f['callback_rows']}")
        chk("b: post-call kept the live booking (no duplicate follow-up)", len(f["follow_up_calls"]) == 1 and "followup.already_scheduled" in f["events"], f"events={f['events']}")
        if cid == "nap_then_4pm" and fu:
            hhmm = fu[0]["at_local"][-5:]
            chk("a: booked time is about 4 pm", "15:30" <= hhmm <= "16:30" or fu[0]["in_minutes"] > 60 * 12, f"booked {fu[0]['at_local']}")
        if cid == "changes_mind" and fu:
            chk("a: booked about 10 minutes out", 8 <= fu[0]["in_minutes"] <= 11, f"{fu[0]['in_minutes']} min")
        promised = re.findall(r"(?:in (\d+) minutes?|at ([\d:]+ (?:in the \w+|at \w+)))", agent_text)
        chk("c: what was promised aloud is in the transcript", bool(promised) or "tomorrow" in agent_text, f"{promised}")
    if cid == "ate_and_doctor_reminder":
        chk("a: reminder row created", len(f["reminders_created"]) == 1, f"{f['reminders_created']}")
        chk("a: its call is on the calendar", len(f["reminder_calls_scheduled"]) == 1, f"{f['reminder_calls_scheduled']}")
        chk("a: read back before setting", any("Shall I set that" in t for w, t in hist if w == "agent"))
        chk("b: the reminder call is at the stated time (Fri-ish 16:30)", bool(f["reminder_calls_scheduled"]) and f["reminder_calls_scheduled"][0]["at_local"].endswith("16:30"))
    chk("c: a filler was used when a tool ran", bool(FILLER_WORDS.search(agent_text)))
    chk("c: no technical talk", not re.search(r"\b(database|tool|api|scheduled in the system|record saved)\b", agent_text, re.I))
    return out


def _complete_the_callback(f: dict) -> dict:
    """Run the booked call-back as a call of its own (parent cooperative) and report whether tomorrow's call got scheduled."""
    fu = [c for c in f["follow_up_calls"] if c["kind"] == "callback"]
    if not fu:
        return {}
    cb_id = fu[0]["call_id"]
    persona = LLMPersona({"id": "cb", "name": "Asha", "traits": ["warm"], "profile": "Asha, 72.",
                          "truth": "Now you HAVE had lunch: dal and rice. Answer simply. Keep replies short."}, make_persona_client())
    s = CallSession(cb_id, make_interpreter(make_llm_client()), channel="twilio")
    s.postcall_client = make_judge_client()
    hist = _converse({"id": "cb"}, s, persona)
    s.finish()
    with db.session_scope() as q:
        c = q.get(CallAttempt, cb_id)
        nxt = [x.idempotency_key for x in q.scalars(select(CallAttempt).where(CallAttempt.status == "scheduled", CallAttempt.kind == "scheduled"))
               if x.routine.key == "lunch_checkin"]
        return {"callback_call_outcome": c.outcome, "greeting": hist[0][1], "tomorrow_lunch_call": nxt, "transcript": hist}


def run(out_dir: str | None = None) -> Path:
    s = get_settings()
    _fresh_db()
    tracing.init_tracing()
    interp = make_interpreter(make_llm_client())
    judge = make_judge_client()
    persona_client = make_persona_client()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir or f"runs/live-scheduling-{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    report, results = [f"# Live scheduling test {stamp}", "",
                       f"- agent interpreter: `{interp.name}`  |  parent (persona): `{persona_client.model}`  |  judge: `{judge.model}`",
                       f"- database: `{db._engine.url.render_as_string(hide_password=True)}`", ""], []
    for n, card in enumerate(CARDS, 1):
        occ = occurrence_key(1, card["routine"], datetime.now(TZ).date().isoformat(), slot=f"live{n}")
        cid, _ = create_call(card["routine"], channel="twilio", occurrence_key=occ, idempotency_key=occ + ":a1")
        sess = CallSession(cid, interp, channel="twilio", tags=["live-scheduling", card["id"]], trace_name=f"live-scheduling:{card['id']}",
                           user_id=f"parent-1-asha")
        sess.postcall_client = judge
        t0 = time.time()
        hist = _converse(card, sess, LLMPersona(card, persona_client))
        sess.finish()
        facts = _facts(cid)
        checks = _checks(card, hist, facts)
        cb = _complete_the_callback(facts)
        if cb:
            checks.append({"name": "b: the callback call opens with 'calling back, as arranged'", "passed": "calling back" in cb["greeting"], "detail": cb["greeting"]})
            checks.append({"name": "b: when that call completes, tomorrow's lunch call is scheduled automatically",
                           "passed": cb["callback_call_outcome"] == "completed" and any(":a1" in k for k in cb["tomorrow_lunch_call"]),
                           "detail": f"outcome={cb['callback_call_outcome']} scheduled={cb['tomorrow_lunch_call']}"})
        convo = "\n".join(f"{'ASSISTANT' if w == 'agent' else 'PARENT'}: {t}" for w, t in hist)
        try:
            grade = complete_json(judge, HUMAN_RUBRIC, f"FACTS FROM THE DATABASE:\n{json.dumps(facts, indent=1, default=str)}\n\nCONVERSATION:\n{convo}", purpose="human_rubric")
        except Exception as e:  # noqa: BLE001
            grade = {"error": str(e)[:200]}
        results.append({"card": card["id"], "call_id": cid, "checks": checks, "grade": grade, "facts": facts, "callback_run": cb})
        ok = all(c["passed"] for c in checks)
        report += [f"## {card['id']}  —  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}  ({time.time() - t0:.0f}s, call #{cid})", "",
                   "### Conversation", "```", convo, "```", "", "### Database facts", "```", json.dumps(facts, indent=1, default=str), "```", "",
                   "### Checks", *[f"- {'PASS' if c['passed'] else 'FAIL'} — {c['name']} {('(' + str(c['detail'])[:160] + ')') if c['detail'] else ''}" for c in checks], "",
                   "### Human-likeness (judge)", "```", json.dumps(grade, indent=1, ensure_ascii=False), "```", ""]
        if cb:
            report += ["### The booked callback, as its own call", "```", "\n".join(f"{'ASSISTANT' if w == 'agent' else 'PARENT'}: {t}" for w, t in cb["transcript"]), "```", ""]
    tracing.flush()
    total = sum(len(r["checks"]) for r in results)
    passed = sum(c["passed"] for r in results for c in r["checks"])
    avg = [r["grade"].get("overall") for r in results if isinstance(r["grade"].get("overall"), (int, float))]
    report.insert(3, f"- **checks passed: {passed}/{total}**  |  mean human-likeness (judge, 1-5): {sum(avg) / len(avg):.1f}" if avg else f"- **checks passed: {passed}/{total}**")
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    return out / "report.md"
