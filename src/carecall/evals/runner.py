"""Eval runner: scripted parent -> agent (text channel) -> deterministic checks -> LLM note -> files.

Each scenario runs in a fresh in-memory database so scenarios can't contaminate each other.
Outputs under runs/<run_id>/ :  notes/<scenario>.md,  summary.md,  results.json,  run.db (TestNote rows)."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from sqlalchemy import select

from .. import tracing
from ..agent.interpret import Interpreter, detect_emergency, make_interpreter
from ..agent.session import CallSession
from ..calls.service import create_call
from ..db import init_db, session_scope
from ..llm.factory import make_llm_client
from ..models import (
    Alert, Callback, CallAttempt, Event, MealLog, MealProfile, Medicine, MedicineLog, ReminderAck, TestNote,
    TranscriptTurn,
)
from ..seed import seed
from . import checks
from .notes import LLMNoteWriter, NoteWriter, TemplateNoteWriter, max_severity, render_markdown

SCENARIO_DIR = Path(__file__).parent / "scenarios"


def load_scenarios(path: Path | None = None) -> list[dict]:
    out = []
    for f in sorted((path or SCENARIO_DIR).glob("*.yaml")):
        out += yaml.safe_load(f.read_text(encoding="utf-8")) or []
    return out


def _collect(call_id: int, session: CallSession, sc: dict) -> dict:
    """Snapshot everything about the call from the DB into a plain dict."""
    with session_scope() as s:
        names = {m.id: m.name for m in s.scalars(select(Medicine))}
        transcript = [{"turn_no": t.turn_no, "speaker": t.speaker, "text": t.text, "latency_ms": t.latency_ms}
                      for t in s.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == call_id).order_by(TranscriptTurn.turn_no))]
        rec = {
            "scenario": {k: sc.get(k) for k in ("id", "routine", "description", "tags", "usecase", "name", "profile", "traits", "truth")},
            "transcript": transcript,
            "meal_logs": [{"meal_type": l.meal_type, "status": l.status, "items": l.items,
                           "source_utterance": l.source_utterance} for l in s.scalars(select(MealLog))],
            "medicine_logs": [{"medicine": names[l.medicine_id], "status": l.status,
                               "source_utterance": l.source_utterance} for l in s.scalars(select(MedicineLog))],
            "reminder_acks": [{"acknowledged": a.acknowledged, "source_utterance": a.source_utterance}
                              for a in s.scalars(select(ReminderAck))],
            "alerts": [{"severity": a.severity, "reason": a.reason} for a in s.scalars(select(Alert))],
            "callbacks": [{"delay_minutes": c.delay_minutes, "reason": c.reason} for c in s.scalars(select(Callback))],
            "meal_profile": {p.meal_type: p.usual_time for p in s.scalars(select(MealProfile))},
            "meal_times_effective": __import__("carecall.agent.routines", fromlist=["x"]).effective_meal_times(s, 1),  # what the schedule would actually follow
            "call_status": s.get(CallAttempt, call_id).status,
        }
    with session_scope() as s:
        evs = s.scalars(select(Event).where(Event.call_id == call_id,
                                            Event.type.in_(["interpretation", "validation.downgrade"])).order_by(Event.id)).all()
        rec["interpretations"] = [{"type": e.type, **e.payload} for e in evs]
    state = session.graph.get_state(session.config).values
    idx = state.get("idx", 0)
    rec["steps"] = [f"{st['id']}: {'resolved' if i < idx else 'not reached'}"
                    + (" (unresolved)" if st["id"] in state.get("unresolved", []) else "")
                    for i, st in enumerate(state.get("steps", []))]
    rec["end_reason"] = session.end_reason
    rec["emergency_words"] = [bool(detect_emergency(t["text"])) for t in transcript if t["speaker"] == "parent"]
    rec["max_latency_ms"] = max([t["latency_ms"] or 0 for t in transcript], default=0)
    rec["interpreter"] = session.interpreter.name
    rec["trace_id"] = session.trace_id
    rec["trace_url"] = tracing.trace_url(session.trace_id)
    return rec


MAX_PERSONA_TURNS = 26  # a call that needs more than this has gone wrong; hang up


def run_scenario(sc: dict, interpreter: Interpreter, writer: NoteWriter, persona=None, run_id: str = "",
                 suite: str = "eval", agent_mode: str | None = None) -> dict:
    """Run one scenario. Fixed `turns` are replayed; with a `persona` the parent answers each question live."""
    init_db("sqlite:///:memory:")
    seed(meal_profile=sc["routine"] != "meal_discovery", preferred_name=(sc.get("name") or "Asha").split(",")[0].strip())
    call_id, _ = create_call(sc["routine"], channel="eval")
    tags = [f"suite:{suite}", f"scenario:{sc['id']}"] + ([f"usecase:{sc['usecase']}"] if sc.get("usecase") else []) \
        + ([f"persona-mode:{persona.mode}"] if persona else []) + ([f"run:{run_id}"] if run_id else [])
    session = CallSession(call_id, interpreter, channel="eval", tags=tags, mode=agent_mode,
                          trace_name=f"{suite}:{sc['id']}", session_id=f"{run_id}:{sc['id']}" if run_id else None,
                          metadata={"scenario": sc["id"], "usecase": sc.get("usecase", ""), "persona": sc.get("name", ""),
                                    "interpreter": interpreter.name, "run": run_id})
    history: list[tuple[str, str]] = []
    for line in session.open():
        history.append(("agent", line))
    if persona is None:
        for utter in sc["turns"]:
            if session.done:
                break
            session.turn(utter)
    else:
        for _ in range(MAX_PERSONA_TURNS):
            if session.done:
                break
            with session._span("persona_reply"):
                utter = persona.reply(session.current_step_id(), history)
            history.append(("parent", utter))
            res = session.turn(utter)
            history.extend(("agent", line) for line in res.reply)
    with session._span("scenario_card", input={k: sc.get(k) for k in ("name", "profile", "traits", "truth", "usecase", "routine", "expect")}):
        pass  # the persona and its expected outcome, visible at the top of the trace
    session.finish(abandoned=not session.done)
    rec = _collect(call_id, session, sc)
    inv = checks.invariants(rec)
    exp = checks.expectations(rec, sc.get("expect", {}))
    gchecks = []
    if persona is not None and sc.get("guardrail", {}).get("expect"):
        from ..llm.factory import make_judge_client
        from .gnani_judge import build_brief, guardrail_rubric, judge

        jc = make_judge_client()
        if jc is not None:  # noqa: SIM102  # offline (rule interpreter, no LLM) there is nobody to judge a detour; deterministic checks still run
            brief = build_brief(sc) | {"database_facts": {  # what the system really did: a promise that matches these is NOT an invention
                "callbacks_booked_by_tool": [{"in_minutes": c.get("delay_minutes"), "reason": c.get("reason")} for c in rec.get("callbacks", [])],
                "alerts_raised": [a["reason"] for a in rec["alerts"]], "meal_logs": [m["status"] for m in rec["meal_logs"]],
                "medicine_logs": [m["status"] for m in rec["medicine_logs"]], "reminder_acks": [a["acknowledged"] for a in rec["reminder_acks"]]}}
            j = judge(jc, sc, [{"speaker": t["speaker"], "text": t["text"]} for t in rec["transcript"]], guardrail_rubric(sc), brief)
            rec["guardrail_judgement"] = j
            gchecks = [{"name": f"guardrail:{it['id']}", "passed": it["verdict"] in ("pass", "na"),
                        "detail": f"{it['why']} (turns {it['turns']})"} for it in j["items"]]
    rec["checks"] = inv + exp + gchecks
    rec["invariants_ok"] = all(c["passed"] for c in inv)
    rec["outcome_ok"] = all(c["passed"] for c in exp + gchecks)
    rec["passed"] = rec["invariants_ok"] and rec["outcome_ok"]
    rec["turns"] = sum(1 for t in rec["transcript"] if t["speaker"] == "parent")
    rec["unresolved"] = [s for s in rec["steps"] if "(unresolved)" in s]
    note = writer.write(rec)
    rec["note"] = note.model_dump()
    rec["note_markdown"] = render_markdown(rec, note, writer.name)
    rec["severity_max"] = max_severity(note)
    rec["writer"] = writer.name
    with session._span("audit", input={"checks": rec["checks"], "alerts": rec["alerts"], "logs": {"meals": rec["meal_logs"], "medicines": rec["medicine_logs"], "acks": rec["reminder_acks"]}}) as span:
        if span:
            span.update(output={"passed": rec["passed"], "note": rec["note"]})
    t = rec["trace_id"]
    tracing.score(t, "eval_passed", 1.0 if rec["passed"] else 0.0, comment=sc["id"], data_type="BOOLEAN")
    tracing.score(t, "outcome_matches_expectation", 1.0 if rec["outcome_ok"] else 0.0, data_type="BOOLEAN")
    tracing.score(t, "invariants_ok", 1.0 if rec["invariants_ok"] else 0.0, data_type="BOOLEAN")
    tracing.score(t, "parent_turns", float(rec["turns"]), data_type="NUMERIC")
    tracing.score(t, "unresolved_steps", float(len(rec["unresolved"])), data_type="NUMERIC")
    tracing.score(t, "alerts_raised", float(len(rec["alerts"])), data_type="NUMERIC")
    tracing.score(t, "note_overall", note.overall, comment=note.summary[:200], data_type="CATEGORICAL")
    return rec


def run_evals(scenario_ids: list[str] | None = None, out_root: str = "runs", interpreter: Interpreter | None = None,
              writer: NoteWriter | None = None, exclude_tags: tuple[str, ...] = ()) -> dict:
    client = make_llm_client()
    interpreter = interpreter or make_interpreter(client)
    writer = writer or (LLMNoteWriter(client) if client else TemplateNoteWriter())
    scenarios = [s for s in load_scenarios() if (not scenario_ids or s["id"] in scenario_ids)
                 and not set(s.get("tags") or []) & set(exclude_tags)]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / run_id

    records = []
    for sc in scenarios:
        rec = run_scenario(sc, interpreter, writer, run_id=run_id)
        records.append(rec)
        print(f"{'PASS' if rec['passed'] else 'FAIL'}  {sc['id']}")
    return finalize_run(records, out, run_id, "Eval run", interpreter, writer)


def finalize_run(records: list[dict], out: Path, run_id: str, title: str, interpreter, writer,
                 extra_summary: list[str] | None = None) -> dict:
    """Write notes/, summary.md, results.json and run.db for a batch of scenario records."""
    (out / "notes").mkdir(parents=True, exist_ok=True)
    for r in records:
        (out / "notes" / f"{r['scenario']['id']}.md").write_text(r["note_markdown"], encoding="utf-8")
    tracing.flush()

    passed = sum(r["passed"] for r in records)
    cats = Counter(i["category"] for r in records for i in r["note"]["issues"])
    failing = [r for r in records if not r["passed"]]
    summary = [f"# {title} {run_id}", "",
               f"- interpreter: `{interpreter.name}`  |  note writer: `{writer.name}`",
               f"- **{passed}/{len(records)} scenarios passed**", ""]
    summary += extra_summary or []
    if failing:
        summary += ["## Read these first (failed)"] + [
            f"- [{r['scenario']['id']}](notes/{r['scenario']['id']}.md): "
            + "; ".join(c["name"] for c in r["checks"] if not c["passed"]) for r in failing] + [""]
    if cats:
        summary += ["## Issue categories (from notes)"] + [f"- {c}: {n}" for c, n in cats.most_common()] + [""]
    summary += ["## All scenarios", "| scenario | result | end | turns | max latency ms |", "|---|---|---|---|---|"]
    summary += [f"| {r['scenario']['id']} | {'PASS' if r['passed'] else 'FAIL'} | {r['end_reason'] or 'hangup'} "
                f"| {r.get('turns', '')} | {r['max_latency_ms']} |" for r in records]
    (out / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "note_markdown"} for r in records], indent=2, default=str), encoding="utf-8")

    init_db(f"sqlite:///{(out / 'run.db').as_posix()}")
    with session_scope() as s:
        for r in records:
            s.add(TestNote(run_id=run_id, scenario_id=r["scenario"]["id"], call_id=None, deterministic_passed=r["passed"],
                           severity_max=r["severity_max"], note_json=r["note"], note_markdown=r["note_markdown"],
                           writer=r["writer"]))
    return {"run_id": run_id, "dir": str(out), "passed": passed, "total": len(records),
            "failed": [r["scenario"]["id"] for r in failing]}
