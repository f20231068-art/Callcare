"""Guards for the defects the persona audit found, plus the persona harness itself."""
import json
from collections import Counter

from sqlalchemy import select

from carecall.agent.graph import validate_interpretation
from carecall.agent.interpret import Interpretation, LLMInterpreter
from carecall.agent.routines import spoken_time
from carecall.agent.session import CallSession
from carecall.calls.service import create_call
from carecall.db import session_scope
from carecall.evals.persona_runner import run_personas
from carecall.evals.personas import LLMPersona, ScriptedPersona, load_personas
from carecall.models import Alert, MealLog, MedicineLog, TranscriptTurn
from carecall.prompts import USECASES, system_prompt

# Personas that genuinely need a language model; the offline keyword baseline is expected to miss them.
NEEDS_LLM = {"ma_hinglish_shanta"}


class Stub:
    model = "stub"

    def __init__(self, *replies):
        self.replies = list(replies)

    def complete(self, system, user):
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_five_personas_per_use_case_and_a_prompt_for_each():
    counts = Counter(p["usecase"] for p in load_personas())
    assert set(counts) == set(USECASES) and all(n == 5 for n in counts.values())
    texts = {system_prompt(u)[0] for u in USECASES}
    assert len(texts) == len(USECASES)  # genuinely different prompt per use case
    assert all(system_prompt(u)[0].startswith("You extract structured data") for u in USECASES)  # shared base


def test_offline_persona_suite(tmp_path):
    res = run_personas("scripted", out_root=str(tmp_path), agent="rules")
    assert res["total"] == 25
    assert set(res["failed"]) <= NEEDS_LLM, res["failed"]
    report = (tmp_path / res["run_id"] / "persona_report.md").read_text(encoding="utf-8")
    assert "Scoreboard by use case" in report and "Every persona" in report


def test_hedged_yes_is_not_a_yes():
    step = {"kind": "yes_no", "handler": "meal_ate"}
    liar = Interpretation(intent="answer", yes_no=True, evidence="Maybe", confidence=0.9)
    out, why = validate_interpretation(liar, step, "Maybe. I am not sure, beta.")
    assert out.intent == "unsure" and out.yes_no is None and why == "hedged"


def test_tablet_described_by_looks_is_not_confirmation():
    step = {"kind": "yes_no", "handler": "medicine_taken"}
    yes = Interpretation(intent="answer", yes_no=True, evidence="I took the white one", confidence=0.9)
    out, why = validate_interpretation(yes, step, "I took the white one")
    assert out.intent == "unclear" and why == "hedged_which_tablet"


def test_filler_is_not_a_food():
    step = {"kind": "items", "handler": "meal_items"}
    out, why = validate_interpretation(Interpretation(intent="answer", items=["hmm"], evidence="Hmm", confidence=0.9), step, "Hmm.")
    assert out.intent == "unclear" and why == "filler_items"


def test_spoken_time_has_no_24h_digits():
    assert spoken_time("09:00") == "9 in the morning"
    assert spoken_time("13:30") == "1:30 in the afternoon"
    assert spoken_time("12:00") == "12 at noon"
    assert spoken_time("20:30") == "8:30 at night"


def run(routine, *utterances, interpreter=None):
    cid, _ = create_call(routine)
    s = CallSession(cid, interpreter)
    s.open()
    for u in utterances:
        if s.done:
            break
        s.turn(u)
    s.finish(abandoned=not s.done)
    return cid, s


def test_refusal_with_symptom_raises_one_alert_not_two():
    run("metformin_reminder", "I do not want to take it, it makes me feel sick")
    with session_scope() as q:
        (alert,) = q.scalars(select(Alert)).all()
        assert "feel sick" in alert.reason and "declined" in alert.reason


def test_honest_dont_remember_is_recorded_without_re_asking():
    cid, _ = run("lunch_checkin", "Did I? I do not remember.")
    with session_scope() as q:
        assert [l.status for l in q.scalars(select(MealLog))] == ["unclear"]
        assert q.scalars(select(Alert)).one().severity == "info"
        agent = [t.text for t in q.scalars(select(TranscriptTurn).where(TranscriptTurn.speaker == "agent"))]
        assert not any("did not catch" in a for a in agent)


def test_unsure_about_medicine_schedules_a_callback_and_warns():
    run("metformin_reminder", "I think I took it, not sure")
    with session_scope() as q:
        assert [l.status for l in q.scalars(select(MedicineLog))] == ["unclear"]
        assert q.scalars(select(Alert)).one().severity == "warning"


def test_same_words_twice_do_not_loop_a_third_time():
    _, s = run("doctor_reminder", "blah blah", "blah blah", "blah blah", "blah blah")
    with session_scope() as q:
        parent_turns = q.scalars(select(TranscriptTurn).where(TranscriptTurn.speaker == "parent")).all()
        assert len(parent_turns) == 2  # gave up after the repeat instead of asking a third time
        alert = q.scalars(select(Alert)).one()
        assert "blah blah" in alert.reason  # the family sees the raw words


def test_identity_question_gets_reassurance():
    cid, _ = run("lunch_checkin", "Who is this? Why are you calling?")
    with session_scope() as q:
        agent = " ".join(t.text for t in q.scalars(select(TranscriptTurn).where(TranscriptTurn.speaker == "agent")))
        assert "never ask for money" in agent


def test_hard_of_hearing_gets_slower_speech():
    cid, _ = create_call("lunch_checkin")
    s = CallSession(cid)
    s.open()
    assert s.turn("What? Say again?").rate < 1.0


def test_llm_interpreter_handles_hinglish_that_rules_cannot():
    reply = json.dumps({"intent": "answer", "yes_no": True, "evidence": "goli bhi le li", "confidence": 0.9})
    card = next(p for p in load_personas(guardrails=False) if p["id"] == "ma_hinglish_shanta")  # plain card: no detour
    cid, _ = create_call(card["routine"])
    s = CallSession(cid, LLMInterpreter(Stub(reply)))
    s.open()
    persona = ScriptedPersona(card)
    s.turn(persona.reply("taken", []))
    with session_scope() as q:
        assert [l.status for l in q.scalars(select(MedicineLog))] == ["taken"]


def test_llm_persona_maps_silence_and_uses_card():
    card = next(p for p in load_personas() if p["id"] == "dr_silent_ganesh")
    assert LLMPersona(card, Stub("<silence>")).reply("ack", [("agent", "Did you get that?")]) == ""
    assert LLMPersona(card, Stub("Okay.")).reply("ack", [("agent", "Did you get that?")]) == "Okay."
