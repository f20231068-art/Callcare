"""Conversation behaviour through CallSession (text channel, rule interpreter)."""
from sqlalchemy import select

from carecall.agent.interpret import Interpretation, RuleInterpreter
from carecall.agent.session import CallSession
from carecall.calls.service import create_call
from carecall.db import session_scope
from carecall.evals.runner import run_evals
from carecall.models import Alert, CallAttempt, Callback, MealLog, MedicineLog


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


def test_lunch_happy_path_logs_with_quote():
    cid, s = run("lunch_checkin", "yes I had lunch", "rice and dal")
    with session_scope() as q:
        (log,) = q.scalars(select(MealLog)).all()
        assert (log.meal_type, log.status, log.items) == ("lunch", "eaten", ["rice", "dal"])
        assert log.source_utterance == "rice and dal"
        assert q.get(CallAttempt, cid).status == "completed"


def test_negation_is_not_a_yes():
    run("metformin_reminder", "I haven't taken it")
    with session_scope() as q:
        (log,) = q.scalars(select(MedicineLog)).all()
        assert log.status == "not_taken"
        assert q.scalars(select(Callback)).one().delay_minutes == 10


def test_emergency_alert_bypasses_interpreter():
    class Boom:
        name = "boom"

        def interpret(self, req):
            raise AssertionError("the model must not be consulted for emergency words")

    _, s = run("lunch_checkin", "I fell down and my chest pain is bad", interpreter=Boom())
    assert s.end_reason == "emergency"
    with session_scope() as q:
        assert q.scalars(select(Alert)).one().severity == "emergency"


def test_fabricated_evidence_is_downgraded_to_unclear():
    """A model that 'quotes' words the parent never said must not produce a log."""
    class Liar:
        name = "liar"

        def interpret(self, req):
            return Interpretation(intent="answer", yes_no=True, evidence="yes I definitely took it", confidence=0.99)

    run("metformin_reminder", "mmm", "hmm", "uh", interpreter=Liar())
    with session_scope() as q:
        (log,) = q.scalars(select(MedicineLog)).all()
        assert log.status == "unclear"


def test_unclear_never_logged_as_taken():
    run("metformin_reminder", "mm", "hmm", "tablet?")
    with session_scope() as q:
        assert [l.status for l in q.scalars(select(MedicineLog))] == ["unclear"]
        assert q.scalars(select(Alert)).one().severity == "warning"


def test_silence_reprompts_then_gives_up():
    _, s = run("lunch_checkin", "", "", "")
    with session_scope() as q:
        assert [l.status for l in q.scalars(select(MealLog))] == ["unclear"]


def test_offline_eval_baseline_passes(tmp_path):
    """Scenarios the keyword baseline is expected to handle. llm_needed ones are excluded on purpose."""
    res = run_evals(out_root=str(tmp_path), exclude_tags=("llm_needed",))
    assert res["failed"] == [], res
    assert (tmp_path / res["run_id"] / "summary.md").exists()
    assert list((tmp_path / res["run_id"] / "notes").glob("*.md"))
