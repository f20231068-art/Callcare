"""Defects found only when a real model played the parents (live persona audit, rounds 2-3)."""
import json

from sqlalchemy import select

from carecall.agent.graph import validate_interpretation
from carecall.agent.interpret import Interpretation, InterpretRequest, LLMInterpreter, RuleInterpreter
from carecall.agent.session import CallSession
from carecall.calls.service import create_call
from carecall.db import session_scope
from carecall.models import Alert, Event, MealLog, ReminderAck


class Stub:
    model = "stub"

    def __init__(self, *replies):
        self.replies = list(replies)

    def complete(self, system, user):
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


class Boom:
    model = "boom"

    def complete(self, system, user):
        raise RuntimeError("provider down")


def run(routine, interpreter, *utterances):
    cid, _ = create_call(routine)
    s = CallSession(cid, interpreter)
    s.open()
    for u in utterances:
        if s.done:
            break
        s.turn(u)
    s.finish(abandoned=not s.done)
    return cid, s


def test_null_fields_from_the_model_do_not_invalidate_the_reply():
    reply = '{"intent":"answer","yes_no":true,"evidence":"Yes","confidence":0.9,"reason":null,"items":null,"concern":null}'
    out = LLMInterpreter(Stub(reply)).interpret(InterpretRequest(step_kind="yes_no", question="q", utterance="Yes"))
    assert out.intent == "answer" and out.yes_no is True and out.reason == ""


def test_answer_inside_a_repeat_request_wins_for_times():
    step = {"kind": "time", "handler": "set_meal_time", "topic": "breakfast"}
    repeat = Interpretation(intent="repeat", evidence="What was that", confidence=0.9)
    out, why = validate_interpretation(repeat, step, "Eh? What was that? Breakfast? About half past seven!")
    assert out.intent == "answer" and out.time == "07:30" and why == "answer_wins"


def test_pure_repeat_request_is_still_a_repeat():
    step = {"kind": "time", "handler": "set_meal_time", "topic": "breakfast"}
    repeat = Interpretation(intent="repeat", evidence="Say again", confidence=0.9)
    out, why = validate_interpretation(repeat, step, "What? Say again?")
    assert out.intent == "repeat" and why is None


def test_i_think_i_got_it_is_an_acknowledgement_not_a_hedge():
    step = {"kind": "ack", "handler": "reminder_ack"}
    ack = Interpretation(intent="answer", yes_no=True, evidence="I think I've got it now", confidence=0.9)
    out, why = validate_interpretation(ack, step, "Sorry, was that Doctor Rao? I think I've got it now.")
    assert out.intent == "answer" and why is None


def test_i_think_i_took_it_is_still_a_hedge():
    step = {"kind": "yes_no", "handler": "medicine_taken"}
    yes = Interpretation(intent="answer", yes_no=True, evidence="I think I took it", confidence=0.9)
    out, why = validate_interpretation(yes, step, "I think I took it")
    assert out.intent == "unsure" and why == "hedged"


def test_llm_outage_falls_back_to_rules_instead_of_failing_the_call():
    cid, s = run("metformin_reminder", LLMInterpreter(Boom()), "Yes, I have taken it")
    with session_scope() as q:
        interp_events = [e.payload for e in q.scalars(select(Event).where(Event.type == "interpretation"))]
    assert interp_events[-1]["fallback"] == "rules" and "provider down" in interp_events[-1]["error"]
    assert s.end_reason == "completed"  # the rules still understood a plain "yes"


def test_every_interpretation_is_recorded_for_audit():
    cid, _ = run("lunch_checkin", RuleInterpreter(), "yes", "rice")
    with session_scope() as q:
        evs = q.scalars(select(Event).where(Event.type == "interpretation")).all()
        assert [e.payload["intent"] for e in evs] == ["answer", "answer"]
        assert evs[0].payload["utterance"] == "yes"


def test_meal_items_given_with_the_yes_are_not_asked_again():
    reply = json.dumps({"intent": "answer", "yes_no": True, "items": ["rice", "sambar"], "evidence": "Yes, rice and sambar",
                        "confidence": 0.95})
    cid, s = run("lunch_checkin", LLMInterpreter(Stub(reply)), "Yes, rice and sambar")
    with session_scope() as q:
        (log,) = q.scalars(select(MealLog)).all()
        assert log.items == ["rice", "sambar"] and s.end_reason == "completed"


def test_symptom_sympathy_is_not_followed_by_good_to_hear():
    cid, s = run("lunch_checkin", RuleInterpreter(), "Yes I had lunch but I felt dizzy")
    out = " ".join(t for t in [])  # placeholder to keep structure simple
    with session_scope() as q:
        from carecall.models import TranscriptTurn
        agent = [t.text for t in q.scalars(select(TranscriptTurn).where(TranscriptTurn.speaker == "agent"))]
    assert any("sorry to hear" in a.lower() for a in agent) and not any(a == "Good to hear." for a in agent)


def test_lonely_remarks_become_one_alert_not_three():
    seq = [json.dumps({"intent": "answer", "time": t, "concern": c, "evidence": e, "confidence": 0.9})
           for t, c, e in [("08:30", "the afternoons feel quiet", "half past eight"),
                           ("13:30", "house is so quiet without her", "half past one"),
                           ("16:30", "It gets a bit lonely at the table", "half past four")]]
    cid, s = run("meal_discovery", LLMInterpreter(Stub(*seq)),
                 "half past eight, the afternoons are quiet", "half past one, the house is so quiet", "half past four, a bit lonely")
    with session_scope() as q:
        assert len(q.scalars(select(Alert)).all()) == 1


def test_parent_asking_for_a_detail_the_reminder_lacks_gets_an_honest_answer_and_the_family_is_told():
    repeat = json.dumps({"intent": "repeat", "evidence": "what time exactly", "confidence": 0.9})
    cid, s = run("doctor_reminder", LLMInterpreter(Stub(repeat)), "what time exactly?", "but what time exactly?")
    with session_scope() as q:
        assert q.scalars(select(ReminderAck)).one().acknowledged is True
        assert "details" in q.scalars(select(Alert)).one().reason


# ---- found when a weaker, faster model (qwen3.7-flash) was switched in ----
def test_one_more_time_is_not_one_oclock():
    """'Could you say that one more time?' once became dinner at 13:00 via the 'answer inside a repeat request' backstop."""
    step = {"kind": "time", "handler": "set_meal_time", "topic": "dinner"}
    repeat = Interpretation(intent="repeat", evidence="say that one more time", confidence=0.9)
    out, why = validate_interpretation(repeat, step, "I'm not sure I heard you correctly. Could you say that one more time?")
    assert out.intent == "repeat" and why is None


def test_implausible_meal_time_is_rejected_whatever_the_model_says():
    step = {"kind": "time", "handler": "set_meal_time", "topic": "dinner"}
    wrong = Interpretation(intent="answer", time="13:00", evidence="one", confidence=0.95)
    out, why = validate_interpretation(wrong, step, "one")
    assert out.intent == "unclear" and why == "implausible_time"
    ok = Interpretation(intent="answer", time="19:30", evidence="half past seven", confidence=0.9)
    assert validate_interpretation(ok, step, "half past seven in the evening")[1] is None
    breakfast = {"kind": "time", "handler": "set_meal_time", "topic": "breakfast"}
    assert validate_interpretation(Interpretation(intent="answer", time="21:00", evidence="nine", confidence=0.9), breakfast, "nine")[1] == "implausible_time"


def test_help_me_remember_is_not_an_emergency_but_help_me_is():
    from carecall.agent.interpret import detect_emergency

    for ok in ("I hope Sree can help me remember", "could you help me find my glasses", "please help me understand this"):
        assert detect_emergency(ok) is None, ok
    for bad in ("help me", "please help", "Help! I fell", "someone help me, I can't breathe"):
        assert detect_emergency(bad), bad


def test_the_closing_never_thanks_twice():
    from carecall.agent.routines import closing

    class C:
        parent_preferred_name = "Asha"

    assert not closing(C()).lower().startswith("thank")
    cid, s = run("metformin_reminder", RuleInterpreter(), "Yes I took it")
    from carecall.models import TranscriptTurn
    with session_scope() as q:
        agent = [t.text for t in q.scalars(select(TranscriptTurn).where(TranscriptTurn.speaker == "agent")).all()]
    assert sum("thank" in a.lower() for a in agent[-2:]) <= 1
