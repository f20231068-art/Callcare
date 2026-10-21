"""The conversational agent: the model holds the conversation (memory) and chooses tools; code supplies the tools' rules and a few rails.
The model is a stand-in that returns scripted JSON, so these tests check the PLUMBING and the GUARANTEES (memory, tool loop, refusals,
fillers, emergencies, silence). How well a real model converses is measured by `carecall personas --agent converse` and the live runs."""
import copy
import json

import pytest
from datetime import timedelta

from sqlalchemy import select

from carecall.agent.session import CallSession
from carecall.config import get_settings
from carecall.calls.keys import occurrence_key
from carecall.calls.service import create_call, schedule_attempt
from carecall.db import session_scope
from carecall.models import Alert, CallAttempt, Callback, Event, MealLog, MealProfile, Routine, TranscriptTurn, utcnow

OCC = occurrence_key(1, "lunch_checkin", utcnow().date().isoformat())


@pytest.fixture(autouse=True)
def _no_quiet_hours_for_relative_times(monkeypatch):
    """'in 60 minutes' must not depend on what time of day the suite runs (the real window, 21:30-07:00, would move it to the morning)."""
    monkeypatch.setenv("QUIET_START", "23:59")
    monkeypatch.setenv("QUIET_END", "00:00")
    get_settings.cache_clear()


def R(say=None, actions=(), end=False):
    return json.dumps({"say": say, "actions": [{"tool": t, "args": a} for t, a in actions], "end_call": end})


class FakeLLM:
    model = "fake-converse"

    def __init__(self, *replies):
        self.replies, self.calls, self.systems = list(replies), [], []

    def chat_json(self, system, messages, max_tokens=None):
        self.calls.append(copy.deepcopy(messages))
        self.systems.append(system)
        return self.replies.pop(0) if self.replies else R("(out of script)")


def start(llm, routine="lunch_checkin", **kw):
    cid, _ = create_call(routine, channel="twilio", occurrence_key=OCC, idempotency_key=OCC + ":a1")
    s = CallSession(cid, channel="twilio", mode="converse", llm=llm, **kw)
    opening = s.open()
    return cid, s, opening


def rows(model, **where):
    with session_scope() as q:
        return q.scalars(select(model).filter_by(**where)).all()


def test_the_opening_is_instant_and_scripted_and_needs_no_model_call():
    llm = FakeLLM()
    cid, s, opening = start(llm)
    assert opening[1] == "Have you had your lunch today?" and "Hello Asha" in opening[0] and llm.calls == []


def test_the_model_sees_the_whole_conversation_every_turn():
    llm = FakeLLM(R("Lovely. What did you have?"), R("Thank you, noted.", [("log_meal", {"meal_type": "lunch", "status": "eaten", "items": ["rice"], "evidence": "rice and dal"})], end=True))
    cid, s, _ = start(llm)
    s.turn("yes I did")
    s.turn("rice and dal")
    second = llm.calls[1]
    texts = " ".join(m["content"] for m in second)
    assert "yes I did" in texts and "What did you have?" in texts and "Have you had your lunch today?" in texts  # earlier turns are in memory
    assert [m["role"] for m in second][:2] == ["user", "assistant"]


def test_a_fact_is_logged_by_the_model_through_a_tool_with_a_real_quote_in_one_model_call():
    llm = FakeLLM(R("Wonderful, thank you. Take care, Asha.", [("log_meal", {"meal_type": "lunch", "status": "eaten", "items": ["dal", "rice"], "evidence": "dal and rice"})], end=True))
    cid, s, _ = start(llm)
    r = s.turn("yes, I had dal and rice")
    assert len(llm.calls) == 1 and r.done and s.end_reason == "completed"  # logging is fire-and-forget: no second model call
    (m,) = rows(MealLog, call_id=cid)
    assert (m.status, m.items, m.source_utterance) == ("eaten", ["dal", "rice"], "dal and rice")


def test_a_quote_the_parent_never_said_is_refused_and_the_model_must_correct_itself():
    llm = FakeLLM(R("Noted.", [("log_meal", {"meal_type": "lunch", "status": "eaten", "items": ["pizza"], "evidence": "I ate pizza"})]),
                  R("Thank you.", [("log_meal", {"meal_type": "lunch", "status": "eaten", "items": ["dal"], "evidence": "dal"})], end=True))
    cid, s, _ = start(llm)
    s.turn("yes I had some dal")
    assert len(llm.calls) == 2 and "evidence" in llm.calls[1][-1]["content"]  # the refusal went back to the model
    (m,) = rows(MealLog, call_id=cid)
    assert m.items == ["dal"]


def test_logging_the_same_fact_twice_does_not_duplicate_it():
    act = ("log_meal", {"meal_type": "lunch", "status": "eaten", "items": ["dal"], "evidence": "dal"})
    llm = FakeLLM(R("Good.", [act]), R("Thanks.", [act], end=True))
    cid, s, _ = start(llm)
    s.turn("dal")
    s.turn("yes dal")
    assert len(rows(MealLog, call_id=cid)) == 1


def test_negotiation_a_refused_time_is_explained_and_the_parent_can_counter_offer():
    """'4 hours later' collides with the next reminder; the tool says why and what would work; the parent proposes 1 hour; it is booked."""
    with session_scope() as q:
        rid = q.scalars(select(Routine.id).where(Routine.key == "metformin_reminder")).first()
        schedule_attempt(q, parent_id=1, routine_id=rid, occurrence_key="1:other:x:s1", attempt_no=1, kind="scheduled", when=utcnow() + timedelta(minutes=150))
    llm = FakeLLM(
        R(None, [("book_callback", {"in_minutes": 240, "reason": "parent asked"})]),
        R("Four hours is after your next reminder, so I can ring before that. Would two hours earlier suit, or sooner?"),
        R(None, [("book_callback", {"in_minutes": 60})]),
        R("Lovely, I will call you in an hour. Take care.", end=True))
    cid, s, _ = start(llm)
    r1 = s.turn("not yet, call me in four hours")
    refusal = json.loads(llm.calls[1][-1]["content"].removeprefix("TOOL RESULTS: "))[0]
    assert refusal["ok"] is False and refusal["reason_code"] == "next_reminder_first" and refusal["latest_possible_local"]  # reason + alternative
    assert not r1.done and any("two hours" in t for t in r1.reply)
    assert [a for a in rows(CallAttempt, prev_attempt_id=cid)] == []  # nothing was promised or booked yet
    r2 = s.turn("what about in an hour then")
    assert r2.done and s.end_reason == "callback"
    (cb,) = rows(CallAttempt, prev_attempt_id=cid)
    assert cb.kind == "callback" and cb.idempotency_key == OCC + ":cb2"
    assert len(rows(Callback, call_id=cid)) == 1
    assert "Sure, I will call you again" in " ".join(r2.reply)  # the parent is told what is about to happen  # a filler while it booked
    assert "TOOL RESULTS" in llm.calls[3][-1]["content"]  # the booking result went back to the model before it spoke


def test_fillers_are_spoken_live_and_not_repeated_by_the_audio_layer():
    llm = FakeLLM(R(None, [("book_callback", {"in_minutes": 30})]), R("I will call you in half an hour.", end=True))
    cid, s, _ = start(llm)
    spoken = []
    s.speak_now = spoken.append
    r = s.turn("not yet, ring me later")
    assert len(spoken) == 1 and spoken[0] not in r.reply
    said = [t.text for t in rows(TranscriptTurn, call_id=cid, speaker="agent")]
    assert spoken[0] in said  # but it is in the transcript


def test_an_emergency_word_pre_empts_the_model_entirely():
    llm = FakeLLM(R("(this must never be asked)"))
    cid, s, _ = start(llm)
    r = s.turn("I fell and my chest hurts")
    assert r.done and s.end_reason == "emergency" and llm.calls == []
    assert any(a.severity == "emergency" for a in rows(Alert, call_id=cid))


def test_three_silences_end_the_call_without_asking_the_model_a_third_time():
    llm = FakeLLM(R("Hello, are you there?"), R("Can you hear me?"))
    cid, s, _ = start(llm)
    s.turn("")
    s.turn("")
    r = s.turn("")
    assert r.done and len(llm.calls) == 2 and any(a.severity == "info" for a in rows(Alert, call_id=cid))


def test_the_model_cannot_use_tools_it_was_not_given():
    """Meal times belong to the caretaker; pausing calls belongs to the engine; the model asking for either is refused."""
    llm = FakeLLM(R(None, [("update_meal_time", {"meal": "lunch", "time": "15:00"}), ("pause_calls", {"reason": "x"})]), R("Sorry, I cannot do that.", end=True))
    cid, s, _ = start(llm)
    s.turn("change my lunch to three and stop calling everyone")
    refusals = json.loads(llm.calls[1][-1]["content"].removeprefix("TOOL RESULTS: "))
    assert [r["ok"] for r in refusals] == [False, False]
    with session_scope() as q:
        assert q.get(MealProfile, (1, "lunch")).usual_time == "12:30"
        from carecall.models import Parent
        assert q.get(Parent, 1).calling_enabled


def test_a_model_that_returns_garbage_gets_one_retry_then_a_polite_fallback():
    llm = FakeLLM("not json at all", "still not json")
    cid, s, _ = start(llm)
    r = s.turn("yes")
    assert not r.done and any("say that again" in t for t in r.reply) and len(llm.calls) == 2


def test_replies_are_capped_and_never_leak_numbers():
    long = " ".join(["This is one short sentence."] * 40)
    llm = FakeLLM(R(long), R("You can reach him on 9876543210 anytime."))
    cid, s, _ = start(llm)
    assert len(s.turn("hello").reply[0].split()) <= 61
    assert "9876543210" not in " ".join(s.turn("what is his number").reply)


def test_after_the_call_the_live_booking_is_kept_and_not_duplicated():
    llm = FakeLLM(R(None, [("book_callback", {"in_minutes": 20})]), R("I will call you in twenty minutes.", end=True))
    cid, s, _ = start(llm)
    s.turn("ring me in twenty minutes")
    s.finish()
    with session_scope() as q:
        assert q.get(CallAttempt, cid).outcome == "callback_requested"
        assert len(q.scalars(select(CallAttempt).where(CallAttempt.prev_attempt_id == cid)).all()) == 1
        assert any(e.type == "followup.already_scheduled" for e in q.scalars(select(Event)))


def test_a_ring_back_gets_the_earlier_conversation_in_its_prompt_so_nothing_is_asked_twice():
    llm1 = FakeLLM(R("Lovely. What did you have?"))
    cid, s, _ = start(llm1)
    s.turn("yes I have eaten")
    s.finish(abandoned=True, hangup={"parent_speaking": True})
    with session_scope() as q:
        nxt = q.scalars(select(CallAttempt).where(CallAttempt.prev_attempt_id == cid)).one()
        nid = nxt.id
    llm2 = FakeLLM(R("Sorry about that. You said you had eaten; what did you have?"))
    s2 = CallSession(nid, channel="twilio", mode="converse", llm=llm2)
    opening = s2.open()
    assert "yes I have eaten" in llm2.systems[0] and "EARLIER ON THIS REMINDER" in llm2.systems[0]
    assert "what did you have" in opening[1].lower()


def test_a_refusal_caused_by_the_night_says_quiet_hours_not_next_reminder(monkeypatch):
    monkeypatch.setenv("QUIET_START", "00:00")  # always night
    monkeypatch.setenv("QUIET_END", "23:59")
    get_settings.cache_clear()
    with session_scope() as q:
        rid = q.scalars(select(Routine.id).where(Routine.key == "metformin_reminder")).first()
        schedule_attempt(q, parent_id=1, routine_id=rid, occurrence_key="1:other:x:s1", attempt_no=1, kind="scheduled", when=utcnow() + timedelta(minutes=150))
    llm = FakeLLM(R(None, [("book_callback", {"in_minutes": 60})]), R("I cannot call at night. Shall I call you tomorrow morning instead?"))
    cid, s, _ = start(llm)
    s.turn("call me in an hour")
    res = json.loads(llm.calls[1][-1]["content"].removeprefix("TOOL RESULTS: "))[0]
    assert res["ok"] is False and res["reason_code"] == "quiet_hours" and res["quiet_hours_start_local"] == "00:00"


def test_the_same_line_a_third_time_ends_the_call_instead_of_looping():
    ask = "Which foods did you have?"
    llm = FakeLLM(R(ask), R(ask), R(ask))
    cid, s, _ = start(llm)
    s.turn("the usual")
    s.turn("the usual food")
    r = s.turn("you know, the usual")
    assert r.done and "leave it there" in " ".join(r.reply)
    with session_scope() as q:
        assert any(e.type == "converse.loop_guard" for e in q.scalars(select(Event)))


def test_a_booking_in_the_same_reply_as_a_question_is_held_until_the_parent_answers():
    llm = FakeLLM(R("That would be around 4:30. Is that okay?", [("book_callback", {"in_minutes": 60})]),
                  R(None, [("book_callback", {"in_minutes": 60})]),
                  R("Done, I will call you then.", [], True))
    cid, s, _ = start(llm)
    first = s.turn("call me later")
    assert first.reply[-1].endswith("Is that okay?") and not rows(Callback, call_id=cid)
    assert any(e.type == "converse.held" for e in rows(Event, call_id=cid))
    s.turn("yes")
    held_memory = json.loads(llm.calls[1][-2]["content"])  # what the model sees it said: the question, with no booking attached
    assert held_memory["actions"] == []
    assert len(rows(Callback, call_id=cid)) == 1


def test_logging_and_alerts_are_never_held_by_a_question():
    llm = FakeLLM(R("Thanks. And did you take the tablet?", [("log_meal", {"meal_type": "lunch", "status": "eaten", "items": ["rice"], "evidence": "I had rice"})]))
    cid, s, _ = start(llm)
    s.turn("I had rice")
    assert rows(MealLog, call_id=cid)


def _linked_call(prev_outcome, kind):
    prev_id, _ = create_call("lunch_checkin", channel="twilio", occurrence_key=OCC, idempotency_key=OCC + ":a1")
    with session_scope() as s:
        s.get(CallAttempt, prev_id).outcome = prev_outcome
    cid, _ = create_call("lunch_checkin", channel="twilio", occurrence_key=OCC, idempotency_key=OCC + ":x2", kind=kind)
    with session_scope() as s:
        s.get(CallAttempt, cid).prev_attempt_id = prev_id
    return cid


def test_a_callback_opens_with_the_scripted_question_and_never_asks_the_model():
    """Call 21 opened by repeating the old goodbye ("I will call you again at ...") because the model was asked to continue from it."""
    cid = _linked_call("callback_requested", "callback")
    llm = FakeLLM()
    opening = CallSession(cid, channel="twilio", mode="converse", llm=llm).open()
    assert llm.calls == [] and any("lunch" in line.lower() for line in opening)


def test_a_call_cut_off_mid_conversation_is_picked_up_by_the_model():
    cid = _linked_call("cut_off", "retry")
    llm = FakeLLM(R("Welcome back. Did you manage to have lunch?"))
    opening = CallSession(cid, channel="twilio", mode="converse", llm=llm).open()
    assert len(llm.calls) == 1 and "Welcome back" in " ".join(opening)


def test_the_parent_is_told_what_is_about_to_happen_before_a_booking_is_made():
    spoken = []
    llm = FakeLLM(R(None, [("book_callback", {"in_minutes": 2})]), R("All right, I will call you at the time given.", end=True))
    cid, s, _ = start(llm)
    s.speak_now = lambda line: spoken.append(line)  # the live-audio bridge: plays a line now, while the tool runs
    r = s.turn("Remind me in 2 minutes.")
    assert spoken == ["Sure, I will call you again in 2 minutes. One moment while I set that up."]
    assert "Sure, I will call you again in 2 minutes." not in r.reply  # already said live, not repeated
    assert len(rows(Callback, call_id=cid)) == 1


def test_on_the_turn_by_turn_phone_mode_the_announcement_leads_the_reply_without_claiming_one_moment():
    llm = FakeLLM(R(None, [("book_callback", {"at": "16:15"})]), R("Done.", end=True))
    cid, s, _ = start(llm)
    r = s.turn("Call me at quarter past four.")
    assert r.reply[0] == "Sure, I will call you again at 4:15 in the afternoon."


def test_a_held_question_makes_no_announcement_because_nothing_is_being_done_yet():
    spoken = []
    llm = FakeLLM(R("That would be around 4:30. Is that okay?", [("book_callback", {"in_minutes": 60})]))
    cid, s, _ = start(llm)
    s.speak_now = lambda line: spoken.append(line)
    s.turn("call me later")
    assert spoken == [] and not rows(Callback, call_id=cid)
