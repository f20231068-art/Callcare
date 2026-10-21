"""Scheduling happens LIVE, inside the call: the agent books the call-back / reminder through a tool, speaks only what the tool
returned, covers the wait with a human-sounding filler, and the post-call engine keeps (never duplicates) what was booked.
The interpreter is replaced by queued interpretations so these tests check the graph + tools + database, not model quality
(model behaviour is checked by the live persona run)."""
from datetime import timedelta

import pytest

from sqlalchemy import select

from carecall.agent.interpret import Interpretation, ToolRequest
from carecall.agent.session import CallSession
from carecall.calls import outcome
from carecall.config import get_settings
from carecall.calls.keys import occurrence_key
from carecall.calls.service import create_call
from carecall.db import session_scope
from carecall.models import Callback, CallAttempt, Event, Parent, Reminder, Routine, utcnow

OCC = occurrence_key(1, "lunch_checkin", utcnow().date().isoformat())


@pytest.fixture(autouse=True)
def _no_quiet_hours_unless_the_test_is_about_them(request, monkeypatch):
    """Relative times ('in 30 minutes') must not depend on the wall clock; tests about the night keep the real window."""
    if "quiet" not in request.node.name and "clock_time" not in request.node.name:
        monkeypatch.setenv("QUIET_START", "23:59")
        monkeypatch.setenv("QUIET_END", "00:00")
        get_settings.cache_clear()


class Queued:
    name = "queued"

    def __init__(self, *interps):
        self.q = list(interps)

    def interpret(self, req):
        return self.q.pop(0)


def ans(**kw):
    base = dict(intent="answer", evidence="x", confidence=0.95)
    base.update(kw)
    return Interpretation(**base)


def start(*interps, routine="lunch_checkin"):
    cid, _ = create_call(routine, channel="twilio", occurrence_key=OCC, idempotency_key=OCC + ":a1")
    s = CallSession(cid, Queued(*interps), channel="twilio")
    s.open()
    return cid, s


def attempts(**where):
    with session_scope() as q:
        return [(c.id, c.idempotency_key, c.kind, c.status, c.scheduled_at, c.prev_attempt_id) for c in
                q.scalars(select(CallAttempt).order_by(CallAttempt.id)) if all(getattr(c, k) == v for k, v in where.items())]


def test_not_yet_books_a_real_callback_during_the_call_and_the_reply_comes_from_the_tool():
    cid, s = start(ans(yes_no=False, evidence="not yet"))
    r = s.turn("not yet")
    text = " ".join(r.reply)
    (row,) = attempts(kind="callback")
    assert row[1] == OCC + ":cb2" and row[3] == "scheduled" and row[5] == cid  # booked NOW, key matches the post-call rule
    assert "in 30 minutes" in text and r.done and s.end_reason == "callback"
    assert any(w in text for w in ("hold on", "second", "moment"))  # the filler leads the reply
    with session_scope() as q:
        cb = q.scalars(select(Callback).where(Callback.call_id == cid)).one()
        assert cb.delay_minutes in (29, 30) and any(e.type == "db_tool.book_callback" for e in q.scalars(select(Event)))


def test_the_post_call_engine_keeps_the_live_booking_and_adds_nothing():
    cid, s = start(ans(yes_no=False, evidence="not yet"))
    s.turn("not yet")
    s.finish()
    cb = attempts(kind="callback")
    assert len(cb) == 1 and len(attempts()) == 2  # no second follow-up, and tomorrow is NOT planned while today's job is still open
    with session_scope() as q:
        assert q.get(CallAttempt, cid).outcome == "callback_requested"
        assert any(e.type == "followup.already_scheduled" for e in q.scalars(select(Event)))


def test_when_the_callback_call_itself_completes_tomorrows_call_is_scheduled():
    cid, s = start(ans(yes_no=False, evidence="not yet"))
    s.turn("not yet")
    s.finish()
    (cb_id, cb_key, *_), = attempts(kind="callback")
    s2 = CallSession(cb_id, Queued(ans(yes_no=True, items=["rice"], evidence="yes rice")), channel="twilio")
    greeting = s2.open()
    assert "calling back, as arranged" in greeting[0]
    s2.turn("yes rice")
    s2.finish()
    tomorrow = occurrence_key(1, "lunch_checkin", (utcnow().astimezone().date() + timedelta(days=1)).isoformat())
    keys = [k for _, k, _, st, *_ in attempts() if st == "scheduled"]
    assert any(k.startswith(tomorrow[:-3].rsplit(":", 1)[0]) or ":a1" in k for k in keys)
    assert len([k for k in keys if k.endswith(":a1")]) == 1


def test_asking_for_a_clock_time_in_quiet_hours_is_booked_honestly_not_as_promised():
    cid, s = start(ans(intent="callback", callback_at="23:30", evidence="call me at 11:30 at night"))
    r = s.turn("call me at 11:30 at night")
    text = " ".join(r.reply)
    (row,) = attempts(kind="callback")
    assert "tomorrow" in text and "in the morning" in text and "11:30" not in text  # said 7 in the morning, not the requested time
    assert row[3] == "scheduled"


def test_changing_your_mind_moves_the_same_booking():
    cid, s = start(ans(intent="callback", callback_minutes=10, evidence="in ten"), ans(intent="callback", callback_minutes=40, evidence="forty"))
    s.turn("call me in ten")
    s2 = CallSession.__new__(CallSession)  # same call, second request (agent ended the call after the first; call the tool directly)
    from carecall.agent.db_tools import call_tool
    r = call_tool("book_callback", {"in_minutes": 40}, {"parent_id": 1, "call_id": cid})
    assert r["ok"] and r["reused"] and len(attempts(kind="callback")) == 1 and 39 <= r["in_minutes"] <= 40


def test_a_reminder_request_is_read_back_then_set_only_after_yes():
    day = (utcnow().astimezone() + timedelta(days=3)).strftime("%Y-%m-%d")
    req = ToolRequest(tool="add_reminder", args={"message": "Dr Rao appointment", "at": f"{day} 16:30"}, evidence="remind me about Dr Rao on Friday at 4:30")
    cid, s = start(ans(yes_no=True, items=["rice"], evidence="yes rice", requests=[req]),
                   ans(yes_no=True, evidence="yes please"))
    r1 = s.turn("yes rice, and remind me about Dr Rao on Friday at 4:30")
    assert any("Shall I set that" in t for t in r1.reply) and not r1.done
    with session_scope() as q:
        assert q.scalars(select(Reminder).where(Reminder.message.like("Dr Rao%"))).all() == []  # nothing written before the parent says yes
    r2 = s.turn("yes please")
    assert any("Done" in t for t in r2.reply) and any(w in " ".join(r2.reply) for w in ("hold on", "second", "moment"))
    with session_scope() as q:
        rem = q.scalars(select(Reminder).where(Reminder.message.like("Dr Rao%"))).one()
        rt = q.scalars(select(Routine).where(Routine.key == f"reminder_{rem.id}")).one()
        assert rt.type == "defined_reminder"
    assert len([a for a in attempts(status="scheduled") if a[2] == "scheduled"]) == 1


def test_saying_no_to_the_read_back_sets_nothing():
    day = (utcnow().astimezone() + timedelta(days=3)).strftime("%Y-%m-%d")
    req = ToolRequest(tool="add_reminder", args={"message": "Dr Rao", "at": f"{day} 16:30"}, evidence="remind me about Dr Rao")
    cid, s = start(ans(intent="request", requests=[req], evidence="remind me about Dr Rao"), ans(yes_no=False, evidence="no"))
    r1 = s.turn("remind me about Dr Rao")
    assert any("Shall I set that" in t for t in r1.reply)
    r2 = s.turn("no")
    assert any("not set" in t for t in r2.reply)
    with session_scope() as q:
        assert q.scalars(select(Reminder).where(Reminder.message.like("Dr Rao%"))).all() == []


def test_a_request_the_parent_never_made_or_in_the_past_is_not_acted_on():
    past = ToolRequest(tool="add_reminder", args={"message": "x", "at": "2020-01-01 10:00"}, evidence="remind me about x")
    cid, s = start(ans(intent="request", requests=[past], evidence="remind me about x"))
    r = s.turn("remind me about x")
    assert any("could not set" in t for t in r.reply)
    invented = ToolRequest(tool="add_reminder", args={"message": "y", "at": "2030-01-01 10:00"}, evidence="remind me about the moon")
    cid2, s2 = start(ans(yes_no=True, items=["dal"], evidence="yes dal", requests=[invented]))
    r2 = s2.turn("yes dal")
    assert not any("Shall I set" in t for t in r2.reply)  # evidence is not in what they said: ignored
    forbidden = ToolRequest(tool="pause_calls", args={"reason": "x"}, evidence="yes dal")
    cid3, s3 = start(ans(yes_no=True, items=["dal"], evidence="yes dal", requests=[forbidden]))
    s3.turn("yes dal")
    with session_scope() as q:
        assert q.get(Parent, 1).calling_enabled


def test_the_filler_is_spoken_live_while_the_tool_runs_and_not_repeated():
    cid, s = start(ans(yes_no=False, evidence="not yet"))
    spoken = []
    s.speak_now = spoken.append
    r = s.turn("not yet")
    assert len(spoken) == 1 and spoken[0] not in r.reply  # went out immediately, so the audio layer must not say it again
    with session_scope() as q:
        from carecall.models import TranscriptTurn
        said = [t.text for t in q.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == cid, TranscriptTurn.speaker == "agent"))]
    assert spoken[0] in said  # but it is in the transcript


def test_a_booking_the_system_refuses_is_never_promised():
    from carecall.agent.db_tools import call_tool
    cid, s = start(ans(intent="callback", callback_minutes=20, evidence="later"))
    with session_scope() as q:
        q.get(Parent, 1).calling_enabled = False
    r = s.turn("call me later")
    text = " ".join(r.reply)
    assert "I will call you again" not in text and "cannot arrange" in text and attempts(kind="callback") == []


def test_regression_a_model_that_answers_the_read_back_with_another_request_cannot_cause_a_loop():
    """Found live: after 'Shall I set that?' the model labelled 'Yes, please' as a NEW request and the agent looped for 10 turns."""
    day = (utcnow().astimezone() + timedelta(days=3)).strftime("%Y-%m-%d")
    req = ToolRequest(tool="add_reminder", args={"message": "Dr Rao appointment", "at": f"{day} 16:30"}, evidence="remind me about Dr Rao")
    again = Interpretation(intent="request", requests=[req], evidence="Yes, please", confidence=0.9)  # the wrong label
    cid, s = start(ans(intent="request", requests=[req], evidence="remind me about Dr Rao"), again)
    s.turn("remind me about Dr Rao")
    r = s.turn("Yes, please. That's right.")
    assert any("Done" in t for t in r.reply)
    with session_scope() as q:
        assert len(q.scalars(select(Reminder).where(Reminder.message.like("Dr Rao%"))).all()) == 1


def test_an_answer_that_also_asks_for_a_callback_does_both():
    cid, s = start(ans(yes_no=True, items=["dal", "rice"], evidence="I had dal and rice", callback_minutes=20))
    r = s.turn("I had dal and rice, call me in twenty minutes")
    text = " ".join(r.reply)
    assert r.done and "in 20 minutes" in text and len(attempts(kind="callback")) == 1
    with session_scope() as q:
        from carecall.models import MealLog
        assert q.scalars(select(MealLog).where(MealLog.call_id == cid)).first() is not None or s.graph.get_state(s.config).values.get("pending_meal")


def test_regression_the_time_the_parent_asked_for_beats_our_default():
    """Found live: 'not yet, call me in ten minutes' was booked at the built-in 30 minutes."""
    cid, s = start(ans(yes_no=False, evidence="not yet", callback_minutes=10))
    r = s.turn("not yet, call me in ten minutes")
    from carecall.models import as_utc
    mins = (as_utc(attempts(kind="callback")[0][4]) - utcnow()).total_seconds() / 60
    assert "in 10 minutes" in " ".join(r.reply) and 9 <= mins <= 10.6


def test_regression_evidence_quoted_in_two_pieces_is_accepted_but_invented_words_are_not():
    from carecall.agent.graph import validate_interpretation
    step = {"kind": "yes_no", "handler": "meal_ate", "prompt": "q", "id": "ate"}
    utt = "Yes, I had dal rice, but I am sleepy now. Could you call me back around 4"
    ok, why = validate_interpretation(ans(yes_no=True, evidence="Yes, I had dal rice... call me back around 4"), step, utt)
    assert why is None and ok.intent == "answer"
    bad, why = validate_interpretation(ans(yes_no=True, evidence="Yes, I had dal rice... and a big sandwich"), step, utt)
    assert why == "evidence_missing"


def test_a_clock_time_asked_for_is_said_as_a_clock_time():
    cid, s = start(ans(intent="callback", callback_at="23:30", evidence="x"))
    assert "at 7" in " ".join(s.turn("call me late").reply)
