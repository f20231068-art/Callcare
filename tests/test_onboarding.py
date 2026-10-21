"""Accounts, onboarding and the dashboard snapshot: a caregiver signs up, adds a parent, and the calendar fills with real calls."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from carecall import accounts
from carecall.api import create_app
from carecall.db import session_scope
from carecall.models import CallAttempt, Parent


def test_phone_numbers_are_normalised_and_bad_ones_refused():
    assert accounts.normalize_phone("98765 43210") == "+919876543210"
    assert accounts.normalize_phone("+1 (415) 555-0100") == "+14155550100"
    assert accounts.normalize_phone("919876543210") == "+919876543210"
    with pytest.raises(accounts.AuthError):
        accounts.normalize_phone("12345")


def test_signup_login_logout_and_wrong_password():
    accounts.signup("Sree@Example.com", "correct horse", "Sree")
    with pytest.raises(accounts.AuthError):
        accounts.signup("sree@example.com", "another one 1", "Sree")  # same email, any case
    with pytest.raises(accounts.AuthError):
        accounts.login("sree@example.com", "wrong password")
    token = accounts.login("sree@example.com", "correct horse")
    assert accounts.account_for_token(token)["email"] == "sree@example.com"
    accounts.logout(token)
    assert accounts.account_for_token(token) is None


def _signed_in(client):
    assert client.post("/auth/signup", json={"name": "Sree", "email": "a@b.co", "password": "longenough"}).status_code == 200
    return client


def test_onboarding_creates_the_parent_routines_and_todays_calls_with_no_hardcoded_data():
    with session_scope() as s:  # the demo parent from the fixture is not what a real user sees
        for p in s.scalars(select(Parent)):
            p.calling_enabled = False
    with TestClient(create_app()) as c:
        _signed_in(c)
        assert c.get("/api/dashboard").json()["parent"] is None  # a new account has no parents yet
        late = (datetime.now(timezone.utc) + timedelta(hours=26)).astimezone()  # irrelevant: meal times are the user's own
        r = c.post("/api/onboarding", json={"parent_name": "Lakshmi Rao", "preferred_name": "Amma", "phone": "98765 43210", "consent": True,
                                            "meals": {"breakfast": "07:45", "lunch": "13:15", "dinner": "20:00"},
                                            "medicines": [{"name": "Metformin", "dose_note": "500mg", "timing": "before_food", "meal_anchor": "lunch"}]})
        assert r.status_code == 201, r.text
        snap = c.get("/api/dashboard").json()["parent"]
        assert snap["phone_masked"].startswith("+91") and "9876543210" not in snap["phone_masked"]
        assert snap["meal_times"]["lunch"] == "13:15"
        assert {x["key"] for x in snap["routines"]} >= {"breakfast_checkin", "lunch_checkin", "dinner_checkin"}
        assert any(x["type"] == "medicine_reminder" and "Metformin" in x["detail"] for x in snap["routines"])
        with session_scope() as s:
            assert s.scalars(select(Parent).where(Parent.phone == "+919876543210")).one().consent_given_at is not None
        assert c.post("/api/onboarding", json={"parent_name": "Again", "phone": "+919876543210", "consent": True}).status_code == 422  # duplicate


def test_consent_is_required_and_other_accounts_cannot_see_a_parent():
    with TestClient(create_app()) as c:
        _signed_in(c)
        assert c.post("/api/onboarding", json={"parent_name": "X", "phone": "9876543210", "consent": False}).status_code == 422
        pid = c.post("/api/onboarding", json={"parent_name": "Y", "phone": "9876543210", "consent": True}).json()["parent_id"]
    with TestClient(create_app()) as other:
        other.post("/auth/signup", json={"name": "Other", "email": "o@b.co", "password": "longenough"})
        assert other.get("/api/dashboard").json()["parent"] is None
        assert other.post(f"/api/parents/{pid}/reminders", json={"message": "hi", "in_minutes": 1}).status_code == 422
        assert other.post(f"/api/parents/{pid}/calling", params={"enabled": False}).status_code == 422


def test_a_reminder_in_one_minute_lands_on_the_calendar_and_the_dispatcher_dials_it_when_due():
    from carecall.calls.scheduler import dispatch_once

    with TestClient(create_app()) as c:
        _signed_in(c)
        pid = c.post("/api/onboarding", json={"parent_name": "Y", "phone": "9876543210", "consent": True}).json()["parent_id"]
        r = c.post(f"/api/parents/{pid}/reminders", json={"message": "Take a walk", "in_minutes": 1})
        assert r.status_code == 201, r.text
        up = [u for u in c.get("/api/dashboard").json()["parent"]["upcoming"] if u["id"] == r.json()["call_id"]]
        assert up and up[0]["routine"].startswith("reminder_") and up[0]["status"] == "scheduled"
        dialed = []
        dispatch_once(lambda cid: dialed.append(cid) or "CA123", now=datetime.now(timezone.utc) + timedelta(seconds=30))
        assert r.json()["call_id"] not in dialed  # not yet due
        dispatch_once(lambda cid: dialed.append(cid) or "CA123", now=datetime.now(timezone.utc) + timedelta(seconds=75))
        assert r.json()["call_id"] in dialed
        with session_scope() as s:
            assert s.get(CallAttempt, r.json()["call_id"]).status == "dialing"
        assert c.post(f"/api/parents/{pid}/reminders", json={"message": "x", "in_minutes": -5}).status_code == 422


def test_admin_routes_refuse_requests_that_came_through_the_tunnel():
    with TestClient(create_app()) as c:
        assert c.get("/admin/parents").status_code == 200
        assert c.get("/admin/parents", headers={"x-forwarded-for": "1.2.3.4"}).status_code == 403
        assert c.post("/calls", json={"routine": "lunch_checkin"}, headers={"x-forwarded-for": "1.2.3.4"}).status_code == 403


def test_voice_webhook_without_a_signature_needs_the_per_call_token_and_a_wrong_signature_is_refused(monkeypatch):
    from carecall.calls.service import create_call
    from carecall.telephony.twilio_api import make_token

    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "x" * 32)
    from carecall.config import get_settings
    get_settings.cache_clear()
    cid, _ = create_call("lunch_checkin", channel="twilio")
    with TestClient(create_app()) as c:
        assert c.post(f"/twilio/voice?call_id={cid}", data={"CallSid": "CA1"}).status_code == 403  # no signature, no token
        assert c.post(f"/twilio/voice?call_id={cid}&t=wrong", data={"CallSid": "CA1"}).status_code == 403
        ok = c.post(f"/twilio/voice?call_id={cid}&t={make_token(cid)}", data={"CallSid": "CA1"})
        assert ok.status_code == 200 and "<Stream" in ok.text
        bad = c.post(f"/twilio/voice?call_id={cid}&t={make_token(cid)}", data={"CallSid": "CA1"}, headers={"X-Twilio-Signature": "nope"})
        assert bad.status_code == 403  # a present-but-wrong signature is never rescued by the token


def test_gather_mode_runs_the_conversation_one_turn_per_request_and_finishes_when_the_agent_is_done(monkeypatch):
    from types import SimpleNamespace

    from carecall.calls.service import create_call
    from carecall.config import get_settings
    from carecall.telephony import gather
    from carecall.telephony.twilio_api import make_token

    class StubSession:
        finished = []

        def __init__(self, call_id, channel="text"):
            self.call_id, self.done = call_id, False

        def open(self):
            return ["Hello Sree. This is your reminder."]

        def turn(self, text, conf=None):
            self.done = "bye" in text
            return SimpleNamespace(reply=["Okay, goodbye."] if self.done else ["Noted. Anything else?"], done=self.done)

        def finish(self, abandoned=False, hangup=None):
            StubSession.finished.append((self.call_id, abandoned))

    monkeypatch.setenv("TWILIO_VOICE_MODE", "gather")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "x" * 32)
    get_settings.cache_clear()
    monkeypatch.setattr(gather, "CALLS", gather.GatherCalls(StubSession))
    monkeypatch.setattr("carecall.llm.factory.make_judge_client", lambda: None)
    cid, _ = create_call("lunch_checkin", channel="twilio")
    t = make_token(cid)
    with TestClient(create_app()) as c:
        first = c.post(f"/twilio/voice?call_id={cid}&t={t}", data={"CallSid": "CA1"})
        assert "<Gather" in first.text and "Hello Sree" in first.text and "<Stream" not in first.text
        mid = c.post(f"/twilio/turn?call_id={cid}&t={t}", data={"SpeechResult": "remind me in 3 minutes"})
        assert "Anything else?" in mid.text and "<Gather" in mid.text
        end = c.post(f"/twilio/turn?call_id={cid}&t={t}", data={"SpeechResult": "ok bye"})
        assert "goodbye" in end.text and "<Hangup" in end.text and "<Gather" not in end.text
        assert c.post(f"/twilio/turn?call_id={cid}", data={"SpeechResult": "x"}).status_code == 403  # no token


def test_gather_mode_says_one_moment_when_the_agent_is_slow_and_answers_when_twilio_comes_back(monkeypatch):
    import time
    from types import SimpleNamespace

    from carecall.calls.service import create_call
    from carecall.config import get_settings
    from carecall.telephony import gather
    from carecall.telephony.twilio_api import make_token

    class SlowSession:
        def __init__(self, call_id, channel="text"):
            self.done = False

        def open(self):
            return ["Hello."]

        def turn(self, text, conf=None):
            time.sleep(0.5)
            return SimpleNamespace(reply=["All sorted."], done=False)

        def finish(self, *a, **k):
            pass

    monkeypatch.setenv("TWILIO_VOICE_MODE", "gather")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "x" * 32)
    get_settings.cache_clear()
    monkeypatch.setattr(gather, "CALLS", gather.GatherCalls(SlowSession, patience_s=0.1))
    monkeypatch.setattr("carecall.llm.factory.make_judge_client", lambda: None)
    cid, _ = create_call("lunch_checkin", channel="twilio")
    t = make_token(cid)
    with TestClient(create_app()) as c:
        c.post(f"/twilio/voice?call_id={cid}&t={t}", data={"CallSid": "CA1"})
        slow = c.post(f"/twilio/turn?call_id={cid}&t={t}", data={"SpeechResult": "hello"})
        assert "One moment" in slow.text and "/twilio/wait" in slow.text and "All sorted" not in slow.text
        time.sleep(0.6)
        back = c.post(f"/twilio/wait?call_id={cid}&t={t}", data={})
        assert "All sorted" in back.text and "<Gather" in back.text


def test_gather_turn_with_no_session_records_the_call_as_abandoned_at_once(monkeypatch):
    """A server restart empties the session dictionary. The call must not stay 'in progress' until the reaper finds it 25 minutes later."""
    import time

    from carecall.calls.service import create_call
    from carecall.calls.state import transition
    from carecall.config import get_settings
    from carecall.models import Event
    from carecall.telephony import gather
    from carecall.telephony.twilio_api import make_token

    monkeypatch.setenv("TWILIO_VOICE_MODE", "gather")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "x" * 32)
    get_settings.cache_clear()
    monkeypatch.setattr(gather, "CALLS", gather.GatherCalls(lambda *a, **k: None))
    monkeypatch.setattr("carecall.llm.factory.make_judge_client", lambda: None)
    cid, _ = create_call("lunch_checkin", channel="twilio")
    with session_scope() as s:
        transition(s, s.get(CallAttempt, cid), "in_progress", "test")
    with TestClient(create_app()) as c:
        r = c.post(f"/twilio/turn?call_id={cid}&t={make_token(cid)}", data={"SpeechResult": "hello"})
        assert "disconnected" in r.text and "<Hangup" in r.text
        for _ in range(40):
            with session_scope() as s:
                done = s.get(CallAttempt, cid).status == "abandoned"
            if done:
                break
            time.sleep(0.1)
        with session_scope() as s:
            assert s.get(CallAttempt, cid).status == "abandoned"
            assert s.scalars(select(Event).where(Event.call_id == cid, Event.type == "call.orphaned")).first() is not None
        for _ in range(40):
            if cid not in gather.CALLS.finishing:
                break
            time.sleep(0.1)
        assert cid not in gather.CALLS.finishing  # nothing is left behind
