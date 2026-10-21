"""Twilio simulator: we play Twilio's side of the Media Streams protocol against our real server code.
STT is scripted (frames in -> canned transcript out) and TTS is fake, so no network, audio models or phone needed."""
import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from starlette.websockets import WebSocketDisconnect

from carecall.api import create_app
from carecall.calls.service import create_call
from carecall.db import session_scope
from carecall.models import CallAttempt, CallSummary, MealLog, TranscriptTurn
from carecall.telephony.stt import ScriptedSTT
from carecall.telephony.tts import FakeTTS
from carecall.telephony.twilio_api import make_token

FRAME = json.dumps({"event": "media", "media": {"track": "inbound", "payload": "/w=="}})  # one mu-law byte


def start_msg(call_id, token=None):
    return json.dumps({"event": "start", "start": {"streamSid": "MZ1", "customParameters": {
        "call_id": str(call_id), "token": token if token is not None else make_token(call_id)}}})


def read_until(ws, event):
    """Collect server messages until `event`; echo marks like Twilio does after playback."""
    seen = []
    while True:
        m = json.loads(ws.receive_text())
        seen.append(m)
        if m["event"] == "mark":
            ws.send_text(json.dumps({"event": "mark", "mark": m["mark"]}))
        if m["event"] == event:
            return seen


def speak(ws, frames=3):
    for _ in range(frames):
        ws.send_text(FRAME)


def wait_for(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def make_client(utterances, silence_timeout=5.0, gnani_like=False):
    tts = FakeTTS()
    app = create_app(stt_factory=lambda: ScriptedSTT(utterances, gnani_like=gnani_like), tts_factory=lambda: tts, silence_timeout=silence_timeout)
    return TestClient(app), tts


def test_full_call_over_twilio_protocol():
    call_id, _ = create_call("lunch_checkin", channel="twilio")
    with make_client(["yes I had lunch", "rice and dal"])[0] as client:
        with client.websocket_connect("/twilio/stream") as ws:
            ws.send_text(json.dumps({"event": "connected"}))
            ws.send_text(start_msg(call_id))
            opening = read_until(ws, "mark")
            assert sum(m["event"] == "media" for m in opening) > 0  # agent audio reached Twilio
            speak(ws); read_until(ws, "mark")          # "yes I had lunch" -> "What did you have?"
            speak(ws)                                   # "rice and dal" -> closing, then server hangs up
            read_until(ws, "mark")
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
        assert wait_for(lambda: _status(call_id) == "completed")
    with session_scope() as s:
        (log,) = s.scalars(select(MealLog)).all()
        assert log.items == ["rice", "dal"]
        assert "lunch" in s.get(CallSummary, call_id).summary_text.lower()


def _status(call_id):
    with session_scope() as s:
        return s.get(CallAttempt, call_id).status


def test_tts_receives_the_agent_lines():
    call_id, _ = create_call("metformin_reminder", channel="twilio")
    client, tts = make_client(["yes I have taken it"])
    with client:
        with client.websocket_connect("/twilio/stream") as ws:
            ws.send_text(start_msg(call_id))
            read_until(ws, "mark")
            speak(ws); read_until(ws, "mark")
    assert any("Metformin" in t and "15 minutes before your lunch" in t for t in tts.spoken)


def test_bad_token_is_rejected_and_no_session_created():
    call_id, _ = create_call("lunch_checkin", channel="twilio")
    client, tts = make_client([])
    with client:
        with client.websocket_connect("/twilio/stream") as ws:
            ws.send_text(start_msg(call_id, token="forged"))
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
    assert tts.spoken == []
    assert _status(call_id) == "scheduled"


def test_barge_in_sends_clear_while_agent_is_speaking():
    call_id, _ = create_call("lunch_checkin", channel="twilio")
    client, _ = make_client(["wait"])
    with client:
        with client.websocket_connect("/twilio/stream") as ws:
            ws.send_text(start_msg(call_id))
            # read opening audio but do NOT echo the mark: Twilio is still 'playing'
            while json.loads(ws.receive_text())["event"] != "mark":
                pass
            speak(ws)  # parent talks over the agent -> partial transcript -> clear
            seen = []
            while True:
                m = json.loads(ws.receive_text())
                seen.append(m["event"])
                if m["event"] == "clear":
                    break
            assert "clear" in seen


def test_barge_in_works_with_gnani_style_stt_that_sends_speech_start_but_no_partial_text():
    call_id, _ = create_call("lunch_checkin", channel="twilio")
    client, _ = make_client(["wait"], gnani_like=True)
    with client:
        with client.websocket_connect("/twilio/stream") as ws:
            ws.send_text(start_msg(call_id))
            while json.loads(ws.receive_text())["event"] != "mark":
                pass
            speak(ws)
            seen = []
            while True:
                m = json.loads(ws.receive_text())
                seen.append(m["event"])
                if m["event"] == "clear":
                    break
            assert "clear" in seen


def test_silence_feeds_empty_turn_and_agent_reasks():
    call_id, _ = create_call("lunch_checkin", channel="twilio")
    client, tts = make_client([], silence_timeout=0.2)
    with client:
        with client.websocket_connect("/twilio/stream") as ws:
            ws.send_text(start_msg(call_id))
            read_until(ws, "mark")   # opening; we echo the mark -> silence timer starts
            read_until(ws, "mark")   # silence fires -> agent: "Are you still there?"
    assert any(t.startswith("Are you still there?") for t in tts.spoken)


def test_stream_drop_midcall_marks_call_abandoned_and_still_post_processes():
    call_id, _ = create_call("lunch_checkin", channel="twilio")
    client, _ = make_client([])
    with client:
        with client.websocket_connect("/twilio/stream") as ws:
            ws.send_text(start_msg(call_id))
            read_until(ws, "mark")
            ws.send_text(json.dumps({"event": "stop"}))  # parent hung up
        assert wait_for(lambda: _status(call_id) == "abandoned")
        assert wait_for(lambda: _has_summary(call_id))


def _has_summary(call_id):
    with session_scope() as s:
        return s.get(CallSummary, call_id) is not None


def test_voice_webhook_returns_stream_twiml_with_token():
    call_id, _ = create_call("lunch_checkin", channel="twilio")
    client, _ = make_client([])
    with client:
        r = client.post(f"/twilio/voice?call_id={call_id}", data={"CallSid": "CA1"})
    assert r.status_code == 200 and "<Stream" in r.text and "wss://example.test/twilio/stream" in r.text
    assert make_token(call_id) in r.text


def test_status_webhook_is_idempotent_and_ordered():
    call_id, _ = create_call("lunch_checkin", channel="twilio")
    client, _ = make_client([])
    with client:
        for status in ["ringing", "ringing", "in-progress", "no-answer"]:  # dup ringing, then illegal no-answer
            assert client.post(f"/twilio/status?call_id={call_id}", data={"CallStatus": status}).status_code == 204
    assert _status(call_id) == "in_progress"
