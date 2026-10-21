"""Live audio pipeline test with NO Twilio carrier: this file plays Twilio's part against the RUNNING server.

    parent line (text) -> Gnani TTS -> 8 kHz mu-law, 20 ms frames -> WebSocket /twilio/stream  (exactly what Twilio sends)
        server: Gnani STT -> LangGraph agent (LLM + tools + database) -> Gnani TTS -> mu-law frames -> WebSocket back
    we decode the audio the server sent back and read it with an independent Gnani STT call, and time the gaps.

What it proves: the stream handshake and token, STT -> agent -> TTS end to end, barge-in-free turn taking, the silence handling,
the database writes and the call-back booking. What it cannot prove: the phone network leg (that needs a paid Twilio account).
A throwaway parent with a dead phone number is used, so nothing can ring a real phone; any call-back the agent books is cancelled."""
import asyncio
import base64
import json
import time

import websockets
from sqlalchemy import select

from ..calls.outcome import cancel_pending
from ..calls.service import create_call
from ..config import get_settings
from ..db import init_db, session_scope
from ..models import Caregiver, CallAttempt, Event, MealProfile, Parent, ParentCaregiver, Routine, TranscriptTurn
from ..agent.routines import CARETAKER, DEFAULT_MEAL_TIMES, MEALS
from ..telephony.audio import frames, mulaw_to_pcm16, pcm16_to_mulaw
from ..telephony.gnani_speech import GnaniTTS
from ..telephony.twilio_api import make_token
from .voice_loopback import transcribe

SILENCE = b"\xff" * 160  # one 20 ms mu-law frame of silence
DEFAULT_LINES = ["Not yet. I will eat later, please call me back in three minutes.", "Okay, that is fine. Thank you, goodbye."]


def _throwaway_call(routine: str = "lunch_checkin") -> tuple[int, int]:
    init_db()
    with session_scope() as s:
        p = Parent(name="Audio Test", preferred_name="Test", phone="+910000000000", language="en", timezone=get_settings().default_timezone)
        cg = Caregiver(name="Audio Test Caregiver", phone="", alert_mode="digest")
        s.add_all([p, cg])
        s.flush()
        s.add(ParentCaregiver(parent_id=p.id, caregiver_id=cg.id, relationship_="child"))
        for meal, t in DEFAULT_MEAL_TIMES.items():
            s.add(MealProfile(parent_id=p.id, meal_type=meal, usual_time=t, source=CARETAKER))
        for meal in MEALS:
            s.add(Routine(parent_id=p.id, key=f"{meal}_checkin", type="meal_checkin", config={"meal_type": meal}))
        s.flush()
        pid = p.id
    cid, _ = create_call(routine, parent_id=pid, channel="text", kind="manual")  # channel 'text': the dispatcher never dials it
    return pid, cid


async def run(lines: list[str] | None = None, host: str = "127.0.0.1:8000", routine: str = "lunch_checkin", verbose: bool = True) -> dict:
    lines = lines or DEFAULT_LINES
    pid, call_id = _throwaway_call(routine)
    tts = GnaniTTS(voice="Devika")
    spoken = {}  # synthesise every parent line first, so a real person's quick reply is not delayed by our own text-to-speech
    for line in lines:
        spoken[line] = frames(pcm16_to_mulaw(b"".join([c async for c in tts.synthesize(line)])))
    t0 = time.monotonic()
    out_q: asyncio.Queue[bytes] = asyncio.Queue()  # mu-law frames to send as the "caller"
    replies: list[dict] = []  # one entry per agent reply: audio bytes, first-frame time, text later
    state = {"last_voice_sent": None, "play_end": 0.0, "stop": False, "cur": None, "closed_by_server": False}
    uri = f"ws://{host}/twilio/stream"
    log = lambda m: print(f"[{time.monotonic() - t0:6.2f}s] {m}", flush=True) if verbose else None  # noqa: E731

    async with websockets.connect(uri, open_timeout=15) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "start": {"streamSid": "MZaudiotest", "callSid": "CAaudiotest",
                                                              "customParameters": {"call_id": str(call_id), "token": make_token(call_id)}}}))
        log(f"stream opened for call {call_id} (throwaway parent {pid})")

        async def feeder():  # Twilio sends a frame every 20 ms, speech or silence, for the whole call
            while not state["stop"]:
                try:
                    fr = out_q.get_nowait()
                    state["last_voice_sent"] = time.monotonic() if fr is not SILENCE else state["last_voice_sent"]
                except asyncio.QueueEmpty:
                    fr = SILENCE
                try:
                    await ws.send(json.dumps({"event": "media", "media": {"track": "inbound", "payload": base64.b64encode(fr).decode()}}))
                except websockets.ConnectionClosed:
                    return
                await asyncio.sleep(0.02)

        marks: asyncio.Queue[str] = asyncio.Queue()

        async def reader():
            try:
                async for raw in ws:
                    m = json.loads(raw)
                    now = time.monotonic()
                    if m.get("event") == "media":
                        if state["cur"] is None or now - state["cur"]["last"] > 2.0:
                            state["cur"] = {"pcm": bytearray(), "first": now, "last": now, "wait_s": None}
                            if state["last_voice_sent"]:
                                state["cur"]["wait_s"] = now - state["last_voice_sent"]
                            replies.append(state["cur"])
                            log(f"agent audio starts ({'%.2fs after the parent stopped talking' % state['cur']['wait_s'] if state['cur']['wait_s'] else 'opening'})")
                        pcm = mulaw_to_pcm16(base64.b64decode(m["media"]["payload"]))
                        state["cur"]["pcm"].extend(pcm)
                        state["cur"]["last"] = now
                        state["play_end"] = max(state["play_end"], now) + 0.02  # Twilio plays 20 ms per frame
                    elif m.get("event") == "mark":
                        await marks.put(m["mark"]["name"])
                    elif m.get("event") == "clear":
                        state["play_end"] = 0.0
            except websockets.ConnectionClosed:
                pass
            state["closed_by_server"] = True

        tasks = [asyncio.create_task(feeder()), asyncio.create_task(reader())]

        async def echo_mark():  # real Twilio echoes a mark once playback reaches it
            name = await asyncio.wait_for(marks.get(), 60)
            await asyncio.sleep(max(0.0, state["play_end"] - time.monotonic()))
            await ws.send(json.dumps({"event": "mark", "mark": {"name": name}}))
            log("agent finished speaking (mark echoed)")

        try:
            await echo_mark()  # the opening
            for line in lines:
                if state["closed_by_server"]:
                    break
                await asyncio.sleep(0.6)
                for fr in spoken[line]:
                    out_q.put_nowait(fr)
                log(f"PARENT says: {line}")
                before = len(replies)
                deadline = time.monotonic() + 40
                while len(replies) == before and time.monotonic() < deadline and not state["closed_by_server"]:
                    await asyncio.sleep(0.05)
                if len(replies) == before:
                    log("no agent reply within 40 s")
                    break
                try:
                    await echo_mark()
                except asyncio.TimeoutError:
                    break
            await asyncio.sleep(2)
        finally:
            state["stop"] = True
            try:
                await ws.send(json.dumps({"event": "stop", "streamSid": "MZaudiotest"}))
            except websockets.ConnectionClosed:
                pass
            for t in tasks:
                t.cancel()

    # the server finishes the call (post-call summary, outcome) after the socket closes
    for _ in range(40):
        with session_scope() as s:
            if s.get(CallAttempt, call_id).status in ("completed", "abandoned", "failed"):
                break
        await asyncio.sleep(1)
    await asyncio.sleep(6)
    heard_back = []
    for r in replies:  # independent check: can a speech recogniser read what the agent said?
        heard_back.append(await transcribe(bytes(r["pcm"])))
    with session_scope() as s:
        call = s.get(CallAttempt, call_id)
        turns = [(t.speaker, t.text) for t in s.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == call_id).order_by(TranscriptTurn.turn_no))]
        ev = [(e.type, e.payload) for e in s.scalars(select(Event).where(Event.call_id == call_id).order_by(Event.id))]
        booked = [(c.id, c.kind, c.scheduled_at) for c in s.scalars(select(CallAttempt).where(CallAttempt.parent_id == pid, CallAttempt.status == "scheduled"))]
        res = {"call_id": call_id, "status": call.status, "outcome": call.outcome, "end_reason": call.end_reason, "turns": turns,
               "tools": [(t, p) for t, p in ev if t.startswith("db_tool.")], "callbacks_booked": booked,
               "reply_wait_s": [round(r["wait_s"], 2) for r in replies if r["wait_s"]], "agent_audio_read_back": heard_back,
               "agent_audio_s": [round(len(r["pcm"]) / 2 / 8000, 1) for r in replies]}
        cancelled = cancel_pending(s, pid, "audio test cleanup: the throwaway parent must never be dialed")
        s.get(Parent, pid).calling_enabled = False
        res["cleanup_cancelled"] = cancelled
    return res
