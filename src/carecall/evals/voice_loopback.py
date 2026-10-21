"""Voice loopback: exercise the persona-on-the-phone path with NO Twilio and NO Gnani agent.

A stand-in plays the far end of the phone (Twilio + a remote agent). The "remote agent" is our own CallSession, so a
conversation runs with REAL audio in both directions:

    our agent text -> Gnani TTS -> 8 kHz mu-law frames -> PersonaCallBridge -> Gnani STT -> persona LLM -> Gnani TTS
        -> mu-law frames back -> Gnani STT (the agent's ears) -> text -> our agent

Two uses: (1) prove the bridge works before wiring Twilio and a Gnani agent; (2) an "audio mode" test of OUR agent, because
the agent now hears speech-to-text output (lowercase, no punctuation, ITN quirks) instead of clean text."""
import asyncio
import base64
import json
import time

from ..agent.interpret import make_interpreter
from ..agent.session import CallSession
from ..calls.service import create_call
from ..db import init_db
from ..llm.factory import make_llm_client
from ..seed import seed
from ..telephony.audio import frames, mulaw_to_pcm16, pcm16_to_mulaw
from ..telephony.gnani_speech import FRAME, GnaniSTT, GnaniTTS
from ..telephony.persona_bridge import PersonaCallBridge, PersonaJob, make_job_token, voice_for
from .personas import LLMPersona

SILENCE = b"\xff" * 160  # one 20 ms frame of mu-law silence


class LoopWS:
    """In-memory stand-in for the Twilio WebSocket, seen from the bridge's side."""

    def __init__(self):
        self.to_bridge: asyncio.Queue = asyncio.Queue()
        self.from_bridge: asyncio.Queue = asyncio.Queue()

    async def receive_text(self):
        m = await self.to_bridge.get()
        if m is None:
            raise ConnectionError("closed")
        return m

    async def send_text(self, s):
        await self.from_bridge.put(s)

    async def close(self, code=1000):
        await self.to_bridge.put(None)


async def transcribe(pcm8k: bytes, min_silence_ms: int = 500) -> str:
    """Run captured audio through Gnani STT at real-time cadence and join the segments (this is the agent's hearing)."""
    stt = GnaniSTT(min_silence_ms=min_silence_ms)
    await stt.start()
    heard: list[str] = []

    async def listen():
        async for ev in stt.events():
            if ev.kind == "transcript":
                heard.append(ev.text)

    task = asyncio.create_task(listen())
    audio = pcm8k + b"\x00" * (8000 * 2 * 3)
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        await stt.send_audio(audio[i:i + FRAME])
        await asyncio.sleep(FRAME / 2 / 8000)
    await asyncio.sleep(3.0)  # transcripts land ~2 s after the speech ends
    await stt.close()
    await task
    return " ".join(heard).strip()


async def run_loopback(card: dict, routine: str | None = None, max_exchanges: int = 14, verbose: bool = True) -> dict:
    """One persona vs our agent, over audio. Returns the transcript as the AGENT experienced it, plus audio diagnostics."""
    init_db("sqlite:///:memory:")
    seed(meal_profile=card["routine"] != "meal_discovery", preferred_name=card["name"].split(",")[0])
    call_id, _ = create_call(routine or card["routine"], channel="voice-loopback")
    client = make_llm_client()
    session = CallSession(call_id, make_interpreter(client), channel="voice-loopback")

    job = PersonaJob(id="loop", card=card, token=make_job_token("loop"))
    persona = LLMPersona(card, client.with_temperature(0.7) if hasattr(client, "with_temperature") else client)
    ws = LoopWS()
    bridge = PersonaCallBridge(ws, job, GnaniSTT(), GnaniTTS(voice=voice_for(card)), persona, quiet_s=1.4, hello_after_s=60)
    bridge_task = asyncio.create_task(bridge.run())

    agent_tts = GnaniTTS(voice="Devika")
    outbound: list[bytes] = []  # mu-law frames the "remote agent" will play, consumed at 20 ms cadence
    rx_pcm = bytearray()  # persona audio as it arrives from the bridge
    rx_last = [0.0]
    stats = {"agent_tts_s": [], "heard_by_agent": [], "agent_said": []}
    stop = asyncio.Event()

    await ws.to_bridge.put(json.dumps({"event": "start", "start": {"streamSid": "MZ-loop", "customParameters": {
        "job_id": "loop", "token": job.token}}}))

    async def line_pump():  # Twilio streams a frame every 20 ms, speech or silence
        while not stop.is_set():
            fr = outbound.pop(0) if outbound else SILENCE
            await ws.to_bridge.put(json.dumps({"event": "media", "media": {"track": "inbound",
                                                                          "payload": base64.b64encode(fr).decode()}}))
            await asyncio.sleep(0.02)

    async def rx_pump():
        while not stop.is_set():
            try:
                m = json.loads(await asyncio.wait_for(ws.from_bridge.get(), 0.2))
            except asyncio.TimeoutError:
                continue
            if m.get("event") == "media":
                rx_pcm.extend(mulaw_to_pcm16(base64.b64decode(m["media"]["payload"])))
                rx_last[0] = time.monotonic()

    pumps = [asyncio.create_task(line_pump()), asyncio.create_task(rx_pump())]

    async def agent_speaks(lines: list[str]):
        for line in lines:
            pcm = b"".join([c async for c in agent_tts.synthesize(line)])
            stats["agent_tts_s"].append(agent_tts.log[-1]["latency_s"])
            stats["agent_said"].append(line)
            outbound.extend(frames(pcm16_to_mulaw(pcm)))
            if verbose:
                print(f"  AGENT > {line}", flush=True)
        while outbound:  # wait until our own audio has been played out
            await asyncio.sleep(0.05)

    try:
        await agent_speaks(session.open())
        for _ in range(max_exchanges):
            if session.done:
                break
            # wait for the persona to start talking (or give up: silence)
            t_wait = time.monotonic()
            while time.monotonic() - t_wait < 25 and (not rx_pcm or time.monotonic() - rx_last[0] < 1.6):
                await asyncio.sleep(0.1)
                if not rx_pcm and time.monotonic() - t_wait > 25:
                    break
            audio = bytes(rx_pcm)
            rx_pcm.clear()
            heard = await transcribe(audio) if len(audio) > 4000 else ""
            stats["heard_by_agent"].append(heard)
            if verbose:
                print(f"  PARENT (as heard by agent STT) > {heard or '(silence)'}", flush=True)
            res = await asyncio.to_thread(session.turn, heard)
            await agent_speaks(res.reply)
    finally:
        stop.set()
        await ws.to_bridge.put(json.dumps({"event": "stop"}))
        await asyncio.wait_for(bridge_task, 20)
        for p in pumps:
            p.cancel()
    await asyncio.to_thread(session.finish, not session.done)
    return {"session": session, "job": job, "call_id": call_id, "stats": stats, "tts_persona": bridge.tts.log}
