"""Persona-on-the-phone: lets an LLM play an elderly parent on a REAL call with Gnani's agent.

Flow (Gnani's agent places the call; we are the person who picks up):

    Gnani agent --phone call--> Twilio number --webhook/stream--> PersonaCallBridge (this file)
        agent's voice -> Gnani STT -> text -> persona LLM -> reply text -> Gnani TTS -> audio -> back into the call

Turn-taking is the hard part. The STT emits one transcript per speech segment, ~1-2 s after the agent stops, and an
agent turn can span several segments. So the persona replies only when the agent has been QUIET for `quiet_s` (no audio
energy, no open STT segment) and at least one segment is pending. If the agent never speaks, the persona says "Hello?".

Everything the persona hears and says is logged with timestamps into the PersonaJob, which the orchestrator reads back."""
import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from ..config import get_settings
from .audio import frames, mulaw_to_pcm16, pcm16_to_mulaw

try:
    import audioop
except ModuleNotFoundError:  # pragma: no cover
    import audioop_lts as audioop

log = logging.getLogger(__name__)
ENERGY_THRESHOLD = 350  # RMS of 16-bit samples above which the line counts as "someone is talking"


def voice_for(card: dict) -> str:
    """Pick an English (India) Timbre voice that roughly matches the persona."""
    male = re.search(r"\b(he|his|him)\b", (card.get("profile", "") + " " + card.get("name", "")).lower())
    return card.get("voice") or ("Pranav" if male else "Kaveri")


@dataclass
class PersonaJob:
    id: str
    card: dict
    token: str
    state: str = "armed"  # armed -> connected -> done | error
    transcript: list = field(default_factory=list)  # {"speaker","text","t"}
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None
    tts_log: list = field(default_factory=list)
    stt_segments: list = field(default_factory=list)

    def say(self, speaker: str, text: str):
        self.transcript.append({"speaker": speaker, "text": text, "t": round(time.time() - (self.started_at or time.time()), 1)})

    def public(self) -> dict:
        return {"id": self.id, "persona": self.card["id"], "state": self.state, "error": self.error,
                "turns": len(self.transcript), "transcript": self.transcript, "tts_log": self.tts_log}


class JobRegistry:
    """In-memory. One inbound call claims the oldest armed job (the number rings once per job)."""

    def __init__(self):
        self.jobs: dict[str, PersonaJob] = {}

    def create(self, card: dict) -> PersonaJob:
        jid = uuid.uuid4().hex[:10]
        job = PersonaJob(id=jid, card=card, token=make_job_token(jid))
        self.jobs[jid] = job
        return job

    def claim_next(self) -> PersonaJob | None:
        armed = [j for j in self.jobs.values() if j.state == "armed"]
        if not armed:
            return None
        job = armed[0]
        job.state = "claimed"
        return job


REGISTRY = JobRegistry()


def make_job_token(job_id: str) -> str:
    return hmac.new(get_settings().stream_secret.encode(), f"persona:{job_id}".encode(), hashlib.sha256).hexdigest()


def verify_job_token(job_id: str, token: str) -> bool:
    return hmac.compare_digest(make_job_token(job_id), token or "")


class PersonaCallBridge:
    def __init__(self, ws, job: PersonaJob, stt, tts, persona, quiet_s: float = 1.6, hello_after_s: float = 7.0,
                 max_seconds: float = 420, max_persona_turns: int = 30):
        self.ws, self.job, self.stt, self.tts, self.persona = ws, job, stt, tts, persona
        self.quiet_s, self.hello_after_s = quiet_s, hello_after_s
        self.max_seconds, self.max_persona_turns = max_seconds, max_persona_turns
        self.stream_sid: str | None = None
        self.pending: list[str] = []
        self.open_segments = 0
        self.last_activity = time.monotonic()
        self.speaking_until = 0.0
        self.persona_turns = 0
        self.heard_agent = False
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    # ---------- Twilio side ----------
    async def run(self):
        try:
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(self.ws.receive_text(), timeout=5)
                except asyncio.TimeoutError:
                    if self.job.started_at and time.time() - self.job.started_at > self.max_seconds:
                        self.job.error = "max call duration reached"
                        break
                    continue
                except Exception:
                    break
                msg = json.loads(raw)
                ev = msg.get("event")
                if ev == "start":
                    if not await self._on_start(msg["start"]):
                        with contextlib.suppress(Exception):
                            await self.ws.close(code=1008)
                        break
                elif ev == "media" and msg["media"].get("track", "inbound") == "inbound":
                    pcm = mulaw_to_pcm16(base64.b64decode(msg["media"]["payload"]))
                    if audioop.rms(pcm, 2) > ENERGY_THRESHOLD:
                        self.last_activity = time.monotonic()
                    await self.stt.send_audio(pcm)
                elif ev == "stop":
                    break
        finally:
            await self._cleanup()

    async def _on_start(self, start: dict) -> bool:
        p = start.get("customParameters", {})
        if p.get("job_id") != self.job.id or not verify_job_token(self.job.id, p.get("token", "")):
            self.job.error = "bad job token"
            return False
        self.stream_sid = start["streamSid"]
        self.job.state, self.job.started_at = "connected", time.time()
        await self.stt.start()
        self.last_activity = time.monotonic()
        self._tasks = [asyncio.create_task(self._pump()), asyncio.create_task(self._monitor())]
        return True

    async def _send_audio(self, text: str):
        self.persona_turns += 1
        pcm = b""
        async for chunk in self.tts.synthesize(text):
            pcm += chunk
        if getattr(self.tts, "log", None):
            self.job.tts_log.append(self.tts.log[-1])
        for fr in frames(pcm16_to_mulaw(pcm)):
            await self.ws.send_text(json.dumps({"event": "media", "streamSid": self.stream_sid,
                                                "media": {"payload": base64.b64encode(fr).decode()}}))
        self.speaking_until = time.monotonic() + len(pcm) / 16000  # 8 kHz * 2 bytes
        self.last_activity = self.speaking_until

    # ---------- hearing the agent ----------
    async def _pump(self):
        async for ev in self.stt.events():
            self.last_activity = time.monotonic()
            if ev.kind == "speech_start":
                self.open_segments += 1
                continue
            self.open_segments = max(0, self.open_segments - 1)
            self.heard_agent = True
            self.pending.append(ev.text)

    # ---------- deciding when and what to say ----------
    async def _monitor(self):
        said_hello = False
        started = time.monotonic()
        while not self._stop.is_set():
            await asyncio.sleep(0.15)
            now = time.monotonic()
            if now < self.speaking_until:
                continue
            if not self.heard_agent and not said_hello and now - started > self.hello_after_s:
                said_hello = True
                self.job.say("parent", "Hello?")
                await self._send_audio("Hello?")
                continue
            if self.pending and not self.open_segments and now - self.last_activity > self.quiet_s:
                heard, self.pending = " ".join(self.pending), []
                self.job.say("agent", heard)
                if self.persona_turns >= self.max_persona_turns:
                    self.job.error = "persona turn cap reached"
                    self._stop.set()
                    return
                history = [("agent" if t["speaker"] == "agent" else "parent", t["text"]) for t in self.job.transcript]
                reply = await asyncio.to_thread(self.persona.reply, None, history)
                if reply.strip():
                    self.job.say("parent", reply)
                    await self._send_audio(reply)
                else:
                    self.job.say("parent", "")  # silence: the line stays quiet, the agent must cope

    async def _cleanup(self):
        for t in self._tasks:
            t.cancel()
        with contextlib.suppress(Exception):
            await self.stt.close()
        self.job.stt_segments = getattr(self.stt, "segments", [])
        self.job.ended_at = time.time()
        if self.job.state != "error":
            self.job.state = "done"


class WebhookInbox:
    """Post-call payloads POSTed by Gnani when a call ends. Matched to persona jobs by arrival time (calls are sequential)."""

    def __init__(self):
        self.items: list[dict] = []

    def add(self, payload: dict):
        self.items.append({"received": time.time(), "consumed": False, "payload": payload})

    def take_after(self, t: float) -> dict | None:
        for it in self.items:
            if not it["consumed"] and it["received"] >= t:
                it["consumed"] = True
                return it["payload"]
        return None


WEBHOOKS = WebhookInbox()
