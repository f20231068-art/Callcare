"""Twilio Media Streams <-> CallSession bridge.

Twilio opens a WebSocket to us when the parent picks up. Messages (JSON text frames):
  in : connected | start{streamSid, customParameters} | media{payload=base64 mulaw 8k} | mark | stop
  out: media{streamSid, payload} (audio to play) | mark{name} (echoed when playback reaches it) | clear (stop playing)

Tasks inside one call:
  receive loop  : Twilio -> STT audio, marks, stop
  pump          : STT events -> inbox queue (and cancels the silence timer when the parent speaks)
  consumer      : inbox -> CallSession.turn (in a worker thread) -> TTS -> Twilio
Barge-in: a partial transcript while we are still playing audio -> send `clear` so the parent isn't talked over.
Silence: if nobody speaks for `silence_timeout` after we finish, feed an empty utterance (the agent re-asks)."""
import asyncio
import base64
import contextlib
import json
import logging
import time

from ..agent.interpret import Interpreter

try:
    import audioop
except ModuleNotFoundError:  # pragma: no cover
    import audioop_lts as audioop
from ..agent.session import CallSession
from .audio import TWILIO_RATE, frames, mulaw_to_pcm16, pcm16_to_mulaw, resample_pcm16
from .stt import STTProvider, SttEvent
from .tts import TTSProvider
from .twilio_api import verify_token

log = logging.getLogger(__name__)


class StreamBridge:
    def __init__(self, ws, stt: STTProvider, tts: TTSProvider, interpreter: Interpreter | None = None,
                 silence_timeout: float = 7.0, playback_timeout: float = 15.0):
        self.ws, self.stt, self.tts, self.interpreter = ws, stt, tts, interpreter
        self.silence_timeout, self.playback_timeout = silence_timeout, playback_timeout
        self.session: CallSession | None = None
        self.stream_sid: str | None = None
        self.inbox: asyncio.Queue[SttEvent] = asyncio.Queue()
        self.playing = False
        self._mark_n = 0
        self._mark_done = asyncio.Event()
        self._silence_task: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._resample_state = None
        self.closing = False
        self.open_segments = 0  # STT segments opened but not yet transcribed: the parent is (or was just) speaking
        self.last_voice = 0.0  # monotonic time of the last inbound frame with real energy

    # ---------- main loop ----------
    async def run(self):
        try:
            while True:
                try:
                    raw = await self.ws.receive_text()
                except Exception:  # disconnect
                    break
                msg = json.loads(raw)
                ev = msg.get("event")
                if ev == "start":
                    if not await self._on_start(msg["start"]):
                        with contextlib.suppress(Exception):
                            await self.ws.close(code=1008)  # policy violation: bad/missing call token
                        break
                elif ev == "media":
                    pcm = mulaw_to_pcm16(base64.b64decode(msg["media"]["payload"]))
                    if audioop.rms(pcm, 2) > 350:
                        self.last_voice = time.monotonic()
                    await self.stt.send_audio(pcm)
                elif ev == "mark":
                    self._on_mark()
                elif ev == "stop":
                    break
        finally:
            await self._cleanup()

    async def _on_start(self, start: dict) -> bool:
        params = start.get("customParameters", {})
        try:
            call_id = int(params.get("call_id", ""))
        except ValueError:
            return False
        if not verify_token(call_id, params.get("token", "")):
            log.warning("rejected media stream: bad token for call %s", call_id)
            return False
        self.stream_sid = start["streamSid"]
        self.session = CallSession(call_id, self.interpreter, channel="twilio")
        from ..llm.factory import make_judge_client
        self.session.postcall_client = make_judge_client()  # the stronger model decides the next step after hang-up
        loop = asyncio.get_running_loop()

        def speak_now(text: str):  # called from the graph's worker thread while a tool runs: play the filler right away, in order
            asyncio.run_coroutine_threadsafe(self._speak([text]), loop).result(timeout=10)
        self.session.speak_now = speak_now
        await self.stt.start()
        self._tasks = [asyncio.create_task(self._pump()), asyncio.create_task(self._consume())]
        opening = await asyncio.to_thread(self.session.open)
        await self._speak(opening)
        return True

    # ---------- agent -> parent ----------
    async def _send(self, obj: dict):
        await self.ws.send_text(json.dumps({**obj, "streamSid": self.stream_sid}))

    async def _speak(self, lines: list[str], rate: float = 1.0):
        for line in lines:
            async for pcm in self.tts.synthesize(line, speed=rate):
                pcm8, self._resample_state = resample_pcm16(pcm, self.tts.sample_rate, TWILIO_RATE, self._resample_state)
                for fr in frames(pcm16_to_mulaw(pcm8)):
                    await self._send({"event": "media", "media": {"payload": base64.b64encode(fr).decode()}})
        self._mark_n += 1
        self._mark_done.clear()
        self.playing = True
        await self._send({"event": "mark", "mark": {"name": f"m{self._mark_n}"}})

    def _on_mark(self):
        """Twilio echoes our mark when it has finished playing everything before it."""
        self.playing = False
        self._mark_done.set()
        if not self.closing and self.open_segments == 0:  # not while the parent is already talking (e.g. they just interrupted)
            self._arm_silence()

    # ---------- parent -> agent ----------
    def _arm_silence(self):
        self._cancel_silence()
        self._silence_task = asyncio.create_task(self._silence())

    def _cancel_silence(self):
        if self._silence_task and not self._silence_task.done():
            self._silence_task.cancel()

    async def _silence(self):
        await asyncio.sleep(self.silence_timeout)
        await self.inbox.put(SttEvent(text="", is_final=True, confidence=1.0))

    async def _pump(self):
        async for ev in self.stt.events():
            if ev.kind == "speech_start":
                # Gnani sends no partial text: its only early signal is "speech started". That is what proves the parent is talking, so it
                # stops the "are you still there?" timer and interrupts the agent (barge-in).
                self.open_segments += 1
                self._cancel_silence()
                if self.playing:
                    self.playing = False
                    await self._send({"event": "clear"})
            elif ev.is_final:
                self.open_segments = max(0, self.open_segments - 1)
            if ev.text.strip():
                self._cancel_silence()
            await self.inbox.put(ev)

    async def _consume(self):
        while True:
            ev = await self.inbox.get()
            if not ev.is_final:
                if self.playing and ev.text.strip():  # barge-in
                    self.playing = False
                    await self._send({"event": "clear"})
                continue
            res = await asyncio.to_thread(self.session.turn, ev.text, ev.confidence)
            await self._speak(res.reply, res.rate)
            if res.done:
                self.closing = True
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._mark_done.wait(), self.playback_timeout)
                with contextlib.suppress(Exception):
                    await self.ws.close()
                return

    # ---------- teardown ----------
    async def _cleanup(self):
        self._cancel_silence()
        for t in self._tasks:
            t.cancel()
        with contextlib.suppress(Exception):
            await self.stt.close()
        if self.session is not None:
            hangup = {"parent_speaking": self.open_segments > 0 or (time.monotonic() - self.last_voice) < 1.2,
                      "pending_text": ""}
            await asyncio.to_thread(self.session.finish, not self.session.done, hangup)
