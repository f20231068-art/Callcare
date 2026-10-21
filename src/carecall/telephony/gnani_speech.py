"""Real Gnani speech adapters, written against docs.gnani.ai.

GnaniSTT  : Prisma realtime WebSocket (wss://api.vachana.ai/stt/v3/stream).
            - config goes in the WebSocket upgrade HEADERS (key, language, sample rate, VAD tuning)
            - audio must be binary frames of EXACTLY 1024 bytes (512 samples) at real-time cadence
            - the server does voice-activity detection and sends ONE `transcript` message per speech segment
GnaniTTS  : Timbre REST (POST /api/v1/tts/inference). Raw 16-bit PCM at 8 kHz, ready for the telephony path.

Both force IPv4 (see net.py): on a network with broken IPv6 the default resolver order stalls every connection."""
import asyncio
import json
import logging
import socket
from typing import AsyncIterator

import websockets

from ..config import get_settings
from ..net import async_http_client
from .stt import SttEvent

log = logging.getLogger(__name__)
STT_URL = "wss://api.vachana.ai/stt/v3/stream"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
SSE_URL = TTS_URL.replace("/inference", "/sse")
FRAME = 1024  # bytes per STT frame, mandated by the protocol


class GnaniSTT:
    """Streaming speech-to-text. Input: 16-bit mono PCM at `sample_rate` (8000 for phone audio)."""

    def __init__(self, language: str = "en-IN", sample_rate: int = 8000, min_silence_ms: int = 600,
                 vad_threshold: float = 0.7, min_speech_ms: int = 200, api_key: str | None = None):
        self.api_key = api_key or get_settings().gnani_api_key
        self.headers = {
            "x-api-key-id": self.api_key, "lang_code": language, "x-sample-rate": str(sample_rate),
            "x-vad-threshold": str(vad_threshold), "x-min-silence-ms": str(min_silence_ms),
            "x-min-speech-ms": str(min_speech_ms),
        }
        self.frame_ms = FRAME / 2 / sample_rate * 1000  # 64 ms at 8 kHz
        self._ws = None
        self._buf = bytearray()
        self._q: asyncio.Queue[SttEvent | None] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self.segments: list[dict] = []  # raw transcript messages, for audits

    async def start(self) -> None:
        kw = {"family": socket.AF_INET} if get_settings().force_ipv4 else {}
        self._ws = await websockets.connect(STT_URL, additional_headers=self.headers, open_timeout=20, **kw)
        self._reader = asyncio.create_task(self._read())

    async def send_audio(self, pcm16: bytes) -> None:
        """Re-chunk arbitrary-size audio into the exact 1024-byte frames the server requires."""
        self._buf += pcm16
        while len(self._buf) >= FRAME:
            frame = bytes(self._buf[:FRAME])
            del self._buf[:FRAME]
            if self._ws is not None:
                try:
                    await self._ws.send(frame)
                except websockets.ConnectionClosed:
                    return

    async def _read(self) -> None:
        try:
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                if m.get("type") == "speech_start":
                    await self._q.put(SttEvent(text="", is_final=False, kind="speech_start"))
                elif m.get("type") == "transcript" and m.get("text", "").strip():
                    self.segments.append(m)
                    await self._q.put(SttEvent(text=m["text"].strip(), is_final=True, confidence=1.0))
                elif m.get("type") == "error":
                    log.warning("Gnani STT error: %s", m.get("message"))
        except websockets.ConnectionClosed:
            pass
        finally:
            await self._q.put(None)

    async def events(self) -> AsyncIterator[SttEvent]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._reader:
            self._reader.cancel()


class GnaniTTS:
    """Text-to-speech via Timbre. `synthesize` yields exact 20 ms chunks of 16-bit PCM at 8 kHz.

    Two ways to get the audio, same voice, same single request for the whole line (so the prosody, the emotion, is the model's own,
    never cut up by us):
      streaming (default)  SSE: the first audio arrives while the rest is still being generated, and is played at once.
      REST                 one response with the entire clip; playback starts only after it is complete.
    The bridge pads the last frame of every chunk with silence, so we re-cut the stream into EXACT 20 ms frames: a chunk boundary
    from the server never becomes an audible gap."""
    sample_rate = 8000

    def __init__(self, voice: str = "Kaveri", language: str = "en-IN", model: str = "timbre-v2.5",
                 api_key: str | None = None, streaming: bool | None = None):
        self.api_key, self.voice, self.language, self.model = api_key or get_settings().gnani_api_key, voice, language, model
        self.streaming = get_settings().tts_streaming if streaming is None else streaming
        self._http = async_http_client(60)
        self.log: list[dict] = []  # per line: mode, time to first audio, total time (for audits)

    def _body(self, text: str, speed: float) -> dict:
        return {"text": text, "voice": self.voice, "model": self.model, "language": self.language,
                "speed": min(1.15, max(0.85, speed)),
                "audio_config": {"sample_rate": self.sample_rate, "num_channels": 1, "sample_width": 2,
                                 "encoding": "linear_pcm", "container": "raw"}}

    async def synthesize(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
        if self.streaming:
            started = False
            try:
                async for chunk in self._stream(text, speed):
                    started = True
                    yield chunk
                return
            except Exception as e:  # noqa: BLE001
                if started:  # part of the line is already on the phone: do not speak it twice, just stop
                    log.warning("Gnani TTS stream broke mid-line: %s", e)
                    return
                log.warning("Gnani TTS streaming unavailable (%s); using the whole-clip endpoint", e)
        async for chunk in self._whole(text, speed):
            yield chunk

    async def _whole(self, text: str, speed: float) -> AsyncIterator[bytes]:
        import time

        t0 = time.perf_counter()
        r = await self._http.post(TTS_URL, headers={"X-API-Key-ID": self.api_key}, json=self._body(text, speed))
        r.raise_for_status()
        done = time.perf_counter() - t0
        self.log.append({"mode": "rest", "text": text, "first_audio_s": round(done, 3), "total_s": round(done, 3), "bytes": len(r.content)})
        audio, step = r.content, self.sample_rate // 50 * 2  # 20 ms
        for i in range(0, len(audio), step):
            yield audio[i:i + step]

    async def _stream(self, text: str, speed: float) -> AsyncIterator[bytes]:
        import base64
        import time

        step = self.sample_rate // 50 * 2  # 320 bytes = 20 ms of 16-bit mono at 8 kHz
        t0, first, total, buf, event = time.perf_counter(), None, 0, b"", ""
        async with self._http.stream("POST", SSE_URL, headers={"X-API-Key-ID": self.api_key, "Accept": "text/event-stream"},
                                     json=self._body(text, speed)) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    if event == "error":
                        raise RuntimeError(line[5:].strip()[:200])
                    if event != "chunk":
                        continue
                    buf += base64.b64decode(json.loads(line[5:]).get("audio", ""))
                    while len(buf) >= step:  # only whole 20 ms frames leave this function
                        if first is None:
                            first = time.perf_counter() - t0
                        total += step
                        yield buf[:step]
                        buf = buf[step:]
        if buf:  # the last partial frame: pad with silence (the clip's own tail)
            yield buf + b"\x00" * (step - len(buf))
            total += step
        if first is None:
            raise RuntimeError("the stream ended without audio")
        self.log.append({"mode": "sse", "text": text, "first_audio_s": round(first, 3), "total_s": round(time.perf_counter() - t0, 3), "bytes": total})
