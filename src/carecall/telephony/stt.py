"""Speech-to-text interface + a scripted fake for tests + the Gnani adapter (to be implemented).

The bridge only depends on this interface, so swapping providers (or testing without audio) is a
one-line change. Audio in is 16-bit PCM mono at 8 kHz (what the bridge produces from Twilio)."""
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class SttEvent:
    text: str
    is_final: bool
    confidence: float = 1.0
    kind: str = "transcript"  # or "speech_start" (a segment opened; its transcript will follow)


class STTProvider(Protocol):
    async def start(self) -> None: ...
    async def send_audio(self, pcm16_8k: bytes) -> None: ...
    def events(self) -> AsyncIterator[SttEvent]: ...
    async def close(self) -> None: ...


class ScriptedSTT:
    """Pretends to transcribe: after `frames_per_utterance` audio frames it 'hears' the next scripted line.
    Lets us exercise the whole Twilio bridge without any real speech model."""

    def __init__(self, utterances: list[str], frames_per_utterance: int = 3, gnani_like: bool = False):
        self.gnani_like = gnani_like  # Gnani sends 'speech started' then the final text, never partial text
        self.utterances = list(utterances)
        self.n = frames_per_utterance
        self._count = 0
        self._q: asyncio.Queue[SttEvent | None] = asyncio.Queue()

    async def start(self) -> None:
        pass

    async def send_audio(self, pcm16_8k: bytes) -> None:
        self._count += 1
        if self.utterances and self._count % self.n == 0:
            text = self.utterances.pop(0)
            if self.gnani_like:
                await self._q.put(SttEvent(text="", is_final=False, kind="speech_start"))
            else:
                await self._q.put(SttEvent(text=text[: max(1, len(text) // 2)], is_final=False))  # a partial first
            await self._q.put(SttEvent(text=text, is_final=True, confidence=0.95))

    async def events(self) -> AsyncIterator[SttEvent]:
        while True:
            ev = await self._q.get()
            if ev is None:
                return
            yield ev

    async def close(self) -> None:
        await self._q.put(None)


def GnaniSTT(*a, **k):  # real adapter lives in gnani_speech.py (imported lazily: needs websockets + key)
    from .gnani_speech import GnaniSTT as _G

    return _G(*a, **k)
