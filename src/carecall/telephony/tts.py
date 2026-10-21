"""Text-to-speech interface + a fake for tests + the Gnani adapter (to be implemented).

`synthesize` yields 16-bit mono PCM chunks at `sample_rate`. The bridge resamples to 8 kHz mu-law."""
from typing import AsyncIterator, Protocol


class TTSProvider(Protocol):
    sample_rate: int

    def synthesize(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]: ...


class FakeTTS:
    """Records what would have been spoken and yields silence of roughly realistic length."""
    sample_rate = 16000

    def __init__(self):
        self.spoken: list[str] = []
        self.speeds: list[float] = []

    async def synthesize(self, text: str, speed: float = 1.0) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        self.speeds.append(speed)
        chunk = b"\x00\x00" * (self.sample_rate // 50)  # 20 ms
        for _ in range(max(2, len(text.split()) // 2)):  # ~0.4 s per 2 words (short, keeps tests fast)
            yield chunk


def GnaniTTS(*a, **k):  # real adapter lives in gnani_speech.py
    from .gnani_speech import GnaniTTS as _G

    return _G(*a, **k)
