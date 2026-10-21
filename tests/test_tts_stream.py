"""Streaming text-to-speech: audio plays while Gnani is still generating it, in exact 20 ms frames, with a safe fallback.
A stand-in server replays Gnani's SSE format (event: start / event: chunk with base64 PCM), so no network is used."""
import asyncio
import base64
import json

import httpx

from carecall.telephony.gnani_speech import GnaniTTS

STEP = 320  # 20 ms of 16-bit mono at 8 kHz


def sse(chunks: list[bytes]) -> list[bytes]:
    out = [b"event: start\n", b'data: {"status": "streaming_started"}\n', b"\n"]
    for i, c in enumerate(chunks, 1):
        out += [b"event: chunk\n", b"data: " + json.dumps({"chunk_index": i, "audio": base64.b64encode(c).decode()}).encode() + b"\n", b"\n"]
    out += [b"event: end\n", b'data: {"status": "done"}\n', b"\n"]
    return out


class Gated(httpx.AsyncByteStream):
    """Yields the first chunk at once, then WAITS for the test to release the rest: proves audio is available before the stream ends."""

    def __init__(self, chunks, gate):
        self.parts, self.gate, self.first_sent = sse(chunks), gate, asyncio.Event()

    async def __aiter__(self):
        for i, part in enumerate(self.parts):
            if i == 6:  # after the first chunk event (3 start lines + 3 chunk lines)
                self.first_sent.set()
                await self.gate.wait()
            yield part


def tts_with(handler, **kw) -> GnaniTTS:
    t = GnaniTTS(api_key="test", **kw)
    t._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return t


def pcm(n_bytes: int, seed: int = 1) -> bytes:
    return bytes((seed * 37 + i * 7) % 251 for i in range(n_bytes))


def test_chunks_of_any_size_come_out_as_exact_20ms_frames_with_no_audio_lost():
    audio = pcm(5000)
    pieces = [audio[:500], audio[500:1277], audio[1277:2000], audio[2000:4321], audio[4321:]]  # none a multiple of 320
    t = tts_with(lambda r: httpx.Response(200, content=b"".join(sse(pieces)), headers={"content-type": "text/event-stream"}))

    async def go():
        return [c async for c in t.synthesize("hello")]

    frames = asyncio.run(go())
    assert all(len(f) == STEP for f in frames)  # the bridge pads short frames with silence: none may be short mid-stream
    joined = b"".join(frames)
    assert joined[:len(audio)] == audio and set(joined[len(audio):]) <= {0}  # every byte, in order; only a silent tail pad
    assert t.log[-1]["mode"] == "sse" and t.log[-1]["first_audio_s"] <= t.log[-1]["total_s"]


def test_the_first_audio_is_available_before_the_stream_has_finished():
    async def go():
        gate = asyncio.Event()
        stream = Gated([pcm(1000), pcm(1000, 2)], gate)
        t = tts_with(lambda r: httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"}))
        it = t.synthesize("hello").__aiter__()
        first = await asyncio.wait_for(it.__anext__(), 5)  # arrives although the server has not sent the rest
        assert not gate.is_set() and len(first) == STEP
        gate.set()
        rest = [c async for c in it]
        return first, rest

    first, rest = asyncio.run(go())
    assert len(first) + sum(len(c) for c in rest) >= 2000


def test_if_streaming_is_unavailable_it_falls_back_to_the_whole_clip():
    audio = pcm(1000)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sse"):
            return httpx.Response(503, text="no")
        return httpx.Response(200, content=audio)

    t = tts_with(handler)

    async def go():
        return b"".join([c async for c in t.synthesize("hello")])

    assert asyncio.run(go())[:1000] == audio and t.log[-1]["mode"] == "rest"


def test_a_stream_that_dies_mid_line_stops_quietly_and_never_repeats_what_was_spoken():
    parts = sse([pcm(1000)])[:6]  # start + one chunk, then the connection ends with no "end" event and no more audio

    t = tts_with(lambda r: httpx.Response(200, content=b"".join(parts), headers={"content-type": "text/event-stream"}))

    async def go():
        return [c async for c in t.synthesize("hello")]

    frames = asyncio.run(go())
    assert frames and all(len(f) == STEP for f in frames)  # what arrived was played; nothing was re-synthesised on top of it


def test_an_error_event_before_any_audio_falls_back_instead_of_going_silent():
    body = b"event: error\ndata: {\"message\": \"bad voice\"}\n\n"

    def handler(request):
        return httpx.Response(200, content=body) if request.url.path.endswith("/sse") else httpx.Response(200, content=pcm(640))

    t = tts_with(handler)

    async def go():
        return b"".join([c async for c in t.synthesize("hi")])

    assert len(asyncio.run(go())) == 640 and t.log[-1]["mode"] == "rest"


def test_streaming_can_be_switched_off():
    called = []

    def handler(request):
        called.append(request.url.path)
        return httpx.Response(200, content=pcm(640))

    t = tts_with(handler, streaming=False)

    async def go():
        return [c async for c in t.synthesize("hi")]

    asyncio.run(go())
    assert called and not any(p.endswith("/sse") for p in called)
