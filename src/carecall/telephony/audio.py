"""Audio plumbing for phone calls.

Telephony audio is 8 kHz, 8-bit mu-law (G.711), delivered by Twilio as base64 in 20 ms frames
(160 bytes). Speech models generally want 16-bit PCM, often at 16 kHz. This module converts
between the two. Getting this wrong sounds like static or chipmunks, which is why it is isolated and tested."""
try:
    import audioop  # removed in Python 3.13; the 'audioop-lts' package provides the same module
except ModuleNotFoundError:  # pragma: no cover
    import audioop_lts as audioop

TWILIO_RATE = 8000
FRAME_BYTES = 160  # 20 ms of 8 kHz mu-law (1 byte per sample)


def mulaw_to_pcm16(data: bytes) -> bytes:
    return audioop.ulaw2lin(data, 2)


def pcm16_to_mulaw(data: bytes) -> bytes:
    return audioop.lin2ulaw(data, 2)


def resample_pcm16(data: bytes, from_rate: int, to_rate: int, state=None):
    """Returns (resampled, state). Pass `state` back in for consecutive chunks of one stream
    so chunk boundaries don't click."""
    if from_rate == to_rate:
        return data, state
    return audioop.ratecv(data, 2, 1, from_rate, to_rate, state)


def frames(mulaw: bytes, size: int = FRAME_BYTES):
    """Split mu-law audio into Twilio-sized 20 ms frames (last frame padded with silence 0xFF)."""
    for i in range(0, len(mulaw), size):
        chunk = mulaw[i : i + size]
        yield chunk + b"\xff" * (size - len(chunk))
