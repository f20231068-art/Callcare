from carecall.telephony.audio import FRAME_BYTES, frames, mulaw_to_pcm16, pcm16_to_mulaw, resample_pcm16


def test_mulaw_roundtrip_is_close():
    import math
    import struct
    pcm = b"".join(struct.pack("<h", int(8000 * math.sin(i / 8))) for i in range(800))
    back = mulaw_to_pcm16(pcm16_to_mulaw(pcm))
    assert len(back) == len(pcm)
    a = struct.unpack("<400h", back[:800]); b = struct.unpack("<400h", pcm[:800])
    assert max(abs(x - y) for x, y in zip(a, b)) < 600  # mu-law is lossy but close


def test_resample_16k_to_8k_halves_length():
    out, _ = resample_pcm16(b"\x00\x00" * 1600, 16000, 8000)
    assert abs(len(out) - 1600) <= 4


def test_frames_are_exactly_20ms_and_padded():
    fr = list(frames(b"\x7f" * 250))
    assert len(fr) == 2 and all(len(f) == FRAME_BYTES for f in fr)
    assert fr[1].endswith(b"\xff")
