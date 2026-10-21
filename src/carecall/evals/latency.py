"""`carecall latency`: how long does a voice turn take, stage by stage?

A turn = the parent stops speaking -> the parent hears the agent start to reply:

    [end of speech] -> STT end-pointing + transcription -> LLM (agent reply) -> TTS (first audio) -> [agent audio begins]

Each stage is measured repeatedly against the real services (real-time audio pacing for STT), then composed. Two pipelines:
  sequential : wait for the full STT text, the FULL LLM reply, the FULL TTS audio (what the code does today)
  streaming  : STT with a shorter silence window, LLM tokens streamed, TTS on the first sentence, first audio chunk (SSE)
The Twilio/phone network adds roughly 100-300 ms more each way; it is not measured here."""
import asyncio
import json
import statistics as st
import time
from datetime import datetime
from pathlib import Path

from ..agent.generative import GenerativeAgent
from ..config import get_settings
from ..llm.factory import make_agent_llm_client, make_llm_client
from ..net import http_client
from ..telephony.gnani_speech import FRAME, TTS_URL, GnaniSTT, GnaniTTS
from .personas import load_personas

PARENT_LINES = ["Yes, I took it just after dinner.", "No, I haven't taken it, it upsets my stomach.",
                "I already ate lunch early. Should I still take it?"]
AGENT_LINES = ["Thank you, that is noted.", "That is a good question for your doctor or pharmacist. I will let Sree know.",
               "It is time for your Metformin. Please take it now, about 15 minutes before your lunch. Have you taken it?"]


def summarize(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "min": round(xs[0], 2), "p50": round(st.median(xs), 2), "p90": round(xs[min(len(xs) - 1, int(len(xs) * .9))], 2),
            "max": round(xs[-1], 2)} if xs else {"n": 0}


def llm_agent_reply(n: int) -> tuple[list[float], list[float]]:
    """Total time and time-to-first-token for the agent's reply (real prompt: ~3.5k chars + conversation)."""
    card = next(p for p in load_personas() if p["usecase"] == "medicine_before_food")
    agent_client = make_agent_llm_client()
    agent = GenerativeAgent(agent_client, card["usecase"], card)
    agent.open()
    totals, ttfts = [], []
    http = http_client(120)
    s = get_settings()
    base = (s.agent_llm_base_url or s.llm_base_url).rstrip("/")
    key = s.agent_llm_api_key or s.llm_api_key
    model = s.agent_llm_model or s.llm_model
    for i in range(n):
        line = PARENT_LINES[i % len(PARENT_LINES)]
        msgs = agent.history + [{"role": "user", "content": line}]
        t0 = time.perf_counter()
        agent_client.chat(agent.system, msgs)
        totals.append(time.perf_counter() - t0)
        body = {"model": model, "stream": True, "temperature": 0.3,
                "messages": [{"role": "system", "content": agent.system}] + msgs, **(agent_client.extra_body or {})}
        t0, first = time.perf_counter(), None
        with http.stream("POST", f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body) as r:
            for raw in r.iter_lines():
                if raw.startswith("data:") and raw[5:].strip() not in ("", "[DONE]"):
                    try:
                        delta = json.loads(raw[5:])["choices"][0]["delta"].get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        first = time.perf_counter() - t0
                        break
        ttfts.append(first if first is not None else float("nan"))
    return totals, [x for x in ttfts if x == x]


def llm_interpreter(n: int) -> list[float]:
    """The LangGraph agent's model call (JSON interpretation of one reply)."""
    from ..agent.interpret import InterpretRequest, make_interpreter

    it = make_interpreter(make_llm_client())
    out = []
    for i in range(n):
        req = InterpretRequest(step_kind="yes_no", question="Have you taken it?", utterance=PARENT_LINES[i % 3],
                               topic="Metformin", usecase="medicine_before_food")
        t0 = time.perf_counter()
        it.interpret(req)
        out.append(time.perf_counter() - t0)
    return out


async def tts_latency(n: int) -> tuple[list[float], list[float], list[float]]:
    """REST (full audio), and SSE (first audio chunk / complete)."""
    tts, http = GnaniTTS(), http_client(60)
    key = get_settings().gnani_api_key
    rest, sse_first, sse_total = [], [], []
    for i in range(n):
        line = AGENT_LINES[i % len(AGENT_LINES)]
        t0 = time.perf_counter()
        _ = [c async for c in tts.synthesize(line)]
        rest.append(time.perf_counter() - t0)
        body = {"text": line, "voice": "Kaveri", "model": "timbre-v2.5", "language": "en-IN", "speed": 1.0,
                "audio_config": {"sample_rate": 8000, "num_channels": 1, "sample_width": 2, "encoding": "linear_pcm", "container": "raw"}}
        t0, first = time.perf_counter(), None
        with http.stream("POST", TTS_URL.replace("/inference", "/sse"), headers={"X-API-Key-ID": key, "Accept": "text/event-stream"}, json=body) as r:
            for ln in r.iter_lines():
                if ln.strip() and first is None:
                    first = time.perf_counter() - t0
        sse_first.append(first if first is not None else float("nan"))
        sse_total.append(time.perf_counter() - t0)
    return rest, [x for x in sse_first if x == x], sse_total


async def stt_latency(silence_ms: int, n: int) -> tuple[list[float], list[int]]:
    """From the END OF SPEECH (first silent frame sent) to the LAST transcript segment arriving, plus how many segments the
    sentence was split into (a short silence window can cut a sentence at a natural pause)."""
    tts = GnaniTTS()
    lat, splits = [], []
    for i in range(n):
        pcm = b"".join([c async for c in tts.synthesize(PARENT_LINES[i % 3])])
        stt = GnaniSTT(min_silence_ms=silence_ms)
        await stt.start()
        arrivals: list[float] = []

        async def listen():
            async for ev in stt.events():
                if ev.kind == "transcript":
                    arrivals.append(time.perf_counter())

        task = asyncio.create_task(listen())
        speech_frames = len(pcm) // FRAME
        t_end = None
        audio = pcm + b"\x00" * (8000 * 2 * 5)
        for k in range(0, len(audio) - FRAME + 1, FRAME):
            await stt.send_audio(audio[k:k + FRAME])
            if k // FRAME == speech_frames:
                t_end = time.perf_counter()  # the moment the parent has stopped talking
            await asyncio.sleep(FRAME / 2 / 8000)
        await asyncio.sleep(1.0)
        await stt.close()
        task.cancel()
        if arrivals and t_end is not None:
            lat.append(max(0.0, arrivals[-1] - t_end))
            splits.append(len(arrivals))
    return lat, splits


def run_latency(n_llm: int = 8, n_speech: int = 4, out_root: str = "runs") -> dict:
    res: dict = {}
    print("LLM agent reply (total + time to first token) ...", flush=True)
    tot, ttft = llm_agent_reply(n_llm)
    res["llm_agent_total"], res["llm_agent_ttft"] = summarize(tot), summarize(ttft)
    print("LLM interpreter JSON call (LangGraph agent) ...", flush=True)
    res["llm_interpreter"] = summarize(llm_interpreter(n_llm))
    print("Gnani TTS: REST vs SSE ...", flush=True)
    rest, first, total = asyncio.run(tts_latency(n_speech + 2))
    res["tts_rest_full_audio"], res["tts_sse_first_chunk"], res["tts_sse_total"] = summarize(rest), summarize(first), summarize(total)
    for ms in (600, 300):
        print(f"Gnani STT end-of-speech -> transcript (min-silence {ms} ms), real-time audio ...", flush=True)
        lat, splits = asyncio.run(stt_latency(ms, n_speech))
        res[f"stt_{ms}ms"] = summarize(lat)
        res[f"stt_{ms}ms_segments_per_sentence"] = splits
    seq = res["stt_600ms"]["p50"] + res["llm_agent_total"]["p50"] + res["tts_rest_full_audio"]["p50"]
    stream = res["stt_300ms"]["p50"] + res["llm_agent_ttft"]["p50"] + res["tts_sse_first_chunk"]["p50"]
    res["turn_sequential_p50"], res["turn_streaming_p50"] = round(seq, 2), round(stream, 2)
    res["models"] = {"agent_llm": make_agent_llm_client().model, "tts": "timbre-v2.5", "stt": "gnani prisma"}
    out = Path(out_root)
    out.mkdir(exist_ok=True)
    (out / f"latency-{datetime.now():%Y%m%d-%H%M%S}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
