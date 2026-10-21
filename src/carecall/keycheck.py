"""`carecall check-keys`: find out what your pasted keys can actually do, without guessing.

Gnani exposes (per docs.gnani.ai) three different surfaces, each with its own key and header:
  1. Speech APIs  (Prisma STT, Timbre TTS)   host api.vachana.ai   header X-API-Key-ID
  2. Agent Builder platform API              host api.inya.ai       header x-api-key
  3. A chat/completions endpoint for the reasoning model: NOT in the public docs. If you have one, put its
     OpenAI-compatible URL in LLM_BASE_URL and set LLM_PROVIDER=openai_compat.
This check exercises each key you set and reports plainly. It never prints a key."""
import io
import time
import wave

from .net import http_client

from .config import get_settings

def _http():
    return http_client(60)


SPEECH = "https://api.vachana.ai"
PLATFORM = "https://api.inya.ai/platform"


def _ok(name, ok, detail):
    return {"name": name, "ok": ok, "detail": detail}


def check_speech(key: str) -> list[dict]:
    """TTS a short sentence, then feed that audio straight back into STT: a free round-trip test of both."""
    out = []
    text = "Yes, I have taken my tablet."
    t0 = time.perf_counter()
    try:
        r = _http().post(f"{SPEECH}/api/v1/tts/inference", timeout=30,
                       headers={"X-API-Key-ID": key, "Content-Type": "application/json"},
                       json={"text": text, "voice": "Kaveri", "model": "timbre-v2.5", "language": "en-IN", "speed": 1.0,
                             "audio_config": {"sample_rate": 16000, "num_channels": 1, "sample_width": 2,
                                              "encoding": "linear_pcm", "container": "wav"}})
        if r.status_code != 200:
            return [_ok("speech: TTS", False, f"HTTP {r.status_code}: {r.text[:200]}")]
        wav_bytes = r.content
        # The WAV header from a streamed response can carry a placeholder length, so count payload bytes instead.
        with wave.open(io.BytesIO(wav_bytes)) as w:
            rate, width = w.getframerate(), w.getsampwidth()
        secs = max(0, len(wav_bytes) - 44) / (rate * width)
        out.append(_ok("speech: TTS (Timbre v2.5)", True,
                       f"{secs:.1f}s of audio in {time.perf_counter() - t0:.1f}s round trip for: {text!r}"))
    except Exception as e:
        return [_ok("speech: TTS", False, str(e))]
    try:
        r = _http().post(f"{SPEECH}/stt/v3", timeout=60, headers={"X-API-Key-ID": key},
                       files={"audio_file": ("check.wav", wav_bytes, "audio/wav")}, data={"language_code": "en-IN"})
        if r.status_code != 200:
            out.append(_ok("speech: STT", False, f"HTTP {r.status_code}: {r.text[:200]}"))
        else:
            out.append(_ok("speech: STT (Prisma v2.5)", True, f"heard back: {r.json().get('transcript')!r}"))
    except Exception as e:
        out.append(_ok("speech: STT", False, str(e)))
    return out


def check_platform(key: str) -> list[dict]:
    try:
        r = _http().get(f"{PLATFORM}/v1/agents/config/llm_models", params={"provider": "gnani"}, timeout=20,
                      headers={"x-api-key": key})
        if r.status_code != 200:
            return [_ok("platform: list models", False, f"HTTP {r.status_code}: {r.text[:200]}")]
        groups = r.json().get("response", {}).get("LLM", [])
        models = [m["modelId"] for g in groups for m in g.get("models", [])]
        return [_ok("platform: list Gnani LLM models", True,
                    f"{models}. NOTE: the public docs expose these only as the model of an Agent Builder agent, "
                    "not as a standalone chat endpoint.")]
    except Exception as e:
        return [_ok("platform: list models", False, str(e))]


def check_llm(base_url: str, key: str, model: str) -> list[dict]:
    from .llm.openai_compat import OpenAICompatClient

    if not key.strip():
        return [_ok(f"llm: {model} @ {base_url}", False, "LLM_API_KEY is empty: paste your key into .env")]

    try:
        text = OpenAICompatClient(base_url, key, model, timeout=30).complete(
            "Reply with JSON only.", 'Return {"ok": true}')
        return [_ok(f"llm: {model} @ {base_url}", "ok" in text.lower(), f"reply: {text[:120]!r}")]
    except Exception as e:
        return [_ok(f"llm: {model} @ {base_url}", False, str(e)[:300])]


def run_checks() -> list[dict]:
    s = get_settings()
    results = []
    if s.gnani_api_key:
        results += check_speech(s.gnani_api_key)
    else:
        results.append(_ok("speech key", False, "GNANI_API_KEY not set in .env"))
    if s.gnani_platform_api_key:
        results += check_platform(s.gnani_platform_api_key)
    else:
        results.append(_ok("platform key", False, "GNANI_PLATFORM_API_KEY not set (optional)"))
    if s.llm_base_url and s.llm_model:
        results += check_llm(s.llm_base_url, s.llm_api_key, s.llm_model)
    else:
        results.append(_ok("reasoning model", False,
                           "LLM_BASE_URL / LLM_MODEL not set: personas will use the offline rule interpreter"))
    return results
