"""`carecall model-bench`: which OpenRouter model is fast AND good enough for this job?

Phase 1 (speed): for each candidate, the two calls our product makes, on the real prompts, thinking disabled,
                 provider routing sorted by latency:
                   agent reply  : multi-turn chat with the 3.5k-char conversation prompt  (total + time to first token)
                   interpretation: JSON-mode call used by the LangGraph agent               (total + JSON validity)
                 and how many hidden reasoning tokens the model actually spent (should be 0).
Phase 2 (quality, finalists only): the 17 scripted scenarios run through that model as the interpreter; pass count = does
                 it still understand elderly speech as well as the big model? Speed means nothing if it guesses."""
import json
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor

from ..agent.generative import GenerativeAgent
from ..agent.interpret import InterpretRequest, LLMInterpreter
from ..config import get_settings
from ..llm.openai_compat import OpenAICompatClient
from ..net import http_client
from .notes import TemplateNoteWriter
from .personas import load_personas
from .runner import load_scenarios, run_scenario

PARENT = ["Yes, I took it just after dinner.", "No, I haven't taken it, it upsets my stomach.",
          "I already ate lunch early. Should I still take it?", "I took the white one, I am not sure which tablet it was.",
          "Not yet, I will take it after the news."]
NO_THINK = {"reasoning": {"enabled": False}, "provider": {"sort": "latency"}}
NO_THINK_EFFORT = {"reasoning": {"effort": "none"}, "provider": {"sort": "latency"}}


def _client(model: str, extra: dict) -> OpenAICompatClient:
    s = get_settings()
    return OpenAICompatClient(s.llm_base_url, s.llm_api_key, model, timeout=60, temperature=0.3, extra_body=extra)


def _raw(client: OpenAICompatClient, system: str, messages: list[dict], json_mode: bool, stream: bool = False):
    body = {"model": client.model, "temperature": client.temperature, "messages": [{"role": "system", "content": system}] + messages,
            **client.extra_body}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    url = f"{client.base_url}/chat/completions"
    h = {"Authorization": f"Bearer {client.api_key}"}
    if not stream:
        t0 = time.perf_counter()
        r = client._http.post(url, headers=h, json=body)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        d = r.json()
        u = d.get("usage", {}) or {}
        return d["choices"][0]["message"].get("content") or "", dt, (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    body["stream"] = True
    t0, first = time.perf_counter(), None
    with client._http.stream("POST", url, headers=h, json=body) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if raw.startswith("data:") and raw[5:].strip() not in ("", "[DONE]"):
                try:
                    delta = json.loads(raw[5:])["choices"][0]["delta"].get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    first = time.perf_counter() - t0
                    break
    return first


def bench_one(model: str, n: int = 5) -> dict:
    card = next(p for p in load_personas() if p["usecase"] == "medicine_before_food")
    row = {"model": model, "error": None}
    for label, extra in (("enabled:false", NO_THINK), ("effort:none", NO_THINK_EFFORT), ("none", {"provider": {"sort": "latency"}})):
        client = _client(model, extra)
        agent = GenerativeAgent(client, card["usecase"], card)
        agent.open()
        try:
            _raw(client, agent.system, agent.history + [{"role": "user", "content": PARENT[0]}], False)  # also warms the connection
            row["no_think_param"] = label
            break
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    else:
        return row
    row["error"] = None
    totals, ttfts, reasoning = [], [], 0
    for i in range(n):
        msgs = agent.history + [{"role": "user", "content": PARENT[i % len(PARENT)]}]
        try:
            _, dt, rt = _raw(client, agent.system, msgs, False)
            totals.append(dt)
            reasoning += rt
            t = _raw(client, agent.system, msgs, False, stream=True)
            if t is not None:
                ttfts.append(t)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    it = LLMInterpreter(client)
    ijson, valid = [], 0
    for i in range(n):
        req = InterpretRequest(step_kind="yes_no", question="Have you taken it?", utterance=PARENT[i % len(PARENT)],
                               topic="Metformin", usecase="medicine_before_food")
        t0 = time.perf_counter()
        out = it.interpret(req)
        ijson.append(time.perf_counter() - t0)
        valid += "invalid" not in out.reason and "llm_failed" not in out.reason
    med = lambda xs: round(st.median(xs), 2) if xs else None  # noqa: E731
    row.update(reply_p50=med(totals), reply_ttft_p50=med(ttfts), reply_max=round(max(totals), 2) if totals else None,
               interp_p50=med(ijson), interp_max=round(max(ijson), 2), json_valid=f"{valid}/{n}", reasoning_tokens=reasoning)
    return row


def quality(model: str) -> dict:
    """The 17 scripted scenarios with this model as the interpreter (template notes: this isolates understanding)."""
    client = _client(model, NO_THINK)
    interp = LLMInterpreter(client)
    passed, failed, t0 = 0, [], time.perf_counter()
    for sc in load_scenarios():
        try:
            rec = run_scenario(sc, interp, TemplateNoteWriter())
            passed += rec["passed"]
            if not rec["passed"]:
                failed.append(sc["id"])
        except Exception as e:  # noqa: BLE001
            failed.append(f"{sc['id']}({type(e).__name__})")
    return {"model": model, "scenarios_passed": passed, "of": len(load_scenarios()), "failed": failed,
            "seconds": round(time.perf_counter() - t0, 1)}


def run_bench(models: list[str], n: int = 5, finalists: int = 5, workers: int = 4) -> dict:
    print(f"Phase 1: speed, {len(models)} models, {n} trials each, {workers} in parallel ...", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda m: bench_one(m, n), models))
    ok = sorted([r for r in rows if not r["error"] and r.get("reply_p50")], key=lambda r: r["reply_p50"] + r["interp_p50"])
    print(f"\n{'model':44} {'reply p50':>9} {'ttft':>6} {'max':>6} {'interp p50':>10} {'max':>6} {'json':>5} {'think-tok':>9}  param")
    for r in ok:
        print(f"{r['model']:44} {r['reply_p50']:>9} {str(r['reply_ttft_p50']):>6} {r['reply_max']:>6} {r['interp_p50']:>10} {r['interp_max']:>6} "
              f"{r['json_valid']:>5} {r['reasoning_tokens']:>9}  {r['no_think_param']}")
    for r in rows:
        if r["error"] and r not in ok:
            print(f"{r['model']:44} ERROR {r['error']}")
    top = [r["model"] for r in ok if r["json_valid"].startswith(str(n))][:finalists]
    print(f"\nPhase 2: quality of the {len(top)} fastest models with valid JSON (17 scripted scenarios) ...", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        quals = list(pool.map(quality, top))
    for q in quals:
        print(f"{q['model']:44} {q['scenarios_passed']}/{q['of']} passed in {q['seconds']}s  failed: {q['failed']}")
    return {"speed": rows, "quality": quals}
