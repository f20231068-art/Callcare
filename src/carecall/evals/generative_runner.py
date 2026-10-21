"""Personas vs a GENERATIVE agent (the model writes the agent's words) over text.

    carecall direct-personas                      uses AGENT_LLM_* as the agent's brain (else LLM_*)
    carecall direct-personas --workers 6 --usecase meal_checkin

Roles: agent = AGENT_LLM_* (put Gnani's model here when you have its endpoint) | parent = LLM_* (temperature 0.7) |
judge = LLM_* (temperature 0). Same report, notes and Langfuse traces as every other run. Text only: no speech in the loop."""
from datetime import datetime
from pathlib import Path

from .. import tracing
from ..agent.generative import GenerativeAgent
from ..llm.factory import make_agent_llm_client, make_judge_client, make_llm_client, make_persona_client
from .gnani_runner import judge_call
from .persona_runner import build_report
from .personas import LLMPersona, load_personas
from .runner import finalize_run

MAX_EXCHANGES = 14


def _work(args):
    card, run_id = args
    try:
        tracing.init_tracing()
        base = make_llm_client()
        agent_client = make_agent_llm_client()
        agent = GenerativeAgent(agent_client, card["usecase"], card)
        persona = LLMPersona(card, make_persona_client())
        judge_client = make_judge_client()
        trace_id = tracing.new_trace_id()
        transcript, history = [], []
        with tracing.call_span(trace_id, "generative_call", 0, 0, card["routine"],
                               tags=["suite:direct-personas", f"usecase:{card['usecase']}", f"run:{run_id}", f"persona:{card['id']}",
                                     f"agent-model:{agent_client.model}"],
                               input={"persona": card["name"], "profile": card["profile"], "truth": card["truth"]},
                               trace_name=f"direct:{card['id']}", session_id=f"{run_id}:{card['id']}",
                               metadata={"agent_model": agent_client.model, "judge_model": judge_client.model}):
            opening = agent.open()
            transcript.append({"speaker": "agent", "text": opening, "t": 0})
            history.append(("agent", opening))
            for _ in range(MAX_EXCHANGES):
                if agent.done:
                    break
                said = persona.reply(None, history)
                transcript.append({"speaker": "parent", "text": said, "t": 0})
                history.append(("parent", said))
                text = agent.reply(said)
                transcript.append({"speaker": "agent", "text": text, "t": 0})
                history.append(("agent", text))
        rec = judge_call(card, transcript, judge_client, trace_id, run_id,
                         {"end_reason": "completed" if agent.done else "turn_cap", "asr_fidelity": 1.0,
                          "interpreter": f"generative agent: {agent_client.model}",
                          "agent_model": agent_client.model, "agent_errors": agent.errors})
        tracing.score(trace_id, "eval_passed", 1.0 if rec["passed"] else 0.0, comment=card["id"], data_type="BOOLEAN")
        tracing.score(trace_id, "rubric_pass_rate", round(sum(c["passed"] for c in rec["checks"]) / len(rec["checks"]), 2), data_type="NUMERIC")
        tracing.flush()
        return {"rec": rec, "agent_model": agent_client.model, "judge_model": judge_client.model}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:300]}", "id": card["id"], "usecase": card["usecase"]}


def run_direct(ids: list[str] | None = None, usecases: list[str] | None = None, out_root: str = "runs", workers: int = 1) -> dict:
    if make_llm_client() is None:
        raise SystemExit("No LLM configured (LLM_* in .env): the parent persona and the judge need one.")
    cards = load_personas(usecases, ids)
    if not cards:
        raise SystemExit("no personas matched")
    run_id = "d-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / run_id
    results: dict[str, dict] = {}

    def show(r):
        if "error" in r:
            print(f"ERROR {r['usecase']:22} {r['id']}: {r['error']}", flush=True)
        else:
            print(f"{'PASS' if r['rec']['passed'] else 'FAIL'}  {r['rec']['scenario']['usecase']:22} {r['rec']['scenario']['id']}", flush=True)

    jobs = [(c, run_id) for c in cards]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_work, j): j[0]["id"] for j in jobs}
            for f in as_completed(futs):
                results[futs[f]] = f.result()
                show(results[futs[f]])
    else:
        for j in jobs:
            results[j[0]["id"]] = _work(j)
            show(results[j[0]["id"]])

    ordered = [results[c["id"]] for c in cards]
    errors = [r for r in ordered if "error" in r]
    done = [r for r in ordered if "error" not in r]
    if not done:
        raise SystemExit("every persona errored:\n" + "\n".join(f"  {e['id']}: {e['error']}" for e in errors))
    records = [r["rec"] for r in done]
    agent_model, judge_model = done[0]["agent_model"], done[0]["judge_model"]

    class _N:
        def __init__(self, name):
            self.name = name

    extra = [f"- **agent brain: `{agent_model}`**  |  parent persona + judge: `{judge_model}`  |  text only (no speech in the loop)",
             "- the agent's replies are generated by the model (no templates); the judge reads the transcript", ""]
    report = build_report(records, run_id, "llm-text", f"generative agent: {agent_model}", f"llm-judge: {judge_model}")
    if errors:
        report += ["## Personas that errored (not scored)"] + [f"- {e['id']}: {e['error']}" for e in errors] + [""]
    result = finalize_run(records, out, run_id, "Direct-model persona run", _N(f"agent: {agent_model}"), _N(f"judge: {judge_model}"),
                          extra_summary=extra)
    (out / "persona_report.md").write_text("\n".join(report), encoding="utf-8")
    result["report"], result["errors"], result["agent_model"] = str(out / "persona_report.md"), [e["id"] for e in errors], agent_model
    return result
