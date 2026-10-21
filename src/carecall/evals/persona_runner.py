"""Run the persona suite and produce an audit report.

    carecall personas                      scripted personas (deterministic) against the configured agent
    carecall personas --mode llm           an LLM plays each persona live (needs LLM_* in .env)
    carecall personas --usecase meal_checkin --persona mc_happy_lakshmi

Outputs runs/<id>/: persona_report.md (read this first), notes/<persona>.md, summary.md, results.json, run.db.
Every call is also a Langfuse trace tagged suite:personas, usecase:<x>, persona-mode:<m>, run:<id>."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..agent.interpret import make_interpreter
from ..llm.factory import make_judge_client, make_llm_client, make_persona_client
from .notes import LLMNoteWriter, TemplateNoteWriter
from .personas import LLMPersona, ScriptedPersona, load_personas
from .runner import finalize_run, run_scenario

REASK_MARKERS = ("did not catch", "still there", "say it again", "try once more", "leave it for now")


def _reasks(rec: dict) -> int:
    return sum(any(m in t["text"] for m in REASK_MARKERS) for t in rec["transcript"] if t["speaker"] == "agent")


def _tail(rec: dict, n: int = 8) -> list[str]:
    return [f"    [{t['turn_no']}] {t['speaker']}: {t['text'] or '(silence)'}" for t in rec["transcript"][-n:]]


def build_report(records: list[dict], run_id: str, mode: str, interpreter_name: str, writer_name: str) -> list[str]:
    by_uc: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uc[r["scenario"]["usecase"]].append(r)
    out = [f"# Persona audit {run_id}", "",
           f"- persona mode: `{mode}`  |  agent interpreter: `{interpreter_name}`  |  note writer: `{writer_name}`",
           f"- **{sum(r['passed'] for r in records)}/{len(records)} personas handled correctly**", "",
           "## Scoreboard by use case", "| use case | passed | personas |", "|---|---|---|"]
    for uc, rs in by_uc.items():
        out.append(f"| {uc} | {sum(r['passed'] for r in rs)}/{len(rs)} | {', '.join(r['scenario']['id'] for r in rs)} |")

    out += ["", "## Every persona", "| persona | traits | result | ended | parent turns | re-asks | alerts | failed checks |",
            "|---|---|---|---|---|---|---|---|"]
    for r in records:
        sc = r["scenario"]
        failed = ", ".join(c["name"] for c in r["checks"] if not c["passed"]) or "-"
        alerts = ", ".join(a["severity"] for a in r["alerts"]) or "-"
        out.append(f"| [{sc['id']}](notes/{sc['id']}.md) | {', '.join(sc['traits'] or [])} | "
                   f"{'PASS' if r['passed'] else 'FAIL'} | {r['end_reason'] or 'hangup'} | {r['turns']} | "
                   f"{_reasks(r)} | {alerts} | {failed} |")

    calls = [i for r in records for i in r.get("interpretations", []) if i["type"] == "interpretation"]
    bad = [i for i in calls if i.get("error")]
    downgrades = [i for r in records for i in r.get("interpretations", []) if i["type"] == "validation.downgrade"]
    out += ["", "## Model-call health",
            f"- interpretations: {len(calls)}  |  model errors / invalid output: {len(bad)}  |  guard downgrades: {len(downgrades)}"]
    out += [f"  - error: {i['error']} (utterance: {i.get('utterance', '')[:60]!r})" for i in bad[:10]]
    out += [f"  - downgrade `{i.get('reason')}` at step {i.get('step')}" for i in downgrades[:10]]
    failing = [r for r in records if not r["passed"]]
    if failing:
        out += ["", "## Failures, with the end of each conversation"]
        for r in failing:
            sc = r["scenario"]
            out += ["", f"### {sc['id']} ({sc['name']}): {sc['profile']}", f"Truth: `{sc['truth']}`"]
            for c in r["checks"]:
                if not c["passed"]:
                    out.append(f"- FAILED `{c['name']}`: {c['detail']}")
            out += ["", "```"] + _tail(r) + ["```"]

    patterns: dict[str, list[str]] = defaultdict(list)
    for r in failing:
        for c in r["checks"]:
            if not c["passed"]:
                patterns[c["name"]].append(r["scenario"]["id"])
    if patterns:
        out += ["", "## Patterns across personas (check -> personas affected)"]
        out += [f"- `{k}`: {len(v)} ({', '.join(v)})" for k, v in sorted(patterns.items(), key=lambda kv: -len(kv[1]))]
    reask_heavy = [r["scenario"]["id"] for r in records if _reasks(r) >= 2]
    if reask_heavy:
        out += ["", f"## Friction: 2+ re-asks needed for {', '.join(reask_heavy)}"]
    return out + [""]


def _work(args):
    """One persona, one process: the DB engine is a per-process global, so processes (not threads) give isolation.
    Never raises: a failed persona (e.g. LLM timeout) comes back as an error entry instead of killing the run."""
    card, mode, agent, run_id = args
    try:
        from .. import tracing

        tracing.init_tracing()
        client = make_llm_client()
        interpreter = make_interpreter(None if agent == "rules" else client)
        writer = LLMNoteWriter(make_judge_client()) if client else TemplateNoteWriter()
        # the persona speaks with a livelier temperature than the (temperature 0) agent
        pclient = make_persona_client()
        persona = LLMPersona(card, pclient) if mode == "llm" else ScriptedPersona(card)
        rec = run_scenario({**card, "description": card["profile"]}, interpreter, writer, persona=persona,
                           run_id=run_id, suite="personas", agent_mode="converse" if agent == "converse" else None)
        tracing.flush()
        return {"rec": rec, "interpreter": interpreter.name, "writer": writer.name}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:300]}", "id": card["id"], "usecase": card["usecase"]}


def run_personas(mode: str = "scripted", usecases: list[str] | None = None, ids: list[str] | None = None,
                 out_root: str = "runs", agent: str = "auto", workers: int = 1) -> dict:
    client = make_llm_client()
    if mode == "llm" and client is None:
        raise SystemExit("--mode llm needs an LLM: set LLM_PROVIDER / LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in .env "
                         "(run `carecall check-keys`)")
    cards = load_personas(usecases, ids)
    if not cards:
        raise SystemExit("no personas matched")
    run_id = "p-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / run_id
    jobs = [(c, mode, agent, run_id) for c in cards]

    results: dict[str, dict] = {}

    def show(r):
        if "error" in r:
            print(f"ERROR {r['usecase']:22} {r['id']}: {r['error']}", flush=True)
        else:
            rec = r["rec"]
            print(f"{'PASS' if rec['passed'] else 'FAIL'}  {rec['scenario']['usecase']:22} {rec['scenario']['id']}"
                  f"  ({rec['turns']} turns)", flush=True)

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
    iname, wname = done[0]["interpreter"], done[0]["writer"]

    class _N:  # finalize_run only needs .name
        def __init__(self, name):
            self.name = name

    report = build_report(records, run_id, mode, iname, wname)
    if errors:
        report += ["## Personas that errored (not scored)"] + [f"- {e['id']}: {e['error']}" for e in errors] + [""]
    result = finalize_run(records, out, run_id, "Persona run", _N(iname), _N(wname))
    (out / "persona_report.md").write_text("\n".join(report), encoding="utf-8")
    result["report"] = str(out / "persona_report.md")
    result["errors"] = [e["id"] for e in errors]
    return result
