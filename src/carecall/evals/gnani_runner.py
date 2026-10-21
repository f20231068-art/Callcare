"""Run the persona suite against GNANI's agent over real phone calls, then judge each transcript.

Per persona (sequential: one persona line = one Twilio number):
  1. configure the Gnani agent for the use case: system prompt, greeting, Pampa model, voice   (Platform API)
  2. arm a persona job on our server (`carecall serve` must be running and reachable at PUBLIC_BASE_URL via a tunnel)
  3. ask the Gnani agent to call our Twilio persona number                                    (Platform API trigger_call)
  4. our server answers; an LLM plays the parent using Gnani STT/TTS                           (telephony/persona_bridge.py)
  5. when the call ends, pull Gnani's own transcript (what ITS ears heard) and judge it         (evals/gnani_judge.py)

Outputs the same artefacts as every other run: persona_report.md, notes/, results.json, run.db and Langfuse traces."""
import difflib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

from .. import tracing
from ..config import get_settings
from ..gnani_platform import GnaniAPIError, GnaniPlatform, agent_context, opening_for, system_prompt_for
from ..llm.factory import make_judge_client, make_llm_client
from ..net import http_client
from .gnani_judge import build_brief, deterministic, judge, rubric_for
from .notes import Issue, Note, max_severity, render_markdown
from .persona_runner import build_report
from .personas import load_personas
from .runner import finalize_run

HIGH = ("safe_no_advice", "no_invention", "emergency_handled", "med_", "family_informed")


def preflight(need_calls: bool = True) -> list[str]:
    """Everything that must exist before a real call can happen, as a list of plain-English problems."""
    s, problems = get_settings(), []
    if not s.gnani_api_key:
        problems.append("GNANI_API_KEY (speech key) is empty: the persona needs Gnani STT/TTS.")
    if make_llm_client() is None:
        problems.append("No LLM configured (LLM_PROVIDER/LLM_BASE_URL/LLM_API_KEY/LLM_MODEL): the persona cannot think.")
    if need_calls:
        if not s.gnani_platform_api_key:
            problems.append("GNANI_PLATFORM_API_KEY is empty: needed to create/configure the Gnani agent and place the call "
                            "(Agent Builder console -> API keys, with 'agents' and 'conversations' permissions).")
        if not (s.persona_phone_e164 and s.persona_phone_cc and s.persona_phone_national):
            problems.append("PERSONA_PHONE_E164 / PERSONA_PHONE_CC / PERSONA_PHONE_NATIONAL are empty: the Twilio number the "
                            "Gnani agent will call. Gnani only dials numbers WHITELISTED on your account (ask your account manager).")
        if not (s.twilio_account_sid and s.twilio_auth_token):
            problems.append("TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN are empty: needed to answer the call and stream its audio.")
        if not s.public_base_url:
            problems.append("PUBLIC_BASE_URL is empty: run a tunnel (e.g. `ngrok http 8000`) and paste its https URL.")
        try:
            r = http_client(5).get(f"{s.persona_server_url}/health")
            r.raise_for_status()
        except Exception:
            problems.append(f"`carecall serve` is not reachable at {s.persona_server_url}: start it in another terminal.")
    return problems


def asr_fidelity(job_parent_lines: list[str], gnani_user_lines: list[str]) -> float:
    """How closely Gnani's ASR transcript of the parent matches what the persona actually said (1.0 = identical)."""
    def norm(x):
        return re.sub(r"[^a-z0-9 ]+", "", x.lower()).strip()
    said, heard = norm(" ".join(job_parent_lines)), norm(" ".join(gnani_user_lines))
    return round(difflib.SequenceMatcher(None, said, heard).ratio(), 2) if said else 1.0


def _note_from_judgement(card: dict, rubric, j: dict, det: list[dict]) -> Note:
    issues = []
    for it in j["items"]:
        if it["verdict"] == "fail":
            sev = "high" if it["id"].startswith(HIGH) else "medium"
            issues.append(Issue(category=it["id"], severity=sev, turn=(it["turns"] or [None])[0],
                                description=f"{it['check']} | {it['why']}"))
    for c in det:
        if not c["passed"]:
            issues.append(Issue(category=c["name"], severity="high" if "advice" in c["name"] or "emergency" in c["name"] else "medium",
                                description=c["detail"] or c["name"]))
    cov = [f"[{it['verdict'].upper()}] {it['id']}: {it['why']} " + (f"(turns {it['turns']})" if it["turns"] else "") for it in j["items"]]
    return Note(summary=j["summary"], coverage=cov, issues=issues, overall="pass" if not issues else "fail",
                judgment_notes="; ".join(j["risks"]))


def judge_call(card: dict, transcript: list[dict], client, trace_id: str | None, run_id: str, extra: dict) -> dict:
    """Build a record (the same shape every other runner produces) from a finished Gnani call."""
    rubric = rubric_for(card)
    det = deterministic(transcript)
    with tracing.call_span(trace_id, "judge", 0, 0, card["routine"], tags=["suite:gnani-personas"], trace_name=f"gnani:{card['id']}",
                           session_id=f"{run_id}:{card['id']}"):
        j = judge(client, card, transcript, rubric, build_brief(card))
    rubric_checks = [{"name": f"rubric:{it['id']}", "passed": it["verdict"] in ("pass", "na"),
                      "detail": f"{it['why']} (turns {it['turns']})"} for it in j["items"]]
    note = _note_from_judgement(card, rubric, j, det)
    checks = det + rubric_checks
    rec = {
        "scenario": {"id": card["id"], "routine": card["routine"], "description": card["profile"], "tags": [],
                     "usecase": card["usecase"], "name": card["name"], "profile": card["profile"],
                     "traits": card["traits"], "truth": card["truth"]},
        "transcript": [{"turn_no": i + 1, "speaker": t["speaker"], "text": t["text"], "latency_ms": None} for i, t in enumerate(transcript)],
        "checks": checks, "invariants_ok": all(c["passed"] for c in det), "outcome_ok": all(c["passed"] for c in rubric_checks),
        "interpreter": "gnani-agent (Pampa)", "trace_id": trace_id, "trace_url": tracing.trace_url(trace_id),
        "alerts": [], "interpretations": [], "meal_logs": [], "medicine_logs": [], "reminder_acks": [], "callbacks": [],
        "meal_profile": {}, "max_latency_ms": 0, "steps": [], **extra,
    }
    rec["passed"] = rec["invariants_ok"] and rec["outcome_ok"]
    rec["turns"] = sum(1 for t in transcript if t["speaker"] == "parent")
    rec["note"] = note.model_dump()
    rec["note_markdown"] = render_markdown(rec, note, "llm-judge")
    rec["severity_max"] = max_severity(note)
    rec["writer"] = "llm-judge"
    return rec


def _wait_job(server: str, job_id: str, timeout: float) -> dict:
    http = http_client(10)
    end = time.time() + timeout
    job = {}
    while time.time() < end:
        job = http.get(f"{server}/persona/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(3)
    job["error"] = job.get("error") or "timed out waiting for the call to finish (did Gnani dial the persona number?)"
    return job


def run_gnani_personas(ids: list[str] | None = None, usecases: list[str] | None = None, out_root: str = "runs",
                       call_timeout: float = 420, dry_run: bool = False) -> dict:
    s = get_settings()
    cards = load_personas(usecases, ids)
    if not cards:
        raise SystemExit("no personas matched")
    problems = preflight(need_calls=not dry_run)
    if dry_run:
        print("DRY RUN: no calls will be placed.\n")
        for c in cards[:5]:
            ctx = agent_context(c)
            print(f"- {c['id']}: greeting = {opening_for(c['usecase'], ctx)[0]}")
        if cards:
            print("\nSystem prompt for", cards[0]["usecase"], "(rendered):\n" + system_prompt_for(cards[0]["usecase"], agent_context(cards[0])))
        print("\nPreflight problems:" if problems else "\nPreflight OK.")
        for p in problems:
            print(" -", p)
        return {"dry_run": True, "problems": problems}
    if problems:
        raise SystemExit("Cannot start real calls yet:\n" + "\n".join(f" - {p}" for p in problems))

    client = make_judge_client()
    platform = GnaniPlatform()
    run_id = "g-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(out_root) / run_id
    server = s.persona_server_url.rstrip("/")
    http = http_client(15)
    bots: dict[str, str] = {}
    backed_up = False
    records, errors = [], []

    for card in cards:
        uc, ctx = card["usecase"], agent_context(card)
        print(f"\n== {card['id']} ({uc})", flush=True)
        try:
            if uc not in bots:
                if s.gnani_bot_id:  # one existing agent is re-purposed per use case: back it up before the first overwrite
                    bots[uc] = s.gnani_bot_id
                    if not backed_up:
                        Path("backups").mkdir(exist_ok=True)
                        dest = Path("backups") / f"gnani_agent_{s.gnani_bot_id}_{run_id}.json"
                        dest.write_text(json.dumps(platform.get_agent(s.gnani_bot_id), indent=2, default=str), encoding="utf-8")
                        print(f"   backed up the current agent configuration to {dest}", flush=True)
                        backed_up = True
                else:
                    bots[uc] = platform.ensure_agent(f"CareCall-{uc}", f"CareCall {uc} agent (persona tests)")
            platform.configure(bots[uc], uc, ctx)
            job_id = http.post(f"{server}/persona/jobs", json={"persona_id": card["id"]}).json()["job_id"]
            ref = f"{run_id}-{card['id']}"[:60]
            platform.trigger_call(bots[uc], s.persona_phone_national, s.persona_phone_cc, card["name"].split(",")[0], ref)
            job = _wait_job(server, job_id, call_timeout)
            conv = platform.find_conversation(bots[uc], ref, card["name"].split(",")[0], timeout=180)
            if conv is None:
                raise GnaniAPIError("Gnani never listed the conversation (call not placed or number not whitelisted)")
            stats = platform.stats(str(conv.get("conversationId") or conv.get("id")))
            transcript = GnaniPlatform.transcript_of(stats)
            if not transcript:
                raise GnaniAPIError("Gnani returned an empty transcript")
            heard = [t["text"] for t in transcript if t["speaker"] == "parent"]
            said = [t["text"] for t in job["transcript"] if t["speaker"] == "parent"]
            fid = asr_fidelity(said, heard)
            trace_id = tracing.new_trace_id()
            with tracing.call_span(trace_id, "gnani_call", 0, 0, card["routine"], tags=["suite:gnani-personas", f"usecase:{uc}", f"run:{run_id}", f"persona:{card['id']}"],
                                   input={"persona": card["name"], "profile": card["profile"], "truth": card["truth"]},
                                   trace_name=f"gnani:{card['id']}", session_id=f"{run_id}:{card['id']}",
                                   metadata={"bot_id": bots[uc], "conversation": conv.get("conversationId", ""), "asr_fidelity": fid}) as span:
                if span:
                    span.update(output={"transcript": transcript, "disposition": stats.get("overallCallDisposition")})
            rec = judge_call(card, transcript, client, trace_id, run_id, {
                "end_reason": str(stats.get("overallCallDisposition") or ""), "asr_fidelity": fid,
                "persona_side_transcript": job["transcript"], "persona_tts_log": job.get("tts_log", []),
                "call_error": job.get("error")})
            tracing.score(trace_id, "eval_passed", 1.0 if rec["passed"] else 0.0, comment=card["id"], data_type="BOOLEAN")
            tracing.score(trace_id, "rubric_pass_rate", round(sum(c["passed"] for c in rec["checks"]) / len(rec["checks"]), 2), data_type="NUMERIC")
            tracing.score(trace_id, "gnani_asr_fidelity", fid, data_type="NUMERIC")
            records.append(rec)
            print(f"{'PASS' if rec['passed'] else 'FAIL'}  {card['id']}  (asr fidelity {fid})", flush=True)
        except (GnaniAPIError, httpx.HTTPError, KeyError) as e:
            errors.append({"id": card["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"ERROR {card['id']}: {errors[-1]['error']}", flush=True)

    if not records:
        raise SystemExit("every persona errored:\n" + "\n".join(f"  {e['id']}: {e['error']}" for e in errors))

    class _N:
        def __init__(self, name):
            self.name = name

    fids = [r["asr_fidelity"] for r in records]
    extra = [f"- Gnani ASR fidelity on the persona's speech: mean {sum(fids) / len(fids):.2f} (1.0 = heard exactly what was said)", ""]
    report = build_report(records, run_id, "llm-on-real-phone-call", "gnani-agent (Pampa Go)", "llm-judge")
    if errors:
        report += ["## Personas that errored (not scored)"] + [f"- {e['id']}: {e['error']}" for e in errors] + [""]
    result = finalize_run(records, out, run_id, "Gnani persona run", _N("gnani-agent (Pampa Go)"), _N("llm-judge"), extra_summary=extra)
    (out / "persona_report.md").write_text("\n".join(report), encoding="utf-8")
    result["report"], result["errors"] = str(out / "persona_report.md"), [e["id"] for e in errors]
    return result
