"""Command line: the Phase-0 control panel.

  carecall seed                         create the hardcoded parent, medicines, routines
  carecall routines                     list routines you can trigger
  carecall chat --routine lunch_checkin talk to the agent by typing (you play the parent)
  carecall eval [--scenario ID ...]     run scripted scenarios, write notes under runs/
  carecall personas [--mode llm]       run 25 personas (5 per use case); writes runs/<id>/persona_report.md
  carecall check-keys                   test the Gnani keys in .env
  carecall gnani-personas [--dry-run]   personas vs GNANI's agent over real phone calls
  carecall gnani-setup / twilio-setup   configure the Gnani agents / point the Twilio number at this server
  carecall voice-selftest               a persona vs OUR agent over real Gnani audio (no Twilio)
  carecall call-test                    ring your phone and read a sentence (proves Twilio creds, no server needed)
  carecall call --routine X             place a real Twilio call (needs .env + public URL)
  carecall serve                        run the API / Twilio webhook / media-stream server
  carecall callbacks                    list callbacks the agent promised to make
  carecall plan-day [--date D]          create the day's calls from onboarding data;  carecall schedule: show the calendar
  carecall scheduler                    run the dispatcher: plan, reap stale calls, dial due calls
  carecall trace-check                  verify the Langfuse connection
"""
import argparse
import sys
from pathlib import Path

from sqlalchemy import select

from . import tracing
from .agent.session import CallSession
from .calls.service import create_call
from .config import get_settings
from .db import init_db, session_scope
from .models import Callback, Routine
from .seed import seed


def _boot():
    init_db()
    seed()
    tracing.init_tracing()


def _live_boot():
    """The real server and scheduler: no demo data, and everything they do is logged to the console and to logs/carecall.log."""
    import logging
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    Path("logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), RotatingFileHandler("logs/carecall.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    init_db()
    tracing.init_tracing()


def cmd_chat(args):
    _boot()
    call_id, _ = create_call(args.routine, channel="text")
    session = CallSession(call_id)
    print(f"[call {call_id} | interpreter {session.interpreter.name} | tracing {'on' if tracing.enabled() else 'off'}]\n")
    for line in session.open():
        print(f"AGENT : {line}")
    while not session.done:
        try:
            said = input("YOU   : ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        res = session.turn(said)
        for line in res.reply:
            print(f"AGENT : {line}")
        print(f"        ({res.latency_ms} ms)")
    session.finish(abandoned=not session.done)
    print(f"\n[call {call_id} ended: {session.end_reason or 'hangup'}]")
    if session.trace_id:
        print(f"[trace] {tracing.trace_url(session.trace_id)}")


def cmd_eval(args):
    from .evals.runner import run_evals

    tracing.init_tracing()
    res = run_evals(args.scenario or None, out_root=args.out, exclude_tags=tuple(args.exclude_tag or ()))
    print(f"\n{res['passed']}/{res['total']} passed. Notes + summary: {res['dir']}")
    if res["failed"]:
        print("Failed:", ", ".join(res["failed"]))
    return 0 if not res["failed"] else 1


def cmd_personas(args):
    from .evals.persona_runner import run_personas

    tracing.init_tracing()
    res = run_personas(args.mode, args.usecase or None, args.persona or None, out_root=args.out, agent=args.agent, workers=args.workers)
    print(f"\n{res['passed']}/{res['total']} personas handled correctly. Read: {res['report']}")
    return 0


def cmd_call(args):
    from .telephony.twilio_api import place_call

    _boot()
    call_id, _ = create_call(args.routine, channel="twilio", idempotency_key=args.key)
    sid = place_call(call_id)
    print(f"call {call_id} placed, Twilio sid {sid}")


def cmd_call_test(args):
    from .telephony.twilio_api import place_test_call

    init_db()
    seed()
    to = args.to
    if not to:
        with session_scope() as s:
            from .models import Parent
            to = s.scalars(select(Parent)).first().phone
    print(f"Twilio sid {place_test_call(to)} (dialing {to})")


def cmd_check_keys(_):
    from .keycheck import run_checks

    bad = 0
    for r in run_checks():
        print(f"[{'ok' if r['ok'] else '--'}] {r['name']}: {r['detail']}")
        bad += not r["ok"]
    return 0

def cmd_langfuse_sync(_):
    from .langfuse_sync import sync

    for line in sync():
        print(line)


def cmd_gnani_personas(args):
    from .evals.gnani_runner import run_gnani_personas

    tracing.init_tracing()
    if args.console:
        from .evals.gnani_console import run_console

        if not args.usecase or len(args.usecase) != 1:
            raise SystemExit("--console needs exactly one --usecase (the console agent holds one prompt at a time)")
        res = run_console(args.usecase[0], args.persona or None, out_root=args.out)
        print(f"\n{res['passed']}/{res['total']} personas handled correctly by Gnani's agent. Read: {res['report']}")
        return
    res = run_gnani_personas(args.persona or None, args.usecase or None, out_root=args.out, dry_run=args.dry_run)
    if not res.get("dry_run"):
        print(f"\n{res['passed']}/{res['total']} personas handled correctly by Gnani's agent. Read: {res['report']}")


def cmd_latency(args):
    from .evals.latency import run_latency

    res = run_latency(args.llm_trials, args.speech_trials)
    print()
    for k, v in res.items():
        print(f"{k:24} {v}")


def cmd_model_bench(args):
    import json

    from .evals.model_bench import run_bench

    res = run_bench([m.strip() for m in args.models.split(",") if m.strip()], n=args.trials, finalists=args.finalists)
    Path("runs").mkdir(exist_ok=True)
    out = Path("runs") / f"model-bench-{__import__('datetime').datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


def cmd_plan_day(args):
    """Put a day's calls on the calendar from the stored onboarding data (idempotent)."""
    from datetime import date

    from .calls.planner import plan_day

    _boot()
    day = date.fromisoformat(args.date) if args.date else None
    ids = plan_day(day)
    print(f"planned {len(ids)} new call(s)" + (f" for {args.date}" if args.date else " for today (each parent's today)"))
    cmd_schedule(None)


def cmd_schedule(_):
    """Show what is on the calendar."""
    from .models import CallAttempt

    _boot()
    with session_scope() as s:
        rows = s.scalars(select(CallAttempt).order_by(CallAttempt.scheduled_at)).all()
        for c in rows:
            print(f"#{c.id:<4} {c.scheduled_at:%Y-%m-%d %H:%M} UTC  {c.status:12} {c.kind:9} {c.idempotency_key or '-'}"
                  + (f"  -> {c.outcome}" if c.outcome else ""))
        if not rows:
            print("(nothing scheduled)")


def cmd_scheduler(_):
    """Run the dispatcher: plan the day, reap stale calls, dial due calls. Ctrl+C to stop."""
    import asyncio

    from .calls.scheduler import scheduler_loop
    from .telephony.twilio_api import place_call

    _live_boot()
    print("scheduler running (plans hourly, dials due calls every few seconds); Ctrl+C to stop")
    try:
        asyncio.run(scheduler_loop(place_call))
    except KeyboardInterrupt:
        pass


def cmd_outcome(args):
    """Show (and apply, if undecided) the outcome engine's decision for one call."""
    from .calls.outcome import apply
    from .llm.factory import make_llm_client
    from .models import CallAttempt

    _boot()
    o = apply(args.call_id, make_llm_client())
    print(f"outcome: {o.kind} ({o.reason})" + (f"\n  follow-up: {o.follow_up}" if o.follow_up else ""))
    with session_scope() as s:
        for c in s.scalars(select(CallAttempt).where(CallAttempt.prev_attempt_id == args.call_id)):
            print(f"  scheduled #{c.id} {c.kind} {c.idempotency_key} at {c.scheduled_at:%Y-%m-%d %H:%M} UTC")


def cmd_live_scheduling(_):
    """Live test of in-call scheduling: LLM parents, real interpreter, real database; writes a report."""
    from .evals.live_scheduling import run

    path = run()
    print(f"report: {path}")


def cmd_tool_use(args):
    """Does the conversational agent call the right tools for the right task? (real model, scripted parent lines)"""
    from .evals.tool_use import run

    raise SystemExit(run(args.runs, args.workers, args.case))


def cmd_direct(args):
    from .evals.generative_runner import run_direct

    tracing.init_tracing()
    res = run_direct(args.persona or None, args.usecase or None, out_root=args.out, workers=args.workers)
    print(f"\n{res['passed']}/{res['total']} personas handled correctly by agent model {res['agent_model']}. Read: {res['report']}")


def cmd_gnani_export(_):
    from .evals.gnani_console import export_setup

    root = export_setup()
    print(f"Paste-ready agent configuration written to {root}/ (see {root}/README.md)")


def cmd_twilio_otp(_):
    """Show recent SMS received by the persona number (e.g. the Gnani whitelisting OTP)."""
    from datetime import datetime, timedelta, timezone

    from twilio.rest import Client

    s = get_settings()
    c = Client(s.twilio_account_sid, s.twilio_auth_token)
    msgs = c.messages.list(to=s.persona_phone_e164, date_sent_after=datetime.now(timezone.utc) - timedelta(minutes=30), limit=10)
    if not msgs:
        print("no SMS received in the last 30 minutes (is the number SMS-capable? did you request the OTP?)")
    for m in msgs:
        print(f"{m.date_sent}  from {m.from_}: {m.body}")


def cmd_gnani_setup(_):
    """Create/refresh one Gnani agent per use case and show what your key can see."""
    from .evals.personas import load_personas
    from .gnani_platform import GnaniPlatform, agent_context
    from .prompts import USECASES

    gp = GnaniPlatform()
    for label, fn in (("LLM models", lambda: gp.llm_models()), ("transcribers", gp.transcriber_config), ("TTS", gp.tts_config)):
        try:
            print(f"[ok] {label}: {str(fn())[:400]}")
        except Exception as e:
            print(f"[--] {label}: {e}")
    for uc in USECASES:
        card = next(p for p in load_personas() if p["usecase"] == uc)
        bot = gp.ensure_agent(f"CareCall-{uc}", f"CareCall {uc} agent (persona tests)")
        gp.configure(bot, uc, agent_context(card))
        print(f"[ok] agent CareCall-{uc}: botId {bot} configured (Pampa Go, en-IN)")


def cmd_twilio_setup(_):
    from .config import get_settings
    from .telephony.twilio_api import point_number_at_server

    s = get_settings()
    sid = point_number_at_server(s.persona_phone_e164)
    print(f"{s.persona_phone_e164} ({sid}) now sends calls to {s.public_base_url.rstrip('/')}/persona/voice")


def cmd_voice_selftest(args):
    import asyncio

    from .evals.personas import load_personas
    from .evals.voice_loopback import run_loopback

    card = load_personas(ids=[args.persona])[0]
    out = asyncio.run(run_loopback(card))
    print("\npersona side (what the persona heard / said):")
    for t in out["job"].transcript:
        print(f"  [{t['t']:>5}s] {t['speaker']:6} {t['text']}")
    print(f"agent end reason: {out['session'].end_reason}")


def cmd_stream_test(args):
    """Live audio pipeline against the RUNNING server, playing Twilio's part (no carrier, no real phone)."""
    import asyncio

    from .evals.stream_test import run

    res = asyncio.run(run(args.line or None, host=args.host, routine=args.routine))
    print("status:", res["status"], "| outcome:", res["outcome"], "| end:", res["end_reason"])
    print("agent waited (s) after the parent stopped talking:", res["reply_wait_s"], "| agent audio lengths (s):", res["agent_audio_s"])
    print("transcript the agent stored:")
    for who, text in res["turns"]:
        print(f"  {who:6} {text}")
    print("what an independent speech recogniser reads back from the agent's audio:")
    for t in res["agent_audio_read_back"]:
        print("  >", t)
    print("tools run:", [(t, p.get("args")) for t, p in res["tools"]])
    print("call-backs booked (then cancelled):", res["callbacks_booked"], "| cleanup cancelled:", res["cleanup_cancelled"])


def cmd_serve(args):
    import uvicorn

    _live_boot()
    uvicorn.run("carecall.api:app", host=args.host, port=args.port, reload=False, log_config=None)


def cmd_tunnel(args):
    """Open an ngrok tunnel to the local server, save its URL as PUBLIC_BASE_URL, and keep the tunnel open until Ctrl+C."""
    import time
    from pathlib import Path

    from .telephony.tunnel import open_tunnel, set_env_value

    url = open_tunnel(args.port, get_settings().ngrok_authtoken)
    set_env_value(Path(".env"), "PUBLIC_BASE_URL", url)
    print(f"tunnel up: {url} -> http://127.0.0.1:{args.port}")
    print("PUBLIC_BASE_URL written to .env. Start `carecall serve` and `carecall scheduler` AFTER this (they read it at start). Ctrl+C closes the tunnel.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("tunnel closed")


def cmd_routines(_):
    _boot()
    with session_scope() as s:
        for r in s.scalars(select(Routine)):
            print(f"{r.key:22} {r.type:18} {r.config}")


def cmd_callbacks(_):
    _boot()
    with session_scope() as s:
        for c in s.scalars(select(Callback)):
            print(f"#{c.id} call {c.call_id}: in {c.delay_minutes} min ({c.reason}) [{c.status}]")


def cmd_trace_check(_):
    s = get_settings()
    ok = tracing.init_tracing()
    print(f"Langfuse at {s.langfuse_host}: {'connected' if ok else 'NOT connected (check keys / docker compose up)'}")
    return 0 if ok else 1


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # Hindi / Tamil transcripts must never crash a command on a cp1252 console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="carecall")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed").set_defaults(fn=lambda a: (_boot(), print("seeded")))
    sub.add_parser("routines").set_defaults(fn=cmd_routines)
    c = sub.add_parser("chat")
    c.add_argument("--routine", required=True)
    c.set_defaults(fn=cmd_chat)
    e = sub.add_parser("eval")
    e.add_argument("--scenario", action="append")
    e.add_argument("--exclude-tag", action="append")
    e.add_argument("--out", default="runs")
    e.set_defaults(fn=cmd_eval)
    pe = sub.add_parser("personas")
    pe.add_argument("--mode", choices=["scripted", "llm"], default="scripted")
    pe.add_argument("--usecase", action="append")
    pe.add_argument("--persona", action="append")
    pe.add_argument("--agent", choices=["auto", "rules", "converse"], default="auto",
                    help="auto = scripted graph with an LLM interpreter if configured; rules = offline baseline; converse = the conversational tool-calling agent")
    pe.add_argument("--workers", type=int, default=1, help="personas to run in parallel (LLM calls are slow; 6-8 is fine)")
    pe.add_argument("--out", default="runs")
    pe.set_defaults(fn=cmd_personas)
    k = sub.add_parser("call")
    k.add_argument("--routine", required=True)
    k.add_argument("--key", help="idempotency key; repeating it will not create a second call")
    k.set_defaults(fn=cmd_call)
    ct = sub.add_parser("call-test")
    ct.add_argument("--to", help="number to dial; defaults to the seeded parent's phone")
    ct.set_defaults(fn=cmd_call_test)
    gp_ = sub.add_parser("gnani-personas", help="run personas against GNANI's agent over real phone calls")
    gp_.add_argument("--persona", action="append")
    gp_.add_argument("--usecase", action="append")
    gp_.add_argument("--dry-run", action="store_true", help="show prompts and what is missing; place no calls")
    gp_.add_argument("--console", action="store_true", help="no Platform API key: you click Trigger Agent Call in the console; the webhook returns the transcript")
    gp_.add_argument("--out", default="runs")
    gp_.set_defaults(fn=cmd_gnani_personas)
    dp = sub.add_parser("direct-personas", help="personas vs a GENERATIVE agent: the model writes the agent's replies (AGENT_LLM_*)")
    dp.add_argument("--persona", action="append")
    dp.add_argument("--usecase", action="append")
    dp.add_argument("--workers", type=int, default=1)
    dp.add_argument("--out", default="runs")
    dp.set_defaults(fn=cmd_direct)
    lt = sub.add_parser("latency", help="measure a voice turn stage by stage (STT, LLM, TTS)")
    lt.add_argument("--llm-trials", type=int, default=8)
    lt.add_argument("--speech-trials", type=int, default=4)
    lt.set_defaults(fn=cmd_latency)
    mb = sub.add_parser("model-bench", help="speed + quality benchmark of candidate OpenRouter models")
    mb.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    mb.add_argument("--trials", type=int, default=5)
    mb.add_argument("--finalists", type=int, default=5)
    mb.set_defaults(fn=cmd_model_bench)
    pd = sub.add_parser("plan-day", help="create the day's calls from the stored onboarding data (idempotent)")
    pd.add_argument("--date", help="parent-local date, YYYY-MM-DD (default: each parent's today)")
    pd.set_defaults(fn=cmd_plan_day)
    sub.add_parser("schedule", help="show what is on the calendar").set_defaults(fn=cmd_schedule)
    sub.add_parser("scheduler", help="run the dispatcher loop (plans, reaps stale calls, dials due calls)").set_defaults(fn=cmd_scheduler)
    oc = sub.add_parser("outcome", help="show/apply the outcome decision for a call id")
    oc.add_argument("call_id", type=int)
    oc.set_defaults(fn=cmd_outcome)
    sub.add_parser("live-scheduling", help="live test: LLM parents book call-backs/reminders in-call; checks DB, next call, human-likeness").set_defaults(fn=cmd_live_scheduling)
    tu = sub.add_parser("tool-use", help="test that the conversational agent calls the right tools for each task")
    tu.add_argument("--runs", type=int, default=1)
    tu.add_argument("--workers", type=int, default=6)
    tu.add_argument("--case")
    tu.set_defaults(fn=cmd_tool_use)
    sub.add_parser("gnani-setup").set_defaults(fn=cmd_gnani_setup)
    sub.add_parser("gnani-export", help="write paste-ready prompts/greetings for the Gnani console").set_defaults(fn=cmd_gnani_export)
    sub.add_parser("twilio-otp", help="show recent SMS to the persona number (Gnani whitelisting OTP)").set_defaults(fn=cmd_twilio_otp)
    sub.add_parser("twilio-setup").set_defaults(fn=cmd_twilio_setup)
    vs = sub.add_parser("voice-selftest", help="one persona vs OUR agent over real Gnani audio, no Twilio needed")
    vs.add_argument("--persona", default="mb_asks_medical_question_usha")
    vs.set_defaults(fn=cmd_voice_selftest)
    st = sub.add_parser("stream-test", help="live audio pipeline vs the running server, playing Twilio's part (no phone)")
    st.add_argument("--host", default="127.0.0.1:8000")
    st.add_argument("--routine", default="lunch_checkin")
    st.add_argument("--line", action="append", help="what the parent says, in order (repeat the flag)")
    st.set_defaults(fn=cmd_stream_test)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)
    tn = sub.add_parser("tunnel", help="ngrok tunnel to the local server; writes PUBLIC_BASE_URL into .env")
    tn.add_argument("--port", type=int, default=8000)
    tn.set_defaults(fn=cmd_tunnel)
    sub.add_parser("callbacks").set_defaults(fn=cmd_callbacks)
    sub.add_parser("trace-check").set_defaults(fn=cmd_trace_check)
    sub.add_parser("check-keys").set_defaults(fn=cmd_check_keys)
    sub.add_parser("langfuse-sync").set_defaults(fn=cmd_langfuse_sync)
    args = p.parse_args(argv)
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
