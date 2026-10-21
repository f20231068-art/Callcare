"""HTTP + WebSocket server.

  POST /calls                 start a call for a routine (REST: create a resource)
  GET  /calls/{id}            status, summary, transcript
  POST /twilio/voice          Twilio asks 'what should I do when she answers?'  -> TwiML (webhook)
  POST /twilio/status         Twilio reports ringing / answered / completed     -> state machine (webhook)
  WS   /twilio/stream         the live audio pipe (see telephony/bridge.py)
  GET  /health

Phase 0 has no user auth: bind to localhost (the default) and expose only via the tunnel for Twilio.
Sync DB calls inside async handlers are short local writes; at scale you'd move them to a threadpool."""
from contextlib import asynccontextmanager

import asyncio
import json

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select

from . import tracing
from .calls.service import create_call
from .calls.state import transition
from .config import get_settings
from .db import init_db, is_initialized, session_scope
from .models import CallAttempt, CallSummary, Callback, TranscriptTurn
from .seed import seed
from .telephony.bridge import StreamBridge
from .telephony.stt import STTProvider
from .telephony.tts import TTSProvider
from .telephony.persona_bridge import REGISTRY, WEBHOOKS, PersonaCallBridge, voice_for
from .telephony.twilio_api import (STATUS_MAP, build_persona_twiml, build_twiml, place_call, validate_signature, verify_token)


class StartCall(BaseModel):
    routine: str
    idempotency_key: str | None = None


def stt_factory_for_persona():
    from .telephony.stt import GnaniSTT

    return GnaniSTT(min_silence_ms=600)


def create_app(stt_factory=None, tts_factory=None, interpreter=None, silence_timeout: float = 7.0) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_):
        if not is_initialized():  # tests (and the CLI) may have bound a database already
            init_db()
        if get_settings().seed_demo_data:
            seed()
        tracing.init_tracing()
        stop, task = asyncio.Event(), None
        if get_settings().enable_scheduler:  # plan the day, dispatch due calls, reap stale ones, forever
            from .calls.scheduler import scheduler_loop

            task = asyncio.create_task(scheduler_loop(place_call, stop))
        yield
        stop.set()
        if task:
            await task
        tracing.flush()

    app = FastAPI(title="CareCall", lifespan=lifespan)

    def make_stt() -> STTProvider:
        if stt_factory:
            return stt_factory()
        from .telephony.stt import GnaniSTT
        return GnaniSTT()

    def make_tts() -> TTSProvider:
        if tts_factory:
            return tts_factory()
        from .telephony.tts import GnaniTTS
        return GnaniTTS()

    def _admin(request: Request):
        tok = get_settings().admin_token
        if tok:
            if request.headers.get("Authorization", "") != f"Bearer {tok}":
                raise HTTPException(401, "admin token required")
        elif request.headers.get("x-forwarded-for") or (request.client and request.client.host not in ("127.0.0.1", "::1", "testclient")):
            raise HTTPException(403, "admin routes are local-only unless ADMIN_TOKEN is set (this request came through the tunnel)")

    @app.get("/health")
    def health():
        return {"ok": True, "tracing": tracing.enabled()}

    @app.post("/calls", status_code=201)
    def start_call(body: StartCall, request: Request):
        _admin(request)
        from .calls.service import has_live_call
        from .models import Parent

        with session_scope() as s:  # an idempotency key cannot tell us the parent is already on a call; this can
            pid = s.scalars(select(Parent.id)).first()
            live = has_live_call(s, pid) if pid else None
            if live is not None:
                raise HTTPException(409, f"parent is already on call {live.id} ({live.status})")
        try:
            call_id, created = create_call(body.routine, channel="twilio", idempotency_key=body.idempotency_key)
        except ValueError as e:
            raise HTTPException(404, str(e))
        sid = place_call(call_id) if created else None  # same idempotency key => no second dial
        return {"call_id": call_id, "created": created, "twilio_sid": sid}

    @app.get("/calls/{call_id}")
    def get_call(call_id: int):
        with session_scope() as s:
            call = s.get(CallAttempt, call_id)
            if call is None:
                raise HTTPException(404, "no such call")
            summ = s.get(CallSummary, call_id)
            turns = s.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == call_id)
                              .order_by(TranscriptTurn.turn_no)).all()
            return {"call_id": call_id, "status": call.status, "summary": summ.summary_text if summ else None,
                    "anomalies": summ.anomalies if summ else [],
                    "transcript": [{"n": t.turn_no, "speaker": t.speaker, "text": t.text} for t in turns]}

    # ---- admin: observe every parent's calls, outcomes and events (Langfuse has the traces; filter its Users tab by parent-<id>-<name>)
    @app.get("/admin/calls")
    def admin_calls(request: Request, parent_id: int | None = None, status: str | None = None, limit: int = 50):
        _admin(request)
        with session_scope() as s:
            q = select(CallAttempt).order_by(CallAttempt.id.desc()).limit(min(limit, 200))
            if parent_id:
                q = q.where(CallAttempt.parent_id == parent_id)
            if status:
                q = q.where(CallAttempt.status == status)
            return [{"call_id": c.id, "parent_id": c.parent_id, "routine": c.routine.key, "status": c.status, "kind": c.kind,
                     "attempt_no": c.attempt_no, "key": c.idempotency_key, "scheduled_at": c.scheduled_at, "outcome": c.outcome,
                     "outcome_reason": c.outcome_reason, "langfuse_user": f"parent-{c.parent_id}-{c.parent.preferred_name.lower()}",
                     "langfuse_session": f"call-{c.id}"} for c in s.scalars(q)]

    @app.get("/admin/calls/{call_id}")
    def admin_call(request: Request, call_id: int):
        _admin(request)
        from .models import Event, Notification

        with session_scope() as s:
            c = s.get(CallAttempt, call_id)
            if c is None:
                raise HTTPException(404, "no such call")
            return {"call": get_call(call_id), "outcome": c.outcome, "outcome_reason": c.outcome_reason, "end_reason": c.end_reason,
                    "langfuse_session": f"call-{c.id}", "events": [{"id": e.id, "type": e.type, "payload": e.payload} for e in s.scalars(
                        select(Event).where(Event.call_id == call_id).order_by(Event.id))],
                    "notifications": [{"caregiver_id": n.caregiver_id, "kind": n.kind, "body": n.body} for n in s.scalars(
                        select(Notification).where(Notification.call_id == call_id))]}

    @app.put("/admin/parents/{parent_id}/meal-times")
    def set_meal_times(request: Request, parent_id: int, times: dict[str, str]):
        """The caretaker fixes the daily meal times, e.g. {"breakfast": "08:00", "lunch": "13:00"}. Calls follow them from the next day planned."""
        _admin(request)
        from .agent.db_tools import call_tool

        out = {}
        for meal, hhmm in times.items():
            out[meal] = call_tool("update_meal_time", {"meal": meal, "time": hhmm, "evidence": "set by the caretaker via the admin API"},
                                  {"parent_id": parent_id, "call_id": None}, scope="admin")
        if not all(r.get("ok") for r in out.values()):
            raise HTTPException(422, out)
        return out

    @app.get("/admin/parents")
    def admin_parents(request: Request):
        _admin(request)
        from .models import Parent

        with session_scope() as s:
            return [{"parent_id": p.id, "name": p.preferred_name, "calling_enabled": p.calling_enabled, "timezone": p.timezone,
                     "langfuse_user": f"parent-{p.id}-{p.preferred_name.lower()}"} for p in s.scalars(select(Parent))]

    @app.get("/callbacks")
    def callbacks():
        with session_scope() as s:
            return [{"id": c.id, "call_id": c.call_id, "delay_minutes": c.delay_minutes, "reason": c.reason,
                     "status": c.status} for c in s.scalars(select(Callback))]


    # ---------------- caregiver accounts, onboarding and the dashboard (cookie session) ----------------
    from . import accounts, onboarding

    class SignupIn(BaseModel):
        name: str
        email: str
        password: str

    class LoginIn(BaseModel):
        email: str
        password: str

    class MedicineIn(BaseModel):
        name: str
        dose_note: str = ""
        timing: str = "after_food"
        meal_anchor: str = "lunch"
        offset_minutes: int | None = None

    class OnboardIn(BaseModel):
        parent_name: str
        preferred_name: str = ""
        phone: str
        timezone: str = ""
        language: str = "en"
        relationship: str = "child"
        consent: bool = False
        meals: dict[str, str] = {}
        medicines: list[MedicineIn] = []

    class ReminderIn(BaseModel):
        message: str
        in_minutes: float | None = None
        at: str | None = None

    def _me(request: Request) -> dict:
        me = accounts.account_for_token(request.cookies.get("cc_session"))
        if me is None:
            raise HTTPException(401, "please log in")
        return me

    def _guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except accounts.AuthError as e:
            raise HTTPException(422, str(e))

    def _set_cookie(request: Request, resp: Response, token: str):
        https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"  # behind the tunnel the browser uses https
        resp.set_cookie("cc_session", token, httponly=True, samesite="lax", secure=https,
                        max_age=get_settings().session_days * 86400)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_page():
        return (Path(__file__).parent / "web" / "dashboard.html").read_text(encoding="utf-8")

    @app.post("/auth/signup")
    def auth_signup(body: SignupIn, request: Request):
        _guard(accounts.signup, body.email, body.password, body.name)
        token = _guard(accounts.login, body.email, body.password)
        resp = JSONResponse({"ok": True})
        _set_cookie(request, resp, token)
        return resp

    @app.post("/auth/login")
    def auth_login(body: LoginIn, request: Request):
        token = _guard(accounts.login, body.email, body.password)
        resp = JSONResponse({"ok": True})
        _set_cookie(request, resp, token)
        return resp

    @app.post("/auth/logout")
    def auth_logout(request: Request):
        accounts.logout(request.cookies.get("cc_session"))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie("cc_session")
        return resp

    @app.get("/api/me")
    def api_me(request: Request):
        from .agent.routines import DEFAULT_MEAL_TIMES

        st = get_settings()
        return {**_me(request), "defaults": {"meals": DEFAULT_MEAL_TIMES, "timezone": st.default_timezone}}

    @app.post("/api/onboarding", status_code=201)
    def api_onboarding(body: OnboardIn, request: Request):
        me = _me(request)
        data = body.model_dump()
        data["medicines"] = [{k: v for k, v in m.items() if v is not None} for m in data["medicines"]]
        return _guard(onboarding.onboard_parent, me["caregiver_id"], data)

    @app.get("/api/dashboard")
    def api_dashboard(request: Request, parent_id: int | None = None):
        me = _me(request)
        return onboarding.snapshot(me["caregiver_id"], parent_id)

    @app.post("/api/parents/{parent_id}/reminders", status_code=201)
    def api_add_reminder(parent_id: int, body: ReminderIn, request: Request):
        me = _me(request)
        return _guard(onboarding.add_reminder, me["caregiver_id"], parent_id, body.message, body.in_minutes, body.at)

    @app.post("/api/parents/{parent_id}/calling")
    def api_calling(parent_id: int, enabled: bool, request: Request):
        me = _me(request)
        return _guard(onboarding.set_calling, me["caregiver_id"], parent_id, enabled)

    async def _verified_form(request: Request, call_id: int | None = None):
        form = dict(await request.form())
        public = get_settings().public_base_url.rstrip("/")
        url = f"{public}{request.url.path}" + (f"?{request.url.query}" if request.url.query else "")
        sig = request.headers.get("X-Twilio-Signature", "")
        if not sig and call_id is not None and verify_token(call_id, request.query_params.get("t", "")):
            # Observed on a trial account: the instructions request for a call arrives UNSIGNED. The per-call secret that we put in the
            # URL (an HMAC only Twilio and we ever see, over https) stands in for the signature, and only when the header is absent:
            # a signature that is present but wrong is still refused.
            import logging

            logging.getLogger("carecall.twilio").warning("call %s: no X-Twilio-Signature; accepted on the per-call token", call_id)
            return form
        if not validate_signature(url, form, sig):
            import logging

            tried = {u: validate_signature(u, form, sig) for u in (url.replace("https://", "http://"), url.split("?")[0],
                                                                  str(request.url), url + "&") if u != url}
            logging.getLogger("carecall.twilio").warning(
                "bad Twilio signature on %s: expected url %s, signature header present=%s, form keys=%s, other urls that would validate=%s, "
                "headers host=%s proto=%s", request.url.path, url, bool(sig), sorted(form), [u for u, ok in tried.items() if ok],
                request.headers.get("host"), request.headers.get("x-forwarded-proto"))
            raise HTTPException(403, "bad Twilio signature")
        return form

    @app.post("/twilio/voice")
    async def twilio_voice(request: Request, call_id: int):
        await _verified_form(request, call_id)
        if get_settings().twilio_voice_mode == "gather":
            from .telephony.gather import CALLS

            return Response(await CALLS.start(call_id), media_type="application/xml")
        return Response(build_twiml(call_id), media_type="application/xml")

    @app.post("/twilio/wait")
    async def twilio_wait(request: Request, call_id: int):
        """Gather mode: Twilio is back after our 'one moment'; answer if the agent has finished thinking."""
        from .telephony.gather import CALLS

        await _verified_form(request, call_id)
        return Response(await CALLS.wait(call_id), media_type="application/xml")

    @app.post("/twilio/turn")
    async def twilio_turn(request: Request, call_id: int):
        """Gather mode: Twilio posts what the parent said; we answer with what the agent says next."""
        from .telephony.gather import CALLS

        form = await _verified_form(request, call_id)
        try:
            conf = float(form.get("Confidence")) if form.get("Confidence") else None
        except ValueError:
            conf = None
        return Response(await CALLS.turn(call_id, (form.get("SpeechResult") or "").strip(), conf), media_type="application/xml")

    @app.post("/twilio/status")
    async def twilio_status(request: Request, call_id: int):
        form = await _verified_form(request)
        status = form.get("CallStatus", "")
        with session_scope() as s:
            call = s.get(CallAttempt, call_id)
            if call is None:
                raise HTTPException(404, "no such call")
            if status == "completed":
                from .telephony.gather import CALLS

                if CALLS.owns(call_id):  # gather mode: the session records how the call ended, not this webhook
                    await CALLS.ended(call_id)
                    return Response(status_code=204)
                # If our session already finished the call this is a duplicate and is ignored (idempotent).
                transition(s, call, "abandoned", "twilio reported completed before the agent finished")
            elif status in STATUS_MAP:
                transition(s, call, STATUS_MAP[status], f"twilio:{status}")
        if status in ("busy", "no-answer", "failed", "canceled"):  # never connected: decide the retry now (idempotent)
            from .calls.outcome import apply as apply_outcome
            from .llm.factory import make_llm_client

            await asyncio.to_thread(apply_outcome, call_id, make_llm_client())
        return Response(status_code=204)

    @app.websocket("/twilio/stream")
    async def twilio_stream(ws: WebSocket):
        await ws.accept()
        await StreamBridge(ws, make_stt(), make_tts(), interpreter, silence_timeout=silence_timeout).run()


    @app.post("/gnani/webhook")
    async def gnani_webhook(request: Request):
        """Gnani calls this when a call ends, with the whole transcript. Protected by a shared bearer token."""
        token = get_settings().gnani_webhook_token
        if token and request.headers.get("authorization", "") != f"Bearer {token}":
            raise HTTPException(401, "bad webhook token")
        WEBHOOKS.add(await request.json())
        return {"ok": True}

    @app.get("/gnani/webhooks/next")
    def gnani_webhook_next(after: float = 0.0):
        """Hand the oldest unconsumed post-call payload received at/after `after` (unix time) to the orchestrator."""
        return {"payload": WEBHOOKS.take_after(after)}

    @app.get("/gnani/webhooks")
    def gnani_webhooks():
        return [{"received": w["received"], "consumed": w["consumed"], "conversation_id": w["payload"].get("conversation_id"),
                 "turns": len(w["payload"].get("transcript", []))} for w in WEBHOOKS.items]

    # ---------------- persona line: an LLM plays a parent on a real call with Gnani's agent ----------------
    class NewPersonaJob(BaseModel):
        persona_id: str
        alias: str | None = None  # the name the caller uses for the parent, e.g. baked into a console agent prompt

    @app.post("/persona/jobs", status_code=201)
    def persona_new_job(body: NewPersonaJob):
        from .evals.personas import load_personas

        cards = load_personas(ids=[body.persona_id])
        if not cards:
            raise HTTPException(404, f"no persona {body.persona_id}")
        card = cards[0]
        if body.alias:  # the persona answers to the name the Gnani agent will use
            card = {**card, "name": f"{body.alias}, {card['name'].split(',')[-1].strip()}"}
        return {"job_id": REGISTRY.create(card).id}

    @app.get("/persona/jobs/{job_id}")
    def persona_job(job_id: str):
        job = REGISTRY.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.public()

    @app.post("/persona/voice")
    async def persona_voice(request: Request):
        """Twilio webhook for the persona number. The caller is Gnani's agent; hand the call to the oldest armed job."""
        await _verified_form(request)
        job = REGISTRY.claim_next()
        if job is None:  # nobody asked for a persona call: refuse rather than talk to a stranger
            return Response("<Response><Reject/></Response>", media_type="application/xml")
        return Response(build_persona_twiml(job.id, job.token), media_type="application/xml")

    @app.websocket("/persona/stream")
    async def persona_stream(ws: WebSocket):
        await ws.accept()
        first = None
        while first is None:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("event") == "start":
                first = raw
            elif msg.get("event") == "stop":
                return
        job = REGISTRY.jobs.get(json.loads(first)["start"].get("customParameters", {}).get("job_id", ""))
        if job is None:
            await ws.close(code=1008)
            return

        class Replay:  # hand the already-consumed 'start' message back to the bridge first
            def __init__(self):
                self.pending = [first]

            async def receive_text(self):
                return self.pending.pop(0) if self.pending else await ws.receive_text()

            async def send_text(self, t):
                await ws.send_text(t)

            async def close(self, code=1000):
                await ws.close(code=code)

        from .evals.personas import LLMPersona
        from .llm.factory import make_llm_client
        from .telephony.stt import GnaniSTT
        from .telephony.tts import GnaniTTS

        client = make_llm_client()
        persona = LLMPersona(job.card, client.with_temperature(0.7) if hasattr(client, "with_temperature") else client)
        bridge = PersonaCallBridge(Replay(), job, stt_factory_for_persona(), GnaniTTS(voice=voice_for(job.card)), persona)
        await bridge.run()

    return app


app = create_app()
