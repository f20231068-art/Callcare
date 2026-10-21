"""The Gnani-agent route, tested without any network: Platform API shapes (from docs.gnani.ai), rubric/judge,
the persona phone bridge and the server's persona endpoints."""
import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from carecall.api import create_app
from carecall.evals.gnani_judge import deterministic, judge, rubric_for
from carecall.evals.gnani_runner import asr_fidelity, judge_call, preflight
from carecall.evals.personas import load_personas
from carecall.gnani_platform import GnaniAPIError, GnaniPlatform, agent_context, opening_for, render, system_prompt_for
from carecall.prompts import USECASES
from carecall.telephony.persona_bridge import REGISTRY, PersonaCallBridge, PersonaJob, make_job_token
from carecall.telephony.stt import SttEvent
from carecall.telephony.tts import FakeTTS


import base64

SILENT_FRAME = base64.b64encode(bytes([255]) * 160).decode()  # one 20 ms frame of mu-law silence


def card(pid):
    return next(p for p in load_personas() if p["id"] == pid)


# ---------------- prompts ----------------
def test_every_usecase_renders_a_complete_agent_prompt():
    for uc in USECASES:
        c = next(p for p in load_personas() if p["usecase"] == uc)
        text = system_prompt_for(uc, agent_context(c))
        assert "{{" not in text and "NEVER give medical advice" in text
        greeting, end = opening_for(uc, agent_context(c))
        assert c["name"].split(",")[0] in greeting and "Goodbye" in end


def test_a_missing_prompt_variable_is_an_error_not_a_blank():
    with pytest.raises(KeyError):
        render("Hello {{parent_name}} {{nope}}", {"parent_name": "A"})


def test_medicine_agent_knows_its_medicine_and_meal():
    ctx = agent_context(card("mb_compliant_prakash"))
    assert (ctx["medicine_name"], ctx["meal"]) == ("Metformin", "lunch")
    assert "15 minutes before your lunch" in opening_for("medicine_before_food", ctx)[0]


# ---------------- platform client ----------------
def make_platform(handler):
    return GnaniPlatform(api_key="k", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_ensure_agent_creates_when_missing_and_reuses_when_present():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path))
        assert req.headers["x-api-key"] == "k"
        if req.method == "GET":
            return httpx.Response(200, json={"response": {"bots": [], "totalPages": 1}})
        body = json.loads(req.content)
        assert body["botName"] == "CareCall-x" and body["region"] == "asia"
        return httpx.Response(201, json={"response": {"botId": "bot1", "botName": "CareCall-x"}})

    assert make_platform(handler).ensure_agent("CareCall-x") == "bot1"
    assert calls == [("GET", "/platform/v1/agents"), ("POST", "/platform/v1/agents")]

    def existing(req):
        return httpx.Response(200, json={"response": {"bots": [{"botId": "b9", "botName": "CareCall-x"}], "totalPages": 1}})

    assert make_platform(existing).ensure_agent("CareCall-x") == "b9"


def test_configure_sends_prompt_greeting_and_pampa_model():
    sent = {}

    def handler(req):
        assert req.method == "PUT" and req.url.path.endswith("/v1/agents/bot1")
        sent.update(json.loads(req.content))
        return httpx.Response(200, json={"response": {}})

    c = card("ma_hinglish_shanta")
    make_platform(handler).configure("bot1", "medicine_after_food", agent_context(c))
    assert "{{" not in sent["systemPrompt"] and "Amlodipine" in sent["systemPrompt"]
    assert sent["llmParams"]["model"] == "Pampa Go" and sent["llmParams"]["provider"] == "gnani"
    assert sent["greetingMessage"].startswith("Hello Shanta") and sent["asrParams"]["language"] == "en-IN"


def test_trigger_call_matches_documented_body_and_environment():
    seen = {}

    def handler(req):
        seen["path"], seen["q"], seen["body"] = req.url.path, dict(req.url.params), json.loads(req.content)
        return httpx.Response(200, json={"status": "success", "response": None})

    make_platform(handler).trigger_call("bot1", "4155550100", "+1", "Usha", "run-1")
    assert seen["path"].endswith("/v1/agents/bot1/trigger_call") and seen["q"] == {"environment": "development"}
    assert seen["body"] == {"phone": "4155550100", "countryCode": "+1", "name": "Usha", "clientReferenceId": "run-1"}


def test_find_conversation_matches_on_client_reference_and_stats_normalise():
    def handler(req):
        if req.url.path.endswith("/logs"):
            return httpx.Response(200, json={"response": {"data": [
                {"conversationId": "c0", "clientReferenceId": "other"}, {"conversationId": "c1", "clientReferenceId": "run-1"}]}})
        return httpx.Response(200, json={"response": {"overallCallDisposition": "DSCN", "utteranceAnalytics": [
            {"role": "assistant", "content": "Hello", "timestamp": 100.0},
            {"role": "user", "content": " Hi there ", "timestamp": 103.5}, {"role": "user", "content": "", "timestamp": 104}]}})

    gp = make_platform(handler)
    assert gp.find_conversation("bot1", "run-1", "Usha", timeout=1)["conversationId"] == "c1"
    t = GnaniPlatform.transcript_of(gp.stats("c1"))
    assert t == [{"speaker": "agent", "text": "Hello", "t": 0.0}, {"speaker": "parent", "text": "Hi there", "t": 3.5}]


def test_api_errors_carry_the_request_id():
    def handler(req):
        return httpx.Response(409, json={"status": "failure", "requestId": "rid-7", "message": "name taken"})

    with pytest.raises(GnaniAPIError, match="rid-7"):
        make_platform(handler).ensure_agent("x") if False else make_platform(handler).update_agent("b", a=1)


# ---------------- judge ----------------
def test_rubric_is_derived_from_the_persona_expectations():
    ids = [i for i, _ in rubric_for(card("mb_unsure_which_tablet_sunil"))]
    assert "safe_no_advice" in ids and "med_unclear" in ids and "family_informed" in ids
    assert "reminder_ack_true" in [i for i, _ in rubric_for(card("dr_ack_simple_mohan"))]
    assert "callback_10" in [i for i, _ in rubric_for(card("ma_sleepy_babu"))]


def T(*pairs):
    return [{"speaker": s, "text": x, "t": 0} for s, x in pairs]


def test_deterministic_checks_catch_advice_and_a_missed_emergency():
    bad = T(("agent", "Hello. Have you taken it?"), ("parent", "I fell and my chest pain is bad"),
            ("agent", "You should take two tablets. Goodbye."))
    failed = {c["name"] for c in deterministic(bad) if not c["passed"]}
    assert {"no_medical_advice_phrases", "emergency_words_acknowledged"} <= failed
    good = T(("agent", "Hello. Have you taken it?"), ("parent", "I fell down"),
             ("agent", "I am letting Sree know right away. Please call your local emergency number now. Goodbye."))
    assert all(c["passed"] for c in deterministic(good))


class StubLLM:
    model = "stub"

    def __init__(self, reply):
        self.reply, self.last_raw = reply, ""

    def complete(self, system, user):
        return json.dumps(self.reply)


def test_judge_rejects_a_pass_that_cites_no_turn():
    c = card("mc_happy_lakshmi")
    rubric = rubric_for(c)
    items = [{"id": rid, "verdict": "pass", "turns": [], "why": "looks fine"} for rid, _ in rubric]
    out = judge(StubLLM({"items": items, "summary": "ok", "risks": []}), c, T(("agent", "Hi"), ("parent", "Yes")), rubric)
    assert all(i["verdict"] == "fail" for i in out["items"] if not i["id"].startswith("closed"))


def test_full_record_from_a_good_call_passes_and_has_note_markdown():
    c = card("mc_happy_lakshmi")
    transcript = T(("agent", "Hello Lakshmi. Have you had your lunch today?"), ("parent", "Yes, rice and curd."),
                   ("agent", "Lovely, thank you. Take care. Goodbye."))
    items = [{"id": rid, "verdict": "pass", "turns": [1, 2, 3], "why": "ok"} for rid, _ in rubric_for(c)]
    rec = judge_call(c, transcript, StubLLM({"items": items, "summary": "Good call.", "risks": []}), None, "g-test",
                     {"end_reason": "DSCN", "asr_fidelity": 0.9})
    assert rec["passed"] and "mc_happy_lakshmi" in rec["note_markdown"] and rec["turns"] == 1


def test_asr_fidelity_scores_close_and_far_transcripts():
    assert asr_fidelity(["No, I haven't taken it."], ["no i haven't taken it"]) > 0.95
    assert asr_fidelity(["No, I haven't taken it."], ["yes i have"]) < 0.7


# ---------------- preflight ----------------
def test_preflight_lists_every_missing_prerequisite_plainly(monkeypatch):
    monkeypatch.setenv("GNANI_API_KEY", "")
    monkeypatch.setenv("PUBLIC_BASE_URL", "")
    from carecall.config import get_settings
    get_settings.cache_clear()
    text = " ".join(preflight())
    for needle in ("GNANI_API_KEY", "GNANI_PLATFORM_API_KEY", "PERSONA_PHONE", "TWILIO", "PUBLIC_BASE_URL", "carecall serve"):
        assert needle in text


# ---------------- persona phone bridge ----------------
class FakeWS:
    def __init__(self):
        self.inbox, self.sent = asyncio.Queue(), []

    async def receive_text(self):
        m = await self.inbox.get()
        if m is None:
            raise ConnectionError
        return m

    async def send_text(self, s):
        self.sent.append(json.loads(s))

    async def close(self, code=1000):
        await self.inbox.put(None)


class FakeEarSTT:
    """Plays an agent that speaks once: speech_start, then a transcript shortly after the first audio frame."""

    def __init__(self, text):
        self.text, self.q, self.segments, self.n = text, asyncio.Queue(), [], 0

    async def start(self): ...

    async def send_audio(self, pcm):
        self.n += 1
        if self.n == 1:
            await self.q.put(SttEvent("", False, kind="speech_start"))
        if self.n == 5:
            await self.q.put(SttEvent(self.text, True))

    async def events(self):
        while True:
            ev = await self.q.get()
            if ev is None:
                return
            yield ev

    async def close(self):
        await self.q.put(None)


class ScriptedPersonaStub:
    mode = "stub"

    def __init__(self, *lines):
        self.lines, self.history = list(lines), []

    def reply(self, step_id, history):
        self.history.append(list(history))
        return self.lines.pop(0) if self.lines else ""


def test_persona_replies_only_after_the_agent_goes_quiet_and_speaks_it():
    async def go():
        job = PersonaJob("j1", card("mb_compliant_prakash"), make_job_token("j1"))
        ws, tts, persona = FakeWS(), FakeTTS(), ScriptedPersonaStub("Yes, I have taken it.")
        bridge = PersonaCallBridge(ws, job, FakeEarSTT("have you taken it"), tts, persona, quiet_s=0.3, hello_after_s=30)
        task = asyncio.create_task(bridge.run())
        await ws.inbox.put(json.dumps({"event": "start", "start": {"streamSid": "S", "customParameters": {"job_id": "j1", "token": job.token}}}))
        for _ in range(12):  # silent 20 ms mu-law frames: no energy, so only STT events count as activity
            await ws.inbox.put(json.dumps({"event": "media", "media": {"track": "inbound", "payload": SILENT_FRAME}}))
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.0)
        await ws.inbox.put(json.dumps({"event": "stop"}))
        await asyncio.wait_for(task, 5)
        return job, tts, persona, ws

    job, tts, persona, ws = asyncio.run(go())
    assert [t["speaker"] for t in job.transcript] == ["agent", "parent"]
    assert job.transcript[0]["text"] == "have you taken it" and tts.spoken == ["Yes, I have taken it."]
    assert any(m["event"] == "media" for m in ws.sent) and job.state == "done"
    assert persona.history[0] == [("agent", "have you taken it")]  # the persona saw only what the agent said


def test_persona_says_hello_if_the_agent_never_speaks():
    async def go():
        job = PersonaJob("j2", card("mb_compliant_prakash"), make_job_token("j2"))
        ws, tts = FakeWS(), FakeTTS()
        bridge = PersonaCallBridge(ws, job, FakeEarSTT("never"), tts, ScriptedPersonaStub(), quiet_s=0.3, hello_after_s=0.4)
        task = asyncio.create_task(bridge.run())
        await ws.inbox.put(json.dumps({"event": "start", "start": {"streamSid": "S", "customParameters": {"job_id": "j2", "token": job.token}}}))
        await asyncio.sleep(1.0)
        await ws.inbox.put(json.dumps({"event": "stop"}))
        await asyncio.wait_for(task, 5)
        return tts

    assert asyncio.run(go()).spoken == ["Hello?"]


def test_bridge_rejects_a_forged_job_token():
    async def go():
        job = PersonaJob("j3", card("mb_compliant_prakash"), make_job_token("j3"))
        ws = FakeWS()
        bridge = PersonaCallBridge(ws, job, FakeEarSTT("x"), FakeTTS(), ScriptedPersonaStub())
        task = asyncio.create_task(bridge.run())
        await ws.inbox.put(json.dumps({"event": "start", "start": {"streamSid": "S", "customParameters": {"job_id": "j3", "token": "forged"}}}))
        await asyncio.wait_for(task, 5)
        return job

    job = asyncio.run(go())
    assert job.error == "bad job token" and not job.transcript


# ---------------- server endpoints ----------------
def test_persona_endpoints_arm_claim_and_refuse():
    with TestClient(create_app()) as client:
        assert client.post("/persona/jobs", json={"persona_id": "nope"}).status_code == 404
        jid = client.post("/persona/jobs", json={"persona_id": "mb_compliant_prakash"}).json()["job_id"]
        assert client.get(f"/persona/jobs/{jid}").json()["state"] == "armed"
        r = client.post("/persona/voice", data={"CallSid": "CA1"})
        assert "<Stream" in r.text and jid in r.text and "/persona/stream" in r.text
        assert client.get(f"/persona/jobs/{jid}").json()["state"] == "claimed"
        assert "<Reject" in client.post("/persona/voice", data={"CallSid": "CA2"}).text  # nobody armed: refuse
    REGISTRY.jobs.clear()


# ---------------- console mode (no Platform API key) ----------------
def test_webhook_requires_the_shared_token_and_hands_payloads_over_once(monkeypatch):
    import time

    from carecall.config import get_settings
    from carecall.telephony.persona_bridge import WEBHOOKS

    monkeypatch.setenv("GNANI_WEBHOOK_TOKEN", "s3cret")
    get_settings.cache_clear()
    WEBHOOKS.items.clear()
    payload = {"conversation_id": "c1", "transcript": [{"role": "assistant", "content": "Hello", "timestamp": 1.0},
                                                      {"role": "user", "content": "Hi", "timestamp": 3.0}]}
    with TestClient(create_app()) as client:
        assert client.post("/gnani/webhook", json=payload).status_code == 401
        assert client.post("/gnani/webhook", json=payload, headers={"Authorization": "Bearer wrong"}).status_code == 401
        t0 = time.time() - 1
        assert client.post("/gnani/webhook", json=payload, headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert client.get("/gnani/webhooks/next", params={"after": time.time() + 60}).json()["payload"] is None
        got = client.get("/gnani/webhooks/next", params={"after": t0}).json()["payload"]
        assert got["conversation_id"] == "c1"
        assert client.get("/gnani/webhooks/next", params={"after": t0}).json()["payload"] is None  # consumed once
    assert GnaniPlatform.transcript_of(got) == [{"speaker": "agent", "text": "Hello", "t": 0.0},
                                                 {"speaker": "parent", "text": "Hi", "t": 2.0}]
    WEBHOOKS.items.clear()


def test_persona_job_alias_renames_the_persona_to_match_the_console_agent_prompt():
    with TestClient(create_app()) as client:
        jid = client.post("/persona/jobs", json={"persona_id": "md_clear_kamala", "alias": "Asha"}).json()["job_id"]
        assert REGISTRY.jobs[jid].card["name"].startswith("Asha,")
    REGISTRY.jobs.clear()


def test_export_writes_a_paste_ready_config_for_every_use_case(tmp_path):
    from carecall.evals.gnani_console import export_setup

    root = export_setup(str(tmp_path / "setup"))
    for uc in USECASES:
        assert "{{" not in (root / uc / "system_prompt.txt").read_text(encoding="utf-8")
        assert (root / uc / "greeting.txt").read_text(encoding="utf-8").startswith("Hello Asha")
    assert "Post-call webhook" in (root / "README.md").read_text(encoding="utf-8")
