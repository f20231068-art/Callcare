"""Gnani Agent Builder (Inya) Platform API client + the per-use-case agent definitions we deploy to it.

Why this exists: Gnani's reasoning model (Pampa) is only reachable as the brain of an Agent Builder agent. So to test
"Gnani's model as the agent" we (1) create an agent through this API, (2) put one system prompt per use case on it,
(3) have it place a real call, and (4) read the transcript back. Docs: docs.gnani.ai/Platform/*.

The agent is the CONVERSATIONAL brain only. It has no access to our database (that needs UI-configured Custom Actions),
so what it logs is judged from the transcript (see evals/gnani_runner.py)."""
import re
import time
from pathlib import Path

from .agent.routines import MEALS, spoken_time
from .config import get_settings
from .net import http_client

BASE = "https://api.inya.ai/platform"
PROMPTS = Path(__file__).parent / "prompts" / "gnani"
CAREGIVER = "Sree"


class GnaniAPIError(RuntimeError):
    pass


def render(template: str, ctx: dict) -> str:
    """Replace {{name}} placeholders. Missing keys are an error: a half-rendered prompt must never reach a call."""
    def sub(m):
        key = m.group(1).strip()
        if key not in ctx:
            raise KeyError(f"prompt variable {key!r} not provided")
        return str(ctx[key])
    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", sub, template)


def agent_context(card: dict | None = None, usecase: str | None = None) -> dict:
    """Variables baked into the system prompt for one call. The seeded data mirrors our own DB seed."""
    name = (card or {}).get("name", "Asha").split(",")[0].strip()
    routine = (card or {}).get("routine", "")
    ctx = {"parent_name": name, "caregiver_name": CAREGIVER, "medicine_name": "", "dose_note": "", "meal": "",
           "reminder_message": "You have an appointment with Doctor Rao tomorrow evening."}
    if routine == "metformin_reminder":
        ctx.update(medicine_name="Metformin", dose_note="500mg", meal="lunch")
    elif routine == "amlodipine_reminder":
        ctx.update(medicine_name="Amlodipine", dose_note="5mg", meal="dinner")
    elif routine.endswith("_checkin"):
        ctx["meal"] = routine.removesuffix("_checkin")
    return ctx


def system_prompt_for(usecase: str, ctx: dict) -> str:
    base = (PROMPTS / "base.md").read_text(encoding="utf-8")
    extra = (PROMPTS / f"{usecase}.md").read_text(encoding="utf-8")
    return render(base.strip() + "\n\n" + extra.strip(), ctx)


def opening_for(usecase: str, ctx: dict, lead_minutes: int = 15) -> tuple[str, str]:
    """(greetingMessage, endMessage). The first question is part of the greeting so the first turn is deterministic;
    the model handles everything after it."""
    hello = f"Hello {ctx['parent_name']}, this is the CareCall assistant, calling for {ctx['caregiver_name']}."
    end = f"Take care, {ctx['parent_name']}. Goodbye."
    if usecase == "meal_discovery":
        first = f"{hello} I would like to learn your usual meal times. What time do you usually have breakfast?"
    elif usecase == "meal_checkin":
        first = f"{hello} Have you had your {ctx['meal']} today?"
    elif usecase == "medicine_before_food":
        first = (f"{hello} It is time for your {ctx['medicine_name']}. Please take it now, about {lead_minutes} minutes "
                 f"before your {ctx['meal']}. Have you taken it?")
    elif usecase == "medicine_after_food":
        first = (f"{hello} It is time for your {ctx['medicine_name']}. Please take it now, after your {ctx['meal']}. "
                 "Have you taken it?")
    else:
        first = f"{hello} This is a gentle reminder. {ctx['reminder_message']} Is there anything else I can help you with? I can remind you again if you like."
    return first, end


class GnaniPlatform:
    def __init__(self, api_key: str | None = None, environment: str = "development", http=None):
        self.key = api_key or get_settings().gnani_platform_api_key
        self.env = environment
        self.http = http or http_client(60)

    # ---- plumbing ----
    def _req(self, method: str, path: str, params: dict | None = None, json: dict | None = None):
        if not self.key:
            raise GnaniAPIError("GNANI_PLATFORM_API_KEY is not set in .env")
        r = self.http.request(method, f"{BASE}{path}", params=params, json=json, headers={"x-api-key": self.key})
        try:
            body = r.json()
        except ValueError:
            body = {"message": r.text[:300]}
        if r.status_code >= 400:
            raise GnaniAPIError(f"{method} {path} -> HTTP {r.status_code} (requestId {body.get('requestId')}): "
                                f"{body.get('message')}")
        return body.get("response", body)

    # ---- discovery ----
    def llm_models(self, provider: str | None = None):
        return self._req("GET", "/v1/agents/config/llm_models", params={"provider": provider} if provider else None)

    def transcriber_config(self):
        return self._req("GET", "/v1/agents/config/transcriber")

    def tts_config(self):
        return self._req("GET", "/v1/agents/config/tts")

    # ---- agents ----
    def list_agents(self, search: str | None = None) -> list[dict]:
        out, page = [], 1
        while True:
            resp = self._req("GET", "/v1/agents", params={"pageNo": page, "pageSize": 50, **({"search": search} if search else {})})
            out += resp.get("bots", [])
            if page >= resp.get("totalPages", 1):
                return out
            page += 1

    def ensure_agent(self, name: str, description: str = "") -> str:
        for b in self.list_agents(search=name):
            if b["botName"] == name:
                return b["botId"]
        return self._req("POST", "/v1/agents", json={"botName": name, "description": description, "region": "asia",
                                                      "timeZone": "Asia/Kolkata"})["botId"]

    def get_agent(self, bot_id: str) -> dict:
        return self._req("GET", f"/v1/agents/{bot_id}")

    def update_agent(self, bot_id: str, **fields):
        return self._req("PUT", f"/v1/agents/{bot_id}", json=fields)

    def configure(self, bot_id: str, usecase: str, ctx: dict, llm_model: str = "Pampa Go", voice: str = "Kaveri",
                  asr_model: str = "gnani_asr", temperature: float = 0.3):
        greeting, end = opening_for(usecase, ctx, get_settings().medicine_lead_minutes)
        return self.update_agent(
            bot_id,
            systemPrompt=system_prompt_for(usecase, ctx), greetingMessage=greeting, endMessage=end,
            botDetails={"language": ["en-IN"], "region": "asia", "timeZone": "Asia/Kolkata",
                        "description": f"CareCall {usecase}"},
            asrParams={"provider": "gnani", "model": asr_model, "language": "en-IN"},
            llmParams={"provider": "gnani", "model": llm_model, "temperature": temperature, "maxTokens": 200},
            ttsParams={"provider": "gnani", "voice": voice, "language": "en-IN", "ttsRate": 1.0, "enableCaching": True},
            advanceSettings={"barge": True, "useDenoiser": True},
        )

    # ---- calls ----
    def trigger_call(self, bot_id: str, phone: str, country_code: str, name: str, client_ref: str):
        return self._req("POST", f"/v1/agents/{bot_id}/trigger_call", params={"environment": self.env},
                         json={"phone": phone, "countryCode": country_code, "name": name, "clientReferenceId": client_ref})

    def conversation_logs(self, bot_id: str, search: str | None = None, page_size: int = 20) -> list[dict]:
        body = {"pageNo": 1, "pageSize": page_size, **({"searchTerm": search} if search else {})}
        resp = self._req("POST", "/v1/conversations/logs", params={"botId": bot_id, "environment": self.env}, json=body)
        return resp.get("data", []) if isinstance(resp, dict) else resp

    def find_conversation(self, bot_id: str, client_ref: str, name: str, timeout: float = 240, poll: float = 6) -> dict | None:
        """The trigger response does not return the conversation id: poll the logs until our call shows up."""
        end = time.time() + timeout
        while time.time() < end:
            for term in (client_ref, name):
                for item in self.conversation_logs(bot_id, search=term):
                    blob = str(item)
                    if client_ref in blob:
                        return item
            time.sleep(poll)
        return None

    def stats(self, conversation_id: str) -> dict:
        return self._req("GET", f"/v1/conversations/{conversation_id}/stats")

    @staticmethod
    def transcript_of(stats: dict) -> list[dict]:
        """Normalise utteranceAnalytics to [{speaker, text, t}] (assistant = agent, user = parent)."""
        turns = stats.get("utteranceAnalytics") or stats.get("transcript") or []
        t0 = turns[0].get("timestamp", 0) if turns else 0
        out = []
        for u in turns:
            text = (u.get("content") or "").strip()
            if text:
                out.append({"speaker": "agent" if u.get("role") == "assistant" else "parent", "text": text,
                            "t": round((u.get("timestamp") or t0) - t0, 1)})
        return out
