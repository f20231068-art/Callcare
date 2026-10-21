"""Settings, read from environment / .env. Every external dependency is optional so the
agent can run fully offline (fake LLM, no Langfuse, no Twilio) for tests."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///carecall.db"
    db_pool_size: int = 10  # Postgres connections kept open
    db_max_overflow: int = 20  # extra connections allowed under a burst of simultaneous calls
    force_ipv4: bool = True  # see net.py: broken IPv6 made every request wait 40-60 s

    # LLM: "fake" (rule-based, offline) | "openai_compat" | "gnani"
    llm_provider: str = "fake"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    # Optional separate model for the AGENT's replies (e.g. Gnani's Pampa). Falls back to LLM_* when empty.
    agent_llm_base_url: str = ""
    agent_llm_api_key: str = ""
    agent_llm_model: str = ""
    agent_llm_extra_body: str = ""
    agent_llm_temperature: float = 0.3
    # Optional separate model for the JUDGE and the test-note writer (quality matters there, latency does not).
    persona_llm_model: str = ""  # model that plays the parent in text tests (fidelity to the backstory matters)
    judge_llm_model: str = ""  # same endpoint/key as LLM_*; falls back to LLM_MODEL when empty
    judge_llm_extra_body: str = ""
    llm_extra_body: str = ""  # JSON merged into each request, e.g. {"reasoning": {"effort": "minimal"}}

    # Langfuse (self-hosted). Tracing is a no-op when keys are missing.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    public_base_url: str = ""  # e.g. https://abc123.ngrok-free.app (written by `carecall tunnel`)
    twilio_voice_mode: str = "stream"  # stream = live audio pipeline (needs a paid Twilio account); gather = turn-by-turn, works on a trial
    gather_voice: str = "Polly.Aditi"  # Twilio <Say> voice for gather mode
    gather_language: str = "en-IN"
    ngrok_authtoken: str = ""  # free token from dashboard.ngrok.com; only used by `carecall tunnel`
    stream_secret: str = "dev-secret-change-me"  # signs the per-call token carried in the media stream
    sms_notifications: bool = False  # send caregiver SMS via Twilio (otherwise printed to console)

    # Persona line: the Twilio number Gnani's agent will call (must be whitelisted by Gnani), split for their API
    persona_phone_e164: str = ""  # e.g. +14155550100
    persona_phone_cc: str = ""  # e.g. +1
    persona_phone_national: str = ""  # e.g. 4155550100
    gnani_bot_id: str = ""  # an existing Agent Builder agent to use for the tests (never overwritten without a backup)
    agent_mode: str = "converse"  # converse = the model holds the conversation and uses tools | graph = the older scripted state machine
    postcall_agent_enabled: bool = True  # a model decides the next step after a call (rules still guard it); False = rules only
    admin_token: str = ""  # if set, /admin/* needs "Authorization: Bearer <token>" (unset = open, localhost only)
    gnani_webhook_token: str = ""  # shared secret the Gnani post-call webhook must send as 'Authorization: Bearer <token>'
    persona_server_url: str = "http://127.0.0.1:8000"  # where `carecall serve` is listening locally

    # Gnani speech
    gnani_api_key: str = ""  # speech key (X-API-Key-ID)
    gnani_platform_api_key: str = ""  # Agent Builder key (x-api-key)
    gnani_stt_url: str = ""
    tts_streaming: bool = True  # speak while Gnani is still generating the audio (SSE); False = wait for the whole clip (REST)
    gnani_tts_url: str = ""

    # Call lifecycle policy (calls/outcome.py, calls/scheduler.py)
    cutoff_retry_delay_s: int = 0  # parent cut off mid-conversation -> ring back almost immediately
    max_cutoff_retries: int = 1
    hangup_retry_delay_min: int = 30  # parent ended the call on purpose -> try again later, once
    unreachable_delays_min: str = "10"  # no answer / busy: wait this long before attempt 2, 3, 4
    technical_delays_min: str = "5"  # our/Twilio failure
    postpone_live_call_s: int = 300  # parent is on another call: try again in 5 min
    max_attempts_per_occurrence: int = 2  # per reminder: the first call + ONE more, then we move on to the next thing
    quiet_start: str = "21:30"  # parent-local: no scheduled retries between these times
    quiet_end: str = "07:00"
    stale_dial_s: int = 180  # a call still dialing/ringing after this is considered lost
    stale_call_s: int = 1500  # an "in progress" call older than this lost its webhook
    scheduler_interval_s: int = 10
    default_country_code: str = "+91"  # a 10-digit number typed at onboarding gets this prefix
    default_timezone: str = "Asia/Kolkata"
    session_days: int = 14
    seed_demo_data: bool = False  # tests only: the hardcoded demo parent. Real data comes from onboarding.
    enable_scheduler: bool = False  # run the dispatcher inside `carecall serve`

    # Agent policy
    max_unclear_retries: int = 2
    medicine_lead_minutes: int = 15  # "before food" reminders fire this long before the meal
    prompt_version: str = "v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
