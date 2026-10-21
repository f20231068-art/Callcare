import pytest

from carecall import tracing
from carecall.config import get_settings
from carecall.db import init_db
from carecall.seed import seed


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path, request):
    """Every test: fresh in-memory DB, no Langfuse, no Twilio creds, rule-based interpreter."""
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                "TWILIO_FROM_NUMBER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    name = request.node.name
    if request.node.module.__name__ != "test_lifecycle" and not any(w in name for w in ("quiet", "night", "clock_time")):
        # a call-back "in 30 minutes" must not depend on what time of day the suite runs (the real window is 21:30-07:00)
        monkeypatch.setenv("QUIET_START", "23:59")
        monkeypatch.setenv("QUIET_END", "00:00")
    monkeypatch.setenv("AGENT_MODE", "graph")  # the older tests exercise the scripted graph; converse tests pass mode="converse" and a fake model
    # A file DB per test (not :memory:): in-memory SQLite shares one connection across threads, which would
    # make the server thread and the test thread corrupt each other's transactions.
    import os
    url = os.getenv("TEST_DATABASE_URL") or f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.test")
    monkeypatch.setenv("SEED_DEMO_DATA", "true")
    monkeypatch.setenv("TWILIO_VOICE_MODE", "stream")  # the real .env may say gather
    get_settings.cache_clear()
    tracing._client = None
    eng = init_db(url)
    if eng.dialect.name == "postgresql":  # a real Postgres is shared: start every test from an empty schema
        from carecall.models import Base
        Base.metadata.drop_all(eng)
        Base.metadata.create_all(eng)
    seed()
    yield
    get_settings.cache_clear()
