import json

from ..config import get_settings
from .gnani import GnaniLLMClient
from .openai_compat import OpenAICompatClient


def make_llm_client():
    """Returns an LLM client, or None for provider 'fake' (callers then use offline rule-based fallbacks)."""
    s = get_settings()
    if s.llm_provider == "openai_compat":
        extra = json.loads(s.llm_extra_body) if s.llm_extra_body.strip() else None
        return OpenAICompatClient(s.llm_base_url, s.llm_api_key, s.llm_model, extra_body=extra)
    if s.llm_provider == "gnani":
        return GnaniLLMClient(s.llm_api_key, s.llm_model or "gnani-reasoning")
    return None


def make_persona_client():
    """Model for the simulated parent in TEXT tests: PERSONA_LLM_MODEL if set (a stronger model keeps to the backstory),
    else LLM_*. Temperature 0.7 so the parent is not robotic."""
    s = get_settings()
    base = make_llm_client()
    if base is None:
        return None
    if s.persona_llm_model:
        return OpenAICompatClient(s.llm_base_url, s.llm_api_key, s.persona_llm_model, temperature=0.7)
    return base.with_temperature(0.7)


def make_judge_client():
    """The model that JUDGES transcripts and writes test notes. JUDGE_LLM_MODEL if set (same endpoint/key as LLM_*), else LLM_*.
    Keep this stronger than the agent model: a weak judge makes every score less trustworthy."""
    s = get_settings()
    base = make_llm_client()
    if base is None or not s.judge_llm_model:
        return base
    extra = json.loads(s.judge_llm_extra_body) if s.judge_llm_extra_body.strip() else None
    return OpenAICompatClient(s.llm_base_url, s.llm_api_key, s.judge_llm_model, extra_body=extra)


def make_agent_llm_client():
    """The model that WRITES THE AGENT'S REPLIES. AGENT_LLM_* if set (e.g. Gnani's model), else the shared LLM_*."""
    s = get_settings()
    if s.agent_llm_base_url and s.agent_llm_model:
        extra = json.loads(s.agent_llm_extra_body) if s.agent_llm_extra_body.strip() else None
        return OpenAICompatClient(s.agent_llm_base_url, s.agent_llm_api_key, s.agent_llm_model,
                                  temperature=s.agent_llm_temperature, extra_body=extra)
    c = make_llm_client()
    return c.with_temperature(s.agent_llm_temperature) if hasattr(c, "with_temperature") else c
