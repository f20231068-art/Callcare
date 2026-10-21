"""Client for any OpenAI-compatible /chat/completions endpoint (OpenRouter, OpenAI, many hosted models).

`extra_body` is merged into every request, which is how provider-specific knobs are passed, e.g. for OpenRouter
reasoning models: LLM_EXTRA_BODY={"reasoning": {"effort": "minimal"}} to cut latency."""
import copy
import time

import httpx

from ..net import http_client


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0,
                 temperature: float = 0.0, extra_body: dict | None = None, json_mode: bool = True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.stats = {"calls": 0, "seconds": 0.0, "call_seconds": [], "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        self.json_mode = json_mode
        self._http = http_client(timeout)  # one pooled client: connections are reused across calls

    def with_temperature(self, temperature: float) -> "OpenAICompatClient":
        """Same endpoint and model, different sampling temperature (e.g. livelier persona speech)."""
        c = copy.copy(self)
        c.temperature = temperature
        return c

    def _post(self, system: str, user: str, json_mode: bool, messages: list | None = None, max_tokens: int | None = None) -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}] + (messages or [{"role": "user", "content": user}]),
            **self.extra_body,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if max_tokens:
            body["max_tokens"] = max_tokens
        last: Exception | None = None
        t0 = time.perf_counter()
        for attempt in range(3):  # transient network / 429 / 5xx: back off and retry
            try:
                r = self._http.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"},
                               json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                u = data.get("usage") or {}
                st = self.stats
                st["calls"] += 1
                st["seconds"] += time.perf_counter() - t0
                st["call_seconds"].append(time.perf_counter() - t0)
                st["prompt_tokens"] += u.get("prompt_tokens", 0)
                st["completion_tokens"] += u.get("completion_tokens", 0)
                st["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                return data["choices"][0]["message"]["content"] or ""
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise last

    def complete(self, system: str, user: str) -> str:
        return self._post(system, user, self.json_mode)

    def complete_text(self, system: str, user: str) -> str:
        """Free text (no JSON mode): used for persona speech."""
        return self._post(system, user, json_mode=False)

    def chat_json(self, system: str, messages: list[dict], max_tokens: int | None = None) -> str:
        """Multi-turn completion in JSON mode: the conversational agent replies with one JSON object per step."""
        return self._post(system, "", json_mode=True, messages=messages, max_tokens=max_tokens)

    def chat(self, system: str, messages: list[dict], max_tokens: int | None = None) -> str:
        """Multi-turn free-text completion: messages = [{"role": "user"|"assistant", "content": ...}, ...]."""
        return self._post(system, "", json_mode=False, messages=messages, max_tokens=max_tokens)
