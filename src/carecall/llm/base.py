"""LLM client interface. The agent only needs one capability: 'give me JSON for this prompt'."""
import json
import re
from typing import Protocol

from .. import tracing


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```fences``` or add a sentence around it. Be forgiving."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


@tracing.traced("llm", as_type="generation")
def complete_json(client: LLMClient, system: str, user: str, purpose: str = "") -> dict:
    """Call the model and parse JSON. Recorded in Langfuse as a generation (prompt + output)."""
    raw = client.complete(system, user)
    client.last_raw = raw  # kept so an unparseable reply can be recorded in the audit trail
    tracing.update_generation(
        model=client.model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        output=raw,
        metadata={"purpose": purpose},
    )
    return parse_json(raw)
