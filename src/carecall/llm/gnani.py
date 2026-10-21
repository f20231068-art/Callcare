"""Gnani reasoning model adapter. TODO: needs the API docs (endpoint, auth, request/response shape).

Implement `complete(system, user) -> str` returning the raw model text; `complete_json` in
base.py handles parsing and tracing. Nothing else in the codebase needs to change."""


class GnaniLLMClient:
    def __init__(self, api_key: str, model: str = "gnani-reasoning"):
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError("Gnani reasoning adapter not implemented yet: add the API details")
