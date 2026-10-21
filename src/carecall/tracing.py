"""Thin wrapper over Langfuse so the rest of the code never imports it directly.

Why a wrapper: tracing must never be able to break a live call, and tests/offline runs should
work with no Langfuse at all. Every helper here is a no-op until `init_tracing()` succeeds.

Mental model: one *trace* per phone call (session_id = "call-<id>"). Inside it, *spans* for each
turn / graph node / tool, and *generations* for every LLM call (input, output, model, tokens).
"""
import contextlib
import functools
import logging

from .config import get_settings

log = logging.getLogger(__name__)
_client = None


def init_tracing():
    """Create the Langfuse client if keys are configured. Returns True when tracing is live."""
    global _client
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return False
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
        if not _client.auth_check():
            log.warning("Langfuse auth check failed at %s; tracing disabled", s.langfuse_host)
            _client = None
    except Exception as e:  # tracing must never stop the app
        log.warning("Langfuse init failed (%s); tracing disabled", e)
        _client = None
    return _client is not None


def enabled() -> bool:
    return _client is not None


def traced(name: str, as_type: str = "span"):
    """Decorator: record the function as a span/generation/tool when tracing is live."""

    def deco(fn):
        wrapped = None

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            nonlocal wrapped
            if _client is None:
                return fn(*args, **kwargs)
            if wrapped is None:
                from langfuse import observe

                wrapped = observe(name=name, as_type=as_type)(fn)
            return wrapped(*args, **kwargs)

        return inner

    return deco


def new_trace_id() -> str | None:
    """One trace per call. We create the id up front and attach every turn to it, because a call's
    turns run in different threads/requests and can't share one open span."""
    if _client is None:
        return None
    return _client.create_trace_id()


def trace_url(trace_id: str | None) -> str | None:
    if _client is None or not trace_id:
        return None
    try:
        return _client.get_trace_url(trace_id=trace_id)  # asks the server for the project id
    except Exception as e:  # Langfuse down must not break a call or an eval run
        log.warning("trace url unavailable: %s", e)
        return None


@contextlib.contextmanager
def call_span(trace_id: str | None, name: str, call_id: int, parent_id: int, routine_key: str,
              tags: list[str] | None = None, input=None, metadata: dict | None = None,
              trace_name: str | None = None, session_id: str | None = None, user_id: str | None = None):
    """A span (e.g. one turn) attached to the call's trace. No-op when tracing is off."""
    if _client is None or not trace_id:
        yield None
        return
    from langfuse import propagate_attributes

    s = get_settings()
    with propagate_attributes(
        session_id=session_id or f"call-{call_id}",
        user_id=user_id or f"parent-{parent_id}",
        tags=(tags or []) + [routine_key],
        version=s.prompt_version,
        trace_name=trace_name or f"call:{routine_key}",
        metadata={"routine": routine_key, **{k: str(v)[:190] for k, v in (metadata or {}).items()}},
    ):
        with _client.start_as_current_observation(
            trace_context={"trace_id": trace_id}, name=name, as_type="span", input=input
        ) as span:
            yield span


def update_span(**kwargs):
    if _client is not None:
        _client.update_current_span(**kwargs)


def update_generation(**kwargs):
    if _client is not None:
        _client.update_current_generation(**kwargs)


def score(trace_id: str | None, name: str, value, comment: str | None = None, data_type=None):
    if _client is None or not trace_id:
        return
    try:
        _client.create_score(trace_id=trace_id, name=name, value=value, comment=comment, data_type=data_type)
    except Exception as e:  # scoring must never break a run
        log.warning("score failed: %s", e)


def flush():
    if _client is not None:
        _client.flush()
