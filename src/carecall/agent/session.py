"""CallSession: one live conversation. This is the single entry point used by every channel
(text CLI, evals, Twilio bridge). Channels only move words in and out; behavior lives here.

Lifecycle:  open() -> turn()* -> finish()
  open():   build context, mark the call in_progress, return the opening words.
  turn():   one parent utterance in -> agent words out. Persists both sides of the transcript.
  finish(): mark the call completed/abandoned and run the post-call pipeline.
"""
import time
from dataclasses import dataclass

from .. import tracing
from ..config import get_settings
from ..db import session_scope
from ..llm.factory import make_llm_client
from ..models import CallAttempt, TranscriptTurn, as_utc
from ..calls.state import transition
from . import tools
from .context import build_context
from .graph import build_graph
from .interpret import Interpreter, make_interpreter


@dataclass
class TurnResult:
    reply: list[str]
    done: bool
    end_reason: str = ""
    latency_ms: int = 0
    rate: float = 1.0  # TTS speaking-rate hint (slower after the parent asked us to repeat)


class CallSession:
    def __init__(self, call_id: int, interpreter: Interpreter | None = None, channel: str = "text",
                 tags: list[str] | None = None, metadata: dict | None = None, trace_name: str | None = None,
                 session_id: str | None = None, followups: bool | None = None, user_id: str | None = None, mode: str | None = None, llm=None):
        self.call_id = call_id
        self.followups = channel == "twilio" if followups is None else followups  # evals/CLI chats must not schedule real calls
        self.tags, self.metadata = tags or [], metadata or {}
        self.trace_name, self.session_id = trace_name, session_id
        self.postcall_client = None  # model for the post-call decision; default: the interpreter's
        self.user_id = user_id  # Langfuse 'user'; set after open() to parent-<id>-<name>, or eval:<scenario> for test runs
        self.channel = channel
        self.mode = mode or ("graph" if interpreter is not None else get_settings().agent_mode)
        self.speak_now = None  # set by the phone bridge: a function that speaks a line immediately (for fillers)
        if self.mode == "converse":
            from types import SimpleNamespace

            from ..llm.factory import make_agent_llm_client
            from .converse import build_conversation_graph

            llm = llm or make_agent_llm_client()
            if llm is None:
                raise RuntimeError("the conversational agent needs an LLM: set LLM_* in .env (or use mode='graph')")
            self.interpreter = SimpleNamespace(name=f"converse:{llm.model}", client=llm)  # same shape the rest of the code reads
            self.graph = build_conversation_graph(llm, lambda: self.speak_now)
        else:
            self.interpreter = interpreter or make_interpreter(make_llm_client())
            self.graph = build_graph(self.interpreter, lambda: self.speak_now)
        self.config = {"configurable": {"thread_id": f"call-{call_id}"}}
        self.turn_no = 0
        self.done = False
        self.end_reason = ""
        self.trace_id = tracing.new_trace_id()
        self.ctx: dict = {}

    # --- internals ---
    def _user_id(self) -> str:
        """Who this trace belongs to in Langfuse: a real parent, or a test scenario (never mixed with real users)."""
        if self.user_id:
            return self.user_id
        if self.session_id:  # eval runs pass session_id="<run>:<scenario>"
            return "eval:" + self.session_id.rsplit(":", 1)[-1]
        name = str(self.ctx.get("parent_preferred_name", "")).lower().replace(" ", "-")
        return f"parent-{self.ctx.get('parent_id', 0)}" + (f"-{name}" if name else "")

    def _span(self, name: str, input=None):
        return tracing.call_span(self.trace_id, name, self.call_id, self.ctx.get("parent_id", 0),
                                 self.ctx.get("routine_key", "?"), tags=[self.channel] + self.tags, input=input,
                                 metadata=self.metadata, trace_name=self.trace_name, session_id=self.session_id, user_id=self._user_id())

    def _save_turn(self, speaker: str, text: str, latency_ms: int | None = None, conf: float | None = None):
        self.turn_no += 1
        with session_scope() as s:
            s.add(TranscriptTurn(call_id=self.call_id, turn_no=self.turn_no, speaker=speaker, text=text,
                                 latency_ms=latency_ms, stt_confidence=conf))

    def current_step_id(self) -> str | None:
        """Which question the agent is waiting on right now (used by scripted personas)."""
        st = self.graph.get_state(self.config).values
        steps, idx = st.get("steps", []), st.get("idx", 0)
        return steps[idx]["id"] if idx < len(steps) else None

    # --- public API ---
    def open(self) -> list[str]:
        with session_scope() as s:
            ctx = build_context(s, self.call_id)
            call = s.get(CallAttempt, self.call_id)
            call.prompt_version = get_settings().prompt_version
            call.model_version = self.interpreter.name
            call.channel = self.channel
            transition(s, call, "in_progress", "session opened")
            tools.add_event(s, self.call_id, "context.built", ctx.model_dump())
        self.ctx = ctx.model_dump()
        with self._span("open", input=self.ctx) as span:
            out = self.graph.invoke({"ctx": self.ctx, "started": False}, self.config)
            reply = out["reply"]
            if span:
                span.update(output=reply)
        for line in reply:
            self._save_turn("agent", line)
        return reply

    def turn(self, utterance: str, stt_confidence: float | None = None) -> TurnResult:
        if self.done:
            return TurnResult([], True, self.end_reason)
        t0 = time.perf_counter()
        self._save_turn("parent", utterance, conf=stt_confidence)
        with self._span(f"turn:{self.turn_no}", input=utterance) as span:
            out = self.graph.invoke({"utterance": utterance}, self.config)
            reply = out["reply"]
            if span:
                span.update(output=reply)
        latency = int((time.perf_counter() - t0) * 1000)
        for line in reply:
            self._save_turn("agent", line, latency_ms=latency)
        to_speak = list(reply)
        for line in out.get("early", []):  # already said live while the tool ran: in the transcript, not spoken twice
            if line in to_speak:
                to_speak.remove(line)
        self.done = bool(out.get("done"))
        self.end_reason = out.get("end_reason", "")
        return TurnResult(to_speak, self.done, self.end_reason, latency, out.get("rate", 1.0))

    def finish(self, abandoned: bool = False, hangup: dict | None = None) -> None:
        """Close the call, record how it ended, run post-call processing, then decide the next call. Safe to call twice.
        `hangup` = what the audio layer saw at the moment the line closed ({"parent_speaking": bool, "pending_text": str})."""
        from ..calls.outcome import apply as apply_outcome
        from ..calls.post_call import run_post_call

        state = self.graph.get_state(self.config).values
        pend = state.get("pending_meal")
        if pend:  # parent said 'yes I ate' but never gave details: still record the meal, items unknown
            tools.log_meal(self.ctx, pend["meal_type"], "eaten", [], pend["evidence"], pend["confidence"])
        with session_scope() as s:
            call = s.get(CallAttempt, self.call_id)
            ok = self.done and not abandoned
            transition(s, call, "completed" if ok else "abandoned", self.end_reason or "hangup")
            call.end_reason = self.end_reason or ""
            steps, idx = state.get("steps", []), state.get("idx", 0)
            tools.add_event(s, self.call_id, "call.finished", {
                "done": self.done, "end_reason": self.end_reason, "steps_total": len(steps), "idx": idx,
                "steps": steps, "steps_resolved": [x["id"] for x in steps[:idx]], "unresolved": state.get("unresolved", []),
                "learned": state.get("learned", {}), "pending_meal": state.get("pending_meal"),
                "parent_speaking_at_hangup": bool((hangup or {}).get("parent_speaking")),
                "pending_stt_text": (hangup or {}).get("pending_text", ""),
                "duration_s": int((as_utc(call.ended_at) - as_utc(call.started_at)).total_seconds()) if call.started_at and call.ended_at else None})
        with self._span("post_call"):
            run_post_call(self.call_id, trace_id=self.trace_id)
            if self.followups:
                apply_outcome(self.call_id, self.postcall_client or (self.interpreter.client if hasattr(self.interpreter, "client") else None))
        tracing.flush()
