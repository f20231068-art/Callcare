"""Turn-by-turn phone mode for Twilio TRIAL accounts.

The live pipeline (telephony/bridge.py) needs `<Connect><Stream>`, which Twilio blocks on trial accounts ("The Stream verb is not
available on trial accounts"). This mode needs nothing but `<Say>` and `<Gather>`, which trials allow: Twilio itself listens
(`<Gather input="speech">`), posts what it heard to /twilio/turn, we answer with the next `<Say>` + `<Gather>`. The same CallSession
(agent, tools, scheduling, tracing) sits behind it. The cost: Twilio's voice and recogniser instead of Gnani's, and one HTTP round trip
per turn instead of a continuous audio stream. Choose it with TWILIO_VOICE_MODE=gather."""
import asyncio
import logging

from twilio.twiml.voice_response import Gather, VoiceResponse

from ..agent.session import CallSession
from ..config import get_settings
from .twilio_api import make_token

log = logging.getLogger("carecall.gather")


PATIENCE_S = 3.0  # how long a webhook waits for the agent before telling the caller "one moment" and asking Twilio to come back
MAX_WAITS = 6


class GatherCalls:
    def __init__(self, session_factory=CallSession, patience_s: float = PATIENCE_S):
        self.patience_s = patience_s
        self.pending: dict[int, asyncio.Task] = {}  # the agent's answer still being worked out, per call
        self.waits: dict[int, int] = {}
        self.sessions: dict[int, CallSession] = {}
        self.finishing: set[int] = set()
        self.factory = session_factory

    def _twiml(self, call_id: int, lines: list[str], done: bool) -> str:
        s = get_settings()
        resp = VoiceResponse()
        say = dict(voice=s.gather_voice, language=s.gather_language)
        if done:
            for line in lines:
                resp.say(line, **say)
            resp.hangup()
            return str(resp)
        action = f"{s.public_base_url.rstrip('/')}/twilio/turn?call_id={call_id}&t={make_token(call_id)}"
        g = Gather(input="speech", action=action, method="POST", language=s.gather_language, speech_timeout="auto", timeout=6,
                   speech_model="phone_call", action_on_empty_result=True)
        for line in lines:  # inside <Gather>, so the parent can talk over the agent
            g.say(line, **say)
        resp.append(g)
        resp.redirect(action, method="POST")  # nothing heard at all: let the agent handle the silence
        return str(resp)

    async def start(self, call_id: int) -> str:
        from ..llm.factory import make_judge_client

        session = self.factory(call_id, channel="twilio")
        session.postcall_client = make_judge_client()
        self.sessions[call_id] = session
        opening = await asyncio.to_thread(session.open)
        log.info("call %s opened (gather mode): %s", call_id, opening)
        return self._twiml(call_id, opening, False)

    def _wait_twiml(self, call_id: int) -> str:
        s = get_settings()
        resp = VoiceResponse()
        resp.say("One moment.", voice=s.gather_voice, language=s.gather_language)
        resp.pause(length=1)
        resp.redirect(f"{s.public_base_url.rstrip('/')}/twilio/wait?call_id={call_id}&t={make_token(call_id)}", method="POST")
        return str(resp)

    def _answer(self, call_id: int, session, heard: str, res) -> str:
        log.info("call %s heard %r -> %s (done=%s)", call_id, heard, res.reply, res.done)
        if res.done:
            self._finish_later(call_id, session, abandoned=False)
        return self._twiml(call_id, res.reply, res.done)

    async def turn(self, call_id: int, heard: str, confidence: float | None) -> str:
        session = self.sessions.get(call_id)
        if session is None:  # the server restarted mid-call: the memory of this conversation is gone
            log.warning("call %s: no session for this turn", call_id)
            self._abandon_orphan(call_id)
            return self._twiml(call_id, ["I am sorry, we got disconnected. I will call you again shortly. Goodbye."], True)
        task = asyncio.create_task(asyncio.to_thread(session.turn, heard, confidence))
        self.pending[call_id], self.waits[call_id] = task, 0
        task.heard = heard  # type: ignore[attr-defined]
        return await self._collect(call_id)

    async def wait(self, call_id: int) -> str:
        """Twilio came back after our 'one moment': hand over the answer if it is ready, else ask for a little more time."""
        if call_id not in self.pending:
            return self._twiml(call_id, ["Sorry, could you say that again?"], False)
        return await self._collect(call_id)

    async def _collect(self, call_id: int) -> str:
        task = self.pending[call_id]
        try:
            res = await asyncio.wait_for(asyncio.shield(task), self.patience_s)
        except asyncio.TimeoutError:
            self.waits[call_id] += 1
            if self.waits[call_id] > MAX_WAITS:
                log.error("call %s: the agent did not answer in time", call_id)
                self.pending.pop(call_id, None)
                return self._twiml(call_id, ["I am sorry, I am having trouble. I will call you again shortly. Goodbye."], True)
            log.info("call %s: slow turn, asking Twilio to come back (%s)", call_id, self.waits[call_id])
            return self._wait_twiml(call_id)
        self.pending.pop(call_id, None)
        return self._answer(call_id, self.sessions.get(call_id), getattr(task, "heard", ""), res)

    def _abandon_orphan(self, call_id: int):
        """The server lost this conversation (a restart). Record the call as abandoned and let the outcome engine retry it now,
        instead of leaving it 'in progress' until the reaper fails it 25 minutes later."""
        if call_id in self.finishing:
            return
        self.finishing.add(call_id)
        asyncio.create_task(asyncio.to_thread(self._abandon_orphan_sync, call_id))

    def _abandon_orphan_sync(self, call_id: int):
        from ..agent import tools
        from ..calls.outcome import apply as apply_outcome
        from ..calls.state import transition
        from ..db import session_scope
        from ..llm.factory import make_judge_client
        from ..models import CallAttempt

        try:
            with session_scope() as s:
                call = s.get(CallAttempt, call_id)
                if call is not None and transition(s, call, "abandoned", "the server lost this conversation (restart)"):
                    tools.add_event(s, call_id, "call.orphaned", {"reason": "no session for this turn"})
            apply_outcome(call_id, make_judge_client())
        except Exception:  # noqa: BLE001
            log.exception("call %s: could not record the orphaned call", call_id)
        finally:
            self.finishing.discard(call_id)

    def _finish_later(self, call_id: int, session, abandoned: bool):
        if call_id in self.finishing:
            return
        self.finishing.add(call_id)
        asyncio.create_task(self._finish(call_id, session, abandoned))

    async def _finish(self, call_id: int, session, abandoned: bool):
        try:
            if session is not None:
                await asyncio.to_thread(session.finish, abandoned or not session.done, None)
        except Exception:  # noqa: BLE001
            log.exception("call %s: finishing failed", call_id)
        finally:
            self.sessions.pop(call_id, None)
            self.finishing.discard(call_id)  # done: nothing grows for the life of the server

    def owns(self, call_id: int) -> bool:
        return call_id in self.sessions or call_id in self.finishing

    async def ended(self, call_id: int):
        """Twilio says the line closed. If the agent had not finished, the parent hung up: record it as abandoned."""
        session = self.sessions.get(call_id)
        if call_id not in self.finishing and session is not None:
            self._finish_later(call_id, session, abandoned=not session.done)


CALLS = GatherCalls()
