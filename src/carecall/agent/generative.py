"""A GENERATIVE agent: the model writes every reply after the opening line. No templates, no state machine.

This is what "use Gnani's model as the agent" means when you have the model's API directly: the same per-use-case conversation
prompt that we deploy to Agent Builder (prompts/gnani/*) is the system prompt, the conversation so far is the message history,
and whatever the model says is what the parent hears.

Trade-off vs the LangGraph agent (agent/graph.py): far more natural, but the flow is not guaranteed. What code still guarantees
here (agent/guardrails.py), whatever the model is:
  - risky requests (injection, OTP/bank, money, stories, tasks, contact details, trivia, opinions) get a CANNED reply and the
    model is never asked, so it cannot be talked into reciting a thousand lines or revealing its instructions;
  - every reply is capped at a sentence boundary (55 words / 140 tokens);
  - a reply that leaks the instructions or a phone number / e-mail is replaced;
  - three off-topic strikes end the call politely.
There are no tools and no database here, so "I will let Sree know" is a PROMISE that nothing fulfils unless the transcript is
processed afterwards. That is a real gap of this route."""
import re

from .. import tracing
from ..config import get_settings
from ..gnani_platform import agent_context, opening_for, system_prompt_for
from . import guardrails

END_RE = re.compile(r"\b(goodbye|bye)\b", re.I)
MAX_WORDS, MAX_TOKENS = 55, 140


class GenerativeAgent:
    def __init__(self, client, usecase: str, card: dict, max_turns: int = 14):
        self.client, self.usecase, self.max_turns = client, usecase, max_turns
        self.ctx = agent_context(card)
        self.system = system_prompt_for(usecase, self.ctx)
        self.history: list[dict] = []  # {"role": "assistant"|"user", "content": ...}
        self.done = False
        self.errors: list[str] = []
        self.strikes = 0
        self.kind_counts: dict[str, int] = {}
        self.q_asked: dict[str, int] = {}
        self.routine_question = ""
        self.guard_events: list[dict] = []  # what the code intercepted: audit trail for the report

    def open(self) -> str:
        """The first message is fixed (like Agent Builder's greetingMessage); everything after is generated."""
        greeting, _ = opening_for(self.usecase, self.ctx, get_settings().medicine_lead_minutes)
        self.history.append({"role": "assistant", "content": greeting})
        self._remember_question(greeting)
        return greeting

    def _remember_question(self, reply: str):
        """The routine question to steer back to. Ignore 'are you still there?' and 'anything to tell Sree?' style questions."""
        qs = [x for x in re.split(r"(?<=[.?!])\s+", reply) if x.strip().endswith("?")]
        qs = [q for q in qs if not re.search(r"still there|is that (okay|correct)|anything you would like", q, re.I)]
        if qs:
            self.routine_question = qs[-1].strip()

    def _redirect(self, n: int = 0) -> str:
        """Lead back into the check-in. Wording varies with how often the parent has wandered off."""
        q = self.routine_question
        return (guardrails.redirect_prefix(n) + q).strip() if q else ""

    @tracing.traced("agent_reply", as_type="generation")
    def reply(self, parent_text: str) -> str:
        cg = self.ctx["caregiver_name"]
        words = len(parent_text.split())
        hit = guardrails.detect(parent_text, cg, guardrails.spoken_date(), repeats=0)
        if hit and self.kind_counts.get(hit.kind):  # the same kind of request again: different words
            hit = guardrails.detect(parent_text, cg, guardrails.spoken_date(), repeats=1)
        self.history.append({"role": "user", "content": parent_text or "(silence: the person said nothing)"})
        text, hint, prefix_text = None, "", ""
        if hit:
            n = self.kind_counts[hit.kind] = self.kind_counts.get(hit.kind, 0) + 1
            self.guard_events.append({"kind": hit.kind, "relay": hit.relay, "canned": hit.kind in guardrails.HARD_CANNED})
            if hit.strike:
                self.strikes += 1
            mixed = words > 14 and hit.kind != "injection"  # a long message may hold an ANSWER next to the request
            if self.strikes >= 3:
                text = (f"{hit.reply} I do not want to keep you. I will let {cg} know and try again later. "
                        f"Take care, {self.ctx['parent_name']}. Goodbye.")
            elif hit.kind in guardrails.HARD_CANNED and not mixed:  # never improvise on these: no model call at all
                text = f"{hit.reply} {self._redirect(n - 1) if not hit.allowed else self.routine_question}".strip()
            elif hit.kind in guardrails.HARD_CANNED:
                # long message: the decline / fact is FIXED by code (so the model cannot invent a date or perform the request);
                # the model only handles whatever else the person said and continues the routine
                prefix = prefix_text = hit.reply
                hint = (f"\n\nNOTE FOR THIS TURN ONLY: your reply will be delivered AFTER this sentence, which has already been "
                        f"said to the person: \"{prefix}\". Do NOT repeat, rephrase or add to it. Just respond briefly to any "
                        f"answer they gave to your question, then continue the routine. Under 30 words.")
            else:
                hint = (f"\n\nNOTE FOR THIS TURN ONLY: the person's last message contains something you must handle with care "
                        f"(type: {hit.kind}). Respond warmly in at most two short sentences (say you will let {cg} know if it "
                        f"matters), then continue the routine. Under 40 words.")
        if text is None:
            try:
                text = self.client.chat(self.system + hint, self.history, max_tokens=MAX_TOKENS).strip()
            except Exception as e:  # never let a provider hiccup end the call silently
                self.errors.append(f"{type(e).__name__}: {str(e)[:120]}")
                text = "I am sorry, I did not catch that. Could you say that again?"
        if prefix_text:
            text = f"{prefix_text} {text}".strip()
        text, truncated = guardrails.cap_reply(text, 55 if prefix_text else (40 if hint else MAX_WORDS))
        if truncated:
            self.guard_events.append({"kind": "output_truncated"})
        if guardrails.leaks(text):  # the model leaked its instructions or a phone number / e-mail: replace the whole reply
            self.guard_events.append({"kind": "output_leak", "found": guardrails.leaks(text)})
            text = f"I can only help with your check-in. {self._redirect()}".strip()
        self._remember_question(text)
        if self.routine_question and self.routine_question in text:
            self.q_asked[self.routine_question] = self.q_asked.get(self.routine_question, 0) + 1
            if self.q_asked[self.routine_question] >= 3:  # the greeting asked it once; three re-asks is a loop, not a conversation
                self.guard_events.append({"kind": "loop_exit"})
                text = (f"I will let {cg} know and try again later. Take care, {self.ctx['parent_name']}. Goodbye.")
        tracing.update_generation(model=self.client.model, input=[{"role": "system", "content": self.system + hint}, *self.history],
                                  output=text, metadata={"guard": [e["kind"] for e in self.guard_events[-2:]]})
        self.history.append({"role": "assistant", "content": text})
        if END_RE.search(text) or sum(m["role"] == "assistant" for m in self.history) > self.max_turns:
            self.done = True
        return text
