"""The per-call conversation graph (LangGraph).

How to read this file:
- State (`CallState`) is everything we know mid-call. LangGraph checkpoints it per call
  (thread_id = call id) so each parent utterance is one `invoke` that resumes where we left off.
- One *turn* = parent says something -> flows through the nodes below -> agent reply.

    START -> [not started?] open -> END
          -> safety --(severe words)--> escalate -> END
                    -> interpret -> validate --(model says emergency)--> escalate -> END
                                             -> act -> advance -> END

- `safety` is deterministic and runs BEFORE any model, so a model outage can't hide an emergency.
- `interpret` is the only place a model is used (prompt chosen per use case, see prompts/).
- `validate` is the false-log guard (code, not model): no verbatim quote, low confidence, hedged answers
  ("maybe", "not sure", "which tablet is this?") and filler "foods" are all downgraded to 'unclear'.
- `act` performs side effects (tools). All alerts raised in one turn are merged into ONE alert.
- `advance` decides the next words to speak. No model involved.
"""
import re
from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .. import tracing
from ..config import get_settings
from . import db_tools, guardrails, tools
from .context import CallContext
from .interpret import (InterpretRequest, Interpretation, Interpreter, RuleInterpreter, detect_emergency, detect_mild_symptom,
                        parse_time_strict)
from .routines import DEFAULT_MEAL_TIMES, build_steps, closing, confirm_tool_step, followup_items_step, greeting, message_offer_step, spoken_time, usecase_of

EMERGENCY_REPLY = ("I am letting {cg} know right away. If you need urgent help, "
                   "please call your local emergency number now.")
SLOW_RATE = 0.85  # speak slower once the parent has asked us to repeat ourselves

SEVERITY_ORDER = {"info": 0, "warning": 1, "emergency": 2}
HEDGE = re.compile(r"\b(maybe|not sure|i think|i guess|perhaps|probably|don'?t remember|do not remember|"
                   r"can'?t remember|cannot remember|which (?:tablet|one|medicine|pill)|what is this)\b")
HEDGE_STRONG = re.compile(r"\b(not sure|don'?t remember|do not remember|can'?t remember|cannot remember|no idea)\b")
APPEARANCE = re.compile(r"\bthe (?:white|small|big|red|blue|yellow|round|pink|green|large|little) (?:one|tablet|pill)\b")
FILLER_ITEMS = {"hmm", "mm", "mmm", "uh", "um", "yes", "no", "ok", "okay", "nothing", "something", "food", "lunch",
                "dinner", "breakfast", "snack", "yeah", "haan", "nahi", "ate", "had"}


class CallState(TypedDict, total=False):
    ctx: dict
    steps: list[dict]
    idx: int
    retries: int
    started: bool
    utterance: str
    last_utter: str  # previous parent utterance (detects a parent repeating the same words)
    interp: dict | None
    emergency: str | None
    decision: dict | None
    pending_meal: dict | None  # 'yes I ate' captured, waiting for the 'what did you eat' answer
    learned: dict  # meal -> "HH:MM" captured during meal_discovery
    unresolved: list[str]
    nudges: int  # repeat/wait/identity asks on the current step (capped)
    mild: str | None  # mild symptom words found this turn (deterministic)
    alerted: list[str]  # symptom/concern texts already reported this call
    rate: float  # TTS speaking-rate hint
    guard: dict | None  # guardrail hit for this turn (deterministic, found before any model)
    guard_done: bool  # the guardrail node already produced the whole reply for this turn
    guard_lead: list[str]  # decline line to put in front of the normal reply when the turn ALSO held an answer
    strikes: int  # off-topic requests so far; 3 ends the call
    guard_relayed: list[str]  # guard kinds already reported to the family
    offered_message: bool  # we already offered to pass a message to the family
    guard_counts: dict  # how many times each kind of request has come up
    early: list[str]  # lines already spoken live this turn (fillers like "umm, hold on"); the audio layer must not repeat them
    filler_n: int  # how many fillers so far (cycles through the wordings so it does not sound canned)
    requests_made: int  # calendar requests proposed so far in this call (capped)
    reply: list[str]
    done: bool
    end_reason: str


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s:']", " ", s.lower())).strip()


def _short(prompt: str) -> str:
    """For re-asks, repeat only the final question, not the whole reminder."""
    parts = [p for p in re.split(r"(?<=[.?])\s+", prompt) if p]
    return parts[-1] if parts else prompt


def validate_interpretation(interp: Interpretation, step: dict, utterance: str) -> tuple[Interpretation, str | None]:
    """Return (possibly downgraded interpretation, reason if downgraded). Pure code: no model involved."""
    if interp.intent in ("repeat", "identity", "wait") and step["kind"] == "time":
        t = parse_time_strict(utterance, step.get("topic", ""))  # "Eh? What was that? About half past seven!" has an answer in it
        if t:
            return (interp.model_copy(update={"intent": "answer", "time": t, "evidence": utterance, "confidence": 0.7,
                                              "reason": "answer found inside a repeat request"}), "answer_wins")
    if interp.intent != "answer":
        return interp, None

    def down(reason: str, why: str):
        return interp.model_copy(update={"intent": "unclear", "reason": reason}), why

    pieces = [_norm(p) for p in re.split(r"\.{3}|\u2026", interp.evidence or "") if _norm(p)]
    if not pieces or not all(p in _norm(utterance) for p in pieces):  # quoting two spans joined by "..." is fine; inventing words is not
        return down("evidence not found in utterance", "evidence_missing")
    if interp.confidence < 0.6:
        return down("low confidence", "low_confidence")
    kind = step["kind"]
    ok = {
        "yes_no": interp.yes_no is not None,
        "ack": interp.yes_no is not None,
        "items": bool(interp.items),
        "time": bool(interp.time and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", interp.time)),
        "message": interp.yes_no is not None,
    }[kind]
    if not ok:
        return down(f"missing field for {kind}", "missing_field")
    if kind == "time" and step.get("topic") in MEAL_WINDOWS and not _plausible_meal_time(step["topic"], interp.time):
        return down(f"implausible time {interp.time} for {step['topic']}", "implausible_time")
    low = utterance.lower()
    if kind in ("yes_no", "ack") and interp.yes_no is True:
        if step["handler"] == "medicine_taken" and (APPEARANCE.search(low) or re.search(r"which (?:tablet|one|medicine|pill)", low)):
            return down("hedged:which_tablet", "hedged_which_tablet")  # cannot verify it is THIS medicine
        # "I think I took it" is a real hedge about a medicine/meal; "I think I got it" about hearing a reminder is not
        m = (HEDGE if step["handler"] in ("meal_ate", "medicine_taken") else HEDGE_STRONG).search(low)
        if m:  # an honest "maybe / not sure" is information in itself, not a failed hearing
            return (interp.model_copy(update={"intent": "unsure", "yes_no": None, "reason": f"hedged:{m.group(0)}"}), "hedged")
    if kind != "items" and interp.items:
        interp = interp.model_copy(update={"items": [i for i in interp.items if i.strip().lower() not in FILLER_ITEMS and len(i.strip()) > 1]})
    if kind == "items":
        real = [i for i in interp.items if i.strip().lower() not in FILLER_ITEMS and len(i.strip()) > 1]
        if not real:
            return down("no concrete food named", "filler_items")
        interp = interp.model_copy(update={"items": real})
    return interp, None


def _concern_key(concern: str) -> str:
    """Group paraphrases of the same worry so a chatty parent does not trigger one alert per sentence."""
    c = concern.lower()
    if re.search(r"lonel|alone|quiet|miss|sad|isolat", c):
        return "concern:loneliness"
    if re.search(r"appetite|hungry|eating", c):
        return "concern:appetite"
    return "concern:" + c


# earliest/latest plausible usual time per meal, in minutes after midnight
MEAL_WINDOWS = {"breakfast": (4 * 60, 11 * 60 + 30), "lunch": (10 * 60 + 30, 16 * 60 + 30),
                "snack": (13 * 60, 19 * 60 + 30), "dinner": (17 * 60, 23 * 60 + 59)}


def _plausible_meal_time(meal: str, hhmm: str) -> bool:
    lo, hi = MEAL_WINDOWS.get(meal, (0, 24 * 60))
    h, m = int(hhmm[:2]), int(hhmm[3:])
    return lo <= h * 60 + m <= hi


def _concern_severity(concern: str) -> str:
    """Health or mood concerns are warnings; logistics (cannot attend) are info."""
    return "warning" if re.search(r"appetite|lonel|alone|isolat|\bsad|pain|sick|unwell|depress|hopeless|worr|anxi|cry|scared|afraid", concern.lower()) else "info"


LIVE_REQUEST_TOOLS = ("add_reminder",)  # the only tools the interpreter may propose by name; everything else is code-initiated
MAX_REQUESTS_PER_CALL = 2

# Said the moment a tool starts running, so the line is never dead silent. Written the way people stall: short, a little hesitant.
FILLERS = {
    "callback": ["Umm... hold on a moment, let me note that down.", "Okay... one second, let me put that in.",
                 "Hmm, just a moment... I am setting that up."],
    "reminder": ["Umm... hold on, let me set that reminder for you.", "One second... I am adding that to the calendar.",
                 "Okay, hold on a moment... let me save that."],
}


def when_phrase(res: dict) -> str:
    """How to SAY the time a tool actually booked: 'in 10 minutes' for near ones, 'at 7 in the morning tomorrow' otherwise."""
    if res.get("by_clock") or res.get("shifted") or res.get("tomorrow") or res.get("in_minutes", 0) > 90:
        return f"at {res['spoken']}" + (" tomorrow" if res.get("tomorrow") else "")
    m = max(1, int(res.get("in_minutes", 1)))
    return f"in {m} minute{'s' if m != 1 else ''}"


def plan_request(interp: Interpretation, utter: str, cd: dict) -> tuple[dict | None, str | None]:
    """Check ONE proposed calendar request. Returns (confirm step, None) to ask the parent, (None, reason) when it cannot be done,
    or (None, None) when it should be ignored (not an allowed tool, or the parent never said it)."""
    req = interp.requests[0]
    if req.tool not in LIVE_REQUEST_TOOLS:
        return None, None
    if not req.evidence or _norm(req.evidence) not in _norm(utter):
        return None, None
    res = db_tools.call_tool(req.tool, req.args, cd, scope="agent", dry_run=True)  # validates + describes, writes nothing
    if not res.get("ok"):
        return None, f'parent asked: "{req.evidence}" but it could not be set ({res.get("error")})'
    q = f"Just to check: a reminder on {res['summary']}, about {res['message']}. Shall I set that?"
    return confirm_tool_step(req.tool, req.args, q, res["summary"], res["message"]), None


def build_graph(interpreter: Interpreter, speak_now=None):
    """`speak_now` is a function returning the live 'say this right now' callback (or None). The phone bridge supplies one so fillers
    are spoken while a tool is still running; text sessions have none and the filler simply leads the reply."""
    max_retries = get_settings().max_unclear_retries
    max_nudges = 3

    def ctxobj(state) -> CallContext:
        return CallContext.model_validate(state["ctx"])

    # ---------------- nodes ----------------
    @tracing.traced("node:open")
    def open_(state: CallState):
        ctx = ctxobj(state)
        steps = build_steps(ctx)
        learned, pending = {}, None
        r = ctx.resume
        if r and r.get("steps") and r.get("idx", 0) < len(r["steps"]) and r["idx"] > 0:
            # the previous call stopped part-way: keep what was already answered and carry on from the unanswered step
            steps, learned, pending = r["steps"][r["idx"]:], r.get("learned", {}), r.get("pending_meal")
        tools.record_event(ctx.call_id, "call.opened", {"routine": ctx.routine_key, "steps": [s["id"] for s in steps],
                                                         "resumed": bool(r), "attempt_no": ctx.attempt_no, "kind": ctx.call_kind})
        return {"steps": steps, "idx": 0, "retries": 0, "started": True, "done": False, "unresolved": [],
                "pending_meal": pending, "nudges": 0, "alerted": [], "mild": None, "learned": learned, "rate": 1.0,
                "last_utter": "", "strikes": 0, "guard_relayed": [], "guard_counts": {}, "offered_message": False, "guard": None,
                "guard_done": False, "guard_lead": [], "reply": [greeting(ctx), steps[0]["prompt"]]}

    @tracing.traced("node:safety")
    def safety(state: CallState):
        utter = state.get("utterance", "")
        ctx = ctxobj(state)
        step = state["steps"][state["idx"]] if state.get("steps") and state.get("idx", 0) < len(state["steps"]) else {}
        hit = None
        if step.get("handler") != "relay_optional":  # that step is already collecting a message for the family
            hit = guardrails.detect(utter, ctx.caregiver_name, guardrails.spoken_date(ctx.timezone), repeats=0)
            if hit and state.get("guard_counts", {}).get(hit.kind):  # same kind again: different wording
                hit = guardrails.detect(utter, ctx.caregiver_name, guardrails.spoken_date(ctx.timezone), repeats=1)
        return {"emergency": detect_emergency(utter), "mild": detect_mild_symptom(utter), "early": [],
                "guard": hit.__dict__ if hit else None, "guard_done": False, "guard_lead": [],
                "interp": None, "decision": None}

    @tracing.traced("node:guardrail")
    def guardrail(state: CallState):
        """Out-of-scope request (story, news, task, OTP, injection...). Decline in code, tell the family when it matters,
        count strikes, and go back to the check-in. No model is asked unless the same reply also answers the question."""
        ctx = ctxobj(state)
        cd = ctx.model_dump()
        cg = ctx.caregiver_name
        hit = state["guard"]
        utter = state.get("utterance", "")
        step = state["steps"][state["idx"]]
        kind = hit["kind"]
        tools.record_event(ctx.call_id, "guardrail.hit", {"kind": kind, "strike": hit["strike"], "relay": hit["relay"], "utterance": utter})
        relayed = list(state.get("guard_relayed", []))
        if hit["relay"] and kind not in relayed:  # one report per kind per call: a bored parent should not spam the family
            tools.raise_alert(cd, hit["relay"], hit["relay_reason"], utter)
            relayed.append(kind)
        strikes = state.get("strikes", 0) + (1 if hit["strike"] else 0)
        req = InterpretRequest(step_kind=step["kind"], question=step["prompt"], utterance=utter, topic=step.get("topic", ""),
                               usecase=usecase_of(ctx))
        rule = RuleInterpreter().interpret(req)
        ev = (rule.evidence or "").lower().strip()
        # "Yes I had lunch. Also tell me a story" has the answer first; "tell me a story, I have nobody to talk to" does not
        up_front = bool(ev) and utter.lower().find(ev) != -1 and utter.lower().find(ev) <= 20
        long_message = len(utter.split()) > 14  # a long reply may hold the answer anywhere: let the interpreter read it
        answer_too = kind != "injection" and ((rule.intent in ("answer", "callback", "refusal") and up_front) or long_message)
        counts = {**state.get("guard_counts", {}), kind: state.get("guard_counts", {}).get(kind, 0) + 1}
        if answer_too and strikes < 3:  # "Yes I had lunch. Also, tell me a story": handle the answer AND decline the story
            return {"guard_lead": [hit["reply"]], "strikes": strikes, "guard_relayed": relayed, "guard_counts": counts}
        if strikes >= 3:
            tools.raise_alert(cd, "info", "parent kept asking for things outside the check-in; the call was ended politely", utter)
            return {"reply": [hit["reply"], f"I do not want to keep you. I will let {cg} know and try again later.", closing(ctx)],
                    "done": True, "end_reason": "off_topic_limit", "guard_done": True, "strikes": strikes, "guard_relayed": relayed,
                    "guard_counts": counts}
        back = guardrails.redirect_prefix(counts[kind] - 1) if not hit["allowed"] else ""
        return {"reply": [hit["reply"], back + _short(step["prompt"])], "guard_done": True, "strikes": strikes,
                "guard_relayed": relayed, "guard_counts": counts}

    @tracing.traced("node:escalate")
    def escalate(state: CallState):
        ctx = ctxobj(state)
        utter = state.get("utterance", "")
        interp = state.get("interp") or {}
        reason = state.get("emergency") or interp.get("evidence") or "possible emergency"
        tools.raise_alert(ctx.model_dump(), "emergency", f"possible emergency: {reason}", utter)
        return {"reply": [EMERGENCY_REPLY.format(cg=ctx.caregiver_name)], "done": True, "end_reason": "emergency"}

    @tracing.traced("node:interpret")
    def interpret(state: CallState):
        step = state["steps"][state["idx"]]
        ctx = ctxobj(state)
        req = InterpretRequest(step_kind=step["kind"], question=step["prompt"], utterance=state.get("utterance", ""),
                               language=ctx.language, topic=step.get("topic", ""), usecase=usecase_of(ctx),
                               now_local=datetime.now(ZoneInfo(ctx.timezone)).strftime("%A %Y-%m-%d %H:%M"))
        if step["handler"] == "confirm_tool":  # a plain yes/no to our read-back: do not let a model reinterpret it as a new request
            rule = RuleInterpreter().interpret(req)
            interp = rule if (rule.intent == "answer" and rule.yes_no is not None) else interpreter.interpret(req)
            if interp.intent == "request":
                interp = interp.model_copy(update={"intent": "unclear", "requests": [], "reason": "request again instead of yes/no"})
        else:
            interp = interpreter.interpret(req)
        tools.record_event(state["ctx"]["call_id"], "interpretation", {
            "step": step["id"], "utterance": state.get("utterance", ""), "interpreter": getattr(interpreter, "name", "?"),
            "intent": interp.intent, "yes_no": interp.yes_no, "items": interp.items, "time": interp.time,
            "evidence": interp.evidence, "confidence": interp.confidence, "reason": interp.reason,
            **getattr(interpreter, "last_diag", {})})
        return {"interp": interp.model_dump()}

    @tracing.traced("node:validate")
    def validate(state: CallState):
        step = state["steps"][state["idx"]]
        interp = Interpretation.model_validate(state["interp"])
        checked, why = validate_interpretation(interp, step, state.get("utterance", ""))
        if why:
            tools.record_event(state["ctx"]["call_id"], "validation.downgrade",
                               {"step": step["id"], "reason": why, "was": interp.model_dump()})
        return {"interp": checked.model_dump()}

    @tracing.traced("node:act")
    def act(state: CallState):
        ctx = ctxobj(state)
        cd = ctx.model_dump()
        cg = ctx.caregiver_name
        step = state["steps"][state["idx"]]
        interp = Interpretation.model_validate(state["interp"])
        utter = state.get("utterance", "")
        steps = list(state["steps"])
        upd: dict = {"last_utter": utter}
        dec = {"advance": False, "end": False, "say": [], "end_reason": ""}
        alerted = list(state.get("alerted", []))
        turn_alerts: list[tuple[str, str]] = []  # merged into a single alert at the end of the turn

        def add_alert(severity: str, reason: str, key: str | None = None) -> bool:
            if key is not None:
                if key in alerted:
                    return False
                alerted.append(key)
            turn_alerts.append((severity, reason))
            return True

        # --- side information that rides along with any reply (symptoms, concerns) ---
        sympathy = False
        sym = state.get("mild") or interp.mild_symptom
        if sym and add_alert("warning", f"parent mentioned: {sym}", key=sym.lower()):
            sympathy = True
        ack_logistics = False
        if interp.concern:
            sev = _concern_severity(interp.concern)
            if add_alert(sev, f"parent concern: {interp.concern}", key=_concern_key(interp.concern)):
                sympathy = sympathy or sev == "warning"
                ack_logistics = sev == "info"
        lead = ([f"I am sorry to hear that. I will let {cg} know."] if sympathy else
                [f"Understood. I will let {cg} know so it can be sorted out."] if ack_logistics and not state.get("guard_lead") else [])

        guard_lead = list(state.get("guard_lead", []))
        fillers: list[str] = []

        def filler(kind: str):
            """Stall like a person while a tool runs. Spoken live when the phone bridge allows, and always kept in the transcript."""
            n = state.get("filler_n", 0)
            line = FILLERS[kind][n % len(FILLERS[kind])]
            upd["filler_n"] = n + 1
            fillers.append(line)
            cb = speak_now() if speak_now else None
            if cb:
                try:
                    cb(line)
                    upd["early"] = list(state.get("early", [])) + [line]
                except Exception:  # noqa: BLE001 - a failed filler must not break the turn
                    pass

        def book(minutes=None, at=None, reason=""):
            """Book the call-back for real, NOW, and return what the tool says. What we promise aloud comes from this result."""
            filler("callback")
            args = {k: v for k, v in (("in_minutes", minutes), ("at", at), ("reason", reason)) if v is not None}
            res = db_tools.call_tool("book_callback", args, cd, scope="agent")
            if not res.get("ok"):
                add_alert("info", f"a call back could not be booked ({res.get('error')})", key="cbfail:" + str(res.get("error")))
            return res

        def asked_for(default_minutes: int):
            """(minutes, clock) the parent actually asked for in this reply, else our default. Never override what they said."""
            at = interp.callback_at if interp.callback_at and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", interp.callback_at) else None
            return (None, at) if at else (interp.callback_minutes or default_minutes, None)

        confirm, req_problem = None, None
        if interp.requests and not guard_lead and step["handler"] not in ("confirm_tool", "relay_optional") and state.get("requests_made", 0) < MAX_REQUESTS_PER_CALL:
            confirm, req_problem = plan_request(interp, utter, cd)
            if req_problem:
                add_alert("info", req_problem)

        def unclear(prefix: str):
            if guard_lead:  # we already spoke (the decline); do not also say "Sorry, I did not catch that"
                prefix = ""
            retries = state.get("retries", 0) + 1
            repeated = bool(utter.strip()) and _norm(utter) == _norm(state.get("last_utter", ""))
            if repeated and retries >= 2:
                retries = max_retries + 1  # same words twice: another identical re-ask would not help
            if retries <= max_retries:
                upd["retries"] = retries
                if step["handler"] == "medicine_taken" and re.search(r"hedged:which|which (tablet|one|medicine)|identify|white|appearance|unsure which|cannot tell", interp.reason, re.I):
                    d = step["data"]
                    dec["say"] = [f"The tablet I mean is {d['name']} {d.get('dose_note', '')}. "
                                  "Please check the name on the strip. Have you taken that one?".replace("  ", " ")]
                else:
                    dec["say"] = [f"{prefix} {_short(step['prompt'])}".strip()]
                return
            # give up on this step: record 'unclear' rather than guessing, and hand the family the raw words
            h = step["handler"]
            pend = state.get("pending_meal")
            heard = f'parent said: "{utter}"' if utter.strip() else "no response"
            if h == "meal_ate":
                tools.log_meal(cd, step["data"]["meal_type"], "unclear", [], utter, 0.0)
                add_alert("info", f"could not confirm {step['data']['meal_type']} ({heard})")
            elif h == "meal_items" and pend:
                tools.log_meal(cd, pend["meal_type"], "eaten", [], pend["evidence"], pend["confidence"])
                upd["pending_meal"] = None
            elif h == "medicine_taken":
                tools.log_medicine(cd, step["data"]["medicine_id"], "unclear", utter, 0.0)
                add_alert("warning", f"could not confirm {step['topic']} was taken ({heard})")
            elif h == "reminder_ack":
                tools.ack_reminder(cd, step["data"]["reminder_id"], False, utter)
                add_alert("warning", f"reminder not acknowledged ({heard})")
            elif h == "set_meal_time":
                meal = step["data"]["meal_type"]
                default = DEFAULT_MEAL_TIMES.get(meal)
                if default:  # no time from the parent: fall back to the standard time, flagged as a default so it can be corrected
                    tools.set_meal_time(cd, meal, default, utter, source="default")
                    upd["learned"] = {**state.get("learned", {}), meal: default}
                    add_alert("info", f"no usual {meal} time given ({heard}); using the default {default}")
                    dec.update(advance=True, say=[f"That is alright, I will note {spoken_time(default)} for {meal} for now."])
                    upd["unresolved"] = state.get("unresolved", []) + [step["id"]]
                    upd["retries"] = 0
                    return
                add_alert("info", f"could not learn usual {meal} time ({heard})")
            upd["unresolved"] = state.get("unresolved", []) + [step["id"]]
            dec.update(advance=True, say=["That is alright, we can leave it for now."])
            upd["retries"] = 0

        def nudge(say: list[str], slow: bool = False):
            """repeat / wait / identity: stay on the step without burning a retry, but cap the loop."""
            n = state.get("nudges", 0) + 1
            upd["nudges"] = n
            if slow:
                upd["rate"] = SLOW_RATE
            if n > max_nudges:
                return unclear("Let me try once more.")
            dec["say"] = say

        i = interp.intent
        if guard_lead and i not in ("answer", "callback", "refusal"):
            # a guardrail already declined this off-topic request. The interpreter may only contribute a real ANSWER: it must not
            # start an unrelated flow (e.g. read "who should I vote for" as a medical question and end the call)
            dec["say"] = [_short(step["prompt"])]
        elif i == "request":
            if not confirm:  # they asked for something we could not set up (unclear time, past, not allowed): say so, do not pretend
                if not req_problem:
                    add_alert("info", f'parent asked for something I could not set up: "{utter[:120]}"', key="req:" + utter.lower()[:60])
                unclear(f"I am sorry, I could not set that up, so I will let {cg} know.")
        elif i == "callback":
            at = interp.callback_at if interp.callback_at and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", interp.callback_at) else None
            res = book(None if at else (interp.callback_minutes or 10), at, f"parent asked for a callback during {step['id']}")
            if res.get("ok"):
                dec.update(end=True, end_reason="callback", say=[f"Sure, I will call you again {when_phrase(res)}."])
            else:  # never promise what the system refused: say so, tell the family, carry on with the check-in
                dec["say"] = [f"I am sorry, I cannot arrange that call myself, so I will let {cg} know.", _short(step["prompt"])]
        elif i == "wait":
            nudge(["Of course, take your time."])
        elif i == "repeat" and step["handler"] == "reminder_ack" and state.get("nudges", 0) >= 1:
            # we already re-read the reminder once and they still want something it does not contain (exact time...)
            tools.ack_reminder(cd, step["data"]["reminder_id"], True, utter)
            add_alert("info", f'parent wants more detail than the reminder holds ("{utter}"): please confirm the details')
            upd["unresolved"] = state.get("unresolved", []) + [step["id"]]
            dec.update(advance=True, say=[f"I only have what I read out. I will ask {cg} to give you the exact details."])
        elif i == "repeat":
            nudge(["Of course.", step["prompt"]], slow=True)
        elif i == "identity":
            nudge([f"I am an automated assistant from CareCall, calling for your family member {cg} just to check on your "
                   f"{step.get('topic') or 'day'}. I will never ask for money or bank details.", _short(step["prompt"])])
        elif i == "medical_question":
            extra = bool(re.search(r"another|more|again|extra|second|two|double", utter.lower()))
            add_alert("warning", ("possible extra dose, " if extra and step["handler"] == "medicine_taken" else "")
                      + f'parent asked a medical question: "{utter}"', key="medq:" + utter.lower())
            if step["handler"] == "medicine_taken":  # we cannot confirm the dose, so record it as unconfirmed
                say = ([f"That is a good question for your doctor or pharmacist. Please check with them before taking "
                        f"any more. I will let {cg} know."] if extra else
                       [f"That is a good question for your doctor or pharmacist. Please check with them first. "
                        f"I will let {cg} know."])
                tools.log_medicine(cd, step["data"]["medicine_id"], "unclear", utter, interp.confidence)
                upd["unresolved"] = state.get("unresolved", []) + [step["id"]]
                dec.update(advance=True, say=say)
            else:
                dec["say"] = [f"That is a good question for your doctor. I will let {cg} know.", _short(step["prompt"])]
        elif i == "unsure" and step["handler"] in ("meal_ate", "medicine_taken"):
            # "I do not remember": record it as unconfirmed, tell the family, never push a forgetful person to guess
            if step["handler"] == "meal_ate":
                meal = step["data"]["meal_type"]
                tools.log_meal(cd, meal, "unclear", [], utter, interp.confidence)
                add_alert("info", f"parent unsure whether they ate {meal}", key="unsure:" + step["id"])
                upd["unresolved"] = state.get("unresolved", []) + [step["id"]]
                dec.update(advance=True, say=[f"That is alright. I will let {cg} know."])
            else:
                tools.log_medicine(cd, step["data"]["medicine_id"], "unclear", utter, interp.confidence)
                res = book(10, None, f"{step['topic']} unconfirmed, parent unsure")
                add_alert("warning", f"parent unsure whether {step['topic']} was taken", key="unsure:" + step["id"])
                if res.get("ok"):
                    dec.update(end=True, end_reason="callback",
                               say=[f"That is alright. Please check your pill box, and I will call you again {when_phrase(res)}."])
                else:
                    dec.update(end=True, end_reason="completed", say=[f"That is alright. Please check your pill box. I will let {cg} know."])
        elif i == "unsure":
            unclear("Sorry, I did not catch that.")
        elif i == "refusal" and step["handler"] == "confirm_tool":  # "no, don't set it" is an answer to our read-back, not a refusal to talk
            dec.update(advance=True, say=["Okay, I will not set that."])
        elif i == "refusal":
            if step["handler"] == "medicine_taken":
                tools.log_medicine(cd, step["data"]["medicine_id"], "not_taken", utter, interp.confidence)
                add_alert("warning", f"parent declined {step['topic']}")
            upd["unresolved"] = state.get("unresolved", []) + [step["id"]]
            dec.update(advance=True, say=["Okay, I understand."] if lead else [f"Okay, I will let {cg} know."])
        elif i == "off_topic" and not guard_lead:
            # the model judged it out of scope although no deterministic guard matched: decline, count a strike, go back
            strikes = state.get("strikes", 0) + 1
            upd["strikes"] = strikes
            if strikes >= 3:
                add_alert("info", "parent kept asking for things outside the check-in; the call was ended politely")
                dec.update(end=True, end_reason="off_topic_limit",
                           say=["I am sorry, I cannot help with that.", f"I do not want to keep you. I will let {cg} know and try again later."])
            elif strikes == 1:
                dec["say"] = ["I am sorry, I am not able to help with that here.", _short(step["prompt"])]
            else:  # asked again: be warm, and tell the family once (a parent who keeps reaching out may want company)
                if "offtopic" not in state.get("guard_relayed", []):
                    add_alert("info", f'parent keeps asking for something outside the check-in: "{utter[:120]}"')
                    upd["guard_relayed"] = state.get("guard_relayed", []) + ["offtopic"]
                dec["say"] = [f"I am sorry, I cannot help with that, but I will let {cg} know you asked.", _short(step["prompt"])]
        elif i in ("unclear", "off_topic"):
            silent = not utter.strip()
            unclear("Are you still there?" if silent else "Sorry, I did not catch that.")
        else:  # answer
            h, d = step["handler"], step["data"]
            if h == "set_meal_time":
                tools.set_meal_time(cd, d["meal_type"], interp.time, interp.evidence)
                upd["learned"] = {**state.get("learned", {}), d["meal_type"]: interp.time}
                dec.update(advance=True, say=[f"Got it, {d['meal_type']} around {spoken_time(interp.time)}."])
            elif h == "confirm_times":
                if interp.yes_no:
                    dec.update(advance=True, say=["Wonderful, thank you."])
                else:
                    add_alert("warning", f"parent said the meal times read back are wrong: {state.get('learned', {})}")
                    upd["unresolved"] = state.get("unresolved", []) + [step["id"]]
                    dec.update(advance=True, say=[f"Okay, I will let {cg} know so we can correct them."])
            elif h == "meal_ate":
                meal = d["meal_type"]
                if interp.yes_no and interp.items:  # "yes, rice and dal": no need to ask again
                    tools.log_meal(cd, meal, "eaten", interp.items, utter, interp.confidence)
                    dec.update(advance=True, say=["Thank you, I have noted that."])
                elif interp.yes_no:
                    upd["pending_meal"] = {"meal_type": meal, "evidence": utter, "confidence": interp.confidence}
                    steps.insert(state["idx"] + 1, followup_items_step(meal))
                    upd["steps"] = steps
                    dec.update(advance=True, say=[] if lead else ["Good to hear."])
                elif interp.skipped:
                    tools.log_meal(cd, meal, "skipped", [], utter, interp.confidence)
                    dec.update(advance=True, say=["Okay, I have noted that."])
                else:
                    tools.log_meal(cd, meal, "not_yet", [], utter, interp.confidence)
                    res = book(*asked_for(30)[:1], asked_for(30)[1], f"{meal} not yet eaten")
                    if res.get("ok"):
                        dec.update(end=True, end_reason="callback",
                                   say=[f"Alright. I will check in again {when_phrase(res)} about your {meal}."])
                    else:
                        dec.update(end=True, end_reason="completed", say=[f"Alright. I will let {cg} know."])
            elif h == "meal_items":
                pend = state.get("pending_meal") or {"meal_type": d["meal_type"]}
                tools.log_meal(cd, pend["meal_type"], "eaten", interp.items, utter, interp.confidence)
                upd["pending_meal"] = None
                dec.update(advance=True, say=["Thank you, I have noted that."])
            elif h == "medicine_taken":
                if interp.yes_no:
                    tools.log_medicine(cd, d["medicine_id"], "taken", utter, interp.confidence)
                    dec.update(advance=True, say=["Thank you, that is noted."])
                else:
                    tools.log_medicine(cd, d["medicine_id"], "not_taken", utter, interp.confidence)
                    res = book(*asked_for(10)[:1], asked_for(10)[1], f"{step['topic']} not yet taken")
                    if res.get("ok"):
                        dec.update(end=True, end_reason="callback", say=[f"No problem. I will remind you again {when_phrase(res)}."])
                    else:
                        add_alert("warning", f"{step['topic']} not taken and a reminder call could not be booked")
                        dec.update(end=True, end_reason="completed", say=[f"No problem. I will let {cg} know."])
            elif h == "confirm_tool":
                if interp.yes_no:
                    filler("reminder")
                    res = db_tools.call_tool(d["tool"], d["args"], cd, scope="agent")
                    if res.get("ok"):
                        dec.update(advance=True, say=[f"Done. I have set that reminder for {d['summary']}."])
                    else:
                        add_alert("info", f"reminder could not be set after the parent confirmed ({res.get('error')}): {d['message']}")
                        dec.update(advance=True, say=[f"I am sorry, I could not set that one. I will let {cg} know."])
                else:
                    dec.update(advance=True, say=["Okay, I will not set that."])
            elif h == "relay_optional":
                if interp.yes_no:
                    tools.relay_message(cd, utter)
                    dec.update(advance=True, say=[f"I will pass that on to {cg}."])
                else:
                    dec.update(advance=True, say=["That is fine."])
            elif h == "reminder_ack":
                if interp.yes_no:
                    tools.ack_reminder(cd, d["reminder_id"], True, utter)
                    dec.update(advance=True, say=["Thank you."])
                else:
                    retries = state.get("retries", 0) + 1
                    if retries <= max_retries:
                        upd["retries"] = retries
                        dec["say"] = [f"Let me say it again. {d['message']} Please say okay once you have it."]
                    else:
                        unclear("")
            else:
                raise ValueError(f"unknown handler {h}")

        if i == "answer" and not dec["end"] and (interp.callback_at or interp.callback_minutes):
            at = interp.callback_at if interp.callback_at and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", interp.callback_at) else None
            res = book(None if at else interp.callback_minutes, at, f"parent asked for a callback while answering {step['id']}")
            if res.get("ok"):
                dec.update(end=True, end_reason="callback")
                dec["say"] = list(dec["say"]) + [f"Sure, I will call you again {when_phrase(res)}."]
            else:
                dec["say"] = list(dec["say"]) + [f"I am sorry, I cannot arrange that call myself, so I will let {cg} know."]
        if confirm:
            if dec["end"]:  # the call is ending anyway (e.g. a call back was booked): do not leave the request half-handled
                add_alert("info", f'parent also asked for a reminder that was not set because the call ended: {confirm["data"]["message"]}')
            else:
                steps.insert(state["idx"] + (1 if dec["advance"] else 0), confirm)  # read it back right after (or instead of) the re-ask
                upd["steps"] = steps
                upd["requests_made"] = state.get("requests_made", 0) + 1
                if not dec["advance"]:
                    dec["say"] = [confirm["prompt"]]
                    upd["retries"] = state.get("retries", 0)
        if turn_alerts:  # one alert per turn, however many reasons: avoids notification fatigue
            sev = max((s for s, _ in turn_alerts), key=SEVERITY_ORDER.get)
            tools.raise_alert(cd, sev, "; ".join(r for _, r in turn_alerts), utter)
        upd["alerted"] = alerted
        dec["say"] = guard_lead + lead + fillers + dec["say"]
        if dec["advance"]:
            upd["retries"] = 0
            upd["nudges"] = 0
        upd["decision"] = dec
        return upd

    @tracing.traced("node:advance")
    def advance(state: CallState):
        ctx = ctxobj(state)
        dec = state["decision"]
        reply = list(dec["say"])
        idx = state["idx"]
        if dec["end"]:
            reply.append(closing(ctx))
            return {"reply": reply, "done": True, "end_reason": dec["end_reason"]}
        if dec["advance"]:
            idx += 1
            steps = list(state["steps"])
            if idx >= len(steps):
                lonely = state.get("alerted") or state.get("strikes", 0) >= 1 or state.get("guard_relayed")
                if ctx.routine_type in ("meal_checkin", "meal_discovery") and lonely and not state.get("offered_message"):
                    steps.append(message_offer_step(ctx.caregiver_name))  # initiate: give the parent a bounded way to talk to family
                    reply.append(steps[idx]["prompt"])
                    return {"reply": reply, "idx": idx, "steps": steps, "offered_message": True}
                reply.append(closing(ctx))
                return {"reply": reply, "idx": idx, "done": True, "end_reason": "completed"}
            if steps[idx]["id"] == "confirm":  # read back what we learned; the prompt depends on the answers
                learned = state.get("learned", {})
                if not learned:  # nothing learned, nothing to confirm
                    reply.append(closing(ctx))
                    return {"reply": reply, "idx": len(steps), "done": True, "end_reason": "completed"}
                said = ", ".join(f"{m} around {spoken_time(t)}" for m, t in learned.items())
                steps[idx] = {**steps[idx], "prompt": f"Let me check I have this right: {said}. Is that correct?"}
                reply.append(steps[idx]["prompt"])
                return {"reply": reply, "idx": idx, "steps": steps}
            reply.append(steps[idx]["prompt"])
            return {"reply": reply, "idx": idx}
        return {"reply": reply}  # re-ask, stay on the same step

    # ---------------- routing ----------------
    def route_start(state):
        return "safety" if state.get("started") else "open"

    def after_safety(state):
        if state.get("emergency"):
            return "escalate"
        return "guardrail" if state.get("guard") else "interpret"

    def after_guardrail(state):
        return "end" if state.get("guard_done") else "interpret"

    def after_validate(state):
        return "escalate" if (state.get("interp") or {}).get("intent") == "emergency" else "act"

    g = StateGraph(CallState)
    for name, fn in [("open", open_), ("safety", safety), ("guardrail", guardrail), ("escalate", escalate), ("interpret", interpret),
                     ("validate", validate), ("act", act), ("advance", advance)]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, route_start, {"open": "open", "safety": "safety"})
    g.add_edge("open", END)
    g.add_conditional_edges("safety", after_safety, {"escalate": "escalate", "guardrail": "guardrail", "interpret": "interpret"})
    g.add_conditional_edges("guardrail", after_guardrail, {"end": END, "interpret": "interpret"})
    g.add_edge("interpret", "validate")
    g.add_conditional_edges("validate", after_validate, {"escalate": "escalate", "act": "act"})
    g.add_edge("act", "advance")
    g.add_edge("advance", END)
    g.add_edge("escalate", END)
    # MemorySaver: a call lives and dies with one server process (a dropped call can't be resumed anyway).
    return g.compile(checkpointer=MemorySaver())
