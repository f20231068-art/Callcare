"""Interpretation: turn what the parent said into a structured, *evidence-backed* answer.

The model only INTERPRETS. It never decides what happens next and never writes to the DB; code does.
Two implementations share one interface:
  - LLMInterpreter: prompts a real model with a per-use-case prompt (see prompts/) and parses JSON.
  - RuleInterpreter: keyword rules for offline tests and as a baseline the real model must beat.
"""
import json
import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, model_validator

from ..llm.base import LLMClient, complete_json
from ..prompts import system_prompt

Intent = Literal["answer", "unclear", "callback", "emergency", "refusal", "off_topic",
                 "repeat", "wait", "identity", "medical_question", "unsure", "request"]


class InterpretRequest(BaseModel):
    step_kind: Literal["yes_no", "items", "time", "ack", "message"]
    question: str
    utterance: str
    language: str = "en"
    topic: str = ""  # human description, e.g. "lunch" / "Metformin"
    usecase: str = ""  # meal_discovery | meal_checkin | medicine_before_food | medicine_after_food | defined_reminder
    now_local: str = ""  # "Sunday 2026-10-18 11:30" in the parent's timezone: lets the model turn "Friday at 4" into a date


class ToolRequest(BaseModel):
    """Something the parent asked to be put on the calendar. The model only PROPOSES it; code checks it, confirms it aloud, then runs the tool."""
    tool: str
    args: dict = Field(default_factory=dict)
    evidence: str = ""


class Interpretation(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _drop_nulls(cls, data):
        """Models often emit null for fields that have a default ("reason": null). Treat null as 'not provided'."""
        return {k: v for k, v in data.items() if v is not None} if isinstance(data, dict) else data

    intent: Intent = "unclear"
    yes_no: bool | None = None
    skipped: bool = False  # parent says they are deliberately not eating this meal
    items: list[str] = Field(default_factory=list)
    time: str | None = None  # "HH:MM" 24h
    callback_minutes: int | None = None
    callback_at: str | None = None  # "HH:MM" parent-local, when they name a clock time
    requests: list[ToolRequest] = Field(default_factory=list)
    mild_symptom: str | None = None  # a minor complaint mentioned alongside an answer
    concern: str | None = None  # something the family should know (cannot attend, low appetite...)
    evidence: str = ""  # exact words from the parent that support the answer
    confidence: float = 0.0
    reason: str = ""  # why unclear / downgraded


class Interpreter(Protocol):
    name: str

    def interpret(self, req: InterpretRequest) -> Interpretation: ...


# ---------- symptom detection (deterministic, runs BEFORE any model) ----------
# Severe: end the call and alert the family now. Mild: alert the family, keep the call going.
SEVERE_PATTERNS = [
    r"chest (pain|hurts?|tight)", r"can'?t breathe", r"cannot breathe", r"short(ness)? of breath", r"\bbreathless\b",
    r"\bfell\b", r"\bfallen\b", r"\bfall(ing)? down\b", r"\bfaint(ed|ing)?\b", r"\bbleeding\b", r"unconscious",
    r"\bstroke\b", r"heart attack", r"\bhelp\s*!", r"need (an )?ambulance", r"severe pain", r"can'?t move",
]
MILD_PATTERNS = [
    r"\bdizz(y|iness)\b", r"\bgiddy\b", r"head ?ache", r"\bnausea\b", r"\bvomit(ing|ed)?\b", r"\bpain\b",
    r"\baching\b", r"\bache\b", r"\bweak(ness)?\b", r"\bconfused\b", r"feel(ing)? (sick|unwell|tired)",
]
_SEVERE_RE = re.compile("|".join(SEVERE_PATTERNS), re.I)
_MILD_RE = re.compile("|".join(MILD_PATTERNS), re.I)


_DISTRESS = re.compile(r"\b(fell|fall|fallen|chest|breath\w*|bleed\w*|faint\w*|collaps\w*|stroke|dying|pain|hurt\w*|emergency|"
                       r"ambulance|unconscious|stuck|fire|thief|robber|attack\w*|intruder)\b", re.I)
_HELP = re.compile(r"\bhelp(?:\s+me)?\b", re.I)
_HELP_STANDALONE_TAIL = {"", "please", "now", "quickly", "someone", "somebody", "anyone", "please someone", "me please"}


def _help_is_distress(utterance: str) -> str | None:
    """'Help me!' is a cry for help; 'ask Sree to help me get the dose lowered' / 'help me remember' are requests for assistance.
    Distress only if 'help (me)' ENDS its sentence (optionally + please/now/someone) or the sentence holds a distress word."""
    for m in _HELP.finditer(utterance):
        before = re.split(r"[.!?]", utterance[: m.start()])[-1]
        rest = re.split(r"[.!?]", utterance[m.end():], 1)[0]
        sentence = before + m.group(0) + rest
        if rest.strip(" ,").lower() in _HELP_STANDALONE_TAIL or _DISTRESS.search(sentence):
            return m.group(0)
    return None


def detect_emergency(utterance: str) -> str | None:
    m = _SEVERE_RE.search(utterance or "")
    return m.group(0) if m else _help_is_distress(utterance or "")


def detect_mild_symptom(utterance: str) -> str | None:
    m = _MILD_RE.search(utterance or "")
    return m.group(0) if m else None


# ---------- LLM interpreter ----------
class LLMInterpreter:
    def __init__(self, client: LLMClient):
        self.client = client
        self.name = f"llm:{client.model}"
        self.last_diag: dict = {}

    def interpret(self, req: InterpretRequest) -> Interpretation:
        from .db_tools import live_tool_prompt

        prompt, version = system_prompt(req.usecase)
        prompt = prompt + "\n\n" + live_tool_prompt()
        user = json.dumps(req.model_dump(), ensure_ascii=False)
        self.last_diag = {"prompt_version": version}
        try:
            data = complete_json(self.client, prompt, user, purpose=f"interpret:{req.usecase}:{version}")
            return Interpretation.model_validate(data)
        except (ValidationError, ValueError, KeyError) as e:
            raw = getattr(self.client, "last_raw", "")
            self.last_diag.update(error=f"invalid model output: {str(e)[:120]}", raw=raw[:300])
            return Interpretation(intent="unclear", reason=f"model output invalid: {str(e)[:80]}")
        except Exception as e:  # network / provider outage: a live call must not die, fall back to keyword rules
            self.last_diag.update(error=f"model call failed ({type(e).__name__}): {str(e)[:120]}", fallback="rules")
            fb = RuleInterpreter().interpret(req)
            return fb.model_copy(update={"reason": (fb.reason + " [llm_failed:rules_fallback]").strip()})


# ---------- rule-based interpreter (offline baseline) ----------
_NO = re.compile(
    r"\b(no|nope|not yet|haven'?t|have not|didn'?t|did not|not taken|not had|not eaten|forgot|skip(?:ped|ping)?)\b"
)
_YES = re.compile(r"\b(yes|yeah|yep|yup|ya|sure|done|took|taken|already|finished|had|ate|eaten)\b|\bi (have|did)\W*$")
_ACK = re.compile(r"\b(ok|okay|got it|fine|understood|noted|thanks|thank you|will do|alright)\b")
_WAIT = re.compile(r"\b(hold on|wait|one (?:minute|moment|sec(?:ond)?)|just a (?:minute|moment|sec(?:ond)?)|give me a (?:minute|moment))\b")
_IDENTITY = re.compile(r"\bwho (?:is this|are you|is it|am i talking)\b|why (?:are you|do you) (?:asking|calling)|which company")
_MEDQ = re.compile(r"\b(should i|shall i|can i|may i|is it (?:ok|okay|alright) (?:to|if)|what if|how many)\b|side effect")
_REFUSE = re.compile(r"don'?t want|do not want|won'?t take|will not take|refuse|not going to|no way|not taking")
_APPETITE = re.compile(r"don'?t feel like eating|no appetite|not hungry|not in the mood to eat|lost my appetite|don'?t feel like it")
_REPEAT = re.compile(r"\b(pardon|say (?:that )?again|repeat|come again|can'?t hear|couldn'?t hear|didn'?t hear|what did you say)\b"
                     r"|^\W*(what|eh|huh|ha|hmm\?)\W*$|^\W*(sorry|what)\W*\?|what was it again")
_DETAIL_Q = re.compile(r"\b(what time|which doctor|when is|where is|which day|what day|when\?)\b")
_CANT = re.compile(r"can'?t (?:go|come|make it|attend)|cannot (?:go|come|make it|attend)|won'?t be able|not able to (?:go|come)")
_UNSURE = re.compile(r"\b(maybe|not sure|i think|i guess|perhaps|probably|(?:don'?t|do not|can'?t|cannot) remember)\b")
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}
_HOURS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"


def _num(tok: str) -> int | None:
    return int(tok) if tok.isdigit() else _NUM.get(tok)


def parse_time(text: str, meal_hint: str = "") -> str | None:
    """Best-effort 'around 1:30', 'one thirty', 'half past eight', '8 am' -> 'HH:MM'. None if no time found."""
    t = text.lower()
    ampm = ""
    half = re.search(rf"half past ({_HOURS})", t)
    quarter_to = re.search(rf"quarter to ({_HOURS})", t)
    digits = re.search(r"\b(\d{1,2})(?!\d)(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", t)
    words = re.search(rf"\b({_HOURS})(?: (thirty|fifteen))?\b", t)
    if half:
        hour, minute = _NUM[half.group(1)], 30
    elif quarter_to:
        hour, minute = _NUM[quarter_to.group(1)] - 1, 45
    elif digits:
        hour, minute = int(digits.group(1)), int(digits.group(2) or 0)
        ampm = (digits.group(3) or "").replace(".", "")
    elif words:
        hour, minute = _NUM[words.group(1)], _NUM.get(words.group(2) or "", 0)
    elif "noon" in t:
        return "12:00"
    else:
        return None
    if hour > 24 or minute > 59:
        return None
    if not ampm and re.search(r"evening|night|afternoon", t):
        ampm = "pm"
    elif not ampm and "morning" in t:
        ampm = "am"
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif not ampm and hour < 12 and meal_hint in ("lunch", "snack", "dinner") and not (meal_hint == "lunch" and hour == 11):
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def parse_time_strict(text: str, meal_hint: str = "") -> str | None:
    """Like parse_time but ONLY for explicit clock phrases: '7:30', '8 am', 'four o'clock', 'half past seven',
    'quarter to nine', 'seven thirty'. A bare 'one' (as in 'one more time') is never a time."""
    t = text.lower()
    explicit = (re.search(r"\b\d{1,2}:\d{2}\b", t) or re.search(r"\b\d{1,2}\s*(am|pm|a\.m\.|p\.m\.)", t)
                or re.search(rf"\b({_HOURS}|\d{{1,2}})\s*o'?clock\b", t) or re.search(rf"\bhalf past ({_HOURS})\b", t)
                or re.search(rf"\bquarter (to|past) ({_HOURS})\b", t) or re.search(rf"\b({_HOURS}) (thirty|fifteen|forty.?five)\b", t)
                or re.search(r"\b(at|around|about)\s+\d{1,2}\b", t))
    return parse_time(text, meal_hint) if explicit else None


def _hit(intent: str, evidence: str, conf: float = 0.8, **kw) -> Interpretation:
    return Interpretation(intent=intent, evidence=evidence, confidence=conf, **kw)


class RuleInterpreter:
    name = "rules"

    def interpret(self, req: InterpretRequest) -> Interpretation:
        u = req.utterance.strip()
        low = u.lower()
        if not low:
            return Interpretation(intent="unclear", reason="silence")

        cb = re.search(r"(call|ring|remind).{0,20}(back|again|later)|call me (later|in)", low)
        if cb:
            mins = re.search(r"in (\w+) (min|minute|minutes|hour|hours)", low)
            minutes = None
            if mins and _num(mins.group(1)):
                minutes = _num(mins.group(1)) * (60 if mins.group(2).startswith("hour") else 1)
            return _hit("callback", u, 0.85, callback_minutes=minutes)
        for rx, intent in ((_WAIT, "wait"), (_IDENTITY, "identity"), (_MEDQ, "medical_question"), (_REFUSE, "refusal")):
            m = rx.search(low)
            if m:
                return _hit(intent, m.group(0), 0.85)
        if _REPEAT.search(low) or (req.step_kind == "ack" and _DETAIL_Q.search(low)):
            return _hit("repeat", (_REPEAT.search(low) or _DETAIL_Q.search(low)).group(0), 0.8)

        if req.step_kind in ("yes_no", "ack"):
            um = _UNSURE.search(low)
            if um and not _NO.search(low):
                return _hit("unsure", um.group(0), 0.8)
            m = _APPETITE.search(low)
            if m:
                return _hit("answer", m.group(0), 0.8, yes_no=False, skipped=True, concern="low appetite")
            cant = _CANT.search(low)
            if cant and req.step_kind == "ack":
                return _hit("answer", cant.group(0), 0.75, yes_no=True, concern="cannot attend")
            no = _NO.search(low)
            yes = _YES.search(low) or (_ACK.search(low) if req.step_kind == "ack" else None)
            if no:  # negations win: "I haven't taken it" contains "taken" but means no
                return _hit("answer", no.group(0), 0.8, yes_no=False, skipped="skip" in low)
            if yes:
                return _hit("answer", yes.group(0), 0.8, yes_no=True)
        elif req.step_kind == "message":  # "anything you'd like me to tell Sree?" -> nothing / a message
            nm = re.search(r"\b(no|nothing|nope|not really|that'?s all|no thanks|nahi|nahin)\b", low)
            if nm and len(low.split()) <= 6:
                return _hit("answer", nm.group(0), 0.8, yes_no=False)
            return _hit("answer", u, 0.7, yes_no=True)
        elif req.step_kind == "items":
            m = re.search(r"(?:had|ate|eaten|having|eating)\s+(.*)", low) or re.match(r"(.*)", low)
            chunk = m.group(1) if m else low
            chunk = re.sub(r"\b(some|a|an|the|my|for (?:breakfast|lunch|dinner)|today|just|only|little)\b", " ", chunk)
            items = [i.strip(" .") for i in re.split(r",|\band\b|\bwith\b", chunk) if i.strip(" .")]
            if items and len(items) <= 6 and not _NO.search(low):
                return _hit("answer", u, 0.7, items=items)
        elif req.step_kind == "time":
            t = parse_time(u, req.topic)
            if t:
                return _hit("answer", u, 0.75, time=t)
        return Interpretation(intent="unclear", reason="no clear answer found")


def make_interpreter(client: LLMClient | None) -> Interpreter:
    return LLMInterpreter(client) if client is not None else RuleInterpreter()
