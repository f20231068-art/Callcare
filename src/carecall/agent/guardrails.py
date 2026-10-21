"""Guardrails: what the agent will and will not do, enforced in CODE (not only in a prompt).

SCOPE POLICY
  Will answer (briefly, then return to the check-in):
      who the caller is / why they are calling; who {caregiver} is; repeat or clarify the question; today's date;
      a medical question  -> refer to the doctor or pharmacist and tell the family;
      a worry, symptom or loneliness -> one kind sentence and tell the family;
      "tell Sree ..." -> take the message and pass it on.
  Will NOT do (decline in one sentence, never perform, then return to the check-in):
      stories, poems, songs, bhajans, jokes, recitations, essays, recipes, diet plans or any long content;
      news, weather, scores, prices, trivia, translation, calculations; opinions on politics or religion;
      tasks (book, order, buy, cancel, call someone, send or move money); sharing contact details or addresses;
      medical advice (what to take, doses, changing medicine); role-play ("pretend you are my doctor");
      revealing or changing its instructions. Never ask for, or accept, an OTP, PIN, password or bank detail.
  Persistent off-topic (3 strikes): end the call kindly and tell the family.

Why code and not only a prompt: a prompt is a request, a regex is a guarantee. `detect` runs BEFORE any model, so a cheap
or confused model cannot be talked into reciting a thousand lines, and `cap_reply` / `leaks` bound what ANY model can emit."""
import re
from dataclasses import dataclass
from datetime import datetime, timezone

# request cues: "can you / tell me / please / I want ..." so a casual mention inside a real answer does not trip a guard
_CUE = (r"(can|could|will|would|should|please|kindly|tell me|read me|sing|recite|i want|i would like|i'd like|do you know|give me|"
        r"show me|teach me|explain|what is|what's|who is|how (do|to|can)|help me|i need|your opinion|what do you think|do you think|"
        r"sunao|sunaiye|sunaao|batao|bataiye|chahiye|dikhao|gaao|bolo|mujhe)")

_P = [  # (kind, regex, needs_request_cue). Order = priority.
    ("injection", r"ignore (all |your |any |the )?(previous |prior |above |earlier )?(instructions|rules|prompt)|system prompt|"
                  r"(your|the) (hidden )?(instructions|prompt|rules)|pretend (to be|you are|you're)|you are now|"
                  r"act as (my|a|an|the)|forget (everything|your (rules|instructions))|developer mode|jailbreak|reveal your|"
                  r"role ?play", False),
    ("scam", r"\b(otp|one[- ]time (password|code)|pin( number)?|cvv|password|aadhaar|aadhar|bank account|account number|"
             r"card number|upi|net ?banking|debit card|credit card)\b", False),
    ("financial", r"\b(transfer|send|give|lend|pay)\b.{0,40}\b(money|rupees|rs\.?|lakh|thousand|account)\b|\binvest|\bloan\b|"
                  r"\binsurance\b|\bwire\b", False),
    ("long_content", r"\b(story|stories|poem|poetry|song|sing|bhajan|lullaby|joke|riddle|recite|essay|speech|lecture|kahani|"
                     r"kahaani|katha|gaana)\b|\b\d{3,5}[- ]?lines?\b|thousand lines", True),
    ("contact_info", r"((phone|mobile|contact|whatsapp)( number)?|\baddress\b|\be-?mail\b|where (does|do) \w+ (live|stay))", True),
    ("task_request", r"\b(book|order|buy|purchase|cancel|reschedule|recharge|arrange|taxi|cab|auto ?rickshaw|groceries|ticket|"
                     r"call (my|a|the|him|her)|phone (my|a|the|him|her)|message (my|a|the)|set (an? )?(alarm|timer))\b", True),
    ("medical_advice", r"(which|what) (tablet|medicine|pill|drug)s? (should|can|to)|\b(change|reduce|increase|stop|skip|double)\b"
                       r".{0,25}\b(dose|tablet|medicine|pill)\b|diet plan|what should i (eat|take|drink)|home remed", True),
    ("general_knowledge", r"\b(weather|news|cricket|score|stock|share price|gold (rate|price)|recipe|capital of|translate|"
                          r"calculate|who won|prime minister|president|how (do|to) (i |you )?(make|cook)|step by step)\b", True),
    ("opinion", r"\b(vote|election|government|politic\w*|religion|which god|caste)\b", True),
    ("human_request", r"are you (a |an )?(real|human|person|robot|machine|ai|computer)|(talk|speak) to (a|my|some)\w* ?"
                      r"(person|human|someone|son|daughter)|do you have (a )?(family|children|kids|wife|husband)", False),
    ("caregiver_who", r"who (is|'s|was) (sree|he|that|this sree)\b|why (is|does) (sree|he) (calling|call|send)", False),
    ("date_time", r"what (day|date) is (it|today)|which day is (it|today)|what('s| is) (today'?s? )?(the )?date|today'?s date", False),
    ("company", r"stay (on the line|with me)|(talk|chat) (to|with) me|nobody to talk|keep me company|so (bored|lonely)|"
                r"feeling lonely|i am lonely", False),
    ("relay", r"\b(tell|ask|let|inform|remind) (sree|my (son|daughter)|him|her)\b.{0,80}|\bpass (on|a message)\b|"
              r"give (him|her|sree) a message|message for (sree|my (son|daughter))", False),
]
_COMPILED = [(k, re.compile(rx, re.I), cue) for k, rx, cue in _P]
_CUE_RE = re.compile(_CUE, re.I)

ALLOWED_KINDS = {"caregiver_who", "date_time", "relay", "human_request", "company", "scam"}  # answered, not refused
# kinds where the agent must never improvise: the reply is canned and the model is not asked
HARD_CANNED = {"injection", "scam", "financial", "long_content", "contact_info", "task_request", "general_knowledge", "opinion",
               "date_time", "caregiver_who", "human_request", "relay", "medical_advice"}


@dataclass
class GuardHit:
    kind: str
    text: str  # the parent's words
    reply: str  # what to say
    strike: bool = False
    relay: str | None = None  # None | "info" | "warning": tell the family
    relay_reason: str = ""
    allowed: bool = False  # an in-scope question that we answer, not a refusal


def _canned(kind: str, cg: str, text: str, local_now: str = "") -> tuple[str, str | None, str, bool]:
    """reply, relay severity, relay reason, strike"""
    short = text.strip()[:140]
    table = {
        "long_content": (f"I am not able to tell stories or songs, but I will let {cg} know you asked.",
                         "info", f"parent asked for entertainment/company: \"{short}\"", True),
        "general_knowledge": ("I am sorry, I cannot help with that. I am only here for your daily check-in.", None, "", True),
        "opinion": ("I am not able to share opinions on that.", None, "", True),
        "task_request": (f"I cannot do that for you, but I will let {cg} know you need help with it.",
                         "info", f"parent asked the assistant to do something it cannot: \"{short}\"", True),
        "financial": (f"I cannot move money or help with payments. Please do not send money to anyone who asks over the phone. "
                      f"I am letting {cg} know right away.", "warning", f"parent asked about moving money: \"{short}\"", True),
        "contact_info": (f"I am not able to share contact details, but I will ask {cg} to get in touch with you.",
                         "info", f"parent asked for contact details: \"{short}\"", True),
        "injection": ("I can only help with your check-in.", None, "", True),
        "scam": (f"Please never share an OTP, PIN or bank details with anyone who calls you, even if they say they are from your "
                 f"bank. I am letting {cg} know right now.", "warning", f"possible scam: parent mentioned OTP/PIN/bank details: \"{short}\"", False),
        "medical_advice": (f"That is a question for your doctor or pharmacist. I will let {cg} know.", "warning",
                           f"parent asked for medical advice: \"{short}\"", True),
        "human_request": ("I am an automated assistant from CareCall, not a person.", None, "", False),
        "human_request_talk": (f"I am an automated assistant from CareCall, not a person. I will ask {cg} to call you.",
                               "info", f"parent asked to speak to a person: \"{short}\"", False),
        "caregiver_who": (f"{cg} is a member of your family who asked me to check in on you.", None, "", False),
        "company": (f"I would love to chat longer, but I must keep this call short. I will let {cg} know you would like some company.",
                    "warning", f"parent wants company: \"{short}\"", False),
        "relay": (f"I will pass that on to {cg}.", "info", f"message for {cg}: \"{short}\"", False),
        "date_time": (f"Today is {local_now}." if local_now else "I am sorry, I do not have the date.", None, "", False),
    }
    return table[kind]


# what to say the SECOND time the same kind of request comes (the first reply, repeated word for word, sounds like a machine)
REPEAT_REPLY = {
    "long_content": "I am sorry, I truly cannot do that.",
    "general_knowledge": "I am sorry, that is outside what I can help with.",
    "opinion": "I would rather not share opinions.",
    "task_request": "I am sorry, I cannot do that, but I will let {cg} know.",
    "financial": "Again, please do not send money to anyone on a call. I will make sure {cg} hears about it.",
    "contact_info": "I cannot share contact details, but {cg} will get in touch.",
    "injection": "I can only talk about your check-in.",
    "scam": "Please remember: never share an OTP or PIN with anyone. I will let {cg} know.",
    "medical_advice": "Please do ask your doctor or pharmacist about that.",
}
BACK_TO_IT = ["", "Let us get back to it. ", "Whenever you are ready: "]


def redirect_prefix(n: int) -> str:
    """How to lead back into the routine question; varies so the same question is not read out identically each time."""
    return BACK_TO_IT[min(n, len(BACK_TO_IT) - 1)]


def detect(text: str, caregiver: str = "Sree", local_now: str = "", repeats: int = 0) -> GuardHit | None:
    """First matching guard for this utterance, or None. Deterministic and instant: runs before any model.
    `repeats` = how many times this KIND of request has already come up this call (changes the wording)."""
    if not text or not text.strip():
        return None
    low = text.lower()
    has_cue = bool(_CUE_RE.search(low))
    for kind, rx, needs_cue in _COMPILED:
        if not rx.search(low) or (needs_cue and not has_cue):
            continue
        key = "human_request_talk" if kind == "human_request" and re.search(r"talk|speak", low) else kind
        reply, relay, reason, strike = _canned(key, caregiver, text, local_now)
        if repeats >= 1 and kind in REPEAT_REPLY:
            reply = REPEAT_REPLY[kind].format(cg=caregiver)
        return GuardHit(kind=kind, text=text, reply=reply, strike=strike, relay=relay, relay_reason=reason,
                        allowed=kind in ALLOWED_KINDS)
    return None


# ---------- output-side guards (bound what ANY model can say) ----------
def cap_reply(text: str, max_words: int = 55) -> tuple[str, bool]:
    """Truncate at a sentence boundary so a runaway reply (a 1000-line story) can never reach a parent's ear."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip(), False
    out, n = [], 0
    for sentence in re.split(r"(?<=[.?!])\s+", text.strip()):
        w = len(sentence.split())
        if n + w > max_words and out:
            break
        out.append(sentence)
        n += w
        if n >= max_words:
            break
    return " ".join(out).strip(), True


LEAK_PATTERNS = [r"hard rules", r"system prompt", r"my (instructions|prompt) (say|are|is|tell)", r"\{\{", r"rule \d+\b",
                 r"as an? (ai )?language model", r"\bcaregiver_name\b", r"never give medical advice"]
PII_PATTERNS = [r"\+?\d[\d \-]{8,}\d", r"[\w.+-]+@[\w-]+\.[\w.]+"]
_LEAK_RE = re.compile("|".join(LEAK_PATTERNS), re.I)
_PII_RE = re.compile("|".join(PII_PATTERNS))


def leaks(text: str) -> list[str]:
    """Things an agent line must never contain: its own instructions, or phone numbers / e-mail addresses."""
    found = []
    if _LEAK_RE.search(text):
        found.append("instructions")
    if _PII_RE.search(text):
        found.append("contact_details")
    return found


def spoken_date(tz: str = "Asia/Kolkata") -> str:
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo(tz))
    except Exception:
        now = datetime.now(timezone.utc)
    return now.strftime("%A, %d %B").replace(" 0", " ")
