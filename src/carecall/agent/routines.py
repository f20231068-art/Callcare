"""Routine -> ordered steps. The steps (and their wording) are code, not model output.

Scripted wording is a deliberate choice: reminders must be exact and consistent. The model's job
is only to interpret the reply (see interpret.py)."""
from dataclasses import asdict, dataclass, field

MEALS = ("breakfast", "lunch", "snack", "dinner")
DEFAULT_MEAL_TIMES = {"breakfast": "08:30", "lunch": "12:30", "dinner": "19:30"}  # used when the caretaker has set none
CARETAKER = "caregiver"  # MealProfile.source of a time the caretaker fixed. The ONLY kind the schedule obeys.


def effective_meal_times(session, parent_id: int) -> dict[str, str]:
    """Meal times the calls follow: the defaults, overridden only by times the CARETAKER set. What a parent says on a call (even a clear
    "I eat at two") never moves the schedule; it is kept as a note for the family. Parents can only ask to be reminded again or
    to be reminded about something else."""
    from sqlalchemy import select

    from ..models import MealProfile

    fixed = {p.meal_type: p.usual_time for p in session.scalars(
        select(MealProfile).where(MealProfile.parent_id == parent_id, MealProfile.source == CARETAKER))}
    return {**DEFAULT_MEAL_TIMES, **fixed}


@dataclass
class Step:
    id: str
    kind: str  # yes_no | items | time | ack
    prompt: str
    topic: str = ""
    handler: str = ""  # which tool-handler consumes the answer
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def spoken_time(hhmm: str) -> str:
    """'13:30' -> '1:30 in the afternoon'. Text goes to a TTS voice, so avoid 24h and leading zeros."""
    h, m = int(hhmm[:2]), int(hhmm[3:])
    period = ("in the morning" if h < 12 else "at noon" if (h == 12 and m == 0) else
              "in the afternoon" if h < 17 else "in the evening" if h < 20 else "at night")
    h12 = h % 12 or 12
    return f"{h12}{':%02d' % m if m else ''} {period}"


def greeting(ctx) -> str:
    name = ctx.parent_preferred_name
    if ctx.call_kind == "retry" and ctx.prev_outcome == "cut_off":
        return f"Hello {name}, this is the CareCall assistant again. Sorry, we got cut off."
    if ctx.call_kind == "callback":
        return f"Hello {name}, this is the CareCall assistant calling back, as arranged."
    if ctx.call_kind == "retry" and ctx.prev_outcome == "hung_up_on_purpose":
        return f"Hello {name}, this is the CareCall assistant. I hope this is a better time."
    return f"Hello {name}, this is the CareCall assistant, calling for {ctx.caregiver_name}."


def build_steps(ctx) -> list[dict]:
    rtype, cfg = ctx.routine_type, ctx.routine_config
    steps: list[Step] = []
    if rtype == "meal_discovery":
        for meal in MEALS:
            steps.append(Step(f"time_{meal}", "time", f"What time do you usually have {meal}?", meal,
                              "set_meal_time", {"meal_type": meal}))
        steps.append(Step("confirm", "yes_no", "Let me check I have this right. Is that correct?", "meal times",
                          "confirm_times", {}))  # prompt is rebuilt from the answers in the graph's advance node
    elif rtype == "meal_checkin":
        meal = cfg["meal_type"]
        steps.append(Step("ate", "yes_no", f"Have you had your {meal} today?", meal, "meal_ate", {"meal_type": meal}))
    elif rtype == "medicine_reminder":
        med = ctx.medicine
        label = f"{med['name']} {med.get('dose_note', '')}".strip()  # the dose is part of the reminder: "Metformin 500mg"
        if med["timing"] == "before_food":
            lead = (f"It is time for your {label}. Please take it now, about "
                    f"{ctx.medicine_lead_minutes} minutes before your {med['meal_anchor']}.")
        elif med["timing"] == "after_food":
            lead = f"It is time for your {label}. Please take it now, after your {med['meal_anchor']}."
        elif med["timing"] == "with_food":
            lead = f"It is time for your {label}. Please take it with your {med['meal_anchor']}."
        else:
            lead = f"It is time for your {label}."
        steps.append(Step("taken", "yes_no", f"{lead} Have you taken it?", med["name"],
                          "medicine_taken", {"medicine_id": med["id"], "name": med["name"], "dose_note": med.get("dose_note", "")}))
    elif rtype == "defined_reminder":
        msg = ctx.reminder["message"]
        steps.append(Step("ack", "ack", f"This is a gentle reminder. {msg} Is there anything else I can help you with? I can remind you again if you like.", "reminder",
                          "reminder_ack", {"reminder_id": ctx.reminder["id"], "message": msg}))
    else:
        raise ValueError(f"unknown routine type {rtype}")
    return [s.to_dict() for s in steps]


def message_offer_step(caregiver: str) -> dict:
    """Added at the end of a check-in when the parent seemed lonely or kept chatting: a bounded way to let them talk to family."""
    return Step("message", "message", f"Is there anything you would like me to tell {caregiver}?", "message",
                "relay_optional", {}).to_dict()


def followup_items_step(meal: str) -> dict:
    return Step("items", "items", "What did you have?", meal, "meal_items", {"meal_type": meal}).to_dict()


def confirm_tool_step(tool: str, args: dict, question: str, summary: str, message: str) -> dict:
    """Inserted after the parent asked for something to go on the calendar: we read it back and only act on a clear yes."""
    return Step("confirm_tool", "yes_no", question, "calendar", "confirm_tool",
                {"tool": tool, "args": args, "summary": summary, "message": message}).to_dict()


def closing(ctx) -> str:
    return f"Take care, {ctx.parent_preferred_name}. Goodbye."


def usecase_of(ctx) -> str:
    """Which prompt/persona family a call belongs to. Medicines split on timing relative to food."""
    if ctx.routine_type == "medicine_reminder":
        return "medicine_before_food" if ctx.medicine["timing"] == "before_food" else "medicine_after_food"
    return {"meal_discovery": "meal_discovery", "meal_checkin": "meal_checkin",
            "defined_reminder": "defined_reminder"}[ctx.routine_type]
