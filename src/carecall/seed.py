"""Hardcoded Phase-0 data: one parent, one caregiver, two medicines, one reminder, and the routines.
No clock times: routines are triggered manually (CLI / API), not scheduled."""
import os

from sqlalchemy import select

from .agent.routines import DEFAULT_MEAL_TIMES, MEALS
from .db import session_scope
from .models import (
    Caregiver, MealProfile, Medicine, Parent, ParentCaregiver, Reminder, Routine,
)



def seed(parent_phone: str | None = None, caregiver_phone: str | None = None, meal_profile: bool = True,
         preferred_name: str = "Asha") -> dict:
    """Idempotent: running twice does not duplicate. Phones default to env SEED_PARENT_PHONE / SEED_CAREGIVER_PHONE."""
    parent_phone = parent_phone or os.getenv("SEED_PARENT_PHONE", "+910000000000")
    caregiver_phone = caregiver_phone or os.getenv("SEED_CAREGIVER_PHONE", "+910000000001")
    with session_scope() as s:
        if s.scalars(select(Parent)).first():
            return {"seeded": False}
        parent = Parent(name=f"{preferred_name} Rao", preferred_name=preferred_name, phone=parent_phone, language="en",
                        timezone="Asia/Kolkata")
        cg = Caregiver(name="Sree", phone=caregiver_phone, alert_mode="digest")
        s.add_all([parent, cg])
        s.flush()
        s.add(ParentCaregiver(parent_id=parent.id, caregiver_id=cg.id, relationship_="child"))
        for meal, t in (DEFAULT_MEAL_TIMES.items() if meal_profile else []):
            s.add(MealProfile(parent_id=parent.id, meal_type=meal, usual_time=t, source="default"))
        met = Medicine(parent_id=parent.id, name="Metformin", dose_note="500mg", timing="before_food",
                       meal_anchor="lunch", offset_minutes=0)
        aml = Medicine(parent_id=parent.id, name="Amlodipine", dose_note="5mg", timing="after_food",
                       meal_anchor="dinner", offset_minutes=30)
        rem = Reminder(parent_id=parent.id, message="You have an appointment with Doctor Rao tomorrow evening.")
        s.add_all([met, aml, rem])
        s.flush()

        def r(key, type_, config):
            s.add(Routine(parent_id=parent.id, key=key, type=type_, config=config))

        r("meal_discovery", "meal_discovery", {})
        for meal in MEALS:
            r(f"{meal}_checkin", "meal_checkin", {"meal_type": meal})
        r("metformin_reminder", "medicine_reminder", {"medicine_id": met.id})
        r("amlodipine_reminder", "medicine_reminder", {"medicine_id": aml.id})
        r("doctor_reminder", "defined_reminder", {"reminder_id": rem.id})
        return {"seeded": True, "parent_id": parent.id}
