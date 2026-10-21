"""The database tools the agent / outcome engine use to read and change the calendar and profile."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from carecall.agent import db_tools
from carecall.agent.db_tools import call_tool, openai_tool_specs
from carecall.calls import outcome
from carecall.calls.keys import occurrence_key
from carecall.db import session_scope
from carecall.models import CallAttempt, Event, MealProfile, Parent, Reminder, Routine

from test_lifecycle import NOW, OCC, followups, make_call, notes

CTX = {"parent_id": 1, "call_id": None}


def waiting():
    with session_scope() as s:
        return [(c.id, c.idempotency_key, c.kind) for c in s.scalars(
            select(CallAttempt).where(CallAttempt.status == "scheduled").order_by(CallAttempt.id))]


def test_every_tool_has_a_model_readable_spec_and_hidden_ones_stay_hidden():
    names = {t["function"]["name"] for t in openai_tool_specs("agent")}
    assert {"schedule_next_call", "get_upcoming_calls", "add_reminder", "cancel_call"} <= names
    assert "resume_calls" not in names and "pause_calls" not in names  # operator / engine only
    json.dumps(openai_tool_specs("admin"))  # serialisable as-is


def test_a_model_cannot_aim_a_tool_at_another_parent_or_use_a_forbidden_one():
    assert not call_tool("get_today_status", {"parent_id": 2}, CTX)["ok"]
    assert not call_tool("resume_calls", {}, CTX, scope="agent")["ok"]
    assert not call_tool("nope", {}, CTX)["ok"]
    assert not call_tool("schedule_next_call", {"routine_key": "lunch_checkin"}, CTX)["ok"]  # needs in_minutes or at
    assert not call_tool("schedule_next_call", {"routine_key": "lunch_checkin", "in_minutes": "5"}, CTX)["ok"]  # wrong type


def test_scheduling_a_call_is_idempotent_and_audited():
    a = call_tool("schedule_next_call", {"routine_key": "lunch_checkin", "at": "2026-10-25 14:00", "reason": "parent asked"}, CTX)
    b = call_tool("schedule_next_call", {"routine_key": "lunch_checkin", "at": "2026-10-25 14:00"}, CTX)
    assert a["ok"] and b["ok"] and a["call_id"] == b["call_id"] and b["already_existed"]
    with session_scope() as s:
        assert any(e.type == "db_tool.schedule_next_call" for e in s.scalars(select(Event)))


def test_reschedule_and_cancel_only_touch_this_parents_waiting_calls():
    cid = call_tool("schedule_next_call", {"routine_key": "lunch_checkin", "at": "2026-10-26 13:00"}, CTX)["call_id"]
    assert call_tool("reschedule_call", {"call_id": cid, "at": "2026-10-26 15:00"}, CTX)["ok"]
    assert call_tool("get_upcoming_calls", {}, CTX)["calls"][0]["call_id"] == cid
    assert call_tool("cancel_call", {"call_id": cid, "reason": "test"}, CTX)["ok"]
    assert not call_tool("cancel_call", {"call_id": cid}, CTX)["ok"]  # already cancelled
    assert not call_tool("cancel_call", {"call_id": cid}, {"parent_id": 999})["ok"]


def test_add_reminder_creates_the_reminder_its_routine_and_the_call_once():
    when = (datetime.now().astimezone() + timedelta(days=3)).strftime("%Y-%m-%d 16:30")
    a = call_tool("add_reminder", {"message": "Doctor appointment with Dr Rao", "at": when}, CTX)
    b = call_tool("add_reminder", {"message": "Doctor appointment with Dr Rao", "at": when}, CTX)
    assert a["ok"] and a["call_id"] and b["already_existed"]
    with session_scope() as s:
        assert len(s.scalars(select(Reminder).where(Reminder.message.like("Doctor%"))).all()) == 1
    assert not call_tool("add_reminder", {"message": "x", "at": "2020-01-01 10:00"}, CTX)["ok"]


def test_only_the_caretaker_side_can_set_a_meal_time_and_the_tool_validates_it():
    ADMIN = {**CTX, "call_id": None}
    assert not call_tool("update_meal_time", {"meal": "lunch", "time": "13:15"}, CTX, scope="agent")["ok"]  # the agent may not
    assert not call_tool("update_meal_time", {"meal": "lunch", "time": "13:15"}, CTX, scope="system")["ok"]  # nor the engine
    assert "update_meal_time" not in {t["function"]["name"] for t in openai_tool_specs("agent")}
    assert not call_tool("update_meal_time", {"meal": "brunch", "time": "10:00"}, ADMIN, scope="admin")["ok"]
    assert not call_tool("update_meal_time", {"meal": "lunch", "time": "25:99"}, ADMIN, scope="admin")["ok"]
    assert call_tool("update_meal_time", {"meal": "lunch", "time": "13:15"}, ADMIN, scope="admin")["ok"]
    with session_scope() as s:
        row = s.get(MealProfile, (1, "lunch"))
        assert (row.usual_time, row.source) == ("13:15", "caregiver")
    assert call_tool("get_parent_profile", {}, CTX)["meal_times"]["lunch"] == "13:15"


def test_what_a_parent_says_never_moves_a_meal_time_the_caretaker_fixed():
    """Parents can only ask for another reminder or a reminder about something else; the daily times stay as the caretaker set them."""
    from carecall.agent import tools as write_tools
    from carecall.agent.routines import effective_meal_times

    call_tool("update_meal_time", {"meal": "lunch", "time": "13:00"}, {**CTX, "call_id": None}, scope="admin")
    cid = make_call("completed", fin={"done": True})
    write_tools.set_meal_time({"parent_id": 1, "call_id": cid}, "lunch", "15:00", "I eat at three now")  # the parent's words, via the agent path
    with session_scope() as s:
        assert effective_meal_times(s, 1)["lunch"] == "13:00"
        assert s.get(MealProfile, (1, "lunch")).usual_time == "13:00"
        assert any(e.type == "tool.set_meal_time" and e.payload.get("ignored") for e in s.scalars(select(Event)))


def test_parent_stated_times_are_notes_not_schedule_so_defaults_apply_until_the_caretaker_sets_one():
    from carecall.agent import tools as write_tools
    from carecall.agent.routines import effective_meal_times

    cid = make_call("completed", fin={"done": True})
    write_tools.set_meal_time({"parent_id": 1, "call_id": cid}, "dinner", "21:00", "late dinner", source="stated")
    with session_scope() as s:
        assert effective_meal_times(s, 1)["dinner"] == "19:30"  # the default still rules


def test_pause_then_scheduling_is_refused_and_only_an_operator_can_resume():
    assert call_tool("pause_calls", {"reason": "test"}, CTX, scope="system")["ok"]
    assert not call_tool("schedule_next_call", {"routine_key": "lunch_checkin", "in_minutes": 5}, CTX)["ok"]
    assert call_tool("resume_calls", {}, CTX, scope="admin")["ok"]
    with session_scope() as s:
        assert s.get(Parent, 1).calling_enabled


def test_when_the_job_is_done_the_next_days_call_is_scheduled_and_stragglers_cancelled():
    """completed lunch call -> the outcome engine plans tomorrow's lunch check-in (key from tomorrow's date, attempt 1)."""
    cid = make_call("completed", turns=[("agent", "Have you eaten?"), ("parent", "yes, rice")], fin={"done": True})
    assert outcome.apply(cid, now=NOW).kind == "completed"
    tomorrow = occurrence_key(1, "lunch_checkin", "2026-10-19") + ":a1"
    assert tomorrow in [k for _, k, _ in waiting()]
    outcome.apply(cid, now=NOW)  # deciding again must not add another
    assert [k for _, k, _ in waiting()].count(tomorrow) == 1


def test_finish_occurrence_tool_cancels_waiting_retries():
    cid = make_call("completed", turns=[("agent", "Have you eaten?")], fin={"done": True})
    from carecall.calls.service import schedule_attempt
    with session_scope() as s:
        rid = s.scalars(select(Routine.id).where(Routine.key == "lunch_checkin")).first()
        schedule_attempt(s, parent_id=1, routine_id=rid, occurrence_key=OCC, attempt_no=2, kind="retry", when=NOW + timedelta(minutes=10))
    r = call_tool("finish_occurrence", {}, {"parent_id": 1, "call_id": cid}, scope="system")
    assert r["ok"] and r["cancelled_waiting"] == 1 and r["next_call_id"]


def test_langfuse_users_are_per_parent_and_evals_are_kept_apart():
    from carecall.agent.session import CallSession
    s = CallSession.__new__(CallSession)
    s.user_id, s.session_id, s.ctx = None, None, {"parent_id": 7, "parent_preferred_name": "Asha Devi"}
    assert s._user_id() == "parent-7-asha-devi"
    s.session_id = "run42:elderly_03"
    assert s._user_id() == "eval:elderly_03"
