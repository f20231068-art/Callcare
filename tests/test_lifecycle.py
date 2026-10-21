"""Call lifecycle: every way a call can end -> an outcome -> the right next call, with keys that match.
Also the dispatcher, the active-call guard, the stale-call reaper and the daily planner. Runs on SQLite by default and on real
Postgres when TEST_DATABASE_URL is set."""
import json
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from carecall.agent.interpret import RuleInterpreter
from carecall.agent.session import CallSession
from carecall.calls import outcome, planner, scheduler
from carecall.calls.keys import attempt_key, occurrence_key, parse_attempt_key
from carecall.calls.service import create_call, has_live_call, in_quiet_hours, schedule_attempt, shift_out_of_quiet_hours
from carecall.calls.state import transition
from carecall.db import session_scope
from carecall.models import (Callback, CallAttempt, Caregiver, Event, Notification, Parent, Routine, TranscriptTurn, utcnow)

NOW = datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc)  # 11:30 in Kolkata
OCC = occurrence_key(1, "lunch_checkin", "2026-10-18")


# ---------------------------------------------------------------- helpers
def make_call(status="abandoned", end_reason="", turns=(), fin=None, attempt_no=1, connected=True, routine="lunch_checkin",
              parent_id=1, secs=60, occ=OCC, kind="scheduled", when=None) -> int:
    """A finished call, built directly, with exactly the facts the outcome engine reads."""
    with session_scope() as s:
        rid = s.scalars(select(Routine.id).where(Routine.parent_id == parent_id, Routine.key == routine)).first()
        key = attempt_key(occ, attempt_no, kind)
        c = s.scalars(select(CallAttempt).where(CallAttempt.idempotency_key == key)).first()  # the follow-up row already exists:
        if c is None:                                                                         # finishing it is what a real call does
            c = CallAttempt(parent_id=parent_id, routine_id=rid, attempt_no=attempt_no, occurrence_key=occ, kind=kind,
                            idempotency_key=key, channel="twilio", scheduled_at=when or NOW)
            s.add(c)
        c.status, c.end_reason = status, end_reason
        if connected:
            c.started_at, c.ended_at = NOW, NOW + timedelta(seconds=secs)
        s.flush()
        for i, (who, text) in enumerate(turns, 1):
            s.add(TranscriptTurn(call_id=c.id, turn_no=i, speaker=who, text=text))
        s.add(Event(call_id=c.id, type="call.finished", payload=fin or {}))
        return c.id


def followups():
    with session_scope() as s:
        return [(c.idempotency_key, c.kind, c.attempt_no, c.status, c.scheduled_at) for c in
                s.scalars(select(CallAttempt).where(CallAttempt.prev_attempt_id.is_not(None)).order_by(CallAttempt.id))]


def delay_of(call_id, row):
    from carecall.models import as_utc
    with session_scope() as s:
        return (as_utc(row[4]) - NOW).total_seconds()


def notes():
    with session_scope() as s:
        return [n.body for n in s.scalars(select(Notification))]


# ---------------------------------------------------------------- keys
def test_keys_are_computed_and_parse_back():
    o = occurrence_key(7, "metformin_reminder", "2026-10-18")
    assert o == "7:metformin_reminder:2026-10-18:s1"
    assert attempt_key(o, 1) == o + ":a1" and attempt_key(o, 2, "retry") == o + ":a2" and attempt_key(o, 3, "callback") == o + ":cb3"
    assert parse_attempt_key(o + ":cb3") == (o, "cb", 3)


# ---------------------------------------------------------------- every way a call can end
def test_completed_call_needs_nothing_more():
    cid = make_call("completed", "completed", fin={"done": True, "unresolved": []})
    o = outcome.apply(cid, now=NOW)
    assert o.kind == "completed" and followups() == []


def test_a_requested_call_back_is_scheduled_with_a_callback_key():
    cid = make_call("completed", "callback", fin={"done": True})
    with session_scope() as s:
        s.add(Callback(parent_id=1, call_id=cid, routine_id=1, delay_minutes=20, reason="not yet eaten"))
    o = outcome.apply(cid, now=NOW)
    (key, kind, n, status, at), = followups()
    assert (o.kind, kind, n, status) == ("callback_requested", "callback", 2, "scheduled")
    assert key == OCC + ":cb2" and (at.replace(tzinfo=timezone.utc) - NOW).total_seconds() == 20 * 60
    with session_scope() as s:
        assert s.scalars(select(Callback)).one().status == "scheduled"


def test_an_emergency_is_never_retried():
    cid = make_call("completed", "emergency", fin={"done": True})
    assert outcome.apply(cid, now=NOW).kind == "emergency" and followups() == []


def test_off_topic_limit_is_not_retried():
    cid = make_call("completed", "off_topic_limit", fin={"done": True})
    assert outcome.apply(cid, now=NOW).kind == "off_topic_limit" and followups() == []


def test_no_answer_is_retried_once_after_10_minutes():
    cid = make_call("no_answer", attempt_no=1, connected=False)
    o = outcome.apply(cid, now=NOW)
    (key, kind, n, _, at), = followups()
    assert o.kind == "never_connected" and key == OCC + ":a2" and n == 2
    assert (at.replace(tzinfo=timezone.utc) - NOW).total_seconds() == 10 * 60


def test_after_two_attempts_we_stop_and_the_family_is_told():
    """The limit is 2 per reminder: no chasing the parent about lunch all afternoon."""
    cid = make_call("no_answer", attempt_no=2, connected=False)
    o = outcome.apply(cid, now=NOW)
    assert o.kind == "never_connected" and followups() == [] and any("could not reach" in n.lower() for n in notes())


def test_technical_failure_retries_quickly_then_tells_the_family():
    assert outcome.apply(make_call("failed", attempt_no=1, connected=False), now=NOW).follow_up.delay_s == 5 * 60
    assert outcome.apply(make_call("failed", attempt_no=2, connected=False, occ=OCC + "x"), now=NOW).follow_up is None
    assert any("kept failing" in n for n in notes())


def test_a_line_that_drops_while_the_parent_is_mid_sentence_gets_an_immediate_call_back():
    turns = [("agent", "Have you had your lunch today?"), ("parent", "Yes, I had rice and then I")]
    cid = make_call("abandoned", turns=turns, fin={"done": False, "parent_speaking_at_hangup": True, "steps_resolved": []})
    o = outcome.apply(cid, now=NOW)
    (key, kind, n, _, at), = followups()
    assert o.kind == "cut_off" and key == OCC + ":a2" and kind == "retry"
    assert (at.replace(tzinfo=timezone.utc) - NOW).total_seconds() == 0  # ring back right away: no waiting


def test_an_unanswered_question_is_also_a_cut_off():
    cid = make_call("abandoned", turns=[("agent", "Have you taken your tablet?")], fin={"done": False})
    assert outcome.apply(cid, now=NOW).kind == "cut_off"


def test_an_unfinished_sentence_is_a_cut_off_even_if_the_agent_spoke_last_is_false():
    cid = make_call("abandoned", turns=[("agent", "Have you had lunch?"), ("parent", "yes and"), ("agent", "What did you have?")],
                    fin={"done": False})
    assert outcome.apply(cid, now=NOW).kind == "cut_off"


def test_cut_offs_get_one_immediate_ring_back_then_the_family_is_told():
    turns = [("agent", "Have you had your lunch today?")]
    o1 = outcome.apply(make_call("abandoned", turns=turns, attempt_no=1, fin={"done": False}), now=NOW)
    o2 = outcome.apply(make_call("abandoned", turns=turns, attempt_no=2, fin={"done": False}), now=NOW)
    assert o1.follow_up and o2.follow_up is None
    assert any("keep getting cut off" in n for n in notes())


def test_a_retry_that_would_land_after_the_next_reminder_is_dropped():
    """Lunch not answered; dinner's call is due before the 10-minute retry would be: skip the retry and move on."""
    sched(NOW + timedelta(minutes=5), routine="dinner_checkin")
    cid = make_call("no_answer", attempt_no=1, connected=False)
    outcome.apply(cid, now=NOW)
    assert followups() == []
    with session_scope() as s:
        assert any(e.type == "followup.skipped" for e in s.scalars(select(Event)))
    assert any("moving on" in n for n in notes())


def test_opting_out_tells_the_family_even_when_the_call_otherwise_completed():
    cid = make_call("completed", turns=[("agent", "Have you eaten?"), ("parent", "Yes. And please stop calling me.")], fin={"done": True})
    assert outcome.apply(cid, now=NOW).kind == "opted_out"
    assert any("asked us to stop calling" in n for n in notes())


def test_a_goodbye_is_hanging_up_on_purpose_and_waits_30_minutes_once():
    turns = [("agent", "Have you had your lunch today?"), ("parent", "I am busy now, bye.")]
    cid = make_call("abandoned", turns=turns, fin={"done": False})
    o = outcome.apply(cid, now=NOW)
    (key, _, _, _, at), = followups()
    assert o.kind == "hung_up_on_purpose" and (at.replace(tzinfo=timezone.utc) - NOW).total_seconds() == 30 * 60
    again = outcome.apply(make_call("abandoned", turns=turns, attempt_no=2, fin={"done": False}), now=NOW)
    assert again.follow_up is None  # does not keep calling someone who ends the call twice


class Judge:
    model = "judge"

    def __init__(self, verdict, confidence=0.9):
        self.reply = json.dumps({"verdict": verdict, "confidence": confidence, "reason": "stub"})
        self.last_raw = ""

    def complete(self, system, user):
        return self.reply


def conflicted():  # cut-off AND goodbye evidence: the rules disagree with each other
    return dict(turns=[("agent", "Have you had your lunch today?"), ("parent", "I have to go, bye, I will")],
                fin={"done": False, "parent_speaking_at_hangup": True})


def test_when_the_signals_conflict_a_model_decides():
    assert outcome.apply(make_call("abandoned", **conflicted()), client=Judge("on_purpose"), now=NOW).kind == "hung_up_on_purpose"


def test_a_model_that_is_unsure_or_unavailable_means_ring_back():
    assert outcome.apply(make_call("abandoned", **conflicted()), client=Judge("on_purpose", 0.3), now=NOW).kind == "cut_off"
    assert outcome.apply(make_call("abandoned", occ=OCC + "n", **conflicted()), client=None, now=NOW).kind == "cut_off"


def test_stop_calling_me_switches_calling_off_and_cancels_what_is_scheduled():
    with session_scope() as s:
        schedule_attempt(s, parent_id=1, routine_id=1, occurrence_key="1:dinner_checkin:2026-10-18:s1", attempt_no=1, kind="scheduled",
                         when=NOW + timedelta(hours=8))
    cid = make_call("abandoned", turns=[("agent", "Have you had lunch?"), ("parent", "Please stop calling me")], fin={"done": False})
    o = outcome.apply(cid, now=NOW)
    with session_scope() as s:
        assert o.kind == "opted_out" and s.get(Parent, 1).calling_enabled is False
        assert [c.status for c in s.scalars(select(CallAttempt).where(CallAttempt.status != "abandoned"))] == ["cancelled"]
    assert followups() == [] and any("stop calling" in n for n in notes())


def test_a_call_still_in_flight_is_not_decided_yet():
    cid = make_call("in_progress")
    assert outcome.apply(cid, now=NOW).kind == "in_flight"
    with session_scope() as s:
        assert s.get(CallAttempt, cid).outcome is None


def test_deciding_twice_schedules_exactly_one_follow_up():
    """The post-call step can run twice (a crash, a repeated webhook). The key is computed, so the second run finds the first."""
    cid = make_call("no_answer", connected=False)
    first, second = outcome.apply(cid, now=NOW), outcome.apply(cid, now=NOW + timedelta(seconds=5))
    assert first.kind == second.kind and len(followups()) == 1
    with session_scope() as s:  # even with the stored outcome cleared, the computed key refuses a duplicate
        s.get(CallAttempt, cid).outcome = None
    outcome.apply(cid, now=NOW)
    assert len(followups()) == 1


def test_a_follow_up_is_pushed_out_of_the_night():
    cid = make_call("no_answer", connected=False)
    late = datetime(2026, 10, 18, 16, 50, tzinfo=timezone.utc)  # 22:20 Kolkata; +10 min is still the night
    outcome.apply(cid, now=late)
    (_, _, _, _, at), = followups()
    assert at.replace(tzinfo=timezone.utc) == datetime(2026, 10, 19, 1, 30, tzinfo=timezone.utc)  # 07:00 the next morning, Kolkata


def test_quiet_hours_window():
    assert in_quiet_hours(datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc), "Asia/Kolkata")  # 22:30
    assert in_quiet_hours(datetime(2026, 10, 18, 0, 0, tzinfo=timezone.utc), "Asia/Kolkata")  # 05:30
    assert not in_quiet_hours(datetime(2026, 10, 18, 7, 0, tzinfo=timezone.utc), "Asia/Kolkata")  # 12:30
    assert shift_out_of_quiet_hours(datetime(2026, 10, 18, 7, 0, tzinfo=timezone.utc), "Asia/Kolkata") == datetime(2026, 10, 18, 7, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- dispatcher
class Dialer:
    def __init__(self, fail=False):
        self.dialed, self.fail = [], fail

    def __call__(self, call_id):
        if self.fail:
            raise RuntimeError("twilio down")
        self.dialed.append(call_id)
        return f"CA{call_id}"


def sched(when, parent_id=1, key_date="2026-10-18", routine="lunch_checkin", n=1, routine_id=None):
    with session_scope() as s:
        rid = routine_id or s.scalars(select(Routine.id).where(Routine.parent_id == parent_id, Routine.key == routine)).first()
        c, _ = schedule_attempt(s, parent_id=parent_id, routine_id=rid, occurrence_key=occurrence_key(parent_id, routine, key_date),
                                attempt_no=n, kind="scheduled", when=when)
        return c.id


def status_of(cid):
    with session_scope() as s:
        return s.get(CallAttempt, cid).status


def test_due_calls_are_dialed_and_future_ones_are_left_alone():
    due, later = sched(NOW - timedelta(minutes=1)), sched(NOW + timedelta(hours=2), key_date="2026-10-19")
    d = Dialer()
    scheduler.dispatch_once(d, now=NOW)
    assert d.dialed == [due] and status_of(due) == "dialing" and status_of(later) == "scheduled"
    scheduler.dispatch_once(d, now=NOW)  # nothing new is due, nothing is dialed twice
    assert d.dialed == [due]


def test_a_parent_is_never_called_twice_at_once():
    """Lunch check-in and a medicine reminder due the same minute: one rings, the other waits."""
    a = sched(NOW - timedelta(minutes=1))
    b = sched(NOW - timedelta(minutes=1), routine="metformin_reminder")
    d = Dialer()
    scheduler.dispatch_once(d, now=NOW)
    assert len(d.dialed) == 1
    waiting = b if d.dialed == [a] else a
    assert status_of(waiting) == "scheduled"
    with session_scope() as s:
        assert (s.get(CallAttempt, waiting).scheduled_at.replace(tzinfo=timezone.utc) - NOW).total_seconds() == 300
        assert any(e.type == "dispatch.postponed" for e in s.scalars(select(Event)))


def test_an_existing_live_call_blocks_a_new_dial():
    make_call("in_progress", occ="live")
    due = sched(NOW - timedelta(minutes=1))
    d = Dialer()
    scheduler.dispatch_once(d, now=NOW)
    assert d.dialed == [] and status_of(due) == "scheduled"
    with session_scope() as s:
        assert has_live_call(s, 1) is not None and has_live_call(s, 2) is None


def test_a_parent_who_opted_out_is_never_dialed():
    due = sched(NOW - timedelta(minutes=1))
    with session_scope() as s:
        s.get(Parent, 1).calling_enabled = False
    d = Dialer()
    scheduler.dispatch_once(d, now=NOW)
    assert d.dialed == [] and status_of(due) == "cancelled"


def test_if_twilio_refuses_the_call_it_is_retried_not_lost():
    due = sched(NOW - timedelta(minutes=1))
    scheduler.dispatch_once(Dialer(fail=True), now=NOW)
    assert status_of(due) == "failed"
    (key, _, n, _, _), = followups()
    assert key == OCC + ":a2" and n == 2


def test_two_parents_are_dialed_in_the_same_tick():
    with session_scope() as s:
        p2 = Parent(name="Ravi Rao", preferred_name="Ravi", phone="+910000000002", timezone="Asia/Kolkata")
        s.add(p2)
        s.flush()
        s.add(Routine(parent_id=p2.id, key="lunch_checkin", type="meal_checkin", config={"meal_type": "lunch"}))
        pid2 = p2.id
    a, b = sched(NOW - timedelta(minutes=1)), sched(NOW - timedelta(minutes=1), parent_id=pid2, key_date="2026-10-18")
    d = Dialer()
    scheduler.dispatch_once(d, now=NOW)
    assert sorted(d.dialed) == sorted([a, b])  # different people can ring at the same time


def test_calls_stuck_dialing_or_in_progress_are_reaped_and_retried():
    dialing = make_call("dialing", connected=False, occ="s1")
    with session_scope() as s:
        s.get(CallAttempt, dialing).dialed_at = NOW - timedelta(minutes=10)
    long_running = make_call("in_progress", occ="s2")
    with session_scope() as s:
        s.get(CallAttempt, long_running).started_at = NOW - timedelta(hours=2)
    reaped = scheduler.reap_stale(now=NOW)
    assert sorted(reaped) == sorted([dialing, long_running]) and status_of(dialing) == "failed" and status_of(long_running) == "failed"
    assert len(followups()) == 2  # each gets its retry
    assert scheduler.reap_stale(now=NOW) == []  # and reaping again does nothing


# ---------------------------------------------------------------- planner
def test_plan_day_turns_onboarding_data_into_dated_calls():
    ids = planner.plan_day(date(2026, 10, 18), now=datetime(2026, 10, 17, 20, 0, tzinfo=timezone.utc))
    with session_scope() as s:
        rows = {c.occurrence_key.split(":")[1]: c.scheduled_at.replace(tzinfo=timezone.utc) for c in
                s.scalars(select(CallAttempt).where(CallAttempt.id.in_(ids)))}
    ist = lambda h, m: datetime(2026, 10, 18, h, m, tzinfo=timezone.utc) - timedelta(hours=5, minutes=30)  # noqa: E731
    assert rows["breakfast_checkin"] == ist(9, 0)  # 08:30 + 30 min
    assert rows["metformin_reminder"] == ist(12, 15)  # before lunch: 12:30 - 15 min
    assert rows["lunch_checkin"] == ist(13, 0)  # 12:30 + 30 min
    assert rows["amlodipine_reminder"] == ist(20, 0)  # after dinner: 19:30 + 30 min
    assert rows["dinner_checkin"] == ist(20, 0)
    assert "snack_checkin" not in rows and "meal_discovery" not in rows and "doctor_reminder" not in rows


def test_plan_day_is_idempotent():
    now = datetime(2026, 10, 17, 20, 0, tzinfo=timezone.utc)
    first = planner.plan_day(date(2026, 10, 18), now=now)
    assert first and planner.plan_day(date(2026, 10, 18), now=now) == []
    with session_scope() as s:
        assert len(s.scalars(select(CallAttempt)).all()) == len(first)


def test_plan_day_skips_moments_that_already_passed_and_paused_parents():
    late = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)  # 14:30 in Kolkata
    ids = planner.plan_day(date(2026, 10, 18), now=late)
    with session_scope() as s:
        names = sorted(c.occurrence_key.split(":")[1] for c in s.scalars(select(CallAttempt).where(CallAttempt.id.in_(ids))))
    assert names == ["amlodipine_reminder", "dinner_checkin"]  # no 1 pm lunch call at 2:30 pm
    with session_scope() as s:
        s.get(Parent, 1).calling_enabled = False
    assert planner.plan_day(date(2026, 10, 19), now=late) == []


def test_a_meal_time_the_caretaker_fixed_moves_the_calls():
    from carecall.models import MealProfile

    with session_scope() as s:
        row = s.get(MealProfile, (1, "lunch"))
        row.usual_time, row.source = "14:00", "caregiver"
    ids = planner.plan_day(date(2026, 10, 18), now=datetime(2026, 10, 17, 20, 0, tzinfo=timezone.utc))
    with session_scope() as s:
        lunch = next(c for c in s.scalars(select(CallAttempt).where(CallAttempt.id.in_(ids))) if "lunch_checkin" in c.occurrence_key)
        assert lunch.scheduled_at.replace(tzinfo=timezone.utc) == datetime(2026, 10, 18, 14, 30, tzinfo=timezone.utc) - timedelta(hours=5, minutes=30)


def test_onboarding_schedules_one_discovery_call():
    first = planner.plan_onboarding(1, now=NOW)
    assert first and planner.plan_onboarding(1, now=NOW) is None


# ---------------------------------------------------------------- resume + end to end
def test_a_cut_off_discovery_call_resumes_at_the_unanswered_question():
    cid, _ = create_call("meal_discovery", channel="twilio", occurrence_key="1:meal_discovery:onboarding:s1",
                         idempotency_key="1:meal_discovery:onboarding:s1:a1")
    s1 = CallSession(cid, RuleInterpreter(), channel="twilio")
    s1.open()
    s1.turn("around 8 am")
    s1.turn("about 1:30")
    s1.finish(abandoned=True, hangup={"parent_speaking": True})  # the line dropped while she was talking
    with session_scope() as s:
        first = s.get(CallAttempt, cid)
        assert first.status == "abandoned" and first.outcome == "cut_off"
        nxt = s.scalars(select(CallAttempt).where(CallAttempt.prev_attempt_id == cid)).one()
        assert nxt.idempotency_key == "1:meal_discovery:onboarding:s1:a2" and nxt.attempt_no == 2 and nxt.kind == "retry"
        nid = nxt.id
    s2 = CallSession(nid, RuleInterpreter(), channel="twilio")
    opening = s2.open()
    assert "Sorry, we got cut off" in opening[0]
    assert opening[1] == "What time do you usually have snack?"  # breakfast and lunch are NOT asked again
    s2.turn("around 5 pm")
    s2.turn("at 8:30 pm")
    r = s2.turn("yes that is right")  # the read-back still includes the times from the first call
    assert r.done
    with session_scope() as s:
        said = [t.text for t in s.scalars(select(TranscriptTurn).where(TranscriptTurn.call_id == nid, TranscriptTurn.speaker == "agent"))]
    assert any("breakfast around 8 in the morning" in t and "lunch around 1:30" in t for t in said)


def test_finishing_a_normal_call_through_the_session_schedules_nothing():
    cid, _ = create_call("lunch_checkin", channel="twilio", occurrence_key=OCC, idempotency_key=OCC + ":a1")
    s = CallSession(cid, RuleInterpreter(), channel="twilio")
    s.open()
    s.turn("yes")
    s.turn("rice and dal")
    s.finish()
    with session_scope() as q:
        assert q.get(CallAttempt, cid).outcome == "completed"
    assert followups() == []


def test_text_and_eval_channels_never_schedule_real_calls():
    cid, _ = create_call("lunch_checkin", channel="text")
    s = CallSession(cid, RuleInterpreter())
    s.open()
    s.finish(abandoned=True, hangup={"parent_speaking": True})
    assert followups() == []
