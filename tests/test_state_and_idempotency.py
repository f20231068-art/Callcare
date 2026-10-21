from sqlalchemy import select

from carecall.calls.service import create_call
from carecall.calls.state import transition
from carecall.db import session_scope
from carecall.models import CallAttempt, Event


def test_legal_transitions_and_duplicates_ignored():
    cid, _ = create_call("lunch_checkin")
    with session_scope() as s:
        c = s.get(CallAttempt, cid)
        assert transition(s, c, "dialing")
        assert transition(s, c, "ringing")
        assert transition(s, c, "in_progress")
        assert not transition(s, c, "in_progress")  # duplicate webhook: no-op
        assert transition(s, c, "completed")
        assert not transition(s, c, "ringing")  # late 'ringing' after completed: ignored
        assert c.status == "completed" and c.ended_at is not None
        ignored = s.scalars(select(Event).where(Event.type == "state.ignored")).all()
        assert len(ignored) == 2


def test_same_idempotency_key_returns_same_call():
    a, created_a = create_call("lunch_checkin", idempotency_key="k1")
    b, created_b = create_call("lunch_checkin", idempotency_key="k1")
    assert a == b and created_a and not created_b
    c, created_c = create_call("lunch_checkin", idempotency_key="k2")
    assert c != a and created_c
