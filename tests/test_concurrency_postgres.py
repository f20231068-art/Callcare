"""Real concurrency, on real Postgres (skipped on SQLite, which allows one writer at a time).
Run:  TEST_DATABASE_URL=postgresql+psycopg://carecall:carecall@127.0.0.1:5433/carecall pytest tests/test_concurrency_postgres.py"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from carecall.calls import scheduler
from carecall.calls.keys import occurrence_key
from carecall.calls.service import create_call, schedule_attempt
from carecall.db import is_postgres, session_scope
from carecall.models import CallAttempt, Parent, Routine

pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="needs a real Postgres (set TEST_DATABASE_URL)")
NOW = datetime(2026, 10, 18, 6, 0, tzinfo=timezone.utc)


def test_we_are_really_on_postgres():
    assert is_postgres()


def test_twenty_workers_racing_to_start_the_same_call_create_exactly_one():
    barrier = threading.Barrier(20)

    def start(_):
        barrier.wait()  # all 20 fire at the same instant
        return create_call("lunch_checkin", channel="twilio", idempotency_key="1:lunch_checkin:2026-10-18:s1:a1")

    with ThreadPoolExecutor(20) as pool:
        results = list(pool.map(start, range(20)))
    assert len({cid for cid, _ in results}) == 1 and sum(created for _, created in results) == 1
    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(CallAttempt)) == 1


def test_several_dispatchers_never_dial_the_same_call_twice():
    with session_scope() as s:  # 12 different parents, each with one call due now
        for i in range(12):
            p = Parent(name=f"P{i}", preferred_name=f"P{i}", phone=f"+9100000010{i:02d}", timezone="Asia/Kolkata")
            s.add(p)
            s.flush()
            r = Routine(parent_id=p.id, key="lunch_checkin", type="meal_checkin", config={"meal_type": "lunch"})
            s.add(r)
            s.flush()
            schedule_attempt(s, parent_id=p.id, routine_id=r.id, occurrence_key=occurrence_key(p.id, "lunch_checkin", "2026-10-18"),
                             attempt_no=1, kind="scheduled", when=NOW - timedelta(minutes=1))
    dialed, lock = [], threading.Lock()

    def dialer(call_id):
        with lock:
            dialed.append(call_id)
        return f"CA{call_id}"

    barrier = threading.Barrier(6)

    def worker(_):
        barrier.wait()
        for _ in range(3):
            scheduler.dispatch_once(dialer, now=NOW, limit=5)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(worker, range(6)))
    assert len(dialed) == len(set(dialed)) == 12  # every due call dialed, none twice, across 6 competing workers
    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(CallAttempt).where(CallAttempt.status == "dialing")) == 12
