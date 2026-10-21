"""Idempotency keys that MATCH the call lifecycle.

Two levels, both computed (never random):

  occurrence key   "1:lunch_checkin:2026-10-18:s1"      one planned call event: parent 1, routine, parent-local date, slot
  attempt key      "<occurrence>:a1"                    the first try
                   "<occurrence>:a2"                    the retry that follows attempt 1 (cut off, no answer...)
                   "<occurrence>:cb3"                   the call the parent asked for, as attempt number 3

The attempt number only ever goes up, so the key of "the call that follows attempt N" is always the same string. If the
post-call step runs twice (a crash, a repeated webhook), it computes the same key and the database refuses the duplicate:
the follow-up is created exactly once. A random key could not do that."""
import re

_ATTEMPT = re.compile(r"^(?P<occ>.+):(?P<kind>a|cb)(?P<n>\d+)$")


def occurrence_key(parent_id: int, routine_key: str, local_date: str, slot: str = "s1") -> str:
    return f"{parent_id}:{routine_key}:{local_date}:{slot}"


def attempt_key(occurrence: str, attempt_no: int, kind: str = "scheduled") -> str:
    """kind 'callback' (the parent asked for it) -> ':cb<n>'; everything else -> ':a<n>'."""
    return f"{occurrence}:{'cb' if kind == 'callback' else 'a'}{attempt_no}"


def parse_attempt_key(key: str) -> tuple[str, str, int] | None:
    m = _ATTEMPT.match(key or "")
    return (m["occ"], m["kind"], int(m["n"])) if m else None
