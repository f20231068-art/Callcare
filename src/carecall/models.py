"""Database schema.

Design rules:
- All timestamps are UTC. Local time is derived from parents.timezone at the edges.
- Facts (meal_logs, medicine_logs) always carry `source_utterance`: the parent's exact words.
  If you can't point at a quote, you shouldn't have a log.
- `events` is append-only: it is the audit trail. Other tables hold current/derived state.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite returns timezone-NAIVE datetimes even for timezone-aware columns; Postgres returns aware ones. Normalise to aware UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class Base(DeclarativeBase):
    pass


class Parent(Base):
    __tablename__ = "parents"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    preferred_name: Mapped[str] = mapped_column(String(100))
    phone: Mapped[str] = mapped_column(String(32))
    language: Mapped[str] = mapped_column(String(16), default="en")
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata")
    consent_given_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    calling_enabled: Mapped[bool] = mapped_column(Boolean, default=True)  # false after "stop calling me": nothing is ever dialed


class Caregiver(Base):
    __tablename__ = "caregivers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    phone: Mapped[str] = mapped_column(String(32))
    alert_mode: Mapped[str] = mapped_column(String(16), default="digest")  # immediate | digest


class ParentCaregiver(Base):
    __tablename__ = "parent_caregiver"
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"), primary_key=True)
    caregiver_id: Mapped[int] = mapped_column(ForeignKey("caregivers.id"), primary_key=True)
    relationship_: Mapped[str] = mapped_column("relationship", String(32), default="child")


class MealProfile(Base):
    """When the parent usually eats. Drives medicine timing. Learned in the discovery call."""
    __tablename__ = "meal_profile"
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"), primary_key=True)
    meal_type: Mapped[str] = mapped_column(String(16), primary_key=True)  # breakfast|lunch|snack|dinner
    usual_time: Mapped[str] = mapped_column(String(5))  # "HH:MM", parent-local
    source: Mapped[str] = mapped_column(String(16), default="stated")  # stated | inferred | seed
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Medicine(Base):
    __tablename__ = "medicines"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"))
    name: Mapped[str] = mapped_column(String(100))
    dose_note: Mapped[str] = mapped_column(String(200), default="")
    timing: Mapped[str] = mapped_column(String(16))  # before_food | after_food | with_food | any
    meal_anchor: Mapped[str | None] = mapped_column(String(16))
    offset_minutes: Mapped[int] = mapped_column(Integer, default=0)  # after_food: minutes after the meal
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Reminder(Base):
    """One-shot alarm-style reminder (doctor appointment etc)."""
    __tablename__ = "reminders"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"))
    message: Mapped[str] = mapped_column(String(300))
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # unset in Phase 0
    ack_required: Mapped[bool] = mapped_column(Boolean, default=True)


class Routine(Base):
    __tablename__ = "routines"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"))
    key: Mapped[str] = mapped_column(String(64))  # e.g. lunch_checkin
    type: Mapped[str] = mapped_column(String(32))  # meal_discovery|meal_checkin|medicine_reminder|defined_reminder
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # {"meal_type": "lunch"} / {"medicine_id": 1}
    version: Mapped[int] = mapped_column(Integer, default=1)
    __table_args__ = (UniqueConstraint("parent_id", "key"),)


class CallAttempt(Base):
    __tablename__ = "call_attempts"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"))
    routine_id: Mapped[int] = mapped_column(ForeignKey("routines.id"))
    status: Mapped[str] = mapped_column(String(24), default="scheduled")  # see calls/state.py
    channel: Mapped[str] = mapped_column(String(16), default="text")  # text | twilio
    twilio_sid: Mapped[str | None] = mapped_column(String(64))
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    dialed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # when we asked Twilio to ring (stale-call detection)
    # One "occurrence" = one planned call event ("Asha, lunch check-in, 18 Oct"). Every attempt to complete it shares this key and
    # carries its own idempotency key: <occurrence>:a1 (first), :a2 (retry), :cb3 (parent asked for a call back)...
    occurrence_key: Mapped[str | None] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="scheduled")  # scheduled | retry | callback | manual
    prev_attempt_id: Mapped[int | None] = mapped_column(Integer)  # the attempt whose outcome created this one
    end_reason: Mapped[str] = mapped_column(String(32), default="")  # how the conversation ended (agent's view)
    outcome: Mapped[str | None] = mapped_column(String(32))  # what the call turned out to be (see calls/outcome.py)
    outcome_reason: Mapped[str] = mapped_column(String(300), default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    prompt_version: Mapped[str] = mapped_column(String(16), default="v1")
    model_version: Mapped[str] = mapped_column(String(64), default="")
    parent: Mapped[Parent] = relationship()
    routine: Mapped[Routine] = relationship()


class TranscriptTurn(Base):
    __tablename__ = "transcript_turns"
    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("call_attempts.id"))
    turn_no: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str] = mapped_column(String(8))  # agent | parent
    text: Mapped[str] = mapped_column(Text)
    stt_confidence: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)  # parent turns: time the agent took to reply
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MealLog(Base):
    __tablename__ = "meal_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"))
    call_id: Mapped[int] = mapped_column(ForeignKey("call_attempts.id"))
    meal_type: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))  # eaten | skipped | not_yet | unclear
    items: Mapped[list] = mapped_column(JSON, default=list)
    source_utterance: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MedicineLog(Base):
    __tablename__ = "medicine_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"))
    call_id: Mapped[int] = mapped_column(ForeignKey("call_attempts.id"))
    medicine_id: Mapped[int] = mapped_column(ForeignKey("medicines.id"))
    status: Mapped[str] = mapped_column(String(16))  # taken | not_taken | unclear
    source_utterance: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReminderAck(Base):
    __tablename__ = "reminder_acks"
    id: Mapped[int] = mapped_column(primary_key=True)
    reminder_id: Mapped[int] = mapped_column(ForeignKey("reminders.id"))
    call_id: Mapped[int] = mapped_column(ForeignKey("call_attempts.id"))
    acknowledged: Mapped[bool] = mapped_column(Boolean)
    source_utterance: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Callback(Base):
    __tablename__ = "callbacks"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"))
    call_id: Mapped[int] = mapped_column(ForeignKey("call_attempts.id"))
    routine_id: Mapped[int] = mapped_column(ForeignKey("routines.id"))
    delay_minutes: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parents.id"))
    call_id: Mapped[int | None] = mapped_column(ForeignKey("call_attempts.id"))
    severity: Mapped[str] = mapped_column(String(16))  # emergency | warning | info
    reason: Mapped[str] = mapped_column(String(300))
    source_utterance: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CallSummary(Base):
    __tablename__ = "call_summaries"
    call_id: Mapped[int] = mapped_column(ForeignKey("call_attempts.id"), primary_key=True)
    summary_text: Mapped[str] = mapped_column(Text)
    anomalies: Mapped[list] = mapped_column(JSON, default=list)
    review_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reason: Mapped[str] = mapped_column(String(300), default="")


class Notification(Base):
    """Outbox for messages to caregivers (SMS in production, console in dev)."""
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(primary_key=True)
    caregiver_id: Mapped[int] = mapped_column(ForeignKey("caregivers.id"))
    call_id: Mapped[int | None] = mapped_column(ForeignKey("call_attempts.id"))
    kind: Mapped[str] = mapped_column(String(16))  # alert | summary
    body: Mapped[str] = mapped_column(Text)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Event(Base):
    """Append-only audit trail. Never update or delete rows here."""
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[int | None] = mapped_column(ForeignKey("call_attempts.id"))
    type: Mapped[str] = mapped_column(String(48))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TestNote(Base):
    __tablename__ = "test_notes"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64))
    scenario_id: Mapped[str] = mapped_column(String(64))
    call_id: Mapped[int | None] = mapped_column(Integer)  # not an FK: eval calls live in per-scenario DBs
    deterministic_passed: Mapped[bool] = mapped_column(Boolean)
    severity_max: Mapped[str] = mapped_column(String(16), default="none")
    note_json: Mapped[dict] = mapped_column(JSON, default=dict)
    note_markdown: Mapped[str] = mapped_column(Text, default="")
    writer: Mapped[str] = mapped_column(String(64), default="")
    human_verdict: Mapped[str | None] = mapped_column(String(16))  # agree | disagree | needs_review
    human_comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Account(Base):
    """A caregiver's login. One account owns one caregiver record, and through parent_caregiver, the parents they look after."""
    __tablename__ = "accounts"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(200), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))  # scrypt$salt$hash, never the password
    caregiver_id: Mapped[int] = mapped_column(ForeignKey("caregivers.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuthSession(Base):
    """A logged-in browser. Only the SHA-256 of the cookie token is stored, so a database leak cannot be replayed as a login."""
    __tablename__ = "auth_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
