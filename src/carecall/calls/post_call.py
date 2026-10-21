"""Post-call pipeline. Runs after hangup, entirely in code (no model), so it is deterministic and testable.

Steps: summarize -> detect anomalies -> flag for human review -> notify caregivers.
Callbacks scheduled during the call stay in the `callbacks` table for a (future) scheduler."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from ..agent import tools
from ..db import session_scope
from ..models import (
    Alert, CallAttempt, CallSummary, Callback, Caregiver, Event, MealLog, Medicine, MedicineLog,
    Notification, ParentCaregiver, ReminderAck, Parent,
)
from .notify import get_notifier


def _meal_line(l: MealLog) -> str:
    label = l.meal_type.capitalize()
    if l.status == "eaten":
        return f"{label}: had {', '.join(l.items)}." if l.items else f"{label}: eaten (no details)."
    return {"skipped": f"{label}: skipped.", "not_yet": f"{label}: not eaten yet.", "unclear": f"{label}: could not confirm."}[l.status]


def run_post_call(call_id: int, trace_id: str | None = None) -> dict:
    with session_scope() as s:
        call = s.get(CallAttempt, call_id)
        parent = s.get(Parent, call.parent_id)
        meals = s.scalars(select(MealLog).where(MealLog.call_id == call_id)).all()
        meds = s.scalars(select(MedicineLog).where(MedicineLog.call_id == call_id)).all()
        acks = s.scalars(select(ReminderAck).where(ReminderAck.call_id == call_id)).all()
        alerts = s.scalars(select(Alert).where(Alert.call_id == call_id)).all()
        callbacks = s.scalars(select(Callback).where(Callback.call_id == call_id)).all()
        downgrades = s.scalars(select(Event).where(Event.call_id == call_id, Event.type == "validation.downgrade")).all()

        lines = [_meal_line(m) for m in meals]
        for m in meds:
            name = s.get(Medicine, m.medicine_id).name
            lines.append(f"{name}: {m.status.replace('_', ' ')}.")
        for a in acks:
            lines.append("Reminder acknowledged." if a.acknowledged else "Reminder NOT acknowledged.")
        for c in callbacks:
            lines.append(f"Will call again in {c.delay_minutes} min ({c.reason}).")
        for a in alerts:
            lines.append(f"ALERT [{a.severity}]: {a.reason}" + (f' (they said: "{a.source_utterance}")' if a.source_utterance and a.source_utterance not in a.reason else ""))
        if call.status != "completed":
            lines.append(f"Call ended early ({call.status}).")
        if not lines:
            lines.append("No information collected.")

        anomalies: list[str] = []
        for m in meals:
            if m.status == "skipped":
                since = datetime.now(timezone.utc) - timedelta(days=2)
                days = {r.created_at.date() for r in s.scalars(select(MealLog).where(
                    MealLog.parent_id == call.parent_id, MealLog.meal_type == m.meal_type,
                    MealLog.status == "skipped", MealLog.created_at >= since))}
                if len(days) >= 2:
                    anomalies.append(f"{m.meal_type} skipped on {len(days)} consecutive days")
        anomalies += [f"medicine {m.status.replace('_', ' ')}" for m in meds if m.status in ("not_taken", "unclear")]
        anomalies += [a.reason for a in alerts]

        unclear = any(m.status == "unclear" for m in meals) or any(m.status == "unclear" for m in meds)
        review_reason = ", ".join(filter(None, [
            "unclear answer" if unclear else "", f"{len(downgrades)} validation downgrade(s)" if downgrades else "",
            "call ended early" if call.status != "completed" else ""]))

        summary_text = f"{parent.preferred_name}: " + " ".join(lines)
        existing = s.get(CallSummary, call_id)
        if existing:
            existing.summary_text, existing.anomalies = summary_text, anomalies
            existing.review_flag, existing.review_reason = bool(review_reason), review_reason
        else:
            s.add(CallSummary(call_id=call_id, summary_text=summary_text, anomalies=anomalies,
                              review_flag=bool(review_reason), review_reason=review_reason))

        caregivers = s.scalars(select(Caregiver).join(ParentCaregiver, ParentCaregiver.caregiver_id == Caregiver.id)
                               .where(ParentCaregiver.parent_id == call.parent_id)).all()
        notifier = get_notifier()
        urgent = bool(alerts) or bool(anomalies)
        for cg in caregivers:
            kind = "alert" if urgent else "summary"
            body = ("URGENT: " if any(a.severity == "emergency" for a in alerts) else "") + summary_text
            send_now = urgent or cg.alert_mode == "immediate"
            note = Notification(caregiver_id=cg.id, call_id=call_id, kind=kind, body=body)
            if send_now:
                notifier.send(cg.phone, body)
                note.sent_at = datetime.now(timezone.utc)
            s.add(note)
        tools.add_event(s, call_id, "post_call.done", {"anomalies": anomalies, "review": review_reason, "trace_id": trace_id})
        return {"summary": summary_text, "anomalies": anomalies, "review_flag": bool(review_reason)}
