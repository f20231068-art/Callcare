"""Caregiver notifications. Console in dev; Twilio SMS when configured."""
import logging
from typing import Protocol

from ..config import get_settings

log = logging.getLogger(__name__)


class Notifier(Protocol):
    def send(self, phone: str, body: str) -> None: ...


class ConsoleNotifier:
    def send(self, phone: str, body: str) -> None:
        line = f"[notify -> {phone}] {body}"
        try:
            print(line)
        except UnicodeEncodeError:  # a Windows console in cp1252 cannot print Hindi script or emoji: never let that kill the call
            print(line.encode("ascii", "replace").decode("ascii"))


class TwilioSMSNotifier:
    def __init__(self):
        from twilio.rest import Client

        s = get_settings()
        self.client = Client(s.twilio_account_sid, s.twilio_auth_token)
        self.from_ = s.twilio_from_number

    def send(self, phone: str, body: str) -> None:
        self.client.messages.create(to=phone, from_=self.from_, body=body)


def get_notifier() -> Notifier:
    s = get_settings()
    if s.twilio_account_sid and s.twilio_auth_token and s.twilio_from_number and s.sms_notifications:
        return TwilioSMSNotifier()
    return ConsoleNotifier()


def notify_caregivers(session, parent_id: int, call_id: int | None, body: str, kind: str = "alert") -> int:
    """Write an outbox row per caregiver and send it now. Returns how many were notified."""
    from datetime import datetime, timezone

    from sqlalchemy import select

    from ..models import Caregiver, Notification, Parent, ParentCaregiver

    name = session.get(Parent, parent_id).preferred_name
    notifier = get_notifier()
    n = 0
    for cg in session.scalars(select(Caregiver).join(ParentCaregiver, ParentCaregiver.caregiver_id == Caregiver.id)
                              .where(ParentCaregiver.parent_id == parent_id)):
        text = f"{name}: {body}"
        note = Notification(caregiver_id=cg.id, call_id=call_id, kind=kind, body=text)
        notifier.send(cg.phone, text)
        note.sent_at = datetime.now(timezone.utc)
        session.add(note)
        n += 1
    return n
