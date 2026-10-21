"""Twilio helpers: placing the call, building TwiML, signing the media-stream token."""
import hashlib
import hmac

from sqlalchemy import select
from twilio.request_validator import RequestValidator
from twilio.rest import Client
from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

from ..calls.state import transition
from ..config import get_settings
from ..db import session_scope
from ..models import CallAttempt, Parent


def make_token(call_id: int) -> str:
    """Twilio can't send auth headers on the media stream, so we put a per-call HMAC token in the
    stream parameters and check it when the stream starts. It stops strangers opening /twilio/stream."""
    return hmac.new(get_settings().stream_secret.encode(), f"call:{call_id}".encode(), hashlib.sha256).hexdigest()


def verify_token(call_id: int, token: str) -> bool:
    return hmac.compare_digest(make_token(call_id), token or "")


def ws_url() -> str:
    base = get_settings().public_base_url.rstrip("/")
    return base.replace("https://", "wss://").replace("http://", "ws://") + "/twilio/stream"


def build_twiml(call_id: int) -> str:
    resp = VoiceResponse()
    connect = Connect()
    stream = Stream(url=ws_url())
    stream.parameter(name="call_id", value=str(call_id))
    stream.parameter(name="token", value=make_token(call_id))
    connect.append(stream)
    resp.append(connect)
    resp.hangup()
    return str(resp)


def validate_signature(url: str, params: dict, signature: str) -> bool:
    s = get_settings()
    if not s.twilio_auth_token:
        return True  # local dev without Twilio creds
    return RequestValidator(s.twilio_auth_token).validate(url, params, signature or "")


def place_call(call_id: int) -> str:
    s = get_settings()
    if not (s.twilio_account_sid and s.twilio_auth_token and s.twilio_from_number and s.public_base_url):
        raise RuntimeError("Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER and PUBLIC_BASE_URL in .env")
    with session_scope() as sess:
        call = sess.get(CallAttempt, call_id)
        to = sess.get(Parent, call.parent_id).phone
    base = s.public_base_url.rstrip("/")
    tw = Client(s.twilio_account_sid, s.twilio_auth_token).calls.create(
        to=to, from_=s.twilio_from_number,
        url=f"{base}/twilio/voice?call_id={call_id}&t={make_token(call_id)}",
        status_callback=f"{base}/twilio/status?call_id={call_id}",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
    )
    with session_scope() as sess:
        call = sess.get(CallAttempt, call_id)
        call.twilio_sid = tw.sid
        transition(sess, call, "dialing", "twilio call created")
    return tw.sid


# Twilio CallStatus -> our state machine
STATUS_MAP = {"ringing": "ringing", "in-progress": "in_progress", "answered": "in_progress",
              "busy": "no_answer", "no-answer": "no_answer", "failed": "failed", "canceled": "failed"}


def place_test_call(to: str, text: str = "Hello, this is a CareCall test. If you can hear this, Twilio calling works.") -> str:
    """Rung 1 of the test ladder: no server, tunnel, STT or model. Twilio dials and reads `text` aloud.
    Proves credentials, your from-number, and (on a trial account) that the destination is verified."""
    s = get_settings()
    if not (s.twilio_account_sid and s.twilio_auth_token and s.twilio_from_number):
        raise RuntimeError("Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER in .env")
    resp = VoiceResponse()
    resp.say(text)
    return Client(s.twilio_account_sid, s.twilio_auth_token).calls.create(
        to=to, from_=s.twilio_from_number, twiml=str(resp)).sid


def build_persona_twiml(job_id: str, token: str) -> str:
    """The persona line: when Gnani's agent calls our Twilio number, connect the call audio to the persona bridge."""
    resp = VoiceResponse()
    connect = Connect()
    stream = Stream(url=ws_url().replace("/twilio/stream", "/persona/stream"))
    stream.parameter(name="job_id", value=job_id)
    stream.parameter(name="token", value=token)
    connect.append(stream)
    resp.append(connect)
    resp.hangup()
    return str(resp)


def point_number_at_server(e164: str) -> str:
    """Set the Voice webhook of our Twilio number to {PUBLIC_BASE_URL}/persona/voice. Returns the number's SID."""
    s = get_settings()
    client = Client(s.twilio_account_sid, s.twilio_auth_token)
    nums = client.incoming_phone_numbers.list(phone_number=e164)
    if not nums:
        raise RuntimeError(f"{e164} is not a number on this Twilio account")
    nums[0].update(voice_url=f"{s.public_base_url.rstrip('/')}/persona/voice", voice_method="POST")
    return nums[0].sid
