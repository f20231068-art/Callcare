"""Caregiver accounts: sign up with email + password, log in with a cookie session.

Passwords are stored as scrypt hashes (memory-hard, in the standard library: no extra dependency). The browser holds a random token in an
HttpOnly cookie; the database stores only its SHA-256, so a leaked table cannot be used to log in, and logging out deletes the row."""
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from .config import get_settings
from .db import session_scope
from .models import Account, AuthSession, Caregiver, as_utc

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(ValueError):
    pass


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        cand = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1, dklen=32)
        return hmac.compare_digest(cand.hex(), h)
    except Exception:  # noqa: BLE001 - a malformed stored value is simply "wrong password"
        return False


def normalize_phone(raw: str, default_cc: str | None = None) -> str:
    """'98765 43210' -> '+919876543210'. A number with its own + or 00 prefix is kept; a bare 10-digit one gets the default country code."""
    cc = (default_cc or get_settings().default_country_code).strip()
    text = (raw or "").strip()
    digits = re.sub(r"\D", "", text)
    if text.startswith("+"):
        num = "+" + digits
    elif text.startswith("00"):
        num = "+" + digits[2:]
    elif len(digits) == 10:
        num = cc + digits
    elif digits.startswith(cc.lstrip("+")) and len(digits) > 10:
        num = "+" + digits
    else:
        raise AuthError("Enter the phone number with its country code, e.g. +91 98765 43210.")
    if not re.fullmatch(r"\+\d{8,15}", num):
        raise AuthError("That does not look like a valid phone number.")
    return num


def signup(email: str, password: str, name: str, phone: str = "") -> int:
    email = email.strip().lower()
    if not EMAIL.match(email):
        raise AuthError("Enter a valid email address.")
    if len(password) < 8:
        raise AuthError("Use a password of at least 8 characters.")
    if not name.strip():
        raise AuthError("Enter your name.")
    with session_scope() as s:
        if s.scalars(select(Account).where(Account.email == email)).first():
            raise AuthError("An account with this email already exists. Log in instead.")
        cg = Caregiver(name=name.strip(), phone=normalize_phone(phone) if phone.strip() else "", alert_mode="immediate")
        s.add(cg)
        s.flush()
        acct = Account(email=email, password_hash=hash_password(password), caregiver_id=cg.id)
        s.add(acct)
        s.flush()
        return acct.id


def login(email: str, password: str) -> str:
    """Returns a new session token (to be put in the cookie)."""
    with session_scope() as s:
        acct = s.scalars(select(Account).where(Account.email == email.strip().lower())).first()
        ok = check_password(password, acct.password_hash if acct else "scrypt$00$00")  # same work either way: no timing hint
        if not (acct and ok):
            raise AuthError("Wrong email or password.")
        token = secrets.token_urlsafe(32)
        s.add(AuthSession(token_hash=hashlib.sha256(token.encode()).hexdigest(), account_id=acct.id,
                          expires_at=datetime.now(timezone.utc) + timedelta(days=get_settings().session_days)))
        return token


def account_for_token(token: str | None) -> dict | None:
    if not token:
        return None
    with session_scope() as s:
        row = s.scalars(select(AuthSession).where(AuthSession.token_hash == hashlib.sha256(token.encode()).hexdigest())).first()
        if row is None or as_utc(row.expires_at) < datetime.now(timezone.utc):
            return None
        acct = s.get(Account, row.account_id)
        cg = s.get(Caregiver, acct.caregiver_id)
        return {"account_id": acct.id, "email": acct.email, "caregiver_id": cg.id, "name": cg.name}


def logout(token: str | None) -> None:
    if not token:
        return
    with session_scope() as s:
        for row in s.scalars(select(AuthSession).where(AuthSession.token_hash == hashlib.sha256(token.encode()).hexdigest())):
            s.delete(row)
