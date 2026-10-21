"""Database engine and the one-transaction-per-block session helper.

SQLite is the zero-setup default (tests, a laptop demo). Postgres is the real target: many calls for many parents at the same
time need real concurrent writers, row locks and a connection pool. Switch with DATABASE_URL:

    postgresql+psycopg://carecall:carecall@localhost:5433/carecall        (docker compose -f docker/app/docker-compose.yml up -d)

Why it matters for concurrency: SQLite allows ONE writer at a time (others get "database is locked"). Postgres lets many
transactions write at once and gives `SELECT ... FOR UPDATE SKIP LOCKED`, which the call dispatcher uses so two workers can never
pick up the same scheduled call."""
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import Base

_engine = None
_Session = None


def is_postgres() -> bool:
    return _engine is not None and _engine.dialect.name == "postgresql"


def init_db(url: str | None = None):
    """(Re)bind the global engine. Tests pass a file / in-memory URL; production passes Postgres."""
    global _engine, _Session
    s = get_settings()
    url = url or s.database_url
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool
            kwargs["poolclass"] = StaticPool
    else:
        # A pool keeps connections open and hands them to concurrent calls. pre_ping drops dead ones (a DB restart must not
        # fail the next call); recycle avoids server-side idle timeouts.
        kwargs.update(pool_size=s.db_pool_size, max_overflow=s.db_max_overflow, pool_pre_ping=True, pool_recycle=1800)
    if _engine is not None:
        _engine.dispose()
    _engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


@contextmanager
def session_scope():
    """One transaction per `with` block: commit on success, roll back on error."""
    if _Session is None:
        init_db()
    s: Session = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def is_initialized() -> bool:
    return _engine is not None
