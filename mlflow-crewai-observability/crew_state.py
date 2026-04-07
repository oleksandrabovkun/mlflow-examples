"""
SQLite-backed shared state for CrewAI task callbacks.

Each task callback persists its output here; the validation tool reads the
research brief back for cross-checking the synthesis.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_DB = Path(__file__).resolve().parent / "crew_state.db"


def _db_path() -> Path:
    raw = os.environ.get("CREW_STATE_DB")
    return Path(raw).expanduser().resolve() if raw else _DEFAULT_DB


def init_db() -> None:
    """Create the ``shared_state`` table if it doesn't exist."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS shared_state "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.commit()


def set_state(key: str, value: str) -> None:
    """Insert or update a key/value pair."""
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            "INSERT INTO shared_state (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, now),
        )
        conn.commit()


def get_state(key: str) -> str | None:
    """Return the value for *key*, or ``None``."""
    path = _db_path()
    if not path.exists():
        return None
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT value FROM shared_state WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else None


def task_output_to_text(output: object) -> str:
    """Normalize a CrewAI ``TaskOutput`` (or similar) to a plain string."""
    if output is None:
        return ""
    for attr in ("raw", "raw_output"):
        val = getattr(output, attr, None)
        if val is not None:
            return str(val)
    return str(output)
