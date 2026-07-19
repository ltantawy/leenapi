"""SQLite storage for todo items.

Deliberately free of any Flask import: the storage layer is exercised directly
in tests without starting a server, and later phases (calendar events, fired
notifications) add tables here without touching the web layer.

A connection is opened per operation rather than held open. Flask serves
requests on multiple threads, and SQLite connections cannot be shared across
threads; at this volume (a handful of writes a day) the cost is irrelevant.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS todos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT    NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);
"""


@dataclass(frozen=True)
class Todo:
    id: int
    text: str
    done: bool
    created_at: str


class Store:
    """Todo persistence. Creates the database file and schema on first use."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, text: str) -> Todo:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO todos (text, done, created_at) VALUES (?, 0, ?)",
                (text, created_at),
            )
            return Todo(
                id=cursor.lastrowid,
                text=text,
                done=False,
                created_at=created_at,
            )

    def list(self) -> list[Todo]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, text, done, created_at FROM todos ORDER BY id"
            ).fetchall()
        return [
            Todo(
                id=row["id"],
                text=row["text"],
                done=bool(row["done"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]
