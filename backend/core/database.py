"""
Database Manager — SQLite persistence for chat history.
Uses Python built-in sqlite3, zero extra dependencies.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "chat_history.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                persona TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)
    print("[DB] SQLite initialized")


def save_session(session_id: str, persona: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, persona, created_at) VALUES (?, ?, ?)",
            (session_id, persona, datetime.utcnow().isoformat())
        )


def save_message(session_id: str, role: str, content: str, sources: list = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(sources or []), datetime.utcnow().isoformat())
        )


def get_sessions() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.session_id, s.persona, s.created_at,
                   COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_session_messages(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()
    return [{**dict(r), "sources": json.loads(r["sources"])} for r in rows]


def delete_session(session_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
