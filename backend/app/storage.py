"""SQLite persistence helpers for the Management UI."""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: bool | int) -> int:
    return 1 if value else 0


def _resolve_db_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Path(os.getenv("MANAGEMENT_DB_PATH", data_dir / "management.db"))


def _dict_from_row(row: sqlite3.Row) -> Dict:
    return {k: row[k] for k in row.keys()}


@contextmanager
def get_connection():
    path = _resolve_db_path()
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def ensure_schema() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                id TEXT PRIMARY KEY CHECK (id = 'singleton'),
                heartbeat_interval INTEGER NOT NULL,
                fallback_budget_per_hour INTEGER NOT NULL,
                auto_restart INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS providers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                status TEXT NOT NULL,
                model TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                priority INTEGER NOT NULL,
                metadata TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                status TEXT NOT NULL,
                role TEXT NOT NULL,
                last_active_at TEXT NOT NULL,
                config TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                level TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        conn.commit()

        row = conn.execute("SELECT 1 FROM settings WHERE id='singleton'").fetchone()
        if not row:
            now = _now_iso()
            conn.execute(
                """
                INSERT INTO settings(id, heartbeat_interval, fallback_budget_per_hour, auto_restart, updated_at)
                VALUES ('singleton', 30, 420, 1, ?)
                """,
                (now,),
            )

        providers = [
            ("openai-primary", "OpenAI", 1, "active", "gpt-4.1-mini", "https://api.openai.com/v1", 1, None),
            ("ollama-secondary", "Ollama", 0, "standby", "llama3.1:8b", "http://127.0.0.1:11434", 2, None),
        ]
        for provider in providers:
            conn.execute(
                """
                INSERT OR IGNORE INTO providers(
                  id, name, enabled, status, model, endpoint, priority, metadata, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                provider + (_now_iso(),),
            )

        agents = [
            ("orchestrator-core", "Orchestrator", 1, "running", "main"),
            ("ops-notify", "Notifier", 1, "running", "support"),
            ("watch-dog", "Security Watchdog", 1, "running", "monitor"),
        ]
        for agent in agents:
            now = _now_iso()
            conn.execute(
                """
                INSERT OR IGNORE INTO agents(
                    id, name, enabled, status, role, last_active_at, config, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (*agent, now),
            )

        defaults = [
            ("evt-1", "orchestration", "Heartbeat healthy.", _now_iso(), "info"),
        ]
        for event in defaults:
            conn.execute("INSERT OR IGNORE INTO events(id, type, message, created_at, level) VALUES(?, ?, ?, ?, ?)", event)

        memories = [
            (
                "memory-1",
                "global",
                "insight",
                "Bootstrap settings reviewed; OpenAI primary and Ollama fallback were loaded.",
                _now_iso(),
            ),
            (
                "memory-2",
                "agent:orchestrator-core",
                "handoff",
                "Orchestrator requested user confirmation for critical security rule updates.",
                _now_iso(),
            ),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO memory(id, scope, type, content, timestamp) VALUES(?, ?, ?, ?, ?)",
            memories,
        )

        chats = [
            (
                "chat-1",
                "orchestrator-core",
                "Session started; user requested provider fallback policy refresh.",
                _now_iso(),
            ),
            (
                "chat-2",
                "ops-notify",
                "Notification policy accepted; waiting for webhook confirmation.",
                _now_iso(),
            ),
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO chat_history(id, agent_id, summary, timestamp) VALUES(?, ?, ?, ?)",
            chats,
        )

        conn.commit()


def get_settings() -> Dict:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM settings WHERE id='singleton'").fetchone()
        if not row:
            ensure_schema()
            row = conn.execute("SELECT * FROM settings WHERE id='singleton'").fetchone()
        payload = _dict_from_row(row)
        payload["auto_restart"] = bool(payload["auto_restart"])
        return payload


def set_settings(settings: Dict) -> Dict:
    with get_connection() as conn:
        payload = get_settings()
        payload.update(settings)
        now = _now_iso()
        conn.execute(
            "UPDATE settings SET heartbeat_interval=?, fallback_budget_per_hour=?, auto_restart=?, updated_at=? WHERE id='singleton'",
            (
                int(payload["heartbeat_interval"]),
                int(payload["fallback_budget_per_hour"]),
                _coerce_bool(payload["auto_restart"]),
                now,
            ),
        )
        conn.commit()
        payload["updated_at"] = now
        payload["auto_restart"] = bool(payload["auto_restart"])
        return payload


def list_providers() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM providers ORDER BY priority ASC").fetchall()
        out: List[Dict] = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["enabled"] = bool(payload["enabled"])
            out.append(payload)
        return out


def update_provider(provider_id: str, values: Dict) -> Dict:
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not existing:
            raise KeyError(f"provider '{provider_id}' not found")
        row = _dict_from_row(existing)
        payload = row | values
        payload["enabled"] = int(bool(payload.get("enabled", row["enabled"])))
        if "enabled" in payload and not payload["enabled"]:
            payload["status"] = "standby"
        elif payload.get("status") is None:
            payload["status"] = row["status"] if row["status"] else "configured"
        now = _now_iso()
        conn.execute(
            """
            UPDATE providers
            SET name=?, enabled=?, status=?, model=?, endpoint=?, priority=?, metadata=?, updated_at=?
            WHERE id=?
            """,
            (
                payload["name"],
                payload["enabled"],
                payload["status"],
                payload["model"],
                payload["endpoint"],
                int(payload["priority"]),
                payload.get("metadata"),
                now,
                provider_id,
            ),
        )
        conn.commit()
        payload["enabled"] = bool(payload["enabled"])
        return payload


def list_agents() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM agents ORDER BY name ASC").fetchall()
        out: List[Dict] = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["enabled"] = bool(payload["enabled"])
            out.append(payload)
        return out


def update_agent(agent_id: str, values: Dict) -> Dict:
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not existing:
            raise KeyError(f"agent '{agent_id}' not found")
        row = _dict_from_row(existing)
        payload = row | values
        payload["enabled"] = int(bool(payload.get("enabled", row["enabled"])))
        if values.get("enabled") is False:
            payload["status"] = "paused"
        elif values.get("enabled") is True:
            payload["status"] = "running"
        payload["last_active_at"] = _now_iso()
        now = _now_iso()
        conn.execute(
            """
            UPDATE agents
            SET name=?, enabled=?, status=?, role=?, last_active_at=?, config=?, updated_at=?
            WHERE id=?
            """,
            (
                payload["name"],
                payload["enabled"],
                payload["status"],
                payload["role"],
                payload["last_active_at"],
                payload.get("config"),
                now,
                agent_id,
            ),
        )
        conn.commit()
        payload["enabled"] = bool(payload["enabled"])
        return payload


def list_events(limit: int = 25) -> List[Dict]:
    normalized_limit = max(1, min(limit, 100))
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY datetime(created_at) DESC LIMIT ?",
            (normalized_limit,),
        ).fetchall()
        return [_dict_from_row(row) for row in rows]


def append_event(event_type: str, message: str, level: str = "info") -> Dict:
    event_id = f"evt-{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    payload = {
        "id": event_id,
        "type": event_type,
        "message": message,
        "created_at": now,
        "level": level,
    }
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO events(id, type, message, created_at, level) VALUES(?, ?, ?, ?, ?)",
            (event_id, event_type, message, now, level),
        )
        conn.commit()
    return payload


def list_memory(scope: Optional[str], item_type: Optional[str]) -> List[Dict]:
    with get_connection() as conn:
        query = "SELECT * FROM memory"
        args: list[str] = []
        clauses: list[str] = []
        if scope:
            clauses.append("(scope = ? OR scope LIKE ?)")
            args.extend([scope, f"{scope}:%"])
        if item_type:
            clauses.append("type = ?")
            args.append(item_type)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY datetime(timestamp) DESC"
        rows = conn.execute(query, tuple(args)).fetchall()
        return [_dict_from_row(row) for row in rows]


def add_memory(scope: str, memory_type: str, content: str) -> Dict:
    memory_id = f"memory-{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    payload = {
        "id": memory_id,
        "scope": scope,
        "type": memory_type,
        "content": content,
        "timestamp": now,
    }
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO memory(id, scope, type, content, timestamp) VALUES(?, ?, ?, ?, ?)",
            (memory_id, scope, memory_type, content, now),
        )
        conn.commit()
    return payload


def list_chat_history(agent_id: Optional[str]) -> List[Dict]:
    with get_connection() as conn:
        if agent_id:
            rows = conn.execute(
                "SELECT * FROM chat_history WHERE agent_id = ? ORDER BY datetime(timestamp) DESC",
                (agent_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_history ORDER BY datetime(timestamp) DESC"
            ).fetchall()
        return [_dict_from_row(row) for row in rows]


def add_chat_entry(agent_id: str, summary: str) -> Dict:
    chat_id = f"chat-{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    payload = {"id": chat_id, "agent_id": agent_id, "summary": summary, "timestamp": now}
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history(id, agent_id, summary, timestamp) VALUES(?, ?, ?, ?)",
            (chat_id, agent_id, summary, now),
        )
        conn.commit()
    return payload
