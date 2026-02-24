"""SQLite persistence helpers for the Management UI."""

from __future__ import annotations

import json
import os
import sqlite3
import hashlib
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: bool | int) -> int:
    return 1 if value else 0


_DB_PATH_OVERRIDE: Optional[Path] = None


def set_db_path_override(path: Optional[str]) -> None:
    global _DB_PATH_OVERRIDE
    if path:
        _DB_PATH_OVERRIDE = Path(path)
    else:
        _DB_PATH_OVERRIDE = None


def _resolve_db_path() -> Path:
    global _DB_PATH_OVERRIDE
    if _DB_PATH_OVERRIDE is not None:
        return _DB_PATH_OVERRIDE

    env_path = os.getenv("MANAGEMENT_DB_PATH")
    if env_path:
        return Path(env_path)

    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Path(os.getenv("MANAGEMENT_DB_PATH", data_dir / "management.db"))


def _dict_from_row(row: sqlite3.Row) -> Dict:
    return {k: row[k] for k in row.keys()}


def _coerce_json_payload(value: Optional[str]) -> Optional[Any]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed


DEFAULT_CONTEXT_SCHEMA_VERSION = "2"


def _normalize_import_metadata(source_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if source_metadata is None:
        return {}
    normalized = source_metadata
    if not isinstance(normalized, dict):
        normalized = _coerce_json_payload(normalized)
        if not isinstance(normalized, dict):
            return {}
    return {
        "source_actor_id": (
            normalized.get("source_actor_id")
            or normalized.get("actor_id")
            or normalized.get("actor")
            or normalized.get("source_actor")
        ),
        "source_mission_id": (
            normalized.get("source_mission_id")
            or normalized.get("mission_id")
            or normalized.get("mission")
        ),
        "context_schema_version": (
            normalized.get("context_schema_version")
            or normalized.get("schema_version")
            or normalized.get("context_schema")
            or DEFAULT_CONTEXT_SCHEMA_VERSION
        ),
    }


def _enrich_import_job_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    source_metadata = _coerce_json_payload(payload.get("source_metadata")) or {}
    payload["source_metadata"] = source_metadata if isinstance(source_metadata, dict) else {}
    derived = _normalize_import_metadata(payload["source_metadata"])
    if derived.get("source_actor_id") is not None:
        payload["source_actor_id"] = derived["source_actor_id"]
        payload["actor_id"] = derived["source_actor_id"]
    if derived.get("source_mission_id") is not None:
        payload["source_mission_id"] = derived["source_mission_id"]
        payload["mission_id"] = derived["source_mission_id"]
    payload["context_schema_version"] = (
        payload["source_metadata"].get("context_schema_version")
        or payload["source_metadata"].get("schema_version")
        or payload["source_metadata"].get("context_schema")
        or DEFAULT_CONTEXT_SCHEMA_VERSION
    )
    payload["rerun_of_import_id"] = payload["source_metadata"].get("rerun_of_import_id")
    return payload


def _coerce_metadata_for_storage(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, separators=(",", ":"))
    except TypeError:
        return None


def _item_checksum(values: Dict[str, Any]) -> str:
    raw = json.dumps(
        {
            "kind": values.get("kind"),
            "content": values.get("content"),
            "agent_scope": values.get("agent_scope"),
            "thread_id": values.get("thread_id"),
            "message_id": values.get("message_id"),
            "author_key": values.get("author_key"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_events_payload_column(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
    if "payload" not in columns:
        conn.execute("ALTER TABLE events ADD COLUMN payload TEXT")


def _ensure_import_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS import_jobs (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            status TEXT NOT NULL,
            checksum TEXT,
            source_metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(source_type, checksum)
        );

        CREATE TABLE IF NOT EXISTS import_items (
            id TEXT PRIMARY KEY,
            import_id TEXT NOT NULL,
            import_index INTEGER NOT NULL,
            kind TEXT NOT NULL,
            agent_scope TEXT NOT NULL,
            content TEXT NOT NULL,
            visibility TEXT NOT NULL,
            source TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            review_state TEXT NOT NULL,
            author_key TEXT,
            metadata TEXT,
            checksum TEXT NOT NULL,
            FOREIGN KEY(import_id) REFERENCES import_jobs(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_import_items_import_checksum
            ON import_items(import_id, checksum);

        CREATE INDEX IF NOT EXISTS idx_import_items_import_id
            ON import_items(import_id);

        CREATE TABLE IF NOT EXISTS import_item_events (
            id TEXT PRIMARY KEY,
            import_id TEXT NOT NULL,
            import_item_id TEXT,
            event_type TEXT NOT NULL,
            review_state TEXT,
            actor TEXT,
            note TEXT,
            payload TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(import_id) REFERENCES import_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(import_item_id) REFERENCES import_items(id) ON DELETE CASCADE
        );
        """
    )


def _ensure_settings_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(settings)")}
    if "proxy_http" not in columns:
        conn.execute("ALTER TABLE settings ADD COLUMN proxy_http TEXT")
    if "proxy_https" not in columns:
        conn.execute("ALTER TABLE settings ADD COLUMN proxy_https TEXT")
    if "proxy_no_proxy" not in columns:
        conn.execute("ALTER TABLE settings ADD COLUMN proxy_no_proxy TEXT")


def _ensure_plugin_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS plugins (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            status TEXT NOT NULL,
            source TEXT NOT NULL,
            kind TEXT NOT NULL,
            command TEXT,
            config TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )


def _coerce_proxy_value(value: Any | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


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
                proxy_http TEXT NOT NULL DEFAULT '',
                proxy_https TEXT NOT NULL DEFAULT '',
                proxy_no_proxy TEXT NOT NULL DEFAULT '',
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
                level TEXT NOT NULL,
                payload TEXT
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
        _ensure_events_payload_column(conn)
        _ensure_import_tables(conn)
        _ensure_settings_columns(conn)
        _ensure_plugin_table(conn)
        conn.commit()

        row = conn.execute("SELECT 1 FROM settings WHERE id='singleton'").fetchone()
        if not row:
            now = _now_iso()
            conn.execute(
                """
                INSERT INTO settings(id, heartbeat_interval, fallback_budget_per_hour, auto_restart, proxy_http, proxy_https, proxy_no_proxy, updated_at)
                VALUES ('singleton', 30, 420, 1, '', '', '', ?)
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
                (*agent, now, now),
            )

        defaults = [
            ("evt-1", "orchestration", "Heartbeat healthy.", _now_iso(), "info"),
        ]
        for event in defaults:
            conn.execute(
                "INSERT OR IGNORE INTO events(id, type, message, created_at, level, payload) VALUES(?, ?, ?, ?, ?, ?)",
                event + (None,),
            )

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
        payload["proxy_http"] = payload.get("proxy_http") or ""
        payload["proxy_https"] = payload.get("proxy_https") or ""
        payload["proxy_no_proxy"] = payload.get("proxy_no_proxy") or ""
        return payload


def set_settings(settings: Dict) -> Dict:
    with get_connection() as conn:
        _ensure_settings_columns(conn)
        payload = get_settings()
        payload.update(settings)
        now = _now_iso()
        conn.execute(
            """
            UPDATE settings
            SET heartbeat_interval=?, fallback_budget_per_hour=?, auto_restart=?, proxy_http=?, proxy_https=?, proxy_no_proxy=?, updated_at=?
            WHERE id='singleton'
            """,
            (
                int(payload["heartbeat_interval"]),
                int(payload["fallback_budget_per_hour"]),
                _coerce_bool(payload["auto_restart"]),
                _coerce_proxy_value(payload.get("proxy_http")),
                _coerce_proxy_value(payload.get("proxy_https")),
                _coerce_proxy_value(payload.get("proxy_no_proxy")),
                now,
            ),
        )
        conn.commit()
        payload["updated_at"] = now
        payload["auto_restart"] = bool(payload["auto_restart"])
        payload["proxy_http"] = _coerce_proxy_value(payload.get("proxy_http"))
        payload["proxy_https"] = _coerce_proxy_value(payload.get("proxy_https"))
        payload["proxy_no_proxy"] = _coerce_proxy_value(payload.get("proxy_no_proxy"))
        return payload


def list_plugins() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM plugins ORDER BY datetime(updated_at) DESC").fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["enabled"] = bool(payload["enabled"])
            payload["config"] = _coerce_json_payload(payload.get("config"))
            payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
            out.append(payload)
        return out


def get_plugin(plugin_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM plugins WHERE id = ?", (plugin_id,)).fetchone()
        if not row:
            return None
        payload = _dict_from_row(row)
        payload["enabled"] = bool(payload["enabled"])
        payload["config"] = _coerce_json_payload(payload.get("config"))
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        return payload


def create_plugin(values: Dict[str, Any]) -> Dict[str, Any]:
    if not values.get("id"):
        raise ValueError("plugin id is required")
    if not values.get("name"):
        raise ValueError("plugin name is required")
    if not values.get("source"):
        raise ValueError("plugin source is required")
    if not values.get("kind"):
        raise ValueError("plugin kind is required")

    with get_connection() as conn:
        existing = conn.execute("SELECT 1 FROM plugins WHERE id = ?", (values["id"],)).fetchone()
        if existing:
            raise ValueError(f"plugin '{values['id']}' already exists")

        now = _now_iso()
        payload = {
            "id": values["id"],
            "name": values["name"],
            "enabled": int(bool(values.get("enabled", True))),
            "status": values.get("status", "configured"),
            "source": values["source"],
            "kind": values["kind"],
            "command": values.get("command"),
            "config": _coerce_metadata_for_storage(values.get("config")),
            "metadata": _coerce_metadata_for_storage(values.get("metadata")),
            "created_at": now,
            "updated_at": now,
        }
        if not payload["enabled"]:
            payload["status"] = "disabled"

        conn.execute(
            """
            INSERT INTO plugins(
              id, name, enabled, status, source, kind, command, config, metadata, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["name"],
                payload["enabled"],
                payload["status"],
                payload["source"],
                payload["kind"],
                payload["command"],
                payload["config"],
                payload["metadata"],
                payload["created_at"],
                payload["updated_at"],
            ),
        )
        conn.commit()
        payload["enabled"] = bool(payload["enabled"])
        payload["config"] = _coerce_json_payload(payload.get("config"))
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        return payload


def update_plugin(plugin_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM plugins WHERE id = ?", (plugin_id,)).fetchone()
        if not existing:
            raise KeyError(f"plugin '{plugin_id}' not found")

        row = _dict_from_row(existing)
        payload = dict(row) | values
        payload["enabled"] = int(bool(payload.get("enabled", row["enabled"])))
        payload["config"] = _coerce_metadata_for_storage(payload.get("config"))
        payload["metadata"] = _coerce_metadata_for_storage(payload.get("metadata"))
        if not payload["enabled"] and not str(payload.get("status")).strip():
            payload["status"] = row["status"] if row["status"] else "disabled"
        if not payload["enabled"]:
            payload["status"] = "disabled"
        elif not payload["status"]:
            payload["status"] = row["status"] if row["status"] else "configured"

        now = _now_iso()
        payload["updated_at"] = now
        conn.execute(
            """
            UPDATE plugins
            SET name=?, enabled=?, status=?, source=?, kind=?, command=?, config=?, metadata=?, updated_at=?
            WHERE id=?
            """,
            (
                payload["name"],
                payload["enabled"],
                payload["status"],
                payload["source"],
                payload["kind"],
                payload.get("command"),
                payload["config"],
                payload["metadata"],
                payload["updated_at"],
                plugin_id,
            ),
        )
        conn.commit()
        payload["enabled"] = bool(payload["enabled"])
        payload["config"] = _coerce_json_payload(payload.get("config"))
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        return payload


def get_provider(provider_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not row:
            return None
        payload = _dict_from_row(row)
        payload["enabled"] = bool(payload["enabled"])
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        return payload


def list_providers() -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM providers ORDER BY priority ASC").fetchall()
        out: List[Dict] = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["enabled"] = bool(payload["enabled"])
            payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
            out.append(payload)
        return out


def update_provider(provider_id: str, values: Dict) -> Dict:
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()
        if not existing:
            raise KeyError(f"provider '{provider_id}' not found")
        row = _dict_from_row(existing)
        payload = row | values
        payload["metadata"] = _coerce_metadata_for_storage(payload.get("metadata"))
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
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        return payload


def create_provider(values: Dict[str, Any]) -> Dict[str, Any]:
    if not values.get("id"):
        raise ValueError("provider id is required")
    if not values.get("name"):
        raise ValueError("provider name is required")
    if not values.get("endpoint"):
        raise ValueError("provider endpoint is required")
    if not values.get("model"):
        raise ValueError("provider model is required")

    with get_connection() as conn:
        existing = conn.execute("SELECT 1 FROM providers WHERE id = ?", (values["id"],)).fetchone()
        if existing:
            raise ValueError(f"provider '{values['id']}' already exists")

        payload = {
            "id": values["id"],
            "name": values["name"],
            "enabled": int(bool(values.get("enabled", True))),
            "status": values.get("status", "standby" if not values.get("enabled", True) else values.get("status", "configured")),
            "model": values["model"],
            "endpoint": values["endpoint"],
            "priority": int(values.get("priority", 0)),
            "metadata": _coerce_metadata_for_storage(values.get("metadata")),
            "updated_at": _now_iso(),
        }
        if not payload["enabled"]:
            payload["status"] = "standby"

        conn.execute(
            """
            INSERT INTO providers(id, name, enabled, status, model, endpoint, priority, metadata, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"],
                payload["name"],
                payload["enabled"],
                payload["status"],
                payload["model"],
                payload["endpoint"],
                payload["priority"],
                payload["metadata"],
                payload["updated_at"],
            ),
        )
        conn.commit()
        payload["enabled"] = bool(payload["enabled"])
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
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
        out = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["payload"] = _coerce_json_payload(payload.get("payload"))
            out.append(payload)
        return out


def append_event(event_type: str, message: str, level: str = "info", payload: Optional[Dict[str, Any]] = None) -> Dict:
    event_id = f"evt-{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    serialized_payload = json.dumps(payload) if payload is not None else None
    event_payload = {
        "id": event_id,
        "type": event_type,
        "message": message,
        "created_at": now,
        "level": level,
        "payload": payload,
    }
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO events(id, type, message, created_at, level, payload) VALUES(?, ?, ?, ?, ?, ?)",
            (event_id, event_type, message, now, level, serialized_payload),
        )
        conn.commit()
    return event_payload


def append_import_progress_event(
    import_id: str,
    phase: str,
    details: Optional[Dict[str, Any]] = None,
    db_connection: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    def _write_event(conn: sqlite3.Connection) -> Dict[str, Any]:
        event_id = f"evt-{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        event_payload = {
            "import_id": import_id,
            "phase": phase,
            **(details or {}),
        }
        conn.execute(
            "INSERT INTO events(id, type, message, created_at, level, payload) VALUES(?, ?, ?, ?, ?, ?)",
            (
                event_id,
                "import.progress",
                f"{import_id}: {phase}",
                now,
                "info",
                json.dumps(event_payload),
            ),
        )
        return {
            "id": event_id,
            "type": "import.progress",
            "message": f"{import_id}: {phase}",
            "created_at": now,
            "level": "info",
            "payload": event_payload,
        }

    if db_connection is not None:
        return _write_event(db_connection)

    return append_event(
        event_type="import.progress",
        message=f"{import_id}: {phase}",
        payload={"import_id": import_id, "phase": phase, **(details or {})},
    )


def append_import_item_review_event(
    import_item_id: str,
    event_type: str,
    review_state: str,
    actor: Optional[str] = None,
    note: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    db_connection: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    def _write_event(conn: sqlite3.Connection) -> Dict[str, Any]:
        row = conn.execute(
            "SELECT import_id FROM import_items WHERE id = ?",
            (import_item_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"import item '{import_item_id}' not found")
        now = _now_iso()
        import_id = row["import_id"]
        event_id = f"import-item-event-{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO import_item_events(
                id,
                import_id,
                import_item_id,
                event_type,
                review_state,
                actor,
                note,
                payload,
                created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                import_id,
                import_item_id,
                event_type,
                review_state,
                actor,
                note,
                json.dumps(payload) if payload is not None else None,
                now,
            ),
        )
        event_payload = {
            "event_type": event_type,
            "import_id": import_id,
            "import_item_id": import_item_id,
            "review_state": review_state,
            "actor": actor,
            "note": note,
            **(payload or {}),
        }
        conn.execute(
            "INSERT INTO events(id, type, message, created_at, level, payload) VALUES(?, ?, ?, ?, ?, ?)",
            (
                f"evt-{uuid.uuid4().hex[:12]}",
                "import.item.reviewed",
                f"{import_item_id}: {review_state}",
                now,
                "info",
                json.dumps(event_payload),
            ),
        )
        return {
            "id": event_id,
            "import_id": import_id,
            "import_item_id": import_item_id,
            "event_type": event_type,
            "review_state": review_state,
            "actor": actor,
            "note": note,
            "payload": payload,
            "created_at": now,
        }

    if db_connection is not None:
        return _write_event(db_connection)

    with get_connection() as conn:
        event_payload = _write_event(conn)
        conn.commit()
        return event_payload


def _item_requires_review(item: Dict[str, Any]) -> bool:
    metadata = _coerce_json_payload(item.get("metadata"))
    if not metadata:
        return False
    if metadata.get("requires_review"):
        return True
    flags = metadata.get("sensitive_flags")
    return bool(flags)


def get_import_job(import_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM import_jobs WHERE id = ?", (import_id,)).fetchone()
        if not row:
            return None
        payload = _dict_from_row(row)
        return _enrich_import_job_payload(payload)


def _build_directory_snapshot_for_import(conn: sqlite3.Connection, import_id: str) -> Optional[Dict[str, Any]]:
    job_row = conn.execute("SELECT * FROM import_jobs WHERE id = ?", (import_id,)).fetchone()
    if not job_row:
        return None

    job = _dict_from_row(job_row)
    source_metadata = _coerce_json_payload(job.get("source_metadata"))
    if not isinstance(source_metadata, dict):
        source_metadata = {}

    approved_rows = conn.execute(
        """
        SELECT
            ii.id, ii.agent_scope, ii.kind, ii.content, ii.visibility, ii.source,
            ii.thread_id, ii.message_id, ii.created_at, ii.metadata
        FROM import_items ii
        WHERE ii.import_id = ? AND ii.review_state = 'approved'
        ORDER BY ii.agent_scope, ii.import_index, datetime(ii.created_at) DESC
        """,
        (import_id,),
    ).fetchall()

    scope_groups: Dict[str, Dict[str, Any]] = {}
    for item_row in approved_rows:
        item = _dict_from_row(item_row)
        metadata = _coerce_json_payload(item.get("metadata"))
        if not isinstance(metadata, dict):
            metadata = {}
        scope = str(item.get("agent_scope") or "global").strip() or "global"
        normalized_title = str(
            metadata.get("agent_name")
            or metadata.get("name")
            or metadata.get("title")
            or metadata.get("display_name")
            or scope
        ).strip() or scope
        responsibility = str(metadata.get("summary") or metadata.get("description") or item.get("content") or "").strip()
        responsibility = " ".join(responsibility.split())
        if len(responsibility) > 180:
            responsibility = f"{responsibility[:177]}…"

        scope_entry = scope_groups.setdefault(
            scope,
            {
                "scope": scope,
                "agent_id": scope,
                "current_owner_role": normalized_title,
                "escalation_target": metadata.get("escalation_target"),
                "responsibilities": [],
            },
        )
        scope_entry["responsibilities"].append(
            {
                "import_item_id": item.get("id"),
                "responsibility_domain": responsibility,
                "visibility": item.get("visibility"),
                "source": item.get("source"),
                "thread_id": item.get("thread_id"),
                "message_id": item.get("message_id"),
                "created_at": item.get("created_at"),
            }
        )

    entries = [scope_groups[scope] for scope in sorted(scope_groups)]
    for entry in entries:
        entry["item_count"] = len(entry.get("responsibilities", []))

    return {
        "artifact_id": f"directory:{import_id}",
        "generated_from_import_id": import_id,
        "source_type": job.get("source_type"),
        "import_status": job.get("status"),
        "source_actor_id": (
            source_metadata.get("source_actor_id")
            or source_metadata.get("actor_id")
            or source_metadata.get("actor")
            or source_metadata.get("source_actor")
        ),
        "source_mission_id": (
            source_metadata.get("source_mission_id")
            or source_metadata.get("mission_id")
            or source_metadata.get("mission")
        ),
        "context_schema_version": (
            source_metadata.get("context_schema_version")
            or source_metadata.get("schema_version")
            or source_metadata.get("context_schema")
            or "2"
        ),
        "directory_version": "1.0",
        "imported_item_count": len(approved_rows),
        "agent_count": len(entries),
        "entries": entries,
        "scope_distribution": {entry["scope"]: entry["item_count"] for entry in entries},
        "generated_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "source_checksum": job.get("checksum"),
    }


def reconcile_agent_directory_snapshot(import_id: str) -> Dict[str, Any]:
    now = _now_iso()
    with get_connection() as conn:
        snapshot = _build_directory_snapshot_for_import(conn, import_id)
        if snapshot is None:
            raise KeyError(f"import '{import_id}' not found")

        row = conn.execute("SELECT source_metadata FROM import_jobs WHERE id = ?", (import_id,)).fetchone()
        metadata = _coerce_json_payload(row["source_metadata"] if row else None) if row else {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        metadata["directory_snapshot"] = snapshot
        conn.execute(
            """
            UPDATE import_jobs
            SET source_metadata = ?, updated_at = ?
            WHERE id = ?
            """,
            (_coerce_metadata_for_storage(metadata), now, import_id),
        )
        append_import_progress_event(
            import_id=import_id,
            phase="directory.snapshot.reconciled",
            details={
                "artifact_id": snapshot.get("artifact_id"),
                "agent_count": snapshot.get("agent_count"),
                "imported_item_count": snapshot.get("imported_item_count"),
            },
            db_connection=conn,
        )
        conn.commit()
        return snapshot


def update_import_job_status(import_id: str, status: str) -> Dict[str, Any]:
    now = _now_iso()
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM import_jobs WHERE id = ?", (import_id,)).fetchone()
        if not existing:
            raise KeyError(f"import '{import_id}' not found")
        conn.execute("UPDATE import_jobs SET status=?, updated_at=? WHERE id = ?", (status, now, import_id))
        conn.commit()
    append_import_progress_event(import_id=import_id, phase="status", details={"status": status})
    return get_import_job(import_id)  # type: ignore[return-value]


def update_import_items_review_state(
    import_item_ids: Sequence[str],
    review_state: str,
    actor: Optional[str] = None,
    note: Optional[str] = None,
    import_id: Optional[str] = None,
    agent_scope: Optional[str] = None,
) -> List[Dict[str, Any]]:
    valid_states = {"pending", "approved", "quarantined", "rejected"}
    if review_state not in valid_states:
        raise ValueError(f"unsupported review state: {review_state}")

    normalized_agent_scope: Optional[str] = None
    if agent_scope is not None:
        normalized_agent_scope = str(agent_scope).strip()
        if not normalized_agent_scope:
            raise ValueError("agent_scope cannot be empty when provided")

    if not import_item_ids:
        raise ValueError("import_item_ids cannot be empty")

    now = _now_iso()
    updated: List[Dict[str, Any]] = []
    updated_ids = set()
    with get_connection() as conn:
        candidate_rows: List[Dict[str, Any]] = []
        for item_id in import_item_ids:
            row = conn.execute("SELECT * FROM import_items WHERE id = ?", (item_id,)).fetchone()
            if not row:
                continue
            item = _dict_from_row(row)
            if import_id and item["import_id"] != import_id:
                raise KeyError(f"import item '{item_id}' is not in import '{import_id}'")
            if review_state in {"approved", "rejected"} and _item_requires_review(item) and not (note and note.strip()):
                raise ValueError(
                    f"review note required for sensitive item '{item_id}' when setting review_state='{review_state}'"
                )
            candidate_rows.append(item)

        for item in candidate_rows:
            item_id = item["id"]
            if item["id"] in updated_ids:
                continue
            scope_before = str(item.get("agent_scope") or "").strip()
            scope_after = normalized_agent_scope if normalized_agent_scope is not None else scope_before

            if normalized_agent_scope is None:
                conn.execute("UPDATE import_items SET review_state = ? WHERE id = ?", (review_state, item_id))
            else:
                conn.execute(
                    "UPDATE import_items SET review_state = ?, agent_scope = ? WHERE id = ?",
                    (review_state, scope_after, item_id),
                )
            appended = append_import_item_review_event(
                import_item_id=item_id,
                event_type="import.item.reviewed",
                review_state=review_state,
                actor=actor,
                note=note,
                payload={
                    "job_id": item["import_id"],
                    "agent_scope_before": scope_before,
                    "agent_scope_after": scope_after,
                },
                db_connection=conn,
            )
            updated.append(
                {
                    **item,
                    "review_state": review_state,
                    "agent_scope": scope_after,
                    "metadata": _coerce_json_payload(item.get("metadata")) or {},
                    "event": appended,
                }
            )
            updated_ids.add(item_id)
        conn.commit()

    return updated


def reassign_import_items_scope(
    source_scope: str,
    target_scope: str,
    actor: Optional[str] = None,
    note: Optional[str] = None,
    import_item_ids: Optional[Sequence[str]] = None,
    import_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    normalized_source = str(source_scope).strip() or "global"
    normalized_target = str(target_scope).strip()
    if not normalized_target:
        raise ValueError("target_scope cannot be empty")
    if normalized_source == normalized_target:
        raise ValueError("source_scope and target_scope cannot be the same")

    with get_connection() as conn:
        if import_item_ids:
            candidate_rows = []
            seen: set[str] = set()
            for item_id in import_item_ids:
                row = conn.execute("SELECT * FROM import_items WHERE id = ?", (item_id,)).fetchone()
                if not row:
                    raise KeyError(f"import item '{item_id}' not found")
                item = _dict_from_row(row)
                if (str(item.get("agent_scope") or "global").strip() or "global") != normalized_source:
                    raise ValueError(
                        f"import item '{item_id}' is not in source scope '{normalized_source}'"
                    )
                if import_id and item["import_id"] != import_id:
                    raise KeyError(f"import item '{item_id}' is not in import '{import_id}'")
                if item_id in seen:
                    continue
                candidate_rows.append(item)
                seen.add(item_id)
        else:
            query = "SELECT * FROM import_items WHERE agent_scope = ?"
            args: List[Any] = [normalized_source]
            if import_id:
                query += " AND import_id = ?"
                args.append(import_id)
            candidate_rows = [dict(row) for row in conn.execute(query, tuple(args)).fetchall()]

        if not candidate_rows:
            return []

        updated: List[Dict[str, Any]] = []
        for item in candidate_rows:
            item_id = item["id"]
            before_scope = str(item.get("agent_scope") or "global").strip() or "global"
            conn.execute(
                "UPDATE import_items SET agent_scope = ? WHERE id = ?",
                (normalized_target, item_id),
            )
            appended = append_import_item_review_event(
                import_item_id=item_id,
                event_type="import.item.scope_reassigned",
                review_state=item.get("review_state", "pending"),
                actor=actor,
                note=note,
                payload={
                    "job_id": item["import_id"],
                    "agent_scope_before": before_scope,
                    "agent_scope_after": normalized_target,
                },
                db_connection=conn,
            )
            updated_item = dict(item)
            updated_item["agent_scope"] = normalized_target
            updated_item["event"] = appended
            updated.append(updated_item)
        conn.commit()

    return updated


def create_import_job(
    source_type: str,
    source_metadata: Optional[Dict[str, Any]],
    checksum: str,
    status: str = "pending",
    allow_duplicate_checksum: bool = False,
) -> tuple[Dict[str, Any], bool]:
    now = _now_iso()
    metadata = source_metadata or {}
    persist_checksum: Optional[str] = checksum
    if allow_duplicate_checksum:
        metadata = dict(metadata)
        if checksum and "source_checksum" not in metadata:
            metadata["source_checksum"] = checksum
        # Allow intentional reruns by keeping checksum nullable in this path while
        # preserving the original checksum in metadata for traceability.
        persist_checksum = None
    with get_connection() as conn:
        if not allow_duplicate_checksum:
            existing = conn.execute(
                "SELECT * FROM import_jobs WHERE source_type = ? AND checksum = ?",
                (source_type, checksum),
            ).fetchone()
            if existing:
                payload = _dict_from_row(existing)
                payload["source_metadata"] = _coerce_json_payload(payload.get("source_metadata")) or {}
                return payload, False

        job_id = f"import-{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO import_jobs(
              id,
              source_type,
              status,
              checksum,
              source_metadata,
              created_at,
              updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, source_type, status, persist_checksum, json.dumps(metadata), now, now),
        )
        conn.commit()
    append_import_progress_event(import_id=job_id, phase="created", details={"status": status})
    return {
        "id": job_id,
        "source_type": source_type,
        "status": status,
        "checksum": checksum,
        "source_metadata": metadata,
        "created_at": now,
        "updated_at": now,
    }, True


def get_latest_import_job_for_source(source_type: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT * FROM import_jobs
            WHERE source_type = ?
            ORDER BY datetime(created_at) DESC
            LIMIT 1
            """,
            (source_type,),
        ).fetchone()
        if not row:
            return None
    payload = _dict_from_row(row)
    return _enrich_import_job_payload(payload)


def add_import_items(import_id: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []

    now = _now_iso()
    inserted: List[Dict[str, Any]] = []
    emitted_events: List[tuple[str, str]] = []
    with get_connection() as conn:
        if not conn.execute("SELECT 1 FROM import_jobs WHERE id = ?", (import_id,)).fetchone():
            raise KeyError(f"import '{import_id}' not found")

        for index, raw_item in enumerate(items):
            item = {
                "kind": raw_item.get("kind", "memory"),
                "agent_scope": raw_item["agent_scope"],
                "content": raw_item["content"],
                "visibility": raw_item.get("visibility", "quarantined"),
                "source": raw_item.get("source", ""),
                "thread_id": raw_item.get("thread_id", ""),
                "message_id": raw_item.get("message_id", ""),
                "created_at": raw_item.get("created_at", now),
                "review_state": raw_item.get("review_state", "pending"),
                "author_key": raw_item.get("author_key"),
                "metadata": raw_item.get("metadata", {}),
            }
            item_checksum = raw_item.get("checksum") or _item_checksum(item)
            item_id = f"import-item-{uuid.uuid4().hex[:12]}"
            try:
                conn.execute(
                    """
                    INSERT INTO import_items(
                        id,
                        import_id,
                        import_index,
                        kind,
                        agent_scope,
                        content,
                        visibility,
                        source,
                        thread_id,
                        message_id,
                        created_at,
                        review_state,
                        author_key,
                        metadata,
                        checksum
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        import_id,
                        index,
                        item["kind"],
                        item["agent_scope"],
                        item["content"],
                        item["visibility"],
                        item["source"],
                        item["thread_id"],
                        item["message_id"],
                        item["created_at"],
                        item["review_state"],
                        item["author_key"],
                        json.dumps(item["metadata"]),
                        item_checksum,
                    ),
                )
            except sqlite3.IntegrityError:
                duplicate = conn.execute(
                    "SELECT * FROM import_items WHERE import_id = ? AND checksum = ?",
                    (import_id, item_checksum),
                ).fetchone()
                if duplicate:
                    payload = _dict_from_row(duplicate)
                    payload["metadata"] = _coerce_json_payload(payload.get("metadata")) or {}
                    inserted.append(payload)
                continue

            emitted_events.append((item_id, item["review_state"]))
            inserted.append(
                {
                    "id": item_id,
                    "import_id": import_id,
                    "import_index": index,
                    "kind": item["kind"],
                    "agent_scope": item["agent_scope"],
                    "content": item["content"],
                    "visibility": item["visibility"],
                    "source": item["source"],
                    "thread_id": item["thread_id"],
                    "message_id": item["message_id"],
                    "created_at": item["created_at"],
                    "review_state": item["review_state"],
                    "author_key": item["author_key"],
                    "metadata": item["metadata"],
                    "checksum": item_checksum,
                }
            )

        for item_id, review_state in emitted_events:
            event_id = f"import-item-event-{uuid.uuid4().hex[:12]}"
            item_event = {
                "id": event_id,
                "import_id": import_id,
                "import_item_id": item_id,
                "event_type": "import.item.reviewed",
                "review_state": review_state,
                "actor": "system",
                "note": "Import item materialized",
                "payload": {"job_id": import_id},
                "created_at": now,
            }
            conn.execute(
                """
                INSERT INTO import_item_events(
                    id,
                    import_id,
                    import_item_id,
                    event_type,
                    review_state,
                    actor,
                    note,
                    payload,
                    created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_event["id"],
                    item_event["import_id"],
                    item_event["import_item_id"],
                    item_event["event_type"],
                    item_event["review_state"],
                    item_event["actor"],
                    item_event["note"],
                    json.dumps(item_event["payload"]),
                    item_event["created_at"],
                ),
            )
            conn.execute(
                """
                INSERT INTO events(id, type, message, created_at, level, payload)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    f"evt-{uuid.uuid4().hex[:12]}",
                    "import.item.reviewed",
                    f"{item_id}: {review_state}",
                    now,
                    "info",
                    json.dumps(item_event),
                ),
            )
        conn.commit()

    append_import_progress_event(import_id=import_id, phase="items-buffered", details={"count": len(items)})

    return inserted


def list_import_items(import_id: Optional[str] = None, review_state: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        filters = []
        args: List[Any] = []
        if import_id is not None:
            filters.append("import_id = ?")
            args.append(import_id)
        if review_state is not None:
            filters.append("review_state = ?")
            args.append(review_state)

        query = "SELECT * FROM import_items"
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY import_id, import_index, datetime(created_at) DESC"
        rows = conn.execute(query, tuple(args)).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["metadata"] = _coerce_json_payload(payload.get("metadata")) or {}
            out.append(payload)
        return out


def list_import_progress_events(import_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE type = 'import.progress' ORDER BY datetime(created_at) ASC"
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _coerce_json_payload(row["payload"])
            if not isinstance(payload, dict):
                continue
            if str(payload.get("import_id") or "") != str(import_id):
                continue
            event_payload = _dict_from_row(row)
            event_payload["payload"] = payload
            out.append(event_payload)
        return out


def list_import_item_events(import_id: str, import_item_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        filters = ["import_id = ?"]
        args: List[Any] = [import_id]
        if import_item_id is not None:
            filters.append("import_item_id = ?")
            args.append(import_item_id)
        query = "SELECT * FROM import_item_events WHERE " + " AND ".join(filters) + " ORDER BY datetime(created_at) ASC"
        rows = conn.execute(query, tuple(args)).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["payload"] = _coerce_json_payload(payload.get("payload"))
            out.append(payload)
        return out


def list_agent_directory() -> List[Dict[str, Any]]:
    def _normalize_title(item_metadata: Dict[str, Any], scope: str) -> str:
        title_fields = ("agent_name", "name", "title", "display_name", "owner")
        for key in title_fields:
            value = item_metadata.get(key)
            if value:
                return str(value).strip() or scope
        return scope

    def _normalize_summary(item_metadata: Dict[str, Any], content: str) -> str:
        summary = str(item_metadata.get("summary") or item_metadata.get("description") or "").strip()
        if not summary:
            summary = str(content or "").strip()
        summary = " ".join(summary.split())
        if len(summary) <= 180:
            return summary
        return f"{summary[:177]}…"

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                ii.id,
                ii.kind,
                ii.agent_scope,
                ii.content,
                ii.visibility,
                ii.source,
                ii.thread_id,
                ii.message_id,
                ii.created_at,
                ii.metadata,
                ii.import_id,
                ij.source_type AS import_source_type,
                ij.created_at AS import_created_at
            FROM import_items ii
            LEFT JOIN import_jobs ij
                ON ij.id = ii.import_id
            WHERE review_state = 'approved'
            ORDER BY agent_scope, import_index, datetime(ii.created_at) DESC
            """
        ).fetchall()
        groups: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            row_data = _dict_from_row(row)
            scope = str(row_data.get("agent_scope") or "global").strip() or "global"
            metadata = _coerce_json_payload(row_data.get("metadata"))
            item_metadata = metadata if isinstance(metadata, dict) else {}
            group = groups.setdefault(
                scope,
                {
                    "agent_scope": scope,
                    "item_count": 0,
                    "responsibilities": [],
                },
            )
            group["item_count"] += 1
            group["responsibilities"].append(
                {
                    "id": row_data.get("id"),
                    "kind": row_data.get("kind"),
                    "title": _normalize_title(item_metadata, scope),
                    "summary": _normalize_summary(item_metadata, str(row_data.get("content", ""))),
                    "source": row_data.get("source"),
                    "visibility": row_data.get("visibility"),
                    "thread_id": row_data.get("thread_id"),
                    "message_id": row_data.get("message_id"),
                    "created_at": row_data.get("created_at"),
                    "import_id": row_data.get("import_id"),
                    "import_source_type": row_data.get("import_source_type"),
                    "import_created_at": row_data.get("import_created_at"),
                }
            )

        directory = [groups[scope] for scope in sorted(groups)]
        return directory


def get_agent_directory_artifact() -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                ii.import_id AS import_id,
                ij.source_type AS source_type,
                ij.created_at AS import_created_at,
                ij.updated_at AS import_updated_at,
                ij.status AS import_status,
                ij.source_metadata AS source_metadata,
                ij.checksum AS import_checksum
            FROM import_items ii
            LEFT JOIN import_jobs ij
                ON ij.id = ii.import_id
            WHERE ii.review_state = 'approved'
            ORDER BY datetime(ii.created_at) DESC, datetime(ij.updated_at) DESC
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            return None

        row_data = _dict_from_row(row)
        source_metadata = _coerce_json_payload(row_data.get("source_metadata"))
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        directory_snapshot = source_metadata.get("directory_snapshot")

        if isinstance(directory_snapshot, dict):
            artifact_payload = dict(directory_snapshot)
            artifact_payload["source_type"] = row_data.get("source_type")
            artifact_payload["source_metadata"] = source_metadata
            if "source_actor_id" not in artifact_payload:
                artifact_payload["source_actor_id"] = (
                    source_metadata.get("source_actor_id")
                    or source_metadata.get("actor_id")
                    or source_metadata.get("actor")
                    or source_metadata.get("source_actor")
                )
            if "source_mission_id" not in artifact_payload:
                artifact_payload["source_mission_id"] = (
                    source_metadata.get("source_mission_id")
                    or source_metadata.get("mission_id")
                    or source_metadata.get("mission")
                )
            artifact_payload["generated_at"] = artifact_payload.get(
                "generated_at",
                row_data.get("import_created_at"),
            )
            artifact_payload["updated_at"] = artifact_payload.get(
                "updated_at",
                row_data.get("import_updated_at"),
            )
            artifact_payload["import_status"] = row_data.get("import_status")
            artifact_payload["source_checksum"] = row_data.get("import_checksum")
            artifact_payload["context_schema_version"] = (
                artifact_payload.get("context_schema_version")
                or source_metadata.get("context_schema_version")
                or source_metadata.get("schema_version")
                or source_metadata.get("context_schema")
            )
            return artifact_payload

        approved_item_count = conn.execute(
            "SELECT COUNT(*) AS total FROM import_items WHERE import_id = ? AND review_state = 'approved'",
            (row_data.get("import_id"),),
        ).fetchone()["total"]

        context_schema_version = (
            source_metadata.get("context_schema_version")
            or source_metadata.get("schema_version")
            or source_metadata.get("context_schema")
        )

        return {
            "artifact_id": f"directory:{row_data.get('import_id')}",
            "generated_from_import_id": row_data.get("import_id"),
            "source_type": row_data.get("source_type"),
            "import_status": row_data.get("import_status"),
            "source_actor_id": (
                source_metadata.get("source_actor_id")
                or source_metadata.get("actor_id")
                or source_metadata.get("actor")
                or source_metadata.get("source_actor")
            ),
            "source_mission_id": (
                source_metadata.get("source_mission_id")
                or source_metadata.get("mission_id")
                or source_metadata.get("mission")
            ),
            "source_metadata": source_metadata,
            "imported_item_count": approved_item_count,
            "context_schema_version": context_schema_version,
            "generated_at": row_data.get("import_created_at"),
            "updated_at": row_data.get("import_updated_at"),
            "source_checksum": row_data.get("import_checksum"),
        }


def list_import_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM import_jobs ORDER BY datetime(created_at) DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _dict_from_row(row)
            out.append(_enrich_import_job_payload(payload))
        return out


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
