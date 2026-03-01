"""SQLite persistence helpers for the Management UI."""

from __future__ import annotations

import json
import os
import sqlite3
import hashlib
import uuid
from contextlib import contextmanager
import fnmatch
from datetime import datetime, timezone
import math
import re
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


def _normalize_day_filter(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        # accept YYYY-MM-DD or ISO timestamps and normalize to the date portion
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        # fast-fail for malformed date values
        return None


DEFAULT_CONTEXT_SCHEMA_VERSION = "2"
AGENT_LIFECYCLE_ACTIVE = "active"
AGENT_LIFECYCLE_ARCHIVED = "archived"
AGENT_LIFECYCLE_ALL = "all"
_AGENT_LIFECYCLE_VALUES = {AGENT_LIFECYCLE_ACTIVE, AGENT_LIFECYCLE_ARCHIVED}
_GOTICKET_STATUSES = {
    "requested",
    "queued",
    "open",
    "in_progress",
    "blocked",
    "complete",
    "resolved",
    "closed",
    "archived",
}
_GOTICKET_PRIORITIES = {"low", "normal", "high", "critical"}
_GOTICKET_CLOSED_STATUSES = {"complete", "resolved", "closed", "archived"}
_GOTICKET_STATUS_TRANSITIONS = {
    "requested": {"queued", "in_progress", "blocked", "archived"},
    "queued": {"in_progress", "blocked", "archived", "requested"},
    "open": {"queued", "in_progress", "blocked", "resolved", "closed", "archived"},
    "in_progress": {"blocked", "complete", "resolved", "closed", "archived"},
    "blocked": {"queued", "in_progress", "closed", "resolved", "archived"},
    "complete": {"resolved", "closed", "archived"},
    "resolved": {"closed", "archived"},
    "closed": {"archived"},
    "archived": set(),
}


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


def _normalize_agent_lifecycle(value: Optional[str]) -> str:
    if value is None:
        return AGENT_LIFECYCLE_ACTIVE
    normalized = str(value).strip().lower()
    if not normalized:
        return AGENT_LIFECYCLE_ACTIVE
    if normalized not in _AGENT_LIFECYCLE_VALUES and normalized != AGENT_LIFECYCLE_ALL:
        raise ValueError(f"invalid agent lifecycle: {value!r}")
    return normalized


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


def _ensure_gm_ticket_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS gm_tickets (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            priority TEXT NOT NULL,
            agent_scope TEXT NOT NULL,
            phase TEXT NOT NULL,
            requested_by TEXT NOT NULL,
            assigned_to TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            closed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS gm_ticket_messages (
            id TEXT PRIMARY KEY,
            gm_ticket_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            message_type TEXT NOT NULL,
            response_required INTEGER NOT NULL DEFAULT 0,
            metadata TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(gm_ticket_id) REFERENCES gm_tickets(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_gm_tickets_status
            ON gm_tickets(status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_gm_tickets_priority
            ON gm_tickets(priority);
        CREATE INDEX IF NOT EXISTS idx_gm_tickets_agent_scope
            ON gm_tickets(agent_scope);
        CREATE INDEX IF NOT EXISTS idx_gm_tickets_phase
            ON gm_tickets(phase);
        CREATE INDEX IF NOT EXISTS idx_gm_tickets_assigned_to
            ON gm_tickets(assigned_to);
        CREATE INDEX IF NOT EXISTS idx_gm_ticket_messages_ticket
            ON gm_ticket_messages(gm_ticket_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_gm_ticket_messages_sender
            ON gm_ticket_messages(sender);
        """
    )


def _ensure_gm_ticket_message_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(gm_ticket_messages)")}
    if "response_required" not in columns:
        conn.execute(
            "ALTER TABLE gm_ticket_messages ADD COLUMN response_required INTEGER NOT NULL DEFAULT 0"
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


def _ensure_agent_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(agents)")}
    if "lifecycle" not in columns:
        conn.execute("ALTER TABLE agents ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'active'")
    if "archived_at" not in columns:
        conn.execute("ALTER TABLE agents ADD COLUMN archived_at TEXT")


def _ensure_tool_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tool_registry (
            id TEXT PRIMARY KEY,
            plugin_id TEXT NOT NULL,
            name TEXT NOT NULL,
            namespace TEXT NOT NULL,
            action TEXT NOT NULL,
            module TEXT,
            description TEXT,
            signature TEXT,
            parameters TEXT,
            container INTEGER NOT NULL DEFAULT 0,
            popularity INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(plugin_id) REFERENCES plugins(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_registry_plugin_namespace_action
            ON tool_registry(plugin_id, namespace, action);
        CREATE INDEX IF NOT EXISTS idx_tool_registry_namespace
            ON tool_registry(namespace);
        CREATE INDEX IF NOT EXISTS idx_tool_registry_module
            ON tool_registry(module);
        CREATE INDEX IF NOT EXISTS idx_tool_registry_status
            ON tool_registry(status);
        CREATE INDEX IF NOT EXISTS idx_tool_registry_popularity
            ON tool_registry(popularity DESC);
        CREATE INDEX IF NOT EXISTS idx_tool_registry_updated_at
            ON tool_registry(updated_at);

        CREATE TABLE IF NOT EXISTS tool_usage (
            id TEXT PRIMARY KEY,
            tool_id TEXT NOT NULL,
            agent_id TEXT,
            session_id TEXT,
            mission_id TEXT,
            request_id TEXT,
            context_type TEXT,
            context_id TEXT,
            memory_id TEXT,
            chat_message_id TEXT,
            chat_session_id TEXT,
            status TEXT NOT NULL,
            duration_ms INTEGER,
            result_status TEXT,
            details TEXT,
            payload TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(tool_id) REFERENCES tool_registry(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_tool_usage_tool_time
            ON tool_usage(tool_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_tool_usage_created_at
            ON tool_usage(created_at);
        """
    )


def _ensure_tool_usage_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(tool_usage)")}
    if "context_type" not in columns:
        conn.execute("ALTER TABLE tool_usage ADD COLUMN context_type TEXT")
    if "context_id" not in columns:
        conn.execute("ALTER TABLE tool_usage ADD COLUMN context_id TEXT")
    if "memory_id" not in columns:
        conn.execute("ALTER TABLE tool_usage ADD COLUMN memory_id TEXT")
    if "chat_message_id" not in columns:
        conn.execute("ALTER TABLE tool_usage ADD COLUMN chat_message_id TEXT")
    if "chat_session_id" not in columns:
        conn.execute("ALTER TABLE tool_usage ADD COLUMN chat_session_id TEXT")
    if "mission_id" not in columns:
        conn.execute("ALTER TABLE tool_usage ADD COLUMN mission_id TEXT")

    indexes = {row["name"] for row in conn.execute("PRAGMA index_list(tool_usage)")}
    if "idx_tool_usage_context" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_context ON tool_usage(context_type, context_id, created_at)")
    if "idx_tool_usage_memory" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_memory ON tool_usage(memory_id, created_at)")
    if "idx_tool_usage_chat_message" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_chat_message ON tool_usage(chat_message_id, created_at)")
    if "idx_tool_usage_chat_session" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_chat_session ON tool_usage(chat_session_id, created_at)")
    if "idx_tool_usage_session" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_session ON tool_usage(session_id)")
    if "idx_tool_usage_mission" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_mission ON tool_usage(mission_id)")
    if "idx_tool_usage_status" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_status ON tool_usage(status)")
    if "idx_tool_usage_tool_id" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_tool_id ON tool_usage(tool_id)")
    if "idx_tool_usage_agent_id" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_usage_agent_id ON tool_usage(agent_id, created_at)")


def _ensure_chat_history_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(chat_history)")}
    if "chat_session_id" not in columns:
        conn.execute("ALTER TABLE chat_history ADD COLUMN chat_session_id TEXT")

    indexes = {row["name"] for row in conn.execute("PRAGMA index_list(chat_history)")}
    if "idx_chat_history_chat_session" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_chat_session ON chat_history(chat_session_id, timestamp)")
    if "idx_chat_history_agent" not in indexes:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_agent ON chat_history(agent_id, timestamp)")


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
                lifecycle TEXT NOT NULL DEFAULT 'active',
                archived_at TEXT,
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
                timestamp TEXT NOT NULL,
                chat_session_id TEXT
            );
            """
        )
        _ensure_events_payload_column(conn)
        _ensure_import_tables(conn)
        _ensure_settings_columns(conn)
        _ensure_plugin_table(conn)
        _ensure_agent_columns(conn)
        _ensure_tool_tables(conn)
        _ensure_tool_usage_columns(conn)
        _ensure_chat_history_columns(conn)
        _ensure_gm_ticket_tables(conn)
        _ensure_gm_ticket_message_columns(conn)
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
                    id, name, enabled, status, role, last_active_at, config, lifecycle, archived_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)
                """,
                (*agent, now, AGENT_LIFECYCLE_ACTIVE, now),
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


def _normalize_tool_metadata(metadata: Any) -> Dict[str, Any]:
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        parsed = _coerce_json_payload(metadata)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_tool_status(raw: Any) -> str:
    if not raw:
        return "active"
    candidate = str(raw).strip().lower()
    if candidate in {"deprecated", "disabled", "retired", "inactive"}:
        return "disabled"
    return "active"


def _tool_id(plugin_id: str, namespace: str, action: str) -> str:
    raw = f"{plugin_id}::{namespace}::{action}"
    return f"tool-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:18]}"


def _normalize_tool_candidate(candidate: Dict[str, Any], plugin: Dict[str, Any], fallback_module: str) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None

    namespace = _normalize_str(candidate.get("namespace"))
    action = _normalize_str(candidate.get("action"))
    maybe_name = _normalize_str(candidate.get("name"))
    if not namespace and ":" in maybe_name:
        namespace, action = maybe_name.split(":", 1)
    if not namespace and maybe_name:
        namespace = _normalize_str(candidate.get("group") or candidate.get("domain"))

    if not namespace:
        namespace = _normalize_str(candidate.get("ns") or plugin.get("kind") or plugin.get("source") or plugin.get("id"))
    if not action and maybe_name and ":" in maybe_name:
        _, action = maybe_name.split(":", 1)
    if not action:
        action = _normalize_str(candidate.get("command") or candidate.get("method") or maybe_name)

    if not namespace or not action:
        return None

    display_name = maybe_name or f"{namespace}:{action}"
    signature = _normalize_str(candidate.get("signature"))
    parameters = candidate.get("parameters")
    return {
        "id": _tool_id(plugin["id"], namespace, action),
        "plugin_id": plugin["id"],
        "name": display_name,
        "namespace": namespace,
        "action": action,
        "module": _normalize_str(candidate.get("module")) or fallback_module,
        "description": _normalize_str(candidate.get("description")),
        "signature": signature or None,
        "parameters": _coerce_metadata_for_storage(parameters),
        "container": bool(candidate.get("container") or candidate.get("is_container") or False),
        "status": _coerce_tool_status(candidate.get("status")),
        "metadata": _coerce_metadata_for_storage(candidate.get("metadata")),
    }


def _normalize_tool_sources(metadata: Dict[str, Any], plugin: Dict[str, Any], fallback_module: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add_from_entry(value: Any) -> None:
        if not value:
            return
        if isinstance(value, dict):
            if "namespace" in value or "action" in value or "name" in value:
                candidate = _normalize_tool_candidate(value, plugin, fallback_module)
                if candidate:
                    key = f"{candidate['namespace']}::{candidate['action']}"
                    if key in seen:
                        return
                    seen.add(key)
                    specs.append(candidate)
            return
        if isinstance(value, list):
            for nested in value:
                _add_from_entry(nested)

    for key in ("tools", "tool_registry", "commands", "command_registry", "capabilities", "actions"):
        entry = metadata.get(key)
        if key == "capabilities" and isinstance(entry, dict):
            for cap_key in ("tools", "actions", "commands"):
                _add_from_entry(entry.get(cap_key))
            continue
        _add_from_entry(entry)

    return specs


def _normalize_plugin_tools(plugin: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = _normalize_tool_metadata(plugin.get("metadata"))
    fallback_module = _normalize_str(plugin.get("kind") or plugin.get("source") or plugin.get("id"))
    specs = _normalize_tool_sources(metadata, plugin, fallback_module)
    return specs


def _determine_tool_registry_status(plugin_payload: Dict[str, Any]) -> str:
    if not bool(plugin_payload.get("enabled", True)):
        return "disabled"
    return _coerce_tool_status(plugin_payload.get("status"))


def _sync_plugin_tools(conn: sqlite3.Connection, plugin_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    now = _now_iso()
    plugin_id = _normalize_str(plugin_payload.get("id"))
    if not plugin_id:
        return []

    specs = _normalize_plugin_tools(plugin_payload)
    normalized_status = _determine_tool_registry_status(plugin_payload)
    current_tool_ids: set[str] = set()
    upserted: List[Dict[str, Any]] = []

    for spec in specs:
        spec_id = spec["id"]
        current_tool_ids.add(spec_id)
        existing = conn.execute("SELECT id, metadata, popularity, created_at FROM tool_registry WHERE id = ?", (spec_id,)).fetchone()
        popularity = int(existing["popularity"]) if existing and existing["popularity"] is not None else 0
        status = normalized_status if normalized_status in {"active", "disabled"} else spec["status"]
        row = {
            "id": spec_id,
            "plugin_id": plugin_id,
            "name": spec["name"],
            "namespace": spec["namespace"],
            "action": spec["action"],
            "module": spec.get("module"),
            "description": spec.get("description"),
            "signature": spec.get("signature"),
            "parameters": spec.get("parameters"),
            "container": int(bool(spec["container"])),
            "status": status if normalized_status == "active" else "disabled",
            "metadata": spec.get("metadata"),
            "updated_at": now,
        }
        if existing:
            conn.execute(
                """
                UPDATE tool_registry
                SET name=?, description=?, signature=?, parameters=?, container=?, status=?, metadata=?, updated_at=?
                WHERE id=?
                """,
                (
                    row["name"],
                    row["description"],
                    row["signature"],
                    row["parameters"],
                    row["container"],
                    row["status"],
                    row["metadata"],
                    row["updated_at"],
                    row["id"],
                ),
            )
            payload = {
                "id": row["id"],
                "plugin_id": plugin_id,
                "name": row["name"],
                "namespace": row["namespace"],
                "action": row["action"],
                "module": row["module"],
                "description": row["description"],
                "signature": row["signature"],
                "parameters": _coerce_json_payload(row["parameters"]),
                "container": bool(row["container"]),
                "status": row["status"],
                "metadata": _coerce_json_payload(row["metadata"]),
                "popularity": popularity,
                "created_at": existing["created_at"] or now,
                "updated_at": row["updated_at"],
            }
        else:
            conn.execute(
                """
                INSERT INTO tool_registry(
                    id,
                    plugin_id,
                    name,
                    namespace,
                    action,
                    module,
                    description,
                    signature,
                    parameters,
                    container,
                    popularity,
                    status,
                    metadata,
                    created_at,
                    updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["plugin_id"],
                    row["name"],
                    row["namespace"],
                    row["action"],
                    row["module"],
                    row["description"],
                    row["signature"],
                    row["parameters"],
                    row["container"],
                    row["status"],
                    row["metadata"],
                    now,
                    now,
                ),
            )
            payload = {
                "id": row["id"],
                "plugin_id": plugin_id,
                "name": row["name"],
                "namespace": row["namespace"],
                "action": row["action"],
                "module": row["module"],
                "description": row["description"],
                "signature": row["signature"],
                "parameters": _coerce_json_payload(row["parameters"]),
                "container": bool(row["container"]),
                "status": row["status"],
                "metadata": _coerce_json_payload(row["metadata"]),
                "popularity": 0,
                "created_at": now,
                "updated_at": now,
            }
        upserted.append(payload)

    if specs:
        placeholders = ",".join(["?"] * len(current_tool_ids))
        conn.execute(
            f"""
            UPDATE tool_registry
            SET status='retired', updated_at=?
            WHERE plugin_id=? AND id NOT IN ({placeholders}) AND status != 'retired'
            """,
            (now, plugin_id, *current_tool_ids),
        )
    else:
        conn.execute(
            """
            UPDATE tool_registry
            SET status='retired', updated_at=?
            WHERE plugin_id=? AND status != 'retired'
            """,
            (now, plugin_id),
        )

    conn.execute(
        "UPDATE tool_registry SET status=? WHERE plugin_id=? AND status='disabled'",
        (normalized_status, plugin_id),
    )
    conn.commit()
    return upserted


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
        _sync_plugin_tools(conn, payload)
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
        _sync_plugin_tools(conn, payload)
        payload["enabled"] = bool(payload["enabled"])
        payload["config"] = _coerce_json_payload(payload.get("config"))
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        return payload


def get_tool_registry(tool_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM tool_registry WHERE id = ?", (tool_id,)).fetchone()
        if not row:
            return None
        payload = _dict_from_row(row)
        payload["container"] = bool(payload["container"])
        payload["enabled"] = payload["status"] == "active"
        payload["popularity"] = int(payload["popularity"] or 0)
        payload["created_at"] = payload.get("created_at")
        payload["updated_at"] = payload.get("updated_at")
        payload["parameters"] = _coerce_json_payload(payload.get("parameters"))
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        return payload


def list_tools(
    q: Optional[str] = None,
    namespace: Optional[str] = None,
    module: Optional[str] = None,
    plugin_id: Optional[str] = None,
    status: Optional[str] = None,
    sort: str = "popularity",
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    normalized_limit = max(1, min(200, int(limit)))
    normalized_offset = max(0, int(offset))
    filters = []
    values: List[Any] = []
    if namespace and str(namespace).strip():
        filters.append("namespace = ?")
        values.append(str(namespace).strip())
    if module and str(module).strip():
        filters.append("module = ?")
        values.append(str(module).strip())
    if plugin_id and str(plugin_id).strip():
        filters.append("plugin_id = ?")
        values.append(str(plugin_id).strip())
    if status and str(status).strip():
        filters.append("status = ?")
        values.append(str(status).strip())
    if q and str(q).strip():
        normalized_q = f"%{str(q).strip().lower()}%"
        filters.append("(LOWER(name) LIKE ? OR LOWER(namespace) LIKE ? OR LOWER(action) LIKE ?)")
        values.extend([normalized_q, normalized_q, normalized_q])

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    normalized_sort = str(sort or "popularity").strip().lower()
    if normalized_sort not in {"popularity", "updated", "name", "namespace"}:
        normalized_sort = "popularity"
    if normalized_sort == "updated":
        order = "datetime(updated_at) DESC"
    elif normalized_sort == "name":
        order = "LOWER(namespace) ASC, LOWER(action) ASC"
    elif normalized_sort == "namespace":
        order = "LOWER(namespace) ASC, popularity DESC"
    else:
        order = "popularity DESC, datetime(updated_at) DESC"

    sql = f"SELECT * FROM tool_registry {where_clause} ORDER BY {order} LIMIT ? OFFSET ?"
    values.extend([normalized_limit, normalized_offset])
    with get_connection() as conn:
        rows = conn.execute(sql, tuple(values)).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["container"] = bool(payload["container"])
            payload["enabled"] = payload["status"] == "active"
            payload["popularity"] = int(payload["popularity"] or 0)
            payload["parameters"] = _coerce_json_payload(payload.get("parameters"))
            payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
            out.append(payload)
        return out


def count_tools(
    q: Optional[str] = None,
    namespace: Optional[str] = None,
    module: Optional[str] = None,
    plugin_id: Optional[str] = None,
    status: Optional[str] = None,
) -> int:
    filters = []
    values: List[Any] = []
    if namespace and str(namespace).strip():
        filters.append("namespace = ?")
        values.append(str(namespace).strip())
    if module and str(module).strip():
        filters.append("module = ?")
        values.append(str(module).strip())
    if plugin_id and str(plugin_id).strip():
        filters.append("plugin_id = ?")
        values.append(str(plugin_id).strip())
    if status and str(status).strip():
        filters.append("status = ?")
        values.append(str(status).strip())
    if q and str(q).strip():
        normalized_q = f"%{str(q).strip().lower()}%"
        filters.append("(LOWER(name) LIKE ? OR LOWER(namespace) LIKE ? OR LOWER(action) LIKE ?)")
        values.extend([normalized_q, normalized_q, normalized_q])

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT COUNT(1) AS total FROM tool_registry {where_clause}",
            tuple(values),
        ).fetchone()
        if not row:
            return 0
        return int(row["total"] or 0)


def search_tools(
    q: Optional[str] = None,
    namespace: Optional[str] = None,
    module: Optional[str] = None,
    plugin_id: Optional[str] = None,
    status: Optional[str] = None,
    match_mode: str = "contains",
    sort: str = "relevance",
    group_by: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    normalized_limit = max(1, min(200, int(limit)))
    normalized_offset = max(0, int(offset))
    normalized_match_mode = str(match_mode or "contains").strip().lower()
    if normalized_match_mode not in {"contains", "prefix", "wildcard", "regex", "exact"}:
        normalized_match_mode = "contains"

    normalized_sort = str(sort or "relevance").strip().lower()
    if normalized_sort not in {"relevance", "popularity", "updated", "name", "namespace"}:
        normalized_sort = "relevance"

    normalized_group = str(group_by or "").strip().lower()
    if normalized_group not in {"", "module", "plugin", "source"}:
        normalized_group = ""

    filters = []
    values: List[Any] = []

    if namespace and str(namespace).strip():
        filters.append("namespace = ?")
        values.append(str(namespace).strip())
    if module and str(module).strip():
        filters.append("module = ?")
        values.append(str(module).strip())
    if plugin_id and str(plugin_id).strip():
        filters.append("plugin_id = ?")
        values.append(str(plugin_id).strip())
    if status and str(status).strip():
        filters.append("status = ?")
        values.append(str(status).strip())

    query_text = str(q).strip().lower() if q else ""
    regex: Optional[re.Pattern[str]] = None

    def _escape_like(value: str) -> str:
        return (
            str(value)
            .replace("!", "!!")
            .replace("%", "!%")
            .replace("_", "!_")
        )

    wildcard_pattern = ""
    if query_text:
        if normalized_match_mode == "exact":
            filters.append("(LOWER(tr.name) = ? OR LOWER(tr.namespace) = ? OR LOWER(tr.action) = ?)")
            values.extend([query_text, query_text, query_text])
        elif normalized_match_mode == "prefix":
            wildcard_pattern = f"{_escape_like(query_text)}%"
        elif normalized_match_mode == "wildcard":
            wildcard_body = ""
            for ch in query_text:
                if ch == "*":
                    wildcard_body += "%"
                elif ch == "?":
                    wildcard_body += "_"
                else:
                    wildcard_body += _escape_like(ch)
            wildcard_pattern = wildcard_body if ("*" in query_text or "?" in query_text) else f"%{wildcard_body}%"
        elif normalized_match_mode == "regex":
            try:
                regex = re.compile(query_text, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
        else:
            wildcard_pattern = f"%{_escape_like(query_text)}%"

        if wildcard_pattern:
            filters.append(
                "(LOWER(tr.name) LIKE ? ESCAPE '!' OR LOWER(tr.namespace) LIKE ? ESCAPE '!' OR LOWER(tr.action) LIKE ? ESCAPE '!')"
            )
            values.extend([wildcard_pattern, wildcard_pattern, wildcard_pattern])

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    base_order = "popularity DESC, datetime(updated_at) DESC, LOWER(namespace) ASC, LOWER(action) ASC"
    from_clause = "FROM tool_registry tr"
    select_clause = "tr.*"
    if normalized_group == "source":
        from_clause = "FROM tool_registry tr LEFT JOIN plugins p ON p.id = tr.plugin_id"
        select_clause = "tr.*, p.source AS plugin_source"
    rows_sql = f"SELECT {select_clause} {from_clause} {where_clause} ORDER BY {base_order}"

    with get_connection() as conn:
        rows = conn.execute(rows_sql, tuple(values)).fetchall()

    candidates: List[Dict[str, Any]] = []
    normalized_query = query_text

    def _parse_updated_at(value: Optional[str]) -> float:
        if not value:
            return 0.0
        try:
            return datetime.fromisoformat(value).timestamp()
        except Exception:
            return 0.0

    def _is_match(payload: Dict[str, Any], matcher: Optional[re.Pattern[str]]) -> tuple[bool, float]:
        if not normalized_query:
            return True, 0.0

        haystacks = [
            str(payload.get("name") or "").lower(),
            str(payload.get("namespace") or "").lower(),
            str(payload.get("action") or "").lower(),
            str(payload.get("description") or "").lower(),
            str(payload.get("module") or "").lower(),
        ]

        if normalized_match_mode == "exact":
            is_match = any(item == normalized_query for item in haystacks[:3])
            return is_match, 100.0 if is_match else 0.0
        if normalized_match_mode == "prefix":
            is_match = any(item.startswith(normalized_query) for item in haystacks)
            return is_match, 95.0 if is_match else 0.0
        if normalized_match_mode == "wildcard":
            fn_pattern = (
                normalized_query if ("*" in query_text or "?" in query_text) else f"*{normalized_query}*"
            )
            is_match = any(fnmatch.fnmatch(item, fn_pattern) for item in haystacks)
            return is_match, 94.0 if is_match else 0.0
        if normalized_match_mode == "regex":
            if matcher is None:
                return False, 0.0
            is_match = any(bool(matcher.search(item)) for item in haystacks)
            return is_match, 94.0 if is_match else 0.0
        is_match = any(normalized_query in item for item in haystacks)
        return is_match, 90.0 if is_match else 0.0

    for row in rows:
        payload = _dict_from_row(row)
        payload["container"] = bool(payload["container"])
        payload["enabled"] = payload["status"] == "active"
        payload["popularity"] = int(payload["popularity"] or 0)
        payload["parameters"] = _coerce_json_payload(payload.get("parameters"))
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        payload["updated_at_epoch"] = _parse_updated_at(payload.get("updated_at"))
        payload["plugin_source"] = payload.pop("plugin_source", None)
        payload["relevance_score"] = 0.0

        is_match, match_score = _is_match(payload, regex)
        if not is_match:
            continue

        namespace_value = str(payload.get("namespace") or "").lower()
        action_value = str(payload.get("action") or "").lower()
        name_value = str(payload.get("name") or "").lower()
        module_value = str(payload.get("module") or "").lower()
        description_value = str(payload.get("description") or "").lower()

        score = match_score
        if normalized_query:
            score += (40.0 if namespace_value == normalized_query else 0.0)
            score += (30.0 if action_value == normalized_query else 0.0)
            score += (12.0 if name_value == normalized_query else 0.0)
            score += (12.0 if namespace_value.startswith(normalized_query) else 0.0)
            score += (8.0 if action_value.startswith(normalized_query) else 0.0)
            score += (4.0 if normalized_query in module_value else 0.0)
            score += (3.0 if normalized_query in name_value else 0.0)
            score += (2.0 if normalized_query in description_value else 0.0)

        payload["relevance_score"] = float(
            score + min(20.0, math.log1p(payload["popularity"]) * 4) + (payload["updated_at_epoch"] / 1e11)
        )
        candidates.append(payload)

    if normalized_sort == "relevance":
        candidates.sort(
            key=lambda item: (
                -item.get("relevance_score", 0.0),
                -int(item.get("popularity") or 0),
                -item.get("updated_at_epoch", 0.0),
                str(item.get("namespace") or ""),
                str(item.get("action") or ""),
                str(item.get("id") or ""),
            )
        )
    elif normalized_sort == "updated":
        candidates.sort(
            key=lambda item: (
                -item.get("updated_at_epoch", 0.0),
                str(item.get("namespace") or ""),
                str(item.get("action") or ""),
            )
        )
    elif normalized_sort == "name":
        candidates.sort(key=lambda item: (str(item.get("namespace") or ""), str(item.get("action") or "")))
    elif normalized_sort == "namespace":
        candidates.sort(key=lambda item: (str(item.get("module") or ""), str(item.get("namespace") or ""), str(item.get("action") or "")))
    else:
        candidates.sort(
            key=lambda item: (
                -int(item.get("popularity") or 0),
                -item.get("updated_at_epoch", 0.0),
                str(item.get("namespace") or ""),
                str(item.get("action") or ""),
            )
        )

    total = len(candidates)
    paged = candidates[normalized_offset : normalized_offset + normalized_limit]

    ranked = sorted(
        candidates,
        key=lambda item: (
            -int(item.get("popularity") or 0),
            -item.get("updated_at_epoch", 0.0),
            str(item.get("namespace") or ""),
            str(item.get("action") or ""),
            str(item.get("id") or ""),
        ),
    )
    popularity_rank_map = {str(item.get("id")): rank for rank, item in enumerate(ranked, start=1)}
    for entry in paged:
        entry["popularity_rank"] = popularity_rank_map.get(str(entry.get("id")), 0)
        score = entry.get("relevance_score")
        if isinstance(score, float):
            entry["relevance_score"] = round(score, 4)
        entry.pop("updated_at_epoch", None)

    groups: List[Dict[str, Any]] = []
    if normalized_group:
        grouped_totals: Dict[str, int] = {}
        grouped_tools: Dict[str, List[str]] = {}
        for candidate in candidates:
            if normalized_group == "module":
                group_key = str(candidate.get("module") or "default")
            elif normalized_group == "plugin":
                group_key = str(candidate.get("plugin_id") or "unbound")
            else:
                group_key = str(candidate.get("plugin_source") or "unbound")
            grouped_totals[group_key] = grouped_totals.get(group_key, 0) + 1
            grouped_tools.setdefault(group_key, []).append(str(candidate.get("id")))

        for group_key in sorted(grouped_totals.keys(), key=str):
            page_tools = [
                tool
                for tool in paged
                if str(
                    (
                        tool.get("module")
                        if normalized_group == "module"
                        else tool.get("plugin_id") if normalized_group == "plugin" else tool.get("plugin_source")
                    )
                    or ""
                ).lower()
                == group_key.lower()
            ]
            groups.append(
                {
                    "key": group_key,
                    "count": grouped_totals[group_key],
                    "tool_ids": grouped_tools[group_key],
                    "tools": [tool["id"] for tool in page_tools],
                }
            )

    for candidate in candidates:
        candidate.pop("plugin_source", None)
        candidate.pop("updated_at_epoch", None)

    return paged, total, groups


def append_tool_usage(
    tool_id: str,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    request_id: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    status: str = "executed",
    duration_ms: Optional[int] = None,
    result_status: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sanitized_tool_id = _normalize_str(tool_id)
    if not sanitized_tool_id:
        raise KeyError("tool id is required")
    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM tool_registry WHERE id = ?", (sanitized_tool_id,)).fetchone()
        if not existing:
            raise KeyError(f"tool '{tool_id}' not found")

        now = _now_iso()
        usage_id = f"tool-use-{uuid.uuid4().hex[:12]}"
        normalized_context_type = _normalize_str(context_type) or None
        normalized_context_id = _normalize_str(context_id) or None
        normalized_memory_id = _normalize_str(memory_id)
        normalized_chat_message_id = _normalize_str(chat_message_id)
        normalized_chat_session_id = _normalize_str(chat_session_id)

        if normalized_memory_id is None and normalized_context_type == "memory":
            normalized_memory_id = normalized_context_id
        if normalized_chat_message_id is None and normalized_context_type == "chat":
            normalized_chat_message_id = normalized_context_id
        if normalized_chat_session_id is None and normalized_context_type == "chat":
            normalized_chat_session_id = _normalize_str(session_id)

        conn.execute(
            """
            INSERT INTO tool_usage(
                id,
                tool_id,
                agent_id,
                session_id,
                mission_id,
                request_id,
                context_type,
                context_id,
                memory_id,
                chat_message_id,
                chat_session_id,
                status,
                duration_ms,
                result_status,
                details,
                payload,
                created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                usage_id,
                sanitized_tool_id,
                _normalize_str(agent_id) or None,
                _normalize_str(session_id) or None,
                _normalize_str(mission_id) or None,
                _normalize_str(request_id) or None,
                normalized_context_type,
                normalized_context_id,
                normalized_memory_id,
                normalized_chat_message_id,
                normalized_chat_session_id,
                _normalize_str(status) or "executed",
                int(duration_ms) if duration_ms is not None else None,
                _normalize_str(result_status) or None,
                _coerce_metadata_for_storage(details),
                _coerce_metadata_for_storage(payload),
                now,
            ),
        )
        if status:
            conn.execute(
                "UPDATE tool_registry SET popularity = popularity + 1, updated_at = ? WHERE id = ?",
                (now, sanitized_tool_id),
            )
        conn.commit()

    usage = {
        "id": usage_id,
        "tool_id": sanitized_tool_id,
        "agent_id": _normalize_str(agent_id) or None,
        "session_id": _normalize_str(session_id) or None,
        "mission_id": _normalize_str(mission_id) or None,
        "request_id": _normalize_str(request_id) or None,
        "context_type": normalized_context_type,
        "context_id": normalized_context_id,
        "memory_id": normalized_memory_id,
        "chat_message_id": normalized_chat_message_id,
        "chat_session_id": normalized_chat_session_id,
        "status": _normalize_str(status) or "executed",
        "duration_ms": int(duration_ms) if duration_ms is not None else None,
        "result_status": _normalize_str(result_status) or None,
        "details": details if isinstance(details, dict) else _coerce_json_payload(_coerce_metadata_for_storage(details)),
        "payload": payload if isinstance(payload, dict) else _coerce_json_payload(_coerce_metadata_for_storage(payload)),
        "created_at": now,
    }
    usage["details"] = _coerce_json_payload(_coerce_metadata_for_storage(usage["details"]))
    usage["payload"] = _coerce_json_payload(_coerce_metadata_for_storage(usage["payload"]))
    return usage


def _collect_tool_usage_rows(
    conn: sqlite3.Connection,
    where_clause: str,
    values: List[Any],
    sort_desc: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    normalized_limit = max(1, min(200, int(limit)))
    normalized_offset = max(0, int(offset))
    order = "datetime(created_at) DESC" if sort_desc else "datetime(created_at) ASC"
    rows = conn.execute(
        f"SELECT * FROM tool_usage {where_clause} ORDER BY {order} LIMIT ? OFFSET ?",
        tuple(values + [normalized_limit, normalized_offset]),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for usage in rows:
        payload = _dict_from_row(usage)
        payload["duration_ms"] = int(payload["duration_ms"]) if payload["duration_ms"] is not None else None
        payload["details"] = _coerce_json_payload(payload.get("details"))
        payload["payload"] = _coerce_json_payload(payload.get("payload"))
        out.append(payload)
    return out


def _append_tool_usage_date_filter(
    filters: List[str],
    values: List[Any],
    date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> None:
    normalized_date = _normalize_day_filter(date)
    if normalized_date:
        filters.append("date(created_at) = ?")
        values.append(normalized_date)
        return

    normalized_from = _normalize_day_filter(date_from)
    if normalized_from:
        filters.append("datetime(created_at) >= datetime(?)")
        values.append(f"{normalized_from}T00:00:00Z")

    normalized_to = _normalize_day_filter(date_to)
    if normalized_to:
        filters.append("datetime(created_at) < datetime(?)")
        values.append(f"{normalized_to}T23:59:59Z")


def list_tool_usage(
    tool_id: str,
    agent_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    sort_desc: bool = True,
) -> List[Dict[str, Any]]:
    normalized_tool_id = _normalize_str(tool_id)
    if not normalized_tool_id:
        return []
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM tool_registry WHERE id = ?", (normalized_tool_id,)).fetchone()
        if not row:
            raise KeyError(f"tool '{tool_id}' not found")

        filters = ["tool_id = ?"]
        values: List[Any] = [normalized_tool_id]
        if agent_id:
            normalized_agent_id = _normalize_str(agent_id)
            if normalized_agent_id:
                filters.append("agent_id = ?")
                values.append(normalized_agent_id)
        if context_type:
            normalized_context_type = _normalize_str(context_type)
            if normalized_context_type:
                filters.append("context_type = ?")
                values.append(normalized_context_type)
        if context_id:
            normalized_context_id = _normalize_str(context_id)
            if normalized_context_id:
                filters.append("context_id = ?")
                values.append(normalized_context_id)
        if memory_id:
            normalized_memory_id = _normalize_str(memory_id)
            if normalized_memory_id:
                filters.append("memory_id = ?")
                values.append(normalized_memory_id)
        if chat_message_id:
            normalized_chat_message_id = _normalize_str(chat_message_id)
            if normalized_chat_message_id:
                filters.append("chat_message_id = ?")
                values.append(normalized_chat_message_id)
        if chat_session_id:
            normalized_chat_session_id = _normalize_str(chat_session_id)
            if normalized_chat_session_id:
                filters.append("chat_session_id = ?")
                values.append(normalized_chat_session_id)
        if mission_id:
            normalized_mission_id = _normalize_str(mission_id)
            if normalized_mission_id:
                filters.append("mission_id = ?")
                values.append(normalized_mission_id)
        if session_id:
            normalized_session_id = _normalize_str(session_id)
            if normalized_session_id:
                filters.append("session_id = ?")
                values.append(normalized_session_id)
        _append_tool_usage_date_filter(filters, values, date=date, date_from=date_from, date_to=date_to)

        return _collect_tool_usage_rows(
            conn=conn,
            where_clause=f"WHERE {' AND '.join(filters)}",
            values=values,
            sort_desc=sort_desc,
            limit=limit,
            offset=offset,
        )


def list_tool_usage_global(
    tool_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    sort_desc: bool = True,
) -> List[Dict[str, Any]]:
    filters = []
    values: List[Any] = []
    with get_connection() as conn:
        if tool_id:
            normalized_tool_id = _normalize_str(tool_id)
            if normalized_tool_id:
                row = conn.execute("SELECT id FROM tool_registry WHERE id = ?", (normalized_tool_id,)).fetchone()
                if not row:
                    raise KeyError(f"tool '{tool_id}' not found")
                filters.append("tool_id = ?")
                values.append(normalized_tool_id)
        if agent_id:
            normalized_agent_id = _normalize_str(agent_id)
            if normalized_agent_id:
                filters.append("agent_id = ?")
                values.append(normalized_agent_id)
        if session_id:
            normalized_session_id = _normalize_str(session_id)
            if normalized_session_id:
                filters.append("session_id = ?")
                values.append(normalized_session_id)
        if context_type:
            normalized_context_type = _normalize_str(context_type)
            if normalized_context_type:
                filters.append("context_type = ?")
                values.append(normalized_context_type)
        if context_id:
            normalized_context_id = _normalize_str(context_id)
            if normalized_context_id:
                filters.append("context_id = ?")
                values.append(normalized_context_id)
        if memory_id:
            normalized_memory_id = _normalize_str(memory_id)
            if normalized_memory_id:
                filters.append("memory_id = ?")
                values.append(normalized_memory_id)
        if chat_message_id:
            normalized_chat_message_id = _normalize_str(chat_message_id)
            if normalized_chat_message_id:
                filters.append("chat_message_id = ?")
                values.append(normalized_chat_message_id)
        if chat_session_id:
            normalized_chat_session_id = _normalize_str(chat_session_id)
            if normalized_chat_session_id:
                filters.append("chat_session_id = ?")
                values.append(normalized_chat_session_id)
        if mission_id:
            normalized_mission_id = _normalize_str(mission_id)
            if normalized_mission_id:
                filters.append("mission_id = ?")
                values.append(normalized_mission_id)
        _append_tool_usage_date_filter(filters, values, date=date, date_from=date_from, date_to=date_to)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        return _collect_tool_usage_rows(
            conn=conn,
            where_clause=where_clause,
            values=values,
            sort_desc=sort_desc,
            limit=limit,
            offset=offset,
        )


def count_tool_usage(
    tool_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> int:
    filters = []
    values: List[Any] = []
    with get_connection() as conn:
        if tool_id:
            normalized_tool_id = _normalize_str(tool_id)
            if normalized_tool_id:
                row = conn.execute("SELECT id FROM tool_registry WHERE id = ?", (normalized_tool_id,)).fetchone()
                if not row:
                    raise KeyError(f"tool '{tool_id}' not found")
                filters.append("tool_id = ?")
                values.append(normalized_tool_id)
        if agent_id:
            normalized_agent_id = _normalize_str(agent_id)
            if normalized_agent_id:
                filters.append("agent_id = ?")
                values.append(normalized_agent_id)
        if session_id:
            normalized_session_id = _normalize_str(session_id)
            if normalized_session_id:
                filters.append("session_id = ?")
                values.append(normalized_session_id)
        if context_type:
            normalized_context_type = _normalize_str(context_type)
            if normalized_context_type:
                filters.append("context_type = ?")
                values.append(normalized_context_type)
        if context_id:
            normalized_context_id = _normalize_str(context_id)
            if normalized_context_id:
                filters.append("context_id = ?")
                values.append(normalized_context_id)
        if memory_id:
            normalized_memory_id = _normalize_str(memory_id)
            if normalized_memory_id:
                filters.append("memory_id = ?")
                values.append(normalized_memory_id)
        if chat_message_id:
            normalized_chat_message_id = _normalize_str(chat_message_id)
            if normalized_chat_message_id:
                filters.append("chat_message_id = ?")
                values.append(normalized_chat_message_id)
        if chat_session_id:
            normalized_chat_session_id = _normalize_str(chat_session_id)
            if normalized_chat_session_id:
                filters.append("chat_session_id = ?")
                values.append(normalized_chat_session_id)
        if mission_id:
            normalized_mission_id = _normalize_str(mission_id)
            if normalized_mission_id:
                filters.append("mission_id = ?")
                values.append(normalized_mission_id)
        _append_tool_usage_date_filter(filters, values, date=date, date_from=date_from, date_to=date_to)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        row = conn.execute(f"SELECT COUNT(1) AS total FROM tool_usage {where_clause}", tuple(values)).fetchone()
        if not row:
            return 0
        return int(row["total"] or 0)


def get_agent_tool_audit(
    agent_id: str,
    mission_id: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_limit: int = 5,
    mission_limit: int = 5,
    session_limit: int = 5,
    recent_limit: int = 25,
) -> Dict[str, Any]:
    normalized_agent_id = _normalize_str(agent_id)
    if not normalized_agent_id:
        raise KeyError("agent id is required")

    normalized_tool_limit = max(1, min(50, int(tool_limit)))
    normalized_session_limit = max(1, min(50, int(session_limit)))
    normalized_recent_limit = max(1, min(200, int(recent_limit)))

    normalized_mission_id = _normalize_str(mission_id)
    normalized_context_type = _normalize_str(context_type)
    normalized_context_id = _normalize_str(context_id)
    normalized_memory_id = _normalize_str(memory_id)
    normalized_chat_message_id = _normalize_str(chat_message_id)
    normalized_chat_session_id = _normalize_str(chat_session_id)
    normalized_session_id = _normalize_str(session_id)

    with get_connection() as conn:
        base_filter = ["agent_id = ?"]
        values: List[Any] = [normalized_agent_id]
        if normalized_mission_id:
            base_filter.append("mission_id = ?")
            values.append(normalized_mission_id)
        if normalized_context_type:
            base_filter.append("context_type = ?")
            values.append(normalized_context_type)
        if normalized_context_id:
            base_filter.append("context_id = ?")
            values.append(normalized_context_id)
        if normalized_memory_id:
            base_filter.append("memory_id = ?")
            values.append(normalized_memory_id)
        if normalized_chat_message_id:
            base_filter.append("chat_message_id = ?")
            values.append(normalized_chat_message_id)
        if normalized_chat_session_id:
            base_filter.append("chat_session_id = ?")
            values.append(normalized_chat_session_id)
        if normalized_session_id:
            base_filter.append("session_id = ?")
            values.append(normalized_session_id)

        where_clause = f"WHERE {' AND '.join(base_filter)}"

        summary = conn.execute(
            f"SELECT COUNT(1) AS total_tool_calls, COUNT(DISTINCT tool_id) AS unique_tools "
            f"FROM tool_usage {where_clause}",
            tuple(values),
        ).fetchone()

        tool_breakdown = conn.execute(
            f"SELECT tool_id, COUNT(1) AS call_count, MAX(datetime(created_at)) AS last_used_at "
            f"FROM tool_usage {where_clause} "
            f"GROUP BY tool_id ORDER BY call_count DESC, datetime(last_used_at) DESC LIMIT ?",
            tuple(values + [normalized_tool_limit]),
        ).fetchall()

        mission_breakdown = conn.execute(
            f"SELECT mission_id, COUNT(1) AS call_count, "
            f"COUNT(DISTINCT CASE WHEN session_id IS NOT NULL AND TRIM(session_id) <> '' THEN session_id END) AS session_count, "
            f"MAX(datetime(created_at)) AS last_used_at "
            f"FROM tool_usage {where_clause} AND mission_id IS NOT NULL AND TRIM(mission_id) <> '' "
            f"GROUP BY mission_id ORDER BY call_count DESC, datetime(last_used_at) DESC LIMIT ?",
            tuple(values + [max(1, min(50, int(mission_limit)))]),
        ).fetchall()

        session_breakdown = conn.execute(
            f"SELECT session_id, COUNT(1) AS call_count, MAX(datetime(created_at)) AS last_used_at "
            f"FROM tool_usage {where_clause} AND session_id IS NOT NULL AND TRIM(session_id) <> '' "
            f"GROUP BY session_id ORDER BY call_count DESC, datetime(last_used_at) DESC LIMIT ?",
            tuple(values + [normalized_session_limit]),
        ).fetchall()

        recent_usage = conn.execute(
            f"SELECT * FROM tool_usage {where_clause} "
            f"ORDER BY datetime(created_at) DESC LIMIT ?",
            tuple(values + [normalized_recent_limit]),
        ).fetchall()

    summary_payload = {
        "total_tool_calls": int(summary["total_tool_calls"] or 0) if summary else 0,
        "unique_tools": int(summary["unique_tools"] or 0) if summary else 0,
        "unique_missions": len(mission_breakdown),
    }
    return {
        "agent_id": normalized_agent_id,
        "filters": {
            "mission_id": normalized_mission_id,
            "context_type": normalized_context_type,
            "context_id": normalized_context_id,
            "memory_id": normalized_memory_id,
            "chat_message_id": normalized_chat_message_id,
            "chat_session_id": normalized_chat_session_id,
            "session_id": normalized_session_id,
        },
        "summary": summary_payload,
        "tool_breakdown": [
            {
                "tool_id": row["tool_id"],
                "call_count": int(row["call_count"] or 0),
                "last_used_at": row["last_used_at"],
            }
            for row in tool_breakdown
        ],
        "mission_breakdown": [
            {
                "mission_id": row["mission_id"],
                "call_count": int(row["call_count"] or 0),
                "session_count": int(row["session_count"] or 0),
                "last_used_at": row["last_used_at"],
            }
            for row in mission_breakdown
        ],
        "session_breakdown": [
            {
                "session_id": row["session_id"],
                "call_count": int(row["call_count"] or 0),
                "last_used_at": row["last_used_at"],
            }
            for row in session_breakdown
        ],
        "recent_calls": [
            {
                **_dict_from_row(entry),
                "duration_ms": int(entry["duration_ms"]) if entry["duration_ms"] is not None else None,
                "details": _coerce_json_payload(entry["details"]),
                "payload": _coerce_json_payload(entry["payload"]),
            }
            for entry in recent_usage
        ],
    }


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


def list_agents(lifecycle: Optional[str] = AGENT_LIFECYCLE_ALL) -> List[Dict]:
    normalized_lifecycle = _normalize_agent_lifecycle(lifecycle)
    with get_connection() as conn:
        query = "SELECT * FROM agents"
        args: List[Any] = []
        if normalized_lifecycle in {AGENT_LIFECYCLE_ACTIVE, AGENT_LIFECYCLE_ARCHIVED}:
            query += " WHERE lifecycle = ?"
            args.append(normalized_lifecycle)
        query += " ORDER BY name ASC"
        rows = conn.execute(query, args).fetchall()
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
        payload["lifecycle"] = _normalize_agent_lifecycle(values.get("lifecycle", row.get("lifecycle")))
        payload["archived_at"] = payload.get("archived_at", row.get("archived_at"))
        if values.get("enabled") is False:
            payload["status"] = "paused"
        elif values.get("enabled") is True:
            payload["status"] = "running"
        if payload["lifecycle"] == AGENT_LIFECYCLE_ARCHIVED:
            payload["enabled"] = 0
            payload["status"] = "archived"
            payload["archived_at"] = payload.get("archived_at") or _now_iso()
        elif payload["lifecycle"] == AGENT_LIFECYCLE_ACTIVE and row.get("lifecycle") == AGENT_LIFECYCLE_ARCHIVED:
            payload["archived_at"] = None
        payload["last_active_at"] = _now_iso()
        now = _now_iso()
        conn.execute(
            """
            UPDATE agents
            SET name=?, enabled=?, status=?, role=?, last_active_at=?, config=?, lifecycle=?, archived_at=?, updated_at=?
            WHERE id=?
            """,
            (
                payload["name"],
                payload["enabled"],
                payload["status"],
                payload["role"],
                payload["last_active_at"],
                payload.get("config"),
                payload["lifecycle"],
                payload.get("archived_at"),
                now,
                agent_id,
            ),
        )
        conn.commit()
        payload["enabled"] = bool(payload["enabled"])
        return payload


def archive_agent(agent_id: str, *, archived_at: Optional[str] = None) -> Dict:
    now = _now_iso()
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not existing:
            raise KeyError(f"agent '{agent_id}' not found")
        row = _dict_from_row(existing)
        payload = row | {"lifecycle": AGENT_LIFECYCLE_ARCHIVED}
        payload["archived_at"] = archived_at or now
        payload["enabled"] = 0
        payload["status"] = row.get("status") if str(row.get("status", "")).strip() == "archived" else "archived"
        payload["last_active_at"] = now
        conn.execute(
            """
            UPDATE agents
            SET lifecycle=?, archived_at=?, enabled=?, status=?, last_active_at=?, updated_at=?
            WHERE id=?
            """,
            (
                AGENT_LIFECYCLE_ARCHIVED,
                payload["archived_at"],
                payload["enabled"],
                payload["status"],
                now,
                now,
                agent_id,
            ),
        )
        conn.commit()
        payload["enabled"] = bool(payload["enabled"])
        payload["status"] = payload["status"]
        payload["lifecycle"] = AGENT_LIFECYCLE_ARCHIVED
        return payload


def restore_agent(agent_id: str) -> Dict:
    now = _now_iso()
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if not existing:
            raise KeyError(f"agent '{agent_id}' not found")
        row = _dict_from_row(existing)
        if row.get("lifecycle") != AGENT_LIFECYCLE_ARCHIVED:
            payload = dict(row)
            payload["enabled"] = bool(payload.get("enabled"))
            payload["lifecycle"] = _normalize_agent_lifecycle(row.get("lifecycle"))
            return payload
        payload = row
        payload["lifecycle"] = AGENT_LIFECYCLE_ACTIVE
        payload["archived_at"] = None
        payload["enabled"] = 1
        payload["status"] = "running"
        payload["last_active_at"] = now
        conn.execute(
            """
            UPDATE agents
            SET lifecycle=?, archived_at=?, enabled=?, status=?, last_active_at=?, updated_at=?
            WHERE id=?
            """,
            (
                AGENT_LIFECYCLE_ACTIVE,
                None,
                1,
                "running",
                now,
                now,
                agent_id,
            ),
        )
        conn.commit()
        payload["enabled"] = True
        payload["lifecycle"] = AGENT_LIFECYCLE_ACTIVE
        payload["status"] = "running"
        return payload


def list_events(limit: int = 25, *, gm_ticket_id: Optional[str] = None) -> List[Dict]:
    normalized_limit = max(1, min(limit, 100))
    normalized_ticket_id = str(gm_ticket_id).strip() if gm_ticket_id is not None else ""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM events ORDER BY datetime(created_at) DESC LIMIT ?",
            (1000,),
        ).fetchall()

    out = []
    for row in rows:
        payload = _dict_from_row(row)
        parsed_payload = _coerce_json_payload(payload.get("payload"))
        payload["payload"] = parsed_payload

        if normalized_ticket_id:
            event_payload = parsed_payload
            if not isinstance(event_payload, dict):
                continue
            if str(event_payload.get("gm_ticket_id") or "").strip() != normalized_ticket_id:
                continue

        out.append(payload)
        if len(out) >= normalized_limit:
            break

    return out


def _normalize_gm_ticket_status(value: Optional[str]) -> str:
    normalized = str(value or "open").strip().lower()
    if not normalized:
        normalized = "open"
    if normalized not in _GOTICKET_STATUSES:
        raise ValueError(f"unsupported ticket status: {value!r}")
    return normalized


def _validate_gm_status_transition(current_status: str, requested_status: str) -> None:
    normalized_current = _normalize_gm_ticket_status(current_status)
    normalized_requested = _normalize_gm_ticket_status(requested_status)
    allowed = _GOTICKET_STATUS_TRANSITIONS.get(normalized_current, set())
    if normalized_requested == normalized_current:
        return
    if normalized_requested not in allowed:
        raise ValueError(
            f"invalid GM ticket status transition: {normalized_current!r} -> {normalized_requested!r}"
        )


def _normalize_gm_ticket_priority(value: Optional[str]) -> str:
    normalized = str(value or "normal").strip().lower()
    if not normalized:
        normalized = "normal"
    if normalized not in _GOTICKET_PRIORITIES:
        raise ValueError(f"unsupported ticket priority: {value!r}")
    return normalized


def create_gm_ticket(values: Dict[str, Any]) -> Dict[str, Any]:
    title = str(values.get("title", "")).strip()
    if not title:
        raise ValueError("title is required")

    ticket_id = str(values.get("id") or f"gm-ticket-{uuid.uuid4().hex[:12]}").strip()
    if not ticket_id:
        raise ValueError("ticket id is required")

    status = _normalize_gm_ticket_status(values.get("status"))
    priority = _normalize_gm_ticket_priority(values.get("priority"))
    agent_scope = str(values.get("agent_scope", "global")).strip() or "global"
    phase = str(values.get("phase", "active")).strip() or "active"
    requested_by = str(values.get("requested_by", "")).strip()
    if not requested_by:
        raise ValueError("requested_by is required")
    assigned_to = str(values.get("assigned_to", "")).strip() or "unassigned"

    now = _now_iso()
    with get_connection() as conn:
        existing = conn.execute("SELECT 1 FROM gm_tickets WHERE id = ?", (ticket_id,)).fetchone()
        if existing:
            raise ValueError(f"ticket '{ticket_id}' already exists")

        metadata = _coerce_metadata_for_storage(values.get("metadata"))
        conn.execute(
            """
            INSERT INTO gm_tickets(
                id,
                title,
                status,
                priority,
                agent_scope,
                phase,
                requested_by,
                assigned_to,
                metadata,
                created_at,
                updated_at,
                closed_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                title,
                status,
                priority,
                agent_scope,
                phase,
                requested_by,
                assigned_to,
                metadata,
                now,
                now,
                None,
            ),
        )
        conn.commit()

    payload = get_gm_ticket(ticket_id)
    if payload is None:
        raise RuntimeError(f"failed to create ticket '{ticket_id}'")
    return payload


def list_gm_tickets(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    assigned_to: Optional[str] = None,
    phase: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    normalized_limit = max(1, min(1000, int(limit)))
    normalized_offset = max(0, int(offset))
    filters: List[str] = []
    args: List[Any] = []

    if status is not None:
        normalized_status = _normalize_gm_ticket_status(status)
        filters.append("status = ?")
        args.append(normalized_status)
    if priority is not None:
        filters.append("priority = ?")
        args.append(_normalize_gm_ticket_priority(priority))
    if assigned_to is not None:
        normalized_assigned_to = str(assigned_to).strip()
        if normalized_assigned_to:
            filters.append("assigned_to = ?")
            args.append(normalized_assigned_to)
    if phase is not None:
        normalized_phase = str(phase).strip()
        if normalized_phase:
            filters.append("phase = ?")
            args.append(normalized_phase)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM gm_tickets {where_clause} ORDER BY datetime(updated_at) DESC LIMIT ? OFFSET ?",
            tuple(args + [normalized_limit, normalized_offset]),
        ).fetchall()

    out: List[Dict[str, Any]] = []
    for row in rows:
        payload = _dict_from_row(row)
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        out.append(payload)
    return out


def get_gm_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    normalized_ticket_id = str(ticket_id).strip()
    if not normalized_ticket_id:
        return None

    with get_connection() as conn:
        row = conn.execute("SELECT * FROM gm_tickets WHERE id = ?", (normalized_ticket_id,)).fetchone()
        if not row:
            return None
        payload = _dict_from_row(row)
        payload["metadata"] = _coerce_json_payload(payload.get("metadata"))
        return payload


def update_gm_ticket(ticket_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
    normalized_ticket_id = str(ticket_id).strip()
    if not normalized_ticket_id:
        raise ValueError("ticket id is required")

    payload: Dict[str, Any] = {}
    title = values.get("title")
    if title is not None:
        payload["title"] = str(title).strip()
        if not payload["title"]:
            raise ValueError("title cannot be empty")

    if values.get("status") is not None:
        payload["status"] = _normalize_gm_ticket_status(values.get("status"))

    if values.get("priority") is not None:
        payload["priority"] = _normalize_gm_ticket_priority(values.get("priority"))

    if values.get("assigned_to") is not None:
        payload["assigned_to"] = str(values.get("assigned_to", "")).strip() or "unassigned"

    if values.get("agent_scope") is not None:
        payload["agent_scope"] = str(values.get("agent_scope", "")).strip() or "global"

    if values.get("phase") is not None:
        payload["phase"] = str(values.get("phase", "")).strip()

    if values.get("metadata") is not None:
        payload["metadata"] = _coerce_metadata_for_storage(values.get("metadata"))

    if values.get("closed_at") is not None:
        payload["closed_at"] = str(values.get("closed_at")).strip() or None

    if not payload:
        return get_gm_ticket(ticket_id)  # type: ignore[return-value]

    now = _now_iso()
    updated_status = payload.get("status")
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM gm_tickets WHERE id = ?", (normalized_ticket_id,)).fetchone()
        if not existing:
            raise KeyError(f"ticket '{normalized_ticket_id}' not found")

        if updated_status is not None:
            existing_status = _normalize_gm_ticket_status(existing["status"])
            _validate_gm_status_transition(existing_status, updated_status)

        if updated_status in _GOTICKET_CLOSED_STATUSES:
            payload.setdefault("closed_at", now)
        elif updated_status in _GOTICKET_STATUSES:
            payload["closed_at"] = None

        updates = [f"{key} = ?" for key in payload]
        updates.append("updated_at = ?")
        args = [payload[key] for key in payload]
        args.append(now)
        args.append(normalized_ticket_id)

        conn.execute(
            f"UPDATE gm_tickets SET {', '.join(updates)} WHERE id = ?",
            tuple(args),
        )
        conn.commit()

    updated = get_gm_ticket(normalized_ticket_id)
    if updated is None:
        raise RuntimeError(f"failed to read ticket '{normalized_ticket_id}'")
    return updated


def append_gm_ticket_message(
    ticket_id: str,
    sender: str,
    content: str,
    message_type: str = "comment",
    response_required: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_ticket_id = str(ticket_id).strip()
    if not normalized_ticket_id:
        raise ValueError("ticket id is required")
    normalized_sender = str(sender).strip()
    if not normalized_sender:
        raise ValueError("sender is required")
    normalized_content = str(content).strip()
    if not normalized_content:
        raise ValueError("content is required")

    normalized_message_type = str(message_type).strip() or "comment"
    if normalized_message_type not in {"comment", "status", "request"}:
        raise ValueError(f"unsupported message_type: {message_type!r}")
    normalized_response_required = _coerce_bool(bool(response_required))

    now = _now_iso()
    message_id = f"gm-msg-{uuid.uuid4().hex[:12]}"
    with get_connection() as conn:
        existing = conn.execute("SELECT 1 FROM gm_tickets WHERE id = ?", (normalized_ticket_id,)).fetchone()
        if not existing:
            raise KeyError(f"ticket '{normalized_ticket_id}' not found")

        conn.execute(
            """
            INSERT INTO gm_ticket_messages(
                id,
                gm_ticket_id,
                sender,
                content,
                message_type,
                response_required,
                metadata,
                created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                normalized_ticket_id,
                normalized_sender,
                normalized_content,
                normalized_message_type,
                normalized_response_required,
                _coerce_metadata_for_storage(metadata),
                now,
            ),
        )
        conn.commit()

    row = get_gm_ticket(normalized_ticket_id)
    if row is None:
        raise RuntimeError(f"ticket '{normalized_ticket_id}' not found")
    return {
        "id": message_id,
        "gm_ticket_id": normalized_ticket_id,
        "sender": normalized_sender,
        "content": normalized_content,
        "message_type": normalized_message_type,
        "response_required": bool(normalized_response_required),
        "metadata": metadata,
        "created_at": now,
    }


def list_gm_ticket_messages(ticket_id: str) -> List[Dict[str, Any]]:
    normalized_ticket_id = str(ticket_id).strip()
    if not normalized_ticket_id:
        return []

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM gm_ticket_messages WHERE gm_ticket_id = ? ORDER BY datetime(created_at) ASC",
            (normalized_ticket_id,),
        ).fetchall()

    return [
        {
            **_dict_from_row(row),
            "response_required": bool(row["response_required"]),
            "metadata": _coerce_json_payload(row["metadata"]),
        }
        for row in rows
    ]


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

        visibility_by_state = {
            "approved": "visible",
            "pending": "visible",
            "quarantined": "quarantined",
            "rejected": "quarantined",
        }
        for item in candidate_rows:
            item_id = item["id"]
            if item["id"] in updated_ids:
                continue
            scope_before = str(item.get("agent_scope") or "").strip()
            scope_after = normalized_agent_scope if normalized_agent_scope is not None else scope_before
            visibility = visibility_by_state.get(review_state, "quarantined")

            if normalized_agent_scope is None:
                conn.execute(
                    "UPDATE import_items SET review_state = ?, visibility = ? WHERE id = ?",
                    (review_state, visibility, item_id),
                )
            else:
                conn.execute(
                    "UPDATE import_items SET review_state = ?, visibility = ?, agent_scope = ? WHERE id = ?",
                    (review_state, visibility, scope_after, item_id),
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
                "visibility_after": visibility,
                },
                db_connection=conn,
            )
            updated.append(
                {
                    **item,
                    "review_state": review_state,
                    "visibility": visibility,
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


def list_memory(
    scope: Optional[str],
    item_type: Optional[str],
    item_id: Optional[str] = None,
) -> List[Dict]:
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
        if item_id:
            clauses.append("id = ?")
            args.append(item_id)
        if clauses:
            query += f" WHERE {' AND '.join(clauses)}"
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


def list_chat_history(
    agent_id: Optional[str] = None,
    item_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
) -> List[Dict]:
    filters: list[str] = []
    args: list[str] = []
    if agent_id:
        filters.append("agent_id = ?")
        args.append(agent_id)
    if item_id:
        filters.append("id = ?")
        args.append(item_id)
    if chat_session_id:
        filters.append("chat_session_id = ?")
        args.append(chat_session_id)
    where = f" WHERE {' AND '.join(filters)}" if filters else ""
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM chat_history{where} ORDER BY datetime(timestamp) DESC",
            tuple(args),
        ).fetchall()
        return [_dict_from_row(row) for row in rows]


def add_chat_entry(agent_id: str, summary: str, chat_session_id: Optional[str] = None) -> Dict:
    chat_id = f"chat-{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    payload = {
        "id": chat_id,
        "agent_id": agent_id,
        "summary": summary,
        "timestamp": now,
        "chat_session_id": _normalize_str(chat_session_id),
    }
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history(id, agent_id, summary, timestamp, chat_session_id) VALUES(?, ?, ?, ?, ?)",
            (chat_id, agent_id, summary, now, _normalize_str(chat_session_id)),
        )
        conn.commit()
    return payload
