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


def append_import_progress_event(import_id: str, phase: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        payload["source_metadata"] = _coerce_json_payload(payload.get("source_metadata")) or {}
        return payload


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
) -> List[Dict[str, Any]]:
    valid_states = {"pending", "approved", "quarantined", "rejected"}
    if review_state not in valid_states:
        raise ValueError(f"unsupported review state: {review_state}")

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
            conn.execute("UPDATE import_items SET review_state = ? WHERE id = ?", (review_state, item_id))
            appended = append_import_item_review_event(
                import_item_id=item_id,
                event_type="import.item.reviewed",
                review_state=review_state,
                actor=actor,
                note=note,
                payload={"job_id": item["import_id"]},
                db_connection=conn,
            )
            updated.append(
                {
                    **item,
                    "review_state": review_state,
                    "metadata": _coerce_json_payload(item.get("metadata")) or {},
                    "event": appended,
                }
            )
            updated_ids.add(item_id)
        conn.commit()

    return updated
def create_import_job(
    source_type: str,
    source_metadata: Optional[Dict[str, Any]],
    checksum: str,
    status: str = "pending",
) -> tuple[Dict[str, Any], bool]:
    now = _now_iso()
    metadata = source_metadata or {}
    with get_connection() as conn:
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
            (job_id, source_type, status, checksum, json.dumps(metadata), now, now),
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
    payload["source_metadata"] = _coerce_json_payload(payload.get("source_metadata")) or {}
    return payload


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


def list_import_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM import_jobs ORDER BY datetime(created_at) DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _dict_from_row(row)
            payload["source_metadata"] = _coerce_json_payload(payload.get("source_metadata")) or {}
            out.append(payload)
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
