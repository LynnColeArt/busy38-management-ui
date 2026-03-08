from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

try:
    from core.bridge.pairing import (
        PairingStateError,
        PairingTokenError,
        build_scoped_pairing_token,
        decode_scoped_pairing_token,
        generate_pairing_code,
        get_pairing_secret,
        load_pairing_state,
        normalize_orchestrator_scope,
        normalize_pairing_code,
        normalize_room_scope_ids,
        pairing_code_hash,
        write_pairing_state,
    )

    _PAIRING_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - defensive import visibility
    PairingStateError = ValueError  # type: ignore[assignment]
    PairingTokenError = ValueError  # type: ignore[assignment]
    _PAIRING_IMPORT_ERROR = str(exc)


PAIRING_CODE_TTL_MAX_SEC = 900
PAIRING_TOKEN_TTL_DEFAULT_SEC = 86400


def _runtime_root() -> str | Path | None:
    configured = (os.getenv("BUSY_RUNTIME_PATH") or "").strip()
    if configured:
        return configured
    return None


def pairing_available() -> tuple[bool, str]:
    if _PAIRING_IMPORT_ERROR:
        return False, f"pairing helpers unavailable: {_PAIRING_IMPORT_ERROR}"
    try:
        get_pairing_secret()
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _require_available() -> None:
    available, reason = pairing_available()
    if not available:
        raise RuntimeError(reason or "pairing is unavailable")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _bridge_url() -> str:
    override = (os.getenv("BUSY38_MOBILE_PAIRING_BRIDGE_URL") or "").strip()
    raw = override or (os.getenv("BUSY_BRIDGE_URL") or "").strip()
    if raw:
        if raw.startswith("http://"):
            raw = raw.replace("http://", "ws://", 1)
        elif raw.startswith("https://"):
            raw = raw.replace("https://", "wss://", 1)
        elif not raw.startswith("ws://") and not raw.startswith("wss://"):
            raw = f"ws://{raw}"
        if raw.endswith("/"):
            raw = raw[:-1]
        if not raw.endswith("/ws"):
            raw = f"{raw}/ws"
        return raw

    host = (os.getenv("BUSY38_BRIDGE_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = (os.getenv("BUSY38_BRIDGE_PORT") or "8787").strip() or "8787"
    return f"ws://{host}:{port}/ws"


def _coerce_device_label(value: Any) -> str:
    label = str(value or "").strip()
    if not label:
        raise PairingStateError("PAIRING_DEVICE_LABEL_INVALID", "device_label is required")
    return label


def _coerce_issue_ttl(value: Any) -> int:
    try:
        ttl = int(value)
    except Exception as exc:
        raise PairingStateError("PAIRING_TTL_INVALID", "ttl_sec must be an integer") from exc
    if ttl <= 0 or ttl > PAIRING_CODE_TTL_MAX_SEC:
        raise PairingStateError(
            "PAIRING_TTL_INVALID",
            f"ttl_sec must be between 1 and {PAIRING_CODE_TTL_MAX_SEC}",
        )
    return ttl


def _normalize_token_id(value: Any) -> str:
    token_id = str(value or "").strip()
    if not token_id:
        raise PairingTokenError("PAIRING_TOKEN_INVALID", "token_id is required")
    return token_id


def _coerce_timestamp_text(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PairingStateError("PAIRING_STATE_INVALID", f"{field_name} is required")
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PairingStateError("PAIRING_STATE_INVALID", f"{field_name} must be ISO8601") from exc
    return text


def _optional_timestamp_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PairingStateError("PAIRING_STATE_INVALID", "timestamp must be ISO8601") from exc
    return text


def _timestamp_not_after_now(value: str) -> bool:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc) <= _now()


def _load_known_pairing_scopes() -> tuple[set[str], set[str]]:
    try:
        from core.bridge import service as bridge_service
    except Exception as exc:
        raise RuntimeError(f"pairing scope validation unavailable: {exc}") from exc

    room_records = bridge_service._build_discovered_room_records()
    known_room_ids = {
        str(record.get("room_id") or "").strip().lower()
        for record in room_records
        if str(record.get("room_id") or "").strip()
    }

    known_orchestrator_inputs = [
        str(record.get("room_owner_identity") or "").strip()
        for record in room_records
        if str(record.get("room_owner_identity") or "").strip()
    ]
    known_orchestrator_inputs.append("GM")
    known_orchestrators = set(normalize_orchestrator_scope(known_orchestrator_inputs))
    return known_room_ids, known_orchestrators


def _validate_known_pairing_scopes(
    *,
    authorized_room_ids: tuple[str, ...],
    orchestrator_scope: tuple[str, ...],
) -> None:
    known_room_ids, known_orchestrators = _load_known_pairing_scopes()

    unknown_room_ids = [room_id for room_id in authorized_room_ids if room_id not in known_room_ids]
    if unknown_room_ids:
        raise PairingStateError(
            "PAIRING_SCOPE_INVALID",
            f"unknown authorized_room_ids: {', '.join(unknown_room_ids)}",
        )

    unknown_orchestrators = [
        orchestrator_id for orchestrator_id in orchestrator_scope if orchestrator_id not in known_orchestrators
    ]
    if unknown_orchestrators:
        raise PairingStateError(
            "PAIRING_SCOPE_INVALID",
            f"unknown orchestrator_scope: {', '.join(unknown_orchestrators)}",
        )


def issue_pairing_code(
    *,
    actor: str,
    device_label: Any,
    authorized_room_ids: Any,
    orchestrator_scope: Any,
    ttl_sec: Any,
) -> Dict[str, Any]:
    _require_available()
    normalized_device_label = _coerce_device_label(device_label)
    normalized_room_ids = normalize_room_scope_ids(authorized_room_ids)
    normalized_orchestrators = normalize_orchestrator_scope(orchestrator_scope)
    _validate_known_pairing_scopes(
        authorized_room_ids=normalized_room_ids,
        orchestrator_scope=normalized_orchestrators,
    )
    resolved_ttl = _coerce_issue_ttl(ttl_sec)
    now = _now()
    expires_at = now + timedelta(seconds=resolved_ttl)
    state = load_pairing_state(_runtime_root())

    chosen_code = ""
    chosen_hash = ""
    for _ in range(10):
        candidate = generate_pairing_code()
        candidate_hash = pairing_code_hash(candidate)
        existing = state["issued_codes"].get(candidate_hash)
        if not isinstance(existing, dict):
            chosen_code = candidate
            chosen_hash = candidate_hash
            break
        existing_expires = str(existing.get("expires_at") or "").strip()
        existing_consumed = str(existing.get("consumed_at") or "").strip()
        if existing_consumed or (existing_expires and existing_expires <= now.isoformat()):
            chosen_code = candidate
            chosen_hash = candidate_hash
            break
    if not chosen_code or not chosen_hash:
        raise PairingStateError("PAIRING_CODE_ISSUE_FAILED", "could not allocate a unique pairing code")

    state["issued_codes"][chosen_hash] = {
        "device_label": normalized_device_label,
        "authorized_room_ids": list(normalized_room_ids),
        "orchestrator_scope": list(normalized_orchestrators),
        "issued_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "issued_by": str(actor or "").strip() or "admin",
        "consumed_at": None,
    }
    write_pairing_state(state, _runtime_root())
    return {
        "pairing_code": chosen_code,
        "pairing_code_ttl_sec": resolved_ttl,
        "expires_at": expires_at.isoformat(),
        "instance_id": str(state["instance_id"]),
        "authorized_room_ids": list(normalized_room_ids),
        "orchestrator_scope": list(normalized_orchestrators),
    }


def exchange_pairing_code(*, pairing_code: Any, device_label: Any) -> Dict[str, Any]:
    _require_available()
    normalized_code = normalize_pairing_code(pairing_code)
    normalized_device_label = str(device_label or "").strip()
    state = load_pairing_state(_runtime_root())
    code_hash = pairing_code_hash(normalized_code)
    record = state["issued_codes"].get(code_hash)
    if not isinstance(record, dict):
        raise PairingStateError("PAIRING_CODE_INVALID", "pairing code is invalid")
    expires_at = datetime.fromisoformat(str(record.get("expires_at")).replace("Z", "+00:00")).astimezone(timezone.utc)
    consumed_at = str(record.get("consumed_at") or "").strip()
    now = _now()
    if consumed_at:
        raise PairingStateError("PAIRING_CODE_CONSUMED", "pairing code has already been used")
    if expires_at <= now:
        raise PairingStateError("PAIRING_CODE_EXPIRED", "pairing code is expired")
    normalized_room_ids = normalize_room_scope_ids(record.get("authorized_room_ids"))
    normalized_orchestrators = normalize_orchestrator_scope(record.get("orchestrator_scope"))
    _validate_known_pairing_scopes(
        authorized_room_ids=normalized_room_ids,
        orchestrator_scope=normalized_orchestrators,
    )

    resolved_device_label = normalized_device_label or str(record.get("device_label") or "").strip() or "Busy mobile device"
    token_expires_at = now + timedelta(
        seconds=int((os.getenv("BUSY38_MOBILE_PAIRING_TOKEN_TTL_SEC") or str(PAIRING_TOKEN_TTL_DEFAULT_SEC)).strip())
    )
    token_id = secrets.token_hex(12)
    token = build_scoped_pairing_token(
        authorized_room_ids=normalized_room_ids,
        orchestrator_scope=normalized_orchestrators,
        expires_at=token_expires_at,
        issued_by=str(record.get("issued_by") or "admin"),
        device_label=resolved_device_label,
        token_id=token_id,
    )
    record["consumed_at"] = now.isoformat()
    record["consumed_device_label"] = resolved_device_label
    record["token_id"] = token_id
    record["token_expires_at"] = token_expires_at.isoformat()
    state["issued_codes"][code_hash] = record
    write_pairing_state(state, _runtime_root())
    return {
        "instance_id": str(state["instance_id"]),
        "bridge_url": _bridge_url(),
        "bridge_token": token,
        "token_id": token_id,
        "expires_at": token_expires_at.isoformat(),
        "authorized_room_ids": list(normalized_room_ids),
        "orchestrator_scope": list(normalized_orchestrators),
    }

def _resolve_token_id_from_state(state: Dict[str, Any], token_id: str) -> str:
    if token_id in state["revoked_token_ids"]:
        raise PairingTokenError("PAIRING_TOKEN_REVOKED", "pairing token has already been revoked")
    for record in state["issued_codes"].values():
        if not isinstance(record, dict):
            continue
        if str(record.get("token_id") or "").strip() == token_id:
            return token_id
    raise PairingTokenError("PAIRING_TOKEN_INVALID", "pairing token id is unknown")


def revoke_pairing_grant(
    *,
    actor: str,
    bridge_token: Any | None = None,
    token_id: Any | None = None,
) -> Dict[str, Any]:
    _require_available()
    state = load_pairing_state(_runtime_root())
    raw_bridge_token = str(bridge_token or "").strip()
    raw_token_id = str(token_id or "").strip()
    if bool(raw_bridge_token) == bool(raw_token_id):
        raise PairingStateError(
            "PAIRING_REVOKE_INVALID",
            "provide exactly one of bridge_token or token_id",
        )
    if raw_bridge_token:
        decoded = decode_scoped_pairing_token(
            raw_bridge_token,
            require_not_expired=False,
            runtime_root=_runtime_root(),
        )
        resolved_token_id = decoded.token_id
        instance_id = decoded.instance_id
    else:
        resolved_token_id = _resolve_token_id_from_state(state, _normalize_token_id(raw_token_id))
        instance_id = str(state["instance_id"])
    revoked_at = _now().isoformat()
    state["revoked_token_ids"][resolved_token_id] = {
        "revoked_at": revoked_at,
        "revoked_by": str(actor or "").strip() or "admin",
    }
    write_pairing_state(state, _runtime_root())
    return {
        "token_id": resolved_token_id,
        "instance_id": instance_id,
        "revoked_at": revoked_at,
    }


def list_pairing_state() -> Dict[str, Any]:
    _require_available()
    state = load_pairing_state(_runtime_root())
    issued_rows: list[Dict[str, Any]] = []
    revoked_rows: list[Dict[str, Any]] = []
    revoked_map = state["revoked_token_ids"]

    for token_id, revoked in revoked_map.items():
        if not isinstance(revoked, dict):
            raise PairingStateError("PAIRING_STATE_INVALID", "revoked token entry must be an object")
        revoked_rows.append(
            {
                "token_id": _normalize_token_id(token_id),
                "revoked_at": _coerce_timestamp_text(revoked.get("revoked_at"), field_name="revoked_at"),
                "revoked_by": str(revoked.get("revoked_by") or "").strip() or "admin",
            }
        )

    for code_hash, record in state["issued_codes"].items():
        if not isinstance(record, dict):
            raise PairingStateError("PAIRING_STATE_INVALID", "issued code entry must be an object")
        issued_at = _coerce_timestamp_text(record.get("issued_at"), field_name="issued_at")
        expires_at = _coerce_timestamp_text(record.get("expires_at"), field_name="expires_at")
        consumed_at = _optional_timestamp_text(record.get("consumed_at"))
        token_id = str(record.get("token_id") or "").strip() or None
        token_expires_at = _optional_timestamp_text(record.get("token_expires_at"))
        revoked = revoked_map.get(token_id) if token_id else None
        revoked_at = None
        revoked_by = None
        if revoked is not None:
            if not isinstance(revoked, dict):
                raise PairingStateError("PAIRING_STATE_INVALID", "revoked token entry must be an object")
            revoked_at = _coerce_timestamp_text(revoked.get("revoked_at"), field_name="revoked_at")
            revoked_by = str(revoked.get("revoked_by") or "").strip() or "admin"

        if revoked_at:
            status = "revoked"
        elif token_id and token_expires_at:
            status = "expired" if _timestamp_not_after_now(token_expires_at) else "active"
        elif consumed_at:
            status = "consumed"
        else:
            status = "expired" if _timestamp_not_after_now(expires_at) else "pending"

        issued_rows.append(
            {
                "pairing_code_hash": str(code_hash),
                "device_label": _coerce_device_label(record.get("device_label")),
                "authorized_room_ids": list(normalize_room_scope_ids(record.get("authorized_room_ids"))),
                "orchestrator_scope": list(normalize_orchestrator_scope(record.get("orchestrator_scope"))),
                "issued_by": str(record.get("issued_by") or "").strip() or "admin",
                "issued_at": issued_at,
                "expires_at": expires_at,
                "consumed_at": consumed_at,
                "consumed_device_label": str(record.get("consumed_device_label") or "").strip() or None,
                "token_id": token_id,
                "token_expires_at": token_expires_at,
                "revoked_at": revoked_at,
                "revoked_by": revoked_by,
                "status": status,
            }
        )

    issued_rows.sort(key=lambda row: (row["issued_at"], row["pairing_code_hash"]), reverse=True)
    revoked_rows.sort(key=lambda row: (row["revoked_at"], row["token_id"]), reverse=True)
    return {
        "instance_id": str(state["instance_id"]),
        "issued": issued_rows,
        "revoked": revoked_rows,
    }
