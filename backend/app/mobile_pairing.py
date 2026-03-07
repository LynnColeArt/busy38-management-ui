from __future__ import annotations

import os
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

    resolved_device_label = normalized_device_label or str(record.get("device_label") or "").strip() or "Busy mobile device"
    token_expires_at = now + timedelta(
        seconds=int((os.getenv("BUSY38_MOBILE_PAIRING_TOKEN_TTL_SEC") or str(PAIRING_TOKEN_TTL_DEFAULT_SEC)).strip())
    )
    token = build_scoped_pairing_token(
        authorized_room_ids=normalize_room_scope_ids(record.get("authorized_room_ids")),
        orchestrator_scope=normalize_orchestrator_scope(record.get("orchestrator_scope")),
        expires_at=token_expires_at,
        issued_by=str(record.get("issued_by") or "admin"),
        device_label=resolved_device_label,
    )
    record["consumed_at"] = now.isoformat()
    record["consumed_device_label"] = resolved_device_label
    state["issued_codes"][code_hash] = record
    write_pairing_state(state, _runtime_root())
    return {
        "instance_id": str(state["instance_id"]),
        "bridge_url": _bridge_url(),
        "bridge_token": token,
        "expires_at": token_expires_at.isoformat(),
        "authorized_room_ids": list(normalize_room_scope_ids(record.get("authorized_room_ids"))),
        "orchestrator_scope": list(normalize_orchestrator_scope(record.get("orchestrator_scope"))),
    }


def revoke_bridge_token(*, actor: str, bridge_token: Any) -> Dict[str, Any]:
    _require_available()
    decoded = decode_scoped_pairing_token(
        str(bridge_token or "").strip(),
        require_not_expired=False,
        runtime_root=_runtime_root(),
    )
    state = load_pairing_state(_runtime_root())
    revoked_at = _now().isoformat()
    state["revoked_token_ids"][decoded.token_id] = {
        "revoked_at": revoked_at,
        "revoked_by": str(actor or "").strip() or "admin",
    }
    write_pairing_state(state, _runtime_root())
    return {
        "token_id": decoded.token_id,
        "instance_id": decoded.instance_id,
        "revoked_at": revoked_at,
    }
