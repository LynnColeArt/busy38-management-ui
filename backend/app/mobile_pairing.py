from __future__ import annotations

import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

try:
    from core.bridge.pairing import (
        PairingStateError,
        PairingTokenError,
        build_scoped_pairing_token,
        decode_scoped_pairing_token,
        generate_pairing_code,
        generate_refresh_grant,
        generate_trusted_device_relationship_id,
        get_pairing_secret,
        load_pairing_state,
        normalize_orchestrator_scope,
        normalize_pairing_code,
        normalize_room_scope_ids,
        pairing_code_hash,
        refresh_grant_hash,
        write_pairing_state,
    )

    _PAIRING_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - defensive import visibility
    PairingStateError = ValueError  # type: ignore[assignment]
    PairingTokenError = ValueError  # type: ignore[assignment]
    _PAIRING_IMPORT_ERROR = str(exc)


PAIRING_CODE_TTL_MAX_SEC = 900
PAIRING_TOKEN_TTL_DEFAULT_SEC = 86400
TRUSTED_DEVICE_TTL_DEFAULT_SEC = 2592000


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


def _normalize_bridge_url(raw: str) -> str:
    normalized = str(raw or "").strip()
    if not normalized:
        raise RuntimeError("bridge URL is unavailable")
    if normalized.startswith("http://"):
        normalized = normalized.replace("http://", "ws://", 1)
    elif normalized.startswith("https://"):
        normalized = normalized.replace("https://", "wss://", 1)
    elif not normalized.startswith("ws://") and not normalized.startswith("wss://"):
        normalized = f"ws://{normalized}"
    if normalized.endswith("/"):
        normalized = normalized[:-1]
    if not normalized.endswith("/ws"):
        normalized = f"{normalized}/ws"
    return normalized


def _request_bridge_origin(request_url: str | None) -> str:
    parsed = urlparse(str(request_url or "").strip())
    host = str(parsed.hostname or "").strip()
    if not host:
        return ""
    scheme = "wss" if parsed.scheme == "https" else "ws"
    port = (os.getenv("BUSY38_BRIDGE_PORT") or "8787").strip() or "8787"
    return f"{scheme}://{host}:{port}"


def _bridge_url(*, request_url: str | None = None) -> str:
    override = (os.getenv("BUSY38_MOBILE_PAIRING_BRIDGE_URL") or "").strip()
    raw = override or (os.getenv("BUSY_BRIDGE_URL") or "").strip()
    if raw:
        return _normalize_bridge_url(raw)

    host = (os.getenv("BUSY38_BRIDGE_HOST") or "").strip()
    if host:
        scheme = "wss" if urlparse(str(request_url or "").strip()).scheme == "https" else "ws"
        port = (os.getenv("BUSY38_BRIDGE_PORT") or "8787").strip() or "8787"
        return _normalize_bridge_url(f"{scheme}://{host}:{port}")

    request_origin = _request_bridge_origin(request_url)
    if request_origin:
        return _normalize_bridge_url(request_origin)

    return _normalize_bridge_url("ws://127.0.0.1:8787")


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


def _normalize_expected_instance_id(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _normalize_device_relationship_id(value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise PairingStateError(
            "PAIRING_TRUSTED_DEVICE_INVALID",
            "device_relationship_id is required",
        )
    return normalized


def _normalize_refresh_grant(value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise PairingStateError(
            "PAIRING_REFRESH_GRANT_INVALID",
            "refresh_grant is required",
        )
    return normalized


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


def _coerce_token_ttl_sec() -> int:
    raw = (os.getenv("BUSY38_MOBILE_PAIRING_TOKEN_TTL_SEC") or str(PAIRING_TOKEN_TTL_DEFAULT_SEC)).strip()
    try:
        ttl = int(raw)
    except Exception as exc:
        raise RuntimeError("BUSY38_MOBILE_PAIRING_TOKEN_TTL_SEC must be an integer") from exc
    if ttl <= 0:
        raise RuntimeError("BUSY38_MOBILE_PAIRING_TOKEN_TTL_SEC must be positive")
    return ttl


def _coerce_trusted_device_ttl_sec() -> int:
    raw = (os.getenv("BUSY38_MOBILE_TRUSTED_DEVICE_TTL_SEC") or str(TRUSTED_DEVICE_TTL_DEFAULT_SEC)).strip()
    try:
        ttl = int(raw)
    except Exception as exc:
        raise RuntimeError("BUSY38_MOBILE_TRUSTED_DEVICE_TTL_SEC must be an integer") from exc
    if ttl <= 0:
        raise RuntimeError("BUSY38_MOBILE_TRUSTED_DEVICE_TTL_SEC must be positive")
    return ttl


def _issue_bridge_token(
    *,
    authorized_room_ids: tuple[str, ...],
    orchestrator_scope: tuple[str, ...],
    issued_by: str,
    device_label: str,
) -> tuple[str, str, str]:
    now = _now()
    token_expires_at = now + timedelta(seconds=_coerce_token_ttl_sec())
    token_id = secrets.token_hex(12)
    token = build_scoped_pairing_token(
        authorized_room_ids=authorized_room_ids,
        orchestrator_scope=orchestrator_scope,
        expires_at=token_expires_at,
        issued_by=issued_by,
        device_label=device_label,
        token_id=token_id,
    )
    return token, token_id, token_expires_at.isoformat()


def _issue_trusted_device_credentials() -> tuple[str, str, str, str]:
    now = _now()
    trusted_device_expires_at = now + timedelta(seconds=_coerce_trusted_device_ttl_sec())
    refresh_grant = generate_refresh_grant()
    return (
        generate_trusted_device_relationship_id(),
        refresh_grant,
        refresh_grant_hash(refresh_grant),
        trusted_device_expires_at.isoformat(),
    )


def _resolve_trusted_device(state: Dict[str, Any], device_relationship_id: str) -> Dict[str, Any]:
    trusted_devices = state.get("trusted_devices")
    if not isinstance(trusted_devices, dict):
        raise PairingStateError("PAIRING_STATE_INVALID", "trusted_devices must be an object")
    record = trusted_devices.get(device_relationship_id)
    if not isinstance(record, dict):
        raise PairingStateError("PAIRING_TRUSTED_DEVICE_UNKNOWN", "trusted device is unknown")
    return record


def _linked_trusted_device_id(state: Dict[str, Any], token_id: str) -> str | None:
    trusted_devices = state.get("trusted_devices")
    if not isinstance(trusted_devices, dict):
        raise PairingStateError("PAIRING_STATE_INVALID", "trusted_devices must be an object")
    for device_relationship_id, record in trusted_devices.items():
        if not isinstance(record, dict):
            continue
        if str(record.get("token_id") or "").strip() == token_id:
            return str(device_relationship_id)
    return None


def _revoke_token_id(
    state: Dict[str, Any],
    *,
    token_id: str,
    actor: str,
    revoked_at: str,
) -> None:
    if token_id in state["revoked_token_ids"]:
        return
    state["revoked_token_ids"][token_id] = {
        "revoked_at": revoked_at,
        "revoked_by": str(actor or "").strip() or "system",
    }


def _trusted_device_status(record: Dict[str, Any]) -> str:
    revoked_at = _optional_timestamp_text(record.get("revoked_at"))
    expires_at = _coerce_timestamp_text(record.get("trusted_device_expires_at"), field_name="trusted_device_expires_at")
    if revoked_at:
        return "revoked"
    return "expired" if _timestamp_not_after_now(expires_at) else "active"


def _advisory_refresh_after(*, expires_at: str) -> str:
    expires_at_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    now = _now()
    remaining_seconds = int((expires_at_dt - now).total_seconds())
    if remaining_seconds <= 0:
        return now.isoformat()
    return (now + timedelta(seconds=max(60, remaining_seconds // 2))).isoformat()


def refresh_error_code(code: str | None) -> str:
    normalized = str(code or "").strip().upper()
    return {
        "PAIRING_TRUSTED_DEVICE_REVOKED": "trusted_device_revoked",
        "PAIRING_TRUSTED_DEVICE_EXPIRED": "trusted_device_expired",
        "PAIRING_REFRESH_GRANT_INVALID": "refresh_grant_invalid",
        "PAIRING_INSTANCE_MISMATCH": "instance_mismatch",
        "PAIRING_ROOM_SCOPE_DENIED": "scope_no_longer_authorized",
        "PAIRING_ORCHESTRATOR_SCOPE_DENIED": "scope_no_longer_authorized",
    }.get(normalized, "")


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


def exchange_pairing_code(
    *,
    pairing_code: Any,
    device_label: Any,
    expected_instance_id: Any | None = None,
    request_url: str | None = None,
) -> Dict[str, Any]:
    _require_available()
    normalized_code = normalize_pairing_code(pairing_code)
    normalized_device_label = str(device_label or "").strip()
    state = load_pairing_state(_runtime_root())
    resolved_expected_instance_id = _normalize_expected_instance_id(expected_instance_id)
    state_instance_id = str(state["instance_id"])
    if (
        resolved_expected_instance_id is not None and
        resolved_expected_instance_id != state_instance_id
    ):
        raise PairingStateError(
            "PAIRING_INSTANCE_MISMATCH",
            f"expected Busy instance {resolved_expected_instance_id}, but control plane is {state_instance_id}",
        )
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
    token, token_id, token_expires_at = _issue_bridge_token(
        authorized_room_ids=normalized_room_ids,
        orchestrator_scope=normalized_orchestrators,
        issued_by=str(record.get("issued_by") or "admin"),
        device_label=resolved_device_label,
    )
    device_relationship_id, refresh_grant, refresh_grant_digest, trusted_device_expires_at = _issue_trusted_device_credentials()
    record["consumed_at"] = now.isoformat()
    record["consumed_device_label"] = resolved_device_label
    record["token_id"] = token_id
    record["token_expires_at"] = token_expires_at
    state["issued_codes"][code_hash] = record
    state["trusted_devices"][device_relationship_id] = {
        "device_label": resolved_device_label,
        "authorized_room_ids": list(normalized_room_ids),
        "orchestrator_scope": list(normalized_orchestrators),
        "issued_by": str(record.get("issued_by") or "admin"),
        "issued_at": now.isoformat(),
        "last_refreshed_at": now.isoformat(),
        "trusted_device_expires_at": trusted_device_expires_at,
        "refresh_grant_hash": refresh_grant_digest,
        "token_id": token_id,
        "token_expires_at": token_expires_at,
        "revoked_at": None,
        "revoked_by": None,
    }
    write_pairing_state(state, _runtime_root())
    return {
        "instance_id": state_instance_id,
        "bridge_url": _bridge_url(request_url=request_url),
        "bridge_token": token,
        "token_id": token_id,
        "expires_at": token_expires_at,
        "device_relationship_id": device_relationship_id,
        "refresh_grant": refresh_grant,
        "trusted_device_expires_at": trusted_device_expires_at,
        "authorized_room_ids": list(normalized_room_ids),
        "orchestrator_scope": list(normalized_orchestrators),
    }


def refresh_trusted_device(
    *,
    device_relationship_id: Any,
    refresh_grant: Any,
    expected_instance_id: Any | None = None,
    device_label: Any | None = None,
    client_platform: Any | None = None,
    last_transport: Any | None = None,
    request_url: str | None = None,
) -> Dict[str, Any]:
    _require_available()
    normalized_relationship_id = _normalize_device_relationship_id(device_relationship_id)
    normalized_refresh_grant = _normalize_refresh_grant(refresh_grant)
    state = load_pairing_state(_runtime_root())
    resolved_expected_instance_id = _normalize_expected_instance_id(expected_instance_id)
    state_instance_id = str(state["instance_id"])
    if (
        resolved_expected_instance_id is not None and
        resolved_expected_instance_id != state_instance_id
    ):
        raise PairingStateError(
            "PAIRING_INSTANCE_MISMATCH",
            f"expected Busy instance {resolved_expected_instance_id}, but control plane is {state_instance_id}",
        )
    record = _resolve_trusted_device(state, normalized_relationship_id)
    if _trusted_device_status(record) == "revoked":
        raise PairingStateError("PAIRING_TRUSTED_DEVICE_REVOKED", "trusted device has been revoked")
    if _trusted_device_status(record) == "expired":
        raise PairingStateError("PAIRING_TRUSTED_DEVICE_EXPIRED", "trusted device is expired")
    expected_hash = str(record.get("refresh_grant_hash") or "").strip()
    presented_hash = refresh_grant_hash(normalized_refresh_grant)
    if not expected_hash or not hmac.compare_digest(expected_hash, presented_hash):
        raise PairingStateError("PAIRING_REFRESH_GRANT_INVALID", "refresh grant is invalid")
    normalized_room_ids = normalize_room_scope_ids(record.get("authorized_room_ids"))
    normalized_orchestrators = normalize_orchestrator_scope(record.get("orchestrator_scope"))
    _validate_known_pairing_scopes(
        authorized_room_ids=normalized_room_ids,
        orchestrator_scope=normalized_orchestrators,
    )

    now = _now()
    previous_token_id = str(record.get("token_id") or "").strip()
    if previous_token_id:
        _revoke_token_id(state, token_id=previous_token_id, actor="trusted-device-refresh", revoked_at=now.isoformat())

    resolved_device_label = str(record.get("device_label") or "").strip() or "Busy mobile device"
    token, token_id, token_expires_at = _issue_bridge_token(
        authorized_room_ids=normalized_room_ids,
        orchestrator_scope=normalized_orchestrators,
        issued_by=str(record.get("issued_by") or "admin"),
        device_label=resolved_device_label,
    )
    rotated_refresh_grant = generate_refresh_grant()
    record["refresh_grant_hash"] = refresh_grant_hash(rotated_refresh_grant)
    record["last_refreshed_at"] = now.isoformat()
    record["trusted_device_expires_at"] = (now + timedelta(seconds=_coerce_trusted_device_ttl_sec())).isoformat()
    record["token_id"] = token_id
    record["token_expires_at"] = token_expires_at
    if str(device_label or "").strip():
        record["device_label"] = _coerce_device_label(device_label)
    if str(client_platform or "").strip():
        record["last_client_platform"] = str(client_platform).strip()
    if isinstance(last_transport, dict):
        record["last_transport_hint"] = dict(last_transport)
    state["trusted_devices"][normalized_relationship_id] = record
    write_pairing_state(state, _runtime_root())
    return {
        "instance_id": state_instance_id,
        "bridge_url": _bridge_url(request_url=request_url),
        "bridge_token": token,
        "token_id": token_id,
        "expires_at": token_expires_at,
        "device_relationship_id": normalized_relationship_id,
        "refresh_grant": rotated_refresh_grant,
        "trusted_device_expires_at": record["trusted_device_expires_at"],
        "display_label": str(record.get("device_label") or "").strip() or resolved_device_label,
        "authorized_room_ids": list(normalized_room_ids),
        "orchestrator_scope": list(normalized_orchestrators),
        "transport": {
            "bridge_url": _bridge_url(request_url=request_url),
            "bridge_token": token,
            "expires_at": token_expires_at,
            "auth_mode": "pairing_scoped_token",
        },
        "refresh": {
            "refresh_grant": rotated_refresh_grant,
            "rotated": True,
            "refresh_after": _advisory_refresh_after(expires_at=token_expires_at),
        },
        "continuity": {
            "refresh_capable": True,
            "re_pair_required": False,
            "policy_requires_reverification": False,
        },
    }

def _resolve_token_id_from_state(state: Dict[str, Any], token_id: str) -> str:
    if token_id in state["revoked_token_ids"]:
        raise PairingTokenError("PAIRING_TOKEN_REVOKED", "pairing token has already been revoked")
    for record in state["issued_codes"].values():
        if not isinstance(record, dict):
            continue
        if str(record.get("token_id") or "").strip() == token_id:
            return token_id
    trusted_devices = state.get("trusted_devices")
    if not isinstance(trusted_devices, dict):
        raise PairingStateError("PAIRING_STATE_INVALID", "trusted_devices must be an object")
    for record in trusted_devices.values():
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
    _revoke_token_id(state, token_id=resolved_token_id, actor=str(actor or "").strip() or "admin", revoked_at=revoked_at)
    linked_device_id = _linked_trusted_device_id(state, resolved_token_id)
    if linked_device_id is not None:
        trusted_device = _resolve_trusted_device(state, linked_device_id)
        trusted_device["revoked_at"] = revoked_at
        trusted_device["revoked_by"] = str(actor or "").strip() or "admin"
        state["trusted_devices"][linked_device_id] = trusted_device
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
    trusted_device_rows: list[Dict[str, Any]] = []
    revoked_map = state["revoked_token_ids"]
    trusted_devices = state.get("trusted_devices")
    if not isinstance(trusted_devices, dict):
        raise PairingStateError("PAIRING_STATE_INVALID", "trusted_devices must be an object")

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
    for device_relationship_id, record in trusted_devices.items():
        if not isinstance(record, dict):
            raise PairingStateError("PAIRING_STATE_INVALID", "trusted device entry must be an object")
        trusted_device_rows.append(
            {
                "device_relationship_id": _normalize_device_relationship_id(device_relationship_id),
                "device_label": _coerce_device_label(record.get("device_label")),
                "authorized_room_ids": list(normalize_room_scope_ids(record.get("authorized_room_ids"))),
                "orchestrator_scope": list(normalize_orchestrator_scope(record.get("orchestrator_scope"))),
                "issued_by": str(record.get("issued_by") or "").strip() or "admin",
                "issued_at": _coerce_timestamp_text(record.get("issued_at"), field_name="issued_at"),
                "last_refreshed_at": _coerce_timestamp_text(record.get("last_refreshed_at"), field_name="last_refreshed_at"),
                "trusted_device_expires_at": _coerce_timestamp_text(
                    record.get("trusted_device_expires_at"),
                    field_name="trusted_device_expires_at",
                ),
                "token_id": str(record.get("token_id") or "").strip() or None,
                "token_expires_at": _optional_timestamp_text(record.get("token_expires_at")),
                "revoked_at": _optional_timestamp_text(record.get("revoked_at")),
                "revoked_by": str(record.get("revoked_by") or "").strip() or None,
                "status": _trusted_device_status(record),
            }
        )
    trusted_device_rows.sort(
        key=lambda row: (row["last_refreshed_at"], row["device_relationship_id"]),
        reverse=True,
    )
    return {
        "instance_id": str(state["instance_id"]),
        "issued": issued_rows,
        "revoked": revoked_rows,
        "trusted_devices": trusted_device_rows,
    }
