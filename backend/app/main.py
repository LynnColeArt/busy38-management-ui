"""FastAPI backend for the Busy38 management UI MVP."""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from .import_adapters import get_import_adapter
from .import_contract import checksum_payload

from . import storage
from .runtime import RuntimeActionResult, load_runtime_adapter


app = FastAPI(title="Busy38 Management UI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SettingsUpdate(BaseModel):
    heartbeat_interval: Optional[int] = None
    fallback_budget_per_hour: Optional[int] = None
    auto_restart: Optional[bool] = None


class ProviderUpdate(BaseModel):
    enabled: Optional[bool] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    priority: Optional[int] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProviderModelDiscovery(BaseModel):
    api_key: Optional[str] = None
    endpoint: Optional[str] = None


class ProviderModelTest(BaseModel):
    api_key: Optional[str] = None
    endpoint: Optional[str] = None


class ProviderTestAllRequest(BaseModel):
    include_disabled: bool = False
    provider_ids: Optional[List[str]] = None


class ProviderSecretAction(BaseModel):
    action: Literal["set", "rotate", "clear"]
    api_key: Optional[str] = None


class ProviderCreate(BaseModel):
    id: str
    name: str
    endpoint: str
    model: str
    status: str = "configured"
    priority: int = 100
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


class AgentUpdate(BaseModel):
    enabled: Optional[bool] = None
    name: Optional[str] = None
    role: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class MemoryCreate(BaseModel):
    scope: str
    type: str
    content: str


class ChatHistoryCreate(BaseModel):
    agent_id: str
    summary: str


class ImportDecisionRequest(BaseModel):
    import_item_ids: List[str]
    review_state: str
    actor: Optional[str] = None
    note: Optional[str] = None


@app.on_event("startup")
def _startup() -> None:
    storage.ensure_schema()


runtime = load_runtime_adapter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_legacy_token = os.getenv("MANAGEMENT_API_TOKEN", "").strip()
_admin_token = os.getenv("MANAGEMENT_ADMIN_TOKEN", "").strip()
_viewer_token = os.getenv("MANAGEMENT_READ_TOKEN", "").strip()

if _admin_token and _viewer_token and _admin_token == _viewer_token:
    _viewer_token = ""
if not _admin_token:
    _admin_token = _legacy_token
if not _viewer_token:
    _viewer_token = _admin_token or _legacy_token

if _viewer_token and not _admin_token:
    _admin_token = _viewer_token
storage.set_db_path_override(os.getenv("MANAGEMENT_DB_PATH"))


def _role_from_token(auth_token: Optional[str] = None, query_token: Optional[str] = None) -> str:
    if not (_admin_token or _viewer_token):
        return "admin"

    token = auth_token or query_token or ""
    if token.startswith("Bearer "):
        token = token[7:].strip()

    if _admin_token and token == _admin_token:
        return "admin"
    if _viewer_token and token == _viewer_token:
        return "viewer"
    return ""


def _token_source(auth_token: Optional[str] = None, query_token: Optional[str] = None) -> str:
    if not (_admin_token or _viewer_token):
        return "open-access"

    token = auth_token or query_token or ""
    if token.startswith("Bearer "):
        token = token[7:].strip()

    if _admin_token and token == _admin_token:
        return "admin-token"
    if _viewer_token and token == _viewer_token:
        return "read-token"
    return "invalid-token"


def _readable_role_name(role: str) -> str:
    if role == "admin":
        return "admin"
    if role == "viewer":
        return "viewer"
    return "unknown"


def _coerce_json_metadata(value: Optional[Dict[str, Any] | str]) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in ("api_key", "token", "secret", "password", "credential"))


def _redact_metadata(metadata: Optional[str] | Dict[str, Any] | list, role: str) -> Optional[Any]:
    if role == "admin":
        return metadata
    if metadata is None:
        return None
    parsed = metadata
    if isinstance(metadata, str):
        parsed = _coerce_json_metadata(metadata)

    if isinstance(parsed, dict):
        return {k: "***redacted***" if _is_sensitive_key(k) else v for k, v in parsed.items()}

    return "***redacted***"


def _require_role(required: str = "viewer"):
    def _checker(
        auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        token: Optional[str] = Query(default=None, alias="token"),
    ) -> str:
        role = _role_from_token(auth.credentials if auth else None, token)
        if not role:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if required == "admin" and role != "admin":
            raise HTTPException(status_code=403, detail="Admin token required for this action.")
        return role

    return _checker


def _event_for(
    _role: str,
    event_type: str,
    message: str,
    level: str = "info",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = storage.append_event(event_type, message, level, payload=payload)
    return payload


def _coerce_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _sanitize_provider(provider: Dict[str, Any], role: str) -> Dict[str, Any]:
    payload = dict(provider)
    payload["metadata"] = _redact_metadata(payload.get("metadata"), role)
    return payload


def _ensure_http_request_body(url: str) -> urllib.request.Request:
    if not url:
        raise HTTPException(status_code=400, detail="provider endpoint is missing")
    return urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "busy38-management-ui/0.2.0"}, method="GET")


def _safe_http_fetch(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 4) -> Any:
    request = _ensure_http_request_body(url)
    if headers:
        for key, value in headers.items():
            if value is not None:
                request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"model discovery request to {url} failed with status {exc.code}",
        )
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"model discovery request to {url} failed: {exc.reason}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"model discovery request to {url} failed: {exc}",
        )

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"invalid JSON returned from {url}: {exc.msg}",
        )


def _extract_models_from_payload(payload: Any) -> List[str]:
    models: List[str] = []
    if isinstance(payload, dict):
        candidates: List[Any] = []
        for key in ("data", "models", "result", "response"):
            if key in payload and isinstance(payload[key], list):
                candidates.extend(payload[key])
        if not candidates and isinstance(payload.get("model"), str):
            candidates.append(payload["model"])
        if not candidates:
            for item in payload.values():
                if isinstance(item, list):
                    candidates.extend(item)
        for entry in candidates:
            if isinstance(entry, str) and entry.strip():
                models.append(entry.strip())
            elif isinstance(entry, dict):
                model = entry.get("id") or entry.get("name") or entry.get("model")
                if isinstance(model, str) and model.strip():
                    models.append(model.strip())
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, str) and entry.strip():
                models.append(entry.strip())
            elif isinstance(entry, dict):
                model = entry.get("id") or entry.get("name") or entry.get("model")
                if isinstance(model, str) and model.strip():
                    models.append(model.strip())
    seen = set()
    unique_models = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        unique_models.append(model)
    return unique_models


def _provider_model_endpoints(provider: Dict[str, Any], endpoint_override: Optional[str] = None) -> List[Dict[str, str]]:
    endpoint = (endpoint_override or provider.get("endpoint") or "").rstrip("/")
    if not endpoint:
        raise HTTPException(status_code=400, detail="provider endpoint is missing")

    normalized_endpoint = endpoint
    if normalized_endpoint.endswith("/v1"):
        normalized_endpoint = normalized_endpoint[:-3]
    for suffix in ("/v1/models", "/models", "/api/tags"):
        if normalized_endpoint.endswith(suffix):
            normalized_endpoint = normalized_endpoint[: -len(suffix)]
            break

    provider_identity = f"{provider.get('id', '').lower()} {provider.get('name', '').lower()}"
    candidates: List[str] = []
    if "ollama" in provider_identity or "llama" in provider_identity:
        candidates.append(f"{normalized_endpoint}/api/tags")
        candidates.append(f"{normalized_endpoint}/v1/models")
    else:
        candidates.append(f"{normalized_endpoint}/v1/models")
        candidates.append(f"{normalized_endpoint}/models")
        candidates.append(f"{normalized_endpoint}/api/tags")
    seen = set()
    deduped: List[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return [{"url": candidate} for candidate in deduped]


def _discover_provider_models(
    provider: Dict[str, Any],
    api_key: Optional[str] = None,
    endpoint_override: Optional[str] = None,
) -> Dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    attempts: List[Dict[str, Any]] = []

    for candidate in _provider_model_endpoints(provider, endpoint_override):
        try:
            payload = _safe_http_fetch(candidate["url"], headers=headers)
            models = _extract_models_from_payload(payload)
            if models:
                return {
                    "models": models,
                    "endpoint": candidate["url"],
                    "source": _maybe_strip_api_path(candidate["url"]),
                    "attempts": attempts,
                }
            attempts.append({"url": candidate["url"], "result": "empty-model-list"})
        except HTTPException as exc:
            attempts.append({"url": candidate["url"], "error": exc.detail})
        except Exception as exc:
            attempts.append({"url": candidate["url"], "error": str(exc)})
    return {
        "models": [],
        "endpoint": endpoint_override or provider.get("endpoint") or "",
        "source": "unresolved",
        "attempts": attempts,
        "error": "No model list endpoint matched this provider.",
    }


def _record_provider_test_metadata(
    metadata: Dict[str, Any],
    source: str,
    endpoint: str,
    latency_ms: float,
    models: List[str],
    attempts: List[Dict[str, Any]],
    error: Optional[str],
) -> Dict[str, Any]:
    payload = dict(metadata or {})
    history = payload.get("test_history") if isinstance(payload.get("test_history"), list) else []
    if not isinstance(history, list):
        history = []

    tested_at = _now_iso()
    test_record = {
        "tested_at": tested_at,
        "status": "pass" if models else "fail",
        "source": source,
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "models_count": len(models),
        "attempts": attempts,
    }
    if error:
        test_record["error"] = error

    history.append(test_record)
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    cleaned_history: List[Dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        tested = _coerce_iso_datetime(item.get("tested_at"))
        if tested and tested >= cutoff:
            cleaned_history.append(item)
    payload["test_history"] = cleaned_history[-200:]
    payload["last_test"] = test_record

    metrics_window_5m = datetime.now(timezone.utc) - timedelta(minutes=5)
    metrics_window_1m = datetime.now(timezone.utc) - timedelta(minutes=1)
    in_5m = [
        item for item in payload["test_history"]
        if isinstance(item, dict) and _coerce_iso_datetime(item.get("tested_at")) and _coerce_iso_datetime(item.get("tested_at")) >= metrics_window_5m
    ]
    in_1m = [
        item for item in payload["test_history"]
        if isinstance(item, dict) and _coerce_iso_datetime(item.get("tested_at")) and _coerce_iso_datetime(item.get("tested_at")) >= metrics_window_1m
    ]
    latencies = [
        item.get("latency_ms")
        for item in in_5m
        if isinstance(item, dict) and isinstance(item.get("latency_ms"), (int, float))
    ]
    fail_count_1m = sum(
        1 for item in in_1m
        if isinstance(item, dict) and item.get("status") == "fail"
    )
    success_count = sum(
        1 for item in in_5m
        if isinstance(item, dict) and item.get("status") == "pass"
    )
    payload["health_metrics"] = {
        "latency_ms_last": test_record.get("latency_ms"),
        "latency_ms_avg_5m": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "latency_ms_p95_5m": None,
        "success_rate_5m": round((success_count / len(in_5m)) * 100, 2) if in_5m else None,
        "failure_count_last_1m": fail_count_1m,
        "last_checked_at": tested_at,
        "last_error_message": test_record.get("error"),
        "last_error_code": None,
    }
    if latencies:
        sorted_latencies = sorted(latencies)
        p95_index = max(0, int(round(0.95 * (len(sorted_latencies) - 1))))
        payload["health_metrics"]["latency_ms_p95_5m"] = sorted_latencies[p95_index]
    return payload


def _coerce_secret_policy(provider: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    policy = metadata.get("secret_policy")
    if policy in {"required", "optional", "none"}:
        return policy

    provider_identity = (
        f"{provider.get('id', '').lower()} "
        f"{provider.get('name', '').lower()} "
        f"{provider.get('endpoint', '').lower()}"
    )
    if "ollama" in provider_identity or "llama" in provider_identity:
        return "none"
    return "required"


def _normalize_secret_metadata(
    metadata: Dict[str, Any],
    provider: Optional[Dict[str, Any]] = None,
    actor: str = "admin",
    has_secret: Optional[bool] = None,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    payload = dict(metadata or {})
    if provider is not None:
        payload["secret_policy"] = _coerce_secret_policy(provider, payload)
    if has_secret is True:
        payload["secret_present"] = True
        payload["secret_last_rotated_at"] = _now_iso()
        payload["secret_touched_by"] = actor
        payload["secret_touched_at"] = _now_iso()
        if action:
            payload["secret_last_action"] = action
    elif has_secret is False:
        payload["secret_present"] = False
        payload["secret_last_rotated_at"] = None
        payload["secret_touched_by"] = actor
        payload["secret_touched_at"] = _now_iso()
        if action:
            payload["secret_last_action"] = action
    return payload


def _maybe_strip_api_path(url: str) -> str:
    suffixes = ["/v1/models", "/api/tags", "/models"]
    for suffix in suffixes:
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def _provider_secret_status(provider: Dict[str, Any]) -> str:
    metadata = provider.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    secret_policy = _coerce_secret_policy(provider, metadata)
    secret_present = bool(metadata.get("secret_present"))
    if secret_policy == "required":
        return "required_present" if secret_present else "required_missing"
    if secret_policy == "none":
        return "none"
    if secret_present:
        return "optional_present"
    return "optional_missing"


def _provider_sort_key(provider: Dict[str, Any], sort_by: str):
    metadata = provider.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    health = metadata.get("health_metrics") or {}
    if not isinstance(health, dict):
        health = {}

    if sort_by == "last_checked":
        checked = health.get("last_checked_at")
        if checked is None:
            return 0
        checked_at = _coerce_iso_datetime(checked)
        if checked_at is None:
            return 0
        return checked_at.timestamp()
    if sort_by == "latency_5m":
        value = health.get("latency_ms_avg_5m")
        if isinstance(value, (int, float)):
            return value
        return float("inf")
    if sort_by == "failures":
        value = health.get("failure_count_last_1m")
        if isinstance(value, (int, float)):
            return value
        return 0
    return int(provider.get("priority", 0))


def _maybe_emit_provider_state_event(
    role: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    provider_id: str,
) -> None:
    before_enabled = bool(before.get("enabled"))
    after_enabled = bool(after.get("enabled"))
    before_status = (before.get("status") or "").lower()
    after_status = (after.get("status") or "").lower()
    if before_enabled != after_enabled or before_status != after_status:
        _event_for(
            role,
            "provider.state_changed",
            f"Provider state changed: {provider_id}",
            "info",
            {"provider_id": provider_id, "enabled": after_enabled, "status": after_status},
        )


def _provider_routing_chain() -> List[Dict[str, Any]]:
    providers = storage.list_providers()
    enabled = [p for p in providers if bool(p.get("enabled"))]
    if not enabled:
        return []
    sorted_enabled = sorted(
        enabled,
        key=lambda item: (
            item.get("status") != "active",
            int(item.get("priority", 0)),
        ),
    )
    chain: List[Dict[str, Any]] = []
    for index, provider in enumerate(sorted_enabled):
        chain.append({
            "id": provider["id"],
            "name": provider["name"],
            "status": provider["status"],
            "enabled": bool(provider.get("enabled")),
            "priority": provider.get("priority", 0),
            "model": provider.get("model"),
            "position": index,
            "active": index == 0,
        })
    return chain


def _run_provider_test(
    provider: Dict[str, Any],
    request: ProviderModelTest,
    role: str,
) -> Dict[str, Any]:
    provider_id = provider.get("id")
    metadata = provider.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    api_key = request.api_key if request.api_key else metadata.get("api_key")
    secret_policy = _coerce_secret_policy(provider, metadata)
    if secret_policy == "required" and not api_key:
        status_payload = _record_provider_test_metadata(
            metadata=dict(metadata),
            source="manual",
            endpoint=provider.get("endpoint") or "",
            latency_ms=0.0,
            models=[],
            attempts=[],
            error="provider secret is required for test",
        )
        provider_payload = storage.update_provider(
            provider_id=provider_id,
            values={"metadata": status_payload},
        )
        _event_for(
            role,
            "provider.tested",
            f"Provider test failed: {provider_id} (missing secret)",
            "error",
        )
        return {
            "provider_id": provider_id,
            "status": status_payload["last_test"]["status"],
            "tested_at": status_payload["last_test"]["tested_at"],
            "latency_ms": status_payload["last_test"]["latency_ms"],
            "source": status_payload["last_test"]["source"],
            "models_count": status_payload["last_test"]["models_count"],
            "error": status_payload["last_test"].get("error"),
            "provider": _sanitize_provider(provider_payload, role),
        }

    started = time.perf_counter()
    try:
        discovery = _discover_provider_models(
            provider=provider,
            api_key=api_key,
            endpoint_override=request.endpoint,
        )
    except HTTPException as exc:
        discovery = {
            "models": [],
            "endpoint": request.endpoint or provider.get("endpoint") or "",
            "source": "unresolved",
            "attempts": [],
            "error": exc.detail,
        }
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

    status_payload = _record_provider_test_metadata(
        metadata=dict(metadata),
        source=discovery.get("source", "manual"),
        endpoint=discovery.get("endpoint") or provider.get("endpoint") or "",
        latency_ms=latency_ms,
        models=discovery.get("models", []),
        attempts=discovery.get("attempts", []),
        error=discovery.get("error"),
    )
    provider_payload = storage.update_provider(
        provider_id=provider.get("id"),
        values={"metadata": status_payload},
    )
    last_test = status_payload.get("last_test", {})
    level = "error" if last_test.get("status") != "pass" else "info"
    _event_for(
        role,
        "provider.tested",
        f"Provider tested: {provider_id} ({last_test.get('status')})",
        level,
    )

    return {
        "provider_id": provider_id,
        "status": last_test.get("status"),
        "tested_at": last_test.get("tested_at"),
        "latency_ms": last_test.get("latency_ms"),
        "source": last_test.get("source"),
        "models_count": last_test.get("models_count", 0),
        "error": last_test.get("error"),
        "provider": _sanitize_provider(provider_payload, role),
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    runtime_status = runtime.get_status()
    return {
        "status": "ok",
        "service": "busy38-management-ui",
        "runtime_connected": bool(runtime_status.get("connected")),
        "updated_at": _now_iso(),
    }


@app.get("/api/settings")
def get_settings(
    role: str = Depends(_require_role("viewer")),
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    token: Optional[str] = Query(default=None, alias="token"),
) -> Dict[str, Any]:
    settings = storage.get_settings()
    settings["auto_restart"] = bool(settings["auto_restart"])
    return {
        "settings": settings,
        "updated_at": settings["updated_at"],
        "role": role,
        "role_source": _token_source(auth.credentials if auth else None, token),
    }


@app.patch("/api/settings")
def update_settings(update: SettingsUpdate, role: str = Depends(_require_role("admin"))) -> Dict[str, Any]:
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No setting fields provided")

    settings = storage.set_settings(payload)
    _event_for(role, "settings", "Settings updated via management UI", "info")
    return {"settings": settings, "updated_at": settings["updated_at"]}


@app.get("/api/providers")
def get_providers(
    role: str = Depends(_require_role("viewer")),
    kind: Optional[str] = None,
    status: Optional[str] = None,
    secret_status: Optional[str] = None,
    sort_by: str = "priority",
    sort_desc: bool = False,
) -> Dict[str, Any]:
    providers = storage.list_providers()
    if kind:
        kind_query = kind.strip().lower()
        providers = [
            provider
            for provider in providers
            if str((provider.get("metadata") or {}).get("kind", "")).strip().lower() == kind_query
        ]
    if status:
        status_query = status.strip().lower()
        providers = [provider for provider in providers if str(provider.get("status", "")).lower() == status_query]
    if secret_status:
        secret_query = secret_status.strip().lower().replace("-", "_")
        providers = [
            provider
            for provider in providers
            if _provider_secret_status(provider) == secret_query
            or (secret_query == "present" and _provider_secret_status(provider).endswith("_present"))
            or (secret_query == "missing" and _provider_secret_status(provider).endswith("_missing"))
            or (secret_query == "required" and _provider_secret_status(provider).startswith("required"))
            or (secret_query == "optional" and _provider_secret_status(provider).startswith("optional"))
            or (secret_query == "none" and _provider_secret_status(provider) == "none")
        ]
    if sort_by not in {"priority", "last_checked", "latency_5m", "failures"}:
        sort_by = "priority"
    providers = sorted(providers, key=lambda provider: _provider_sort_key(provider, sort_by), reverse=sort_desc)
    providers = [_sanitize_provider(p, role) for p in providers]
    return {"providers": providers, "updated_at": _now_iso()}


@app.post("/api/providers")
def create_provider(provider: ProviderCreate, role: str = Depends(_require_role("admin"))) -> Dict[str, Any]:
    payload = provider.model_dump()
    payload["id"] = str(payload["id"]).strip()
    payload["name"] = str(payload["name"]).strip()
    payload["endpoint"] = str(payload["endpoint"]).strip()
    payload["model"] = str(payload["model"]).strip()
    if not payload["id"]:
        raise HTTPException(status_code=400, detail="provider id is required")
    if not payload["name"]:
        raise HTTPException(status_code=400, detail="provider name is required")
    if not payload["endpoint"]:
        raise HTTPException(status_code=400, detail="provider endpoint is required")
    if not payload["model"]:
        raise HTTPException(status_code=400, detail="provider model is required")

    provider_payload = {
        "id": payload["id"],
        "name": payload["name"],
        "endpoint": payload["endpoint"],
        "model": payload["model"],
        "priority": int(payload["priority"]),
        "enabled": bool(payload["enabled"]),
        "status": payload.get("status") or "configured",
        "metadata": payload.get("metadata") or {},
    }
    provider_payload["metadata"] = _normalize_secret_metadata(
        metadata=dict(provider_payload["metadata"]),
        provider=provider_payload,
        actor=role,
        has_secret=bool(provider_payload["metadata"].get("api_key")),
    )
    if not bool(provider_payload["enabled"]):
        provider_payload["status"] = "standby"
    try:
        provider_row = storage.create_provider(provider_payload)
    except ValueError as exc:
        raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc))

    _event_for(
        role,
        "provider.created",
        f"Provider created: {provider_payload['id']}",
        "info",
        {"provider_id": provider_payload["id"]},
    )
    return {"provider": _sanitize_provider(provider_row, role), "updated_at": _now_iso()}


@app.get("/api/providers/routing-chain")
def get_provider_chain(role: str = Depends(_require_role("viewer"))) -> Dict[str, Any]:
    chain = _provider_routing_chain()
    return {"chain": chain, "updated_at": _now_iso()}


@app.get("/api/providers/{provider_id}/history")
def get_provider_history(
    provider_id: str,
    limit: int = Query(default=25, ge=1, le=500),
    role: str = Depends(_require_role("viewer")),
) -> Dict[str, Any]:
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    metadata = provider.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    history = metadata.get("test_history") or []
    if not isinstance(history, list):
        history = []
    return {
        "provider_id": provider_id,
        "test_history": list(reversed(history[-limit:])),
        "health_metrics": metadata.get("health_metrics") or {},
        "updated_at": _now_iso(),
    }


@app.get("/api/providers/{provider_id}/metrics")
def get_provider_metrics(
    provider_id: str,
    role: str = Depends(_require_role("viewer")),
) -> Dict[str, Any]:
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    metadata = provider.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "provider_id": provider_id,
        "metrics": metadata.get("health_metrics") or {},
        "last_test": metadata.get("last_test") or {},
        "updated_at": _now_iso(),
    }


@app.patch("/api/providers/{provider_id}")
def patch_provider(
    provider_id: str,
    update: ProviderUpdate,
    role: str = Depends(_require_role("admin")),
) -> Dict[str, Any]:
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No provider fields provided")

    before_provider = storage.get_provider(provider_id)
    if not before_provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    try:
        provider = storage.update_provider(provider_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    _maybe_emit_provider_state_event(
        role=role,
        before=before_provider,
        after=provider,
        provider_id=provider_id,
    )
    _event_for(
        role,
        "provider.updated",
        f"Provider updated: {provider_id}",
        "info",
        {"provider_id": provider_id},
    )
    provider["enabled"] = bool(provider["enabled"])
    return {"provider": _sanitize_provider(provider, role), "updated_at": _now_iso()}


@app.post("/api/providers/{provider_id}/discover-models")
def discover_provider_models(
    provider_id: str,
    request: ProviderModelDiscovery,
    role: str = Depends(_require_role("admin")),
) -> Dict[str, Any]:
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    discovery = _discover_provider_models(
        provider=provider,
        api_key=request.api_key,
        endpoint_override=request.endpoint,
    )

    metadata = provider.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata.pop("discovered_models", None)
    metadata["model_discovery"] = {
        "discovered_at": _now_iso(),
        "status": "complete" if discovery["models"] else "manual",
        "source": discovery.get("source"),
        "endpoint": discovery.get("endpoint"),
        "attempts": discovery.get("attempts"),
        "count": len(discovery.get("models") or []),
        "error": discovery.get("error"),
    }
    if request.endpoint:
        metadata["model_discovery"]["requested_endpoint"] = request.endpoint
    if discovery["models"]:
        metadata["discovered_models"] = discovery["models"]
        provider_payload = storage.update_provider(
            provider_id=provider_id,
            values={"metadata": metadata},
        )
        return {
            "provider": _sanitize_provider(provider_payload, role),
            "discovered_models": discovery["models"],
            "count": len(discovery["models"]),
            "source": discovery["source"],
            "updated_at": _now_iso(),
        }

    discovery["message"] = (
        "Automatic model discovery failed. Use the model field below to set values manually."
    )
    provider_payload = storage.update_provider(
        provider_id=provider_id,
        values={"metadata": metadata},
    )
    return {
        "provider": _sanitize_provider(provider_payload, role),
        "discovered_models": [],
        "count": 0,
        "source": "manual",
        "error": discovery.get("error"),
        "message": discovery.get("message"),
        "updated_at": _now_iso(),
    }


@app.post("/api/providers/{provider_id}/test")
def test_provider_models(
    provider_id: str,
    request: ProviderModelTest,
    role: str = Depends(_require_role("admin")),
) -> Dict[str, Any]:
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    result = _run_provider_test(provider, request=request, role=role)

    return {
        "provider": result["provider"],
        "tested_at": result["tested_at"],
        "latency_ms": result["latency_ms"],
        "status": result["status"],
        "source": result["source"],
        "models_count": result["models_count"],
        "error": result.get("error"),
        "updated_at": _now_iso(),
    }


@app.post("/api/providers/test-all")
def test_providers_all(
    request: ProviderTestAllRequest,
    role: str = Depends(_require_role("admin")),
) -> Dict[str, Any]:
    providers = storage.list_providers()
    candidates = [p for p in providers if request.include_disabled or bool(p.get("enabled"))]
    if request.provider_ids:
        candidate_ids = set(request.provider_ids)
        candidates = [p for p in candidates if p.get("id") in candidate_ids]
    candidates.sort(key=lambda item: (int(item.get("priority", 0)), item.get("id")))

    results = []
    pass_count = 0
    fail_count = 0
    for provider in candidates:
        provider_test_request = ProviderModelTest(api_key=None, endpoint=None)
        result = _run_provider_test(provider, request=provider_test_request, role=role)
        results.append(result)
        if result["status"] == "pass":
            pass_count += 1
        else:
            fail_count += 1

    return {
        "checked": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "results": results,
        "updated_at": _now_iso(),
    }


@app.post("/api/providers/{provider_id}/secret")
def update_provider_secret(
    provider_id: str,
    request: ProviderSecretAction,
    role: str = Depends(_require_role("admin")),
) -> Dict[str, Any]:
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    metadata = provider.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    secret_policy = _coerce_secret_policy(provider, metadata)
    if secret_policy == "none" and request.action in {"set", "rotate"}:
        raise HTTPException(
            status_code=409,
            detail="secret updates are not required for this provider",
        )

    payload = dict(metadata)
    if request.action in {"set", "rotate"}:
        if not request.api_key:
            raise HTTPException(status_code=400, detail="api_key is required")
        payload["api_key"] = request.api_key
        payload = _normalize_secret_metadata(
            metadata=payload,
            provider=provider,
            actor=role,
            has_secret=True,
            action=request.action,
        )
    else:
        payload.pop("api_key", None)
        payload = _normalize_secret_metadata(
            metadata=payload,
            provider=provider,
            actor=role,
            has_secret=False,
            action=request.action,
        )

    provider_payload = storage.update_provider(
        provider_id=provider_id,
        values={"metadata": payload},
    )
    _event_for(
        role,
        "provider.secret",
        f"Provider secret {request.action}: {provider_id}",
        "info",
        {"provider_id": provider_id, "action": request.action},
    )
    secret_event = "provider.secret.cleared" if request.action == "clear" else "provider.secret.updated"
    _event_for(
        role,
        secret_event,
        f"Provider secret {request.action}: {provider_id}",
        "info",
        {"provider_id": provider_id, "action": request.action},
    )

    return {"provider": _sanitize_provider(provider_payload, role), "updated_at": _now_iso()}


@app.get("/api/agents")
def get_agents(role: str = Depends(_require_role("viewer"))) -> Dict[str, Any]:
    agents = storage.list_agents()
    return {"agents": agents, "updated_at": _now_iso()}


@app.patch("/api/agents/{agent_id}")
def patch_agent(agent_id: str, update: AgentUpdate, role: str = Depends(_require_role("admin"))) -> Dict[str, Any]:
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No agent fields provided")

    try:
        agent = storage.update_agent(agent_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")

    _event_for(role, "agent", f"Agent updated: {agent_id}", "info")
    return {"agent": agent, "updated_at": _now_iso()}


def _normalise_source_type(source_type: str) -> str:
    return str(source_type or "").strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_import_payload(raw: bytes | str) -> bytes:
    if isinstance(raw, bytes):
        return raw
    return raw.encode("utf-8")


@app.post("/api/agents/import")
async def create_import_job(
    source_type: str = Form(...),
    source_file: UploadFile = File(...),
    append_to_latest: bool = Form(default=False),
    role: str = Depends(_require_role("admin")),
) -> Dict[str, Any]:
    source_type_key = _normalise_source_type(source_type)
    adapter = get_import_adapter(source_type_key)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported source_type "
                f"'{source_type}'. Supported values: openai, openai_codex, codex, gemini, "
                "twitter, gemini_cli, copilot, busy, busy_local, openclaw"
            ),
        )

    raw_data = await source_file.read()
    if not raw_data:
        raise HTTPException(status_code=400, detail="source_file is empty")

    checksum = checksum_payload(raw_data)
    try:
        parse_result = adapter.parse(_coerce_import_payload(raw_data))
    except ValueError as exc:
        _event_for(role, "import", f"Import parse failed for {source_type_key}: {exc}", "error")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _event_for(role, "import", f"Import parse failed for {source_type_key}: {exc}", "error")
        raise HTTPException(status_code=422, detail="Unable to parse provided source payload")

    if parse_result.errors:
        _event_for(role, "import", f"Import warnings for {source_type_key}", "info")

    created = False
    if append_to_latest:
        existing_job = storage.get_latest_import_job_for_source(source_type_key)
        if existing_job:
            job = existing_job
            created = False
            storage.update_import_job_status(job["id"], "parsed")
        else:
            job, created = storage.create_import_job(
                source_type=source_type_key,
                source_metadata=parse_result.source_metadata,
                checksum=checksum,
                status="parsed",
            )
    else:
        job, created = storage.create_import_job(
            source_type=source_type_key,
            source_metadata=parse_result.source_metadata,
            checksum=checksum,
            status="parsed",
        )

    attempted_count = len(parse_result.items)
    if parse_result.items:
        items = [
            {
                "kind": item.kind,
                "agent_scope": item.agent_scope,
                "content": item.content,
                "visibility": item.visibility,
                "source": item.source,
                "thread_id": item.thread_id,
                "message_id": item.message_id,
                "created_at": item.created_at,
                "review_state": item.review_state,
                "author_key": item.author_key,
                "metadata": item.metadata,
                "checksum": item.checksum,
            }
            for item in parse_result.items
        ]
        if append_to_latest:
            existing_checksums = {
                stored["checksum"]
                for stored in storage.list_import_items(import_id=job["id"])
                if stored.get("checksum")
            }
            items = [item for item in items if item.get("checksum") not in existing_checksums]
            if not items:
                storage.append_import_progress_event(
                    import_id=job["id"],
                    phase="parsed",
                    details={"warning_count": len(parse_result.warnings), "new_item_count": 0},
                )
                storage.update_import_job_status(job["id"], "awaiting_review")
                return {
                    "import_id": job["id"],
                    "job": job,
                    "created": created,
                    "counts": parse_result.counts,
                    "items": [],
                    "warnings": list(parse_result.warnings),
                    "errors": list(parse_result.errors),
                    "redaction_hints": adapter.redaction_hints(),
                    "dedupe": {
                        "attempted": attempted_count,
                        "inserted": 0,
                        "skipped": attempted_count,
                    },
                }
            inserted = storage.add_import_items(job["id"], items)
        else:
            inserted = storage.add_import_items(job["id"], items)
    else:
        inserted = []

    storage.update_import_job_status(job["id"], "awaiting_review")
    if parse_result.warnings:
        storage.append_import_progress_event(
            import_id=job["id"],
            phase="parsed",
            details={"warning_count": len(parse_result.warnings), "errors": parse_result.errors},
        )

    return {
        "import_id": job["id"],
        "job": job,
        "created": created,
        "counts": parse_result.counts,
        "items": inserted if parse_result.items else [],
        "warnings": list(parse_result.warnings),
        "errors": list(parse_result.errors),
        "redaction_hints": adapter.redaction_hints(),
        "dedupe": {
            "attempted": attempted_count,
            "inserted": len(inserted),
            "skipped": max(0, attempted_count - len(inserted)),
        } if append_to_latest else None,
    }


def _summarize_import_items(items: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {"total": 0, "pending": 0, "approved": 0, "quarantined": 0, "rejected": 0}
    for item in items:
        summary["total"] += 1
        state = str(item.get("review_state", "pending")).lower()
        if state in summary:
            summary[state] += 1
        else:
            summary["pending"] += 1
    return summary


@app.get("/api/agents/imports")
def list_import_jobs(
    role: str = Depends(_require_role("viewer")),
    limit: int = 25,
    source_type: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    jobs = storage.list_import_jobs(limit=limit)
    if source_type:
        normalized_source = source_type.strip().lower()
        jobs = [job for job in jobs if str(job.get("source_type", "")).lower() == normalized_source]
    if status:
        normalized_status = status.strip().lower()
        jobs = [job for job in jobs if str(job.get("status", "")).lower() == normalized_status]
    enriched_jobs = []
    for job in jobs:
        import_items = storage.list_import_items(import_id=job["id"])
        job_payload = dict(job)
        job_payload["item_counts"] = _summarize_import_items(import_items)
        enriched_jobs.append(job_payload)
    return {"imports": enriched_jobs, "updated_at": _now_iso(), "role": role}


@app.get("/api/agents/import/{import_id}")
def get_import_job(
    import_id: str,
    review_state: Optional[str] = Query(default=None),
    role: str = Depends(_require_role("viewer")),
) -> Dict[str, Any]:
    job = storage.get_import_job(import_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"import '{import_id}' not found")

    items = storage.list_import_items(import_id=import_id, review_state=review_state)
    return {
        "job": job,
        "items": items,
        "updated_at": _now_iso(),
    }


@app.post("/api/agents/import/{import_id}/decision")
def set_import_item_decisions(
    import_id: str,
    payload: ImportDecisionRequest,
    role: str = Depends(_require_role("admin")),
) -> Dict[str, Any]:
    try:
        updated = storage.update_import_items_review_state(
            import_item_ids=payload.import_item_ids,
            review_state=payload.review_state,
            actor=payload.actor or role,
            note=payload.note,
            import_id=import_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not updated:
        raise HTTPException(status_code=404, detail="No matching import items found")

    return {
        "import_id": import_id,
        "updated": updated,
        "updated_count": len(updated),
        "updated_at": _now_iso(),
    }


@app.get("/api/events")
def get_events(limit: int = 25, role: str = Depends(_require_role("viewer"))) -> Dict[str, Any]:
    events = storage.list_events(limit)
    return {"events": events, "updated_at": _now_iso()}


@app.websocket("/api/events/ws")
async def events_ws(ws: WebSocket) -> None:
    query_token = ws.query_params.get("token", "")
    role = _role_from_token(
        auth_token=ws.headers.get("authorization") or "",
        query_token=query_token,
    )
    if not role:
        await ws.close(code=4401)
        return

    await ws.accept()
    last_event_id = None
    try:
        await ws.send_json(
            {
                "type": "events",
                "role": _readable_role_name(role),
                "role_source": _token_source(ws.headers.get("authorization") or "", query_token),
                "events": storage.list_events(25),
                "updated_at": _now_iso(),
                "status": "stream-start",
            }
        )
        while True:
            rows = storage.list_events(25)
            current = rows[0]["id"] if rows else None
            if current != last_event_id:
                last_event_id = current
                await ws.send_json(
                    {
                        "type": "events",
                        "role": _readable_role_name(role),
                        "role_source": _token_source(ws.headers.get("authorization") or "", query_token),
                        "events": rows,
                        "updated_at": _now_iso(),
                    }
                )
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        return
    except Exception:
        await ws.close(code=1011)


@app.get("/api/memory")
def get_memory(
    scope: Optional[str] = None,
    item_type: Optional[str] = Query(default=None, alias="type"),
    role: str = Depends(_require_role("viewer")),
) -> Dict[str, Any]:
    rows = storage.list_memory(scope=scope, item_type=item_type)
    return {"memory": rows, "updated_at": _now_iso()}


@app.post("/api/memory")
def add_memory_entry(payload: MemoryCreate, role: str = Depends(_require_role("admin"))) -> Dict[str, Any]:
    memory = storage.add_memory(scope=payload.scope, memory_type=payload.type, content=payload.content)
    _event_for(role, "memory", f"Memory item added: {memory['id']}", "info")
    return {"memory": memory}


@app.get("/api/chat_history")
def get_chat_history(agent_id: Optional[str] = None, role: str = Depends(_require_role("viewer"))) -> Dict[str, Any]:
    rows = storage.list_chat_history(agent_id=agent_id)
    return {"chat_history": rows, "updated_at": _now_iso()}


@app.post("/api/chat_history")
def add_chat_entry(payload: ChatHistoryCreate, role: str = Depends(_require_role("admin"))) -> Dict[str, Any]:
    row = storage.add_chat_entry(agent_id=payload.agent_id, summary=payload.summary)
    _event_for(role, "chat", f"Chat history entry added for {payload.agent_id}", "info")
    return {"chat": row}


@app.get("/api/runtime/status")
def get_runtime_status(role: str = Depends(_require_role("viewer"))) -> Dict[str, Any]:
    return {"runtime": runtime.get_status(), "updated_at": _now_iso()}


@app.get("/api/runtime/services")
def list_runtime_services(role: str = Depends(_require_role("viewer"))) -> Dict[str, Any]:
    return {**runtime.get_services(), "updated_at": _now_iso()}


@app.post("/api/runtime/services/{service_name}/start")
def start_runtime_service(service_name: str, role: str = Depends(_require_role("admin"))) -> Dict[str, Any]:
    result = _run_runtime_action("start", service_name, role=role)
    return {"updated_at": _now_iso(), **result}


@app.post("/api/runtime/services/{service_name}/stop")
def stop_runtime_service(service_name: str, role: str = Depends(_require_role("admin"))) -> Dict[str, Any]:
    result = _run_runtime_action("stop", service_name, role=role)
    return {"updated_at": _now_iso(), **result}


@app.post("/api/runtime/services/{service_name}/restart")
def restart_runtime_service(service_name: str, role: str = Depends(_require_role("admin"))) -> Dict[str, Any]:
    result = _run_runtime_action("restart", service_name, role=role)
    return {"updated_at": _now_iso(), **result}


def _run_runtime_action(action: str, service_name: str, role: str) -> Dict[str, Any]:
    action_result: RuntimeActionResult = runtime.control_service(service_name, action)
    if action_result.success:
        _event_for(role, "runtime", action_result.message, "info")
        return {
            "success": action_result.success,
            "message": action_result.message,
            "payload": action_result.payload,
        }
    return {
        "success": False,
        "message": action_result.message,
        "payload": action_result.payload,
    }
