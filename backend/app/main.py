"""FastAPI backend for the Busy38 management UI MVP."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

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


def _event_for(_role: str, event_type: str, message: str, level: str = "info") -> Dict[str, Any]:
    payload = storage.append_event(event_type, message, level)
    return payload


def _redact_metadata(metadata: Optional[str], role: str) -> Optional[str]:
    if role == "admin":
        return metadata
    if not metadata:
        return metadata
    try:
        parsed = json.loads(metadata)
        if isinstance(parsed, dict):
            return json.dumps({k: "***redacted***" for k in parsed.keys()}, separators=(",", ":"))
    except Exception:
        pass
    return "***redacted***"


def _sanitize_provider(provider: Dict[str, Any], role: str) -> Dict[str, Any]:
    payload = dict(provider)
    payload["metadata"] = _redact_metadata(payload.get("metadata"), role)
    if role != "admin" and isinstance(payload.get("metadata"), str):
        if payload["metadata"] != "***redacted***":
            payload["metadata"] = "{...}"
    return payload


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
def get_providers(role: str = Depends(_require_role("viewer"))) -> Dict[str, Any]:
    providers = [_sanitize_provider(p, role) for p in storage.list_providers()]
    return {"providers": providers, "updated_at": _now_iso()}


@app.patch("/api/providers/{provider_id}")
def patch_provider(
    provider_id: str,
    update: ProviderUpdate,
    role: str = Depends(_require_role("admin")),
) -> Dict[str, Any]:
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No provider fields provided")

    try:
        provider = storage.update_provider(provider_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    _event_for(role, "provider", f"Provider updated: {provider_id}", "info")
    provider["enabled"] = bool(provider["enabled"])
    return {"provider": _sanitize_provider(provider, role), "updated_at": _now_iso()}


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
