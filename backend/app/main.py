"""FastAPI backend for the Busy38 management UI MVP."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from . import storage


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


def _check_auth(token: Optional[str] = None, query_token: Optional[str] = None) -> None:
    expected = os.getenv("MANAGEMENT_API_TOKEN", "").strip()
    if not expected:
        return

    if token and token.startswith("Bearer "):
        token = token[7:].strip()
    token = token or query_token or ""
    if token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_auth(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    token: Optional[str] = Query(default=None, alias="token"),
) -> None:
    _check_auth(auth.credentials if auth else None, token)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "busy38-management-ui",
        "updated_at": _now_iso(),
    }


@app.get("/api/settings", dependencies=[Depends(_require_auth)])
def get_settings() -> Dict[str, Any]:
    settings = storage.get_settings()
    settings["auto_restart"] = bool(settings["auto_restart"])
    return {"settings": settings, "updated_at": settings["updated_at"]}


@app.patch("/api/settings", dependencies=[Depends(_require_auth)])
def update_settings(update: SettingsUpdate) -> Dict[str, Any]:
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No setting fields provided")

    settings = storage.set_settings(payload)
    storage.append_event("settings", "Settings updated via management UI", "info")
    return {"settings": settings, "updated_at": settings["updated_at"]}


@app.get("/api/providers", dependencies=[Depends(_require_auth)])
def get_providers() -> Dict[str, Any]:
    providers = storage.list_providers()
    return {"providers": providers, "updated_at": _now_iso()}


@app.patch("/api/providers/{provider_id}", dependencies=[Depends(_require_auth)])
def patch_provider(provider_id: str, update: ProviderUpdate) -> Dict[str, Any]:
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No provider fields provided")

    try:
        provider = storage.update_provider(provider_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    storage.append_event("provider", f"Provider updated: {provider_id}", "info")
    provider["enabled"] = bool(provider["enabled"])
    return {"provider": provider, "updated_at": _now_iso()}


@app.get("/api/agents", dependencies=[Depends(_require_auth)])
def get_agents() -> Dict[str, Any]:
    agents = storage.list_agents()
    return {"agents": agents, "updated_at": _now_iso()}


@app.patch("/api/agents/{agent_id}", dependencies=[Depends(_require_auth)])
def patch_agent(agent_id: str, update: AgentUpdate) -> Dict[str, Any]:
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No agent fields provided")

    try:
        agent = storage.update_agent(agent_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")

    storage.append_event("agent", f"Agent updated: {agent_id}", "info")
    return {"agent": agent, "updated_at": _now_iso()}


@app.get("/api/events", dependencies=[Depends(_require_auth)])
def get_events(limit: int = 25) -> Dict[str, Any]:
    events = storage.list_events(limit)
    return {"events": events, "updated_at": _now_iso()}


@app.get("/api/memory", dependencies=[Depends(_require_auth)])
def get_memory(scope: Optional[str] = None, item_type: Optional[str] = Query(default=None, alias="type")) -> Dict[str, Any]:
    rows = storage.list_memory(scope=scope, item_type=item_type)
    return {"memory": rows, "updated_at": _now_iso()}


@app.post("/api/memory", dependencies=[Depends(_require_auth)])
def add_memory_entry(payload: MemoryCreate) -> Dict[str, Any]:
    memory = storage.add_memory(scope=payload.scope, memory_type=payload.type, content=payload.content)
    storage.append_event("memory", f"Memory item added: {memory['id']}", "info")
    return {"memory": memory}


@app.get("/api/chat_history", dependencies=[Depends(_require_auth)])
def get_chat_history(agent_id: Optional[str] = None) -> Dict[str, Any]:
    rows = storage.list_chat_history(agent_id=agent_id)
    return {"chat_history": rows, "updated_at": _now_iso()}


@app.post("/api/chat_history", dependencies=[Depends(_require_auth)])
def add_chat_entry(payload: ChatHistoryCreate) -> Dict[str, Any]:
    row = storage.add_chat_entry(agent_id=payload.agent_id, summary=payload.summary)
    storage.append_event("chat", f"Chat history entry added for {payload.agent_id}", "info")
    return {"chat": row}
