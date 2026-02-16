"""FastAPI backend for the Busy38 management UI MVP."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="Busy38 Management UI API", version="0.1.0")

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


_STATE: Dict[str, Any] = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "settings": {
        "heartbeat_interval": 30,
        "fallback_budget_per_hour": 420,
        "auto_restart": True,
    },
    "providers": [
        {
            "id": "openai-primary",
            "name": "OpenAI",
            "enabled": True,
            "status": "active",
            "model": "gpt-4.1-mini",
            "endpoint": "https://api.openai.com/v1",
            "priority": 1,
        },
        {
            "id": "ollama-secondary",
            "name": "Ollama",
            "enabled": False,
            "status": "standby",
            "model": "llama3.1:8b",
            "endpoint": "http://127.0.0.1:11434",
            "priority": 2,
        },
    ],
    "agents": [
        {
            "id": "orchestrator-core",
            "name": "Orchestrator",
            "enabled": True,
            "status": "running",
            "role": "main",
            "last_active_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "ops-notify",
            "name": "Notifier",
            "enabled": True,
            "status": "running",
            "role": "support",
            "last_active_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "watch-dog",
            "name": "Security Watchdog",
            "enabled": True,
            "status": "running",
            "role": "monitor",
            "last_active_at": datetime.now(timezone.utc).isoformat(),
        },
    ],
    "events": [
        {
            "id": "evt-1",
            "type": "orchestration",
            "message": "Heartbeat healthy.",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "level": "info",
        },
    ],
    "memory": [
        {
            "id": "memory-1",
            "scope": "global",
            "type": "insight",
            "content": "Bootstrap settings reviewed; OpenAI primary and Ollama fallback were loaded.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "memory-2",
            "scope": "agent:orchestrator-core",
            "type": "handoff",
            "content": "Orchestrator requested user confirmation for critical security rule updates.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ],
    "chat_history": [
        {
            "id": "chat-1",
            "agent_id": "orchestrator-core",
            "summary": "Session started; user requested provider fallback policy refresh.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "chat-2",
            "agent_id": "ops-notify",
            "summary": "Notification policy accepted; waiting for webhook confirmation.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    ],
}


def _touch_state() -> None:
    _STATE["updated_at"] = datetime.now(timezone.utc).isoformat()


def _find_item(container: str, item_id: str) -> Dict[str, Any]:
    for item in _STATE[container]:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"{container[:-1]} '{item_id}' not found")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "busy38-management-ui",
        "updated_at": _STATE["updated_at"],
    }


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    return {"settings": _STATE["settings"], "updated_at": _STATE["updated_at"]}


@app.patch("/api/settings")
def update_settings(update: SettingsUpdate) -> Dict[str, Any]:
    payload = update.model_dump(exclude_unset=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No setting fields provided")
    _STATE["settings"].update({k: v for k, v in payload.items() if v is not None})
    _touch_state()
    return {"settings": _STATE["settings"], "updated_at": _STATE["updated_at"]}


@app.get("/api/providers")
def get_providers() -> Dict[str, Any]:
    return {"providers": _STATE["providers"], "updated_at": _STATE["updated_at"]}


@app.patch("/api/providers/{provider_id}")
def patch_provider(provider_id: str, update: ProviderUpdate) -> Dict[str, Any]:
    target = _find_item("providers", provider_id)
    payload = update.model_dump(exclude_unset=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No provider fields provided")
    target.update({k: v for k, v in payload.items() if v is not None})
    target["status"] = "configured" if target.get("enabled") else "standby"
    _touch_state()
    return {"provider": target, "updated_at": _STATE["updated_at"]}


@app.get("/api/agents")
def get_agents() -> Dict[str, Any]:
    return {"agents": _STATE["agents"], "updated_at": _STATE["updated_at"]}


@app.patch("/api/agents/{agent_id}")
def patch_agent(agent_id: str, update: AgentUpdate) -> Dict[str, Any]:
    target = _find_item("agents", agent_id)
    payload = update.model_dump(exclude_unset=True)
    if not payload:
        raise HTTPException(status_code=400, detail="No agent fields provided")
    target.update({k: v for k, v in payload.items() if v is not None})
    if update.enabled is False:
        target["status"] = "paused"
    elif update.enabled is True:
        target["status"] = "running"
    target["last_active_at"] = datetime.now(timezone.utc).isoformat()
    _touch_state()
    return {"agent": target, "updated_at": _STATE["updated_at"]}


@app.get("/api/events")
def get_events(limit: int = 25) -> Dict[str, Any]:
    normalized = sorted(
        _STATE["events"], key=lambda item: item["created_at"], reverse=True
    )
    return {"events": normalized[:max(1, min(limit, 100))], "updated_at": _STATE["updated_at"]}


@app.get("/api/memory")
def get_memory(scope: Optional[str] = None, item_type: Optional[str] = None) -> Dict[str, Any]:
    rows = _STATE["memory"]
    if scope:
        rows = [row for row in rows if row["scope"] == scope or row["scope"].startswith(scope)]
    if item_type:
        rows = [row for row in rows if row["type"] == item_type]
    return {"memory": rows, "updated_at": _STATE["updated_at"]}


@app.get("/api/chat_history")
def get_chat_history(agent_id: Optional[str] = None) -> Dict[str, Any]:
    rows = _STATE["chat_history"]
    if agent_id:
        rows = [row for row in rows if row["agent_id"] == agent_id]
    return {"chat_history": rows, "updated_at": _STATE["updated_at"]}
