"""Busy runtime adapter used by the Management UI."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RuntimeActionResult:
    success: bool
    message: str
    payload: Dict[str, Any]


class RuntimeAdapter:
    """Thin optional adapter for Busy runtime orchestration controls."""

    def __init__(self) -> None:
        self.runtime_path = os.getenv("BUSY_RUNTIME_PATH", "").strip()
        self.bridge_url = os.getenv("BUSY_BRIDGE_URL", "").strip()
        self.default_service = (os.getenv("BUSY_RUNTIME_SERVICE", "busy") or "busy").strip()

        self._core_services = None
        self._core_bridge = None
        self._load_reason: Optional[str] = None
        self._use_direct = False

        if self.runtime_path:
            self._load_reason = self._bootstrap_direct(self.runtime_path)
            self._use_direct = self._core_services is not None
        elif self.bridge_url:
            # HTTP fallback is still useful with only bridge URL
            self._load_reason = "direct runtime path not configured"
        else:
            self._load_reason = "no runtime adapter configured"

    def _bootstrap_direct(self, runtime_path: str) -> str:
        if not os.path.isdir(runtime_path):
            return f"runtime path does not exist: {runtime_path}"

        if runtime_path not in sys.path:
            sys.path.insert(0, runtime_path)

        try:
            from core.service.definitions import get_default_service_definitions
            from core.service.manager import get_status, list_statuses, restart_service, start_service, stop_service
            from core.bridge.runtime import get_orchestrator

            self._core_services = {
                "get_default_service_definitions": get_default_service_definitions,
                "get_status": get_status,
                "list_statuses": list_statuses,
                "start_service": start_service,
                "stop_service": stop_service,
                "restart_service": restart_service,
                "get_orchestrator": get_orchestrator,
            }
            return ""
        except Exception as exc:  # pragma: no cover - defensive
            self._core_services = None
            return f"could not import busy runtime modules: {exc}"

    def _request_json(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if not self.bridge_url:
            return None
        url = f"{self.bridge_url.rstrip('/')}" + path
        data = None
        headers = {"accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"

        req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=4) as response:
                raw = response.read()
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except urllib.error.URLError:
            return None
        except Exception:
            return None

    def _serialize_direct_service(self, service_status: Any) -> Dict[str, Any]:
        return {
            "name": service_status.name,
            "running": bool(service_status.running),
            "pid": service_status.pid,
            "pid_file": str(service_status.pid_file),
            "log_file": str(service_status.log_file),
        }

    def is_connected(self) -> bool:
        if self._core_services is not None:
            return True
        if self.bridge_url:
            health = self._request_json("GET", "/health")
            return isinstance(health, dict)
        return False

    def get_status(self) -> Dict[str, Any]:
        if self._core_services is not None:
            defs = self._core_services["get_default_service_definitions"]()
            service_statuses = self._core_services["list_statuses"](defs)
            payload: Dict[str, Any] = {
                "source": "direct",
                "connected": True,
                "services": {
                    name: self._serialize_direct_service(status) for name, status in service_statuses.items()
                },
            }
            orch = self._core_services["get_orchestrator"]()
            if orch is not None:
                try:
                    payload["orchestrator"] = orch.get_status()
                except Exception:
                    payload["orchestrator"] = {"available": False, "error": "orchestrator status unavailable"}
            return payload

        if self.bridge_url:
            health = self._request_json("GET", "/health")
            if isinstance(health, dict):
                return {"source": "bridge", "connected": True, **health}

        return {
            "connected": False,
            "source": "none",
            "error": self._load_reason,
            "services": {},
        }

    def get_services(self) -> Dict[str, Any]:
        if self._core_services is not None:
            defs = self._core_services["get_default_service_definitions"]()
            service_statuses = self._core_services["list_statuses"](defs)
            return {
                "source": "direct",
                "connected": True,
                "services": [
                    self._serialize_direct_service(service_statuses[name])
                    for name in sorted(service_statuses)
                ],
            }

        if self.bridge_url:
            data = self._request_json("GET", "/api/runtime/services")
            if isinstance(data, dict):
                return {
                    "source": "bridge",
                    "connected": True,
                    **data,
                }

        return {
            "connected": False,
            "source": "none",
            "error": self._load_reason,
            "services": [],
        }

    def _run_action(self, service_name: str, action: str) -> RuntimeActionResult:
        action = action.strip().lower()
        if self._core_services is not None:
            defs = self._core_services["get_default_service_definitions"]()
            service = defs.get(service_name)
            if service is None:
                return RuntimeActionResult(
                    success=False,
                    message=f"unknown service: {service_name}",
                    payload={"service": service_name},
                )

            if action == "start":
                ok, msg = self._core_services["start_service"](service)
            elif action == "stop":
                ok, msg = self._core_services["stop_service"](service_name)
            elif action == "restart":
                ok, msg = self._core_services["restart_service"](service)
            else:
                return RuntimeActionResult(
                    success=False,
                    message=f"unsupported action: {action}",
                    payload={"service": service_name},
                )

            return RuntimeActionResult(
                success=bool(ok),
                message=str(msg),
                payload={"service": service_name, "action": action},
            )

        if self.bridge_url:
            for path in (
                f"/api/runtime/services/{service_name}/{action}",
                f"/api/runtime/{service_name}/{action}",
                f"/runtime/services/{service_name}/{action}",
            ):
                data = self._request_json("POST", path)
                if isinstance(data, dict):
                    if "success" in data:
                        return RuntimeActionResult(
                            success=bool(data.get("success")),
                            message=str(data.get("message", "runtime action executed")),
                            payload=data,
                        )

        return RuntimeActionResult(
            success=False,
            message="runtime control unavailable",
            payload={"service": service_name, "action": action},
        )

    def control_service(self, service_name: str, action: str) -> RuntimeActionResult:
        return self._run_action(service_name, action)


def load_runtime_adapter() -> RuntimeAdapter:
    return RuntimeAdapter()
