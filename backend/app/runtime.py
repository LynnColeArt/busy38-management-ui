"""Busy runtime adapter used by the Management UI."""

from __future__ import annotations

import json
import os
import importlib.util
import inspect
import sys
from pathlib import Path
import urllib.error
import urllib.parse
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

        # Cache lazily-created overlay accessor helpers.
        self._overlay_store = None
        self._overlay_error: Optional[str] = None

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

    @staticmethod
    def _snapshot_to_overlay_payload(snapshot: Any) -> Dict[str, Any]:
        return {
            "overlay_id": str(snapshot.overlay_id),
            "overlay_version": int(snapshot.overlay_version),
            "actor_id": str(snapshot.actor_id),
            "source": str(snapshot.source),
            "source_hash": str(snapshot.source_hash),
            "content": str(snapshot.content),
            "token_count": int(snapshot.token_count),
            "reduced": bool(snapshot.reduced),
            "requested_token_cap": int(snapshot.requested_token_cap),
            "created_by": str(snapshot.created_by),
            "created_at": str(snapshot.created_at),
        }

    def _coerce_limit(self, value: Any, *, default: int = 20, max_limit: int = 200) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(1, min(parsed, max_limit))

    def _load_overlay_store(self) -> bool:
        if self._overlay_store is not None:
            return True
        if self._overlay_error is not None:
            return False
        try:
            from core.context.actor_overlays import ActorOverlayStore
        except Exception as exc:  # pragma: no cover
            self._overlay_error = f"overlay store unavailable: {exc}"
            return False

        try:
            self._overlay_store = ActorOverlayStore
            return True
        except Exception as exc:  # pragma: no cover
            self._overlay_store = None
            self._overlay_error = f"overlay store unavailable: {exc}"
            return False

    def _serialize_overlay_response(self, actor_id: str, payload: Any, *, success: bool, error: Optional[str] = None) -> Dict[str, Any]:
        response = {
            "success": success,
            "actor_id": actor_id,
        }
        if error is not None:
            response["error"] = error
            return response
        if payload is None:
            response["found"] = False
            response["overlay"] = None
            return response
        if isinstance(payload, dict):
            response.update(payload)
            if "overlay" not in payload:
                response["overlay"] = None
        else:
            response["overlay"] = payload
        return response

    def _bridge_get_overlay(self, actor_id: str) -> Optional[Dict[str, Any]]:
        if not self.bridge_url:
            return None

        payload = {"actor_id": actor_id}
        get_paths = (
            "/api/overlay/get",
            "/overlay/get",
            "/runtime/overlay/get",
            f"/api/overlay/{actor_id}",
        )
        for path in get_paths:
            if "{" in path:
                response = self._request_json("GET", path)
            else:
                response = self._request_json("POST", path, payload=payload)
            if isinstance(response, dict) and response:
                if "success" in response or "overlay" in response or "error" in response:
                    return response
        return None

    def _bridge_get_overlay_history(self, actor_id: str, *, limit: int) -> Optional[Dict[str, Any]]:
        if not self.bridge_url:
            return None

        payload = {"actor_id": actor_id, "limit": int(limit)}
        get_paths = (
            "/api/overlay/history",
            "/overlay/history",
            "/runtime/overlay/history",
            f"/api/overlay/{actor_id}/history",
            f"/overlay/{actor_id}/history",
            f"/runtime/overlay/{actor_id}/history",
        )
        for path in get_paths:
            if "{" in path:
                response = self._request_json("GET", path)
            else:
                response = self._request_json("POST", path, payload=payload)
            if isinstance(response, dict) and response:
                if "success" in response or "history" in response or "error" in response:
                    return response
        return None

    def _bridge_write_overlay(
        self,
        actor_id: str,
        content: str,
        *,
        token_cap: Optional[int] = None,
        source: str = "management-ui",
        provenance: Optional[Dict[str, Any]] = None,
        editor_actor: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.bridge_url:
            return None
        payload = {
            "actor_id": actor_id,
            "content": content,
            "source": source,
            "provenance": provenance,
            "editor_actor": editor_actor,
        }
        if token_cap is not None:
            payload["token_cap"] = int(token_cap)
        write_paths = (
            "/api/overlay/write",
            "/overlay/write",
            "/runtime/overlay/write",
        )
        for path in write_paths:
            response = self._request_json("POST", path, payload=payload)
            if isinstance(response, dict) and response:
                return response
        return None

    def get_actor_overlay(self, actor_id: str) -> Dict[str, Any]:
        normalized = str(actor_id or "").strip()
        if not normalized:
            return {"success": False, "actor_id": normalized, "found": False, "error": "actor_id is required"}

        if self._core_services is not None and self._load_overlay_store():
            try:
                with self._overlay_store() as store:  # type: ignore[operator]
                    snapshot = store.get_latest_overlay(normalized)
                if snapshot is None:
                    return {"success": True, "actor_id": normalized, "found": False, "overlay": None}
                return self._serialize_overlay_response(
                    normalized,
                    {
                        "found": True,
                        "overlay": {
                            **self._snapshot_to_overlay_payload(snapshot),
                            "truncated": bool(snapshot.reduced),
                        },
                    },
                    success=True,
                )
            except Exception as exc:
                return self._serialize_overlay_response(
                    normalized,
                    None,
                    success=False,
                    error=str(exc),
                )

        response = self._bridge_get_overlay(normalized)
        if isinstance(response, dict):
            response.setdefault("actor_id", normalized)
            response.setdefault("success", response.get("success", True))
            if "overlay" not in response and response.get("found") is True:
                response["overlay"] = response.get("overlay_data")
            return response

        return {
            "success": bool(self.bridge_url),
            "actor_id": normalized,
            "found": False,
            "overlay": None,
            "error": self._overlay_error,
        }

    def get_actor_overlay_history(self, actor_id: str, *, limit: int = 20) -> Dict[str, Any]:
        normalized = str(actor_id or "").strip()
        if not normalized:
            return {
                "success": False,
                "actor_id": normalized,
                "history": [],
                "error": "actor_id is required",
            }

        normalized_limit = self._coerce_limit(limit, default=20, max_limit=200)

        if self._core_services is not None and self._load_overlay_store():
            try:
                with self._overlay_store() as store:  # type: ignore[operator]
                    snapshots = store.list_overlays(normalized, limit=normalized_limit)
                return {
                    "success": True,
                    "actor_id": normalized,
                    "history": [
                        {
                            **self._snapshot_to_overlay_payload(snapshot),
                            "truncated": bool(snapshot.reduced),
                        }
                        for snapshot in snapshots
                    ],
                    "count": len(snapshots),
                }
            except Exception as exc:
                return {
                    "success": False,
                    "actor_id": normalized,
                    "history": [],
                    "error": str(exc),
                }

        response = self._bridge_get_overlay_history(normalized, limit=normalized_limit)
        if isinstance(response, dict):
            response.setdefault("actor_id", normalized)
            response.setdefault("success", response.get("success", True))
            response.setdefault("count", len(response.get("history", []) if isinstance(response.get("history"), list) else []))
            response.setdefault("history", [])
            return response

        return {
            "success": bool(self.bridge_url),
            "actor_id": normalized,
            "history": [],
            "count": 0,
            "error": self._overlay_error,
        }

    def _bridge_plugin_ui_action(
        self,
        plugin_id: str,
        action_id: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        method: str = "POST",
    ) -> Optional[Dict[str, Any]]:
        if not self.bridge_url:
            return None

        quoted_plugin = urllib.parse.quote_plus(str(plugin_id or "").strip())
        quoted_action = urllib.parse.quote_plus(str(action_id or "").strip())
        if not quoted_plugin or not quoted_action:
            return None

        body = dict(payload or {})
        request_paths = (
            f"/api/plugins/{quoted_plugin}/ui/{quoted_action}",
            f"/runtime/plugins/{quoted_plugin}/ui/{quoted_action}",
            f"/plugin/{quoted_plugin}/ui/{quoted_action}",
            f"/api/plugin/{quoted_plugin}/ui/{quoted_action}",
        )
        method_name = method.strip().upper() or "POST"
        for path in request_paths:
            if method_name == "GET":
                response = self._request_json("GET", path, payload=None)
            else:
                response = self._request_json("POST", path, payload=body)
            if isinstance(response, dict) and response:
                return response
        return None

    def _resolve_plugin_source_path(self, source: Optional[str]) -> Optional[Path]:
        if not isinstance(source, str):
            return None
        raw_source = source.strip()
        if not raw_source:
            return None
        if "://" in raw_source:
            return None

        source_path = Path(raw_source)
        if source_path.is_absolute():
            return source_path if source_path.exists() else None

        candidate_paths = [
            source_path,
            Path(__file__).resolve().parent.parent / source_path,
            Path.cwd() / source_path,
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _parse_entry_point(action: Optional[Dict[str, Any]], action_id: str) -> tuple[str, str]:
        default_module = "actions"
        fallback_function = f"handle_{str(action_id).strip()}"
        if not isinstance(action, dict):
            return default_module, fallback_function

        raw_entry = str(action.get("entry_point") or "").strip()
        if not raw_entry:
            return default_module, fallback_function

        if ":" in raw_entry:
            module_expr, function_expr = raw_entry.split(":", 1)
            module_expr = module_expr.strip()
            function_expr = function_expr.strip()
        else:
            separator = raw_entry.rfind(".")
            if separator < 0:
                return default_module, raw_entry
            module_expr = raw_entry[:separator].strip()
            function_expr = raw_entry[separator + 1 :].strip()

        if not module_expr:
            module_expr = default_module
        if not function_expr:
            function_expr = fallback_function
        return module_expr, function_expr

    @staticmethod
    def _resolve_ui_module_file(ui_path: Path, module_name: str) -> Optional[Path]:
        module_parts = [part for part in module_name.split(".") if part]
        if not module_parts:
            module_parts = ["actions"]

        module_base = ui_path.joinpath(*module_parts)
        module_file = module_base.with_suffix(".py")
        package_init = module_base / "__init__.py"
        if package_init.is_file():
            return package_init
        if module_file.is_file():
            return module_file
        return None

    @staticmethod
    def _coerce_payload_for_runtime_result(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict):
            normalized = dict(payload)
            normalized.pop("success", None)
            normalized.pop("message", None)
            nested_payload = normalized.pop("payload", None)
            if isinstance(nested_payload, dict):
                for key, value in nested_payload.items():
                    if key not in normalized:
                        normalized[key] = value
            normalized["payload"] = nested_payload
            return normalized
        return {"value": payload}

    @staticmethod
    def _coerce_runtime_result(payload: Any) -> RuntimeActionResult:
        if isinstance(payload, RuntimeActionResult):
            return payload
        if not isinstance(payload, dict):
            return RuntimeActionResult(
                success=False,
                message="plugin ui handler returned non-mapping result",
                payload={"value": payload},
            )

        message = str(payload.get("message", "plugin ui action executed"))
        return RuntimeActionResult(
            success=bool(payload.get("success", True)),
            message=message,
            payload=RuntimeAdapter._coerce_payload_for_runtime_result(payload),
        )

    def _invoke_local_ui_handler(
        self,
        plugin_id: str,
        action_id: str,
        action: Optional[Dict[str, Any]],
        source_path: Path,
        method: str,
        payload: Optional[Dict[str, Any]],
    ) -> Optional[RuntimeActionResult]:
        ui_path = source_path / "ui"
        if not ui_path.is_dir():
            return None

        module_name, function_name = self._parse_entry_point(action, action_id)
        module_file = self._resolve_ui_module_file(ui_path, module_name)
        if module_file is None:
            return None

        try:
            module_id = f"busy38_plugin_ui_{id(module_file)}_{module_name}"
            module_spec = importlib.util.spec_from_file_location(module_id, str(module_file))
            if module_spec is None or module_spec.loader is None:
                return RuntimeActionResult(
                    success=False,
                    message="plugin ui module spec was not loadable",
                    payload={
                        "plugin_id": str(plugin_id),
                        "action_id": str(action_id),
                        "module": module_name,
                    },
                )

            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        except Exception as exc:
            return RuntimeActionResult(
                success=False,
                message="plugin ui module load failed",
                payload={
                    "plugin_id": str(plugin_id),
                    "action_id": str(action_id),
                    "module": module_name,
                    "error": str(exc),
                },
            )

        handler = module
        for piece in function_name.split("."):
            if not isinstance(handler, object):
                return None
            handler = getattr(handler, piece, None)
            if handler is None:
                return None

        if not callable(handler):
            return RuntimeActionResult(
                success=False,
                message="plugin ui entry point is not callable",
                payload={
                    "plugin_id": str(plugin_id),
                    "action_id": str(action_id),
                    "module": module_name,
                    "entry_point": function_name,
                },
            )

        normalized_payload = dict(payload or {})
        method_value = str(method or "POST").strip().upper() or "POST"
        context = {
            "plugin_id": str(plugin_id),
            "action_id": str(action_id),
            "method": method_value,
            "source_path": str(source_path),
        }

        try:
            try:
                signature = inspect.signature(handler)
                kwargs: Dict[str, Any] = {}
                if "payload" in signature.parameters:
                    kwargs["payload"] = normalized_payload
                if "method" in signature.parameters:
                    kwargs["method"] = method_value
                if "context" in signature.parameters:
                    kwargs["context"] = context
                if kwargs:
                    return self._coerce_runtime_result(handler(**kwargs))
                return self._coerce_runtime_result(handler())
            except TypeError:
                return self._coerce_runtime_result(handler(normalized_payload, method_value, context))
        except Exception as exc:
            return RuntimeActionResult(
                success=False,
                message=f"plugin ui handler execution failed for '{action_id}'",
                payload={
                    "plugin_id": str(plugin_id),
                    "action_id": str(action_id),
                    "error": str(exc),
                },
            )

    def _run_plugin_local_ui_action(
        self,
        plugin_id: str,
        action_id: str,
        action: Optional[Dict[str, Any]],
        payload: Optional[Dict[str, Any]],
        *,
        plugin_source: Optional[str],
        method: str,
    ) -> Optional[RuntimeActionResult]:
        source_path = self._resolve_plugin_source_path(plugin_source)
        if source_path is None:
            return None
        return self._invoke_local_ui_handler(plugin_id, action_id, action, source_path, method=method, payload=payload)

    def write_actor_overlay(
        self,
        actor_id: str,
        content: str,
        *,
        token_cap: Optional[int] = None,
        source: str = "management-ui",
        provenance: Optional[Dict[str, Any]] = None,
        editor_actor: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_actor = str(actor_id or "").strip()
        normalized_editor = str(editor_actor or normalized_actor or "management-ui").strip()
        if not normalized_actor:
            return {"success": False, "actor_id": normalized_actor, "error": "actor_id is required"}

        if self._core_services is not None and self._load_overlay_store():
            try:
                with self._overlay_store() as store:  # type: ignore[operator]
                    snapshot = store.write_overlay(
                        actor_id=normalized_actor,
                        content=content or "",
                        editor_actor=normalized_editor,
                        source=str(source or "management-ui").strip() or "management-ui",
                        provenance=provenance or {},
                        token_cap=int(token_cap or 5000) if token_cap is not None else 5000,
                    )
                return self._serialize_overlay_response(
                    normalized_actor,
                    {
                        "found": True,
                        "overlay": {
                            **self._snapshot_to_overlay_payload(snapshot),
                            "truncated": bool(snapshot.reduced),
                        },
                        "requested_token_cap": int(token_cap or 5000),
                    },
                    success=True,
                )
            except Exception as exc:
                return self._serialize_overlay_response(
                    normalized_actor,
                    None,
                    success=False,
                    error=str(exc),
                )

        response = self._bridge_write_overlay(
            normalized_actor,
            content,
            token_cap=token_cap,
            source=source,
            provenance=provenance,
            editor_actor=normalized_editor,
        )
        if isinstance(response, dict):
            response.setdefault("actor_id", normalized_actor)
            response.setdefault("success", response.get("success", True))
            return response

        return {
            "success": False,
            "actor_id": normalized_actor,
            "error": self._overlay_error or "overlay write bridge unavailable",
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
                "default_service": self.default_service,
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
                return {
                    "source": "bridge",
                    "connected": True,
                    **health,
                    "default_service": str(health.get("default_service") or self.default_service),
                }

        return {
            "connected": False,
            "source": "none",
            "error": self._load_reason,
            "default_service": self.default_service,
            "services": {},
        }

    def get_services(self) -> Dict[str, Any]:
        if self._core_services is not None:
            defs = self._core_services["get_default_service_definitions"]()
            service_statuses = self._core_services["list_statuses"](defs)
            return {
                "source": "direct",
                "connected": True,
                "default_service": self.default_service,
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
                    "default_service": str(data.get("default_service") or self.default_service),
                }

        return {
            "connected": False,
            "source": "none",
            "error": self._load_reason,
            "default_service": self.default_service,
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

    def _bridge_dispatch_gm_ticket(
        self,
        ticket_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.bridge_url:
            return None

        quoted_ticket = urllib.parse.quote_plus(str(ticket_id or "").strip())
        if not quoted_ticket:
            return None

        body = dict(payload or {})
        for path in (
            f"/api/gm-tickets/{quoted_ticket}/dispatch-mission",
            f"/runtime/gm-tickets/{quoted_ticket}/dispatch-mission",
            f"/gm-tickets/{quoted_ticket}/dispatch-mission",
        ):
            response = self._request_json("POST", path, payload=body)
            if isinstance(response, dict) and response:
                return response
        return None

    def dispatch_gm_ticket(
        self,
        ticket_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> RuntimeActionResult:
        normalized_ticket_id = str(ticket_id or "").strip()
        if not normalized_ticket_id:
            return RuntimeActionResult(
                success=False,
                message="ticket_id is required",
                payload={"ticket_id": ticket_id},
            )

        response = self._bridge_dispatch_gm_ticket(normalized_ticket_id, payload=payload)
        if isinstance(response, dict) and response:
            result_message = str(response.get("message", "gm ticket dispatch queued"))
            if "success" in response:
                return RuntimeActionResult(
                    success=bool(response.get("success")),
                    message=result_message,
                    payload=self._coerce_payload_for_runtime_result(response),
                )
            return RuntimeActionResult(
                success=True,
                message=result_message,
                payload=self._coerce_payload_for_runtime_result(response),
            )

        return RuntimeActionResult(
            success=False,
            message="gm ticket dispatch unavailable",
            payload={"ticket_id": normalized_ticket_id},
        )

    def plugin_ui_action(
        self,
        plugin_id: str,
        action_id: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        method: str = "POST",
        action: Optional[Dict[str, Any]] = None,
        plugin_source: Optional[str] = None,
    ) -> RuntimeActionResult:
        if not plugin_id:
            return RuntimeActionResult(
                success=False,
                message="plugin_id is required",
                payload={"plugin_id": plugin_id, "action_id": action_id},
            )
        if not action_id:
            return RuntimeActionResult(
                success=False,
                message="action_id is required",
                payload={"plugin_id": str(plugin_id)},
            )

        normalized_plugin = str(plugin_id).strip()
        method_value = str(method or "POST").strip().upper()
        local_result = self._run_plugin_local_ui_action(
            normalized_plugin,
            action_id,
            action=action,
            payload=payload,
            plugin_source=plugin_source,
            method=method_value,
        )
        if isinstance(local_result, RuntimeActionResult):
            return local_result

        response = self._bridge_plugin_ui_action(
            normalized_plugin,
            action_id,
            payload=payload,
            method=method_value,
        )

        if isinstance(response, dict) and response:
            return RuntimeActionResult(
                success=bool(response.get("success", True)),
                message=str(response.get("message", "plugin ui action executed")),
                payload=response,
            )

        if not self.bridge_url:
            return RuntimeActionResult(
                success=False,
                message="plugin ui action bridge unavailable",
                payload={"plugin_id": str(plugin_id), "action_id": action_id},
            )

        return RuntimeActionResult(
            success=False,
            message="plugin ui action request did not return a response",
            payload={"plugin_id": str(plugin_id), "action_id": action_id},
        )


def load_runtime_adapter() -> RuntimeAdapter:
    return RuntimeAdapter()
