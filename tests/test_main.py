"""Backend API tests for role constraints, runtime wiring, and websocket auth."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from typing import Dict, List, Tuple

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.runtime import RuntimeActionResult


def _load_main_module(admin_token: str, read_token: str, db_path: str):
    env = {
        "MANAGEMENT_ADMIN_TOKEN": admin_token,
        "MANAGEMENT_READ_TOKEN": read_token,
        "MANAGEMENT_API_TOKEN": "",
        "MANAGEMENT_DB_PATH": db_path,
        "BUSY_RUNTIME_PATH": "",
        "BUSY_BRIDGE_URL": "",
    }
    with patch.dict(os.environ, env, clear=False):
        if "backend.app.main" in sys.modules:
            return importlib.reload(sys.modules["backend.app.main"])
        return importlib.import_module("backend.app.main")


class _MockRuntime:
    def __init__(self) -> None:
        self.actions: List[Tuple[str, str]] = []

    def get_status(self) -> Dict[str, object]:
        return {"connected": True, "source": "mock"}

    def get_services(self) -> Dict[str, object]:
        return {
            "connected": True,
            "source": "mock",
            "services": [
                {"name": "busy", "running": True, "pid": 1001, "pid_file": "/tmp/busy.pid", "log_file": "/tmp/busy.log"},
            ],
        }

    def control_service(self, service_name: str, action: str) -> RuntimeActionResult:
        self.actions.append((service_name, action))
        if service_name != "busy":
            return RuntimeActionResult(
                success=False,
                message=f"service '{service_name}' not found",
                payload={"service": service_name, "action": action},
            )
        if action not in {"start", "stop", "restart"}:
            return RuntimeActionResult(
                success=False,
                message=f"unsupported action: {action}",
                payload={"service": service_name, "action": action},
            )
        return RuntimeActionResult(
            success=True,
            message=f"{action} queued for {service_name}",
            payload={"service": service_name, "action": action},
        )


class TestManagementApiRolesAndRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admin_token = "admin-token"
        cls.read_token = "read-token"

    def setUp(self):
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_file.close()
        self.main = _load_main_module(
            admin_token=self.admin_token,
            read_token=self.read_token,
            db_path=self.db_file.name,
        )
        self.client = TestClient(self.main.app)
        self.client.__enter__()
        self.runtime = _MockRuntime()
        self.main.runtime = self.runtime

    def tearDown(self):
        self.client.__exit__(None, None, None)
        os.remove(self.db_file.name)

    def test_viewer_and_admin_tokens(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        settings = self.client.get("/api/settings", headers=read_headers)
        self.assertEqual(settings.status_code, 200)
        self.assertEqual(settings.json()["role"], "viewer")

        denied = self.client.patch("/api/settings", headers=read_headers, json={"auto_restart": False})
        self.assertEqual(denied.status_code, 403)

        updated = self.client.patch("/api/settings", headers=admin_headers, json={"auto_restart": False})
        self.assertEqual(updated.status_code, 200)
        self.assertIn("settings", updated.json())

        unauthorized = self.client.get("/api/settings")
        self.assertEqual(unauthorized.status_code, 401)

    def test_runtime_actions_require_admin(self):
        read_headers = {"Authorization": f"Bearer {self.read_token}"}
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        service_list = self.client.get("/api/runtime/services", headers=read_headers)
        self.assertEqual(service_list.status_code, 200)
        self.assertIn("services", service_list.json())

        blocked = self.client.post("/api/runtime/services/busy/start", headers=read_headers)
        self.assertEqual(blocked.status_code, 403)

        started = self.client.post("/api/runtime/services/busy/start", headers=admin_headers)
        self.assertEqual(started.status_code, 200)
        payload = started.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["payload"]["service"], "busy")
        self.assertEqual(self.runtime.actions[-1], ("busy", "start"))

    def test_events_websocket_requires_token_and_returns_role(self):
        with self.assertRaises(WebSocketDisconnect) as exc:
            with self.client.websocket_connect("/api/events/ws"):
                pass
        self.assertEqual(exc.exception.code, 4401)

        with self.client.websocket_connect("/api/events/ws?token=read-token") as ws:
            first = ws.receive_json()
            self.assertEqual(first["type"], "events")
            self.assertEqual(first["role"], "viewer")
            self.assertIn("events", first)
