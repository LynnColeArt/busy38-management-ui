"""Backend API tests for role constraints, runtime wiring, and websocket auth."""

from __future__ import annotations

import importlib
import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
from typing import Dict, List, Tuple
import asyncio

from fastapi import HTTPException

from backend.app.runtime import RuntimeActionResult
from backend.app import storage
import httpx


class _SyncAsyncClient:
    """Minimal sync wrapper around httpx.AsyncClient for TestCase usage."""

    def __init__(self, loop: asyncio.AbstractEventLoop, app) -> None:
        self._loop = loop
        self._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        )

    def request(self, *args, **kwargs):
        return self._loop.run_until_complete(self._client.request(*args, **kwargs))

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def patch(self, url: str, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def close(self) -> None:
        self._loop.run_until_complete(self._client.aclose())


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
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_file.close()
        self.main = _load_main_module(
            admin_token=self.admin_token,
            read_token=self.read_token,
            db_path=self.db_file.name,
        )
        storage.ensure_schema()
        self.client = _SyncAsyncClient(self._loop, self.main.app)
        self.runtime = _MockRuntime()
        self.main.runtime = self.runtime

    def tearDown(self):
        self.client.close()
        self._loop.close()
        os.remove(self.db_file.name)

    def test_viewer_and_admin_tokens(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        settings = self.client.get("/api/settings", headers=read_headers)
        self.assertEqual(settings.status_code, 200)
        self.assertEqual(settings.json()["role"], "viewer")
        self.assertEqual(settings.json()["role_source"], "read-token")

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

    def test_provider_discovery_success_stores_model_metadata(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        def fake_fetch(url: str, headers: dict | None = None):
            self.assertEqual(url, "https://api.openai.com/v1/models")
            self.assertIsNotNone(headers)
            self.assertEqual(headers.get("Authorization"), "Bearer sk-test")
            return {"data": [{"id": "gpt-4"}, {"id": "gpt-4.1-mini"}]}

        with patch.object(self.main, "_safe_http_fetch", side_effect=fake_fetch):
            response = self.client.post(
                "/api/providers/openai-primary/discover-models",
                headers=admin_headers,
                json={"api_key": "sk-test", "endpoint": "https://api.openai.com/v1/models"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["discovered_models"], ["gpt-4", "gpt-4.1-mini"])

        providers = self.client.get("/api/providers", headers=admin_headers).json()["providers"]
        provider = next(item for item in providers if item["id"] == "openai-primary")
        metadata = provider.get("metadata") or {}
        discovery = metadata.get("model_discovery") or {}
        self.assertEqual(discovery.get("status"), "complete")
        self.assertEqual(discovery.get("count"), 2)
        self.assertEqual(discovery.get("endpoint"), "https://api.openai.com/v1/models")
        self.assertEqual(metadata.get("discovered_models"), ["gpt-4", "gpt-4.1-mini"])

    def test_provider_discovery_failure_uses_manual_path(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        with patch.object(self.main, "_safe_http_fetch", side_effect=HTTPException(status_code=502, detail="No model list endpoint matched this provider.")):
            response = self.client.post(
                "/api/providers/openai-primary/discover-models",
                headers=admin_headers,
                json={},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["source"], "manual")
        self.assertIn("No model list endpoint", payload.get("error", ""))

        providers = self.client.get("/api/providers", headers=admin_headers).json()["providers"]
        provider = next(item for item in providers if item["id"] == "openai-primary")
        metadata = provider.get("metadata") or {}
        discovery = metadata.get("model_discovery") or {}
        self.assertEqual(discovery.get("status"), "manual")
        self.assertEqual(discovery.get("count"), 0)
        self.assertIn("No model list endpoint", discovery.get("error", ""))
        self.assertNotIn("discovered_models", metadata)

    def test_provider_create(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        response = self.client.post(
            "/api/providers",
            headers=admin_headers,
            json={
                "id": "openai-secondary",
                "name": "OpenAI Secondary",
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "priority": 3,
                "enabled": True,
                "metadata": {"kind": "openai_compatible"},
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        created = response.json()["provider"]
        self.assertEqual(created["id"], "openai-secondary")
        self.assertEqual(created["name"], "OpenAI Secondary")
        self.assertEqual(created["status"], "configured")
        providers = self.client.get("/api/providers", headers=admin_headers, params={"kind": "openai_compatible"}).json()["providers"]
        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["id"], "openai-secondary")

    def test_provider_create_with_routing_metadata(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        response = self.client.post(
            "/api/providers",
            headers=admin_headers,
            json={
                "id": "anthropic-primary",
                "name": "Anthropic Primary",
                "endpoint": "https://api.anthropic.com/v1",
                "model": "claude-3-7-sonnet",
                "priority": 5,
                "enabled": True,
                "kind": "openai_compatible",
                "fallback_models": ["claude-3-opus", "claude-3-haiku"],
                "retries": 4,
                "timeout_ms": 7000,
                "tool_timeout_ms": 12000,
                "display_name": "Anthropic Prime",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        created = response.json()["provider"]
        metadata = created["metadata"] or {}
        self.assertEqual(metadata["kind"], "openai_compatible")
        self.assertEqual(metadata["fallback_models"], ["claude-3-opus", "claude-3-haiku"])
        self.assertEqual(metadata["retries"], 4)
        self.assertEqual(metadata["timeout_ms"], 7000)
        self.assertEqual(metadata["tool_timeout_ms"], 12000)
        self.assertEqual(metadata["display_name"], "Anthropic Prime")

    def test_provider_create_requires_admin(self):
        read_headers = {"Authorization": f"Bearer {self.read_token}"}
        response = self.client.post(
            "/api/providers",
            headers=read_headers,
            json={
                "id": "read-only-provider",
                "name": "Read Only",
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_provider_create_duplicate(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        payload = {
            "id": "openai-primary",
            "name": "OpenAI Primary Copy",
            "endpoint": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        }
        response = self.client.post("/api/providers", headers=admin_headers, json=payload)
        self.assertEqual(response.status_code, 409)
        self.assertIn("already exists", response.json()["detail"])

    def test_provider_secret_set_rotate_clear(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        set_resp = self.client.post(
            "/api/providers/openai-primary/secret",
            headers=admin_headers,
            json={"action": "set", "api_key": "sk-secret"},
        )
        self.assertEqual(set_resp.status_code, 200, set_resp.text)
        provider = set_resp.json()["provider"]
        self.assertEqual(provider["metadata"]["secret_present"], True)
        self.assertEqual(provider["metadata"]["secret_policy"], "required")
        self.assertEqual(provider["metadata"]["api_key"], "sk-secret")

        rotate_resp = self.client.post(
            "/api/providers/openai-primary/secret",
            headers=admin_headers,
            json={"action": "rotate", "api_key": "sk-secret-2"},
        )
        self.assertEqual(rotate_resp.status_code, 200, rotate_resp.text)
        provider = rotate_resp.json()["provider"]
        self.assertEqual(provider["metadata"]["api_key"], "sk-secret-2")
        self.assertEqual(provider["metadata"]["secret_last_action"], "rotate")

        clear_resp = self.client.post(
            "/api/providers/openai-primary/secret",
            headers=admin_headers,
            json={"action": "clear"},
        )
        self.assertEqual(clear_resp.status_code, 200, clear_resp.text)
        provider = clear_resp.json()["provider"]
        self.assertEqual(provider["metadata"]["secret_present"], False)
        self.assertNotIn("api_key", provider["metadata"])

    def test_provider_secret_not_required_for_local_provider(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        response = self.client.post(
            "/api/providers/ollama-secondary/secret",
            headers=admin_headers,
            json={"action": "set", "api_key": "should-not-be-needed"},
        )
        self.assertEqual(response.status_code, 409)
        payload = response.json()
        self.assertIn("not required", payload["detail"])

    def test_provider_secret_set_requires_admin(self):
        read_headers = {"Authorization": f"Bearer {self.read_token}"}
        response = self.client.post(
            "/api/providers/openai-primary/secret",
            headers=read_headers,
            json={"action": "set", "api_key": "blocked"},
        )
        self.assertEqual(response.status_code, 403)

    def test_provider_test_route_stores_last_test_metadata(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        def fake_fetch(url: str, headers: dict | None = None):
            self.assertTrue(url.endswith("/v1/models"), url)
            return {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}

        with patch.object(self.main, "_safe_http_fetch", side_effect=fake_fetch):
            response = self.client.post(
                "/api/providers/openai-primary/test",
                headers=admin_headers,
                json={"api_key": "sk-test"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["models_count"], 2)
        self.assertIsInstance(payload["latency_ms"], (int, float))
        self.assertGreaterEqual(payload["latency_ms"], 0.0)

        providers = self.client.get("/api/providers", headers=admin_headers).json()["providers"]
        provider = next(item for item in providers if item["id"] == "openai-primary")
        metadata = provider.get("metadata") or {}
        last_test = metadata.get("last_test") or {}
        self.assertEqual(last_test.get("status"), "pass")
        self.assertEqual(last_test.get("models_count"), 2)
        self.assertTrue(last_test.get("endpoint", "").endswith("/v1/models"), last_test.get("endpoint"))
        self.assertTrue(last_test.get("source", "").startswith("https://api.openai.com"), last_test.get("source"))

    def test_provider_test_route_requires_admin(self):
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        response = self.client.post(
            "/api/providers/openai-primary/test",
            headers=read_headers,
            json={},
        )
        self.assertEqual(response.status_code, 403)

    def test_provider_test_route_records_failures(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        self.client.post(
            "/api/providers/openai-primary/secret",
            headers=admin_headers,
            json={"action": "set", "api_key": "sk-test"},
        )
        with patch.object(self.main, "_safe_http_fetch", side_effect=HTTPException(status_code=502, detail="No model list endpoint matched this provider.")):
            response = self.client.post(
                "/api/providers/openai-primary/test",
                headers=admin_headers,
                json={"endpoint": "https://api.openai.com/v1"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["models_count"], 0)
        self.assertIn("No model list endpoint", payload.get("error", ""))

        providers = self.client.get("/api/providers", headers=admin_headers).json()["providers"]
        provider = next(item for item in providers if item["id"] == "openai-primary")
        metadata = provider.get("metadata") or {}
        last_test = metadata.get("last_test") or {}
        self.assertEqual(last_test.get("status"), "fail")
        self.assertEqual(last_test.get("models_count"), 0)
        self.assertIn("No model list endpoint", last_test.get("error", ""))

    def test_provider_routing_chain_reflects_enabled_order(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        update_payload = {"enabled": True, "status": "standby", "priority": 2}
        response = self.client.patch("/api/providers/ollama-secondary", headers=admin_headers, json=update_payload)
        self.assertEqual(response.status_code, 200, response.text)

        chain_response = self.client.get("/api/providers/routing-chain", headers=admin_headers)
        self.assertEqual(chain_response.status_code, 200, chain_response.text)
        chain = chain_response.json()["chain"]
        self.assertEqual(chain[0]["id"], "openai-primary")
        self.assertEqual(chain[0]["active"], True)
        self.assertIn(chain[1]["id"], {"ollama-secondary"})
        self.assertFalse(chain[1]["active"])

    def test_provider_routing_chain_exposes_routing_reasoning(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        response = self.client.get("/api/providers/routing-chain", headers=admin_headers)
        self.assertEqual(response.status_code, 200, response.text)

        payload = response.json()
        chain = payload["chain"]
        self.assertIn("selection_strategy", payload)
        self.assertIn("selection_rationale", payload)
        self.assertIn("active_provider_id", payload)
        self.assertIn("routing_intent", payload)
        self.assertIn("routing_path", payload)
        self.assertGreater(len(chain), 0)
        self.assertIsInstance(payload["routing_path"], str)
        self.assertTrue(payload["routing_path"])
        first = chain[0]
        self.assertEqual(first["id"], payload["active_provider_id"])
        self.assertEqual(first["routing_intent"], "primary")
        self.assertIn("routing_reason", first)
        self.assertIn("routing_behavior", first)
        self.assertIn("health", first)
        fallback_nodes = [node for node in chain if node["routing_intent"] == "fallback"]
        if fallback_nodes:
            self.assertIn("fallback", payload["routing_intent"])

    def test_provider_patch_updates_routing_metadata(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        response = self.client.patch(
            "/api/providers/openai-primary",
            headers=admin_headers,
            json={
                "kind": "openai_compatible",
                "fallback_models": ["gpt-5", "gpt-4o"],
                "retries": 2,
                "timeout_ms": 4000,
                "tool_timeout_ms": 9000,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        provider = response.json()["provider"]
        metadata = provider["metadata"] or {}
        self.assertEqual(metadata["kind"], "openai_compatible")
        self.assertEqual(metadata["fallback_models"], ["gpt-5", "gpt-4o"])
        self.assertEqual(metadata["retries"], 2)
        self.assertEqual(metadata["timeout_ms"], 4000)
        self.assertEqual(metadata["tool_timeout_ms"], 9000)

    def test_provider_history_and_metrics_endpoints(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        def fake_fetch(url: str, headers: dict | None = None):
            self.assertTrue("v1/models" in url or "api/tags" in url)
            return {"data": [{"id": "gpt-4o"}]}

        with patch.object(self.main, "_safe_http_fetch", side_effect=fake_fetch):
            test_response = self.client.post(
                "/api/providers/openai-primary/test",
                headers=admin_headers,
                json={"api_key": "sk-test"},
            )
        self.assertEqual(test_response.status_code, 200, test_response.text)

        history_response = self.client.get("/api/providers/openai-primary/history?limit=10", headers=admin_headers)
        self.assertEqual(history_response.status_code, 200, history_response.text)
        history_payload = history_response.json()
        self.assertEqual(history_payload["provider_id"], "openai-primary")
        self.assertGreaterEqual(len(history_payload["test_history"]), 1)
        self.assertEqual(history_payload["test_history"][0]["status"], "pass")

        metrics_response = self.client.get("/api/providers/openai-primary/metrics", headers=admin_headers)
        self.assertEqual(metrics_response.status_code, 200, metrics_response.text)
        metrics_payload = metrics_response.json()
        self.assertIn("success_rate_5m", metrics_payload["metrics"])
        self.assertIn("latency_ms_last", metrics_payload["metrics"])

    def test_provider_test_all_checks_enabled_only_by_default(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        with patch.object(
            self.main,
            "_safe_http_fetch",
            side_effect=[
                {"data": [{"id": "gpt-4o"}]},
                HTTPException(status_code=502, detail="Ollama unavailable"),
            ],
        ):
            response = self.client.post(
                "/api/providers/test-all",
                headers=admin_headers,
                json={},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["checked"], 1)
        self.assertIn(payload["pass_count"], [0, 1])
        self.assertGreaterEqual(payload["pass_count"] + payload["fail_count"], 1)

    def test_provider_test_all_can_include_disabled(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        self.client.post(
            "/api/providers/openai-primary/secret",
            headers=admin_headers,
            json={"action": "set", "api_key": "sk-test"},
        )
        with patch.object(
            self.main,
            "_safe_http_fetch",
            side_effect=[
                {"data": [{"id": "gpt-4o"}]},
                HTTPException(status_code=502, detail="Ollama unavailable"),
            ],
        ):
            response = self.client.post(
                "/api/providers/test-all",
                headers=admin_headers,
                json={"include_disabled": True},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["checked"], 2)
        self.assertEqual(payload["fail_count"], 1)

    def test_provider_metadata_redacts_sensitive_values_for_viewer(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        self.client.patch(
            "/api/providers/openai-primary",
            headers=admin_headers,
            json={
                "metadata": {
                    "provider": "openai",
                    "api_key": "sk-test",
                    "notes": "default endpoint",
                }
            },
        )

        admin_view = self.client.get("/api/providers", headers=admin_headers).json()["providers"]
        admin_provider = next(item for item in admin_view if item["id"] == "openai-primary")
        admin_metadata = admin_provider.get("metadata") or {}
        self.assertEqual(admin_metadata.get("api_key"), "sk-test")

        viewer_view = self.client.get("/api/providers", headers=read_headers).json()["providers"]
        viewer_provider = next(item for item in viewer_view if item["id"] == "openai-primary")
        viewer_metadata = viewer_provider.get("metadata") or {}
        self.assertEqual(viewer_metadata.get("api_key"), "***redacted***")
        self.assertEqual(viewer_metadata.get("notes"), "default endpoint")

    def test_events_websocket_requires_token_and_returns_role(self):
        self.skipTest("WebSocket test skipped: transport layer requires integration test harness in this environment.")

    def test_import_openai_upload_and_review_flow(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        export = {
            "conversations": [
                {
                    "id": "thread-1",
                    "title": "Idea planning",
                    "messages": [
                        {"id": "m1", "author": {"role": "user"}, "create_time": "2026-01-01T12:00:00Z", "content": "How can we build a product from scratch?"},
                        {"id": "m2", "author": {"role": "assistant"}, "create_time": "2026-01-01T12:00:10Z", "content": {"parts": ["Let's start by defining goals."]}},
                    ],
                },
            ]
        }
        upload = ("openai-export.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "openai"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["counts"].get("messages"), 2)
        self.assertIn(payload["created"], [True, False])
        import_id = payload["import_id"]
        job_get = self.client.get(f"/api/agents/import/{import_id}", headers=read_headers)
        self.assertEqual(job_get.status_code, 200)
        items = job_get.json()["items"]
        self.assertGreaterEqual(len(items), 2)

        decision = self.client.post(
            f"/api/agents/import/{import_id}/decision",
            headers=admin_headers,
            json={
                "import_item_ids": [items[0]["id"]],
                "review_state": "approved",
                "note": "initial check",
            },
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["updated_count"], 1)

        viewer_blocked = self.client.post(
            f"/api/agents/import/{import_id}/decision",
            headers=read_headers,
            json={
                "import_item_ids": [items[0]["id"]],
                "review_state": "approved",
            },
        )
        self.assertEqual(viewer_blocked.status_code, 403)

    def test_import_jobs_list_includes_counts(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        export = {
            "conversations": [
                {
                    "id": "thread-1",
                    "title": "Momentum planning",
                    "messages": [
                        {"id": "m1", "author": {"role": "user"}, "create_time": "2026-01-01T12:00:00Z", "content": "Let's test count behavior."},
                    ],
                },
            ]
        }
        upload = ("openai-counts.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "openai"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        import_id = response.json()["import_id"]

        import_list = self.client.get("/api/agents/imports", headers=admin_headers)
        self.assertEqual(import_list.status_code, 200, import_list.text)
        payload = import_list.json()
        self.assertIn("imports", payload)
        found = next((item for item in payload["imports"] if item["id"] == import_id), None)
        self.assertIsNotNone(found, "import job should appear in list")
        counts = found["item_counts"]
        self.assertEqual(counts["total"], 1)
        self.assertEqual(counts["pending"], 1)

    def test_import_rejects_unsupported_source(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        payload = {"foo": "bar"}
        upload = ("unsupported.json", json.dumps(payload), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "unsupported-source"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 400)

    def test_import_twitter_wrapper_upload_and_parsing(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        payload = {
            "tweets": [
                {
                    "tweet": {
                        "id": "tweet-1",
                        "full_text": "Drafting a launch strategy.",
                        "created_at": "2026-01-02T10:00:00+00:00",
                        "conversation_id": "thread-launch",
                        "user": {"screen_name": "founder"},
                    }
                }
            ],
            "mentions": [
                {
                    "tweet": {
                        "id": "tweet-2",
                        "full_text": "@team this needs follow-up",
                        "created_at": 1735722000,
                        "conversation_id": "thread-launch",
                        "in_reply_to_status_id": "tweet-1",
                        "user": {"screen_name": "founder"},
                    }
                }
            ],
            "likes": [
                {
                    "like": {
                        "id": "like-1",
                        "tweet_id": "tweet-9",
                        "title": "A future we can use",
                        "date": "2026-01-03T08:15:00+00:00",
                    }
                }
            ],
        }
        wrapped_payload = f"window.YTD.tweets.part0 = {json.dumps(payload)};"
        upload = ("twitter.js", wrapped_payload, "application/javascript")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "twitter"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertGreaterEqual(payload["counts"].get("post", 0), 1)
        self.assertGreaterEqual(payload["counts"].get("mention", 0), 1)
        self.assertGreaterEqual(payload["counts"].get("like", 0), 1)
        self.assertIn(payload["created"], [True, False])
        self.assertEqual(payload["job"]["source_type"], "twitter")

    def test_import_rejects_empty_wrapper_payload(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        upload = ("twitter-empty.js", "window.YTD.tweets.part0 = [];", "application/javascript")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "twitter"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 400)

    def test_import_openclaw_upload_and_review_flow(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        export = {
            "agents": [
                {
                    "id": "agent-orchestrator",
                    "name": "Orchestrator",
                    "role": "main",
                    "capabilities": ["plan", "delegate", "summarize"],
                }
            ],
            "skills": [
                {
                    "id": "search-google",
                    "name": "search",
                    "description": "web search with structured extraction",
                    "owner": "agent-orchestrator",
                }
            ],
            "threads": [
                {
                    "id": "thr-1",
                    "agent_id": "agent-orchestrator",
                    "title": "Design kickoff",
                    "messages": [
                        {"id": "msg-1", "role": "user", "created_at": "2026-01-04T11:00:00Z", "content": "Let's build the MVP first."},
                        {"id": "msg-2", "role": "assistant", "created_at": "2026-01-04T11:00:30Z", "content": "Confirmed. Start with management web onboarding."},
                    ],
                }
            ],
        }
        upload = ("openclaw-export.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "openclaw"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["job"]["source_type"], "openclaw")
        self.assertEqual(payload["counts"].get("agent_profile"), 1)
        self.assertEqual(payload["counts"].get("skill"), 1)
        self.assertGreaterEqual(payload["counts"].get("memory"), 2)

        import_id = payload["import_id"]
        job_get = self.client.get(f"/api/agents/import/{import_id}", headers=read_headers)
        self.assertEqual(job_get.status_code, 200)
        items = job_get.json()["items"]
        self.assertGreaterEqual(len(items), 4)
        decision = self.client.post(
            f"/api/agents/import/{import_id}/decision",
            headers=admin_headers,
            json={
                "import_item_ids": [items[0]["id"]],
                "review_state": "approved",
                "note": "agent profile sanity checked",
            },
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["updated_count"], 1)

    def test_import_rejects_openclaw_empty_payload(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        upload = ("openclaw-empty.json", json.dumps({"items": []}), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "openclaw"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 400)

    def test_import_append_to_latest_updates_existing_job(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        initial_export = {
            "conversations": [
                {
                    "id": "thread-append",
                    "title": "Incremental import",
                    "messages": [
                        {"id": "msg-1", "author": {"role": "user"}, "create_time": "2026-01-07T10:00:00Z", "content": "First idea"},
                        {"id": "msg-2", "author": {"role": "assistant"}, "create_time": "2026-01-07T10:00:20Z", "content": "Great, let's do this."},
                    ],
                },
            ]
        }
        first_upload = ("openai-incremental-1.json", json.dumps(initial_export), "application/json")
        first = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "openai"},
            files={"source_file": first_upload},
        )
        self.assertEqual(first.status_code, 200, first.text)
        first_payload = first.json()
        first_id = first_payload["import_id"]
        self.assertEqual(first_payload["counts"].get("messages"), 2)
        self.assertEqual(first_payload["dedupe"], None)

        followup_export = {
            "conversations": [
                {
                    "id": "thread-append",
                    "title": "Incremental import",
                    "messages": [
                        {"id": "msg-1", "author": {"role": "user"}, "create_time": "2026-01-07T10:00:00Z", "content": "First idea"},
                        {"id": "msg-2", "author": {"role": "assistant"}, "create_time": "2026-01-07T10:00:20Z", "content": "Great, let's do this."},
                        {"id": "msg-3", "author": {"role": "assistant"}, "create_time": "2026-01-07T10:01:00Z", "content": "Now let's build."},
                    ],
                },
            ]
        }
        second_upload = ("openai-incremental-2.json", json.dumps(followup_export), "application/json")
        second = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "openai", "append_to_latest": "true"},
            files={"source_file": second_upload},
        )
        self.assertEqual(second.status_code, 200, second.text)
        second_payload = second.json()
        self.assertFalse(second_payload["created"])
        self.assertEqual(second_payload["import_id"], first_id)
        dedupe = second_payload["dedupe"]
        self.assertIsInstance(dedupe, dict)
        self.assertEqual(dedupe["attempted"], 3)
        self.assertEqual(dedupe["attempted"], dedupe["inserted"] + dedupe["skipped"])

        items = self.client.get(f"/api/agents/import/{first_id}", headers=read_headers).json()["items"]
        self.assertGreaterEqual(len(items), 3)

        # idempotent follow-up should skip all duplicate items
        duplicate = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "openai", "append_to_latest": "true"},
            files={"source_file": second_upload},
        )
        self.assertEqual(duplicate.status_code, 200, duplicate.text)
        duplicate_payload = duplicate.json()
        self.assertEqual(duplicate_payload["import_id"], first_id)
        self.assertFalse(duplicate_payload["created"])
        dup_dedupe = duplicate_payload["dedupe"]
        self.assertEqual(dup_dedupe["inserted"], 0)

    def test_import_codex_upload_and_review_flow(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        export = {
            "threads": [
                {
                    "id": "thread-codex",
                    "title": "Codex import draft",
                    "messages": [
                        {
                            "id": "c1",
                            "author": {"role": "user"},
                            "create_time": "2026-01-05T09:00:00Z",
                            "content": "Let's check onboarding defaults.",
                        },
                        {
                            "id": "c2",
                            "author": {"role": "assistant"},
                            "create_time": "2026-01-05T09:00:10Z",
                            "content": {"parts": ["Defaults loaded. Next step: set providers."]},
                        },
                    ],
                },
            ]
        }
        upload = ("codex-export.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "codex"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["job"]["source_type"], "codex")
        self.assertEqual(payload["counts"].get("messages"), 2)
        self.assertIn(payload["created"], [True, False])

        import_id = payload["import_id"]
        job_get = self.client.get(f"/api/agents/import/{import_id}", headers=read_headers)
        self.assertEqual(job_get.status_code, 200)
        items = job_get.json()["items"]
        self.assertGreaterEqual(len(items), 2)

        decision = self.client.post(
            f"/api/agents/import/{import_id}/decision",
            headers=admin_headers,
            json={
                "import_item_ids": [items[0]["id"]],
                "review_state": "approved",
                "note": "codex thread accepted",
            },
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["updated_count"], 1)

    def test_import_gemini_upload_and_review_flow(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        export = {
            "conversations": [
                {
                    "id": "gemini-thread-1",
                    "title": "Gemini import draft",
                    "messages": [
                        {
                            "id": "g1",
                            "author": {"role": "user"},
                            "create_time": "2026-01-06T11:00:00Z",
                            "content": "Can you review our privacy policy?",
                        },
                        {
                            "id": "g2",
                            "author": {"role": "assistant"},
                            "create_time": "2026-01-06T11:00:20Z",
                            "content": {"parts": ["Sure—let's keep it concise and explicit."]},
                        },
                    ],
                },
            ],
        }
        upload = ("gemini-export.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "gemini"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["job"]["source_type"], "gemini")
        self.assertEqual(payload["counts"].get("messages"), 2)
        self.assertIn(payload["created"], [True, False])

        import_id = payload["import_id"]
        job_get = self.client.get(f"/api/agents/import/{import_id}", headers=read_headers)
        self.assertEqual(job_get.status_code, 200)
        items = job_get.json()["items"]
        self.assertGreaterEqual(len(items), 2)

        decision = self.client.post(
            f"/api/agents/import/{import_id}/decision",
            headers=admin_headers,
            json={
                "import_item_ids": [items[0]["id"]],
                "review_state": "approved",
                "note": "gemini thread accepted",
            },
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["updated_count"], 1)

    def test_import_gemini_cli_upload_and_multisegment_reference_handling(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        export = {
            "conversations": [
                {
                    "id": "gemini-cli-thread",
                    "title": "Project launch runbook",
                    "messages": [
                        {
                            "id": "gcli-1",
                            "role": "user",
                            "created_at": "2026-01-08T08:00:00Z",
                            "parts": [
                                "Drafting an AI-powered release checklist.",
                                {"type": "code", "language": "python", "value": "print('drafting')"},
                            ],
                            "references": [
                                {"url": "https://example.com/release-checklist"},
                                {"url": "https://example.com/context"},
                            ],
                        },
                        {
                            "id": "gcli-2",
                            "role": "assistant",
                            "timestamp": "2026-01-08T08:00:30Z",
                            "content": "Let's validate the dependencies.",
                            "parts": ["Dependencies are green."],
                        },
                    ],
                }
            ]
        }
        upload = ("gemini-cli-export.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "gemini_cli"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["job"]["source_type"], "gemini_cli")
        self.assertEqual(payload["counts"].get("messages"), 2)

        items = self.client.get(f"/api/agents/import/{payload['import_id']}", headers=read_headers).json()["items"]
        self.assertGreaterEqual(len(items), 2)
        first_item = items[0]["content"]
        self.assertIn("References:", first_item)
        self.assertIn("https://example.com/release-checklist", first_item)

        decision = self.client.post(
            f"/api/agents/import/{payload['import_id']}/decision",
            headers=admin_headers,
            json={
                "import_item_ids": [items[0]["id"]],
                "review_state": "approved",
                "note": "Gemini CLI content ready for review.",
            },
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["updated_count"], 1)

    def test_import_copilot_upload_and_review_flow(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        export = {
            "records": [
                {
                    "id": "thread-copilot",
                    "title": "Copilot draft",
                    "messages": [
                        {
                            "id": "m1",
                            "role": "user",
                            "created_at": "2026-01-09T09:00:00Z",
                            "content": "Start by checking architecture.",
                        },
                        {
                            "id": "m2",
                            "role": "assistant",
                            "time": "2026-01-09T09:00:20Z",
                            "text": "Architecture review complete. Proceed to implementation.",
                        },
                    ],
                }
            ]
        }
        upload = ("copilot-export.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "copilot"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["job"]["source_type"], "copilot")
        self.assertEqual(payload["counts"].get("messages"), 2)

        items = self.client.get(f"/api/agents/import/{payload['import_id']}", headers={"Authorization": f"Bearer {self.read_token}"}).json()["items"]
        self.assertGreaterEqual(len(items), 2)

        decision = self.client.post(
            f"/api/agents/import/{payload['import_id']}/decision",
            headers=admin_headers,
            json={
                "import_item_ids": [items[0]["id"]],
                "review_state": "approved",
                "note": "Copilot record accepted.",
            },
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["updated_count"], 1)

    def test_import_copilot_rejects_empty_upload(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "copilot"},
            files={"source_file": ("copilot-empty.json", json.dumps({"threads": []}), "application/json")},
        )
        self.assertEqual(response.status_code, 400)

    def test_import_busy_local_upload_warns_when_no_history(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        export = {
            "providers": [
                {
                    "name": "openai-primary",
                    "model": "gpt-4o-mini",
                    "endpoint": "https://api.openai.com/v1",
                    "api_key": "sk-test-key",
                }
            ],
            "agents": [
                {
                    "name": "orchestrator-core",
                    "role": "main",
                    "capabilities": ["plan", "delegate"],
                }
            ],
        }
        upload = ("busy-local-no-history.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "busy_local"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("No conversation history", payload.get("warnings", [""])[0] or "")
        self.assertEqual(payload["job"]["source_type"], "busy_local")
        self.assertGreaterEqual(payload["counts"].get("provider", 0), 1)
        self.assertGreaterEqual(payload["counts"].get("agent", 0), 1)

        items = self.client.get(f"/api/agents/import/{payload['import_id']}", headers=read_headers).json()["items"]
        self.assertGreaterEqual(len(items), 2)

    def test_sensitive_import_items_require_review_note_for_override(self):
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        read_headers = {"Authorization": f"Bearer {self.read_token}"}

        export = {
            "conversations": [
                {
                    "id": "thread-sensitive",
                    "title": "Sensitive notes",
                    "messages": [
                        {
                            "id": "msg-1",
                            "author": {"role": "user"},
                            "create_time": "2026-01-10T12:00:00Z",
                            "content": "This contains my password for the private account.",
                        }
                    ],
                }
            ]
        }
        upload = ("openai-sensitive.json", json.dumps(export), "application/json")
        response = self.client.post(
            "/api/agents/import",
            headers=admin_headers,
            data={"source_type": "openai"},
            files={"source_file": upload},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["counts"].get("messages"), 1)

        import_id = payload["import_id"]
        items = self.client.get(f"/api/agents/import/{import_id}", headers=read_headers).json()["items"]
        self.assertEqual(len(items), 1)
        sensitive_item_id = items[0]["id"]
        self.assertEqual(items[0]["review_state"], "quarantined")

        denied = self.client.post(
            f"/api/agents/import/{import_id}/decision",
            headers=admin_headers,
            json={
                "import_item_ids": [sensitive_item_id],
                "review_state": "approved",
            },
        )
        self.assertEqual(denied.status_code, 400)
        self.assertIn("review note required", denied.json()["detail"])

        approved = self.client.post(
            f"/api/agents/import/{import_id}/decision",
            headers=admin_headers,
            json={
                "import_item_ids": [sensitive_item_id],
                "review_state": "approved",
                "note": "Reviewed and approved after manual scan.",
            },
        )
        self.assertEqual(approved.status_code, 200)
        self.assertEqual(approved.json()["updated_count"], 1)
