#!/usr/bin/env python3
"""Run a live trusted-device refresh smoke test against sibling Busy sources."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Busy38 trusted-device refresh smoke test")
    parser.add_argument("--busy-repo", help="Path to sibling Busy checkout")
    parser.add_argument("--busy-python", help="Python interpreter from the Busy virtualenv")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the temporary management API server")
    parser.add_argument("--port", type=int, help="Port to bind the temporary management API server")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary runtime directory and logs instead of deleting them",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_busy_repo(root: Path, configured: str | None) -> Path:
    if configured:
        candidate = Path(configured).expanduser().resolve()
    else:
        candidate = (root.parent / "busy").resolve()
    if not candidate.is_dir():
        raise SystemExit(f"Busy repo not found: {candidate}")
    return candidate


def resolve_busy_python(busy_repo: Path, configured: str | None) -> Path:
    if configured:
        candidate = Path(configured).expanduser()
    else:
        candidate = busy_repo / ".venv" / "bin" / "python"
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(str(candidate)))
    if not candidate.is_file():
        raise SystemExit(f"Busy Python not found: {candidate}")
    return candidate


def pick_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    payload = None
    merged_headers = {"accept": "application/json"}
    if headers:
        merged_headers.update(headers)
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        merged_headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=payload, headers=merged_headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return exc.code, parsed


def wait_for_health(base_url: str, *, timeout: float, log_path: Path) -> None:
    deadline = time.time() + timeout
    last_error = "server did not become healthy"
    while time.time() < deadline:
        try:
            status, payload = request_json("GET", f"{base_url}/api/health")
            if status == 200 and payload.get("status") == "ok":
                return
            last_error = f"unexpected health payload: {payload!r}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.1)
    stderr = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    raise SystemExit(f"{last_error}\n\n{stderr}")


def assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    args = parse_args()
    root = repo_root()
    busy_repo = resolve_busy_repo(root, args.busy_repo)
    busy_python = resolve_busy_python(busy_repo, args.busy_python)
    host = args.host
    port = args.port or pick_port(host)
    base_url = f"http://{host}:{port}"
    admin_token = f"admin-{secrets.token_hex(8)}"
    pairing_secret = f"pairing-{secrets.token_hex(16)}"

    temp_root = Path(tempfile.mkdtemp(prefix="busy38-trusted-refresh-smoke-"))
    runtime_root = temp_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    log_path = temp_root / "trusted-device-refresh-smoke.log"

    bootstrap = f"""
import sys

sys.path.insert(0, {json.dumps(str(root))})
sys.path.insert(0, {json.dumps(str(busy_repo))})

from core.bridge import service as bridge_service

bridge_service._GM_TICKETS.clear()
bridge_service._GM_STANDUPS.clear()
bridge_service._GM_TICKETS["smoke-ticket-1"] = {{
    "ticket_id": "smoke-ticket-1",
    "title": "Trusted refresh smoke seeded room",
    "state": "queued",
    "team_id": "qa",
    "assigned_to": "carlo",
    "phase2_mission": "phase2-room-seed",
    "implementation_owner": "carlo",
    "cadence_owner": "gm",
    "created_at": "2026-03-15T00:00:00+00:00",
    "updated_at": "2026-03-15T00:00:00+00:00",
}}

import uvicorn
uvicorn.run("backend.app.main:app", host={json.dumps(host)}, port={port}, reload=False, log_level="warning")
"""

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(root), str(busy_repo)]),
            "BUSY_RUNTIME_PATH": str(runtime_root),
            "BUSY38_INSTANCE_ID": "busy-local",
            "BUSY38_MOBILE_PAIRING_SECRET": pairing_secret,
            "MANAGEMENT_ADMIN_TOKEN": admin_token,
            "MANAGEMENT_READ_TOKEN": f"viewer-{secrets.token_hex(8)}",
            "MANAGEMENT_API_TOKEN": "",
        }
    )

    process = subprocess.Popen(
        [str(busy_python), "-c", bootstrap],
        cwd=str(root),
        env=env,
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    try:
        wait_for_health(base_url, timeout=20.0, log_path=log_path)

        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        issue_payload = {
            "device_label": "Smoke iPhone",
            "authorized_room_ids": ["team-room-qa"],
            "orchestrator_scope": ["carlo"],
            "ttl_sec": 300,
        }
        status, issued = request_json(
            "POST",
            f"{base_url}/api/mobile/pairing/issue",
            headers=admin_headers,
            body=issue_payload,
        )
        assert_equal(status, 200, "pairing issue should succeed")
        issued_pairing = issued["pairing"]
        pairing_code = issued_pairing["pairing_code"]

        status, exchanged = request_json(
            "POST",
            f"{base_url}/api/mobile/pairing/exchange",
            body={
                "pairing_code": pairing_code,
                "device_label": "Smoke iPhone",
                "expected_instance_id": "busy-local",
            },
        )
        assert_equal(status, 200, "pairing exchange should succeed")
        exchanged_pairing = exchanged["pairing"]
        assert_true(exchanged_pairing["bridge_token"].startswith("busy_pair_v1."), "exchange should return scoped token")
        assert_true(exchanged_pairing["refresh_grant"].startswith("busy_refresh_v1."), "exchange should return refresh grant")

        status, state_before = request_json("GET", f"{base_url}/api/mobile/pairing/state", headers=admin_headers)
        assert_equal(status, 200, "pairing state should load for admin")
        state_before_json = json.dumps(state_before["pairing"], sort_keys=True)
        assert_true(pairing_code not in state_before_json, "pairing state must not expose raw pairing code")
        assert_true(exchanged_pairing["bridge_token"] not in state_before_json, "pairing state must not expose raw bridge token")
        assert_true(exchanged_pairing["refresh_grant"] not in state_before_json, "pairing state must not expose raw refresh grant")

        status, refreshed = request_json(
            "POST",
            f"{base_url}/api/mobile/trust/refresh",
            body={
                "device_relationship_id": exchanged_pairing["device_relationship_id"],
                "refresh_grant": exchanged_pairing["refresh_grant"],
                "expected_instance_id": "busy-local",
            },
        )
        assert_equal(status, 200, "trusted-device refresh should succeed")
        refreshed_pairing = refreshed["pairing"]
        assert_true(
            refreshed_pairing["token_id"] != exchanged_pairing["token_id"],
            "refresh should rotate token id",
        )
        assert_true(
            refreshed_pairing["refresh_grant"] != exchanged_pairing["refresh_grant"],
            "refresh should rotate refresh grant",
        )

        status, revoked = request_json(
            "POST",
            f"{base_url}/api/mobile/pairing/revoke",
            headers=admin_headers,
            body={"token_id": refreshed_pairing["token_id"]},
        )
        assert_equal(status, 200, "revoke should succeed")
        assert_equal(revoked["pairing"]["token_id"], refreshed_pairing["token_id"], "revoke should target refreshed token")

        status, state_after = request_json("GET", f"{base_url}/api/mobile/pairing/state", headers=admin_headers)
        assert_equal(status, 200, "pairing state should reload after revoke")
        issued_records = state_after["pairing"]["issued"]
        trusted_devices = state_after["pairing"]["trusted_devices"]
        assert_true(any(record.get("status") == "revoked" for record in issued_records), "issued state should show revoked token")
        assert_true(any(record.get("status") == "revoked" for record in trusted_devices), "trusted device state should show revoked relationship")

        print(f"trusted-device refresh smoke passed against {base_url}")
        print(f"log: {log_path}")
        return 0
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
