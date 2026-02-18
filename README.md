# Busy38 Management UI

Busy38 Management UI is the centralized web control plane for configuring and monitoring Busy deployments.

## What this repo is for

This is the management application for:

- model provider configuration and fallbacks
- credential and API binding status
- agent fleet topology and enable/disable toggles
- orchestrator run/stop state and live status
- system health and recent activity logs

The goal is to move operations off ad-hoc mobile flows and into a first-class management interface for human operators.

## Local development

Start backend + static UI:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload --port 8031
```

Optional token protection:

```bash
export MANAGEMENT_API_TOKEN="your-shared-secret"
```

If set, the API requires either:

- `Authorization: Bearer <token>`
- or `?token=<token>` on GET/POST/PATCH endpoints.

You can also configure role-specific tokens:

- `MANAGEMENT_ADMIN_TOKEN` for privileged write actions (settings, provider, agent, runtime, chat/memory writes)
- `MANAGEMENT_READ_TOKEN` for read-only access to status and telemetry
- legacy `MANAGEMENT_API_TOKEN` still works for both roles when role tokens are not set.

Token is stored locally in browser for this UI only (client-side convenience) via the `Save token` control.

Then open `web/index.html` in a browser (or serve it from any static host).

Set this environment variable if you want different backend routing:

```bash
export MANAGEMENT_API_BASE=http://127.0.0.1:8031
```

## Current behavior

- Seeded defaults are loaded on first run and persisted in local SQLite storage.
- Persistence is now backed by SQLite in `backend/data/management.db` (auto-created).
- API is versioned by endpoint conventions and can be swapped behind a proxy later.

## API surface (MVP)

- `GET /api/health`
- `GET /api/settings`
- `PATCH /api/settings`
- `GET /api/providers`
- `POST /api/providers`
- `POST /api/providers/{provider_id}/discover-models`
- `POST /api/providers/{provider_id}/test`
- `POST /api/providers/test-all`
- `POST /api/providers/{provider_id}/secret`
- `GET /api/providers/routing-chain`
- `GET /api/providers/{provider_id}/history`
- `GET /api/providers/{provider_id}/metrics`
- `PATCH /api/providers/{provider_id}`
- `POST /api/agents/import`
- `GET /api/agents/imports`
- `GET /api/agents/import/{import_id}`
- `POST /api/agents/import/{import_id}/decision`
- `GET /api/agents`
- `PATCH /api/agents/{agent_id}`
- `GET /api/events` for latest event list
- `GET /api/events/ws` for live websocket event stream (role included in each frame)
- `GET /api/memory`
- `GET /api/chat_history`
- `POST /api/memory`
- `POST /api/chat_history`
- `GET  /api/runtime/status`
- `GET  /api/runtime/services`
- `POST /api/runtime/services/{service_name}/start`
- `POST /api/runtime/services/{service_name}/stop`
- `POST /api/runtime/services/{service_name}/restart`

The web UI now shows a role badge at the top of the page:

- `admin` — token has write permission
- `viewer` — read-only access
- `unauthorized` — invalid or missing credentials when auth is configured

The badge tooltip displays the token source currently in use (for example `admin-token`, `read-token`, or `invalid-or-missing-token`).

## Test and validation

```bash
python3 -m py_compile backend/app/main.py backend/app/runtime.py
node -c web/app.js
pip install -r backend/requirements-dev.txt
PYTHONPATH=. .venv/bin/pytest tests
```

## License

GPL-3.0-only.
