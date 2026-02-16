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

Token is stored locally in browser for this UI only (client-side convenience) via the `Save token` control.

Then open `web/index.html` in a browser (or serve it from any static host).

Set this environment variable if you want different backend routing:

```bash
export MANAGEMENT_API_BASE=http://127.0.0.1:8031
```

## Current behavior

- Mock data is used to boot quickly and prove flow.
- Persistence is now backed by SQLite in `backend/data/management.db` (auto-created).
- API is versioned by endpoint conventions and can be swapped behind a proxy later.

## API surface (MVP)

- `GET /api/health`
- `GET /api/settings`
- `PATCH /api/settings`
- `GET /api/providers`
- `PATCH /api/providers/{provider_id}`
- `GET /api/agents`
- `PATCH /api/agents/{agent_id}`
- `GET /api/events`
- `POST /api/events` not yet available (event stream is append-only from internal actions)
- `GET /api/memory`
- `GET /api/chat_history`
- `POST /api/memory`
- `POST /api/chat_history`

## License

GPL-3.0-only.
