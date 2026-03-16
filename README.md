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

Start the management backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
PYTHONPATH="/path/to/Busy:$PWD" \
BUSY_RUNTIME_PATH="/path/to/Busy" \
uvicorn app.main:app --reload --port 8031
```

`backend/requirements.txt` includes the websocket transport dependency needed for the live event stream at `/api/events/ws`. A standard local install should not require an extra manual `pip install websockets`.

The backend imports Busy core modules directly. Pairing endpoints additionally
require:

```bash
export BUSY38_MOBILE_PAIRING_SECRET="replace-with-local-secret"
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

Then open `http://127.0.0.1:8031/` in a browser. The backend now serves the
management web app at the same origin as the API, and unknown non-API browser
paths such as `/admin` fall back to the SPA entrypoint.

If you need a different backend base than the default:

```bash
export MANAGEMENT_API_BASE=http://127.0.0.1:8031
```

For the shipped static page, runtime resolution is literal in this order:

1. `window.MANAGEMENT_API_BASE`
2. `meta[name="busy38-management-api-base"]`
3. served `window.location.origin` for HTTP(S) pages
4. `http://127.0.0.1:8031` only for local file/offline dev

Setting a shell variable alone does not inject it into `web/index.html`; use a
served page override if the UI and API are not on the same origin.

Appearance preference authority uses the same Busy runtime path:

- the browser reads and writes Busy-owned appearance preferences through
  `/api/appearance`
- `PATCH /api/appearance` is admin-authenticated; viewer tokens remain read-only
- default behavior is `system`
- app override supports `system`, `light`, and `dark`
- when override is enabled, sync remains on by default so desktop and mobile
  share one app-owned theme preference unless the user explicitly splits them
- default accessibility/readability policy is WCAG 2.2 AA
- the same Busy-owned record now also carries:
  - `AAA` contrast override
  - reduced motion
  - stronger color separation
  - increased text spacing

## Current behavior

- Seeded defaults are loaded on first run and persisted in local SQLite storage.
- Persistence is now backed by SQLite in `backend/data/management.db` (auto-created).
- API is versioned by endpoint conventions and can be swapped behind a proxy later.
- GM ticket workflow now has a management-plane interface for creation, filtering, assignment, status updates, and threaded operator notes.
- Plugin debugger warnings/errors and plugin UI action failures now emit structured records to the browser console for operator-side debugging.
- Successful plugin UI actions now also log warning-oriented metadata from both
  top-level result fields and payload-nested result bodies, matching the
  warning shapes returned by current plugin UI handlers.
- Mobile pairing is now plugin-owned in this repo for the first bounded slice:
  - admin-authenticated issue/revoke endpoints and an unauthenticated exchange endpoint now exist under `/api/mobile/pairing/*`
  - issued pairing state is short-lived and single-use
  - persisted pairing state must match the live Busy instance id; stale instance state fails closed before issue/exchange
  - exchange returns a scoped Busy bridge token and authoritative bridge URL
  - exchange bridge URL resolution is literal: explicit bridge URL override, then explicit bridge host, then exchange request host, then loopback fallback only for local dev
  - the browser now includes an admin-only pairing panel for issue + inspect + revoke
  - browser inspection uses safe state summaries only; it does not recover raw pairing codes or bridge tokens from persisted state
  - revoke is now keyed by explicit `token_id`, not pasted bearer tokens
  - the browser now also renders a QR locally from the live issue response plus the resolved control-plane URL
  - QR control-plane URL resolution is literal: explicit runtime override, then document override, then served origin, then loopback fallback
  - QR copy/render is live-response-only; reload requires issuing a new code
- Desktop appearance preferences are now part of the control plane:
  - `GET /api/appearance` and `PATCH /api/appearance` read/write the canonical
    Busy appearance record
  - the browser applies the resolved desktop theme to the document root on load
    and after save
  - the browser also applies the shared accessibility/readability policy to the
    document immediately after load/save

## API surface (MVP)

- `GET /api/health`
- `GET /api/settings`
- `PATCH /api/settings`
- `GET /api/plugins`
- `POST /api/plugins`
- `PATCH /api/plugins/{plugin_id}`
- `GET /api/plugins/core`
- `GET /api/plugins/core/reference`
- `POST /api/plugins/{plugin_id}/ui/{action_id}`
- `GET /api/plugins/{plugin_id}/ui/debug` (plugin diagnostics; includes signature checks, dependency checks, tool conflict diagnostics, optional runtime probe, and warning/error summary)
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
- `POST /api/gm-tickets`
- `POST /api/gm/message`
- `GET /api/gm-tickets`
- `GET /api/gm-tickets/{ticket_id}`
- `PATCH /api/gm-tickets/{ticket_id}`
- `POST /api/gm-tickets/{ticket_id}/messages`

Import review boundary:
- this app is local-first, not a SaaS control plane,
- raw sensitive import content may remain available for authenticated human review,
- downstream agent/provider-facing summaries must use security-redacted derivatives instead of raw sensitive import text.
- `GET /api/gm-tickets/{ticket_id}/messages`
- `GET /api/gm-tickets/{ticket_id}/audit`
- `GET /api/events` for latest event list
- `GET /api/events/ws` for live websocket event stream (role included in each frame)
- `GET /api/memory`
- `GET /api/chat_history`
- `POST /api/memory`
- `POST /api/chat_history`
- `POST /api/mobile/pairing/issue`
- `POST /api/mobile/pairing/exchange`
- `GET /api/mobile/pairing/state`
- `POST /api/mobile/pairing/revoke`
- `GET  /api/runtime/status`
- `GET  /api/runtime/services`
- `POST /api/runtime/services/{service_name}/start`
- `POST /api/runtime/services/{service_name}/stop`
- `POST /api/runtime/services/{service_name}/restart`
- `GET /api/appearance`
- `PATCH /api/appearance`

### Notable settings fields

- `heartbeat_interval`
- `fallback_budget_per_hour`
- `auto_restart`
- `proxy_http`
- `proxy_https`
- `proxy_no_proxy`

The web UI now shows a role badge at the top of the page:

- `admin` — token has write permission
- `viewer` — read-only access
- `unauthorized` — invalid or missing credentials when auth is configured

The badge tooltip displays the token source currently in use (for example `admin-token`, `read-token`, or `invalid-or-missing-token`).

## Test and validation

```bash
python3 -m py_compile backend/app/main.py backend/app/runtime.py
node -c web/plugin_ui_console.js
node -c web/app.js
node --test tests/test_plugin_ui_console_logging.mjs
node --test tests/test_mobile_pairing_ui.mjs
pip install -r backend/requirements-dev.txt
PYTHONPATH=. .venv/bin/pytest tests
```

See `docs/PLUGIN_DEBUGGER.md` for detailed debugger semantics, warning/error codes,
and expected payload structure.

## License

GPL-3.0-only.
