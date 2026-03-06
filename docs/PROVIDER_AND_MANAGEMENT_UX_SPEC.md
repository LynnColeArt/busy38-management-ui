# Provider and Management UX Specification

Status: Draft  
Scope: busy38-management-ui  
Owner: platform  
Date: 2026-02-18  

## 1) Purpose

Define the management web UX for provider configuration, health observability, and secure secret handling so operators can safely configure, test, and troubleshoot multi-provider inference without exposing sensitive material in the UI or logs.

## 2) Design goals

- Keep secrets isolated from normal provider settings.
- Make routing/fallback behavior deterministic and inspectable.
- Surface runtime impact (including latency) in operator terms.
- Keep destructive operations explicit and reversible.
- Make viewer/admin behavior obvious at all times.

## 3) Core data model

### 3.1 Provider record (UI-facing model)
These fields are normal and editable in UI forms:
- `id`
- `name`
- `display_name`
- `kind` (`openai_compatible`, `ollama`, `llama_cpp`, `other`)
- `endpoint`
- `default_model`
- `fallback_models[]` (ordered)
- `status` (`active`, `standby`, `degraded`, `unreachable`, `disabled`)
- `enabled` (bool)
- `priority` (int)
- `capabilities` (features / limits)
- `secret_policy` (`required`, `optional`, `none`)
- `retries`, `timeout_ms`, `tool_timeout_ms`
- `metadata` (non-sensitive, redacted for viewer)
- `routing_intent` (resolved target for default and fallback chain)

### 3.2 Secret record (secure plane)
These values are never shown back to client UI as clear text:
- `api_key`
- `token`
- `secret_salt`
- `webhook_secret`
- `tls_identity` fragments if present

The only UI-safe projection is:
- `secret_present` (bool)
- `secret_last_rotated_at`
- `secret_touched_by` / `secret_touched_at`
- `secret_policy` (mirrors how provider handles secrets)

Provider secret behavior:
- `secret_required` is derived from `secret_policy`:
  - `required`: key must be present for test and usage
  - `optional`: key optional
  - `none`: no key required
- Local/self-hosted providers (for example Ollama, Llama.cpp) should default to `secret_policy=none`.

### 3.3 Provider health metrics
Track and display:
- `latency_ms_last`
- `latency_ms_avg_5m`
- `latency_ms_p95_5m`
- `success_rate_5m`
- `failure_count_last_1m`
- `last_checked_at`
- `last_error_code`
- `last_error_message` (redacted/no secrets)

## 4) User experience requirements

### 4.1 Provider list
- Cards should separate:
  - **Identity** (`name`, `kind`, `endpoint`)
  - **Model routing** (`default_model`, fallback order, enabled)
  - **Security** (`secret_present` badge, no raw secrets)
  - **Status** (`status`, `latency` badges, `last_checked_at`)
- Filters:
  - kind
  - status
  - secret status (`required`, `missing`, `present`)
  - health (`ok`, `degraded`, `unreachable`)
- Sort:
  - priority
  - last checked
  - avg latency
  - failures

### 4.2 Provider edit flow
- Editing public fields should only touch non-sensitive config.
- `POST / PATCH` that changes endpoint/model/priority/status should:
  - refresh local optimistic row
  - write audit event
  - show result toast + event stream entry
- If a change could impact routing, show preview of resulting chain immediately.

### 4.3 Secret management flow
- Separate action, separate form (`Secret` panel/action button), not in inline provider card.
- Modes:
  - `set`
  - `rotate`
  - `clear`
- Admin-only.
- Success and failures should not expose/echo secret values.
- Secret actions should be logged as audit events with operation, actor, time, provider.
- Providers with `secret_policy=none` should show secret actions as disabled/readonly with an inline message (for example, “No secret required”).

### 4.4 Provider health test flow
- Add `Test` action per provider.
- Test should:
  - use stored provider config + secret handle
  - skip secret requirement when `secret_policy=none`
  - require present secret when `secret_policy=required`
  - run short round-trip call
  - record latency and status
- Failed checks should stay in UI with actionable reason:
  - auth failure
  - timeout
  - unreachable
  - bad endpoint
- Add "Run all checks" action for all enabled providers.

### 4.5 Routing chain visibility
- Add explicit chain visualization:
  - `primary -> fallback1 -> fallback2`
  - indicates which provider is currently active and why.
- Chain should update live when priority/status changes.
- Clicking a chain node opens provider detail.

## 5) Management page context

### 5.1 Role model
- `viewer`: read-only, cannot mutate provider configs, secrets, or runtime.
- `admin`: full mutation and secret rotation.
- UI badges must remain visible and lock editing controls automatically by role.

### 5.2 Error and event feedback
- All mutations show:
  - toast status
  - event stream event row
- Event should include:
  - `event`: provider.created|updated|secret.updated|secret.cleared|provider.tested|provider.state_changed
  - `provider_id`
- Viewer should see event summaries even if payload fields are masked.

## 6) Proposed endpoints (backend contract)

- `GET /api/providers`
- `POST /api/providers`
- `PATCH /api/providers/{provider_id}` (supports `secret_policy`)
- `POST /api/providers/{provider_id}/test`
- `POST /api/providers/{provider_id}/secret` with action (`set|rotate|clear`) and optional `secret_id_hint`
- `GET /api/providers/{provider_id}/history` (optional)
- `GET /api/providers/{provider_id}/metrics`

## 7) Data safety

- Never place secret values into:
  - `GET /api/providers`
  - events
  - logs
  - websocket payloads
  - client cached storage
- Clear old provider test history older than configured TTL (default 30d).
- Secret operations should invalidate cached test credentials immediately.

## 8) Acceptance criteria

- An operator can:
  - create and enable a provider.
  - set public routing fields without touching secrets.
  - run provider test and see pass/fail + latency values.
  - rotate provider secret without exposing secret in payload/UI.
  - identify routing chain and active provider within one screen.
- Viewer role cannot access secret operations or mutate providers.
- Provider health and latency metrics update on test and are persisted for short-range diagnostics.

## 9) Non-goals (initial)

- Full multi-tenant key vault integration (must be a follow-up behind a feature flag).
- Deep provider policy editing UI beyond safe defaults.

## 10) Deployment status

- This is the implementation blueprint for the next management UI and API iteration after current routing/risk hardening.
