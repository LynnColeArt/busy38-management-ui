# Current State

## 2026-03-07

- Plugin-owned mobile pairing is now implemented through the management API:
  - `POST /api/mobile/pairing/issue` is admin-authenticated and issues a
    short-lived single-use pairing code with explicit room/orchestrator scope.
  - `POST /api/mobile/pairing/exchange` exchanges that pairing code into a
    scoped Busy bridge bearer token plus authoritative bridge URL.
  - `POST /api/mobile/pairing/revoke` is admin-authenticated and revokes an
    issued scoped bridge token by token ID.
- The bounded operator/browser pairing surface is now also implemented:
  - admins can issue pairing codes directly in the browser,
  - the browser now loads an admin-only safe pairing-state summary from
    `GET /api/mobile/pairing/state`,
  - exchanged grants can be revoked by `token_id` from the browser without
    pasting raw bridge bearer tokens,
  - raw pairing codes remain visible only from the live issuance response and
    are not recoverable from persisted state after refresh.
- Pairing authority remains API-owned in this first slice:
  - this repo is the canonical pairing authority surface,
  - Busy bridge core only validates the plugin-issued scoped token and enforces
    room/orchestrator scope at runtime,
  - no duplicate Busy-core pairing issuance endpoint exists.
- Pairing state uses the shared Busy runtime artifact and signature secret:
  - backend pairing helpers import Busy core pairing utilities directly,
  - pairing requires `BUSY_RUNTIME_PATH` plus `PYTHONPATH` access to the Busy
    checkout,
  - pairing also requires `BUSY38_MOBILE_PAIRING_SECRET` to be set explicitly.
- Canonical details live in:
  - `docs/internal/PAIRING_CONTROL_PLANE_SLICE_SPEC.md` in Busy
  - `docs/internal/PAIRING_PLUGIN_SCOPED_BRIDGE_TOKEN_VALIDATION_CHANGE_REQUEST.md` in Busy
  - `docs/internal/PAIRING_OPERATOR_BROWSER_SURFACE_SPEC.md` in Busy

## 2026-03-05

- Plugin UI diagnostics and plugin UI action handlers now emit structured warning/error records to the browser console.
- Diagnostics loaded from `GET /api/plugins/{plugin_id}/ui/debug` log:
  - `console.warn` for warning entries returned by the debugger payload
  - `console.error` for error entries returned by the debugger payload
- Plugin UI action execution logs:
  - `console.error` for transport failures and handler failures (`result.success=false`)
  - `console.warn` for successful actions that still surface warning-oriented payload data such as `warnings`, `warning_codes`, or `reason_codes`
  - warning/reason metadata is normalized from both top-level action results and payload-nested result bodies so operator console telemetry does not under-report actionable warnings
  - failed-action console records now also preserve top-level `errors` so validation/debug failure details survive into browser telemetry
- Console output is browser-only telemetry for operators. It does not alter backend dispatch, plugin authority, or runtime fallback behavior.
- Canonical details live in:
  - `docs/PLUGIN_DEBUGGER.md`
  - `docs/PROVIDER_AND_MANAGEMENT_UX_SPEC.md`
  - `docs/internal/PLUGIN_UI_BROWSER_CONSOLE_LOGGING_CHANGE_REQUEST.md`
- Local management-ui bootstrap now includes websocket transport support for `/api/events/ws` through repository backend requirements.
- The websocket auth/role test is no longer skipped; it now exercises a real handshake against the FastAPI app.
- Canonical details live in:
  - `docs/internal/WEBSOCKET_EVENT_STREAM_BOOTSTRAP_CHANGE_REQUEST.md`
- Automated import intake `block` results now land in quarantine for human review instead of being marked rejected immediately.
- Human review remains the only path that may set `review_state="rejected"`.
- Canonical details live in:
  - `docs/internal/IMPORT_BLOCK_QUARANTINE_REVIEW_CHANGE_REQUEST.md`
- Sensitive import items now retain a local human-review copy while explicitly marking raw content as reviewer-local-only in metadata.
- The security pipeline now persists a redacted preview derivative for sensitive/quarantined imports, and downstream agent-directory surfaces use that redacted derivative instead of raw import text.
- Canonical details live in:
  - `docs/internal/IMPORT_LOCAL_REVIEW_ISOLATION_CHANGE_REQUEST.md`
