# Current State

## 2026-03-05

- Plugin UI diagnostics and plugin UI action handlers now emit structured warning/error records to the browser console.
- Diagnostics loaded from `GET /api/plugins/{plugin_id}/ui/debug` log:
  - `console.warn` for warning entries returned by the debugger payload
  - `console.error` for error entries returned by the debugger payload
- Plugin UI action execution logs:
  - `console.error` for transport failures and handler failures (`result.success=false`)
  - `console.warn` for successful actions that still surface warning-oriented payload data such as `warnings`, `warning_codes`, or `reason_codes`
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
