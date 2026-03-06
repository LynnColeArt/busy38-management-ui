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
