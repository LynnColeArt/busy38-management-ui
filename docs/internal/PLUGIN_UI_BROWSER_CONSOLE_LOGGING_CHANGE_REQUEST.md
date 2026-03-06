# Plugin UI Browser Console Logging Change Request

**Status**: implemented
**Date**: 2026-03-05

## Summary

The management UI already surfaces plugin debug warnings and plugin action failures in visible page status regions, but operators debugging browser-side integration work still need those warning/error details in the browser console.

This change adds explicit browser-console telemetry for plugin UI diagnostics and plugin UI action execution results.

## Desired behavior

- `GET /api/plugins/{plugin_id}/ui/debug` responses should emit:
  - `console.warn(...)` when the debugger payload contains warning entries
  - `console.error(...)` when the debugger payload contains error entries
- `POST /api/plugins/{plugin_id}/ui/{action_id}` responses should emit:
  - `console.error(...)` when the transport request fails
  - `console.error(...)` when the handler result reports `success=false`
  - `console.warn(...)` when the handler result succeeds but still returns warning-oriented metadata
- Console payloads must remain structured and include enough operator context to debug without scraping DOM status text.

## Constraints

- Browser console logging is telemetry only. It must not mutate UI state, alter request flow, or introduce any new authority path.
- Logging must stay literal. The client may only log warning/error material already present in the server response or transport failure.
- The helper must be testable without booting the full DOM app.

## Structured payload contract

Console records should include, when available:

- `pluginId`
- `actionId`
- `status`
- `message`
- `updatedAt`
- `warnings`
- `errors`
- `reasonCodes`
- `payload`

## Implementation touch points

- `web/plugin_ui_console.js`
- `web/app.js`
- `web/index.html`
- `docs/PLUGIN_DEBUGGER.md`
- `docs/PROVIDER_AND_MANAGEMENT_UX_SPEC.md`
- `README.md`

## Validation

- Add a focused node-based unit test for the console helper.
- Keep `node -c web/app.js` and `node -c web/plugin_ui_console.js` in the validation sweep.
