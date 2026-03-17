# Current State

## 2026-03-13

- The management backend now owns the first trusted-device continuity slice:
  - `POST /api/mobile/pairing/exchange` now persists a durable trusted-device
    relationship in the shared Busy pairing state and returns
    `device_relationship_id`, a refresh grant, and
    `trusted_device_expires_at`
  - `POST /api/mobile/trust/refresh` now rotates the short-lived bridge token
    and the refresh grant for an active trusted device
  - refresh revokes the superseded bridge token instead of accumulating
    parallel long-lived active grants for one device relationship
  - `POST /api/mobile/pairing/revoke` now also revokes the linked trusted
    device relationship when revoking its active token
  - `GET /api/mobile/pairing/state` now exposes safe trusted-device summaries
    with no raw refresh-grant recovery

## 2026-03-09

- Busy-owned appearance preferences are now implemented in this repo's control
  plane:
  - `GET /api/appearance` returns the current canonical Busy appearance record
  - `PATCH /api/appearance` updates that record with fail-closed validation for
    admin-authenticated callers only
  - the browser now applies the resolved desktop theme literally from one
    shared preference model:
    - default `system`
    - app override `system` / `light` / `dark`
    - sync-on-by-default when override is enabled
    - default accessibility/readability policy is WCAG 2.2 AA
    - stronger overrides now exist for `AAA` contrast, reduced motion,
      stronger color separation, and increased text spacing
  - the browser uses the same document theme helper for both initial load and
    post-save updates, so served pages now switch between light/dark without
    separate local-only theme state
  - the browser now also applies those accessibility/readability preferences
    immediately at the document level through the same Busy-owned authority
    path.

## 2026-03-07

- Plugin-owned mobile pairing is now implemented through the management API:
  - `POST /api/mobile/pairing/issue` is admin-authenticated and issues a
    short-lived single-use pairing code with explicit room/orchestrator scope.
  - issuance now also validates requested room ids against the current
    GM-derived logical room set and validates requested orchestrator ids against
    the bounded known orchestrator set for this slice before minting a code.
  - `POST /api/mobile/pairing/exchange` exchanges that pairing code into a
    scoped Busy bridge bearer token plus authoritative bridge URL.
  - exchange now also revalidates the persisted issued-code room/orchestrator
    scope against the current bounded known set before minting a live bridge
    token, and invalid persisted scope leaves the code pending instead of
    consuming it.
  - pairing state now also fails closed unless its persisted `instance_id`
    matches the live Busy instance id used for scoped bridge token minting, so
    QR issue/exchange and minted tokens share one literal instance authority.
  - `POST /api/mobile/pairing/revoke` is admin-authenticated and revokes an
    issued scoped bridge token by token ID.
- The bounded operator/browser pairing surface is now also implemented:
  - admins can issue pairing codes directly in the browser,
  - the browser now loads an admin-only safe pairing-state summary from
    `GET /api/mobile/pairing/state`,
  - exchanged grants can be revoked by `token_id` from the browser without
    pasting raw bridge bearer tokens,
  - raw pairing codes remain visible only from the live issuance response and
    are not recoverable from persisted state after refresh,
  - the browser now also derives a QR locally from the live issue response plus
    a literally resolved control-plane URL,
  - exchange now derives returned `bridge_url` literally from explicit bridge
    URL override -> explicit bridge host -> exchange request host -> loopback
    fallback only for local dev,
  - control-plane URL resolution now uses:
    - `window.MANAGEMENT_API_BASE` first,
    - `meta[name="busy38-management-api-base"]` second,
    - served `window.location.origin` for HTTP(S) pages third,
    - loopback fallback only for local file/offline dev,
  - QR payload copy/render remains browser-local and is not recoverable from
    persisted state after refresh.
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
