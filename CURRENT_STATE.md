# Current State

## 2026-03-18

- Mobile pairing now tolerates the one legacy state shape that predates trusted
  devices:
  - exchange, revoke, refresh, and state inspection now treat a missing
    `trusted_devices` collection as an empty legacy map
  - any present non-object `trusted_devices` value still fails closed as an
    invalid pairing-state artifact
- Provider overview remediation jumps now preserve each item's own provider
  status when opening diagnostics, so summary-driven drill-downs land on the
  correct bounded provider filter instead of inheriting the card-wide status.
- Same-origin management hardening now preserves the richer operator surface
  while closing two routing/auth gaps:
  - `GET /api/mobile/pairing/discovery` now requires viewer-or-admin auth
    through either `Authorization: Bearer <token>` or `?token=<token>`
  - bare `/api` remains a literal JSON 404 and does not fall through to the
    SPA shell
- Documentation, backend tests, and the live smoke path now all exercise that
  secured discovery contract.

## 2026-03-15

- The management repo now includes a live owner-level smoke path:
  - `python3 scripts/run_management_plane_smoke.py` boots a temporary local
    control plane against the sibling Busy checkout and verifies same-origin
    app serving plus appearance, discovery, pairing, refresh, revoke, and safe
    state inspection behavior over real HTTP
- The runtime operator surface is now closer to the repo's primary mission:
  - runtime status now exposes the configured default service in both status
    and service-list payloads
  - the browser now shows source-aware runtime summary, default service,
    service counts, and orchestrator notes instead of only connected yes/no
  - runtime start/stop/restart controls now fail closed in the browser for
    non-admin tokens rather than depending on backend 403s alone
  - each runtime service card now has local start/stop/restart controls plus
    literal pid, pid-file, and log-file visibility for operators
  - runtime actions now also emit structured `runtime.service_action` events,
    and the generic event feed renders service/action/result/source fields
    directly instead of reducing runtime history to plain message strings
  - the event feed now also supports bounded operator filtering by domain,
    level, text query, and limit, so runtime, import, GM, and provider
    activity can be navigated without leaving the main control plane
  - the page now starts with an operator overview layer for runtime, provider
    routing health, GM ticket pressure, and startup/plugin warnings so the
    control plane opens on the highest-attention slices first
  - those overview cards now also act as drill-down entry points into the
    existing runtime, provider, GM, and startup panels, including bounded
    filter handoff for provider and GM problem states
  - event feed and overview now share a local seen-state model keyed by event
    id, so new operator activity is visually distinct from already-reviewed
    events without mutating backend audit history
  - overview card selection now also exposes a bounded attention-history slice
    showing the recent unseen events behind that summary before opening the
    larger runtime/provider/GM/startup panels
  - provider routing health now uses one shared browser-side classifier in the
    overview and provider-chain panel, explicitly separating healthy routing,
    available-but-degraded routing, and blocked routing instead of only
    counting enabled providers
  - provider cards and diagnostics now also synthesize explicit cause mixes
    like missing secret, failed test, standby-only fallback, or recent request
    failures, so operators get a next action without parsing raw metrics first
  - the top provider overview card and its attention-history slice now also
    carry those remediation cues, so degraded routing can name the first fix
    directly from the summary layer before the operator opens full diagnostics
  - opening provider diagnostics now also selects and emphasizes the affected
    provider card in the main provider list, including overview-driven
    provider remediation jumps, so operators land on the editable record and
    diagnostics context together instead of scanning manually
  - provider diagnostics now also keep a local last-action strip for secret
    changes, provider tests, and model discovery attempts, so the selected
    provider shows the most recent fix attempt and outcome without depending
    on the global health status line
  - the provider overview card now also considers those local provider action
    results, so a fresh failed remediation attempt or successful verification
    can change the top-level provider guidance immediately before diagnostics
    are opened
  - the runtime overview card now also considers the latest structured
    `runtime.service_action` event, so failed restarts or recoveries can alter
    the top-level runtime guidance immediately before the operator opens the
    runtime panel
  - overview cards and the attention-history slice now also classify the most
    recent relevant signal as `fresh`, `recent`, or `older`, so operators can
    distinguish new incidents from aging history without dropping the audit
    detail or remediation hint
  - overview cards now also keep a browser-local per-card review timestamp, so
    the top summary and attention-history slice can distinguish `unreviewed`,
    `updated since review`, and `reviewed` states without mutating backend
    events or hiding fresh critical summaries
  - overview ordering now also uses that local review state as a tiebreaker
    after severity and freshness, so equally noisy cards stop competing for
    the first slot once one has already been reviewed in this browser
  - attention history now separates `Mark summary reviewed` from `Mark attached
    events seen`, so operators can locally deprioritize a summary without
    collapsing the unseen event trail that still backs the audit surface
  - attention-history event rows now also support per-event `Mark event seen`
    actions, so operators can peel down one noisy summary incrementally
    without bulk-acknowledging the rest of that card's unseen trail
  - already-seen attention-history rows now stay collapsed behind a local
    show/hide toggle, so the default summary view remains unresolved-first
    while recent seen context is still available on demand, and each card now
    states the bounded scope of that seen-history slice before expansion,
    including how old the oldest shown seen event is
  - seen-history toggle labels now also reflect whether that hidden context is
    background-only relative to the card's active unresolved attention state
- Revocation now correctly accepts the active post-refresh `token_id` for a
  trusted device:
  - refreshed bridge tokens remain revocable by `token_id`
  - the regression is covered in backend tests so refresh rotation and revoke
    stay coherent
## 2026-03-13

- The management backend now serves the browser app at the same origin:
  - `GET /` returns `web/index.html`
  - unknown non-API browser paths such as `/admin` fall back to the same SPA
    entrypoint
  - local operator launch can now open `http://127.0.0.1:8031/` directly
- The management backend now also owns the first trusted-device continuity
  slice:
  - `POST /api/mobile/pairing/exchange` now persists a durable trusted-device
    relationship in the shared Busy pairing state and returns
    `device_relationship_id`, a refresh grant, and `trusted_device_expires_at`
  - `POST /api/mobile/trust/refresh` now rotates the short-lived bridge token
    and the refresh grant for an active trusted device
  - refresh revokes the superseded bridge token instead of accumulating
    parallel long-lived active grants for one device relationship
  - `POST /api/mobile/pairing/revoke` now also revokes the linked trusted
    device relationship when revoking its active token
  - `GET /api/mobile/pairing/state` now exposes safe trusted-device summaries
    with no raw refresh-grant recovery
- The management backend now also owns the bounded LAN discovery descriptor:
  - `GET /api/mobile/pairing/discovery` exposes a viewer-authenticated
    read-only candidate surface for local-network pairing discovery
  - the descriptor returns `version`, `service_type`, `instance_id`,
    `display_label`, `control_plane_url`, `bridge_url`, `bootstrap_methods`,
    and `supports_pairing_code`
  - the descriptor does not issue trust; short pairing-code confirmation
    remains mandatory through `POST /api/mobile/pairing/exchange`

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
  - exchange now also persists a trusted-device relationship so successful
    first-pair bootstrap can later refresh and recover continuity without
    re-scanning QR.
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
