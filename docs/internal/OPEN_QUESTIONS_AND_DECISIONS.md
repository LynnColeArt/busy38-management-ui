# Open Questions and Decisions

Last updated: 2026-03-19

This file is the repo-level ledger for unresolved authority, parsing,
permission, and execution-boundary ambiguities referenced by `AGENTS.md`.

## Open questions

### Provider status normalization boundary

- Status: open
- Scope: provider filtering / diagnostics drill-down
- Question: should provider-status normalization become a write-time storage
  invariant, or remain a read/filter concern so the control plane preserves the
  literal upstream status string?
- Current decision boundary: the management UI now normalizes filter matching
  and drill-down behavior, while persisted provider rows may still retain their
  original casing or surrounding whitespace.

### Viewer-authenticated LAN discovery parity

- Status: open
- Scope: client parity for `GET /api/mobile/pairing/discovery`
- Question: which client repos have clean, committed proof that they handle the
  current viewer-auth discovery contract over bearer token or query token?
- Current evidence:
  - `pillowfort-ios-native` still shows a bare discovery GET in the checked-in
    runtime path and needs a focused parity rerun.
  - `pillowfort-kotlin` still lacks a committed in-app discovery consumer path
    against the live management control plane in this checkout.

## Recording rule

- Add a dated entry here whenever an authority-path requirement is
  underspecified enough that implementation would otherwise guess.
- When an open question is resolved, update the status here and summarize the
  behavior change in `CURRENT_STATE.md`.
