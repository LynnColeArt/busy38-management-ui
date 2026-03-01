# Plugin UI Local `/ui` Directory Change Request

**Status**: implemented
**Date**: 2026-02-28

## Summary
Management UI currently dispatches plugin actions through a central bridge endpoint contract even when plugin source is local.

The goal is to move executable UI behavior ownership into each plugin’s local `ui` directory, while preserving the existing manifest contract and bridge fallback behavior.

## Desired behavior
- Runtime action execution should prefer plugin-local handlers when plugin source resolves to a local directory.
- Plugin source is expected at:
  - `<plugin_source>/ui/`
- Manifest/action metadata remains authoritative for validation and visibility.
- Execution path:
  1. Validate action exists in `metadata.ui` contract.
  2. Try local dispatch:
     - module/function defaults to `actions.handle_<action_id>`
     - optional explicit entry point via `action.entry_point`
     - support both `module.function` and `module:function`
  3. Fall back to existing bridge endpoint flow when local handler is missing/unavailable.

## Unresolved questions
- Should local handler call semantics standardize strictly on keyword-only `(payload, method, context)` or remain flexible?
- Should async handler functions be supported at all (and how should they be executed from sync call path)?
- What failures should block fallback when local import succeeds but handler raises?
- Do we need `ui/manifest.json` augmentation beyond `metadata.ui` in plugin registries?

## Resolutions

- Handler call semantics: keep flexible invocation and support keyword arguments if provided (`payload`, `method`, `context`) with positional fallback support for backward compatibility.
- Async handlers: not supported yet; handlers are invoked through synchronous execution on the current runtime path.
- Local failures: module import/load and non-callable symbol failures surface immediate warning codes and do not trigger bridge fallback; missing handler or path falls back to bridge.
- Optional plugin-local `ui/manifest.json` is supported as a source for UI contract and entry-point hints; if present it augments/overrides `metadata.ui` defaults.

## Implementation touch points
- `backend/app/main.py`: pass plugin source/action spec into runtime UI dispatch calls.
- `backend/app/runtime.py`: add local `/ui` action resolution + handler invocation, then bridge fallback.
- `tests/test_main.py`: extend runtime mock to accept new optional dispatch args.

## Required core plugins coverage reference (post-implementation)

Management API now exposes required plugin metadata and coverage through:

- `GET /api/plugins/core`
- `GET /api/plugins/core/reference`

Current required set from core plugin reference:

- `busy-38-squidkeys`
- `busy-38-rangewriter`
- `busy-38-blossom`
- `busy-38-management-ui`
- `busy-38-gticket`
- `busy-installer`
- `busy38-security-agent`
- `busy-38-git`
- `openclaw-browser-for-busy38`
- `busy-38-watchdog`

Coverage status (current):
- `busy-38-squidkeys` — management UI contract covered via `vendor/busy-38-squidkeys/ui/manifest.json`; installer coverage via source-of-truth clone or pre-vendored workspace dependency.
- `busy-38-rangewriter` — management UI contract covered via `vendor/busy-38-rangewriter/ui/manifest.json`; installer coverage via manifest repo sync.
- `busy-38-blossom` — management UI contract covered via `vendor/busy-38-blossom/ui/manifest.json`; installer coverage via manifest repo sync.
- `busy-38-management-ui` — management UI contract covered in plugin-local `ui`; installer coverage represented in bootstrap manifest dependency/source-of-truth mappings.
- `busy-38-gticket` — management UI contract covered via `vendor/busy-38-gticket/ui/manifest.json`; installer coverage via manifest repo sync.
- `busy-installer` — management UI contract covered in plugin-local `ui`; installer coverage represented in bootstrap manifest dependency/source-of-truth mappings.
- `busy38-security-agent` — management UI contract covered in plugin-local `ui`; installer coverage represented in bootstrap manifest dependency/source-of-truth mappings.
- `busy-38-git` — management UI contract covered in plugin-local `ui`; installer coverage represented in bootstrap manifest dependency/source-of-truth mappings.
- `openclaw-browser-for-busy38` — required by core and debug checks; required bootstrap coverage is represented in required bootstrap manifest/dependency mappings. This plugin does not currently include a local `/ui` directory in this repository snapshot.
- `busy-38-watchdog` — management UI contract covered in plugin-local `ui`; installer coverage represented in bootstrap manifest dependency/source-of-truth mappings.

`/api/plugins/core` returns required plugin presence plus UI coverage fields (`has_ui_contract`, `has_debug_action`, `coverage_state`) so installer/deployment tooling can quickly identify required plugins that need installer support or management-debug surface fixes.  
`/api/plugins/core/reference` provides the canonical required/optional plugin matrix from current core reference configuration (including alias set, required status, and signature requirement), useful for installer parity audits.
