# Plugin UI Debugger

The Management UI plugin debugger endpoint is the single canonical place to validate
plugin registration quality and diagnose runtime availability.

## Endpoint

`GET /api/plugins/{plugin_id}/ui/debug`

Requires admin role.

## Purpose

- Validate `metadata.ui` contract shape.
- Detect tool registry conflicts and missing dependency context.
- Probe a plugin `debug` UI action when declared.
- Surface signature and required-core-plugin expectations.
- Return a deterministic, machine-readable diagnosis block.

## Response shape

```json
{
  "plugin": { /* redacted plugin record */ },
  "reference": {
    "required": false,
    "signature_required": false,
    "aliases": ["..."],
    "matched_alias": "...",
    "depends_on": ["..."]
  },
  "ui": {
    "actions": [...],
    "has_debug_action": false,
    "action_count": 0
  },
  "tools": {
    "total": 3,
    "active": 2,
    "conflicts": [...],
    "conflict_count": 0
  },
  "dependencies": {
    "declared": ["busy38-management-ui"],
    "warnings": [...],
    "count": 1
  },
  "warnings": {
    "count": 2,
    "entries": [...]
  },
  "errors": {
    "count": 1,
    "entries": [...]
  },
  "status": "warn",
  "runtime": {
    "action": "debug",
    "method": "GET",
    "success": false,
    "message": "debug action unavailable in plugin ui contract",
    "payload": {...},
    "runtime_called": false
  },
  "updated_at": "..."
}
```

## Status meaning

- `ok` — no warning or error entries.
- `warn` — at least one warning.
- `error` — at least one error.

## Warning and error codes

- `P_PLUGIN_UI_METADATA_MISSING` — no `metadata` block found.
- `P_PLUGIN_UI_CONTRACT_MISSING` — no `metadata.ui` block found.
- `P_PLUGIN_DEBUG_ACTION_MISSING` — no `debug` action in UI contract.
- `P_PLUGIN_RUNTIME_DEBUG_FAILED` — runtime returned a failed result for `debug` action.
- `P_PLUGIN_SIGNATURE_MISSING_REQUIRED` — required plugin is missing signature metadata.
- `P_PLUGIN_SIGNATURE_MISSING_OPTIONAL` — optional plugin has no signature metadata.
- `P_PLUGIN_TOOL_NAMESPACE_CONFLICT` — tool namespace/action collisions.
- `P_PLUGIN_TOOLS_QUERY_FAILED` — tool registry lookup failed.
- `P_PLUGIN_DEPENDENCY_MISSING` — declared dependency plugin is absent.
- `P_PLUGIN_DEPENDENCY_DISABLED` — declared dependency plugin is registered but disabled.

## Plugin guidance

To reduce noise and improve debugger quality, plugin UI metadata should:

1. include a `ui` section with actions and stable IDs,
2. include a `debug` action when diagnostic hooks are available,
3. emit stable, redacted-friendly payloads from debug action responses,
4. declare optional dependencies in `metadata.depends_on` (or via core plugin index for built-ins).

For built-ins, dependency checks use the core plugin reference table so declared core
dependencies are validated even when plugin metadata is sparse.
