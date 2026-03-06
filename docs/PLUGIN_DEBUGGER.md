# Plugin UI Debugger

The Management UI plugin debugger endpoint is the single canonical place to validate
plugin registration quality and diagnose runtime availability.

## Endpoint

`GET /api/plugins/{plugin_id}/ui/debug`

Requires admin role.

Startup debug execution also emits per-plugin status events during startup and a startup summary event at `plugin.startup_debug_summary`.

The summary appears in management events and is surfaced as a dedicated startup diagnostics block in the UI.
The same startup path also writes `startup plugin debug: ...` lines to process stdout for container/runtime logs.
Payload includes all checked installed plugins plus required core plugin presence summary from the core reference table.
When required core plugins are absent from the registry, they are surfaced in `required_missing`/`required_missing_plugins` and promoted to startup-level errors.

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
    "base_required": false,
    "required_override": false,
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
  "core_plugins": {
    "plugins": [
      {
        "plugin_id": "squidkeys",
        "required": true,
        "base_required": true,
        "required_override": false,
        "alias_match": "squidkeys",
        "matched_plugin_id": "squidkeys",
        "present": true,
        "reason_code": "P_PLUGIN_PRESENT_OK"
      },
      {
        "plugin_id": "busy38-iphone",
        "required": false,
        "base_required": false,
        "required_override": false,
        "alias_match": null,
        "matched_plugin_id": null,
        "present": false,
        "reason_code": "P_PLUGIN_MISSING_OPTIONAL"
      }
    ],
    "summary": {
      "required_total": 1,
      "required_present": 1,
      "required_missing": [],
      "optional_total": 3,
      "optional_present": 2,
      "optional_missing": ["busy38-iphone"],
      "state": "READY",
      "conflicts": []
    }
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
- `P_PLUGIN_UI_ASSET_MISSING` — plugin declares UI contract but local source path has no `ui/` directory.
- `P_PLUGIN_UI_ASSET_EMPTY` — plugin `ui/` directory exists but has no files.
- `P_PLUGIN_UI_HANDLER_MISSING` — local `ui` module/function for an action is missing.
- `P_PLUGIN_UI_HANDLER_LOAD_FAILED` — action `ui` module failed to import/execute during diagnostics.
- `P_PLUGIN_UI_HANDLER_NOT_CALLABLE` — resolved action handler is present but not callable.
- `P_PLUGIN_RUNTIME_DEBUG_FAILED` — runtime returned a failed result for `debug` action.
- `P_PLUGIN_SIGNATURE_MISSING_REQUIRED` — required plugin is missing signature metadata.
- `P_PLUGIN_SIGNATURE_MISSING_OPTIONAL` — optional plugin has no signature metadata.
- `P_PLUGIN_TOOL_NAMESPACE_CONFLICT` — tool namespace/action collisions.
- `P_PLUGIN_TOOLS_QUERY_FAILED` — tool registry lookup failed.
- `P_PLUGIN_DEPENDENCY_MISSING` — declared dependency plugin is absent.
- `P_PLUGIN_DEPENDENCY_DISABLED` — declared dependency plugin is registered but disabled.
- `P_PLUGIN_MISSING_REQUIRED` — required core plugin alias not found in the plugin registry.
- `P_PLUGIN_MISSING_OPTIONAL` — optional core plugin alias not found in the plugin registry.
- `P_PLUGIN_NAMESPACE_CONFLICT` — multiple registry entries match a single required/optional core plugin.
- `P_PLUGIN_PRESENT_OK` — core plugin alias resolved to a single enabled registry entry.

## Plugin guidance

To reduce noise and improve debugger quality, plugin UI metadata should:

1. include a `ui` section with actions and stable IDs,
2. include a `debug` action when diagnostic hooks are available,
3. emit stable, redacted-friendly payloads from debug action responses,
4. declare optional dependencies in `metadata.depends_on` (or via core plugin index for built-ins).

For built-ins, dependency checks use the core plugin reference table so declared core
dependencies are validated even when plugin metadata is sparse.

## Local UI handler resolution details

- Default local dispatch target for each action is `ui/actions.py` function `handle_<action_id>`.
- An action may override with `entry_point` using:
  - `module.function`
  - `module:function`
- During diagnostics, each action in `metadata.ui.sections` is validated against the local `ui` tree:
  - module/file discovery under `ui/`
  - import/load success
  - handler symbol presence and callability
