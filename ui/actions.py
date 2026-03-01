from pathlib import Path


def handle_ping(payload: dict | None, method: str, context: dict | None) -> dict:
    normalized_payload = dict(payload or {})
    normalized_context = dict(context or {})
    return {
        "success": True,
        "message": "plugin ui ping handler executed",
        "payload": {
            "plugin": normalized_context.get("plugin_id", "unknown"),
            "method": str(method).strip().upper() or "POST",
            "payload": normalized_payload,
            "source": normalized_context.get("source_path"),
        },
    }


def handle_debug(payload: dict | None, method: str, context: dict | None) -> dict:
    normalized_payload = dict(payload or {})
    normalized_context = dict(context or {})
    source_path = str(normalized_context.get("source_path") or "").strip()
    plugin_id = str(normalized_context.get("plugin_id") or "unknown").strip()
    plugin_root = Path(source_path) if source_path else None

    return {
        "success": True,
        "message": "plugin ui debug handler executed",
        "payload": {
            "plugin": plugin_id,
            "method": str(method).strip().upper() or "GET",
            "payload": normalized_payload,
            "source_path": source_path,
            "source_exists": bool(plugin_root) and plugin_root.exists(),
            "manifests": {
                "plugin_manifest_exists": bool(plugin_root and (plugin_root / "manifest.json").is_file()),
                "ui_manifest_exists": bool(plugin_root and (plugin_root / "ui" / "manifest.json").is_file()),
            },
            "entrypoint": "actions:handle_debug",
        },
    }
