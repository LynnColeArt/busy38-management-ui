"""FastAPI backend for the Busy38 management UI MVP."""

from __future__ import annotations

import asyncio
import logging
import hashlib
import importlib.util
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
import re

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Query,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

try:
    from core.cognition.attachment_intake import (
        ATTACHMENT_DECISION_ACCEPT,
        ATTACHMENT_DECISION_BLOCK,
        ATTACHMENT_DECISION_QUARANTINE,
        make_intake_decision as _make_import_intake_decision,
    )
except Exception:  # pragma: no cover
    from core.attachments.intake import (
        ATTACHMENT_DECISION_ACCEPT,
        ATTACHMENT_DECISION_BLOCK,
        ATTACHMENT_DECISION_QUARANTINE,
        assess_attachment_intake as _make_import_intake_decision,
    )

try:
    from core.inference.provider_catalog import filter_catalog_models as _filter_models_from_catalog
except Exception:  # pragma: no cover
    _filter_models_from_catalog = None

from .import_adapters import get_import_adapter
from .import_contract import checksum_payload

from . import mobile_pairing, storage
from .runtime import RuntimeActionResult, load_runtime_adapter
from core.bridge.appearance_preferences import (
    AppearancePreferencesError,
    apply_appearance_update,
    load_appearance_preferences,
    write_appearance_preferences,
)

try:
    from core.plugins.state import (
        get_core_plugin_requirements,
        is_base_required_core_plugin,
        is_known_core_plugin,
        is_required_core_plugin,
        resolve_core_plugin_alias,
        set_required_plugin,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("busy38 plugin state module unavailable") from exc
 

app = FastAPI(title="Busy38 Management UI API", version="0.2.0")
_STARTUP_DEBUG_LOGGER = logging.getLogger("busy38.management.startup")
_STARTUP_PLUGIN_DEBUG_CHECKS_RAN = False
_WEB_ROOT = Path(__file__).resolve().parents[2] / "web"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_TICKETING_PROVIDER_DEFAULT = "busy-38-gticket"
_TICKETING_PROVIDER_MISSING_PLUGIN = "P_TICKETING_PROVIDER_MISSING_PLUGIN"
_TICKETING_PROVIDER_MISSING_MANIFEST = "P_TICKETING_PROVIDER_MISSING_MANIFEST"
_TICKETING_PROVIDER_INVALID_MANIFEST = "P_TICKETING_PROVIDER_INVALID_MANIFEST"
_TICKETING_PROVIDER_MISSING_REQUIRED_API = "P_TICKETING_PROVIDER_MISSING_REQUIRED_API"
_TICKETING_PROVIDER_CAPABILITY_MISMATCH = "P_TICKETING_PROVIDER_CAPABILITY_MISMATCH"
_TICKETING_PROVIDER_INCOMPATIBLE = "P_TICKETING_PROVIDER_INCOMPATIBLE"
_TICKETING_PROVIDER_OK = "P_TICKETING_PROVIDER_OK"
_TICKETING_PROVIDER_EXTERNAL_ENABLE_ENV = "BUSY38_TICKETING_ALLOW_EXTERNAL_PROVIDER"


_CORE_PLUGIN_REFERENCE = [
    {
        "plugin_id": str(requirement.canonical),
        "aliases": list(requirement.aliases),
        "signature_required": bool(requirement.signed),
    }
    for requirement in get_core_plugin_requirements()
]

_CORE_PLUGIN_DEPENDENCIES: Dict[str, List[str]] = {
    "openclaw-canvas-for-busy38": ["busy-38-management-ui"],
}
_GM_TICKET_EXECUTION_ROLES = {
    "portia",
    "nora",
    "mini",
    "mission_agent",
}


def _iter_core_plugin_reference() -> List[Dict[str, Any]]:
    reference = []
    for entry in _CORE_PLUGIN_REFERENCE:
        if not isinstance(entry, dict):
            continue
        plugin_id = str(entry.get("plugin_id") or "").strip()
        if not plugin_id:
            continue
        required = bool(is_required_core_plugin(plugin_id))
        base_required = bool(is_base_required_core_plugin(plugin_id))
        dynamic_entry = dict(entry)
        dynamic_entry.update(
            {
                "required": required,
                "base_required": base_required,
                "required_override": bool(required and not base_required),
                "aliases": list(entry.get("aliases") or []),
                "depends_on": list(_CORE_PLUGIN_DEPENDENCIES.get(plugin_id, [])),
            }
        )
        reference.append(dynamic_entry)
    return reference


def _core_alias_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


_CORE_PLUGIN_INDEX = {
    _core_alias_key(alias): entry
    for entry in _iter_core_plugin_reference()
    for alias in (
        [str(entry.get("plugin_id") or "").strip()]
        + [str(candidate).strip() for candidate in (entry.get("aliases") or [])]
    )
    if alias
}

_PLUGIN_MANIFEST_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
_PLUGIN_UI_MANIFEST_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def _coerce_json_obj(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _normalize_core_plugin_dependency_id(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized.startswith("busy-38-"):
        return f"busy38-{normalized[len('busy-38-'):]}"
    return normalized


def _core_plugin_display_id(entry: Dict[str, Any]) -> str:
    aliases = [str(alias).strip().lower() for alias in (entry.get("aliases") or []) if str(alias).strip()]
    for alias in aliases:
        if "-" not in alias and "_" not in alias:
            return alias

    plugin_id = str(entry.get("plugin_id") or "").strip().lower()
    if plugin_id.startswith("busy-38-"):
        return f"busy38-{plugin_id[len('busy-38-'):]}"
    return plugin_id


def _load_plugin_manifest(source: str) -> Optional[Dict[str, Any]]:
    source_path = _resolve_plugin_source_path(source)
    if source_path is None or not source_path.is_dir():
        return None

    manifest_path = source_path / "manifest.json"
    if not manifest_path.exists():
        return None

    cache_key = str(manifest_path.resolve())
    if cache_key in _PLUGIN_MANIFEST_CACHE:
        return _PLUGIN_MANIFEST_CACHE[cache_key]

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (json.JSONDecodeError, OSError):
        loaded = None

    payload = loaded if isinstance(loaded, dict) else None
    _PLUGIN_MANIFEST_CACHE[cache_key] = payload
    return payload


def _load_plugin_ui_manifest(source: str) -> Optional[Dict[str, Any]]:
    source_path = _resolve_plugin_source_path(source)
    if source_path is None or not source_path.is_dir():
        return None

    ui_manifest_path = source_path / "ui" / "manifest.json"
    if not ui_manifest_path.exists():
        return None

    cache_key = str(ui_manifest_path.resolve())
    if cache_key in _PLUGIN_UI_MANIFEST_CACHE:
        return _PLUGIN_UI_MANIFEST_CACHE[cache_key]

    try:
        with open(ui_manifest_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (json.JSONDecodeError, OSError):
        loaded = None

    payload = loaded if isinstance(loaded, dict) else None
    _PLUGIN_UI_MANIFEST_CACHE[cache_key] = payload
    return payload


def _extract_ui_contract(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    if isinstance(value.get("ui"), dict):
        return _coerce_json_obj(value["ui"])

    if "sections" in value or "summary" in value or "type" in value:
        return _coerce_json_obj(value)
    return None


def _merge_ui_contracts(base: Dict[str, Any], local: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(base, dict):
        return _coerce_json_obj(local)
    if not isinstance(local, dict):
        return dict(base)

    merged = dict(base)
    base_sections = merged.get("sections")
    local_sections = local.get("sections")
    if isinstance(base_sections, list) and isinstance(local_sections, list):
        merged_sections = [dict(section) if isinstance(section, dict) else {} for section in base_sections]
        index: Dict[str, int] = {
            str(section.get("id")).strip(): idx
            for idx, section in enumerate(merged_sections)
            if isinstance(section, dict) and str(section.get("id") or "").strip()
        }

        for section in local_sections:
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("id") or "").strip()
            section_actions = [
                dict(action)
                for action in section.get("actions", [])
                if isinstance(action, dict)
            ]
            if not section_id or section_id not in index:
                merged_sections.append(dict(section))
                continue

            existing_index = index[section_id]
            existing_section = merged_sections[existing_index]
            existing_actions = [
                dict(action)
                for action in existing_section.get("actions", [])
                if isinstance(action, dict)
            ]
            existing_action_ids = {
                str(action.get("id") or "").strip()
                for action in existing_actions
                if str(action.get("id") or "").strip()
            }
            for action in section_actions:
                action_id = str(action.get("id") or "").strip()
                if not action_id or action_id in existing_action_ids:
                    continue
                existing_actions.append(action)
                existing_action_ids.add(action_id)
            if existing_actions:
                existing_section["actions"] = existing_actions

            for key, value in section.items():
                if key == "id":
                    continue
                if key in existing_section:
                    continue
                existing_section[key] = value
            merged_sections[existing_index] = existing_section

        merged["sections"] = merged_sections

    merged_required_api = list(merged.get("required_api") or [])
    local_required_api = local.get("required_api")
    if local_required_api is not None:
        required_api_items = [item for item in merged_required_api if item]
        for item in local_required_api if isinstance(local_required_api, list) else [local_required_api]:
            normalized = str(item).strip()
            if normalized and normalized not in required_api_items:
                required_api_items.append(normalized)
        merged["required_api"] = required_api_items

    for key in ("type", "version", "summary", "plugin_identity"):
        if key not in merged and key in local:
            merged[key] = local[key]

    return merged


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_ticketing_provider_id(provider_id: str) -> str:
    normalized = str(provider_id or "").strip()
    if not normalized:
        normalized = _TICKETING_PROVIDER_DEFAULT
    resolved = resolve_core_plugin_alias(normalized)
    return str(resolved or normalized)


def _normalize_ticketing_endpoint(entry: str) -> str:
    return str(entry or "").strip().lower()


def _resolve_ticketing_provider_path(provider_id: str) -> tuple[str, Optional[Path]]:
    vendor_root = Path(os.getenv("BUSY38_VENDOR_DIR") or "./vendor").resolve()
    configured = str(provider_id or "").strip() or _TICKETING_PROVIDER_DEFAULT
    candidates: list[str] = []

    alias = resolve_core_plugin_alias(configured)
    if alias:
        candidates.append(alias)
    if configured not in candidates:
        candidates.append(configured)

    selected_provider = configured
    selected_path: Optional[Path] = None
    for candidate in candidates:
        candidate_path = vendor_root / str(candidate)
        if candidate_path.exists() and candidate_path.is_dir():
            selected_provider = str(candidate)
            selected_path = candidate_path
            break

    return selected_provider, selected_path


def _required_ticketing_api_variants(plugin_id: str, template: str) -> set[str]:
    normalized_plugin = str(plugin_id).strip().lower()
    return {
        _normalize_ticketing_endpoint(template),
        _normalize_ticketing_endpoint(template.replace("{plugin_id}", normalized_plugin)),
    }


def _collect_ticketing_contract_issues(manifest: Dict[str, Any], provider_id: str) -> list[tuple[str, str]]:
    ticketing_contract = manifest.get("ticketing")
    if not isinstance(ticketing_contract, dict):
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest ticketing contract is required",
        )]

    contract_version = ticketing_contract.get("contract_version")
    if not isinstance(contract_version, int) or contract_version != 1:
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest.ticketing.contract_version must be exactly 1 (received {contract_version!r})",
        )]

    required_api = ticketing_contract.get("required_api")
    if not isinstance(required_api, list):
        return [(
            _TICKETING_PROVIDER_MISSING_REQUIRED_API,
            f"{provider_id} manifest.ticketing.required_api is required and must be a list",
        )]

    normalized_required_api = {
        _normalize_ticketing_endpoint(value)
        for value in required_api
        if isinstance(value, str) and _normalize_ticketing_endpoint(value)
    }
    if not normalized_required_api:
        return [(
            _TICKETING_PROVIDER_MISSING_REQUIRED_API,
            f"{provider_id} manifest.ticketing.required_api is empty",
        )]

    expected_paths = {
        "create": _required_ticketing_api_variants(provider_id, "/api/plugins/{plugin_id}/tickets"),
        "read": _required_ticketing_api_variants(provider_id, "/api/plugins/{plugin_id}/tickets/{ticket_id}"),
        "comment": _required_ticketing_api_variants(
            provider_id,
            "/api/plugins/{plugin_id}/tickets/{ticket_id}/comments",
        ),
        "assign": _required_ticketing_api_variants(
            provider_id,
            "/api/plugins/{plugin_id}/tickets/{ticket_id}/assign",
        ),
        "close": _required_ticketing_api_variants(
            provider_id,
            "/api/plugins/{plugin_id}/tickets/{ticket_id}/close",
        ),
        "ui_debug": _required_ticketing_api_variants(
            provider_id,
            "/api/plugins/{plugin_id}/ui/debug",
        ),
    }
    missing_api: list[str] = []
    for name, alternatives in expected_paths.items():
        if alternatives.isdisjoint(normalized_required_api):
            missing_api.append(name)
    if missing_api:
        return [(
            _TICKETING_PROVIDER_MISSING_REQUIRED_API,
            f"{provider_id} manifest.ticketing.required_api missing entries for: {', '.join(sorted(missing_api))}",
        )]

    capabilities = ticketing_contract.get("capabilities")
    if not isinstance(capabilities, list):
        return [(
            _TICKETING_PROVIDER_CAPABILITY_MISMATCH,
            f"{provider_id} manifest.ticketing.capabilities is required and must be a list",
        )]

    normalized_capabilities = [
        str(capability).strip()
        for capability in capabilities
        if isinstance(capability, str) and str(capability).strip()
    ]
    required_capabilities = {"create", "read", "comment", "assign", "close"}
    missing_capabilities = required_capabilities.difference(set(normalized_capabilities))
    if missing_capabilities:
        return [(
            _TICKETING_PROVIDER_CAPABILITY_MISMATCH,
            f"{provider_id} manifest.ticketing.capabilities missing required values: {', '.join(sorted(missing_capabilities))}",
        )]

    lifecycle_contract = ticketing_contract.get("lifecycle")
    if not isinstance(lifecycle_contract, dict):
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest.ticketing.lifecycle is required and must be an object",
        )]

    dispatch_required = lifecycle_contract.get("dispatch_required")
    if dispatch_required is not None and not isinstance(dispatch_required, bool):
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest.ticketing.lifecycle.dispatch_required must be a boolean",
        )]

    if "phase2_required" not in lifecycle_contract:
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest.ticketing.lifecycle.phase2_required is required",
        )]

    phase2_required = lifecycle_contract.get("phase2_required")
    if not isinstance(phase2_required, list):
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest.ticketing.lifecycle.phase2_required is required and must be a list",
        )]

    normalized_phase2_required: list[str] = []
    for value in phase2_required:
        if isinstance(value, str):
            normalized_value = value.strip()
            if normalized_value:
                normalized_phase2_required.append(normalized_value)

    if not normalized_phase2_required:
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest.ticketing.lifecycle.phase2_required is empty",
        )]

    required_phase2 = {"request_id", "build_ticket_id", "build_batch_id"}
    missing_phase2_fields = required_phase2.difference(set(normalized_phase2_required))
    if missing_phase2_fields:
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest.ticketing.lifecycle.phase2_required is missing required values: "
            f"{', '.join(sorted(missing_phase2_fields))}",
        )]

    supports_hard_close = lifecycle_contract.get("supports_hard_close")
    if supports_hard_close is not None and not isinstance(supports_hard_close, bool):
        return [(
            _TICKETING_PROVIDER_INCOMPATIBLE,
            f"{provider_id} manifest.ticketing.lifecycle.supports_hard_close must be a boolean",
        )]

    return []


def _collect_ticketing_provider_status(plugin_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    configured_provider_id = os.getenv("BUSY38_TICKETING_PROVIDER", "").strip() or _TICKETING_PROVIDER_DEFAULT
    selected_provider, provider_path = _resolve_ticketing_provider_path(configured_provider_id)
    provider_path_resolved = str(provider_path) if provider_path else None
    selected_normalized = _normalize_ticketing_provider_id(selected_provider)
    default_normalized = _normalize_ticketing_provider_id(_TICKETING_PROVIDER_DEFAULT)
    allow_external = _env_truthy(os.getenv(_TICKETING_PROVIDER_EXTERNAL_ENABLE_ENV, ""))

    status: Dict[str, Any] = {
        "configured_provider_id": configured_provider_id,
        "selected_provider_id": selected_provider,
        "provider_path": provider_path_resolved,
        "provider_is_default": selected_normalized == default_normalized,
        "status": "ok",
        "reason_code": _TICKETING_PROVIDER_OK,
        "message": None,
        "ticketing_contract": None,
        "provider_record": None,
        "allow_external_provider": allow_external,
        "contract_issues": [],
        "has_ui_contract": False,
    }

    if not status["provider_is_default"] and not allow_external:
        status.update(
            status="error",
            reason_code=_TICKETING_PROVIDER_INCOMPATIBLE,
            message=(
                f"ticketing provider '{selected_provider}' requires explicit promotion via "
                f"{_TICKETING_PROVIDER_EXTERNAL_ENABLE_ENV}=1 to replace default provider"
            ),
        )
        return status

    if provider_path is None:
        status.update(
            status="error",
            reason_code=_TICKETING_PROVIDER_MISSING_PLUGIN,
            message=f"ticketing provider '{configured_provider_id}' is missing from vendor directory",
        )
        return status

    manifest = _load_plugin_manifest(str(provider_path)) if provider_path is not None else None
    if manifest is None:
        status.update(
            status="error",
            reason_code=_TICKETING_PROVIDER_MISSING_MANIFEST,
            message=f"{selected_provider} manifest.json is required",
        )
        return status

    issues = _collect_ticketing_contract_issues(manifest, selected_provider)
    if issues:
        reason_code, message = issues[0]
        status.update(status="error", reason_code=reason_code, message=message)
        status["contract_issues"] = [{"reason_code": reason_code, "message": message}]
    else:
        status["ticketing_contract"] = {
            "contract_version": manifest.get("ticketing", {}).get("contract_version"),
            "required_api": manifest.get("ticketing", {}).get("required_api"),
            "capabilities": manifest.get("ticketing", {}).get("capabilities"),
            "lifecycle": manifest.get("ticketing", {}).get("lifecycle"),
        }

    ui_contract = manifest.get("ui") if isinstance(manifest, dict) else None
    if isinstance(ui_contract, dict):
        status["has_ui_contract"] = bool(_extract_ui_contract(ui_contract))

    selected_record = None
    selected_record_id = ""
    if isinstance(plugin_map, dict):
        lookup_keys = [selected_provider, configured_provider_id, selected_normalized]
        for lookup_id in lookup_keys:
            normalized_lookup = _normalize_ticketing_provider_id(lookup_id)
            if not normalized_lookup:
                continue
            for candidate_id, candidate_record in plugin_map.items():
                if not isinstance(candidate_record, dict):
                    continue
                candidate_normalized = _normalize_ticketing_provider_id(str(candidate_id))
                if candidate_normalized == normalized_lookup:
                    selected_record = candidate_record
                    selected_record_id = str(candidate_id).strip()
                    break
            if selected_record is not None:
                break
    if isinstance(selected_record, dict):
        status["provider_record"] = {
            "plugin_id": selected_record_id or str(selected_provider).strip(),
            "name": selected_record.get("name"),
            "source": selected_record.get("source"),
            "enabled": bool(selected_record.get("enabled", False)),
            "present_in_registry": True,
        }
    else:
        status["provider_record"] = {
            "plugin_id": str(selected_provider).strip(),
            "present_in_registry": False,
        }
    return status


def _enrich_plugin_metadata(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}

    metadata = _coerce_json_obj(contract.get("metadata"))
    metadata_copy = dict(metadata)

    source_path = contract.get("source")
    manifest = _load_plugin_manifest(str(source_path)) if isinstance(source_path, str) else None
    ui_manifest = _load_plugin_ui_manifest(str(source_path)) if isinstance(source_path, str) else None
    if not manifest and not ui_manifest:
        return metadata_copy

    manifest_payload = manifest or {}
    local_ui = _extract_ui_contract(ui_manifest) if isinstance(ui_manifest, dict) else None
    manifest_ui = _extract_ui_contract(manifest_payload.get("ui")) if isinstance(manifest_payload.get("ui"), dict) else None
    if "ui" not in metadata_copy:
        if local_ui is not None:
            metadata_copy["ui"] = local_ui
        elif manifest_ui is not None:
            metadata_copy["ui"] = manifest_ui
    elif isinstance(metadata_copy.get("ui"), dict):
        metadata_copy["ui"] = _merge_ui_contracts(metadata_copy["ui"], local_ui)
        if "required_api" not in metadata_copy["ui"] and manifest_ui is not None:
            metadata_copy["ui"] = _merge_ui_contracts(metadata_copy["ui"], manifest_ui)

    for key in ("depends_on", "required_api", "required_core_plugins", "signature"):
        if key not in metadata_copy and key in manifest:
            metadata_copy[key] = manifest[key]
        elif key not in metadata_copy and ui_manifest and key in ui_manifest:
            metadata_copy[key] = ui_manifest[key]

    if "ticketing" not in metadata_copy and isinstance(manifest.get("ticketing"), dict):
        metadata_copy["ticketing"] = _coerce_json_obj(manifest.get("ticketing"))

    if "plugin_identity" not in metadata_copy and isinstance(manifest.get("plugin_identity"), dict):
        metadata_copy["plugin_identity"] = manifest["plugin_identity"]

    if "type" not in metadata_copy and "type" in manifest:
        metadata_copy["type"] = manifest["type"]

    return metadata_copy


def _enrich_plugin_record(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}

    payload = dict(contract)
    payload["metadata"] = _enrich_plugin_metadata(payload)
    return payload


def _collect_core_plugin_presence_report(plugin_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    plugin_rows = {str(plugin_id): plugin for plugin_id, plugin in (plugin_map or {}).items()}
    plugin_ids = list(plugin_rows.keys())

    reference_lookup = [r for r in _iter_core_plugin_reference() if isinstance(r, dict)]
    report_entries: List[Dict[str, Any]] = []
    required_total = 0
    required_present = 0
    optional_total = 0
    optional_present = 0
    required_missing: List[str] = []
    optional_missing: List[str] = []
    conflicts: List[str] = []

    for entry in reference_lookup:
        plugin_id = _core_plugin_display_id(entry)
        if not plugin_id:
            continue

        aliases = [plugin_id]
        aliases.extend([str(alias).strip() for alias in (entry.get("aliases") or []) if str(alias).strip()])
        reference_aliases = {_core_alias_key(alias) for alias in aliases if str(alias).strip()}
        reference_matches = {_core_alias_key(resolve_core_plugin_alias(alias)) for alias in aliases if resolve_core_plugin_alias(alias)}

        matches: List[str] = []
        for plugin_id_entry in plugin_ids:
            normalized_entry = _core_alias_key(plugin_id_entry)
            if normalized_entry in reference_aliases:
                matches.append(plugin_id_entry)
                continue

            resolved = resolve_core_plugin_alias(plugin_id_entry)
            if resolved and _core_alias_key(resolved) in reference_matches:
                matches.append(plugin_id_entry)

        # Deduplicate preserve order for deterministic output
        deduped: List[str] = []
        seen_matches: set[str] = set()
        for match in matches:
            if match in seen_matches:
                continue
            seen_matches.add(match)
            deduped.append(match)
        matches = deduped

        required = bool(entry.get("required", False))
        required_override = bool(entry.get("required_override", False))
        base_required = bool(entry.get("base_required", required))
        matched_id: Optional[str] = None
        alias_match: Optional[str] = None
        reason_code = "P_PLUGIN_MISSING_REQUIRED" if required else "P_PLUGIN_MISSING_OPTIONAL"
        present = False

        if not matches:
            if required:
                required_missing.append(plugin_id)
            else:
                optional_missing.append(plugin_id)
        elif len(matches) > 1:
            reason_code = "P_PLUGIN_NAMESPACE_CONFLICT"
            alias_match = _core_alias_key(matches[0])
            conflicts.append(plugin_id)
            matched_id = matches[0]
            if required:
                required_missing.append(plugin_id)
            else:
                optional_missing.append(plugin_id)
        else:
            matched_id = matches[0]
            plugin_record = plugin_rows.get(matched_id, {})
            plugin_enabled = bool(plugin_record.get("enabled", True))
            alias_match = str(matched_id)
            if plugin_enabled:
                reason_code = "P_PLUGIN_PRESENT_OK"
                present = True
                if required:
                    required_present += 1
                else:
                    optional_present += 1
            else:
                if required:
                    required_missing.append(plugin_id)
                else:
                    optional_missing.append(plugin_id)

        if required:
            required_total += 1
        else:
            optional_total += 1

        report_entries.append(
            {
                "plugin_id": plugin_id,
                "required": required,
                "base_required": base_required,
                "required_override": required_override,
                "signature_required": bool(entry.get("signature_required", False)),
                "aliases": aliases,
                "alias_match": alias_match,
                "matched_plugin_id": matched_id,
                "present": present,
                "reason_code": reason_code,
            },
        )

    return {
        "plugins": report_entries,
        "summary": {
            "required_total": required_total,
            "required_present": required_present,
            "required_missing": required_missing,
            "optional_total": optional_total,
            "optional_present": optional_present,
            "optional_missing": optional_missing,
            "state": "READY" if not required_missing and not conflicts else "BLOCKED",
            "conflicts": conflicts,
        },
    }


def _build_core_plugin_coverage_report(plugin_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    presence = _collect_core_plugin_presence_report(plugin_map)
    coverage: List[Dict[str, Any]] = []
    for entry in presence.get("plugins", []):
        if not bool(entry.get("required")):
            continue

        plugin_id = str(entry.get("plugin_id") or "").strip()
        matched_plugin_id = entry.get("matched_plugin_id")
        plugin_record = plugin_map.get(str(matched_plugin_id), {}) if isinstance(matched_plugin_id, str) else {}
        metadata = plugin_record.get("metadata") if isinstance(plugin_record, dict) else None
        has_ui_contract = isinstance(metadata, dict) and isinstance(metadata.get("ui"), dict)
        has_debug_action = bool(_find_plugin_ui_action(plugin_record, "debug")) if isinstance(plugin_record, dict) else False
        plugin_enabled = bool(plugin_record.get("enabled")) if isinstance(plugin_record, dict) else False
        present = bool(entry.get("present"))

        if not present:
            coverage_state = "missing"
        elif not plugin_enabled:
            coverage_state = "disabled"
        elif not has_ui_contract:
            coverage_state = "ui_contract_missing"
        elif not has_debug_action:
            coverage_state = "debug_missing"
        else:
            coverage_state = "covered"

        aliases = [str(item) for item in (entry.get("aliases") or []) if str(item).strip()]
        if not aliases and plugin_id:
            aliases = [plugin_id]

        coverage.append(
            {
                "plugin_id": plugin_id,
                "required": bool(entry.get("required")),
                "base_required": bool(entry.get("base_required")),
                "required_override": bool(entry.get("required_override")),
                "signature_required": bool(entry.get("signature_required")),
                "aliases": aliases,
                "alias_match": entry.get("alias_match"),
                "matched_plugin_id": matched_plugin_id,
                "present": present,
                "plugin_enabled": plugin_enabled,
                "has_ui_contract": has_ui_contract,
                "has_debug_action": has_debug_action,
                "reason_code": entry.get("reason_code"),
                "coverage_state": coverage_state,
            },
        )

    covered_total = sum(1 for item in coverage if item["coverage_state"] == "covered")
    summary = {
        "required_total": len(coverage),
        "required_present": len([item for item in coverage if item["present"]]),
        "required_missing": len([item for item in coverage if item["coverage_state"] == "missing"]),
        "required_disabled": len([item for item in coverage if item["coverage_state"] == "disabled"]),
        "required_ui_missing": len([item for item in coverage if item["coverage_state"] == "ui_contract_missing"]),
        "required_debug_missing": len([item for item in coverage if item["coverage_state"] == "debug_missing"]),
        "required_covered": covered_total,
    }

    return {
        "required_plugins": coverage,
        "summary": summary,
        "required_overall_state": "READY" if summary["required_missing"] == 0 and summary["required_disabled"] == 0 and summary["required_ui_missing"] == 0 and summary["required_debug_missing"] == 0 else "BLOCKED",
        "updated_at": _now_iso(),
    }

class SettingsUpdate(BaseModel):
    heartbeat_interval: Optional[int] = None
    fallback_budget_per_hour: Optional[int] = None
    auto_restart: Optional[bool] = None
    proxy_http: Optional[str] = None
    proxy_https: Optional[str] = None
    proxy_no_proxy: Optional[str] = None


class AppearanceUpdate(BaseModel):
    override_enabled: Optional[bool] = None
    sync_theme_preferences: Optional[bool] = None
    shared_theme_mode: Optional[Literal["system", "light", "dark"]] = None
    desktop_theme_mode: Optional[Literal["system", "light", "dark"]] = None
    contrast_policy: Optional[Literal["aa", "aaa"]] = None
    motion_policy: Optional[Literal["default", "reduced"]] = None
    color_separation_policy: Optional[Literal["default", "stronger"]] = None
    text_spacing_policy: Optional[Literal["default", "increased"]] = None


class PluginUpdate(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    status: Optional[str] = None
    source: Optional[str] = None
    kind: Optional[str] = None
    command: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class PluginCreate(BaseModel):
    id: str
    name: str
    source: str
    kind: str
    status: str = "configured"
    enabled: bool = True
    command: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ProviderUpdate(BaseModel):
    enabled: Optional[bool] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    priority: Optional[int] = None
    status: Optional[str] = None
    display_name: Optional[str] = None
    kind: Optional[str] = None
    fallback_models: Optional[List[str]] = None
    retries: Optional[int] = None
    timeout_ms: Optional[int] = None
    tool_timeout_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ProviderModelDiscovery(BaseModel):
    api_key: Optional[str] = None
    endpoint: Optional[str] = None


class ProviderModelTest(BaseModel):
    api_key: Optional[str] = None
    endpoint: Optional[str] = None


class ProviderTestAllRequest(BaseModel):
    include_disabled: bool = False
    provider_ids: Optional[List[str]] = None


class ProviderSecretAction(BaseModel):
    action: Literal["set", "rotate", "clear"]
    api_key: Optional[str] = None


class ProviderCreate(BaseModel):
    id: str
    name: str
    endpoint: str
    model: Optional[str] = None
    status: str = "configured"
    priority: int = 100
    enabled: bool = True
    display_name: Optional[str] = None
    kind: Optional[str] = None
    fallback_models: Optional[List[str]] = None
    retries: Optional[int] = None
    timeout_ms: Optional[int] = None
    tool_timeout_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentUpdate(BaseModel):
    enabled: Optional[bool] = None
    name: Optional[str] = None
    role: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    overlay_content: Optional[str] = None
    overlay_token_cap: Optional[int] = None


class AgentLifecycleAction(BaseModel):
    reason: Optional[str] = None
    replacement_agent_id: Optional[str] = None
    actor: Optional[str] = None


class MemoryCreate(BaseModel):
    scope: str
    type: str
    content: str


class ChatHistoryCreate(BaseModel):
    agent_id: str
    summary: str
    chat_session_id: Optional[str] = None


class ToolUsageCreate(BaseModel):
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    mission_id: Optional[str] = None
    request_id: Optional[str] = None
    context_type: Optional[str] = None
    context_id: Optional[str] = None
    memory_id: Optional[str] = None
    chat_message_id: Optional[str] = None
    chat_session_id: Optional[str] = None
    status: str = "executed"
    duration_ms: Optional[int] = None
    result_status: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None


class ImportDecisionRequest(BaseModel):
    import_item_ids: List[str]
    review_state: str
    actor: Optional[str] = None
    note: Optional[str] = None
    agent_scope: Optional[str] = None


class DirectoryScopeReassignRequest(BaseModel):
    source_scope: str
    target_scope: str
    import_item_ids: Optional[List[str]] = None
    import_id: Optional[str] = None
    actor: Optional[str] = None
    note: Optional[str] = None


class GmTicketCreate(BaseModel):
    title: str
    status: Optional[str] = None
    priority: Optional[str] = None
    agent_scope: Optional[str] = None
    phase: Optional[str] = None
    requested_by: str
    assigned_to: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GmTicketUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    agent_scope: Optional[str] = None
    phase: Optional[str] = None
    assigned_to: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    closed_at: Optional[str] = None


class GmTicketMessageCreate(BaseModel):
    sender: str
    content: str
    message_type: Optional[str] = None
    response_required: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None


class GmDirectMessageCreate(BaseModel):
    sender: str
    content: str
    ticket_id: Optional[str] = None
    title: Optional[str] = None
    message_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GmTicketDispatch(BaseModel):
    objective: Optional[str] = None
    mission_id: Optional[str] = None
    assigned_to: Optional[str] = None
    role: Optional[str] = None
    request_id: Optional[str] = None
    gm_request_id: Optional[str] = None
    dispatch_token: Optional[str] = None
    dispatch_nonce: Optional[str] = None
    build_ticket_id: Optional[str] = None
    build_batch_id: Optional[str] = None
    max_steps: Optional[int] = None
    qa_max_retries: Optional[int] = None
    acceptance_criteria: Optional[List[str]] = None
    allowed_namespaces: Optional[List[str]] = None


class PairingIssueRequest(BaseModel):
    device_label: str
    authorized_room_ids: List[str]
    orchestrator_scope: List[str]
    ttl_sec: int


class PairingExchangeRequest(BaseModel):
    pairing_code: str
    device_label: Optional[str] = None
    expected_instance_id: Optional[str] = None


class PairingRevokeRequest(BaseModel):
    bridge_token: Optional[str] = None
    token_id: Optional[str] = None


@app.on_event("startup")
async def _startup() -> None:
    storage.ensure_schema()
    _run_startup_plugin_debug_checks_once()


runtime = load_runtime_adapter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_legacy_token = os.getenv("MANAGEMENT_API_TOKEN", "").strip()
_admin_token = os.getenv("MANAGEMENT_ADMIN_TOKEN", "").strip()
_viewer_token = os.getenv("MANAGEMENT_READ_TOKEN", "").strip()

if _admin_token and _viewer_token and _admin_token == _viewer_token:
    _viewer_token = ""
if not _admin_token:
    _admin_token = _legacy_token
if not _viewer_token:
    _viewer_token = _admin_token or _legacy_token

if _viewer_token and not _admin_token:
    _admin_token = _viewer_token
storage.set_db_path_override(os.getenv("MANAGEMENT_DB_PATH"))


def _role_from_token(auth_token: Optional[str] = None, query_token: Optional[str] = None) -> str:
    if not (_admin_token or _viewer_token):
        return "admin"

    token = auth_token or query_token or ""
    if token.startswith("Bearer "):
        token = token[7:].strip()

    if _admin_token and token == _admin_token:
        return "admin"
    if _viewer_token and token == _viewer_token:
        return "viewer"
    return ""


def _token_source(auth_token: Optional[str] = None, query_token: Optional[str] = None) -> str:
    if not (_admin_token or _viewer_token):
        return "open-access"

    token = auth_token or query_token or ""
    if token.startswith("Bearer "):
        token = token[7:].strip()

    if _admin_token and token == _admin_token:
        return "admin-token"
    if _viewer_token and token == _viewer_token:
        return "read-token"
    return "invalid-token"


def _readable_role_name(role: str) -> str:
    if role == "admin":
        return "admin"
    if role == "viewer":
        return "viewer"
    return "unknown"


def _coerce_json_metadata(value: Optional[Dict[str, Any] | str]) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed


def _is_sensitive_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in ("api_key", "token", "secret", "password", "credential"))


def _redact_metadata(metadata: Optional[str] | Dict[str, Any] | list, role: str) -> Optional[Any]:
    if role == "admin":
        return metadata
    if metadata is None:
        return None
    parsed = metadata
    if isinstance(metadata, str):
        parsed = _coerce_json_metadata(metadata)

    if isinstance(parsed, dict):
        return {k: "***redacted***" if _is_sensitive_key(k) else v for k, v in parsed.items()}

    return "***redacted***"


def _normalize_provider_metadata_fields(
    payload: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = {}
    if isinstance(metadata, dict):
        merged = dict(metadata)
    provided_metadata = payload.get("metadata")
    if isinstance(provided_metadata, dict):
        merged.update(provided_metadata)

    value = payload.get("display_name")
    if value is not None:
        display_name = str(value).strip()
        if display_name:
            merged["display_name"] = display_name
        else:
            merged.pop("display_name", None)

    value = payload.get("kind")
    if value is not None:
        normalized_kind = str(value).strip().lower()
        if normalized_kind:
            merged["kind"] = normalized_kind
        else:
            merged.pop("kind", None)

    value = payload.get("fallback_models")
    if value is not None:
        if isinstance(value, str):
            candidate_models = value.split(",")
        elif isinstance(value, list):
            candidate_models = value
        else:
            candidate_models = []
        merged["fallback_models"] = [str(item).strip() for item in candidate_models if str(item).strip()]

    value = payload.get("retries")
    if value is not None:
        merged["retries"] = int(value)

    value = payload.get("timeout_ms")
    if value is not None:
        merged["timeout_ms"] = int(value)

    value = payload.get("tool_timeout_ms")
    if value is not None:
        merged["tool_timeout_ms"] = int(value)

    return merged


def _require_role(
    request: Request,
    required: str = "viewer",
    token: Optional[str] = None,
) -> str:
    auth_token = request.headers.get("Authorization")
    role = _role_from_token(auth_token, token)
    if not role:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if required == "admin" and role != "admin":
        raise HTTPException(status_code=403, detail="Admin token required for this action.")
    return role


def _event_for(
    _role: str,
    event_type: str,
    message: str,
    level: str = "info",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = storage.append_event(event_type, message, level, payload=payload)
    return payload


def _coerce_agent_lifecycle_payload(
    role: str,
    action: Optional[AgentLifecycleAction],
    *,
    fallback_reason: str,
    replacement_allowed: bool = True,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"actor": role}
    if action is None:
        payload["reason"] = fallback_reason
        return payload

    values = action.model_dump(exclude_unset=True)
    reason = str(values.get("reason") or "").strip()
    actor = str(values.get("actor") or "").strip()
    replacement_agent_id = str(values.get("replacement_agent_id") or "").strip()

    payload["reason"] = reason or fallback_reason
    if actor:
        payload["request_actor"] = actor
    if replacement_agent_id and replacement_allowed:
        payload["replacement_agent_id"] = replacement_agent_id
    return payload


def _coerce_replacement_agent(agent_id: str, replacement_agent_id: str) -> Optional[str]:
    normalized_replacement = str(replacement_agent_id or "").strip()
    if not normalized_replacement:
        return None
    if normalized_replacement == agent_id:
        raise HTTPException(status_code=400, detail="replacement_agent_id must be different from target agent")

    candidates = storage.list_agents(storage.AGENT_LIFECYCLE_ACTIVE)
    for candidate in candidates:
        if candidate.get("id") == normalized_replacement:
            return normalized_replacement
    raise HTTPException(status_code=404, detail=f"replacement agent '{normalized_replacement}' not found or not active")


def _coerce_gm_dispatch_role(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized not in _GM_TICKET_EXECUTION_ROLES:
        raise HTTPException(status_code=400, detail=f"unsupported dispatch role: {value!r}")
    return normalized


def _coerce_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _sanitize_gm_ticket_payload(ticket: Dict[str, Any], role: str) -> Dict[str, Any]:
    payload = dict(ticket)
    payload["metadata"] = _redact_metadata(payload.get("metadata"), role)
    return payload


def _sanitize_gm_ticket_messages(messages: List[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    return [
        dict(msg, metadata=_redact_metadata(msg.get("metadata"), role), response_required=bool(msg.get("response_required", False)))
        for msg in messages
    ]


def _sanitize_gm_ticket_events(events: List[Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    return [
        dict(event, payload=_redact_metadata(event.get("payload"), role))
        for event in events
    ]


def _build_gm_ticket_audit_payload(
    ticket_id: str,
    role: str,
    *,
    event_limit: int = 100,
) -> Dict[str, Any]:
    ticket = storage.get_gm_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"ticket '{ticket_id}' not found")

    messages = _sanitize_gm_ticket_messages(storage.list_gm_ticket_messages(ticket_id), role)
    events = _sanitize_gm_ticket_events(storage.list_events(event_limit, gm_ticket_id=ticket_id), role)
    return {
        "ticket": _sanitize_gm_ticket_payload(ticket, role),
        "messages": messages,
        "events": events,
        "summary": {
            "message_count": len(messages),
            "event_count": len(events),
        },
        "updated_at": _now_iso(),
    }


def _sanitize_provider(provider: Dict[str, Any], role: str) -> Dict[str, Any]:
    payload = dict(provider)
    payload["metadata"] = _redact_metadata(payload.get("metadata"), role)
    return payload


def _sanitize_plugin(plugin: Dict[str, Any], role: str) -> Dict[str, Any]:
    payload = dict(plugin)
    payload["metadata"] = _redact_metadata(payload.get("metadata"), role)
    return payload


def _find_core_plugin_reference(plugin_id: str) -> Optional[Dict[str, Any]]:
    normalized = str(plugin_id).strip().lower()
    if not normalized:
        return None
    candidate = _CORE_PLUGIN_INDEX.get(_core_alias_key(normalized))
    if candidate is None:
        canonical = resolve_core_plugin_alias(normalized)
        if canonical:
            candidate = _CORE_PLUGIN_INDEX.get(_core_alias_key(canonical))
            if candidate is None:
                candidate = _CORE_PLUGIN_INDEX.get(_core_alias_key(str(canonical)))
    if candidate is None:
        for entry in _iter_core_plugin_reference():
            plugin_id_entry = str(entry.get("plugin_id") or "").strip().lower()
            aliases = {str(alias).strip().lower() for alias in (entry.get("aliases") or [])}
            if _core_alias_key(normalized) == _core_alias_key(plugin_id_entry) or (
                canonical and _core_alias_key(normalized) == _core_alias_key(canonical)
            ):
                candidate = entry
                break
            if canonical and canonical in aliases:
                candidate = entry
                break
    if candidate is None:
        return None

    canonical = str(candidate.get("plugin_id", "")).strip() or normalized
    required = bool(is_required_core_plugin(canonical))
    base_required = bool(is_base_required_core_plugin(canonical))
    display_id = _core_plugin_display_id(candidate)
    return {
        "plugin_id": display_id,
        "required": required,
        "base_required": base_required,
        "required_override": bool(required and not base_required),
        "signature_required": bool(candidate.get("signature_required", False)),
        "aliases": list(candidate.get("aliases") or [normalized]),
        "depends_on": list(candidate.get("depends_on") or []),
        "matched_alias": normalized,
        "canonical_plugin_id": canonical,
    }


def _plugin_ui_contract_diagnostics(
    plugin: Dict[str, Any],
) -> List[Dict[str, str]]:
    metadata = plugin.get("metadata") if isinstance(plugin, dict) else None
    ui_warnings: List[Dict[str, str]] = []
    if not isinstance(metadata, dict):
        ui_warnings.append(
            {
                "code": "P_PLUGIN_UI_METADATA_MISSING",
                "severity": "warn",
                "source": "ui",
                "message": "plugin metadata.ui contract is missing",
            },
        )
        return ui_warnings

    ui = metadata.get("ui")
    if not isinstance(ui, dict):
        ui_warnings.append(
            {
                "code": "P_PLUGIN_UI_CONTRACT_MISSING",
                "severity": "warn",
                "source": "ui",
                "message": "plugin has no ui contract",
            },
        )
        return ui_warnings

    source = plugin.get("source")
    if isinstance(source, str):
        source_path = _resolve_plugin_source_path(source)
        if source_path:
            ui_path = source_path / "ui"
            if not ui_path.exists():
                ui_warnings.append(
                    {
                        "code": "P_PLUGIN_UI_ASSET_MISSING",
                        "severity": "warn",
                        "source": "runtime",
                        "message": "plugin source exists but has no ui directory",
                    },
                )
            elif not any(ui_path.iterdir()):
                ui_warnings.append(
                    {
                        "code": "P_PLUGIN_UI_ASSET_EMPTY",
                        "severity": "warn",
                        "source": "runtime",
                        "message": "plugin ui directory exists but is empty",
                    },
                )

    sections = ui.get("sections")
    if not isinstance(sections, list):
        ui_warnings.append(
            {
                "code": "P_PLUGIN_UI_SECTIONS_INVALID",
                "severity": "warn",
                "source": "ui",
                "message": "plugin.ui.sections must be a list",
            },
        )
        ui_warnings.extend(_plugin_ui_handler_warnings_for_contract(plugin))
        return ui_warnings
    if not sections:
        ui_warnings.append(
            {
                "code": "P_PLUGIN_UI_EMPTY",
                "severity": "warn",
                "source": "ui",
                "message": "plugin.ui.sections is present but empty",
            },
        )
        ui_warnings.extend(_plugin_ui_handler_warnings_for_contract(plugin))
        return ui_warnings

    seen_action_ids: set[str] = set()
    for section in sections:
        if not isinstance(section, dict):
            ui_warnings.append(
                {
                    "code": "P_PLUGIN_UI_SECTION_INVALID",
                    "severity": "warn",
                    "source": "ui",
                    "message": "plugin.ui.sections contains a non-object section",
                },
            )
            continue
        section_title = str(section.get("id") or section.get("title") or "section")
        for action in section.get("actions", []):
            if not isinstance(action, dict):
                ui_warnings.append(
                    {
                        "code": "P_PLUGIN_UI_ACTION_INVALID",
                        "severity": "warn",
                        "source": "ui",
                        "message": f"plugin ui section '{section_title}' contains non-object action",
                    },
                )
                continue
            action_id = str(action.get("id") or "").strip()
            if not action_id:
                ui_warnings.append(
                    {
                        "code": "P_PLUGIN_UI_ACTION_ID_INVALID",
                        "severity": "warn",
                        "source": "ui",
                        "message": f"plugin ui section '{section_title}' has an action without an id",
                    },
                )
                continue
            if action_id in seen_action_ids:
                ui_warnings.append(
                    {
                        "code": "P_PLUGIN_UI_ACTION_DUPLICATE_ID",
                        "severity": "warn",
                        "source": "ui",
                        "message": f"plugin ui action id '{action_id}' appears multiple times",
                    },
                )
            seen_action_ids.add(action_id)

            method = str(action.get("method") or "POST").strip().upper()
            if method not in {"GET", "POST", "PATCH", "PUT", "DELETE", "HEAD", "OPTIONS"}:
                ui_warnings.append(
                    {
                        "code": "P_PLUGIN_UI_ACTION_METHOD_INVALID",
                        "severity": "warn",
                        "source": "ui",
                        "message": f"plugin ui action '{action_id}' uses unsupported method '{method}'",
                    },
                )

    ui_warnings.extend(_plugin_ui_handler_warnings_for_contract(plugin))
    return ui_warnings


def _resolve_plugin_source_path(source: str) -> Optional[Path]:
    """Resolve plugin `source` to a local path when possible."""
    if not isinstance(source, str):
        return None

    raw_source = source.strip()
    if not raw_source:
        return None

    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw_source):
        return None

    source_path = Path(raw_source)
    if source_path.is_absolute():
        return source_path if source_path.exists() else None

    candidate_paths = [
        Path(__file__).resolve().parent.parent / source_path,
        Path.cwd() / source_path,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return None


def _collect_plugin_dependencies(plugin: Dict[str, Any]) -> List[str]:
    metadata = plugin.get("metadata") if isinstance(plugin, dict) else None
    if not isinstance(metadata, dict):
        return []

    raw_dependencies = metadata.get("depends_on") or plugin.get("depends_on")
    if raw_dependencies is None:
        return []

    if isinstance(raw_dependencies, str):
        split_values = [part for part in raw_dependencies.replace(",", " ").split() if part.strip()]
    elif isinstance(raw_dependencies, (list, tuple)):
        split_values = list(raw_dependencies)
    else:
        return []

    normalized: List[str] = []
    for value in split_values:
        if not isinstance(value, str):
            continue
        dependency = _normalize_core_plugin_dependency_id(value)
        if dependency and dependency not in normalized:
            normalized.append(dependency)
    return normalized


def _collect_plugin_tool_conflicts(plugin_id: str) -> List[Dict[str, Any]]:
    normalized = str(plugin_id).strip()
    own_tools = storage.list_tools(plugin_id=normalized, status="active")
    if not own_tools:
        return []

    collisions: List[Dict[str, Any]] = []
    namespace_index: Dict[str, set[str]] = {}
    for tool in own_tools:
        namespace = str(tool.get("namespace") or "").strip()
        action = str(tool.get("action") or "").strip()
        if not namespace or not action:
            continue
        namespace_index.setdefault(namespace, set()).add(action)

    checked: set[tuple[str, str]] = set()
    for namespace, actions in namespace_index.items():
        if not actions:
            continue
        namespace_tools = storage.list_tools(namespace=namespace, status="active")
        for action in sorted(actions):
            key = (namespace, action)
            if key in checked:
                continue
            checked.add(key)
            owners = sorted({
                str(candidate.get("plugin_id") or "")
                for candidate in namespace_tools
                if str(candidate.get("action") or "").strip() == action
                and str(candidate.get("plugin_id") or "").strip() != normalized
                and str(candidate.get("plugin_id") or "").strip()
            })
            if owners:
                collisions.append(
                    {
                        "namespace": namespace,
                        "action": action,
                        "owners": owners,
                    },
                )

    return collisions


def _find_plugin_ui_action(contract: Dict[str, Any], action_id: str) -> Optional[Dict[str, Any]]:
    if not action_id:
        return None

    metadata = contract.get("metadata") if isinstance(contract, dict) else None
    if not isinstance(metadata, dict):
        return None

    ui = metadata.get("ui") if isinstance(metadata.get("ui"), dict) else None
    if ui is None:
        return None

    for section in ui.get("sections") or []:
        if not isinstance(section, dict):
            continue
        for action in section.get("actions") or []:
            if not isinstance(action, dict):
                continue
            if str(action.get("id") or "").strip() == action_id:
                return action
    return None


def _collect_plugin_ui_actions(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = contract.get("metadata") if isinstance(contract, dict) else None
    if not isinstance(metadata, dict):
        return []

    ui = metadata.get("ui") if isinstance(metadata.get("ui"), dict) else None
    if ui is None:
        return []

    sections = ui.get("sections") if isinstance(ui.get("sections"), list) else []
    actions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "").strip()
        section_title = section.get("title")
        for action in section.get("actions") or []:
            if not isinstance(action, dict):
                continue
            action_id = str(action.get("id") or "").strip()
            if not action_id or action_id in seen:
                continue
            seen.add(action_id)
            actions.append(
                {
                    "id": action_id,
                    "section_id": section_id,
                    "section_title": section_title,
                    "label": action.get("label"),
                    "method": str(action.get("method") or "POST").strip().upper() or "POST",
                }
            )
    return actions


def _log_startup_debug_message(level: str, message: str, payload: Dict[str, Any]) -> None:
    payload_text = "{}" if not payload else json.dumps(payload, sort_keys=True)
    log_line = f"startup plugin debug: {message} | {payload_text}"
    print(log_line)
    log_level = (level or "info").strip().lower()
    if log_level == "error":
        _STARTUP_DEBUG_LOGGER.error(log_line)
    elif log_level in {"warn", "warning"}:
        _STARTUP_DEBUG_LOGGER.warning(log_line)
    else:
        _STARTUP_DEBUG_LOGGER.info(log_line)


def _run_startup_plugin_debug_check(plugin_id: str, plugin: Dict[str, Any]) -> Dict[str, Any]:
    normalized_plugin_id = str(plugin_id).strip()
    if not normalized_plugin_id:
        return {"plugin_id": "", "runtime_called": False, "runtime_success": False, "has_debug_action": False, "status": "warn", "message": "invalid plugin id"}

    debug_action = _find_plugin_ui_action(plugin, "debug")
    has_debug_action = debug_action is not None
    if has_debug_action:
        runtime_called = True
        try:
            runtime_result = runtime.plugin_ui_action(
                plugin_id=normalized_plugin_id,
                action_id="debug",
                method="GET",
                action=debug_action,
                plugin_source=plugin.get("source"),
            )
        except Exception as exc:  # pragma: no cover
            runtime_result = RuntimeActionResult(
                success=False,
                message=f"startup plugin debug probe raised exception: {exc}",
                payload={
                    "action_id": "debug",
                    "method": "GET",
                    "reason": "runtime_exception",
                    "error": str(exc),
                },
            )
            runtime_called = True
    else:
        runtime_called = False
        runtime_result = RuntimeActionResult(
            success=False,
            message="debug action unavailable in plugin ui contract",
            payload={
                "action_id": "debug",
                "method": "GET",
                "reason": "action_not_declared_in_ui_contract",
            },
        )

    event_level = "info"
    if not runtime_result.success:
        if has_debug_action:
            event_level = "error"
        else:
            event_level = "warn"
    _event_for(
        "admin",
        "plugin.startup_debug",
        f"Startup plugin debug probe executed: {normalized_plugin_id}",
        event_level,
        {
            "plugin_id": normalized_plugin_id,
            "action_id": "debug",
            "method": "GET",
            "runtime_called": runtime_called,
            "runtime_success": runtime_result.success,
            "has_debug_action": has_debug_action,
            "message": runtime_result.message,
            "runtime_payload": runtime_result.payload,
            "plugin_enabled": bool(plugin.get("enabled")),
            "plugin_status": plugin.get("status"),
        },
    )
    _log_startup_debug_message(
        event_level,
        f"{normalized_plugin_id}: {runtime_result.message}",
        {
            "plugin_id": normalized_plugin_id,
            "action_id": "debug",
            "runtime_called": runtime_called,
            "runtime_success": runtime_result.success,
            "has_debug_action": has_debug_action,
            "plugin_enabled": bool(plugin.get("enabled")),
            "plugin_status": plugin.get("status"),
        },
    )
    return {
        "plugin_id": normalized_plugin_id,
        "runtime_called": runtime_called,
        "runtime_success": runtime_result.success,
        "has_debug_action": has_debug_action,
        "status": event_level,
        "message": runtime_result.message,
        "plugin_enabled": bool(plugin.get("enabled")),
        "plugin_status": plugin.get("status"),
    }


def _run_startup_plugin_debug_checks_once() -> None:
    global _STARTUP_PLUGIN_DEBUG_CHECKS_RAN
    if _STARTUP_PLUGIN_DEBUG_CHECKS_RAN:
        return
    _STARTUP_PLUGIN_DEBUG_CHECKS_RAN = True
    try:
        _run_startup_plugin_debug_checks()
    except Exception as exc:  # pragma: no cover
        _log_startup_debug_message(
            "error",
            "Startup plugin debug checks failed",
            {"error": str(exc)},
        )


def _run_startup_plugin_debug_checks() -> None:
    plugin_map = _plugin_map()
    ticketing_provider_status = _collect_ticketing_provider_status(plugin_map)
    ticketing_summary_label = (
        f"Ticketing provider '{ticketing_provider_status.get('selected_provider_id')}' "
        f"selection status: {ticketing_provider_status.get('status')}"
    )
    ticketing_level = "info"
    if str(ticketing_provider_status.get("status") or "").strip().lower() != "ok":
        ticketing_level = "error" if ticketing_provider_status.get("provider_is_default") else "warn"
    _event_for(
        "admin",
        "plugin.startup_debug",
        ticketing_summary_label,
        ticketing_level,
        dict(ticketing_provider_status),
    )
    _log_startup_debug_message(
        ticketing_level,
        ticketing_summary_label,
        dict(ticketing_provider_status),
    )

    core_presence = _collect_core_plugin_presence_report(plugin_map)
    core_summary = core_presence.get("summary", {})
    core_coverage = _build_core_plugin_coverage_report(plugin_map)
    core_coverage_summary = core_coverage.get("summary", {})
    required_missing_plugins = [str(item) for item in core_summary.get("required_missing", [])]
    required_disabled_plugins = [
        str(item.get("plugin_id") or "")
        for item in core_coverage.get("required_plugins", [])
        if item.get("coverage_state") == "disabled"
    ]
    required_disabled_plugins = [item for item in required_disabled_plugins if item]

    if not plugin_map:
        _event_for(
            "admin",
            "plugin.startup_debug",
            "Startup plugin debug probe skipped: no plugins discovered",
            "warn",
            {"count": 0},
        )
        _log_startup_debug_message(
            "warn",
            "Startup plugin debug probe skipped: no plugins discovered",
            {"count": 0},
        )
        error_count = len(required_missing_plugins)
        warn_count = len(required_missing_plugins)
        if str(ticketing_provider_status.get("status") or "").strip().lower() != "ok":
            if ticketing_provider_status.get("provider_is_default"):
                error_count += 1
            else:
                warn_count += 1
        startup_level = "error" if error_count else "warn"
        for plugin_id in required_missing_plugins:
            _event_for(
                "admin",
                "plugin.startup_debug",
                f"Required core plugin missing from plugin registry: {plugin_id}",
                "error",
                {
                    "plugin_id": plugin_id,
                    "required": True,
                    "runtime_called": False,
                    "runtime_success": False,
                    "presence": "missing",
                },
            )
            _log_startup_debug_message(
                "error",
                f"{plugin_id}: required core plugin missing from plugin registry",
                {"plugin_id": plugin_id, "required": True},
            )
        _event_for(
            "admin",
            "plugin.startup_debug_summary",
            "Startup plugin debug probe summary: no plugins discovered",
            startup_level,
            {
                "plugin_count": 0,
                "checked": 0,
                "runtime_called": 0,
                "runtime_success": 0,
                "required_total": int(core_summary.get("required_total", 0)),
                "required_present": int(core_summary.get("required_present", 0)),
                "required_missing": len(required_missing_plugins),
                "required_disabled": len(required_disabled_plugins),
                "missing_debug": 0,
                "warn_count": warn_count,
                "error_count": error_count,
                "required_missing_plugins": required_missing_plugins,
                "required_disabled_plugins": required_disabled_plugins,
                "missing_debug_plugins": [],
                "warn_plugins": [],
                "error_plugins": [],
                "checked_plugins": [],
                "ticketing_provider": ticketing_provider_status,
            },
        )
        return

    summary = {
        "plugin_count": len(plugin_map),
        "checked": 0,
        "runtime_called": 0,
        "runtime_success": 0,
        "required_total": int(core_summary.get("required_total", 0)),
        "required_present": int(core_summary.get("required_present", 0)),
        "required_missing": len(required_missing_plugins),
        "required_disabled": int(core_coverage_summary.get("required_disabled", 0)),
        "missing_debug": 0,
        "warn_count": 0,
        "error_count": 0,
        "required_missing_plugins": [],
        "required_disabled_plugins": required_disabled_plugins,
        "missing_debug_plugins": [],
        "warn_plugins": [],
        "error_plugins": [],
        "checked_plugins": [],
        "ticketing_provider": ticketing_provider_status,
    }
    if str(ticketing_provider_status.get("status") or "").strip().lower() != "ok":
        if ticketing_provider_status.get("provider_is_default"):
            summary["error_count"] += 1
            summary["error_plugins"].append(
                f"ticketing_provider:{ticketing_provider_status.get('selected_provider_id')}"
            )
            summary["warn_plugins"].append(
                f"ticketing_provider:{ticketing_provider_status.get('selected_provider_id')}"
            )
        else:
            summary["warn_count"] += 1
            summary["warn_plugins"].append(
                f"ticketing_provider:{ticketing_provider_status.get('selected_provider_id')}"
            )

    for plugin_id in sorted(required_missing_plugins):
        summary["required_missing_plugins"].append(plugin_id)
        summary["error_count"] += 1
        summary["warn_count"] += 1
        summary["warn_plugins"].append(plugin_id)
        summary["error_plugins"].append(plugin_id)
        _event_for(
            "admin",
            "plugin.startup_debug",
            f"Required core plugin missing from plugin registry: {plugin_id}",
            "error",
            {
                "plugin_id": plugin_id,
                "required": True,
                "runtime_called": False,
                "runtime_success": False,
                "presence": "missing",
                "core_reported": True,
            },
        )
        _log_startup_debug_message(
            "error",
            f"{plugin_id}: required core plugin missing from plugin registry",
            {"plugin_id": plugin_id, "required": True, "core_reported": True},
        )

    for normalized_plugin_id in sorted(plugin_map):
        plugin = plugin_map[normalized_plugin_id]
        summary["checked"] += 1
        summary["checked_plugins"].append(normalized_plugin_id)
        try:
            result = _run_startup_plugin_debug_check(normalized_plugin_id, plugin)
            if result["runtime_called"]:
                summary["runtime_called"] += 1
                if result["runtime_success"]:
                    summary["runtime_success"] += 1
                else:
                    summary["error_count"] += 1
                    summary["error_plugins"].append(result["plugin_id"])
                    summary["warn_plugins"].append(result["plugin_id"])
            elif result["has_debug_action"]:
                summary["warn_count"] += 1
                summary["warn_plugins"].append(result["plugin_id"])
            else:
                summary["missing_debug"] += 1
                summary["warn_count"] += 1
                summary["missing_debug_plugins"].append(result["plugin_id"])
                summary["warn_plugins"].append(result["plugin_id"])
        except Exception as exc:  # pragma: no cover
            summary["error_count"] += 1
            summary["warn_count"] += 1
            summary["warn_plugins"].append(normalized_plugin_id)
            if normalized_plugin_id not in summary["error_plugins"]:
                summary["error_plugins"].append(normalized_plugin_id)
            _event_for(
                "admin",
                "plugin.startup_debug",
                f"Startup plugin debug check failed before execution: {normalized_plugin_id}",
                "error",
                {
                    "plugin_id": normalized_plugin_id,
                    "action_id": "debug",
                    "runtime_called": False,
                    "runtime_success": False,
                    "error": str(exc),
                },
            )
            _log_startup_debug_message(
                "error",
                f"{normalized_plugin_id}: startup debug check failed before execution",
                {
                    "plugin_id": normalized_plugin_id,
                    "error": str(exc),
                },
            )

    summary_level = "info"
    if summary["error_count"]:
        summary_level = "error"
    elif summary["warn_count"]:
        summary_level = "warn"

    summary_suffix = ""
    if str(ticketing_provider_status.get("status") or "").strip().lower() != "ok":
        summary_suffix = (
            f"; ticketing provider={ticketing_provider_status.get('selected_provider_id')} "
            f"reason={ticketing_provider_status.get('reason_code')}"
        )

    _event_for(
        "admin",
        "plugin.startup_debug_summary",
        f"Startup plugin debug probe summary: {summary['runtime_success']}/{summary['runtime_called']} debug checks succeeded; {summary['missing_debug']} missing debug handlers; {summary['required_missing']} required core plugin(s) missing; {summary['required_disabled']} required core plugin(s) disabled{summary_suffix}",
        summary_level,
        summary,
    )
    _log_startup_debug_message(
        summary_level,
        f"Startup plugin debug summary complete: {summary['runtime_success']}/{summary['runtime_called']} passed",
        summary,
    )


def _parse_local_ui_entry_point(action: Dict[str, Any], action_id: str) -> tuple[str, str]:
    raw_entry_point = str(action.get("entry_point") or "").strip()
    if not raw_entry_point:
        return "actions", f"handle_{str(action_id).strip()}"

    if ":" in raw_entry_point:
        module_expr, function_expr = raw_entry_point.split(":", 1)
        module_expr = module_expr.strip()
        function_expr = function_expr.strip()
    else:
        separator = raw_entry_point.rfind(".")
        if separator < 0:
            return "actions", raw_entry_point
        module_expr = raw_entry_point[:separator].strip()
        function_expr = raw_entry_point[separator + 1 :].strip()

    if not module_expr:
        module_expr = "actions"
    if not function_expr:
        function_expr = f"handle_{str(action_id).strip()}"
    return module_expr, function_expr


def _resolve_plugin_ui_module_file(ui_path: Path, module_name: str) -> Optional[Path]:
    module_parts = [part for part in module_name.split(".") if part]
    if not module_parts:
        module_parts = ["actions"]

    module_base = ui_path.joinpath(*module_parts)
    module_file = module_base.with_suffix(".py")
    package_init = module_base / "__init__.py"
    if package_init.is_file():
        return package_init
    if module_file.is_file():
        return module_file
    return None


def _plugin_ui_handler_warnings_for_contract(plugin: Dict[str, Any]) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    source = plugin.get("source")
    if not isinstance(source, str):
        return warnings

    source_path = _resolve_plugin_source_path(source)
    if source_path is None:
        return warnings

    ui_path = source_path / "ui"
    if not ui_path.exists() or not ui_path.is_dir():
        return warnings

    metadata = plugin.get("metadata")
    if not isinstance(metadata, dict):
        return warnings
    ui = metadata.get("ui")
    if not isinstance(ui, dict):
        return warnings

    sections = ui.get("sections")
    if not isinstance(sections, list):
        return warnings

    for section in sections:
        if not isinstance(section, dict):
            continue
        for action in section.get("actions") or []:
            if not isinstance(action, dict):
                continue
            action_id = str(action.get("id") or "").strip()
            if not action_id:
                continue

            module_expr, function_expr = _parse_local_ui_entry_point(action, action_id)
            module_file = _resolve_plugin_ui_module_file(ui_path, module_expr)
            if module_file is None:
                warnings.append(
                    {
                        "code": "P_PLUGIN_UI_HANDLER_MISSING",
                        "severity": "warn",
                        "source": "ui",
                        "message": f"ui action '{action_id}' has no local entry point module '{module_expr}'",
                    },
                )
                continue

            try:
                module_id = f"busy38_ui_diagnostic_{id(module_file)}_{module_expr}"
                module_spec = importlib.util.spec_from_file_location(module_id, str(module_file))
                if module_spec is None or module_spec.loader is None:
                    raise RuntimeError("module spec unavailable")
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
            except Exception as exc:
                warnings.append(
                    {
                        "code": "P_PLUGIN_UI_HANDLER_LOAD_FAILED",
                        "severity": "warn",
                        "source": "ui",
                        "message": f"ui action '{action_id}' failed to load handler module '{module_expr}': {exc}",
                    },
                )
                continue

            handler = module
            for piece in function_expr.split("."):
                handler = getattr(handler, piece, None)
                if handler is None:
                    warnings.append(
                        {
                            "code": "P_PLUGIN_UI_HANDLER_MISSING",
                            "severity": "warn",
                            "source": "ui",
                            "message": f"ui action '{action_id}' has missing handler '{function_expr}' in module '{module_expr}'",
                        },
                    )
                    break
            else:
                if not callable(handler):
                    warnings.append(
                        {
                            "code": "P_PLUGIN_UI_HANDLER_NOT_CALLABLE",
                            "severity": "warn",
                            "source": "ui",
                            "message": f"ui action '{action_id}' handler '{function_expr}' in module '{module_expr}' is not callable",
                        },
                    )

    return warnings


def _sanitize_plugin_for_debug(plugin: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plugin, dict):
        return {}
    payload = dict(plugin)
    payload["metadata"] = _redact_metadata(payload.get("metadata"), "viewer")
    return payload


def _redact_sensitive_payload(payload: Any, role: str) -> Any:
    if role == "admin":
        return payload
    if isinstance(payload, dict):
        return {
            k: "***redacted***" if _is_sensitive_key(k) else _redact_sensitive_payload(v, role)
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [_redact_sensitive_payload(v, role) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_redact_sensitive_payload(v, role) for v in payload)
    return payload


def _dependency_warning(
    dependency: str,
    plugin_catalog: Dict[str, Dict[str, Any]],
    *,
    owner_required: bool,
) -> Optional[Dict[str, str]]:
    normalized_dependency = _normalize_core_plugin_dependency_id(str(dependency).strip())
    if not normalized_dependency:
        return None

    reference = _find_core_plugin_reference(normalized_dependency)
    aliases = reference.get("aliases") if reference else [normalized_dependency]
    if not aliases:
        aliases = [normalized_dependency]

    dependency_lookup = {_core_alias_key(plugin_id): plugin_id for plugin_id in plugin_catalog.keys()}
    for alias in aliases:
        matched = dependency_lookup.get(_core_alias_key(alias))
        if matched is None:
            continue
        dependency_plugin = plugin_catalog[matched]
        if not dependency_plugin.get("enabled", False):
            return {
                "code": "P_PLUGIN_DEPENDENCY_DISABLED",
                "severity": "warn",
                "source": "runtime",
                "message": f"plugin dependency '{normalized_dependency}' is registered but disabled",
            }
        return None

    severity = "error" if owner_required else "warn"
    return {
        "code": "P_PLUGIN_DEPENDENCY_MISSING",
        "severity": severity,
        "source": "runtime",
        "message": f"plugin dependency '{normalized_dependency}' is missing from plugin registry",
    }


def _plugin_map() -> Dict[str, Dict[str, Any]]:
    return {
        str(plugin.get("id")).strip(): _enrich_plugin_record(plugin)
        for plugin in storage.list_plugins()
    }


def _sanitize_tool_payload(
    tool: Dict[str, Any],
    role: str,
    plugin_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    payload = dict(tool)
    payload["metadata"] = _redact_metadata(payload.get("metadata"), role)
    plugin_id = str(payload.get("plugin_id") or "").strip()
    if plugin_id:
        plugin = plugin_map.get(plugin_id, {})
        if plugin:
            payload["plugin"] = {
                "id": plugin_id,
                "name": plugin.get("name"),
                "source": plugin.get("source"),
            }
        else:
            payload["plugin"] = {"id": plugin_id}
    else:
        payload["plugin"] = None
    return payload


def _sanitize_tool_usage_payload(usage: Dict[str, Any], role: str) -> Dict[str, Any]:
    if role == "admin":
        return dict(usage)

    payload = dict(usage)
    payload["details"] = _redact_metadata(payload.get("details"), role)
    payload["payload"] = _redact_metadata(payload.get("payload"), role)
    if isinstance(payload.get("agent_id"), str):
        payload["agent_id"] = "***redacted***"
    if isinstance(payload.get("session_id"), str):
        payload["session_id"] = "***redacted***"
    if isinstance(payload.get("request_id"), str):
        payload["request_id"] = "***redacted***"
    return payload


def _coerce_overlay_token_cap(token_cap: Optional[int]) -> Optional[int]:
    if token_cap is None:
        return None
    if token_cap <= 0:
        raise HTTPException(status_code=400, detail="overlay_token_cap must be a positive integer")
    return int(token_cap)


def _sanitize_agent_overlay(overlay_result: Any, role: str) -> Dict[str, Any]:
    if not isinstance(overlay_result, dict):
        return {"found": False, "overlay": None}

    if not overlay_result.get("success", True):
        payload = {"found": False, "overlay": None}
        if isinstance(overlay_result.get("error"), str):
            payload["error"] = overlay_result.get("error")
        return payload

    raw = overlay_result.get("overlay")
    if raw is None and overlay_result.get("overlay_data") is not None:
        raw = overlay_result.get("overlay_data")

    if raw is None:
        return {"found": bool(overlay_result.get("found", False)), "overlay": None}

    if not isinstance(raw, dict):
        return {"found": False, "overlay": None}

    overlay_payload = dict(raw)
    overlay_payload.pop("source_hash", None)
    if role != "admin":
        content = str(overlay_payload.get("content", ""))
        if content:
            overlay_payload["content_preview"] = content[:240] + ("…" if len(content) > 240 else "")
        overlay_payload.pop("content", None)

    return {
        "found": True,
        "overlay": overlay_payload,
        "actor_id": str(overlay_result.get("actor_id", "")),
    }


def _sanitize_overlay_history_record(overlay_record: Any, role: str) -> Dict[str, Any]:
    if not isinstance(overlay_record, dict):
        return {}
    payload = dict(overlay_record)
    payload.pop("source_hash", None)

    if role != "admin":
        content = str(payload.get("content", ""))
        if content:
            payload["content_preview"] = content[:240] + ("…" if len(content) > 240 else "")
        payload.pop("content", None)

    return payload


def _agent_with_overlay(agent: Dict[str, Any], role: str) -> Dict[str, Any]:
    payload = dict(agent)
    try:
        payload["overlay"] = _sanitize_agent_overlay(runtime.get_actor_overlay(str(agent.get("id", "")).strip()), role)
    except Exception as exc:  # pragma: no cover - defensive
        payload["overlay"] = {"found": False, "overlay": None, "error": str(exc)}
    return payload


def _ensure_http_request_body(url: str) -> urllib.request.Request:
    if not url:
        raise HTTPException(status_code=400, detail="provider endpoint is missing")
    return urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "busy38-management-ui/0.2.0"}, method="GET")


def _safe_http_fetch(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 4) -> Any:
    request = _ensure_http_request_body(url)
    if headers:
        for key, value in headers.items():
            if value is not None:
                request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"model discovery request to {url} failed with status {exc.code}",
        )
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"model discovery request to {url} failed: {exc.reason}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"model discovery request to {url} failed: {exc}",
        )

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"invalid JSON returned from {url}: {exc.msg}",
        )


def _provider_catalog_key(provider: Dict[str, Any]) -> str:
    metadata = provider.get("metadata")
    if isinstance(metadata, dict):
        kind = str(metadata.get("kind") or "").strip().lower()
        if kind:
            return kind
    return str(provider.get("kind") or provider.get("id") or provider.get("name") or "").strip().lower()


def _apply_catalog_filter(
    provider: Dict[str, Any],
    models: list[str],
) -> tuple[list[str], bool, Optional[str]]:
    if not models:
        return [], False, None

    raw_models = [str(model).strip() for model in models if str(model).strip()]
    if not raw_models:
        return [], False, None

    if _filter_models_from_catalog is None:
        return raw_models, False, None

    try:
        return _filter_models_from_catalog(
            provider=_provider_catalog_key(provider),
            models=raw_models,
            base_url=provider.get("endpoint"),
        )
    except Exception:
        return raw_models, False, None


def _extract_models_from_payload(payload: Any) -> List[str]:
    models: List[str] = []
    if isinstance(payload, dict):
        candidates: List[Any] = []
        for key in ("data", "models", "result", "response"):
            if key in payload and isinstance(payload[key], list):
                candidates.extend(payload[key])
        if not candidates and isinstance(payload.get("model"), str):
            candidates.append(payload["model"])
        if not candidates:
            for item in payload.values():
                if isinstance(item, list):
                    candidates.extend(item)
        for entry in candidates:
            if isinstance(entry, str) and entry.strip():
                models.append(entry.strip())
            elif isinstance(entry, dict):
                model = entry.get("id") or entry.get("name") or entry.get("model")
                if isinstance(model, str) and model.strip():
                    models.append(model.strip())
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, str) and entry.strip():
                models.append(entry.strip())
            elif isinstance(entry, dict):
                model = entry.get("id") or entry.get("name") or entry.get("model")
                if isinstance(model, str) and model.strip():
                    models.append(model.strip())
    seen = set()
    unique_models = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        unique_models.append(model)
    return unique_models


def _provider_model_endpoints(provider: Dict[str, Any], endpoint_override: Optional[str] = None) -> List[Dict[str, str]]:
    endpoint = (endpoint_override or provider.get("endpoint") or "").rstrip("/")
    if not endpoint:
        raise HTTPException(status_code=400, detail="provider endpoint is missing")

    normalized_endpoint = endpoint
    if normalized_endpoint.endswith("/v1"):
        normalized_endpoint = normalized_endpoint[:-3]
    for suffix in ("/v1/models", "/models", "/api/tags"):
        if normalized_endpoint.endswith(suffix):
            normalized_endpoint = normalized_endpoint[: -len(suffix)]
            break

    provider_identity = f"{provider.get('id', '').lower()} {provider.get('name', '').lower()}"
    candidates: List[str] = []
    if "ollama" in provider_identity or "llama" in provider_identity:
        candidates.append(f"{normalized_endpoint}/api/tags")
        candidates.append(f"{normalized_endpoint}/v1/models")
    else:
        candidates.append(f"{normalized_endpoint}/v1/models")
        candidates.append(f"{normalized_endpoint}/models")
        candidates.append(f"{normalized_endpoint}/api/tags")
    seen = set()
    deduped: List[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return [{"url": candidate} for candidate in deduped]


def _discover_provider_models(
    provider: Dict[str, Any],
    api_key: Optional[str] = None,
    endpoint_override: Optional[str] = None,
) -> Dict[str, Any]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    attempts: List[Dict[str, Any]] = []
    catalog_filter_error: Optional[str] = None

    for candidate in _provider_model_endpoints(provider, endpoint_override):
        try:
            payload = _safe_http_fetch(candidate["url"], headers=headers)
            models = _extract_models_from_payload(payload)
            if models:
                filtered, _was_filtered, catalog_error = _apply_catalog_filter(provider, models)
                if catalog_error:
                    catalog_filter_error = catalog_error
                if not filtered:
                    attempts.append({
                        "url": candidate["url"],
                        "result": "catalog-filtered-empty",
                        "error": catalog_error or "No discovered model passed catalog validation.",
                    })
                    continue
                return {
                    "models": filtered,
                    "endpoint": candidate["url"],
                    "source": _maybe_strip_api_path(candidate["url"]),
                    "attempts": attempts,
                }
            attempts.append({"url": candidate["url"], "result": "empty-model-list"})
        except HTTPException as exc:
            attempts.append({"url": candidate["url"], "error": exc.detail})
        except Exception as exc:
            attempts.append({"url": candidate["url"], "error": str(exc)})

    terminal_error = "No model list endpoint matched this provider."
    if catalog_filter_error:
        terminal_error = catalog_filter_error

    return {
        "models": [],
        "endpoint": endpoint_override or provider.get("endpoint") or "",
        "source": "unresolved",
        "attempts": attempts,
        "error": terminal_error,
    }


def _record_provider_test_metadata(
    metadata: Dict[str, Any],
    source: str,
    endpoint: str,
    latency_ms: float,
    models: List[str],
    attempts: List[Dict[str, Any]],
    error: Optional[str],
) -> Dict[str, Any]:
    payload = dict(metadata or {})
    history = payload.get("test_history") if isinstance(payload.get("test_history"), list) else []
    if not isinstance(history, list):
        history = []

    tested_at = _now_iso()
    test_record = {
        "tested_at": tested_at,
        "status": "pass" if models else "fail",
        "source": source,
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "models_count": len(models),
        "attempts": attempts,
    }
    if error:
        test_record["error"] = error

    history.append(test_record)
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    cleaned_history: List[Dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        tested = _coerce_iso_datetime(item.get("tested_at"))
        if tested and tested >= cutoff:
            cleaned_history.append(item)
    payload["test_history"] = cleaned_history[-200:]
    payload["last_test"] = test_record

    metrics_window_5m = datetime.now(timezone.utc) - timedelta(minutes=5)
    metrics_window_1m = datetime.now(timezone.utc) - timedelta(minutes=1)
    in_5m = [
        item for item in payload["test_history"]
        if isinstance(item, dict) and _coerce_iso_datetime(item.get("tested_at")) and _coerce_iso_datetime(item.get("tested_at")) >= metrics_window_5m
    ]
    in_1m = [
        item for item in payload["test_history"]
        if isinstance(item, dict) and _coerce_iso_datetime(item.get("tested_at")) and _coerce_iso_datetime(item.get("tested_at")) >= metrics_window_1m
    ]
    latencies = [
        item.get("latency_ms")
        for item in in_5m
        if isinstance(item, dict) and isinstance(item.get("latency_ms"), (int, float))
    ]
    fail_count_1m = sum(
        1 for item in in_1m
        if isinstance(item, dict) and item.get("status") == "fail"
    )
    success_count = sum(
        1 for item in in_5m
        if isinstance(item, dict) and item.get("status") == "pass"
    )
    payload["health_metrics"] = {
        "latency_ms_last": test_record.get("latency_ms"),
        "latency_ms_avg_5m": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "latency_ms_p95_5m": None,
        "success_rate_5m": round((success_count / len(in_5m)) * 100, 2) if in_5m else None,
        "failure_count_last_1m": fail_count_1m,
        "last_checked_at": tested_at,
        "last_error_message": test_record.get("error"),
        "last_error_code": None,
    }
    if latencies:
        sorted_latencies = sorted(latencies)
        p95_index = max(0, int(round(0.95 * (len(sorted_latencies) - 1))))
        payload["health_metrics"]["latency_ms_p95_5m"] = sorted_latencies[p95_index]
    return payload


def _build_model_discovery_metadata(
    discovery: Dict[str, Any],
    status: str,
) -> Dict[str, Any]:
    return {
        "discovered_at": _now_iso(),
        "status": status,
        "source": discovery.get("source", "manual"),
        "endpoint": discovery.get("endpoint"),
        "attempts": discovery.get("attempts", []),
        "count": len(discovery.get("models") or []),
        "error": discovery.get("error"),
    }


def _coerce_secret_policy(provider: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    policy = metadata.get("secret_policy")
    if policy in {"required", "optional", "none"}:
        return policy

    provider_identity = (
        f"{provider.get('id', '').lower()} "
        f"{provider.get('name', '').lower()} "
        f"{provider.get('endpoint', '').lower()}"
    )
    if "ollama" in provider_identity or "llama" in provider_identity:
        return "none"
    return "required"


def _normalize_secret_metadata(
    metadata: Dict[str, Any],
    provider: Optional[Dict[str, Any]] = None,
    actor: str = "admin",
    has_secret: Optional[bool] = None,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    payload = dict(metadata or {})
    if provider is not None:
        payload["secret_policy"] = _coerce_secret_policy(provider, payload)
    if has_secret is True:
        payload["secret_present"] = True
        payload["secret_last_rotated_at"] = _now_iso()
        payload["secret_touched_by"] = actor
        payload["secret_touched_at"] = _now_iso()
        if action:
            payload["secret_last_action"] = action
    elif has_secret is False:
        payload["secret_present"] = False
        payload["secret_last_rotated_at"] = None
        payload["secret_touched_by"] = actor
        payload["secret_touched_at"] = _now_iso()
        if action:
            payload["secret_last_action"] = action
    return payload


def _maybe_strip_api_path(url: str) -> str:
    suffixes = ["/v1/models", "/api/tags", "/models"]
    for suffix in suffixes:
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def _provider_secret_status(provider: Dict[str, Any]) -> str:
    metadata = provider.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    secret_policy = _coerce_secret_policy(provider, metadata)
    secret_present = bool(metadata.get("secret_present"))
    if secret_policy == "required":
        return "required_present" if secret_present else "required_missing"
    if secret_policy == "none":
        return "none"
    if secret_present:
        return "optional_present"
    return "optional_missing"


def _assert_catalog_models_allowed(
    *,
    provider: Dict[str, Any],
    primary_model: Optional[str],
    fallback_models: Optional[List[str]] = None,
) -> None:
    candidates = []
    if primary_model:
        candidates.append(primary_model)
    if fallback_models:
        candidates.extend(fallback_models)

    for model in candidates:
        normalized = str(model).strip()
        if not normalized:
            continue
        filtered, _, catalog_error = _apply_catalog_filter(provider, [normalized])
        if not filtered:
            raise HTTPException(
                status_code=400,
                detail=catalog_error or f"model '{normalized}' is not permitted by provider catalog.",
            )


def _provider_sort_key(provider: Dict[str, Any], sort_by: str):
    metadata = provider.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    health = metadata.get("health_metrics") or {}
    if not isinstance(health, dict):
        health = {}

    if sort_by == "last_checked":
        checked = health.get("last_checked_at")
        if checked is None:
            return 0
        checked_at = _coerce_iso_datetime(checked)
        if checked_at is None:
            return 0
        return checked_at.timestamp()
    if sort_by == "latency_5m":
        value = health.get("latency_ms_avg_5m")
        if isinstance(value, (int, float)):
            return value
        return float("inf")
    if sort_by == "failures":
        value = health.get("failure_count_last_1m")
        if isinstance(value, (int, float)):
            return value
        return 0
    return int(provider.get("priority", 0))


def _maybe_emit_provider_state_event(
    role: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    provider_id: str,
) -> None:
    before_enabled = bool(before.get("enabled"))
    after_enabled = bool(after.get("enabled"))
    before_status = (before.get("status") or "").lower()
    after_status = (after.get("status") or "").lower()
    if before_enabled != after_enabled or before_status != after_status:
        _event_for(
            role,
            "provider.state_changed",
            f"Provider state changed: {provider_id}",
            "info",
            {"provider_id": provider_id, "enabled": after_enabled, "status": after_status},
        )


def _provider_routing_chain() -> Dict[str, Any]:
    providers = storage.list_providers()
    enabled = [p for p in providers if bool(p.get("enabled"))]
    if not enabled:
        return {
            "selection_strategy": "enabled providers sorted by ascending priority",
            "selection_rationale": "No enabled providers are currently available for routing.",
            "active_provider_id": None,
            "enabled_count": 0,
            "total_count": len(providers),
            "routing_intent": "no-providers",
            "routing_path": "no providers available",
            "chain": [],
        }

    sorted_enabled = sorted(
        enabled,
        key=lambda item: (
            item.get("status") != "active",
            int(item.get("priority", 0)),
        ),
    )
    chain: List[Dict[str, Any]] = []
    for index, provider in enumerate(sorted_enabled):
        provider_status = str(provider.get("status") or "configured").lower()
        metadata = provider.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        last_test = metadata.get("last_test") if isinstance(metadata.get("last_test"), dict) else {}
        health_metrics = metadata.get("health_metrics") if isinstance(metadata.get("health_metrics"), dict) else {}

        if index == 0:
            routing_reason = "Primary route candidate from enabled provider list."
            routing_behavior = "Used first for normal routing."
        else:
            routing_reason = "Fallback route candidate."
            routing_behavior = f"Used if all {index} higher-priority provider(s) fail or are unavailable."

        if provider_status != "active":
            routing_reason = f"{routing_reason} Provider status is currently {provider_status}."
            if provider_status == "unreachable":
                routing_behavior = "Avoided until provider recovers."

        chain.append({
            "id": provider["id"],
            "name": provider["name"],
            "status": provider_status,
            "enabled": bool(provider.get("enabled")),
            "priority": provider.get("priority", 0),
            "model": provider.get("model"),
            "position": index,
            "active": index == 0,
            "routing_intent": "primary" if index == 0 else "fallback",
            "fallback_position": 0 if index == 0 else index,
            "routing_reason": routing_reason,
            "routing_behavior": routing_behavior,
            "health": {
                "last_test_status": last_test.get("status") if isinstance(last_test, dict) else None,
                "last_tested_at": last_test.get("tested_at") if isinstance(last_test, dict) else None,
                "latency_ms_last": health_metrics.get("latency_ms_last") if isinstance(health_metrics, dict) else None,
                "last_error": health_metrics.get("last_error_message") if isinstance(health_metrics, dict) else None,
            },
            "selection_strategy": "enabled providers sorted by priority, then active status",
        })
    strategy_summary = "Enabled providers sorted by priority, with active state-first ordering."
    if all(item.get("status") == "standby" for item in sorted_enabled):
        strategy_summary = "All enabled providers are in standby mode; fallback ordering still preserved."
    return {
        "selection_strategy": strategy_summary,
        "selection_rationale": (
            "Providers are ordered deterministically by active state and configured priority. "
            "The first active entry is the primary route and each subsequent entry is fallback."
        ),
        "active_provider_id": sorted_enabled[0]["id"],
        "enabled_count": len(sorted_enabled),
        "total_count": len(providers),
        "routing_intent": "primary then fallback",
        "routing_path": " -> ".join(
            [
                f"{item['id']} ({'primary' if item['active'] else 'fallback'})"
                for item in chain
            ]
        ),
        "chain": chain,
    }


def _run_provider_test(
    provider: Dict[str, Any],
    request: ProviderModelTest,
    role: str,
) -> Dict[str, Any]:
    provider_id = provider.get("id")
    metadata = provider.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    api_key = request.api_key if request.api_key else metadata.get("api_key")
    secret_policy = _coerce_secret_policy(provider, metadata)
    if secret_policy == "required" and not api_key:
        status_payload = _record_provider_test_metadata(
            metadata=dict(metadata),
            source="manual",
            endpoint=provider.get("endpoint") or "",
            latency_ms=0.0,
            models=[],
            attempts=[],
            error="provider secret is required for test",
        )
        provider_payload = storage.update_provider(
            provider_id=provider_id,
            values={"metadata": status_payload},
        )
        _event_for(
            role,
            "provider.tested",
            f"Provider test failed: {provider_id} (missing secret)",
            "error",
        )
        return {
            "provider_id": provider_id,
            "status": status_payload["last_test"]["status"],
            "tested_at": status_payload["last_test"]["tested_at"],
            "latency_ms": status_payload["last_test"]["latency_ms"],
            "source": status_payload["last_test"]["source"],
            "models_count": status_payload["last_test"]["models_count"],
            "error": status_payload["last_test"].get("error"),
            "provider": _sanitize_provider(provider_payload, role),
        }

    started = time.perf_counter()
    try:
        discovery = _discover_provider_models(
            provider=provider,
            api_key=api_key,
            endpoint_override=request.endpoint,
        )
    except HTTPException as exc:
        discovery = {
            "models": [],
            "endpoint": request.endpoint or provider.get("endpoint") or "",
            "source": "unresolved",
            "attempts": [],
            "error": exc.detail,
        }
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

    status_payload = _record_provider_test_metadata(
        metadata=dict(metadata),
        source=discovery.get("source", "manual"),
        endpoint=discovery.get("endpoint") or provider.get("endpoint") or "",
        latency_ms=latency_ms,
        models=discovery.get("models", []),
        attempts=discovery.get("attempts", []),
        error=discovery.get("error"),
    )
    provider_payload = storage.update_provider(
        provider_id=provider.get("id"),
        values={"metadata": status_payload},
    )
    last_test = status_payload.get("last_test", {})
    level = "error" if last_test.get("status") != "pass" else "info"
    _event_for(
        role,
        "provider.tested",
        f"Provider tested: {provider_id} ({last_test.get('status')})",
        level,
    )

    return {
        "provider_id": provider_id,
        "status": last_test.get("status"),
        "tested_at": last_test.get("tested_at"),
        "latency_ms": last_test.get("latency_ms"),
        "source": last_test.get("source"),
        "models_count": last_test.get("models_count", 0),
        "error": last_test.get("error"),
        "provider": _sanitize_provider(provider_payload, role),
    }


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    runtime_status = runtime.get_status()
    return {
        "status": "ok",
        "service": "busy38-management-ui",
        "runtime_connected": bool(runtime_status.get("connected")),
        "updated_at": _now_iso(),
    }


@app.get("/api/settings")
async def get_settings(
    request: Request,
    token: Optional[str] = Query(default=None, alias="token"),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer", token=token)
    settings = storage.get_settings()
    settings["auto_restart"] = bool(settings["auto_restart"])
    auth_header = request.headers.get("Authorization") if request else None
    return {
        "settings": settings,
        "updated_at": settings["updated_at"],
        "role": role,
        "role_source": _token_source(auth_header, token),
    }


@app.patch("/api/settings")
async def update_settings(request: Request, update: SettingsUpdate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No setting fields provided")

    settings = storage.set_settings(payload)
    _event_for(role, "settings", "Settings updated via management UI", "info")
    return {"settings": settings, "updated_at": settings["updated_at"]}


@app.get("/api/appearance")
async def get_appearance(
    request: Request,
    token: Optional[str] = Query(default=None, alias="token"),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer", token=token)
    auth_header = request.headers.get("Authorization") if request else None
    try:
        appearance = load_appearance_preferences()
    except AppearancePreferencesError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    return {
        "appearance_preferences": appearance,
        "updated_at": appearance["updated_at"],
        "role": role,
        "role_source": _token_source(auth_header, token),
    }


@app.patch("/api/appearance")
async def update_appearance(request: Request, update: AppearanceUpdate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No appearance fields provided")
    try:
        current = load_appearance_preferences()
        updated = apply_appearance_update(current, payload)
        write_appearance_preferences(updated)
    except AppearancePreferencesError as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    _event_for(role, "appearance", "Appearance preferences updated via management UI", "info")
    return {
        "appearance_preferences": updated,
        "updated_at": updated["updated_at"],
    }


@app.get("/api/providers")
async def get_providers(
    request: Request,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    secret_status: Optional[str] = None,
    sort_by: str = "priority",
    sort_desc: bool = False,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    providers = storage.list_providers()
    if kind:
        kind_query = kind.strip().lower()
        providers = [
            provider
            for provider in providers
            if str((provider.get("metadata") or {}).get("kind", "")).strip().lower() == kind_query
        ]
    if status:
        status_query = status.strip().lower()
        providers = [provider for provider in providers if str(provider.get("status", "")).lower() == status_query]
    if secret_status:
        secret_query = secret_status.strip().lower().replace("-", "_")
        providers = [
            provider
            for provider in providers
            if _provider_secret_status(provider) == secret_query
            or (secret_query == "present" and _provider_secret_status(provider).endswith("_present"))
            or (secret_query == "missing" and _provider_secret_status(provider).endswith("_missing"))
            or (secret_query == "required" and _provider_secret_status(provider).startswith("required"))
            or (secret_query == "optional" and _provider_secret_status(provider).startswith("optional"))
            or (secret_query == "none" and _provider_secret_status(provider) == "none")
        ]
    if sort_by not in {"priority", "last_checked", "latency_5m", "failures"}:
        sort_by = "priority"
    providers = sorted(providers, key=lambda provider: _provider_sort_key(provider, sort_by), reverse=sort_desc)
    providers = [_sanitize_provider(p, role) for p in providers]
    return {"providers": providers, "updated_at": _now_iso()}


@app.post("/api/providers")
async def create_provider(request: Request, provider: ProviderCreate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = provider.model_dump()
    payload["id"] = str(payload["id"]).strip()
    payload["name"] = str(payload["name"]).strip()
    payload["endpoint"] = str(payload["endpoint"]).strip()
    payload["model"] = (
        str(payload["model"]).strip() if payload.get("model") is not None else ""
    )
    if not payload["id"]:
        raise HTTPException(status_code=400, detail="provider id is required")
    if not payload["name"]:
        raise HTTPException(status_code=400, detail="provider name is required")
    if not payload["endpoint"]:
        raise HTTPException(status_code=400, detail="provider endpoint is required")
    provider_payload = {
        "id": payload["id"],
        "name": payload["name"],
        "endpoint": payload["endpoint"],
        "model": payload["model"],
        "priority": int(payload["priority"]),
        "enabled": bool(payload["enabled"]),
        "status": payload.get("status") or "configured",
        "metadata": _normalize_provider_metadata_fields(
            payload=payload,
            metadata=payload.get("metadata") or {},
        ),
    }
    provider_payload["metadata"] = _normalize_secret_metadata(
        metadata=dict(provider_payload["metadata"]),
        provider=provider_payload,
        actor=role,
        has_secret=bool(provider_payload["metadata"].get("api_key")),
    )

    if not provider_payload["model"]:
        discovery = _discover_provider_models(provider=provider_payload, api_key=None)
        provider_payload["metadata"]["model_discovery"] = _build_model_discovery_metadata(
            discovery=discovery,
            status="complete" if discovery["models"] else "manual",
        )
        if discovery["models"]:
            provider_payload["model"] = discovery["models"][0]
            provider_payload["metadata"]["discovered_models"] = discovery["models"]
        else:
            raise HTTPException(
                status_code=400,
                detail="provider model is required unless discovery returns at least one model; run discovery first or provide model manually",
            )

    _assert_catalog_models_allowed(
        provider=provider_payload,
        primary_model=provider_payload.get("model"),
        fallback_models=provider_payload.get("metadata", {}).get("fallback_models"),
    )

    if not bool(provider_payload["enabled"]):
        provider_payload["status"] = "standby"
    try:
        provider_row = storage.create_provider(provider_payload)
    except ValueError as exc:
        raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc))

    _event_for(
        role,
        "provider.created",
        f"Provider created: {provider_payload['id']}",
        "info",
        {"provider_id": provider_payload["id"]},
    )
    return {"provider": _sanitize_provider(provider_row, role), "updated_at": _now_iso()}


@app.get("/api/providers/routing-chain")
async def get_provider_chain(request: Request) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    payload = _provider_routing_chain()
    chain = payload["chain"] if isinstance(payload, dict) else []
    payload_without_chain = {
        "selection_strategy": payload.get("selection_strategy", "enabled providers sorted by priority"),
        "selection_rationale": payload.get(
            "selection_rationale",
            "Enabled providers are ordered deterministically for fallback planning.",
        ),
        "active_provider_id": payload.get("active_provider_id"),
        "enabled_count": payload.get("enabled_count", len(chain)),
        "total_count": payload.get("total_count", len(chain)),
        "routing_intent": payload.get("routing_intent", "primary then fallback"),
        "routing_path": payload.get("routing_path"),
    }
    return {"chain": chain, **payload_without_chain, "updated_at": _now_iso()}


@app.get("/api/providers/{provider_id}/history")
async def get_provider_history(
    provider_id: str,
    request: Request,
    limit: int = Query(default=25, ge=1, le=500),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    metadata = provider.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    history = metadata.get("test_history") or []
    if not isinstance(history, list):
        history = []
    return {
        "provider_id": provider_id,
        "test_history": list(reversed(history[-limit:])),
        "health_metrics": metadata.get("health_metrics") or {},
        "updated_at": _now_iso(),
    }


@app.get("/api/providers/{provider_id}/metrics")
async def get_provider_metrics(
    provider_id: str,
    request: Request,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    metadata = provider.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "provider_id": provider_id,
        "metrics": metadata.get("health_metrics") or {},
        "last_test": metadata.get("last_test") or {},
        "updated_at": _now_iso(),
    }


@app.patch("/api/providers/{provider_id}")
async def patch_provider(
    provider_id: str,
    update: ProviderUpdate,
    request: Request,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No provider fields provided")

    before_provider = storage.get_provider(provider_id)
    if not before_provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")
    provider_metadata = before_provider.get("metadata")
    if not isinstance(provider_metadata, dict):
        provider_metadata = {}
    payload["metadata"] = _normalize_provider_metadata_fields(
        payload=payload,
        metadata=provider_metadata,
    )

    merged_provider = dict(before_provider)
    merged_provider.update(payload)
    merged_metadata = dict(before_provider.get("metadata") or {})
    merged_metadata.update(payload.get("metadata") or {})
    merged_provider["metadata"] = merged_metadata
    _assert_catalog_models_allowed(
        provider=merged_provider,
        primary_model=merged_provider.get("model") if "model" in payload else before_provider.get("model"),
        fallback_models=merged_provider.get("metadata", {}).get("fallback_models"),
    )

    try:
        provider = storage.update_provider(provider_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    _maybe_emit_provider_state_event(
        role=role,
        before=before_provider,
        after=provider,
        provider_id=provider_id,
    )
    _event_for(
        role,
        "provider.updated",
        f"Provider updated: {provider_id}",
        "info",
        {"provider_id": provider_id},
    )
    provider["enabled"] = bool(provider["enabled"])
    return {"provider": _sanitize_provider(provider, role), "updated_at": _now_iso()}


@app.post("/api/providers/{provider_id}/discover-models")
async def discover_provider_models(
    provider_id: str,
    request: ProviderModelDiscovery,
    request_context: Request,
) -> Dict[str, Any]:
    role = _require_role(request_context, required="admin")
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    discovery = _discover_provider_models(
        provider=provider,
        api_key=request.api_key,
        endpoint_override=request.endpoint,
    )

    metadata = provider.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    metadata.pop("discovered_models", None)
    metadata["model_discovery"] = _build_model_discovery_metadata(
        discovery=discovery,
        status="complete" if discovery["models"] else "manual",
    )
    if request.endpoint:
        metadata["model_discovery"]["requested_endpoint"] = request.endpoint
    if discovery["models"]:
        metadata["discovered_models"] = discovery["models"]
        provider_payload = storage.update_provider(
            provider_id=provider_id,
            values={"metadata": metadata},
        )
        return {
            "provider": _sanitize_provider(provider_payload, role),
            "discovered_models": discovery["models"],
            "count": len(discovery["models"]),
            "source": discovery["source"],
            "updated_at": _now_iso(),
        }

    discovery["message"] = (
        "Automatic model discovery failed. Use the model field below to set values manually."
    )
    provider_payload = storage.update_provider(
        provider_id=provider_id,
        values={"metadata": metadata},
    )
    return {
        "provider": _sanitize_provider(provider_payload, role),
        "discovered_models": [],
        "count": 0,
        "source": "manual",
        "error": discovery.get("error"),
        "message": discovery.get("message"),
        "updated_at": _now_iso(),
    }


@app.post("/api/providers/{provider_id}/test")
async def test_provider_models(
    provider_id: str,
    request: ProviderModelTest,
    request_context: Request,
) -> Dict[str, Any]:
    role = _require_role(request_context, required="admin")
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    result = _run_provider_test(provider, request=request, role=role)

    return {
        "provider": result["provider"],
        "tested_at": result["tested_at"],
        "latency_ms": result["latency_ms"],
        "status": result["status"],
        "source": result["source"],
        "models_count": result["models_count"],
        "error": result.get("error"),
        "updated_at": _now_iso(),
    }


@app.post("/api/providers/test-all")
async def test_providers_all(
    request: ProviderTestAllRequest,
    request_context: Request,
) -> Dict[str, Any]:
    role = _require_role(request_context, required="admin")
    providers = storage.list_providers()
    candidates = [p for p in providers if request.include_disabled or bool(p.get("enabled"))]
    if request.provider_ids:
        candidate_ids = set(request.provider_ids)
        candidates = [p for p in candidates if p.get("id") in candidate_ids]
    candidates.sort(key=lambda item: (int(item.get("priority", 0)), item.get("id")))

    results = []
    pass_count = 0
    fail_count = 0
    for provider in candidates:
        provider_test_request = ProviderModelTest(api_key=None, endpoint=None)
        result = _run_provider_test(provider, request=provider_test_request, role=role)
        results.append(result)
        if result["status"] == "pass":
            pass_count += 1
        else:
            fail_count += 1

    return {
        "checked": len(results),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "results": results,
        "updated_at": _now_iso(),
    }


@app.post("/api/providers/{provider_id}/secret")
async def update_provider_secret(
    provider_id: str,
    request: ProviderSecretAction,
    request_context: Request,
) -> Dict[str, Any]:
    role = _require_role(request_context, required="admin")
    provider = storage.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"provider '{provider_id}' not found")

    metadata = provider.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    secret_policy = _coerce_secret_policy(provider, metadata)
    if secret_policy == "none" and request.action in {"set", "rotate"}:
        raise HTTPException(
            status_code=409,
            detail="secret updates are not required for this provider",
        )

    payload = dict(metadata)
    if request.action in {"set", "rotate"}:
        if not request.api_key:
            raise HTTPException(status_code=400, detail="api_key is required")
        payload["api_key"] = request.api_key
        payload = _normalize_secret_metadata(
            metadata=payload,
            provider=provider,
            actor=role,
            has_secret=True,
            action=request.action,
        )
    else:
        payload.pop("api_key", None)
        payload = _normalize_secret_metadata(
            metadata=payload,
            provider=provider,
            actor=role,
            has_secret=False,
            action=request.action,
        )

    provider_payload = storage.update_provider(
        provider_id=provider_id,
        values={"metadata": payload},
    )
    _event_for(
        role,
        "provider.secret",
        f"Provider secret {request.action}: {provider_id}",
        "info",
        {"provider_id": provider_id, "action": request.action},
    )
    secret_event = "provider.secret.cleared" if request.action == "clear" else "provider.secret.updated"
    _event_for(
        role,
        secret_event,
        f"Provider secret {request.action}: {provider_id}",
        "info",
        {"provider_id": provider_id, "action": request.action},
    )

    return {"provider": _sanitize_provider(provider_payload, role), "updated_at": _now_iso()}


@app.get("/api/plugins")
async def list_plugins(request: Request) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    plugins = [_sanitize_plugin(_enrich_plugin_record(plugin), role) for plugin in storage.list_plugins()]
    return {"plugins": plugins, "updated_at": _now_iso()}


@app.get("/api/plugins/core")
async def list_core_plugins(request: Request) -> Dict[str, Any]:
    _require_role(request, required="viewer")
    plugin_map = _plugin_map()
    core_presence = _collect_core_plugin_presence_report(plugin_map)
    core_coverage = _build_core_plugin_coverage_report(plugin_map)
    ticketing_provider = _collect_ticketing_provider_status(plugin_map)
    core_plugins = core_coverage.get("required_plugins") or []
    core_summary = core_coverage.get("summary") or {}
    return {
        "core_plugins": core_plugins,
        "plugins": core_plugins,
        "summary": core_summary,
        "required_state": core_coverage.get("required_overall_state") or "UNKNOWN",
        "presence_report": core_presence,
        "ticketing_provider": ticketing_provider,
        "updated_at": _now_iso(),
    }


@app.get("/api/plugins/core/reference")
async def list_core_plugin_reference(request: Request) -> Dict[str, Any]:
    _require_role(request, required="viewer")
    reference = _iter_core_plugin_reference()
    sorted_reference = sorted(reference, key=lambda item: str(item.get("plugin_id") or ""))
    required_total = sum(1 for item in sorted_reference if bool(item.get("required")))
    optional_total = len(sorted_reference) - required_total
    return {
        "plugins": sorted_reference,
        "summary": {
            "required_total": required_total,
            "optional_total": optional_total,
            "total": len(sorted_reference),
        },
        "updated_at": _now_iso(),
    }


@app.post("/api/plugins")
async def create_plugin(request: Request, plugin: PluginCreate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = plugin.model_dump()
    payload["id"] = str(payload["id"]).strip()
    payload["name"] = str(payload["name"]).strip()
    payload["source"] = str(payload["source"]).strip()
    payload["kind"] = str(payload["kind"]).strip()
    if not payload["id"]:
        raise HTTPException(status_code=400, detail="plugin id is required")
    if not payload["name"]:
        raise HTTPException(status_code=400, detail="plugin name is required")
    if not payload["source"]:
        raise HTTPException(status_code=400, detail="plugin source is required")
    if not payload["kind"]:
        raise HTTPException(status_code=400, detail="plugin kind is required")

    try:
        plugin_payload = storage.create_plugin(values=payload)
    except ValueError as exc:
        raise HTTPException(status_code=409 if "already exists" in str(exc) else 400, detail=str(exc))

    _event_for(role, "plugin.created", f"Plugin created: {payload['id']}", "info", {"plugin_id": payload["id"]})
    return {"plugin": _sanitize_plugin(plugin_payload, role), "updated_at": _now_iso()}


@app.patch("/api/plugins/{plugin_id}")
async def patch_plugin(request: Request, plugin_id: str, update: PluginUpdate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="No plugin fields provided")

    try:
        plugin = storage.update_plugin(plugin_id=plugin_id, values=payload)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"plugin '{plugin_id}' not found")

    _event_for(role, "plugin.updated", f"Plugin updated: {plugin_id}", "info", {"plugin_id": plugin_id})
    return {"plugin": _sanitize_plugin(plugin, role), "updated_at": _now_iso()}


@app.post("/api/plugins/{plugin_id}/required")
async def promote_core_plugin_requiredness(request: Request, plugin_id: str) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    normalized = str(plugin_id).strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="plugin_id is required")
    if not is_known_core_plugin(normalized):
        raise HTTPException(status_code=404, detail=f"plugin '{plugin_id}' is not a known core plugin")

    state = set_required_plugin(normalized, True)
    canonical = str(resolve_core_plugin_alias(normalized) or normalized).strip()
    _event_for(
        role,
        "plugin.required_required",
        f"Core plugin promoted to required: {normalized}",
        "info",
        {"plugin_id": normalized},
    )

    return {
        "plugin_id": normalized,
        "required": bool(is_required_core_plugin(canonical)),
        "base_required": bool(is_base_required_core_plugin(canonical)),
        "required_override": bool(is_required_core_plugin(canonical) and not is_base_required_core_plugin(canonical)),
        "required_overrides": sorted(state.required_overrides),
        "updated_at": _now_iso(),
    }


@app.delete("/api/plugins/{plugin_id}/required")
async def clear_core_plugin_requiredness(request: Request, plugin_id: str) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    normalized = str(plugin_id).strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="plugin_id is required")
    if not is_known_core_plugin(normalized):
        raise HTTPException(status_code=404, detail=f"plugin '{plugin_id}' is not a known core plugin")

    state = set_required_plugin(normalized, False)
    canonical = str(resolve_core_plugin_alias(normalized) or normalized).strip()
    _event_for(
        role,
        "plugin.required_cleared",
        f"Core plugin required override cleared: {normalized}",
        "info",
        {"plugin_id": normalized},
    )

    return {
        "plugin_id": normalized,
        "required": bool(is_required_core_plugin(canonical)),
        "base_required": bool(is_base_required_core_plugin(canonical)),
        "required_override": bool(is_required_core_plugin(canonical) and not is_base_required_core_plugin(canonical)),
        "required_overrides": sorted(state.required_overrides),
        "updated_at": _now_iso(),
    }


@app.post("/api/plugins/{plugin_id}/ui/{action_id}")
async def execute_plugin_ui_action(
    request: Request,
    plugin_id: str,
    action_id: str,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    plugin = _plugin_map().get(str(plugin_id).strip())
    if not plugin:
        raise HTTPException(status_code=404, detail=f"plugin '{plugin_id}' not found")

    action = _find_plugin_ui_action(plugin, action_id)
    if action is None:
        raise HTTPException(status_code=404, detail=f"action '{action_id}' not found for plugin '{plugin_id}'")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}

    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="invalid request payload")

    method = str(body.get("method") or "POST").strip().upper() or "POST"
    payload: Dict[str, Any] = {}
    defaults = action.get("defaults")
    if isinstance(defaults, dict):
        payload.update({k: v for k, v in defaults.items()})

    for key, value in body.items():
        if key == "method":
            continue
        payload[str(key)] = value

    result = runtime.plugin_ui_action(
        plugin_id=plugin_id,
        action_id=action_id,
        payload=payload,
        method=method,
        action=action,
        plugin_source=plugin.get("source"),
    )
    _event_for(
        role,
        "plugin.ui_action",
        f"Plugin action executed: {plugin_id}/{action_id}",
        "info",
        {
            "plugin_id": str(plugin_id),
            "action_id": str(action_id),
            "method": method,
            "success": result.success,
        },
    )

    if not result.success:
        raise HTTPException(status_code=502, detail=result.message)

    result_payload = dict(result.payload) if isinstance(result.payload, dict) else {"value": result.payload}
    inner_payload = result_payload.get("payload")
    if isinstance(inner_payload, dict):
        nested_payload = inner_payload.get("payload")
        if (
            isinstance(nested_payload, dict)
            and "plugin" in inner_payload
            and "method" in inner_payload
            and "payload" in inner_payload
        ):
            result_payload["payload"] = nested_payload

    return {
        "result": {
            "success": result.success,
            "message": result.message,
            "payload": result_payload,
        },
        "updated_at": _now_iso(),
    }


@app.get("/api/plugins/{plugin_id}/ui/debug")
async def debug_plugin_ui(request: Request, plugin_id: str) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    normalized_plugin = str(plugin_id).strip()
    plugin_map = _plugin_map()
    core_plugins_report = _collect_core_plugin_presence_report(plugin_map)
    plugin = plugin_map.get(normalized_plugin)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"plugin '{plugin_id}' not found")

    ui_actions = _collect_plugin_ui_actions(plugin)
    debug_action = _find_plugin_ui_action(plugin, "debug")
    has_debug = debug_action is not None
    reference = _find_core_plugin_reference(normalized_plugin)
    reference_warnings: List[Dict[str, str]] = []
    if reference and reference.get("required"):
        signature_metadata = plugin.get("metadata") if isinstance(plugin, dict) else {}
        signature_block = signature_metadata.get("signature") if isinstance(signature_metadata, dict) else None
        if reference.get("signature_required") and not signature_block:
            reference_warnings.append(
                {
                    "code": "P_PLUGIN_SIGNATURE_MISSING_REQUIRED",
                    "severity": "error",
                    "source": "registry",
                    "message": "required plugin signature is missing from metadata",
                },
            )
        elif reference.get("signature_required") and not signature_block.get("status") and isinstance(signature_block, dict):
            reference_warnings.append(
                {
                    "code": "P_PLUGIN_SIGNATURE_INVALID",
                    "severity": "error",
                    "source": "registry",
                    "message": "required plugin signature record indicates unresolved status",
                },
            )
    if reference and not reference.get("required") and not reference.get("signature_required"):
        signature_metadata = plugin.get("metadata") if isinstance(plugin, dict) else {}
        if not isinstance(signature_metadata, dict) or not signature_metadata.get("signature"):
            reference_warnings.append(
                {
                    "code": "P_PLUGIN_SIGNATURE_MISSING_OPTIONAL",
                    "severity": "warn",
                    "source": "registry",
                    "message": "optional plugin metadata signature is missing",
                },
            )

    runtime_payload: Dict[str, Any]
    dependency_warnings: List[Dict[str, str]] = []
    declared_dependencies = [
        _normalize_core_plugin_dependency_id(str(dependency).strip())
        for dependency in (list(reference.get("depends_on") if reference else []) + _collect_plugin_dependencies(plugin))
        if str(dependency).strip()
    ]
    seen_dependencies: set[str] = set()
    unique_dependencies = []
    for dependency in declared_dependencies:
        if dependency not in seen_dependencies:
            seen_dependencies.add(dependency)
            unique_dependencies.append(dependency)

    for dependency in unique_dependencies:
        warning = _dependency_warning(dependency, plugin_map, owner_required=bool(reference.get("required")) if reference else False)
        if warning:
            dependency_warnings.append(warning)

    runtime_called = False
    if has_debug:
        runtime_called = True
        runtime_result = runtime.plugin_ui_action(
            plugin_id=normalized_plugin,
            action_id="debug",
            method="GET",
            action=debug_action,
            plugin_source=plugin.get("source"),
        )
    else:
        runtime_result = RuntimeActionResult(
            success=False,
            message="debug action unavailable in plugin ui contract",
            payload={
                "action_id": "debug",
                "method": "GET",
                "reason": "action_not_declared_in_ui_contract",
            },
        )

    tool_count = 0
    active_tool_count = 0
    tool_warnings: List[Dict[str, str]] = []
    try:
        tool_rows = storage.list_tools(plugin_id=normalized_plugin)
        tool_count = len(tool_rows)
        active_tool_count = len([tool for tool in tool_rows if str(tool.get("status") or "").strip() != "retired"])
    except Exception as exc:  # pragma: no cover
        tool_warnings.append(
            {
                "code": "P_PLUGIN_TOOLS_QUERY_FAILED",
                "severity": "warn",
                "source": "tools",
                "message": f"could not query tool registry for {normalized_plugin}: {exc}",
            },
        )

    conflicts = _collect_plugin_tool_conflicts(normalized_plugin)
    if conflicts:
        tool_warnings.append(
            {
                "code": "P_PLUGIN_TOOL_NAMESPACE_CONFLICT",
                "severity": "warn",
                "source": "tools",
                "message": f"tool namespace/action conflicts detected for {len(conflicts)} action(s)",
            },
        )

    local_handler_warnings = _plugin_ui_handler_warnings_for_contract(plugin)
    ui_warnings = _plugin_ui_contract_diagnostics(plugin)
    ui_warnings.extend(local_handler_warnings)
    if not has_debug:
        ui_warnings.append(
            {
                "code": "P_PLUGIN_DEBUG_ACTION_MISSING",
                "severity": "warn",
                "source": "ui",
                "message": "plugin ui contract does not expose a debug action",
            },
        )
    if not runtime_result.success:
        ui_warnings.append(
            {
                "code": "P_PLUGIN_RUNTIME_DEBUG_FAILED",
                "severity": "warn",
                "source": "runtime",
                "message": "plugin debug action returned failure",
            },
        )
    elif has_debug and any(
        str(entry.get("code") or "").startswith("P_PLUGIN_UI_HANDLER_")
        for entry in local_handler_warnings
    ):
        ui_warnings.append(
            {
                "code": "P_PLUGIN_RUNTIME_DEBUG_FAILED",
                "severity": "warn",
                "source": "runtime",
                "message": "plugin debug action succeeded but ui handler diagnostics reported failures",
            },
        )

    diagnostics = [
        *ui_warnings,
        *reference_warnings,
        *tool_warnings,
        *dependency_warnings,
    ]
    errors = [entry for entry in diagnostics if entry.get("severity") == "error"]
    status = "error" if any(entry.get("severity") == "error" for entry in diagnostics) else ("warn" if diagnostics else "ok")
    warnings = {
        "count": len(diagnostics),
        "entries": diagnostics,
    }

    _event_for(
        role,
        "plugin.ui_debug",
        f"Plugin UI debug probe executed: {plugin_id}",
        "info",
        {
            "plugin_id": normalized_plugin,
            "action_id": "debug",
            "runtime_success": runtime_result.success,
            "runtime_called": runtime_called,
            "warning_count": len(diagnostics),
        },
    )

    return {
        "plugin": _sanitize_plugin_for_debug(plugin),
        "reference": {
            "required": bool(reference.get("required")) if reference else False,
            "base_required": bool(reference.get("base_required")) if reference else False,
            "required_override": bool(reference.get("required_override")) if reference else False,
            "signature_required": bool(reference.get("signature_required")) if reference else False,
            "aliases": reference.get("aliases") if reference else [],
            "matched_alias": reference.get("matched_alias") if reference else normalized_plugin,
            "depends_on": reference.get("depends_on") if reference else [],
        },
        "ui": {
            "actions": ui_actions,
            "has_debug_action": has_debug,
            "action_count": len(ui_actions),
        },
        "tools": {
            "total": tool_count,
            "active": active_tool_count,
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
        },
        "warnings": warnings,
        "runtime": {
            "action": "debug",
            "method": "GET",
            "success": runtime_result.success,
            "message": runtime_result.message,
            "payload": _redact_sensitive_payload(runtime_result.payload, role),
            "runtime_called": runtime_called,
        },
        "core_plugins": core_plugins_report,
        "dependencies": {
            "declared": unique_dependencies,
            "warnings": dependency_warnings,
            "count": len(unique_dependencies),
        },
        "status": status,
        "errors": {
            "count": len(errors),
            "entries": errors,
        },
        "updated_at": _now_iso(),
    }


@app.get("/api/tools")
async def list_tools(
    request: Request,
    q: Optional[str] = None,
    namespace: Optional[str] = None,
    module: Optional[str] = None,
    plugin_id: Optional[str] = None,
    status: Optional[str] = None,
    sort: str = "popularity",
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    plugins = _plugin_map()
    tools = storage.list_tools(
        q=q,
        namespace=namespace,
        module=module,
        plugin_id=plugin_id,
        status=status,
        sort=sort,
        limit=limit,
        offset=offset,
    )
    return {
        "tools": [_sanitize_tool_payload(tool, role, plugins) for tool in tools],
        "count": storage.count_tools(
            q=q,
            namespace=namespace,
            module=module,
            plugin_id=plugin_id,
            status=status,
        ),
        "updated_at": _now_iso(),
    }


@app.get("/api/tools/search")
async def search_tools(
    request: Request,
    q: Optional[str] = None,
    namespace: Optional[str] = None,
    module: Optional[str] = None,
    plugin_id: Optional[str] = None,
    status: Optional[str] = None,
    match_mode: str = "contains",
    sort: str = "popularity",
    group_by: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    plugins = _plugin_map()
    try:
        tools, count, groups = storage.search_tools(
            q=q,
            namespace=namespace,
            module=module,
            plugin_id=plugin_id,
            status=status,
            match_mode=match_mode,
            sort=sort,
            group_by=group_by,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    response: Dict[str, Any] = {
        "tools": [_sanitize_tool_payload(tool, role, plugins) for tool in tools],
        "count": count,
        "updated_at": _now_iso(),
    }
    if groups:
        response["groups"] = groups
        response["group_by"] = group_by
    return response


@app.get("/api/tools/usage")
async def get_tool_usage_global(
    request: Request,
    tool_id: Optional[str] = Query(default=None),
    agent_id: Optional[str] = Query(default=None),
    mission_id: Optional[str] = Query(default=None),
    memory_id: Optional[str] = Query(default=None),
    chat_message_id: Optional[str] = Query(default=None),
    chat_session_id: Optional[str] = Query(default=None),
    context_type: Optional[str] = Query(default=None),
    context_id: Optional[str] = Query(default=None),
    date: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort_desc: bool = True,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    usage = storage.list_tool_usage_global(
        tool_id=tool_id,
        agent_id=agent_id,
        mission_id=mission_id,
        memory_id=memory_id,
        chat_message_id=chat_message_id,
        chat_session_id=chat_session_id,
        context_type=context_type,
        context_id=context_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
        sort_desc=sort_desc,
    )
    count = storage.count_tool_usage(
        tool_id=tool_id,
        agent_id=agent_id,
        mission_id=mission_id,
        memory_id=memory_id,
        chat_message_id=chat_message_id,
        chat_session_id=chat_session_id,
        context_type=context_type,
        context_id=context_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
    )
    return {
        "tool_id": tool_id,
        "agent_id": agent_id,
        "memory_id": memory_id,
        "chat_message_id": chat_message_id,
        "chat_session_id": chat_session_id,
        "context_type": context_type,
        "context_id": context_id,
        "count": count,
        "usage": [_sanitize_tool_usage_payload(entry, role) for entry in usage],
        "updated_at": _now_iso(),
    }


@app.get("/api/tools/{tool_id}")
async def get_tool(request: Request, tool_id: str) -> Dict[str, Any]:
    if tool_id in {"usage", "search"}:
        raise HTTPException(status_code=404, detail=f"tool '{tool_id}' not found")
    role = _require_role(request, required="viewer")
    tool = storage.get_tool_registry(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"tool '{tool_id}' not found")
    return {
        "tool": _sanitize_tool_payload(tool, role, _plugin_map()),
        "updated_at": _now_iso(),
    }


@app.get("/api/tools/{tool_id}/usage")
async def get_tool_usage(
    request: Request,
    tool_id: str,
    agent_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    date: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort_desc: bool = True,
) -> Dict[str, Any]:
    if tool_id in {"usage", "search"}:
        raise HTTPException(status_code=404, detail=f"tool '{tool_id}' not found")
    role = _require_role(request, required="viewer")
    count = storage.count_tool_usage(
        tool_id=tool_id,
        agent_id=agent_id,
        mission_id=mission_id,
        memory_id=memory_id,
        chat_message_id=chat_message_id,
        chat_session_id=chat_session_id,
        context_type=context_type,
        context_id=context_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
    )
    usage = storage.list_tool_usage(
        tool_id=tool_id,
        agent_id=agent_id,
        mission_id=mission_id,
        memory_id=memory_id,
        chat_message_id=chat_message_id,
        chat_session_id=chat_session_id,
        context_type=context_type,
        context_id=context_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
        sort_desc=sort_desc,
    )
    return {
        "tool_id": tool_id,
        "memory_id": memory_id,
        "chat_message_id": chat_message_id,
        "chat_session_id": chat_session_id,
        "count": count,
        "usage": [_sanitize_tool_usage_payload(entry, role) for entry in usage],
        "updated_at": _now_iso(),
    }


@app.post("/api/tools/{tool_id}/usage")
async def record_tool_usage(
    request: Request,
    tool_id: str,
    payload: ToolUsageCreate,
) -> Dict[str, Any]:
    if tool_id in {"usage", "search"}:
        raise HTTPException(status_code=404, detail=f"tool '{tool_id}' not found")
    role = _require_role(request, required="admin")
    try:
        usage = storage.append_tool_usage(
            tool_id=tool_id,
            agent_id=payload.agent_id,
            session_id=payload.session_id,
            mission_id=payload.mission_id,
            request_id=payload.request_id,
            memory_id=payload.memory_id,
            chat_message_id=payload.chat_message_id,
            chat_session_id=payload.chat_session_id,
            context_type=payload.context_type,
            context_id=payload.context_id,
            status=payload.status,
            duration_ms=payload.duration_ms,
            result_status=payload.result_status,
            details=payload.details,
            payload=payload.payload,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    _event_for(role, "tool.usage", f"Tool usage recorded: {tool_id}", "info", {
        "tool_id": tool_id,
        "agent_id": payload.agent_id,
        "mission_id": payload.mission_id,
        "session_id": payload.session_id,
        "request_id": payload.request_id,
        "memory_id": payload.memory_id,
        "chat_message_id": payload.chat_message_id,
        "chat_session_id": payload.chat_session_id,
        "context_type": payload.context_type,
        "context_id": payload.context_id,
        "result_status": payload.result_status,
    })
    return {"usage": usage, "updated_at": _now_iso()}


@app.get("/api/tool-log")
async def get_tool_log(
    request: Request,
    tool_id: Optional[str] = Query(default=None),
    agent_id: Optional[str] = Query(default=None),
    mission_id: Optional[str] = Query(default=None),
    memory_id: Optional[str] = Query(default=None),
    chat_message_id: Optional[str] = Query(default=None),
    chat_session_id: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    context_type: Optional[str] = Query(default=None),
    context_id: Optional[str] = Query(default=None),
    date: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort_desc: bool = True,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    filters = {
        "tool_id": tool_id,
        "agent_id": agent_id,
        "mission_id": mission_id,
        "memory_id": memory_id,
        "chat_message_id": chat_message_id,
        "chat_session_id": chat_session_id,
        "context_type": context_type,
        "context_id": context_id,
    }
    usage = storage.list_tool_usage_global(
        tool_id=tool_id,
        agent_id=agent_id,
        session_id=session_id,
        mission_id=mission_id,
        memory_id=memory_id,
        chat_message_id=chat_message_id,
        chat_session_id=chat_session_id,
        context_type=context_type,
        context_id=context_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
        sort_desc=sort_desc,
    )
    count = storage.count_tool_usage(
        tool_id=tool_id,
        agent_id=agent_id,
        session_id=session_id,
        mission_id=mission_id,
        memory_id=memory_id,
        chat_message_id=chat_message_id,
        chat_session_id=chat_session_id,
        context_type=context_type,
        context_id=context_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
    )
    return {
        "tool_id": tool_id,
        "agent_id": agent_id,
        "mission_id": mission_id,
        "memory_id": memory_id,
        "chat_message_id": chat_message_id,
        "chat_session_id": chat_session_id,
        "session_id": session_id,
        "context_type": context_type,
        "context_id": context_id,
        "count": count,
        "usage": [_sanitize_tool_usage_payload(entry, role) for entry in usage],
        "updated_at": _now_iso(),
        "legacy_filters": filters,
    }


@app.get("/api/tool-log/session/{session_id}")
async def get_tool_log_by_session(
    request: Request,
    session_id: str,
    date: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    usage = storage.list_tool_usage_global(
        session_id=session_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
        limit=200,
        sort_desc=True,
    )
    return {
        "session_id": session_id,
        "count": len(usage),
        "usage": [_sanitize_tool_usage_payload(entry, role) for entry in usage],
        "updated_at": _now_iso(),
    }


@app.get("/api/tool-log/{tool_call_id}")
async def get_tool_log_entry(request: Request, tool_call_id: str) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    usage = storage.list_tool_usage_global(limit=1000, sort_desc=True)
    for entry in usage:
        if entry.get("id") == tool_call_id:
            return {
                "tool_call_id": tool_call_id,
                "usage": _sanitize_tool_usage_payload(entry, role),
                "updated_at": _now_iso(),
            }
    raise HTTPException(status_code=404, detail=f"tool usage '{tool_call_id}' not found")


@app.get("/api/agents/{agent_id}/tool_usage")
async def get_agent_tool_usage(
    request: Request,
    agent_id: str,
    tool_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    date: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort_desc: bool = True,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    count = storage.count_tool_usage(
        tool_id=tool_id,
        agent_id=agent_id,
        session_id=session_id,
        mission_id=mission_id,
        memory_id=memory_id,
        chat_message_id=chat_message_id,
        chat_session_id=chat_session_id,
        context_type=context_type,
        context_id=context_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
    )
    usage = storage.list_tool_usage_global(
        tool_id=tool_id,
        agent_id=agent_id,
        session_id=session_id,
        mission_id=mission_id,
        memory_id=memory_id,
        chat_message_id=chat_message_id,
        chat_session_id=chat_session_id,
        context_type=context_type,
        context_id=context_id,
        date=date,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
        sort_desc=sort_desc,
    )
    return {
        "agent_id": agent_id,
        "tool_id": tool_id,
        "memory_id": memory_id,
        "chat_message_id": chat_message_id,
        "chat_session_id": chat_session_id,
        "session_id": session_id,
        "mission_id": mission_id,
        "count": count,
        "usage": [_sanitize_tool_usage_payload(entry, role) for entry in usage],
        "updated_at": _now_iso(),
    }


@app.get("/api/agents/{agent_id}/audit")
async def get_agent_audit(
    request: Request,
    agent_id: str,
    mission_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    chat_message_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tool_limit: int = Query(default=5, ge=1, le=50),
    session_limit: int = Query(default=5, ge=1, le=50),
    mission_limit: int = Query(default=5, ge=1, le=50),
    recent_limit: int = Query(default=25, ge=1, le=200),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    try:
        payload = storage.get_agent_tool_audit(
            agent_id=agent_id,
            mission_id=mission_id,
            memory_id=memory_id,
            chat_message_id=chat_message_id,
            chat_session_id=chat_session_id,
            context_type=context_type,
            context_id=context_id,
            session_id=session_id,
            tool_limit=tool_limit,
            session_limit=session_limit,
            mission_limit=mission_limit,
            recent_limit=recent_limit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    payload["summary"] = payload.get("summary", {})
    payload["summary"]["unique_sessions"] = len(payload.get("session_breakdown", []))
    payload["summary"]["unique_missions"] = len(payload.get("mission_breakdown", []))
    payload["recent_calls"] = [
        _sanitize_tool_usage_payload(entry, role) for entry in payload.get("recent_calls", [])
    ]
    payload["tool_breakdown"] = payload.get("tool_breakdown", [])
    payload["session_breakdown"] = payload.get("session_breakdown", [])
    payload["mission_breakdown"] = payload.get("mission_breakdown", [])
    payload["updated_at"] = _now_iso()
    return payload


@app.get("/api/agents/{agent_id}/overlay/history")
async def get_agent_overlay_history(
    request: Request,
    agent_id: str,
    limit: int = Query(default=20, ge=1, le=200),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    normalized_agent_id = str(agent_id or "").strip()
    history_payload = runtime.get_actor_overlay_history(normalized_agent_id, limit=limit)
    if not isinstance(history_payload, dict):
        raise HTTPException(status_code=502, detail="agent overlay history runtime response invalid")

    history = history_payload.get("history", [])
    sanitized = []
    if isinstance(history, list):
        sanitized = [
            _sanitize_overlay_history_record(item, role)
            for item in history
            if isinstance(item, dict)
        ]

    response = {
        "success": bool(history_payload.get("success", True)),
        "actor_id": str(history_payload.get("actor_id", normalized_agent_id)),
        "history": sanitized,
        "count": int(history_payload.get("count", len(sanitized))),
    }
    if isinstance(history_payload.get("error"), str):
        response["error"] = history_payload.get("error")
    return response


@app.get("/api/agents")
async def get_agents(
    request: Request,
    lifecycle: str = Query(default=storage.AGENT_LIFECYCLE_ALL),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    try:
        agents = [_agent_with_overlay(agent, role) for agent in storage.list_agents(lifecycle=lifecycle)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"agents": agents, "updated_at": _now_iso()}


@app.get("/api/agents/directory")
async def get_agent_directory(request: Request) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    directory = storage.list_agent_directory()
    directory_artifact = storage.get_agent_directory_artifact()
    if isinstance(directory_artifact, dict):
        import_id = directory_artifact.get("generated_from_import_id")
        if import_id:
            directory_artifact["lineage"] = _collect_import_lineage(str(import_id))
        directory_artifact["source_metadata"] = _redact_metadata(
            directory_artifact.get("source_metadata"),
            role,
        )
    return {
        "directory": directory,
        "directory_artifact": directory_artifact,
        "updated_at": _now_iso(),
        "role": role,
    }


@app.patch("/api/agents/{agent_id}")
async def patch_agent(request: Request, agent_id: str, update: AgentUpdate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = {k: v for k, v in update.model_dump(exclude_unset=True).items()}
    overlay_content = payload.pop("overlay_content", None)
    overlay_token_cap = payload.pop("overlay_token_cap", None)

    if overlay_content is None and overlay_token_cap is not None:
        raise HTTPException(status_code=400, detail="overlay_content is required when overlay_token_cap is provided")

    normalized_overlay = {k: v for k, v in payload.items() if v is not None}

    if overlay_content is not None:
        try:
            overlay_result = runtime.write_actor_overlay(
                str(agent_id).strip(),
                str(overlay_content),
                token_cap=_coerce_overlay_token_cap(overlay_token_cap),
                source="management-ui",
            )
        except HTTPException:
            raise
        except Exception as exc:
            _event_for(role, "agent.overlay.failed", f"Agent overlay update failed: {agent_id}", "warn", {"agent_id": str(agent_id), "error": str(exc)})
            raise HTTPException(status_code=502, detail=f"agent overlay update failed: {exc}") from exc

        if not isinstance(overlay_result, dict) or not overlay_result.get("success", False):
            raise HTTPException(
                status_code=502,
                detail=overlay_result.get("error", "overlay write failed") if isinstance(overlay_result, dict) else "overlay write failed",
            )

    if not normalized_overlay and overlay_content is None:
        raise HTTPException(status_code=400, detail="No agent fields provided")

    try:
        agent = storage.update_agent(agent_id, normalized_overlay)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")

    if overlay_content is not None:
        _event_for(role, "agent.overlay", f"Agent overlay updated: {agent_id}", "info", {"agent_id": str(agent_id)})

    _event_for(role, "agent", f"Agent updated: {agent_id}", "info")
    response_agent = _agent_with_overlay(agent, role)
    return {"agent": response_agent, "updated_at": _now_iso()}


@app.post("/api/agents/{agent_id}/archive")
async def archive_agent(
    request: Request,
    agent_id: str,
    action: Optional[AgentLifecycleAction] = None,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = _coerce_agent_lifecycle_payload(
        role,
        action,
        fallback_reason="agent archived by operator",
    )
    replacement_agent_id = _coerce_replacement_agent(agent_id, payload.pop("replacement_agent_id", "")) if payload.get("replacement_agent_id") else None
    if replacement_agent_id:
        payload["replacement_agent_id"] = replacement_agent_id
    try:
        agent = storage.archive_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")

    _event_for(
        role,
        "agent.lifecycle.archive",
        f"Agent archived: {agent_id}",
        "warn",
        payload={"agent_id": str(agent_id), **payload},
    )
    return {"agent": _agent_with_overlay(agent, role), "updated_at": _now_iso()}


@app.post("/api/agents/{agent_id}/restore")
async def restore_agent(
    request: Request,
    agent_id: str,
    action: Optional[AgentLifecycleAction] = None,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    payload = _coerce_agent_lifecycle_payload(
        role,
        action,
        fallback_reason="agent restored by operator",
        replacement_allowed=False,
    )
    try:
        agent = storage.restore_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"agent '{agent_id}' not found")

    _event_for(
        role,
        "agent.lifecycle.restore",
        f"Agent restored: {agent_id}",
        "info",
        payload={"agent_id": str(agent_id), **payload},
    )
    return {"agent": _agent_with_overlay(agent, role), "updated_at": _now_iso()}


@app.post("/api/gm-tickets")
async def create_gm_ticket(request: Request, payload: GmTicketCreate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=400, detail="ticket payload is required")
    try:
        ticket = storage.create_gm_ticket(values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ticket = _sanitize_gm_ticket_payload(ticket, role)
    _event_for(
        role,
        "gm_ticket.created",
        f"GM ticket created: {ticket['id']}",
        "info",
        payload={
            "gm_ticket_id": str(ticket.get("id")),
            "status": ticket.get("status"),
            "priority": ticket.get("priority"),
            "actor": role,
        },
    )
    return {"ticket": ticket, "updated_at": _now_iso()}


@app.post("/api/gm/message")
async def send_direct_gm_message(request: Request, payload: GmDirectMessageCreate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    values = payload.model_dump(exclude_unset=True)

    sender = str(values.get("sender", "")).strip()
    if not sender:
        raise HTTPException(status_code=400, detail="sender is required")
    content = str(values.get("content", "")).strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    ticket_id = str(values.get("ticket_id") or "").strip()
    message_type = str(values.get("message_type") or "comment").strip() or "comment"
    metadata = values.get("metadata")

    created_ticket = False
    resolved_ticket_id = ticket_id
    if not resolved_ticket_id:
        created_ticket = True
        title = str(values.get("title") or "").strip() or f"GM direct message from {sender}"
        ticket_payload = {
            "title": title[:140],
            "requested_by": sender,
            "status": "requested",
            "priority": "normal",
            "agent_scope": "global",
            "phase": "direct_message",
            "metadata": {
                "source": "direct_message",
                "initiator": sender,
            },
        }
        try:
            ticket = storage.create_gm_ticket(ticket_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        resolved_ticket_id = str(ticket.get("id") or "").strip()
        if not resolved_ticket_id:
            raise HTTPException(status_code=500, detail="failed to initialize direct GM ticket")
    else:
        existing = storage.get_gm_ticket(resolved_ticket_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"ticket '{resolved_ticket_id}' not found")

        ticket = existing

    try:
        message = storage.append_gm_ticket_message(
            resolved_ticket_id,
            sender=sender,
            content=content,
            message_type=message_type,
            metadata=metadata,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"ticket '{resolved_ticket_id}' not found",
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sanitized_ticket = _sanitize_gm_ticket_payload(ticket, role)
    _event_for(
        role,
        "gm_ticket.direct_message",
        f"Direct GM message sent to ticket: {resolved_ticket_id}",
        "info",
        payload={
            "gm_ticket_id": resolved_ticket_id,
            "sender": message.get("sender"),
            "message_type": message.get("message_type"),
            "created_ticket": created_ticket,
            "actor": role,
        },
    )
    return {
        "ticket": sanitized_ticket,
        "message": dict(message, metadata=_redact_metadata(message.get("metadata"), role)),
        "created_ticket": created_ticket,
        "updated_at": _now_iso(),
    }


@app.get("/api/gm-tickets")
async def list_gm_tickets(
    request: Request,
    status: Optional[str] = Query(default=None),
    priority: Optional[str] = Query(default=None),
    assigned_to: Optional[str] = Query(default=None),
    phase: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    try:
        tickets = storage.list_gm_tickets(
            status=status,
            priority=priority,
            assigned_to=assigned_to,
            phase=phase,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sanitized = [_sanitize_gm_ticket_payload(ticket, role) for ticket in tickets]
    return {
        "tickets": sanitized,
        "count": len(sanitized),
        "updated_at": _now_iso(),
    }


@app.get("/api/gm-tickets/{ticket_id}")
async def get_gm_ticket(request: Request, ticket_id: str) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    ticket = storage.get_gm_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"ticket '{ticket_id}' not found")
    return {
        "ticket": _sanitize_gm_ticket_payload(ticket, role),
        "updated_at": _now_iso(),
    }


@app.get("/api/gm-tickets/{ticket_id}/audit")
async def get_gm_ticket_audit(
    request: Request,
    ticket_id: str,
    event_limit: int = Query(default=100, ge=1, le=1000),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    normalized_ticket_id = str(ticket_id).strip()
    if not normalized_ticket_id:
        raise HTTPException(status_code=404, detail=f"ticket '{ticket_id}' not found")

    return _build_gm_ticket_audit_payload(normalized_ticket_id, role, event_limit=event_limit)


@app.get("/api/gm-tickets/{ticket_id}/audit/export")
async def export_gm_ticket_audit(
    request: Request,
    ticket_id: str,
    event_limit: int = Query(default=100, ge=1, le=1000),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    normalized_ticket_id = str(ticket_id).strip()
    if not normalized_ticket_id:
        raise HTTPException(status_code=404, detail=f"ticket '{ticket_id}' not found")

    audit = _build_gm_ticket_audit_payload(normalized_ticket_id, role, event_limit=event_limit)
    generated_at = _now_iso()
    updated_at = _now_iso()
    export_payload = {
        "export_format": "gm-ticket-audit-v1",
        "ticket_id": normalized_ticket_id,
        "generated_at": generated_at,
        "generated_by": role,
        "role": role,
        "audit": audit,
        "updated_at": updated_at,
    }
    canonical = json.dumps(export_payload, sort_keys=True, separators=(",", ":"))
    export_payload["audit_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return export_payload


@app.patch("/api/gm-tickets/{ticket_id}")
async def update_gm_ticket(request: Request, ticket_id: str, payload: GmTicketUpdate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=400, detail="No ticket fields provided")
    try:
        ticket = storage.update_gm_ticket(ticket_id, values)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"ticket '{ticket_id}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ticket = _sanitize_gm_ticket_payload(ticket, role)
    _event_for(
        role,
        "gm_ticket.updated",
        f"GM ticket updated: {ticket_id}",
        "info",
        payload={
            "gm_ticket_id": str(ticket_id),
            "fields": sorted(values.keys()),
            "actor": role,
        },
    )
    return {"ticket": ticket, "updated_at": _now_iso()}


@app.post("/api/gm-tickets/{ticket_id}/dispatch")
async def dispatch_gm_ticket(request: Request, ticket_id: str, payload: GmTicketDispatch) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    normalized_ticket_id = str(ticket_id).strip()
    if not normalized_ticket_id:
        raise HTTPException(status_code=400, detail="ticket_id is required")

    ticket = storage.get_gm_ticket(normalized_ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"ticket '{ticket_id}' not found")

    values = payload.model_dump(exclude_unset=True)
    objective = str(values.get("objective") or "").strip() or str(ticket.get("title") or "").strip()
    if not objective:
        objective = f"GM ticket {normalized_ticket_id}"

    request_payload: Dict[str, Any] = {
        "objective": objective,
    }

    def _coerce_optional_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _coerce_assigned_to(value: Any) -> Optional[str]:
        text = _coerce_optional_text(value)
        if not text:
            return None
        if text.lower() == "unassigned":
            return None
        return text

    ticket_request_id = _coerce_optional_text(ticket.get("gm_request_id")) or normalized_ticket_id
    ticket_build_ticket_id = _coerce_optional_text(ticket.get("build_ticket_id")) or normalized_ticket_id
    ticket_build_batch_id = _coerce_optional_text(ticket.get("build_batch_id")) or f"batch-{normalized_ticket_id}"
    ticket_dispatch_token = _coerce_optional_text(ticket.get("dispatch_token"))
    ticket_dispatch_nonce = _coerce_optional_text(ticket.get("dispatch_nonce"))
    ticket_assigned_to = _coerce_assigned_to(ticket.get("assigned_to"))
    ticket_phase2_mission = _coerce_optional_text(ticket.get("phase2_mission") or ticket.get("mission_id"))

    for field in (
        "mission_id",
        "assigned_to",
        "request_id",
        "gm_request_id",
        "dispatch_token",
        "dispatch_nonce",
        "build_ticket_id",
        "build_batch_id",
        "max_steps",
        "qa_max_retries",
        "acceptance_criteria",
        "allowed_namespaces",
    ):
        value = values.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        if value is not None:
            request_payload[str(field)] = value

    dispatch_role = _coerce_gm_dispatch_role(values.get("role"))
    if dispatch_role is not None:
        request_payload["role"] = dispatch_role

    request_payload.setdefault("request_id", request_payload.get("gm_request_id") or ticket_request_id)
    request_payload.setdefault("gm_request_id", request_payload.get("request_id"))
    request_payload.setdefault("build_ticket_id", ticket_build_ticket_id)
    request_payload.setdefault("build_batch_id", ticket_build_batch_id)
    request_payload.setdefault("dispatch_token", ticket_dispatch_token)
    request_payload.setdefault("dispatch_nonce", ticket_dispatch_nonce)
    if (not request_payload.get("mission_id") and ticket_phase2_mission):
        request_payload.setdefault("mission_id", ticket_phase2_mission)
    if not request_payload.get("assigned_to") and ticket_assigned_to:
        request_payload.setdefault("assigned_to", ticket_assigned_to)

    if not request_payload.get("dispatch_token") or not request_payload.get("dispatch_nonce"):
        raise HTTPException(
            status_code=400,
            detail=(
                "phase-2 dispatch requires dispatch_token and dispatch_nonce from the ticket lineage or request payload"
            ),
        )

    result = runtime.dispatch_gm_ticket(
        ticket_id=normalized_ticket_id,
        payload=request_payload,
    )
    if not result.success:
        raise HTTPException(status_code=502, detail=result.message)

    _event_for(
        role,
        "gm_ticket.dispatch",
        f"GM ticket dispatched: {normalized_ticket_id}",
        "info",
        payload={
            "gm_ticket_id": normalized_ticket_id,
            "objective": objective,
            "assigned_to": request_payload.get("assigned_to"),
            "mission_id": request_payload.get("mission_id"),
            "role": request_payload.get("role"),
        },
    )

    return {
        "success": result.success,
        "message": result.message,
        "result": result.payload,
        "updated_at": _now_iso(),
    }


@app.post("/api/gm-tickets/{ticket_id}/messages")
async def create_gm_ticket_message(
    request: Request,
    ticket_id: str,
    payload: GmTicketMessageCreate,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    values = payload.model_dump(exclude_unset=True)
    try:
        message = storage.append_gm_ticket_message(
            ticket_id,
            sender=values["sender"],
            content=values["content"],
            message_type=values.get("message_type") or "comment",
            response_required=values.get("response_required", False),
            metadata=values.get("metadata"),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"ticket '{ticket_id}' not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _event_for(
        role,
        "gm_ticket.message",
        f"Message added to GM ticket: {ticket_id}",
        "info",
        payload={
            "gm_ticket_id": str(ticket_id),
            "sender": message.get("sender"),
            "message_type": message.get("message_type"),
            "response_required": bool(values.get("response_required", False)),
            "actor": role,
        },
    )
    return {
        "message": dict(message, metadata=_redact_metadata(message.get("metadata"), role)),
        "updated_at": _now_iso(),
    }


@app.get("/api/gm-tickets/{ticket_id}/messages")
async def list_gm_ticket_messages(request: Request, ticket_id: str) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    if not storage.get_gm_ticket(ticket_id):
        raise HTTPException(status_code=404, detail=f"ticket '{ticket_id}' not found")
    messages = _sanitize_gm_ticket_messages(storage.list_gm_ticket_messages(ticket_id), role)
    return {
        "ticket_id": str(ticket_id).strip(),
        "messages": messages,
        "count": len(messages),
        "updated_at": _now_iso(),
    }


def _normalise_source_type(source_type: str) -> str:
    return str(source_type or "").strip().lower().replace("-", "_").replace(" ", "_")


def _coerce_import_payload(raw: bytes | str) -> bytes:
    if isinstance(raw, bytes):
        return raw
    return raw.encode("utf-8")


def _normalise_intake_reason(value: str) -> str:
    value = str(value or "").strip()
    mapping = {
        "redaction_required": "redacted_sensitive_patterns",
        "blocked_attachment_type": "blocked_by_intake_policy",
    }
    return mapping.get(value, value)


_IMPORT_LOCAL_REVIEW_ONLY_PLACEHOLDER = "[REDACTED: local human review required]"


def _coerce_import_item_intake(
    item: Dict[str, Any],
    *,
    source_type: str,
    source_index: int = 0,
) -> tuple[Dict[str, Any], list[str]]:
    content = str(item.get("content") or "").strip()
    thread_ref = str(item.get("thread_id") or f"thread-{source_index}").strip()
    message_ref = str(item.get("message_id") or f"message-{source_index}").strip()

    file_name = f"{source_type}_{thread_ref}_{message_ref}.txt"
    file_name = file_name.replace("/", "_").replace("\\", "_")

    entry = {
        "filename": file_name,
        "mime_type": "text/plain",
        "size": len(content.encode("utf-8")),
        "source": f"management-import/{source_type}",
        "text_preview": content,
        "id": f"import:{source_type}:{thread_ref}:{message_ref}",
        "source_type": source_type,
    }

    decision = _make_import_intake_decision(entry)  # type: ignore[arg-type]
    reasons = [_normalise_intake_reason(str(reason)) for reason in list(entry.get("intake_reasons") or [])]
    reasons = list(dict.fromkeys([reason for reason in reasons if str(reason).strip()]))
    redacted_preview = str(entry.get("text_preview") or "").strip()
    redaction_changed_preview = bool(content) and redacted_preview and redacted_preview != content

    if decision == ATTACHMENT_DECISION_BLOCK and "blocked_by_intake_policy" not in reasons:
        reasons.append("blocked_by_intake_policy")

    metadata = dict(item.get("metadata") or {})
    raw_content_local_only = bool(
        decision != ATTACHMENT_DECISION_ACCEPT
        or redaction_changed_preview
        or metadata.get("requires_review")
        or metadata.get("sensitive_flags")
    )
    downstream_content_mode = "redacted_preview" if raw_content_local_only else "raw"
    downstream_redacted_preview = ""
    if raw_content_local_only:
        downstream_redacted_preview = redacted_preview if redaction_changed_preview else _IMPORT_LOCAL_REVIEW_ONLY_PLACEHOLDER
    metadata.update(
        {
            "requires_review": decision != ATTACHMENT_DECISION_ACCEPT,
            "intake_decision": decision,
            "intake_reasons": reasons,
            "intake_policy_version": entry.get("intake_policy_version"),
            "intake_source": entry.get("source"),
            "intake_warnings": reasons,
            "intake_text_preview_present": bool(content),
            "contains_sensitive_content": raw_content_local_only,
            "raw_content_local_only": raw_content_local_only,
            "agent_content_mode": downstream_content_mode,
            "provider_content_mode": downstream_content_mode,
            "redacted_preview": downstream_redacted_preview,
        }
    )

    item_payload = dict(item)
    item_payload["metadata"] = metadata

    if decision == ATTACHMENT_DECISION_BLOCK:
        item_payload["visibility"] = "quarantined"
        item_payload["review_state"] = "quarantined"
    elif decision == ATTACHMENT_DECISION_QUARANTINE:
        item_payload["visibility"] = "quarantined"
        item_payload["review_state"] = "quarantined"

    return item_payload, reasons


def _collect_import_lineage(import_id: str) -> List[Dict[str, Any]]:
    lineage: List[Dict[str, Any]] = []
    seen: set[str] = set()
    current = import_id

    while current:
        if current in seen:
            break
        seen.add(current)

        job = storage.get_import_job(current)
        if not job:
            break

        source_metadata = job.get("source_metadata") or {}
        if isinstance(source_metadata, str):
            source_metadata = _coerce_json_metadata(source_metadata)
            if source_metadata is None:
                source_metadata = {}
        elif not isinstance(source_metadata, dict):
            source_metadata = {}

        lineage.append(
            {
                "import_id": job.get("id"),
                "source_type": job.get("source_type"),
                "source_actor_id": source_metadata.get("source_actor_id"),
                "source_mission_id": source_metadata.get("source_mission_id"),
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at"),
                "context_schema_version": source_metadata.get("context_schema_version"),
                "rerun_of_import_id": source_metadata.get("rerun_of_import_id"),
                "checksum": job.get("checksum"),
            }
        )

        current = source_metadata.get("rerun_of_import_id")
        if not current:
            break

    return lineage


@app.post("/api/agents/import")
async def create_import_job(
    request: Request,
    source_type: str = Form(...),
    source_file: UploadFile = File(...),
    append_to_latest: bool = Form(default=False),
    actor_id: Optional[str] = Form(default=None),
    mission_id: Optional[str] = Form(default=None),
    context_schema_version: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    return await _process_import_upload(
        role=role,
        source_type=source_type,
        source_file=source_file,
        append_to_latest=append_to_latest,
        source_type_source="request",
        actor_id=actor_id,
        mission_id=mission_id,
        context_schema_version=context_schema_version,
    )


async def _process_import_upload(
    *,
    role: str,
    source_type: str,
    source_file: UploadFile,
    append_to_latest: bool = False,
    allow_duplicate_checksum: bool = False,
    rerun_of_import_id: Optional[str] = None,
    source_type_source: str = "request",
    actor_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    context_schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    source_type_key = _normalise_source_type(source_type)
    adapter = get_import_adapter(source_type_key)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported source_type "
                f"'{source_type}'. Supported values: openai, openai_codex, codex, gemini, "
                "twitter, gemini_cli, copilot, busy, busy_local, openclaw"
            ),
        )

    raw_data = await source_file.read()
    if not raw_data:
        raise HTTPException(status_code=400, detail="source_file is empty")

    checksum = checksum_payload(raw_data)
    try:
        parse_result = adapter.parse(_coerce_import_payload(raw_data))
    except ValueError as exc:
        _event_for(role, "import", f"Import parse failed for {source_type_key}: {exc}", "error")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        _event_for(role, "import", f"Import parse failed for {source_type_key}: {exc}", "error")
        raise HTTPException(status_code=422, detail="Unable to parse provided source payload")

    if parse_result.errors:
        _event_for(role, "import", f"Import warnings for {source_type_key}", "info")

    source_metadata = dict(parse_result.source_metadata or {})
    if actor_id and str(actor_id).strip():
        source_metadata["actor_id"] = str(actor_id).strip()
        source_metadata["source_actor_id"] = str(actor_id).strip()
    if mission_id and str(mission_id).strip():
        source_metadata["mission_id"] = str(mission_id).strip()
        source_metadata["source_mission_id"] = str(mission_id).strip()
    if context_schema_version and str(context_schema_version).strip():
        source_metadata["context_schema_version"] = str(context_schema_version).strip()

    if rerun_of_import_id:
        source_metadata["rerun_of_import_id"] = rerun_of_import_id
    if source_type_source == "rerun":
        source_metadata["source_type_source"] = "rerun"

    created = False
    if append_to_latest:
        existing_job = storage.get_latest_import_job_for_source(source_type_key)
        if existing_job:
            job = existing_job
            created = False
            storage.update_import_job_status(job["id"], "parsed")
        else:
            job, created = storage.create_import_job(
                source_type=source_type_key,
                source_metadata=source_metadata,
                checksum=checksum,
                status="parsed",
                allow_duplicate_checksum=allow_duplicate_checksum,
            )
    else:
        job, created = storage.create_import_job(
            source_type=source_type_key,
            source_metadata=source_metadata,
            checksum=checksum,
            status="parsed",
            allow_duplicate_checksum=allow_duplicate_checksum,
        )

    attempted_count = len(parse_result.items)
    if parse_result.items:
        raw_items = [
            {
                "kind": item.kind,
                "agent_scope": item.agent_scope,
                "content": item.content,
                "visibility": item.visibility,
                "source": item.source,
                "thread_id": item.thread_id,
                "message_id": item.message_id,
                "created_at": item.created_at,
                "review_state": item.review_state,
                "author_key": item.author_key,
                "metadata": item.metadata,
                "checksum": item.checksum,
            }
            for item in parse_result.items
        ]

        warnings = list(parse_result.warnings)
        items = []
        for index, raw_item in enumerate(raw_items):
            prepared, intake_warnings = _coerce_import_item_intake(
                raw_item,
                source_type=source_type_key,
                source_index=index,
            )
            items.append(prepared)
            warnings.extend(intake_warnings)
        warnings = list(dict.fromkeys([w for w in warnings if str(w).strip()]))
        if append_to_latest:
            existing_checksums = {
                stored["checksum"]
                for stored in storage.list_import_items(import_id=job["id"])
                if stored.get("checksum")
            }
            items = [item for item in items if item.get("checksum") not in existing_checksums]
            if not items:
                storage.append_import_progress_event(
                    import_id=job["id"],
                    phase="parsed",
                    details={"warning_count": len(warnings), "new_item_count": 0},
                )
                storage.update_import_job_status(job["id"], "awaiting_review")
                try:
                    storage.reconcile_agent_directory_snapshot(job["id"])
                except Exception as exc:  # pragma: no cover - defensive telemetry path
                    _event_for(
                        role,
                        "import",
                        f"Failed to reconcile directory snapshot for '{job['id']}': {exc}",
                        "error",
                    )
                    raise HTTPException(status_code=500, detail=str(exc))
                return {
                    "import_id": job["id"],
                    "job": job,
                    "created": created,
                    "counts": parse_result.counts,
                    "items": [],
                    "warnings": warnings,
                    "errors": list(parse_result.errors),
                    "redaction_hints": adapter.redaction_hints(),
                    "dedupe": {
                        "attempted": attempted_count,
                        "inserted": 0,
                        "skipped": attempted_count,
                    },
                }
            inserted = storage.add_import_items(job["id"], items)
        else:
            inserted = storage.add_import_items(job["id"], items)
    else:
        inserted = []
        warnings = list(parse_result.warnings)

    storage.update_import_job_status(job["id"], "awaiting_review")
    try:
        storage.reconcile_agent_directory_snapshot(job["id"])
    except Exception as exc:  # pragma: no cover - defensive telemetry path
        _event_for(
            role,
            "import",
            f"Failed to reconcile directory snapshot for '{job['id']}': {exc}",
            "error",
        )
        raise HTTPException(status_code=500, detail=str(exc))
    job = storage.get_import_job(job["id"])  # type: ignore[assignment]

    if warnings:
        storage.append_import_progress_event(
            import_id=job["id"],
            phase="parsed",
            details={"warning_count": len(warnings), "errors": parse_result.errors},
        )

    return {
        "import_id": job["id"],
        "job": job,
        "created": created,
        "counts": parse_result.counts,
        "items": inserted if parse_result.items else [],
        "warnings": warnings,
        "errors": list(parse_result.errors),
        "redaction_hints": adapter.redaction_hints(),
        "dedupe": {
            "attempted": attempted_count,
            "inserted": len(inserted),
            "skipped": max(0, attempted_count - len(inserted)),
        } if append_to_latest else None,
    }


@app.post("/api/agents/import/{import_id}/rerun")
async def rerun_import_job(
    request: Request,
    import_id: str,
    source_file: UploadFile = File(...),
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    prior = storage.get_import_job(import_id)
    if not prior:
        raise HTTPException(status_code=404, detail=f"import '{import_id}' not found")
    source_type = str(prior.get("source_type") or "").strip()
    if not source_type:
        raise HTTPException(status_code=400, detail="Import source type missing for rerun")

    prior_metadata = dict(prior.get("source_metadata") or {})

    return await _process_import_upload(
        role=role,
        source_type=source_type,
        source_file=source_file,
        allow_duplicate_checksum=True,
        rerun_of_import_id=import_id,
        source_type_source="rerun",
        actor_id=prior_metadata.get("source_actor_id") or prior_metadata.get("actor_id"),
        mission_id=prior_metadata.get("source_mission_id") or prior_metadata.get("mission_id"),
        context_schema_version=(
            prior_metadata.get("context_schema_version")
            or prior_metadata.get("schema_version")
            or prior_metadata.get("context_schema")
        ),
    )


def _summarize_import_items(items: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {"total": 0, "pending": 0, "approved": 0, "quarantined": 0, "rejected": 0}
    for item in items:
        summary["total"] += 1
        state = str(item.get("review_state", "pending")).lower()
        if state in summary:
            summary[state] += 1
        else:
            summary["pending"] += 1
    return summary


@app.get("/api/agents/imports")
async def list_import_jobs(
    request: Request,
    limit: int = 25,
    source_type: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    jobs = storage.list_import_jobs(limit=limit)
    if source_type:
        normalized_source = source_type.strip().lower()
        jobs = [job for job in jobs if str(job.get("source_type", "")).lower() == normalized_source]
    if status:
        normalized_status = status.strip().lower()
        jobs = [job for job in jobs if str(job.get("status", "")).lower() == normalized_status]
    enriched_jobs = []
    for job in jobs:
        import_items = storage.list_import_items(import_id=job["id"])
        job_payload = dict(job)
        job_payload["item_counts"] = _summarize_import_items(import_items)
        enriched_jobs.append(job_payload)
    return {"imports": enriched_jobs, "updated_at": _now_iso(), "role": role}


@app.get("/api/agents/import/{import_id}")
async def get_import_job(
    import_id: str,
    request: Request,
    review_state: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    job = storage.get_import_job(import_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"import '{import_id}' not found")

    items = storage.list_import_items(import_id=import_id, review_state=review_state)
    return {
        "job": job,
        "items": items,
        "updated_at": _now_iso(),
    }


@app.get("/api/agents/import/{import_id}/audit")
async def get_import_audit(
    import_id: str,
    request: Request,
) -> Dict[str, Any]:
    _require_role(request, required="viewer")
    job = storage.get_import_job(import_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"import '{import_id}' not found")

    progress_events = storage.list_import_progress_events(import_id)
    item_events = storage.list_import_item_events(import_id=import_id)
    timeline: List[Dict[str, Any]] = []
    for event in progress_events:
        timeline.append(
            {
                "created_at": event.get("created_at"),
                "event_type": "import.progress",
                "phase": (event.get("payload") or {}).get("phase") if isinstance(event.get("payload"), dict) else None,
                "details": event.get("payload") or {},
            }
        )
    for event in item_events:
        timeline.append(
            {
                "created_at": event.get("created_at"),
                "event_type": "import.item.event",
                "phase": event.get("event_type"),
                "details": {
                    "import_item_id": event.get("import_item_id"),
                    "review_state": event.get("review_state"),
                    "actor": event.get("actor"),
                    "note": event.get("note"),
                    "payload": event.get("payload") or {},
                },
            }
        )

    timeline.sort(key=lambda item: str(item.get("created_at") or ""))
    return {
        "job": job,
        "lineage": _collect_import_lineage(import_id),
        "progress_events": progress_events,
        "item_events": item_events,
        "timeline": timeline,
        "updated_at": _now_iso(),
    }


@app.post("/api/agents/import/{import_id}/decision")
async def set_import_item_decisions(
    import_id: str,
    payload: ImportDecisionRequest,
    request: Request,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    try:
        updated = storage.update_import_items_review_state(
            import_item_ids=payload.import_item_ids,
            review_state=payload.review_state,
            actor=payload.actor or role,
            note=payload.note,
            import_id=import_id,
            agent_scope=payload.agent_scope,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not updated:
        raise HTTPException(status_code=404, detail="No matching import items found")
    try:
        storage.reconcile_agent_directory_snapshot(import_id)
    except Exception as exc:  # pragma: no cover - defensive telemetry path
        _event_for(
            role,
            "import",
            f"Failed to reconcile directory snapshot for '{import_id}': {exc}",
            "error",
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "import_id": import_id,
        "updated": updated,
        "updated_count": len(updated),
        "updated_at": _now_iso(),
    }


@app.patch("/api/agents/directory/reassign")
async def reassign_directory_scopes(
    request: Request,
    payload: DirectoryScopeReassignRequest,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    if not payload.source_scope.strip() or not payload.target_scope.strip():
        raise HTTPException(status_code=400, detail="source_scope and target_scope are required")
    if payload.source_scope.strip() == payload.target_scope.strip():
        raise HTTPException(status_code=400, detail="source_scope and target_scope must differ")
    try:
        updated = storage.reassign_import_items_scope(
            source_scope=payload.source_scope,
            target_scope=payload.target_scope,
            actor=payload.actor or role,
            note=payload.note,
            import_item_ids=payload.import_item_ids,
            import_id=payload.import_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not updated:
        raise HTTPException(status_code=404, detail="No matching import items found for reassignment")

    source_scope = str(payload.source_scope).strip() or "global"
    target_scope = str(payload.target_scope).strip()
    import_ids = {item["import_id"] for item in updated if item.get("import_id")}
    for import_id in sorted(import_ids):
        try:
            storage.reconcile_agent_directory_snapshot(import_id)
        except Exception as exc:  # pragma: no cover - defensive telemetry path
            _event_for(
                role,
                "import",
                f"Failed to reconcile directory snapshot for '{import_id}': {exc}",
                "error",
            )
            raise HTTPException(status_code=500, detail=str(exc))

    return {
        "source_scope": source_scope,
        "target_scope": target_scope,
        "updated_count": len(updated),
        "updated": [{"id": item["id"], "import_id": item.get("import_id"), "agent_scope": item.get("agent_scope")} for item in updated],
        "updated_at": _now_iso(),
    }


@app.get("/api/events")
async def get_events(request: Request, limit: int = 25) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    _run_startup_plugin_debug_checks_once()
    events = storage.list_events(limit)
    return {"events": events, "updated_at": _now_iso()}


@app.websocket("/api/events/ws")
async def events_ws(ws: WebSocket) -> None:
    query_token = ws.query_params.get("token", "")
    role = _role_from_token(
        auth_token=ws.headers.get("authorization") or "",
        query_token=query_token,
    )
    if not role:
        await ws.close(code=4401)
        return
    _run_startup_plugin_debug_checks_once()

    await ws.accept()
    last_event_id = None
    try:
        await ws.send_json(
            {
                "type": "events",
                "role": _readable_role_name(role),
                "role_source": _token_source(ws.headers.get("authorization") or "", query_token),
                "events": storage.list_events(25),
                "updated_at": _now_iso(),
                "status": "stream-start",
            }
        )
        while True:
            rows = storage.list_events(25)
            current = rows[0]["id"] if rows else None
            if current != last_event_id:
                last_event_id = current
                await ws.send_json(
                    {
                        "type": "events",
                        "role": _readable_role_name(role),
                        "role_source": _token_source(ws.headers.get("authorization") or "", query_token),
                        "events": rows,
                        "updated_at": _now_iso(),
                    }
                )
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        return
    except Exception:
        await ws.close(code=1011)


@app.get("/api/memory")
async def get_memory(
    request: Request,
    scope: Optional[str] = None,
    item_type: Optional[str] = Query(default=None, alias="type"),
    item_id: Optional[str] = None,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    rows = storage.list_memory(scope=scope, item_type=item_type, item_id=item_id)
    return {"memory": rows, "updated_at": _now_iso()}


@app.post("/api/memory")
async def add_memory_entry(request: Request, payload: MemoryCreate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    memory = storage.add_memory(scope=payload.scope, memory_type=payload.type, content=payload.content)
    _event_for(role, "memory", f"Memory item added: {memory['id']}", "info")
    return {"memory": memory}


@app.get("/api/chat_history")
async def get_chat_history(
    request: Request,
    agent_id: Optional[str] = None,
    item_id: Optional[str] = None,
    chat_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    rows = storage.list_chat_history(
        agent_id=agent_id,
        item_id=item_id,
        chat_session_id=chat_session_id,
    )
    return {"chat_history": rows, "updated_at": _now_iso()}


@app.get("/api/chat_history/session/{chat_session_id}")
async def get_chat_history_by_session(
    request: Request,
    chat_session_id: str,
) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    rows = storage.list_chat_history(chat_session_id=chat_session_id)
    return {"chat_history": rows, "updated_at": _now_iso()}


@app.post("/api/chat_history")
async def add_chat_entry(request: Request, payload: ChatHistoryCreate) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    row = storage.add_chat_entry(
        agent_id=payload.agent_id,
        summary=payload.summary,
        chat_session_id=payload.chat_session_id,
    )
    _event_for(role, "chat", f"Chat history entry added for {payload.agent_id}", "info")
    return {"chat": row}


@app.get("/api/runtime/status")
async def get_runtime_status(request: Request) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    return {"runtime": runtime.get_status(), "updated_at": _now_iso()}


@app.get("/api/runtime/services")
async def list_runtime_services(request: Request) -> Dict[str, Any]:
    role = _require_role(request, required="viewer")
    return {**runtime.get_services(), "updated_at": _now_iso()}


@app.post("/api/runtime/services/{service_name}/start")
async def start_runtime_service(request: Request, service_name: str) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    result = _run_runtime_action("start", service_name, role=role)
    return {"updated_at": _now_iso(), **result}


@app.post("/api/runtime/services/{service_name}/stop")
async def stop_runtime_service(request: Request, service_name: str) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    result = _run_runtime_action("stop", service_name, role=role)
    return {"updated_at": _now_iso(), **result}


@app.post("/api/runtime/services/{service_name}/restart")
async def restart_runtime_service(request: Request, service_name: str) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    result = _run_runtime_action("restart", service_name, role=role)
    return {"updated_at": _now_iso(), **result}


@app.post("/api/mobile/pairing/issue")
async def issue_mobile_pairing_code(
    request: Request,
    payload: PairingIssueRequest,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    try:
        result = mobile_pairing.issue_pairing_code(
            actor=role,
            device_label=payload.device_label,
            authorized_room_ids=payload.authorized_room_ids,
            orchestrator_scope=payload.orchestrator_scope,
            ttl_sec=payload.ttl_sec,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        detail = getattr(exc, "message", str(exc))
        raise HTTPException(status_code=400, detail=detail) from exc
    _event_for(
        role,
        "mobile.pairing.issue",
        f"Mobile pairing code issued for {payload.device_label}",
        "info",
        {
            "device_label": payload.device_label,
            "authorized_room_ids": list(result["authorized_room_ids"]),
            "orchestrator_scope": list(result["orchestrator_scope"]),
            "expires_at": result["expires_at"],
        },
    )
    return {"pairing": result, "updated_at": _now_iso()}


@app.post("/api/mobile/pairing/exchange")
async def exchange_mobile_pairing_code(
    request: Request,
    payload: PairingExchangeRequest,
) -> Dict[str, Any]:
    try:
        result = mobile_pairing.exchange_pairing_code(
            pairing_code=payload.pairing_code,
            device_label=payload.device_label,
            expected_instance_id=payload.expected_instance_id,
            request_url=str(request.url),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        detail = getattr(exc, "message", str(exc))
        raise HTTPException(status_code=400, detail=detail) from exc
    return {"pairing": result, "updated_at": _now_iso()}


@app.get("/api/mobile/pairing/state")
async def get_mobile_pairing_state(request: Request) -> Dict[str, Any]:
    _require_role(request, required="admin")
    try:
        result = mobile_pairing.list_pairing_state()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        detail = getattr(exc, "message", str(exc))
        raise HTTPException(status_code=400, detail=detail) from exc
    return {"pairing": result, "updated_at": _now_iso()}


@app.post("/api/mobile/pairing/revoke")
async def revoke_mobile_pairing_token(
    request: Request,
    payload: PairingRevokeRequest,
) -> Dict[str, Any]:
    role = _require_role(request, required="admin")
    try:
        result = mobile_pairing.revoke_pairing_grant(
            actor=role,
            bridge_token=payload.bridge_token,
            token_id=payload.token_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        detail = getattr(exc, "message", str(exc))
        raise HTTPException(status_code=400, detail=detail) from exc
    _event_for(
        role,
        "mobile.pairing.revoke",
        f"Mobile pairing token revoked: {result['token_id']}",
        "info",
        {
            "token_id": result["token_id"],
            "instance_id": result["instance_id"],
            "revoked_at": result["revoked_at"],
        },
    )
    return {"pairing": result, "updated_at": _now_iso()}


def _run_runtime_action(action: str, service_name: str, role: str) -> Dict[str, Any]:
    action_result: RuntimeActionResult = runtime.control_service(service_name, action)
    if action_result.success:
        _event_for(role, "runtime", action_result.message, "info")
        return {
            "success": action_result.success,
            "message": action_result.message,
            "payload": action_result.payload,
        }
    return {
        "success": False,
        "message": action_result.message,
        "payload": action_result.payload,
    }


@app.get("/", include_in_schema=False)
async def serve_web_root() -> FileResponse:
    return FileResponse(_WEB_ROOT / "index.html")


@app.get("/{asset_path:path}", include_in_schema=False)
async def serve_web_asset(asset_path: str) -> FileResponse:
    if asset_path == "api" or asset_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")

    candidate = (_WEB_ROOT / asset_path).resolve()
    if candidate.is_file() and candidate.is_relative_to(_WEB_ROOT):
        return FileResponse(candidate)
    return FileResponse(_WEB_ROOT / "index.html")
