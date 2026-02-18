"""Shared contracts for context import adapters.

This module defines immutable data models and a provider-agnostic adapter
interface that all import implementations should follow before they are wired
into API endpoints in later tickets.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Tuple

ImportKind = str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def checksum_payload(payload: Mapping[str, Any] | Iterable[Mapping[str, Any]] | str | bytes) -> str:
    """Create a deterministic checksum for import dedupe and idempotency."""

    if isinstance(payload, (str, bytes)):
        raw = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        normalized = raw
    elif isinstance(payload, Mapping):
        normalized = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    elif isinstance(payload, Iterable):
        normalized = json.dumps(list(payload), sort_keys=True, separators=(",", ":"))
    else:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalImportItem:
    """A normalized import row stored by the Management UI intake pipeline."""

    kind: str
    content: str
    agent_scope: str
    visibility: str
    source: str
    thread_id: str
    message_id: str
    created_at: str
    author_key: str
    review_state: str
    metadata: Dict[str, Any]
    checksum: str


@dataclass(frozen=True)
class ImportParseResult:
    """Adapter parse output for one import payload."""

    import_id: str
    source_type: str
    source_metadata: Dict[str, Any]
    items: Tuple[CanonicalImportItem, ...]
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]
    counts: Dict[str, int]

    @property
    def total_items(self) -> int:
        return len(self.items)


class ImportSourceAdapter(ABC):
    """Contract shared by each provider-specific import adapter."""

    source_type: str = "generic"

    @abstractmethod
    def parse(self, source_payload: Mapping[str, Any] | str | bytes) -> ImportParseResult:
        """Parse raw source data into immutable canonical segments."""

    @abstractmethod
    def source_metadata(self, source_payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
        """Return stable metadata for audit and provenance tracking."""

    @abstractmethod
    def redaction_hints(self) -> Dict[str, Any]:
        """Return redaction guidance before item preview is surfaced."""
