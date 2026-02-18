"""Import adapters for context ingestion."""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .import_contract import CanonicalImportItem, ImportParseResult, ImportSourceAdapter, checksum_payload


class OpenAIManualImportAdapter(ImportSourceAdapter):
    """Adapter for manual export bundles from OpenAI-compatible providers."""

    source_type = "openai"
    _sensitive_markers: Sequence[str] = (
        "ssn",
        "social security",
        "credit card",
        "password",
        "private key",
        "api key",
        "secret",
        "sex",
        "sexual",
        "suicide",
        "depression",
        "anxiety",
        "therapy",
        "therapy",
    )

    def parse(self, source_payload: Mapping[str, Any] | str | bytes) -> ImportParseResult:
        data = self._coerce_payload(source_payload)
        source_metadata = self.source_metadata(data)
        threads = self._extract_threads(data)
        if not threads:
            raise ValueError("No importable OpenAI export threads found")

        counts: Counter[str] = Counter()
        items: List[CanonicalImportItem] = []
        warnings: List[str] = []

        for thread_index, thread in enumerate(threads):
            counts["threads"] += 1
            thread_id = str(thread.get("id") or thread.get("uuid") or f"thread-{thread_index}")
            thread_title = str(thread.get("title") or "OpenAI Thread")
            messages = list(self._extract_messages(thread))
            if not messages:
                warnings.append(f"Thread {thread_title} has no parseable messages")
                continue

            for msg_index, message in enumerate(messages):
                content = self._message_text(message)
                if not content.strip():
                    continue
                created_at = self._message_created_at(message) or datetime.now(timezone.utc).isoformat()
                sensitivity = self._sensitive_signals(content)
                is_sensitive = bool(sensitivity)
                author = str(self._message_author(message) or "unknown")
                item = CanonicalImportItem(
                    kind="memory",
                    content=content.strip(),
                    agent_scope=f"openai:{thread_id}",
                    visibility="quarantined" if is_sensitive else "pending",
                    source="openai_manual_upload",
                    thread_id=thread_id,
                    message_id=str(message.get("id") or message.get("uuid") or f"msg-{msg_index}"),
                    created_at=created_at,
                    author_key=author,
                    review_state="quarantined" if is_sensitive else "pending",
                    metadata={
                        "title": thread_title,
                        "thread_index": thread_index,
                        "message_index": msg_index,
                        "sensitive_flags": sensitivity,
                        "requires_review": is_sensitive,
                        "source_adapter": self.source_type,
                    },
                    checksum="",
                )
                items.append(
                    CanonicalImportItem(
                        kind=item.kind,
                        content=item.content,
                        agent_scope=item.agent_scope,
                        visibility=item.visibility,
                        source=item.source,
                        thread_id=item.thread_id,
                        message_id=item.message_id,
                        created_at=item.created_at,
                        author_key=item.author_key,
                        review_state=item.review_state,
                        metadata=item.metadata,
                        checksum=checksum_payload(item.__dict__),
                    )
                )
                counts["messages"] += 1

        if not items:
            errors = tuple(["No parseable message records were found"])
            return ImportParseResult(
                import_id=checksum_payload(source_payload),
                source_type=self.source_type,
                source_metadata=source_metadata,
                items=tuple(),
                warnings=tuple(warnings),
                errors=errors,
                counts=dict(counts),
            )

        return ImportParseResult(
            import_id=checksum_payload(source_payload),
            source_type=self.source_type,
            source_metadata=source_metadata,
            items=tuple(items),
            warnings=tuple(warnings),
            errors=tuple(),
            counts=dict(counts),
        )

    def source_metadata(self, source_payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
        data = self._coerce_payload(source_payload)
        threads = list(self._extract_threads(data))
        return {
            "source_type": self.source_type,
            "thread_count": len(threads),
            "adapter": self.__class__.__name__,
        }

    def redaction_hints(self) -> Dict[str, Any]:
        return {
            "redact_fields": ["author_key", "metadata.sensitive_flags"],
            "default_review_state": "pending",
            "default_visibility": "quarantined",
        }

    def _coerce_payload(self, source_payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
        if isinstance(source_payload, bytes):
            raw = source_payload.decode("utf-8", errors="replace")
            return json.loads(raw)
        if isinstance(source_payload, str):
            return json.loads(source_payload)
        return dict(source_payload)

    def _extract_threads(self, data: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("conversations", "threads", "sessions"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        return []

    def _extract_messages(self, thread: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
        if isinstance(thread.get("messages"), list):
            return [item for item in thread["messages"] if isinstance(item, dict)]

        mapping = thread.get("mapping")
        if isinstance(mapping, dict):
            messages = []
            for node in mapping.values():
                if not isinstance(node, dict):
                    continue
                message = node.get("message")
                if isinstance(message, dict):
                    messages.append(message)
            return sorted(messages, key=self._message_sort_key)

        if isinstance(thread.get("data"), dict):
            messages = thread["data"].get("messages")
            if isinstance(messages, list):
                return [item for item in messages if isinstance(item, dict)]
        return []

    def _message_text(self, message: Mapping[str, Any]) -> str:
        content = message.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if isinstance(content.get("parts"), list):
                parts = [self._coerce_text(p) for p in content.get("parts", []) if self._coerce_text(p)]
                return "\n".join(parts).strip()
            if isinstance(content.get("text"), str):
                return content["text"]
        return self._coerce_text(content)

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join([self._coerce_text(v) for v in value]).strip()
        if isinstance(value, dict):
            if "text" in value and isinstance(value["text"], str):
                return value["text"]
        return ""

    def _message_author(self, message: Mapping[str, Any]) -> str:
        author = message.get("author")
        if isinstance(author, str):
            return author
        if isinstance(author, dict):
            return str(author.get("role") or author.get("name") or "unknown")
        role = message.get("role")
        return str(role) if role is not None else "unknown"

    def _message_created_at(self, message: Mapping[str, Any]) -> str:
        created = message.get("create_time") or message.get("created_at") or message.get("timestamp")
        if created is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(created, (int, float)):
            return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
        if isinstance(created, str):
            return created
        return datetime.now(timezone.utc).isoformat()

    def _message_sort_key(self, message: Mapping[str, Any]) -> str:
        created = self._message_created_at(message)
        return created

    def _sensitive_signals(self, text: str) -> List[str]:
        lowered = text.lower()
        return [marker for marker in self._sensitive_markers if marker in lowered]


class OpenAICodexManualImportAdapter(OpenAIManualImportAdapter):
    source_type = "openai_codex"

    def redaction_hints(self) -> Dict[str, Any]:
        hints = super().redaction_hints()
        hints["source_type"] = "openai_codex"
        return hints


class CodexManualImportAdapter(OpenAICodexManualImportAdapter):
    """Alias adapter for developer-local Codex exports."""

    source_type = "codex"

    def redaction_hints(self) -> Dict[str, Any]:
        hints = super().redaction_hints()
        hints["source_type"] = "codex"
        return hints


class GeminiManualImportAdapter(OpenAIManualImportAdapter):
    """Alias adapter for Gemini chat/thread style exports."""

    source_type = "gemini"

    def redaction_hints(self) -> Dict[str, Any]:
        hints = super().redaction_hints()
        hints["source_type"] = "gemini"
        return hints


class GeminiCLIManualImportAdapter(OpenAIManualImportAdapter):
    """Adapter for Gemini CLI export bundles and local snapshots."""

    source_type = "gemini_cli"
    _sensitive_markers: Sequence[str] = (
        "ssn",
        "social security",
        "credit card",
        "password",
        "private key",
        "api key",
        "secret",
        "sex",
        "sexual",
        "suicide",
        "depression",
        "anxiety",
        "therapy",
    )

    def parse(self, source_payload: Mapping[str, Any] | str | bytes) -> ImportParseResult:
        data = self._coerce_payload(source_payload)
        source_metadata = self.source_metadata(data)
        entries = list(self._extract_entries(data))
        if not entries:
            raise ValueError("No importable Gemini CLI records found")

        counts: Counter[str] = Counter()
        items: List[CanonicalImportItem] = []
        warnings: List[str] = []

        for entry_index, entry in enumerate(entries):
            message, source_file, conversation_id, title = entry
            content = self._build_message_content(message, title)
            if not content.strip():
                warnings.append(f"Skipping empty Gemini CLI message from {conversation_id}")
                continue

            created_at = self._coerce_timestamp(
                message.get("created_at")
                or message.get("createdAt")
                or message.get("timestamp")
                or message.get("time")
            )
            sensitivity = self._sensitive_signals(content)
            is_sensitive = bool(sensitivity)
            message_id = str(
                message.get("id")
                or message.get("uuid")
                or message.get("message_id")
                or f"gemini-cli-{entry_index}"
            )
            scope_owner = self._coerce_text(message.get("path") or message.get("session") or conversation_id)
            item_scope = f"gemini_cli:{scope_owner}" if scope_owner else "gemini_cli:import"
            metadata = self._build_metadata(
                message=message,
                record_type="message",
                source_file=source_file,
                thread_title=title,
                thread_id=conversation_id,
            )
            author_key = self._coerce_author(message)
            item = CanonicalImportItem(
                kind="memory",
                content=content.strip(),
                agent_scope=item_scope,
                visibility="quarantined" if is_sensitive else "pending",
                source="gemini_cli_manual_upload",
                thread_id=conversation_id,
                message_id=message_id,
                created_at=created_at,
                author_key=author_key,
                review_state="quarantined" if is_sensitive else "pending",
                metadata=metadata,
                checksum="",
            )
            items.append(
                CanonicalImportItem(
                    kind=item.kind,
                    content=item.content,
                    agent_scope=item.agent_scope,
                    visibility=item.visibility,
                    source=item.source,
                    thread_id=item.thread_id,
                    message_id=item.message_id,
                    created_at=item.created_at,
                    author_key=item.author_key,
                    review_state=item.review_state,
                    metadata=item.metadata,
                    checksum=checksum_payload(item.__dict__),
                )
            )
            counts["messages"] += 1

        if not items:
            errors = tuple(["No parseable Gemini CLI messages were found"])
            return ImportParseResult(
                import_id=checksum_payload(source_payload),
                source_type=self.source_type,
                source_metadata=source_metadata,
                items=tuple(),
                warnings=tuple(warnings),
                errors=errors,
                counts=dict(counts),
            )

        return ImportParseResult(
            import_id=checksum_payload(source_payload),
            source_type=self.source_type,
            source_metadata=source_metadata,
            items=tuple(items),
            warnings=tuple(warnings),
            errors=tuple(),
            counts=dict(counts),
        )

    def source_metadata(self, source_payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
        payload = self._coerce_payload(source_payload)
        entries = list(self._extract_entries(payload))
        if not entries:
            return {
                "source_type": self.source_type,
                "record_count": 0,
                "adapter": self.__class__.__name__,
            }
        counts = Counter(
            entry[3].strip() or "Gemini CLI Session" for entry in entries
        )
        return {
            "source_type": self.source_type,
            "record_count": len(entries),
            "record_types": dict(Counter((count for count in counts.keys()))),
            "adapter": self.__class__.__name__,
            "files": len(payload.get("files")) if isinstance(payload, dict) and isinstance(payload.get("files"), list) else 0,
        }

    def redaction_hints(self) -> Dict[str, Any]:
        return {
            "redact_fields": ["author_key", "metadata.sensitive_flags"],
            "default_review_state": "pending",
            "default_visibility": "quarantined",
        }

    def _coerce_payload(self, source_payload: Mapping[str, Any] | str | bytes) -> Mapping[str, Any]:
        if isinstance(source_payload, bytes):
            if zipfile.is_zipfile(io.BytesIO(source_payload)):
                return self._extract_archive_payload(source_payload)
            raw = source_payload.decode("utf-8", errors="replace")
            return self._coerce_payload(raw)
        if isinstance(source_payload, str):
            return self._coerce_wrapped_json(source_payload)
        return dict(source_payload)

    def _coerce_wrapped_json(self, raw: str) -> Mapping[str, Any]:
        payload_text = raw.strip()
        if not payload_text:
            return {}
        if payload_text.startswith("window.") and "=" in payload_text:
            try:
                payload_text = payload_text[payload_text.find("=") + 1 :].strip().rstrip(";")
            except Exception:
                pass
        return self._normalise_payload(json.loads(payload_text))

    def _normalise_payload(self, payload: Any) -> Mapping[str, Any]:
        if isinstance(payload, list):
            return {"conversations": payload}
        if isinstance(payload, dict):
            return dict(payload)
        return {"records": []}

    def _extract_archive_payload(self, archive_bytes: bytes) -> Dict[str, Any]:
        extracted: List[Tuple[str, Mapping[str, Any]]] = []
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zipped:
            for name in sorted(zipped.namelist()):
                lower_name = name.lower()
                if not (lower_name.endswith(".json") or lower_name.endswith(".jsonl") or lower_name.endswith(".js")):
                    continue
                try:
                    raw = zipped.read(name).decode("utf-8", errors="replace")
                except Exception:
                    continue
                try:
                    payload = self._coerce_wrapped_json(raw)
                except Exception:
                    continue
                extracted.append((name, payload))
        return {"files": extracted}

    def _extract_entries(
        self,
        payload: Mapping[str, Any],
    ) -> List[Tuple[Dict[str, Any], Optional[str], str, str]]:
        if not payload:
            return []

        if "files" in payload and isinstance(payload["files"], list):
            entries: List[Tuple[Dict[str, Any], Optional[str], str, str]] = []
            for file_name, file_payload in payload["files"]:
                if isinstance(file_payload, Mapping):
                    nested_payload = dict(file_payload)
                    nested_payload["__source_file"] = file_name
                    entries.extend(self._extract_entries(nested_payload))
                elif isinstance(file_payload, list):
                    for item in file_payload:
                        if isinstance(item, Mapping):
                            entries.extend(
                                self._extract_entries({"__source_file": file_name, "records": [item]})
                            )
            return entries

        container = payload.get("data", payload)
        source_file = payload.get("__source_file")

        if isinstance(container, list):
            out: List[Tuple[Dict[str, Any], Optional[str], str, str]] = []
            for item in container:
                if isinstance(item, Mapping):
                    out.extend(self._extract_entries({"record": item, "__source_file": source_file}))
            return out

        if not isinstance(container, Mapping):
            return []

        out: List[Tuple[Dict[str, Any], Optional[str], str, str]] = []
        for key, title_key in (("conversations", "title"), ("sessions", "name"), ("history", "title"), ("transcripts", "title"), ("runs", "name")):
            records = container.get(key)
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, Mapping):
                        out.extend(self._extract_conversation_messages(record, source_file, title_key))
            elif isinstance(records, Mapping):
                out.extend(self._extract_conversation_messages(records, source_file, title_key))

        if "record" in container and isinstance(container["record"], Mapping):
            out.extend(self._extract_conversation_messages(container["record"], source_file, "title"))
        elif "messages" in container and isinstance(container.get("messages"), list):
            conversation_id = str(
                container.get("id")
                or container.get("session_id")
                or container.get("session")
                or "gemini-session"
            )
            title = self._coerce_text(
                container.get("title")
                or container.get("name")
                or container.get("session")
                or conversation_id
            )
            for message in container["messages"]:
                if isinstance(message, Mapping):
                    out.append((dict(message), source_file, conversation_id, title))
        return out

    def _extract_conversation_messages(
        self,
        conversation: Mapping[str, Any],
        source_file: Optional[str],
        title_key: str,
    ) -> List[Tuple[Dict[str, Any], Optional[str], str, str]]:
        if not isinstance(conversation, Mapping):
            return []

        conversation_id = str(
            conversation.get("id")
            or conversation.get("session_id")
            or conversation.get("conversation_id")
            or conversation.get("uuid")
            or "gemini-session"
        )
        title = self._coerce_text(conversation.get(title_key) or conversation.get("title") or conversation.get("name") or conversation_id)
        raw_messages = (
            conversation.get("messages")
            or conversation.get("turns")
            or conversation.get("entries")
            or conversation.get("history")
            or []
        )
        if not isinstance(raw_messages, list):
            return []

        out: List[Tuple[Dict[str, Any], Optional[str], str, str]] = []
        for message in sorted(
            raw_messages,
            key=lambda item: self._coerce_timestamp(
                item.get("created_at") if isinstance(item, Mapping) else None
            ),
        ):
            if isinstance(message, Mapping):
                out.append((dict(message), source_file, conversation_id, title))
        return out

    def _build_message_content(self, message: Mapping[str, Any], fallback_title: str) -> str:
        parts: List[str] = []
        fallback = self._coerce_text(
            message.get("content")
            or message.get("text")
            or message.get("message")
            or message.get("body")
        )
        body_parts = self._coerce_segments(message.get("parts") or message.get("content_parts"))
        if body_parts:
            parts.extend(body_parts)
        elif fallback:
            parts.append(fallback)

        references = message.get("references") or message.get("sources") or message.get("citations")
        rendered_refs = self._render_references(references)
        if rendered_refs:
            parts.append(f"\n\nReferences:\n{rendered_refs}")

        body = "\n\n".join([part for part in parts if part]).strip()
        if not body and fallback_title:
            return f"Gemini message from {fallback_title}"
        return body

    def _coerce_segments(self, segments: Any) -> List[str]:
        if segments is None:
            return []
        if isinstance(segments, str):
            return [segments]
        if not isinstance(segments, list):
            return [self._coerce_text(segments)]

        out: List[str] = []
        for segment in segments:
            if isinstance(segment, str):
                if segment.strip():
                    out.append(segment.strip())
                continue
            if not isinstance(segment, Mapping):
                continue
            if segment.get("text"):
                text_value = self._coerce_text(segment.get("text"))
                if text_value:
                    out.append(text_value)
            if segment.get("content"):
                content_value = self._coerce_text(segment.get("content"))
                if content_value:
                    out.append(content_value)
            if segment.get("code") or segment.get("snippet"):
                code = self._coerce_text(segment.get("code") or segment.get("snippet"))
                if code:
                    language = segment.get("language") or segment.get("lang") or ""
                    out.append(f"```{language}\n{code}\n```")
            if segment.get("type") in {"code", "code_block", "codeblock"}:
                code = self._coerce_text(segment.get("value") or segment.get("text") or segment.get("code"))
                if code:
                    language = segment.get("language") or segment.get("lang") or ""
                    out.append(f"```{language}\n{code}\n```")
        return out

    def _render_references(self, raw_references: Any) -> str:
        references: List[str] = []
        if raw_references is None:
            return ""
        if isinstance(raw_references, str):
            return raw_references.strip()
        if isinstance(raw_references, list):
            for reference in raw_references:
                if isinstance(reference, str):
                    references.append(reference.strip())
                elif isinstance(reference, Mapping):
                    reference_value = (
                        reference.get("url")
                        or reference.get("href")
                        or reference.get("uri")
                        or reference.get("source")
                    )
                    if reference_value:
                        references.append(str(reference_value).strip())
        elif isinstance(raw_references, Mapping):
            for value in raw_references.values():
                if isinstance(value, str):
                    references.append(value.strip())
                elif isinstance(value, (list, tuple)):
                    for nested in value:
                        if isinstance(nested, str):
                            references.append(nested.strip())
        return "\n".join([f"- {item}" for item in references if item])

    def _coerce_author(self, message: Mapping[str, Any]) -> str:
        for key in ("author", "role", "sender", "actor", "user", "source"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, Mapping):
                candidate = (
                    value.get("name")
                    or value.get("username")
                    or value.get("handle")
                    or value.get("role")
                )
                if isinstance(candidate, str) and candidate.strip():
                    return str(candidate).strip()
        return "gemini-cli-user"

    def _build_metadata(
        self,
        message: Mapping[str, Any],
        record_type: str,
        source_file: Optional[str],
        thread_title: str,
        thread_id: str,
    ) -> Dict[str, Any]:
        flags = self._sensitive_signals(self._build_message_content(message, thread_title))
        return {
            "record_type": record_type,
            "source_adapter": self.source_type,
            "source_file": source_file,
            "thread_id": thread_id,
            "thread_title": thread_title,
            "requires_review": bool(flags),
            "sensitive_flags": flags,
            "references": message.get("references"),
            "created_at": message.get("created_at") or message.get("timestamp"),
        }

    def _coerce_timestamp(self, raw: Any) -> str:
        if raw is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(raw, (int, float)):
            if raw > 10**15:
                return datetime.fromtimestamp(raw / 1_000_000, tz=timezone.utc).isoformat()
            if raw > 10**12:
                return datetime.fromtimestamp(raw / 1000, tz=timezone.utc).isoformat()
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
        if isinstance(raw, str):
            candidate = raw.strip()
            if not candidate:
                return datetime.now(timezone.utc).isoformat()
            try:
                if candidate.endswith("Z"):
                    return datetime.fromisoformat(candidate.replace("Z", "+00:00")).isoformat()
                return datetime.fromisoformat(candidate).isoformat()
            except Exception:
                return candidate
        return str(raw)

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return " ".join([self._coerce_text(v) for v in value]).strip()
        if isinstance(value, Mapping):
            if "text" in value and isinstance(value["text"], str):
                return value["text"].strip()
        return ""

    def _sensitive_signals(self, text: str) -> List[str]:
        lowered = text.lower()
        return [marker for marker in self._sensitive_markers if marker in lowered]


class CopilotManualImportAdapter(ImportSourceAdapter):
    """Adapter for Copilot / VS Code style conversation exports."""

    source_type = "copilot"
    _sensitive_markers: Sequence[str] = (
        "ssn",
        "social security",
        "credit card",
        "password",
        "private key",
        "api key",
        "secret",
        "sex",
        "sexual",
        "suicide",
        "depression",
        "anxiety",
        "therapy",
    )

    def parse(self, source_payload: Mapping[str, Any] | str | bytes) -> ImportParseResult:
        data = self._coerce_payload(source_payload)
        source_metadata = self.source_metadata(data)
        records = list(self._extract_records(data))
        if not records:
            raise ValueError("No importable Copilot records found")

        counts: Counter[str] = Counter()
        items: List[CanonicalImportItem] = []
        warnings: List[str] = []

        for record_index, record in enumerate(records):
            message, source_file, conversation_id, conversation_title = record
            content = self._coerce_text(
                message.get("content")
                or message.get("text")
                or message.get("message")
                or message.get("body")
            )
            if not content.strip():
                warnings.append(f"Skipping empty Copilot message in {conversation_id}")
                continue

            created_at = self._coerce_timestamp(
                message.get("created_at")
                or message.get("createdAt")
                or message.get("timestamp")
                or message.get("time")
            )
            sensitivity = self._sensitive_signals(content)
            is_sensitive = bool(sensitivity)
            role = self._coerce_author_role(message)
            message_id = str(
                message.get("id")
                or message.get("message_id")
                or message.get("uuid")
                or f"copilot-msg-{record_index}"
            )
            metadata = self._build_metadata(message, "message", source_file, conversation_id, conversation_title)
            item = CanonicalImportItem(
                kind="memory",
                content=self._render_message_block(role, content, conversation_title),
                agent_scope=f"copilot:{conversation_id}",
                visibility="quarantined" if is_sensitive else "pending",
                source="copilot_manual_upload",
                thread_id=conversation_id,
                message_id=message_id,
                created_at=created_at,
                author_key=role,
                review_state="quarantined" if is_sensitive else "pending",
                metadata=metadata,
                checksum="",
            )
            items.append(
                CanonicalImportItem(
                    kind=item.kind,
                    content=item.content,
                    agent_scope=item.agent_scope,
                    visibility=item.visibility,
                    source=item.source,
                    thread_id=item.thread_id,
                    message_id=item.message_id,
                    created_at=item.created_at,
                    author_key=item.author_key,
                    review_state=item.review_state,
                    metadata=item.metadata,
                    checksum=checksum_payload(item.__dict__),
                )
            )
            counts["messages"] += 1

        if not items:
            errors = tuple(["No parseable Copilot messages were found"])
            return ImportParseResult(
                import_id=checksum_payload(source_payload),
                source_type=self.source_type,
                source_metadata=source_metadata,
                items=tuple(),
                warnings=tuple(warnings),
                errors=errors,
                counts=dict(counts),
            )

        return ImportParseResult(
            import_id=checksum_payload(source_payload),
            source_type=self.source_type,
            source_metadata=source_metadata,
            items=tuple(items),
            warnings=tuple(warnings),
            errors=tuple(),
            counts=dict(counts),
        )

    def source_metadata(self, source_payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
        data = self._coerce_payload(source_payload)
        records = list(self._extract_records(data))
        return {
            "source_type": self.source_type,
            "record_count": len(records),
            "adapter": self.__class__.__name__,
            "files": len(data.get("files")) if isinstance(data, dict) and isinstance(data.get("files"), list) else 0,
        }

    def redaction_hints(self) -> Dict[str, Any]:
        return {
            "redact_fields": ["author_key", "metadata.sensitive_flags"],
            "default_review_state": "pending",
            "default_visibility": "quarantined",
        }

    def _coerce_payload(self, source_payload: Mapping[str, Any] | str | bytes) -> Mapping[str, Any]:
        if isinstance(source_payload, bytes):
            if zipfile.is_zipfile(io.BytesIO(source_payload)):
                return self._extract_archive_payload(source_payload)
            raw = source_payload.decode("utf-8", errors="replace")
            return self._coerce_payload(raw)
        if isinstance(source_payload, str):
            payload_text = source_payload.strip()
            if not payload_text:
                return {}
            try:
                payload = json.loads(payload_text)
            except Exception:
                lines = [line.strip() for line in payload_text.splitlines() if line.strip()]
                parsed: List[Dict[str, Any]] = []
                for line in lines:
                    try:
                        loaded = json.loads(line)
                        if isinstance(loaded, Mapping):
                            parsed.append(dict(loaded))
                    except Exception:
                        pass
                if parsed:
                    return {"records": parsed}
                raise
            if isinstance(payload, list):
                return {"records": payload}
            if isinstance(payload, dict):
                return dict(payload)
            return {"records": []}
        return dict(source_payload)

    def _extract_archive_payload(self, archive_bytes: bytes) -> Dict[str, Any]:
        extracted: List[Tuple[str, Mapping[str, Any]]] = []
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zipped:
            for name in sorted(zipped.namelist()):
                lower_name = name.lower()
                if lower_name.endswith((".json", ".jsonl", ".js")):
                    try:
                        raw = zipped.read(name).decode("utf-8", errors="replace")
                        parsed = self._coerce_payload(raw)
                    except Exception:
                        continue
                    extracted.append((name, parsed))
                elif lower_name.endswith(".md") or lower_name.endswith(".txt"):
                    extracted.append((name, {"kind": "text_dump", "path": name, "content": zipped.read(name).decode("utf-8", errors="replace")}))
        return {"files": extracted}

    def _extract_records(self, payload: Mapping[str, Any]) -> List[Tuple[Dict[str, Any], Optional[str], str, str]]:
        if not payload:
            return []
        if "files" in payload and isinstance(payload["files"], list):
            out: List[Tuple[Dict[str, Any], Optional[str], str, str]] = []
            for file_name, file_payload in payload["files"]:
                if isinstance(file_payload, Mapping):
                    container = dict(file_payload)
                    container["__source_file"] = file_name
                    out.extend(self._extract_records(container))
                elif isinstance(file_payload, list):
                    for item in file_payload:
                        if isinstance(item, Mapping):
                            out.extend(self._extract_records({"record": item, "__source_file": file_name}))
            return out

        source_file = payload.get("__source_file")
        container = payload.get("data", payload)
        if isinstance(container, list):
            out: List[Tuple[Dict[str, Any], Optional[str], str, str]] = []
            for item in container:
                if isinstance(item, Mapping):
                    out.extend(self._extract_records({"record": item, "__source_file": source_file}))
            return out
        if not isinstance(container, Mapping):
            return []

        records: List[Tuple[Dict[str, Any], Optional[str], str, str]] = []
        for collection_key in (
            "records",
            "threads",
            "conversations",
            "sessions",
            "chats",
            "messages",
        ):
            values = container.get(collection_key)
            if isinstance(values, Mapping):
                records.extend(self._extract_records({"record": values, "__source_file": source_file}))
            elif isinstance(values, list):
                for item in values:
                    if isinstance(item, Mapping):
                        records.extend(self._extract_records({"record": item, "__source_file": source_file}))

        if "record" in container and isinstance(container["record"], Mapping):
            record = dict(container["record"])
            conversation_id = str(
                record.get("conversation_id")
                or record.get("thread_id")
                or record.get("id")
                or "copilot-conversation"
            )
            conversation_title = str(record.get("title") or record.get("name") or conversation_id)
            messages = record.get("messages") or record.get("turns") or record.get("items") or []
            if isinstance(messages, Mapping):
                messages = list(messages.values())
            if isinstance(messages, list):
                for message in sorted(messages, key=lambda item: self._coerce_timestamp(item.get("created_at") if isinstance(item, Mapping) else None)):
                    if isinstance(message, Mapping):
                        records.append((dict(message), source_file, conversation_id, conversation_title))
            elif any(k in record for k in ("content", "text", "message", "body")):
                records.append((record, source_file, conversation_id, conversation_title))
        return records

    def _build_metadata(self, message: Mapping[str, Any], record_type: str, source_file: Optional[str], conversation_id: str, conversation_title: str) -> Dict[str, Any]:
        flags = self._sensitive_signals(self._coerce_text(message.get("content") or message.get("text") or message.get("message") or message.get("body")))
        role = self._coerce_author_role(message)
        return {
            "record_type": record_type,
            "source_adapter": self.source_type,
            "source_file": source_file,
            "conversation_id": conversation_id,
            "conversation_title": conversation_title,
            "requires_review": bool(flags),
            "sensitive_flags": flags,
            "author_role": role,
            "created_at": message.get("created_at") or message.get("timestamp"),
        }

    def _render_message_block(self, role: str, content: str, conversation_title: str) -> str:
        if conversation_title:
            return f"# {conversation_title}\n\n## {role}\n{content}"
        return f"## {role}\n{content}"

    def _coerce_author_role(self, message: Mapping[str, Any]) -> str:
        role_value = (
            message.get("role")
            or message.get("speaker")
            or message.get("sender")
            or message.get("author")
            or message.get("user")
        )
        if not isinstance(role_value, str):
            return "unknown"
        role = role_value.strip().lower()
        if role in {"user", "human", "me", "requester"}:
            return "user"
        if role in {"assistant", "copilot", "agent", "system", "ai", "assistant_agent"}:
            return "assistant"
        return role

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join([self._coerce_text(item) for item in value]).strip()
        if isinstance(value, Mapping):
            if "text" in value and isinstance(value["text"], str):
                return value["text"]
            if "content" in value and isinstance(value["content"], str):
                return value["content"]
        return ""

    def _coerce_timestamp(self, raw: Any) -> str:
        if raw is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(raw, (int, float)):
            if raw > 10**15:
                return datetime.fromtimestamp(raw / 1_000_000, tz=timezone.utc).isoformat()
            if raw > 10**12:
                return datetime.fromtimestamp(raw / 1000, tz=timezone.utc).isoformat()
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
        if isinstance(raw, str):
            candidate = raw.strip()
            if not candidate:
                return datetime.now(timezone.utc).isoformat()
            try:
                if candidate.endswith("Z"):
                    return datetime.fromisoformat(candidate.replace("Z", "+00:00")).isoformat()
                return datetime.fromisoformat(candidate).isoformat()
            except Exception:
                return candidate
        return str(raw)

    def _sensitive_signals(self, text: str) -> List[str]:
        lowered = text.lower()
        return [marker for marker in self._sensitive_markers if marker in lowered]


class BusyLocalImportAdapter(ImportSourceAdapter):
    """Adapter for Busy local project export snapshots."""

    source_type = "busy_local"
    _sensitive_markers: Sequence[str] = (
        "ssn",
        "social security",
        "credit card",
        "password",
        "private key",
        "api key",
        "secret",
        "sex",
        "sexual",
        "suicide",
        "depression",
        "anxiety",
        "therapy",
    )

    def parse(self, source_payload: Mapping[str, Any] | str | bytes) -> ImportParseResult:
        data = self._coerce_payload(source_payload)
        source_metadata = self.source_metadata(data)
        records = list(self._extract_records(data))
        if not records:
            raise ValueError("No importable Busy local export records found")

        counts: Counter[str] = Counter()
        items: List[CanonicalImportItem] = []
        warnings: List[str] = []
        history_found = False

        for record_index, record in enumerate(records):
            kind, payload, source_file, parent_id, parent_title = record
            content = self._coerce_content(kind, payload, parent_title)
            if not content.strip():
                warnings.append(f"Skipping empty {kind} record from {parent_id}")
                continue

            if kind == "history":
                history_found = True

            created_at = self._coerce_timestamp(
                payload.get("created_at")
                or payload.get("createdAt")
                or payload.get("timestamp")
                or payload.get("time")
            )
            sensitivity = self._sensitive_signals(content)
            is_sensitive = bool(sensitivity)
            item_id = str(
                payload.get("id")
                or payload.get("uuid")
                or payload.get("agent_id")
                or payload.get("provider_id")
                or f"{kind}-{record_index}"
            )
            thread_id = str(parent_id or payload.get("thread_id") or payload.get("id") or f"busy-{record_index}")
            metadata = self._build_metadata(kind, payload, source_file, parent_id, parent_title, sensitivity)
            item = CanonicalImportItem(
                kind="provider" if kind == "provider" else "agent_profile" if kind == "agent" else "memory",
                content=content,
                agent_scope=f"busy:{thread_id}",
                visibility="quarantined" if is_sensitive else "pending",
                source="busy_local_export_upload",
                thread_id=thread_id,
                message_id=item_id,
                created_at=created_at,
                author_key=self._coerce_author(payload),
                review_state="quarantined" if is_sensitive else "pending",
                metadata=metadata,
                checksum="",
            )
            items.append(
                CanonicalImportItem(
                    kind=item.kind,
                    content=item.content,
                    agent_scope=item.agent_scope,
                    visibility=item.visibility,
                    source=item.source,
                    thread_id=item.thread_id,
                    message_id=item.message_id,
                    created_at=item.created_at,
                    author_key=item.author_key,
                    review_state=item.review_state,
                    metadata=item.metadata,
                    checksum=checksum_payload(item.__dict__),
                )
            )
            counts[kind] += 1

        if not history_found:
            warnings.append(
                "No conversation history was found in this Busy local export; imported onboarding and provider metadata only."
            )

        if not items:
            errors = tuple(["No parseable Busy local export records were found"])
            return ImportParseResult(
                import_id=checksum_payload(source_payload),
                source_type=self.source_type,
                source_metadata=source_metadata,
                items=tuple(),
                warnings=tuple(warnings),
                errors=errors,
                counts=dict(counts),
            )

        return ImportParseResult(
            import_id=checksum_payload(source_payload),
            source_type=self.source_type,
            source_metadata=source_metadata,
            items=tuple(items),
            warnings=tuple(warnings),
            errors=tuple(),
            counts=dict(counts),
        )

    def source_metadata(self, source_payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
        data = self._coerce_payload(source_payload)
        records = list(self._extract_records(data))
        history_count = len([record for record in records if record[0] == "history"])
        return {
            "source_type": self.source_type,
            "record_count": len(records),
            "provider_count": len([record for record in records if record[0] == "provider"]),
            "agent_count": len([record for record in records if record[0] == "agent"]),
            "history_count": history_count,
            "adapter": self.__class__.__name__,
            "files": len(data.get("files")) if isinstance(data, dict) and isinstance(data.get("files"), list) else 0,
        }

    def redaction_hints(self) -> Dict[str, Any]:
        return {
            "redact_fields": ["author_key", "metadata.sensitive_flags"],
            "default_review_state": "pending",
            "default_visibility": "quarantined",
        }

    def _coerce_payload(self, source_payload: Mapping[str, Any] | str | bytes) -> Mapping[str, Any]:
        if isinstance(source_payload, bytes):
            if zipfile.is_zipfile(io.BytesIO(source_payload)):
                return self._extract_archive_payload(source_payload)
            raw = source_payload.decode("utf-8", errors="replace")
            return self._coerce_payload(raw)
        if isinstance(source_payload, str):
            payload_text = source_payload.strip()
            if not payload_text:
                return {}
            payload = json.loads(payload_text)
            if isinstance(payload, list):
                return {"records": payload}
            return dict(payload)
        return dict(source_payload)

    def _extract_archive_payload(self, archive_bytes: bytes) -> Dict[str, Any]:
        extracted: List[Tuple[str, Mapping[str, Any]]] = []
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zipped:
            for name in sorted(zipped.namelist()):
                lower_name = name.lower()
                if lower_name.endswith((".json", ".yaml", ".yml")):
                    try:
                        raw = zipped.read(name).decode("utf-8", errors="replace")
                        parsed = self._coerce_payload(raw)
                    except Exception:
                        continue
                    extracted.append((name, parsed))
                elif lower_name.endswith((".txt", ".md", ".markdown")):
                    try:
                        raw = zipped.read(name).decode("utf-8", errors="replace").strip()
                    except Exception:
                        continue
                    if raw:
                        extracted.append((name, {"kind": "text_dump", "path": name, "content": raw}))
                elif lower_name.endswith(".csv"):
                    extracted.append((name, {"kind": "unsupported", "path": name}))
        return {"files": extracted}

    def _extract_records(
        self,
        payload: Mapping[str, Any],
    ) -> List[Tuple[str, Dict[str, Any], Optional[str], str, str]]:
        if not payload:
            return []
        if "files" in payload and isinstance(payload["files"], list):
            out: List[Tuple[str, Dict[str, Any], Optional[str], str, str]] = []
            for file_name, file_payload in payload["files"]:
                if isinstance(file_payload, Mapping):
                    nested = dict(file_payload)
                    nested["__source_file"] = file_name
                    out.extend(self._extract_records(nested))
                elif isinstance(file_payload, list):
                    for item in file_payload:
                        if isinstance(item, Mapping):
                            out.extend(self._extract_records({"record": item, "__source_file": file_name}))
            return out

        container = payload.get("data", payload)
        source_file = payload.get("__source_file")
        if isinstance(container, list):
            out: List[Tuple[str, Dict[str, Any], Optional[str], str, str]] = []
            for item in container:
                if isinstance(item, Mapping):
                    out.extend(self._extract_records({"record": item, "__source_file": source_file}))
            return out

        if not isinstance(container, Mapping):
            return []

        out: List[Tuple[str, Dict[str, Any], Optional[str], str, str]] = []
        if "record" in container and isinstance(container["record"], Mapping):
            record = dict(container["record"])
            if record.get("kind") == "provider":
                out.append(("provider", record, source_file, str(record.get("name") or record.get("id") or "provider"), str(record.get("name") or "Provider")))
            elif record.get("kind") in {"agent_profile", "agent"} or "agents" in str(record.get("kind", "")):
                out.append(("agent", record, source_file, str(record.get("name") or record.get("id") or "agent"), str(record.get("name") or "Agent")))
            elif "conversation_id" in record or "thread_id" in record or "messages" in record:
                parent_id = str(record.get("conversation_id") or record.get("thread_id") or "thread")
                title = str(record.get("title") or "Busy conversation")
                for message in self._coerce_records(record.get("messages") or []):
                    if isinstance(message, Mapping):
                        out.append(("history", dict(message), source_file, parent_id, title))
            else:
                out.append(("note", record, source_file, str(record.get("id") or "import"), str(record.get("title") or "Busy snapshot")))

        for collection_key, kind in (
            ("providers", "provider"),
            ("provider_profiles", "provider"),
            ("agents", "agent"),
            ("agent_profiles", "agent"),
            ("threads", "history"),
            ("conversations", "history"),
            ("messages", "history"),
            ("history", "history"),
            ("notes", "note"),
            ("context", "note"),
            ("onboarding", "note"),
        ):
            values = container.get(collection_key)
            if isinstance(values, Mapping):
                out.extend(self._extract_records({"record": values, "__source_file": source_file}))
            elif isinstance(values, list):
                for item in values:
                    if not isinstance(item, Mapping):
                        continue
                    if kind in {"provider", "agent"}:
                        out.append((kind, dict(item), source_file, str(item.get("name") or item.get("id") or kind), str(item.get("name") or kind)))
                    elif kind == "history":
                        thread_id = str(item.get("id") or item.get("thread_id") or f"thread-{collection_key}")
                        title = str(item.get("title") or item.get("name") or thread_id)
                        messages = item.get("messages")
                        if isinstance(messages, list) and messages:
                            for message in messages:
                                if isinstance(message, Mapping):
                                    out.append(("history", dict(message), source_file, thread_id, title))
                        else:
                            out.append(("history", dict(item), source_file, thread_id, title))
                    else:
                        out.append(("note", dict(item), source_file, str(item.get("id") or kind), str(item.get("title") or kind)))
        return out

    def _coerce_records(self, raw_records: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_records, list):
            return [item for item in raw_records if isinstance(item, dict)]
        if isinstance(raw_records, Mapping):
            return [dict(raw_records)]
        return []

    def _coerce_content(self, kind: str, payload: Mapping[str, Any], title: str) -> str:
        if kind == "provider":
            provider_name = self._coerce_text(payload.get("name") or payload.get("provider") or "Provider")
            model = self._coerce_text(payload.get("model") or payload.get("model_name") or "")
            endpoint = self._coerce_text(payload.get("endpoint") or payload.get("base_url") or "")
            credentials = self._coerce_text(
                payload.get("api_key")
                or payload.get("token")
                or payload.get("secret")
                or ""
            )
            lines = [f"# Provider: {provider_name}", f"Model: {model}" if model else "", f"Endpoint: {endpoint}" if endpoint else ""]
            if credentials:
                lines.append("Credentials: [REDACTED]")
            details = self._coerce_text(payload.get("notes") or payload.get("description"))
            if details:
                lines.append(details)
            return "\n".join([line for line in lines if line]).strip() or f"Provider record for {provider_name}"
        if kind == "agent":
            name = self._coerce_text(payload.get("name") or payload.get("id") or "Agent")
            role = self._coerce_text(payload.get("role") or payload.get("title"))
            capabilities = self._coerce_text(payload.get("capabilities"))
            lines = [f"# Agent: {name}"]
            if role:
                lines.append(f"Role: {role}")
            if capabilities:
                lines.append(f"Capabilities: {capabilities}")
            details = self._coerce_text(payload.get("notes") or payload.get("description"))
            if details:
                lines.append(details)
            return "\n".join([line for line in lines if line]).strip() or f"Agent record for {name}"
        if kind == "history":
            text = self._coerce_text(payload.get("content") or payload.get("text") or payload.get("body") or payload.get("message"))
            if text:
                return text
            fallback = self._coerce_text(payload.get("summary"))
            return f"{title}\n\n{fallback}".strip() if fallback else ""
        return self._coerce_text(payload.get("content") or payload.get("text") or payload.get("body") or payload.get("summary") or title)

    def _coerce_author(self, payload: Mapping[str, Any]) -> str:
        return (
            self._coerce_text(payload.get("author"))
            or self._coerce_text(payload.get("agent"))
            or self._coerce_text(payload.get("role"))
            or "busy-user"
        )

    def _coerce_timestamp(self, raw: Any) -> str:
        if raw is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(raw, (int, float)):
            if raw > 10**15:
                return datetime.fromtimestamp(raw / 1_000_000, tz=timezone.utc).isoformat()
            if raw > 10**12:
                return datetime.fromtimestamp(raw / 1000, tz=timezone.utc).isoformat()
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
        if isinstance(raw, str):
            candidate = raw.strip()
            if not candidate:
                return datetime.now(timezone.utc).isoformat()
            try:
                if candidate.endswith("Z"):
                    return datetime.fromisoformat(candidate.replace("Z", "+00:00")).isoformat()
                return datetime.fromisoformat(candidate).isoformat()
            except Exception:
                return candidate
        return str(raw)

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join([self._coerce_text(v) for v in value]).strip()
        if isinstance(value, Mapping):
            if "text" in value and isinstance(value["text"], str):
                return value["text"]
        return ""

    def _build_metadata(
        self,
        kind: str,
        payload: Mapping[str, Any],
        source_file: Optional[str],
        parent_id: str,
        parent_title: str,
        sensitive_flags: List[str],
    ) -> Dict[str, Any]:
        return {
            "record_type": kind,
            "source_adapter": self.source_type,
            "requires_review": bool(sensitive_flags),
            "sensitive_flags": sensitive_flags,
            "source_file": source_file,
            "parent_id": parent_id,
            "parent_title": parent_title,
            "export_scope": "busy_local",
        }

    def _sensitive_signals(self, text: str) -> List[str]:
        lowered = text.lower()
        return [marker for marker in self._sensitive_markers if marker in lowered]


class TwitterManualImportAdapter(ImportSourceAdapter):
    """Adapter for manual export bundles from Twitter/X history exports."""

    source_type = "twitter"
    _sensitive_markers: Sequence[str] = (
        "ssn",
        "social security",
        "credit card",
        "password",
        "private key",
        "api key",
        "secret",
        "sex",
        "sexual",
        "suicide",
        "depression",
        "anxiety",
        "therapy",
    )

    def parse(self, source_payload: Mapping[str, Any] | str | bytes) -> ImportParseResult:
        data = self._coerce_payload(source_payload)
        source_metadata = self.source_metadata(data)
        entries = list(self._extract_entries(data))
        if not entries:
            raise ValueError("No importable Twitter/X export posts found")

        counts: Counter[str] = Counter()
        items: List[CanonicalImportItem] = []
        warnings: List[str] = []

        for item_index, entry in enumerate(entries):
            record, record_type, source_file = entry
            text = self._extract_content(record, record_type)
            if not text.strip():
                warnings.append(
                    f"Skipping empty {record_type} item"
                )
                continue

            created_at = self._coerce_timestamp(record.get("created_at") or record.get("createdAt") or record.get("date"))
            sensitivity = self._sensitive_signals(text)
            is_sensitive = bool(sensitivity)
            tweet_id = str(
                record.get("id")
                or record.get("id_str")
                or record.get("tweet_id")
                or record.get("message_id")
                or f"{record_type}-{item_index}"
            )
            conversation_id = str(
                record.get("conversation_id")
                or record.get("thread_id")
                or record.get("conversationId")
                or tweet_id
            )
            author_key = self._extract_author(record)
            scope_owner = self._extract_account_root(data)
            item_scope = f"x:{scope_owner}" if scope_owner else "x:import"
            metadata = self._build_metadata(entry)
            item = CanonicalImportItem(
                kind="memory",
                content=text,
                agent_scope=item_scope,
                visibility="quarantined" if is_sensitive else "pending",
                source="twitter_manual_upload",
                thread_id=conversation_id,
                message_id=tweet_id,
                created_at=created_at,
                author_key=author_key,
                review_state="quarantined" if is_sensitive else "pending",
                metadata=metadata,
                checksum="",
            )
            items.append(
                CanonicalImportItem(
                    kind=item.kind,
                    content=item.content,
                    agent_scope=item.agent_scope,
                    visibility=item.visibility,
                    source=item.source,
                    thread_id=item.thread_id,
                    message_id=item.message_id,
                    created_at=item.created_at,
                    author_key=item.author_key,
                    review_state=item.review_state,
                    metadata=item.metadata,
                    checksum=checksum_payload(item.__dict__),
                )
            )
            counts[record_type] += 1

        if not items:
            errors = tuple(["No parseable Twitter/X entries were found"])
            return ImportParseResult(
                import_id=checksum_payload(source_payload),
                source_type=self.source_type,
                source_metadata=source_metadata,
                items=tuple(),
                warnings=tuple(warnings),
                errors=errors,
                counts=dict(counts),
            )

        return ImportParseResult(
            import_id=checksum_payload(source_payload),
            source_type=self.source_type,
            source_metadata=source_metadata,
            items=tuple(items),
            warnings=tuple(warnings),
            errors=tuple(),
            counts=dict(counts),
        )

    def source_metadata(self, source_payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
        payload = self._coerce_payload(source_payload)
        entries = list(self._extract_entries(payload))
        if not entries:
            return {
                "source_type": self.source_type,
                "record_count": 0,
                "adapter": self.__class__.__name__,
            }

        by_type = Counter(entry[1] for entry in entries)
        return {
            "source_type": self.source_type,
            "record_count": len(entries),
            "record_types": dict(by_type),
            "adapter": self.__class__.__name__,
            "account": self._extract_account_root(payload),
        }

    def redaction_hints(self) -> Dict[str, Any]:
        return {
            "redact_fields": ["author_key", "metadata.sensitive_flags"],
            "default_review_state": "pending",
            "default_visibility": "quarantined",
        }

    def _coerce_payload(self, source_payload: Mapping[str, Any] | str | bytes) -> Mapping[str, Any]:
        if isinstance(source_payload, bytes):
            if zipfile.is_zipfile(io.BytesIO(source_payload)):
                return self._extract_archive_payload(source_payload)
            raw = source_payload.decode("utf-8", errors="replace")
            return self._coerce_payload(raw)
        if isinstance(source_payload, str):
            payload_text = source_payload.strip()
            if payload_text.startswith("window.") and "=" in payload_text:
                try:
                    payload_text = payload_text[payload_text.find("=") + 1 :].strip().rstrip(";")
                except Exception:
                    pass
            return self._coerce_wrapped_json(payload_text)
        return dict(source_payload)

    def _coerce_wrapped_json(self, raw: str) -> Mapping[str, Any]:
        payload = json.loads(raw)
        return {"data": payload} if isinstance(payload, list) else dict(payload)

    def _extract_archive_payload(self, archive_bytes: bytes) -> Dict[str, Any]:
        extracted: List[Tuple[str, Mapping[str, Any]]] = []
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zipped:
            for name in sorted(zipped.namelist()):
                lower_name = name.lower()
                if not (lower_name.endswith(".json") or lower_name.endswith(".js") or lower_name.endswith(".txt")):
                    continue
                if lower_name.endswith(".jpg") or lower_name.endswith(".png") or lower_name.endswith(".gif"):
                    continue
                try:
                    raw = zipped.read(name).decode("utf-8", errors="replace")
                    parsed = self._coerce_payload(raw)
                    if isinstance(parsed, dict) and "data" in parsed and len(parsed) == 1 and isinstance(parsed["data"], list):
                        parsed = parsed["data"]
                    extracted.append((name, parsed if isinstance(parsed, dict) else {"records": list(parsed)}))
                except Exception:
                    continue
        if extracted:
            return {"source_type": "twitter_archive", "files": extracted}
        return {"source_type": "twitter_archive", "files": []}

    def _extract_entries(self, payload: Mapping[str, Any]) -> List[Tuple[Dict[str, Any], str, Optional[str]]]:
        if not payload:
            return []

        if "files" in payload and isinstance(payload["files"], list):
            entries: List[Tuple[Dict[str, Any], str, Optional[str]]] = []
            for file_name, file_payload in payload["files"]:
                if isinstance(file_payload, list):
                    entries.extend(self._extract_entries({"data": file_payload, "__source_file": file_name}))
                elif isinstance(file_payload, dict):
                    container = dict(file_payload)
                    container["__source_file"] = file_name
                    entries.extend(self._extract_entries(container))
            return entries

        container = payload.get("data", payload)
        source_file = payload.get("__source_file")

        if isinstance(container, list):
            out: List[Tuple[Dict[str, Any], str, Optional[str]]] = []
            for raw_record in container:
                out.extend(self._extract_entries({ "record": raw_record, "__source_file": source_file}))
            return out

        if not isinstance(container, dict):
            return []

        out: List[Tuple[Dict[str, Any], str, Optional[str]]] = []

        if "record" in container:
            wrapped = container["record"]
            if isinstance(wrapped, Mapping):
                return self._normalise_record(wrapped, "post", source_file)

        if "tweet" in container and isinstance(container["tweet"], Mapping):
            return self._normalise_record(container["tweet"], "post", source_file)

        if "like" in container and isinstance(container["like"], Mapping):
            return self._normalise_record(container["like"], "like", source_file)

        if "thread" in container and isinstance(container["thread"], Mapping):
            thread_root = container["thread"]
            out.extend(self._normalise_record(thread_root, "thread", source_file))
            posts = thread_root.get("posts") or thread_root.get("tweets")
            if isinstance(posts, list):
                for entry in posts:
                    out.extend(self._normalise_record(entry, "thread", source_file))
            return out

        for record_key, record_type in (
            ("tweets", "post"),
            ("tweet", "post"),
            ("posts", "post"),
            ("likes", "like"),
            ("mentions", "mention"),
            ("threads", "thread"),
            ("conversations", "conversation"),
        ):
            records = container.get(record_key)
            if isinstance(records, list):
                for item in records:
                    if isinstance(item, Mapping):
                        out.extend(self._normalise_record(item, record_type, source_file))
            elif isinstance(records, Mapping):
                out.extend(self._normalise_record(records, record_type, source_file))

        return out

    def _normalise_record(self, candidate: Mapping[str, Any], record_type: str, source_file: Optional[str]) -> List[Tuple[Dict[str, Any], str, Optional[str]]]:
        if not isinstance(candidate, Mapping):
            return []

        out: List[Tuple[Dict[str, Any], str, Optional[str]]] = []
        if "tweet" in candidate and isinstance(candidate.get("tweet"), Mapping):
            return self._normalise_record(candidate["tweet"], record_type, source_file)

        if "like" in candidate and isinstance(candidate.get("like"), Mapping):
            return self._normalise_record(candidate["like"], "like", source_file)

        if "thread" in candidate and isinstance(candidate.get("thread"), Mapping):
            out.extend(self._normalise_record(candidate["thread"], "thread", source_file))
            thread_payload = candidate["thread"]
            nested_items = thread_payload.get("posts") or thread_payload.get("tweets")
            if isinstance(nested_items, list):
                for nested in nested_items:
                    out.extend(self._normalise_record(nested, "thread", source_file))
            return out

        if "records" in candidate and isinstance(candidate.get("records"), list):
            for item in candidate["records"]:
                out.extend(self._normalise_record(item, record_type, source_file))
            return out

        return [(dict(candidate), self._classify_record(candidate, record_type), source_file)]

    def _classify_record(self, record: Mapping[str, Any], default_type: str) -> str:
        in_reply_to = record.get("in_reply_to_status_id") or record.get("in_reply_to_user_id") or record.get("in_reply_to_tweet_id")
        if in_reply_to:
            return "mention"
        if record.get("retweet_count") is not None:
            return default_type
        if "retweeted_status" in record:
            return "retweet"
        return default_type

    def _extract_content(self, record: Mapping[str, Any], record_type: str) -> str:
        text = self._coerce_text(record.get("full_text") or record.get("text") or record.get("tweet") or record.get("note_text"))
        if text:
            return text

        if record.get("legacy") and isinstance(record["legacy"], Mapping):
            legacy = record["legacy"]
            legacy_text = legacy.get("full_text") or legacy.get("text")
            if legacy_text:
                return self._coerce_text(legacy_text)

        if record_type in {"like"} and record.get("title"):
            return self._coerce_text(record["title"])

        if record.get("content") and isinstance(record["content"], str):
            return self._coerce_text(record["content"])

        note: List[str] = []
        for _, value in record.items():
            if isinstance(value, str) and len(value) > 30:
                note.append(value)
        return "\n".join(note).strip()

    def _extract_author(self, record: Mapping[str, Any]) -> str:
        for key in ("screen_name", "username", "handle", "user", "author"):
            value = record.get(key)
            if isinstance(value, str):
                return value.strip("@")
            if isinstance(value, Mapping):
                user_candidate = value.get("screen_name") or value.get("username") or value.get("name")
                if isinstance(user_candidate, str):
                    return str(user_candidate).strip("@")
                if value.get("id_str") and isinstance(value.get("id_str"), str):
                    return str(value["id_str"])
        return "x-user"

    def _extract_account_root(self, payload: Mapping[str, Any]) -> str:
        if "account" in payload and isinstance(payload["account"], Mapping):
            account = payload["account"]
            for key in ("username", "screen_name", "handle", "id_str"):
                value = account.get(key)
                if value:
                    return str(value).strip("@")
        return ""

    def _coerce_timestamp(self, raw: Any) -> str:
        if raw is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(raw, (int, float)):
            if raw > 10**15:
                return datetime.fromtimestamp(raw / 1_000_000, tz=timezone.utc).isoformat()
            if raw > 10**12:
                return datetime.fromtimestamp(raw / 1000, tz=timezone.utc).isoformat()
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()

        if isinstance(raw, str):
            candidate = raw.strip()
            if not candidate:
                return datetime.now(timezone.utc).isoformat()
            try:
                if candidate.endswith("Z"):
                    return datetime.fromisoformat(candidate.replace("Z", "+00:00")).isoformat()
                return datetime.fromisoformat(candidate).isoformat()
            except Exception:
                return candidate
        return str(raw)

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join([self._coerce_text(v) for v in value]).strip()
        if isinstance(value, Mapping):
            if "text" in value and isinstance(value["text"], str):
                return value["text"]
        return ""

    def _sensitive_signals(self, text: str) -> List[str]:
        lowered = text.lower()
        return [marker for marker in self._sensitive_markers if marker in lowered]

    def _build_metadata(self, entry: Tuple[Dict[str, Any], str, Optional[str]]) -> Dict[str, Any]:
        record, record_type, source_file = entry
        flags: List[str] = self._sensitive_signals(self._coerce_text(record.get("full_text") or record.get("text")))
        metadata = {
            "record_type": record_type,
            "source_adapter": self.source_type,
            "requires_review": bool(flags),
            "source_file": source_file,
            "sensitive_flags": flags,
        }
        for key in ("urls", "hashtags", "user_mentions", "media"):
            if key in record:
                metadata[key] = record.get(key)
        if "conversation_id" in record:
            metadata["conversation_id"] = record.get("conversation_id")
        if "reply_count" in record:
            metadata["reply_count"] = record.get("reply_count")
        return metadata


class OpenClawManualImportAdapter(ImportSourceAdapter):
    """Adapter for OpenClaw rich exports with markdown memories and agent/skill context."""

    source_type = "openclaw"
    _sensitive_markers: Sequence[str] = (
        "ssn",
        "social security",
        "credit card",
        "password",
        "private key",
        "api key",
        "secret",
        "sex",
        "sexual",
        "suicide",
        "depression",
        "anxiety",
        "therapy",
    )

    def parse(self, source_payload: Mapping[str, Any] | str | bytes) -> ImportParseResult:
        data = self._coerce_payload(source_payload)
        source_metadata = self.source_metadata(data)
        entries = list(self._extract_entries(data))
        if not entries:
            raise ValueError("No importable OpenClaw export records found")

        counts: Counter[str] = Counter()
        items: List[CanonicalImportItem] = []
        warnings: List[str] = []

        for item_index, entry in enumerate(entries):
            record, record_type, source_file = entry
            content = self._extract_content(record, record_type)
            if not content.strip():
                warnings.append(f"Skipping empty {record_type} item")
                continue

            created_at = self._coerce_timestamp(
                record.get("created_at")
                or record.get("createdAt")
                or record.get("timestamp")
                or record.get("time")
            )
            sensitivity = self._sensitive_signals(content)
            is_sensitive = bool(sensitivity)
            item_id = str(
                record.get("id")
                or record.get("uuid")
                or record.get("thread_id")
                or record.get("item_id")
                or f"{record_type}-{item_index}"
            )
            thread_id = str(
                record.get("thread_id")
                or record.get("conversation_id")
                or record.get("session_id")
                or item_id
            )
            author_key = self._extract_author(record)
            item_scope = self._extract_scope(record, source_file)
            metadata = self._build_metadata(entry)

            item_kind = self._classify_item_kind(record_type)
            item = CanonicalImportItem(
                kind=item_kind,
                content=content,
                agent_scope=item_scope,
                visibility="quarantined" if is_sensitive else "pending",
                source="openclaw_manual_upload",
                thread_id=thread_id,
                message_id=item_id,
                created_at=created_at,
                author_key=author_key,
                review_state="quarantined" if is_sensitive else "pending",
                metadata=metadata,
                checksum="",
            )
            items.append(
                CanonicalImportItem(
                    kind=item.kind,
                    content=item.content,
                    agent_scope=item.agent_scope,
                    visibility=item.visibility,
                    source=item.source,
                    thread_id=item.thread_id,
                    message_id=item.message_id,
                    created_at=item.created_at,
                    author_key=item.author_key,
                    review_state=item.review_state,
                    metadata=item.metadata,
                    checksum=checksum_payload(item.__dict__),
                )
            )
            counts[item_kind] += 1

        if not items:
            errors = tuple(["No parseable OpenClaw entries were found"])
            return ImportParseResult(
                import_id=checksum_payload(source_payload),
                source_type=self.source_type,
                source_metadata=source_metadata,
                items=tuple(),
                warnings=tuple(warnings),
                errors=errors,
                counts=dict(counts),
            )

        return ImportParseResult(
            import_id=checksum_payload(source_payload),
            source_type=self.source_type,
            source_metadata=source_metadata,
            items=tuple(items),
            warnings=tuple(warnings),
            errors=tuple(),
            counts=dict(counts),
        )

    def source_metadata(self, source_payload: Mapping[str, Any] | str | bytes) -> Dict[str, Any]:
        data = self._coerce_payload(source_payload)
        entries = list(self._extract_entries(data))
        by_type = Counter(entry[1] for entry in entries)
        return {
            "source_type": self.source_type,
            "record_count": len(entries),
            "record_types": dict(by_type),
            "adapter": self.__class__.__name__,
            "archive_files": len(data.get("files")) if isinstance(data, dict) and isinstance(data.get("files"), list) else 0,
        }

    def redaction_hints(self) -> Dict[str, Any]:
        return {
            "redact_fields": ["author_key", "metadata.sensitive_flags"],
            "default_review_state": "pending",
            "default_visibility": "quarantined",
        }

    def _coerce_payload(self, source_payload: Mapping[str, Any] | str | bytes) -> Mapping[str, Any]:
        if isinstance(source_payload, bytes):
            if zipfile.is_zipfile(io.BytesIO(source_payload)):
                return self._extract_archive_payload(source_payload)
            raw = source_payload.decode("utf-8", errors="replace")
            return self._coerce_payload(raw)

        if isinstance(source_payload, str):
            return self._coerce_wrapped_json(source_payload)

        return dict(source_payload)

    def _coerce_wrapped_json(self, raw: str) -> Mapping[str, Any]:
        payload_text = raw.strip()
        if not payload_text:
            return {}
        if payload_text.startswith("window.") and "=" in payload_text:
            try:
                payload_text = payload_text[payload_text.find("=") + 1 :].strip().rstrip(";")
            except Exception:
                pass
        payload = json.loads(payload_text)
        return dict(payload) if isinstance(payload, dict) else {"records": list(payload)}

    def _extract_archive_payload(self, archive_bytes: bytes) -> Dict[str, Any]:
        extracted: List[Tuple[str, Mapping[str, Any]]] = []
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zipped:
            for name in sorted(zipped.namelist()):
                lower_name = name.lower()
                if lower_name.endswith(".md") or lower_name.endswith(".markdown"):
                    try:
                        raw = zipped.read(name).decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    extracted.append((name, {"kind": "markdown_note", "path": name, "content": raw}))
                    continue
                if lower_name.endswith(".json") or lower_name.endswith(".js"):
                    try:
                        raw = zipped.read(name).decode("utf-8", errors="replace")
                        parsed = self._coerce_payload(raw)
                    except Exception:
                        continue
                    extracted.append((name, parsed if isinstance(parsed, dict) else {"records": list(parsed)}))
                    continue
                if lower_name.endswith(".txt"):
                    try:
                        raw = zipped.read(name).decode("utf-8", errors="replace").strip()
                    except Exception:
                        continue
                    if raw:
                        extracted.append((name, {"kind": "text_note", "path": name, "content": raw}))
        return {"files": extracted}

    def _extract_entries(self, payload: Mapping[str, Any]) -> List[Tuple[Dict[str, Any], str, Optional[str]]]:
        if not payload:
            return []

        if "files" in payload and isinstance(payload["files"], list):
            entries: List[Tuple[Dict[str, Any], str, Optional[str]]] = []
            for file_name, file_payload in payload["files"]:
                if isinstance(file_payload, list):
                    entries.extend(self._extract_entries({"records": file_payload, "__source_file": file_name}))
                elif isinstance(file_payload, Mapping):
                    container = dict(file_payload)
                    container["__source_file"] = file_name
                    entries.extend(self._extract_entries(container))
            return entries

        container = payload.get("data", payload)
        source_file = payload.get("__source_file")

        if isinstance(container, list):
            out: List[Tuple[Dict[str, Any], str, Optional[str]]] = []
            for raw_record in container:
                out.extend(self._extract_entries({"record": raw_record, "__source_file": source_file}))
            return out

        if not isinstance(container, dict):
            return []

        out: List[Tuple[Dict[str, Any], str, Optional[str]]] = []

        if "record" in container and isinstance(container["record"], Mapping):
            out.extend(self._normalise_record(container["record"], "record", source_file))
            return out

        for key, kind in (
            ("agents", "agent_profile"),
            ("agent_profiles", "agent_profile"),
            ("skills", "skill"),
            ("skill_profiles", "skill"),
            ("memories", "memory"),
            ("messages", "memory"),
            ("threads", "thread"),
            ("conversations", "thread"),
            ("sessions", "session"),
            ("notes", "note"),
            ("context", "context"),
            ("contexts", "context"),
            ("events", "event"),
        ):
            values = container.get(key)
            if isinstance(values, Mapping):
                out.extend(self._normalise_record(values, kind, source_file))
            elif isinstance(values, list):
                for item in values:
                    if isinstance(item, Mapping):
                        out.extend(self._normalise_record(item, kind, source_file))

        if container.get("kind") in {"markdown_note", "text_note"} and container.get("content"):
            return self._normalise_record(container, "note", source_file)

        if not out and container.get("content"):
            out.append((dict(container), "context", source_file))

        return out

    def _normalise_record(
        self,
        candidate: Mapping[str, Any],
        default_kind: str,
        source_file: Optional[str],
    ) -> List[Tuple[Dict[str, Any], str, Optional[str]]]:
        if not isinstance(candidate, Mapping):
            return []

        nested_keys = (
            ("messages", "memory"),
            ("notes", "note"),
            ("threads", "thread"),
            ("skills", "skill"),
            ("agents", "agent_profile"),
        )
        if "records" in candidate and isinstance(candidate["records"], list):
            out: List[Tuple[Dict[str, Any], str, Optional[str]]] = []
            for nested in candidate["records"]:
                if isinstance(nested, Mapping):
                    out.extend(self._normalise_record(nested, default_kind, source_file))
            if out:
                return out

        for group_key, group_kind in nested_keys:
            values = candidate.get(group_key)
            if isinstance(values, list):
                out: List[Tuple[Dict[str, Any], str, Optional[str]]] = []
                for nested in values:
                    if isinstance(nested, Mapping):
                        out.extend(self._normalise_record(nested, group_kind, source_file))
                if out:
                    return out
            if isinstance(values, Mapping):
                return self._normalise_record(values, group_kind, source_file)

        payload = dict(candidate)
        payload["__source_file"] = source_file
        kind = payload.get("kind") if isinstance(payload.get("kind"), str) else default_kind
        return [(payload, kind, source_file)]

    def _extract_content(self, record: Mapping[str, Any], record_type: str) -> str:
        if record_type == "thread":
            if isinstance(record.get("messages"), list):
                messages = self._coerce_message_list(record.get("messages"))
                if messages:
                    return self._messages_to_markdown(record.get("title") or "Thread", messages)
            if record.get("summary"):
                return f"# {record.get('title', 'Thread')}\n\n{self._coerce_text(record.get('summary'))}"

        if record_type in {"skill", "agent_profile"}:
            parts: List[str] = []
            title = str(record.get("name") or record.get("id") or record.get("agent") or "OpenClaw Record")
            if record.get("name"):
                parts.append(f"# {record.get('name')}")
            if record.get("description"):
                parts.append(self._coerce_text(record["description"]))
            if record.get("summary"):
                parts.append(self._coerce_text(record["summary"]))
            if record_type == "agent_profile" and record.get("role"):
                parts.append(f"Role: {record.get('role')}")
            if record_type == "agent_profile" and isinstance(record.get("capabilities"), list):
                cap_text = ", ".join([self._coerce_text(item) for item in record["capabilities"]])
                if cap_text:
                    parts.append(f"Capabilities: {cap_text}")
            if not parts:
                raw_body = self._coerce_text(record.get("content") or record.get("text") or record.get("details"))
                if raw_body:
                    parts.append(raw_body)
            if record_type in {"skill", "agent_profile"}:
                return "\n\n".join(parts).strip() if parts else self._pretty_dump(record)
            return "\n\n".join(parts).strip()

        if record.get("content_markdown"):
            return self._coerce_text(record["content_markdown"])
        if isinstance(record.get("content"), str):
            return record["content"]
        if isinstance(record.get("body"), str):
            return record["body"]
        if record.get("text"):
            return self._coerce_text(record["text"])
        if record.get("summary"):
            return self._coerce_text(record["summary"])
        return self._pretty_dump(record)

    def _coerce_message_list(self, raw_messages: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_messages, list):
            return []
        messages: List[Dict[str, Any]] = []
        for item in raw_messages:
            if isinstance(item, Mapping):
                messages.append(dict(item))
        return messages

    def _messages_to_markdown(self, title: str, messages: List[Dict[str, Any]]) -> str:
        lines: List[str] = [f"# {title}"]
        for message in messages:
            actor = self._coerce_text(
                message.get("role")
                or message.get("author")
                or message.get("actor")
                or message.get("agent")
            )
            timestamp = self._coerce_timestamp(
                message.get("created_at")
                or message.get("timestamp")
                or message.get("time")
            )
            body = self._coerce_text(message.get("content") or message.get("text") or message.get("body"))
            if not body:
                body = self._pretty_dump(message)
            heading = f"## {actor or 'Unknown'} ({timestamp})" if actor or timestamp else "## Message"
            lines.append(heading)
            lines.append(body)
        return "\n\n".join(lines)

    def _coerce_timestamp(self, raw: Any) -> str:
        if raw is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(raw, (int, float)):
            if raw > 10**15:
                return datetime.fromtimestamp(raw / 1_000_000, tz=timezone.utc).isoformat()
            if raw > 10**12:
                return datetime.fromtimestamp(raw / 1000, tz=timezone.utc).isoformat()
            return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat()
        if isinstance(raw, str):
            candidate = raw.strip()
            if not candidate:
                return datetime.now(timezone.utc).isoformat()
            try:
                if candidate.endswith("Z"):
                    return datetime.fromisoformat(candidate.replace("Z", "+00:00")).isoformat()
                return datetime.fromisoformat(candidate).isoformat()
            except Exception:
                return candidate
        return str(raw)

    def _coerce_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join([self._coerce_text(v) for v in value]).strip()
        if isinstance(value, Mapping):
            if "text" in value and isinstance(value["text"], str):
                return value["text"]
        return ""

    def _pretty_dump(self, value: Mapping[str, Any]) -> str:
        try:
            return json.dumps(value, indent=2, sort_keys=True)
        except Exception:
            return str(value)

    def _extract_author(self, record: Mapping[str, Any]) -> str:
        for key in ("author", "role", "agent", "agent_name", "actor", "user", "name"):
            value = record.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping):
                for nested in ("name", "username", "handle", "id", "id_str"):
                    nested_value = value.get(nested)
                    if nested_value:
                        return str(nested_value)
        return "openclaw-user"

    def _extract_scope(self, record: Mapping[str, Any], source_file: Optional[str]) -> str:
        agent_id = (
            record.get("agent_id")
            or record.get("agent")
            or record.get("owner")
            or record.get("scope")
        )
        if isinstance(agent_id, str) and agent_id:
            return f"openclaw:{agent_id}"
        if source_file:
            return f"openclaw:file:{source_file}"
        return "openclaw:import"

    def _classify_item_kind(self, record_type: str) -> str:
        normalized = str(record_type).strip().lower().replace("-", "_")
        if normalized in {"agent_profile", "agent", "agent_profile"}:
            return "agent_profile"
        if normalized in {"skill", "skills"}:
            return "skill"
        if normalized in {"thread", "conversation", "session"}:
            return "thread"
        if normalized in {"memory", "message", "note", "context", "event", "note"}:
            return normalized
        return "memory"

    def _sensitive_signals(self, text: str) -> List[str]:
        lowered = text.lower()
        return [marker for marker in self._sensitive_markers if marker in lowered]

    def _build_metadata(self, entry: Tuple[Dict[str, Any], str, Optional[str]]) -> Dict[str, Any]:
        record, record_type, source_file = entry
        content_text = self._coerce_text(record.get("content") or record.get("summary"))
        flags = self._sensitive_signals(content_text)
        metadata = {
            "record_type": self._classify_item_kind(record_type),
            "source_adapter": self.source_type,
            "source_file": source_file,
            "requires_review": bool(flags),
            "sensitive_flags": flags,
        }
        if record.get("agent_id"):
            metadata["agent_id"] = record.get("agent_id")
        if record.get("thread_id"):
            metadata["thread_id"] = record.get("thread_id")
        if record.get("path"):
            metadata["path"] = record.get("path")
        if record.get("name"):
            metadata["name"] = record.get("name")
        if record.get("created_at"):
            metadata["created_at"] = record.get("created_at")
        return metadata


_ADAPTERS = {
    OpenAIManualImportAdapter.source_type: OpenAIManualImportAdapter(),
    OpenAICodexManualImportAdapter.source_type: OpenAICodexManualImportAdapter(),
    CodexManualImportAdapter.source_type: CodexManualImportAdapter(),
    GeminiManualImportAdapter.source_type: GeminiManualImportAdapter(),
    GeminiCLIManualImportAdapter.source_type: GeminiCLIManualImportAdapter(),
    CopilotManualImportAdapter.source_type: CopilotManualImportAdapter(),
    BusyLocalImportAdapter.source_type: BusyLocalImportAdapter(),
    TwitterManualImportAdapter.source_type: TwitterManualImportAdapter(),
    OpenClawManualImportAdapter.source_type: OpenClawManualImportAdapter(),
}


def get_import_adapter(source_type: str) -> ImportSourceAdapter | None:
    if source_type in {"x", "twitter_x", "tw", "x_com"}:
        return _ADAPTERS.get("twitter")
    if source_type in {"codex", "openai_codex", "codex_cli", "openai-codex", "codex_export"}:
        return _ADAPTERS.get("codex")
    if source_type in {"gemini", "gemini_cli", "gemini_cli_export", "google_gemini"}:
        return _ADAPTERS.get("gemini_cli" if source_type in {"gemini_cli", "gemini_cli_export"} else "gemini")
    if source_type in {"copilot", "vscode", "vscode_copilot", "copilot_export"}:
        return _ADAPTERS.get("copilot")
    if source_type in {"openclaw", "open_claw", "claw", "claw_export"}:
        return _ADAPTERS.get("openclaw")
    if source_type in {"busy", "busy_local", "busy_export"}:
        return _ADAPTERS.get("busy_local")
    return _ADAPTERS.get(source_type)
