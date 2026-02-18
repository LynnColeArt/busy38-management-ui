"""Tests for import contract and queue primitives."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError

from backend.app import storage
from backend.app.import_contract import CanonicalImportItem, ImportParseResult, checksum_payload


class TestImportContract(unittest.TestCase):
    def test_models_are_immutable(self) -> None:
        item = CanonicalImportItem(
            kind="memory",
            content="Hello world",
            agent_scope="orchestrator",
            visibility="quarantined",
            source="openai",
            thread_id="thread-1",
            message_id="message-1",
            created_at="2026-02-18T00:00:00Z",
            author_key="author-1",
            review_state="pending",
            metadata={},
            checksum="abc",
        )
        with self.assertRaises(FrozenInstanceError):
            item.review_state = "approved"

        result = ImportParseResult(
            import_id="import-demo",
            source_type="openai",
            source_metadata={"provider": "openai"},
            items=(item,),
            warnings=("w",),
            errors=(),
            counts={"items": 1},
        )
        self.assertEqual(result.total_items, 1)

    def test_checksum_is_deterministic(self) -> None:
        payload = {"a": 1, "b": 2}
        reordered = {"b": 2, "a": 1}
        self.assertEqual(checksum_payload(payload), checksum_payload(reordered))


class TestImportQueueSchema(unittest.TestCase):
    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        handle.close()
        self._db_file = handle.name
        os.environ["MANAGEMENT_DB_PATH"] = self._db_file
        storage.ensure_schema()

    def tearDown(self) -> None:
        os.remove(self._db_file)

    def test_import_job_dedupes_by_checksum(self) -> None:
        checksum = checksum_payload({"source": "openai"})
        first, created = storage.create_import_job("openai", {"provider": "openai"}, checksum, status="pending")
        second, created_again = storage.create_import_job("openai", {"provider": "openai"}, checksum, status="pending")

        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(storage.list_import_jobs()), 1)

    def test_import_items_are_deduped_with_progress_events(self) -> None:
        checksum = checksum_payload({"source": "openai"})
        (job, _) = storage.create_import_job("openai", {"provider": "openai"}, checksum, status="pending")
        items = [
            {
                "kind": "memory",
                "agent_scope": "orchestrator",
                "content": "first note",
                "thread_id": "thread-1",
                "message_id": "msg-1",
                "author_key": "a",
                "visibility": "quarantined",
                "source": "openai",
                "review_state": "pending",
                "metadata": {"imported": True},
            }
        ]
        first_pass = storage.add_import_items(job["id"], items)
        second_pass = storage.add_import_items(job["id"], items)

        self.assertEqual(len(storage.list_import_items(import_id=job["id"])), 1)
        self.assertEqual(len(first_pass), 1)
        self.assertEqual(len(second_pass), 1)

        events = storage.list_events(limit=20)
        self.assertTrue(any(event["type"] == "import.progress" for event in events))
        self.assertTrue(any(event["type"] == "import.item.reviewed" for event in events))


if __name__ == "__main__":
    unittest.main()
