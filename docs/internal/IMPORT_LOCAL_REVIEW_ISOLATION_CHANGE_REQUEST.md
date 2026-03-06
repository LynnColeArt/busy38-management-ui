# Import Local Review Isolation Change Request

**Status**: implemented
**Date**: 2026-03-05

## Summary

Management UI import intake already quarantines sensitive items for review, but the non-SaaS threat model is narrower than a blanket "hide everything from local operators" rule:

- local human reviewers still need access to the raw imported material,
- downstream agent/provider/runtime surfaces must not inherit that raw content,
- security must provide redacted derivatives for any non-review use.

## Decision

- Raw imported content may remain stored locally for authenticated human review.
- Sensitive or quarantined import items must carry explicit metadata that marks raw content as local-review-only.
- Security-produced redacted preview text must be persisted alongside the raw review copy when intake detects sensitive content or redaction need.
- Downstream management surfaces that synthesize agent-facing summaries or artifacts must use the redacted derivative instead of raw import content whenever the metadata says raw content is local-review-only.

## Scope

- `backend/app/main.py`
- `backend/app/storage.py`
- `docs/ARCHITECTURE.md`
- `README.md`
- `tests/test_main.py`
- `CURRENT_STATE.md`

## Non-goals

- No attempt to remove the local human review copy from the import-review UI/API.
- No new remote processing path.
- No change to the rule that human review is required before sensitive items may be approved or rejected.
