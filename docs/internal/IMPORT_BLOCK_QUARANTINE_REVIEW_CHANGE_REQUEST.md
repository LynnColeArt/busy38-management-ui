# Import Block Quarantine Review Change Request

**Status**: implemented
**Date**: 2026-03-05

## Summary

Management UI import intake currently maps automated `ATTACHMENT_DECISION_BLOCK` results directly to `review_state="rejected"`.

That collapses the distinction between:

- an automated intake block, and
- a human rejection decision.

## Decision

- Automated `ATTACHMENT_DECISION_BLOCK` results must land in:
  - `visibility="quarantined"`
  - `review_state="quarantined"`
- The original intake decision must remain visible in item metadata:
  - `metadata.intake_decision = "block"`
  - `metadata.intake_reasons = [...]`
- Only an explicit human review action may set `review_state="rejected"`.

## Scope

- `backend/app/main.py`
- `tests/test_main.py`
- `CURRENT_STATE.md`

## Non-goals

- No change to the underlying Busy intake policy decision engine.
- No change to the rule that reviewed sensitive items require a note for `approved` or `rejected`.
