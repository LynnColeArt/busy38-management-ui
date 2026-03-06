# Core Plugin Reference Test Alignment Change Request

**Status**: implemented
**Date**: 2026-03-05

## Summary

The management-ui test for `/api/plugins/core/reference` still expected the older required-core count.

The canonical required plugin set now includes `busy-38-onboarding`, and the runtime endpoint already reports that expanded set.

## Decision

- Align the test expectation to the current canonical required plugin matrix.
- Do not change runtime behavior as part of this fix.

## Scope

- `tests/test_main.py`
- no backend behavior change
