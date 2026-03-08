# WebSocket Event Stream Bootstrap Change Request

**Status**: implemented
**Date**: 2026-03-05

## Summary

The management UI exposes a live event stream at `GET /api/events/ws`, but the local bootstrap dependencies only installed plain `uvicorn`.

That configuration is incomplete for a real runtime smoke session: `uvicorn` without a websocket transport library rejects the upgrade request and the browser reports websocket handshake failures.

## Desired behavior

- A standard local install from repository requirements must support the `/api/events/ws` route without extra manual package installation.
- The repository test suite must exercise the websocket route directly instead of skipping it.
- The local-development docs must state that the standard backend dependency set includes websocket transport support.

## Implementation

- Add an explicit websocket transport dependency to backend runtime requirements.
- Make dev requirements include the runtime requirements to avoid drift between local runtime and local test environments.
- Replace the skipped websocket test with a real handshake test that verifies:
  - missing token is rejected
  - valid token connects
  - first payload reports the resolved role

## Non-goals

- No application websocket protocol changes.
- No fallback polling transport changes.
