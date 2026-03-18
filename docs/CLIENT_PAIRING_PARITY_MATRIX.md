# Client Pairing Parity Matrix

Last updated: 2026-03-15

## Purpose

This document records the current control-plane pairing and trusted-device
parity across the management API and the active mobile client repos.

It exists to prevent management-plane work from drifting toward one client
implementation while other supported clients fall behind.

Current client priority for parity decisions:
- `pillowfort-ios-native`
- `pillowfort-kotlin`
- `busy38-iphone` remains relevant, but it is not the primary reference client
  for the current slice.

## Canonical control-plane contract

`busy38-management-ui` is the canonical authority for the current bounded
mobile pairing slice:
- `POST /api/mobile/pairing/issue`
- `GET /api/mobile/pairing/discovery`
- `POST /api/mobile/pairing/exchange`
- `POST /api/mobile/trust/refresh`
- `GET /api/mobile/pairing/state`
- `POST /api/mobile/pairing/revoke`

The management repo currently implements:
- short-lived single-use pairing issue,
- unauthenticated read-only LAN discovery descriptors,
- scoped bridge-token exchange,
- durable trusted-device relationship persistence,
- refresh-grant rotation through `POST /api/mobile/trust/refresh`,
- safe trusted-device summaries through `GET /api/mobile/pairing/state`,
- revoke-by-`token_id`, including active post-refresh tokens.

## Repo snapshot

Management control plane:
- repo: `../busy38-management-ui`
- checked-out branch: `feature/realtime-mobile-readiness`
- current local head during this audit: `4411866`

Native iOS client:
- repo: `../pillowfort-ios-native`
- checked-out branch: `feature/ios-native-parity-spike`
- current local head during this audit: `249b5e4`
- local worktree is dirty, so its newest shell/runtime work is in-flight and
  not yet represented by one clean reviewed commit boundary

Kotlin client:
- repo: `../pillowfort-kotlin`
- checked-out branch: `main`
- current local head during this audit: `227b221`
- a `feature/realtime-mobile-readiness` branch exists at `71395ec`, but the
  checked-out worktree also contains substantial untracked Android app/docs
  work that is not yet committed

## Parity matrix

### 1. Pairing issue authority

Management UI:
- implemented and canonical

Native iOS:
- consumes the management control plane as designed
- physical-device probes already exercise real management-owned pairing issue
  and bootstrap flows

Kotlin:
- no management-owned admin issue surface in the Android repo
- local smoke harness still centers on a mock pairing control plane rather than
  the real management repo

Status:
- native iOS aligned
- Kotlin behind

### 2. QR / launch-url bootstrap

Management UI:
- browser can issue a live pairing code and derive a canonical
  `busy38pair://pair` payload locally

Native iOS:
- implemented through launch-url intake, QR import, camera scanning, and
  physical-device proof scripts

Kotlin:
- implemented in local Android code for payload parsing and launch-url / QR
  intake
- local emulator/physical-device smoke harnesses generate canonical
  `busy38pair://pair` artifacts

Status:
- broadly aligned across all three repos

### 3. LAN discovery descriptor consumption

Management UI:
- implemented at `GET /api/mobile/pairing/discovery`

Native iOS:
- implemented and explicitly documented as Bonjour-backed local-network pairing
  discovery with short-code confirmation
- physical-device proof path exists for LAN discovery plus chained refresh

Kotlin:
- no committed evidence during this audit that the Android runtime consumes the
  real discovery descriptor from management UI
- current local docs and smoke harnesses still emphasize mock pairing exchange
  and launch-url delivery rather than management-owned LAN discovery

Status:
- native iOS aligned
- Kotlin behind

### 4. Durable trusted-device persistence

Management UI:
- implemented in the control plane and persisted in shared Busy pairing state

Native iOS:
- implemented with protected native persistence and durable trusted-session
  storage

Kotlin:
- local Android runtime now persists `device_relationship_id`,
  `refresh_grant`, and `trusted_device_expires_at`
- pairing scope can survive cold start and recover the bridge session without
  forcing a fresh pairing-code exchange

Status:
- native iOS aligned
- Kotlin aligned for trusted-device persistence

### 5. Trusted-device refresh via `POST /api/mobile/trust/refresh`

Management UI:
- implemented and tested

Native iOS:
- implemented
- repo-local hardware probes explicitly prove real control-plane token rotation
  and revocation of the superseded token

Kotlin:
- local Android runtime now calls `POST /api/mobile/trust/refresh`
- reconnect and cold-start connect reuse persisted bridge credentials when
  still valid, refresh them when expired but the trusted-device lease is still
  active, and only fall back to fresh pairing exchange when needed
- focused unit coverage now exists for refresh payload parsing and refresh
  decision policy

Status:
- native iOS aligned
- Kotlin aligned for trusted-device refresh continuity

### 6. Safe admin inspection and revoke coherence

Management UI:
- implemented through `GET /api/mobile/pairing/state` and
  `POST /api/mobile/pairing/revoke`

Native iOS:
- client does not own admin inspection, but current probe paths already depend
  on revoke/refresh coherence in the management plane

Kotlin:
- no committed Android proof was found that exercises revoke coherence against
  the real management repo

Status:
- native iOS indirectly validated
- Kotlin behind

## In-flight vs not yet addressed

### In-flight

Native iOS:
- the repo is clearly in active use for pairing/trusted-device continuity and
  already has deeper real-device proof than the Flutter client
- the current risk is process hygiene, not contract absence: the local dirty
  worktree needs commit boundaries and validation reruns before being treated
  as settled parity truth

Kotlin:
- the repo has meaningful local Android scaffolding, pairing payload parsing,
  emulator smoke, physical-device smoke harnesses, and appearance/readability
  docs
- however, much of that state is still uncommitted on `main`, and the tracked
  `feature/realtime-mobile-readiness` branch does not yet expose the same
  runtime surface as the local worktree

### Not yet addressed

Kotlin still lacks committed parity for the most important management-owned
continuity behaviors:
- real management-owned LAN discovery consumption,
- proof paths that use `busy38-management-ui` instead of a local mock control
  plane for the above behaviors.

## Recommendation

The optimal next step is not more management-plane contract expansion.

The highest-value move is:
1. treat PillowFort iOS Native (`pillowfort-ios-native`) as the current non-Flutter reference client
   for management-plane pairing continuity,
2. bring PillowFort Kotlin (`pillowfort-kotlin`) up to the existing management contract,
3. require future management pairing changes to name their parity impact on
   native iOS and Kotlin explicitly before they are treated as ready.

Concretely, the next implementation slice should be:
- Kotlin proof automation that boots the real `busy38-management-ui` control
  plane instead of only a mock pairing exchange server
- Kotlin consumption of the real management-owned LAN discovery descriptor
  instead of only QR/manual bootstrap artifacts

Until Kotlin reaches that baseline, the management API should be treated as
stable enough for this slice, and native parity effort should focus on client
adoption rather than new endpoint invention.
