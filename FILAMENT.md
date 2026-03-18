# Filament Integration

Busy38 Management UI (`busy38-management-ui`) is a downstream consumer and
integration target for Filament when it needs efficient real-time data access.

Rules:

- Filament is the upstream shared realtime transport/runtime/binary layer.
- Busy38 Management UI keeps its domain semantics, data models, adapters,
  caching, auth, and rollout logic local.
- Reusable transport, packaging, and portability findings should flow back into
  Filament instead of forking shared realtime glue here.
- Prefer Filament manifests, vendored packages, wrappers, and release assets
  before introducing repo-local transport abstractions.

Reference:

- https://github.com/crucible-energy/filament
