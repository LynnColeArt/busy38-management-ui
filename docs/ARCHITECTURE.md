# Busy38 Management UI Architecture

## Related docs

- [AGENTS.md](../AGENTS.md)
- [CURRENT_STATE.md](../CURRENT_STATE.md)
- [Provider and Management UX Specification](PROVIDER_AND_MANAGEMENT_UX_SPEC.md)
- [Plugin UI Debugger](PLUGIN_DEBUGGER.md)
- [Client Pairing Parity Matrix](CLIENT_PAIRING_PARITY_MATRIX.md)

## Mermaid architecture diagram

```mermaid
flowchart LR
  User((Operator / Human))
  SPA["Busy38 Management Web UI<br/>web/index.html + web/app.js"]
  BrowserToken["Authorization header or ?token query"]

  subgraph "Network Boundary"
    Http["HTTP/WS Clients"]
  end

  subgraph "Backend API (backend/app/main.py)"
    Startup["FastAPI app startup<br/>@app.on_event('startup')"]
    Auth["Token + role resolver<br/>admin/viewer/open-access"]
    Settings["Settings endpoints /api/settings"]
    Providers["Provider APIs<br/>/api/providers*"]
    Plugins["Plugin APIs<br/>/api/plugins*"]
    Tools["Tool APIs<br/>/api/tools*"]
    Imports["Import APIs<br/>/api/agents/import*"]
    Agents["Agent APIs<br/>/api/agents*"]
    GmTickets["GM Ticket APIs<br/>/api/gm-tickets*"]
    RuntimeAPI["Runtime APIs<br/>/api/runtime/*"]
    Memory["Memory & chat APIs<br/>/api/memory, /api/chat_history"]
    Events["Event APIs<br/>/api/events, /api/events/ws<br/>requires websocket transport dependency in backend requirements"]
  end

  subgraph "Runtime bridge (backend/app/runtime.py)"
    Runtime["RuntimeAdapter"]
    Overlay["Actor overlays (read/write)"]
    DirectBusy["Direct Busy runtime modules<br/>BUSY_RUNTIME_PATH"]
    BusyBridge["Bridge runtime endpoints<br/>BUSY_BRIDGE_URL"]
  end

  subgraph "Import pipeline (backend/app)"
    Contract["import_contract.py<br/>CanonicalImportItem / ImportParseResult"]
    Adapters["import_adapters.py<br/>OpenAI/Codex/Gemini/Copilot/CLI adapters"]
    BusyIntake["core.cognition/attachments intake decision"]
  end

  subgraph "Persistence (backend/app/storage.py)"
    Schema["SQLite schema bootstrap"]
    DB["management.db<br/>settings / providers / plugins / tools / agents / events / imports / memory / chat_history / gm_tickets / gm_ticket_messages"]
    PluginSync["plugin->tool_registry sync"]
  end

 subgraph "External systems"
    ProviderNet["Model provider endpoints<br/>for discover/test"]
    Webhost["Static host for web UI"]
    BusyCore["Busy core packages (optional)"]
  end

  User --> Http
  User -->|pastes token| BrowserToken
  Http -->|REST + WS| SPA
  Webhost -->|serves files| SPA
  SPA --> BrowserToken
  SPA -->|/api/*| Auth
  Auth --> Settings
  Auth --> Providers
  Auth --> Plugins
  Auth --> Tools
  Auth --> Imports
  Auth --> Agents
  Auth --> GmTickets
  Auth --> RuntimeAPI
  Auth --> Memory
  Auth --> Events

  Startup --> Schema
  Schema --> DB

  Startup --> Runtime

  Settings --> DB
  Agents --> DB
  Memory --> DB
  Events --> DB

  Plugins --> PluginSync
  PluginSync --> DB
  PluginSync --> Tools
  Tools --> DB

  Providers -->|tool model discovery| ProviderNet
  Providers --> DB
  DB -->|history/metrics| Providers

  Imports -->|multipart/json upload| Adapters
  Adapters --> Contract
  Contract --> BusyIntake
  BusyIntake -->|review state + visibility| Imports
  Imports --> DB
  Imports --> Events

  Agents --> DB
  Agents --> Runtime
  Runtime -->|read/write actor overlay| Overlay
  Overlay -->|stored via store/sqlite path| DB
  Runtime --> DirectBusy
  Runtime --> BusyBridge
  DirectBusy --> BusyCore
  BusyBridge --> BusyCore
  BusyCore --> Overlay

  RuntimeAPI --> Runtime
  RuntimeAPI --> DB
  GmTickets --> DB
  ToolUsage["tool usage writes<br/>/api/tools/{id}/usage"]
  ToolUsage --> DB

  Tools --> ToolUsage

  Events -->|new event rows| DB
  Events -->|live pushes| SPA
  SPA -->|auth token persisted| SPA

  SPA -->|stores token| BrowserToken
```

## Runtime flow map

```mermaid
sequenceDiagram
  autonumber
  actor Operator as Operator Browser
  participant U as Web SPA
  participant API as FastAPI Main
  participant S as SQLite Storage
  participant I as Import Pipeline
  participant R as Runtime Adapter
  participant B as Busy runtime / Bridge
  participant ProviderNet as Provider endpoints

  Operator->>U: Open UI and set token
  U->>API: GET /api/health, /api/settings
  API->>S: get_settings / auth role resolution
  S-->>API: redacted config
  API-->>U: role badge + health state

  Operator->>U: Trigger provider test / discovery
  U->>API: POST /api/providers/{id}/discover-models or /test
  API->>S: load provider row + validate metadata
  API->>ProviderNet: discover models / send health probe
  API->>S: persist provider history/metrics
  API-->>U: updated provider status + diagnostics

  Operator->>U: Import context archive
  U->>API: POST /api/agents/import (multipart/json)
  API->>I: select adapter + parse via contract
  I->>S: store immutable import job + items
  API->>S: emit audit/progress events
  API-->>U: import job + review state
  Operator->>U: Approve / quarantine / reject
  U->>API: POST /api/agents/import/{id}/decision
  API->>S: update item states + directory snapshot

  Operator->>U: Start/stop service
  U->>API: POST /api/runtime/services/{name}/start
  API->>R: start/stop/restart
  R->>B: direct import or HTTP bridge call
  R-->>API: runtime status/error
  API-->>U: runtime action result
```

## Notes

- Default API contract is defined in `main.py`.
- Persistence lives in a single SQLite file configured via `MANAGEMENT_DB_PATH`.
- Runtime control is best-effort in both paths:
  - direct core import when `BUSY_RUNTIME_PATH` is present
  - bridge mode when `BUSY_BRIDGE_URL` is configured
- Import processing is contract-first:
  canonical dataclasses -> sensitivity + visibility + review state -> storage + events.
- Import review is local-first:
  - raw sensitive content may remain available on the authenticated local review path,
  - security-generated redacted derivatives must be used for downstream agent/provider-facing summaries and artifacts.

## Adversarial failure modes to preserve

- `/api/*` routes must remain literal API surfaces. Unknown API paths fail as
  JSON 404s instead of falling through to the SPA shell.
- Parsing, schema validation, auth, and dispatch remain separate layers.
  Parsed structure never implies permission or side effects.
- Import review keeps the local-raw versus downstream-redacted split.
  Downstream summaries and derivatives must not source directly from raw
  sensitive payloads.

## Open questions

- None for the current slice. Record new authority ambiguities in
  `docs/internal/OPEN_QUESTIONS_AND_DECISIONS.md`; summarize behavior-shaping
  outcomes in `CURRENT_STATE.md`; add the relevant
  `docs/internal/*_CHANGE_REQUEST.md` note when an implementation slice needs
  design tracking.
