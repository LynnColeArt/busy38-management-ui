const API_BASE = (() => {
  const configured = (window.MANAGEMENT_API_BASE || "").trim();
  return configured !== "" ? configured : "http://127.0.0.1:8031";
})();

const TOKEN_KEY = "busy38-management-token";
const state = {
  settings: {},
  providers: [],
  agents: [],
  events: [],
  runtimeServices: [],
  role: "unknown",
  roleSource: "unknown",
  providerChain: [],
  providerDiagnostics: null,
  importJobs: [],
  importItems: [],
  selectedImportId: "",
};
let eventSocket = null;

const qs = (selector) => document.querySelector(selector);

function getToken() {
  return (qs("#apiToken")?.value || localStorage.getItem(TOKEN_KEY) || "").trim();
}

function saveToken() {
  const token = qs("#apiToken")?.value.trim() || "";
  if (token === "") {
    localStorage.removeItem(TOKEN_KEY);
    setStatus("#healthState", "token cleared", "");
    updateRoleBadge("unknown");
  } else {
    localStorage.setItem(TOKEN_KEY, token);
    setStatus("#healthState", "token saved", "ok");
    updateRoleBadge("unknown");
  }
  boot();
}

function escapeHtml(value) {
  const text = `${value}`;
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return text.replace(/[&<>"']/g, (char) => map[char]);
}

function setStatus(id, text, kind = "") {
  const el = qs(id);
  if (!el) {
    return;
  }
  el.textContent = text;
  el.className = `status ${kind}`;
}

function apiBaseForWs() {
  if (API_BASE.startsWith("https://")) {
    return API_BASE.replace(/^https:/, "wss:");
  }
  if (API_BASE.startsWith("http://")) {
    return API_BASE.replace(/^http:/, "ws:");
  }
  return API_BASE.replace(/^ws:/, "ws:").replace(/^wss:/, "wss:");
}

function authHeaders(overrides = {}) {
  const token = getToken();
  const headers = {
    "Content-Type": "application/json",
    ...overrides,
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

function authQuery(path) {
  const token = getToken();
  if (!token) {
    return path;
  }
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}token=${encodeURIComponent(token)}`;
}

async function apiRequest(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    const err = new Error(`${res.status} ${res.statusText}: ${text}`);
    err.status = res.status;
    throw err;
  }
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    return {};
  }
  return res.json();
}

async function fetchJson(path) {
  return apiRequest(path);
}

async function postForm(path, formData) {
  return apiRequest(authQuery(path), {
    method: "POST",
    headers: {},
    body: formData,
  });
}

async function postJson(path, body) {
  return apiRequest(authQuery(path), {
    method: "POST",
    body: JSON.stringify(body),
  });
}

async function patchJson(path, body) {
  return apiRequest(authQuery(path), {
    method: "PATCH",
    body: JSON.stringify(body),
  });
}

function renderCards(targetId, items, renderRow) {
  const container = qs(targetId);
  if (!container) {
    return;
  }
  const cards = (items || []).map(renderRow).join("");
  container.innerHTML = `<div class="card-list">${cards || "<p>No data</p>"}</div>`;
}

function formatImportStateCountLabel(counts) {
  const safe = counts || {};
  const total = Number(safe.total || 0);
  const pending = Number(safe.pending || 0);
  const approved = Number(safe.approved || 0);
  const quarantined = Number(safe.quarantined || 0);
  const rejected = Number(safe.rejected || 0);
  return `${total} total • ${pending} pending • ${approved} approved • ${quarantined} quarantined • ${rejected} rejected`;
}

function renderImportJobs(jobs) {
  const root = qs("#importJobs");
  if (!root) {
    return;
  }
  if (!Array.isArray(jobs) || jobs.length === 0) {
    root.innerHTML = "<p>No import jobs.</p>";
    return;
  }
  const cards = jobs
    .map((job) => {
      const counts = formatImportStateCountLabel(job.item_counts || {});
      const status = escapeHtml(job.status || "unknown");
      const source = escapeHtml(job.source_type || "unknown");
      const selected = state.selectedImportId === job.id;
      return `
        <div class="card${selected ? " selected" : ""}">
          <h3>Source ${source}</h3>
          <p><strong>Job:</strong> ${escapeHtml(job.id)}</p>
          <p><strong>Status:</strong> ${status}</p>
          <p><strong>Items:</strong> ${counts}</p>
          <p>
            <button type="button" data-action="open-import" data-id="${job.id}">
              Open review
            </button>
          </p>
        </div>
      `;
    })
    .join("");
  root.innerHTML = `<div class="card-list">${cards}</div>`;
}

function renderImportItems(items) {
  const root = qs("#importItems");
  if (!root) {
    return;
  }
  if (!Array.isArray(items) || items.length === 0) {
    root.innerHTML = "<p>No items for this import.</p>";
    return;
  }
  const cards = items
    .map((item) => {
      const metadata = item.metadata || {};
      const metadataLabel = metadata.summary || metadata.title || metadata.name || "";
      const reviewState = escapeHtml(item.review_state || "pending");
      const content = escapeHtml(String(item.content || "").slice(0, 600));
      const kind = escapeHtml(item.kind || "memory");
      const scope = escapeHtml(item.agent_scope || "global");
      const note = escapeHtml(item.note || "");
      return `
        <div class="card">
          <h3>${kind} — ${scope}</h3>
          <p><strong>State:</strong> ${reviewState}</p>
          <p><strong>Thread:</strong> ${escapeHtml(item.thread_id || "-")}</p>
          <p>${content}${item.content && item.content.length > 600 ? "…" : ""}</p>
          ${metadataLabel ? `<p><strong>Metadata:</strong> ${escapeHtml(metadataLabel)}</p>` : ""}
          ${note ? `<p><strong>Note:</strong> ${note}</p>` : ""}
          <p>
            <button type="button" data-action="import-item-decision" data-id="${item.id}" data-state="approved">Approve</button>
            <button type="button" data-action="import-item-decision" data-id="${item.id}" data-state="quarantined">Quarantine</button>
            <button type="button" data-action="import-item-decision" data-id="${item.id}" data-state="rejected">Reject</button>
          </p>
        </div>
      `;
    })
    .join("");
  root.innerHTML = `<div class="card-list">${cards}</div>`;
}

function getProviderFilterState() {
  return {
    kind: qs("#providerFilterKind")?.value || "",
    status: qs("#providerFilterStatus")?.value || "",
    secretStatus: qs("#providerFilterSecret")?.value || "",
    sortBy: qs("#providerSortBy")?.value || "priority",
    sortDesc: Boolean(qs("#providerSortDesc")?.checked),
  };
}

function renderProviderChain(chain) {
  const container = qs("#providerChain");
  if (!container) {
    return;
  }
  if (!Array.isArray(chain) || chain.length === 0) {
    container.innerHTML = "<p>No enabled providers in chain.</p>";
    return;
  }
  container.innerHTML = chain
    .map(
      (provider) => `
        <button class="chain-node ${provider.active ? "is-active" : ""}" type="button" data-action="show-provider-diagnostics" data-id="${provider.id}">
          <span>${escapeHtml(provider.name)} (${escapeHtml(provider.id)})</span>
          <span>${provider.active ? "active" : "fallback"}</span>
        </button>
      `
    )
    .join("");
}

function formatDateOrDash(value) {
  if (!value) {
    return "n/a";
  }
  return `${value}`;
}

function renderProviderDiagnosticsHtml(payload) {
  if (!payload) {
    return "<p>Diagnostics unavailable.</p>";
  }
  const metrics = payload.metrics || {};
  const history = Array.isArray(payload.test_history) ? payload.test_history : [];
  const rows = history
    .map((item) => `<li>${formatDateOrDash(item.tested_at)} — ${escapeHtml(item.status || "unknown")} (${item.models_count || 0} model(s), ${item.latency_ms || 0}ms)</li>`)
    .join("");

  return `
    <div class="status-grid">
      <div><strong>Provider:</strong> ${escapeHtml(payload.provider_id || "unknown")}</div>
      <div><strong>Latency last:</strong> ${metrics.latency_ms_last != null ? `${metrics.latency_ms_last}ms` : "n/a"}</div>
      <div><strong>Latency avg (5m):</strong> ${metrics.latency_ms_avg_5m != null ? `${metrics.latency_ms_avg_5m}ms` : "n/a"}</div>
      <div><strong>P95 latency (5m):</strong> ${metrics.latency_ms_p95_5m != null ? `${metrics.latency_ms_p95_5m}ms` : "n/a"}</div>
      <div><strong>Success rate (5m):</strong> ${metrics.success_rate_5m != null ? `${metrics.success_rate_5m}%` : "n/a"}</div>
      <div><strong>Failures (1m):</strong> ${metrics.failure_count_last_1m || 0}</div>
      <div><strong>Last checked:</strong> ${formatDateOrDash(metrics.last_checked_at || payload.last_test?.tested_at)}</div>
      <div><strong>Last error:</strong> ${escapeHtml(metrics.last_error_message || "n/a")}</div>
    </div>
    <h4>Recent checks</h4>
    <ul class="compact-list">
      ${rows || "<li>No checks run yet.</li>"}
    </ul>
  `;
}

function parseProviderMetadata(value) {
  if (value == null) {
    return {};
  }
  if (typeof value === "object") {
    return value;
  }
  try {
    return JSON.parse(value);
  } catch (err) {
    return {};
  }
}

function formatProviderStatusChip(status) {
  const stateText = `${status || "unknown"}`.toLowerCase();
  return `<span class="status-badge status-${stateText.replace(/[^a-z]+/g, "")}">${escapeHtml(stateText)}</span>`;
}

function updateRoleBadge(role, source) {
  const el = qs("#authRoleBadge");
  if (!el) {
    return;
  }
  const nextRole = role === "admin" ? "admin" : role === "viewer" ? "viewer" : role === "unauthorized" ? "unauthorized" : "unknown";
  const tokenSource = source || "unknown";
  state.role = nextRole;
  state.roleSource = tokenSource;
  el.textContent = `Role: ${nextRole}`;
  el.className = `role-badge role-${nextRole}`;
  el.title = `Token source: ${tokenSource}`;
}

function buildProviderCard(provider) {
  const metadata = parseProviderMetadata(provider.metadata);
  const discoveredModels = Array.isArray(metadata.discovered_models) ? metadata.discovered_models : [];
  const discovery = metadata.model_discovery || {};
  const discoveredEndpoint = escapeHtml(provider.endpoint || "");
  const lastTest = metadata.last_test || {};
  const secretPolicy = metadata.secret_policy || "required";
  const secretPresent = Boolean(metadata.secret_present);
  const testStatus = (lastTest.status || "").toLowerCase();
  const testText = lastTest.tested_at
    ? `Last test: ${testStatus || "unknown"} (${lastTest.models_count || 0} model(s), ${lastTest.latency_ms ? `${lastTest.latency_ms}ms` : "n/a"})`
    : "Last test: not run";
  const testError = lastTest.error ? `<p><strong>Test note:</strong> ${escapeHtml(lastTest.error)}</p>` : "";
  const isViewer = state.role === "viewer";
  const isLocked = isViewer ? "disabled" : "";
  const discoveredLabel = discoveredModels.length
    ? `<p><strong>Discovered models:</strong> ${discoveredModels
      .map((model) => `<button type="button" data-action="apply-discovered-model" data-id="${provider.id}" data-model="${escapeHtml(model)}" ${isLocked}>${escapeHtml(model)}</button>`)
      .join(" ")}</p>`
    : "<p><strong>Discovered models:</strong> none</p>";
  const discoveryText = discovery.status === "complete"
    ? `Auto-discovery from ${discovery.endpoint || "provider endpoint"}`
    : "Manual entry recommended";

  return `
    <div class="card">
      <h3>${provider.name}</h3>
      <p><strong>Status:</strong> ${formatProviderStatusChip(provider.status)} priority ${provider.priority}</p>
      <p><strong>Model:</strong> ${provider.model}</p>
      <p><strong>Endpoint:</strong> ${provider.endpoint || "n/a"}</p>
      <p><strong>Secret policy:</strong> ${escapeHtml(secretPolicy)}${secretPresent ? " (present)" : " (missing)"}</p>
      ${isViewer ? "<p><strong>Secret:</strong> read-only role</p>" : secretPolicy !== "none" ? `<p><strong>Secret action:</strong>
        <input type="password" data-action="provider-secret-key" data-id="${provider.id}" placeholder="api key" ${isLocked}/>
        <button type="button" data-action="set-provider-secret" data-id="${provider.id}" data-mode="${secretPresent ? "rotate" : "set"}" ${isLocked}>
          ${secretPresent ? "Rotate" : "Set"} secret
        </button>
        <button type="button" data-action="clear-provider-secret" data-id="${provider.id}" ${isLocked}>
          Clear
        </button>
      </p>` : `<p><strong>Secret:</strong> not required</p>`}
      <p><strong>Model discovery:</strong> ${discoveryText}</p>
      ${discovery.error ? `<p><strong>Discovery note:</strong> ${escapeHtml(discovery.error)}</p>` : ""}
      ${discoveredLabel}
      <p><strong>Provider test:</strong> ${escapeHtml(testText)}</p>
      ${testError}
      <p>
        <label>
          Discover endpoint (optional):
          <input
            type="text"
            data-action="provider-discovery-endpoint"
            data-id="${provider.id}"
            placeholder="e.g. https://api.openai.com/v1"
            value="${discoveredEndpoint}"${isLocked}
          />
        </label>
      </p>
      <p>
        <label>
          Private endpoint key (optional):
          <input
            type="password"
            data-action="provider-discovery-key"
            data-id="${provider.id}"
            placeholder="api key used for model list endpoint"${isLocked}
          />
        </label>
      </p>
      <p>
        <button type="button" data-action="test-provider-models" data-id="${provider.id}" ${isLocked}>Run provider test</button>
      </p>
      <p>
        <button type="button" data-action="discover-provider-models" data-id="${provider.id}" ${isLocked}>Discover models</button>
      </p>
      <p>
        <button type="button" data-action="show-provider-diagnostics" data-id="${provider.id}">
          View diagnostics
        </button>
      </p>
      <p>
        <label>
          Enabled:
          <input type="checkbox" data-action="toggle-provider" data-id="${provider.id}" ${provider.enabled ? "checked" : ""} ${isLocked} />
        </label>
      </p>
      <p>
        <label>
          Priority:
          <input type="number" data-action="set-provider-priority" data-id="${provider.id}" value="${provider.priority}" ${isLocked} />
        </label>
      </p>
      <p>
        <label>
          Model:
          <input type="text" data-action="set-provider-model" data-id="${provider.id}" value="${escapeHtml(provider.model || "")}" placeholder="manually set model"${isLocked} />
        </label>
      </p>
    </div>
  `;
}

function buildAgentCard(agent) {
  return `
    <div class="card">
      <h3>${agent.name}</h3>
      <p><strong>Status:</strong> ${agent.status}</p>
      <p><strong>Role:</strong> ${agent.role}</p>
      <p><strong>Last active:</strong> ${agent.last_active_at || "unknown"}</p>
      <p>
        <label>
          Enabled:
          <input type="checkbox" data-action="toggle-agent" data-id="${agent.id}" ${agent.enabled ? "checked" : ""} />
        </label>
      </p>
    </div>
  `;
}

function renderRuntimeServices(services) {
  const hasServices = Array.isArray(services) && services.length > 0;
  const options = hasServices
    ? services.map((service) => `<option value="${service.name}">${service.name}</option>`).join("")
    : "<option value=\"busy\">busy</option>";

  const select = qs("#runtimeService");
  if (select) {
    select.innerHTML = options;
    if (!select.value && hasServices) {
      select.value = "busy";
    }
  }

  const renderService = (service) => `
    <div class="card">
      <h3>${service.name}</h3>
      <p><strong>Running:</strong> ${service.running ? "yes" : "no"}</p>
      <p><strong>PID:</strong> ${service.pid || "-"}</p>
      <p><strong>Log:</strong> ${service.log_file || "-"}</p>
    </div>
  `;

  renderCards("#runtimeServices", services || [], renderService);
}

function renderEvents(items) {
  const list = qs("#events");
  if (!list) {
    return;
  }
  list.innerHTML = (items || [])
    .map((event) => `<li>${event.created_at} — [${event.level}] ${event.message}</li>`)
    .join("");
}

function renderMemory(items) {
  renderCards("#memory", items || [], (row) => `
    <div class="card">
      <h3>${row.type}</h3>
      <p><strong>Scope:</strong> ${row.scope}</p>
      <p>${row.content}</p>
      <small>${row.timestamp}</small>
    </div>
  `);
}

function renderChat(rows) {
  const list = qs("#chatHistory");
  if (!list) {
    return;
  }
  list.innerHTML = (rows || [])
    .map(
      (row) => `<li><strong>${row.agent_id}</strong> (${row.id}): ${row.summary} — <small>${row.timestamp}</small></li>`
    )
    .join("");
}

function connectEvents() {
  if (eventSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(eventSocket.readyState)) {
    return;
  }

  const token = getToken();
  const query = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(`${apiBaseForWs()}/api/events/ws${query}`);
  eventSocket = ws;
  setStatus("#eventStreamState", "Event stream: connecting", "");

  ws.onopen = () => {
    setStatus("#eventStreamState", "Event stream: live", "ok");
  };

  ws.onmessage = (evt) => {
    let payload = null;
    try {
      payload = JSON.parse(evt.data);
    } catch (err) {
      return;
    }
    if (payload?.type === "events" && Array.isArray(payload.events)) {
      if (payload?.role) {
        updateRoleBadge(payload.role, payload?.role_source);
      }
      state.events = payload.events;
      renderEvents(payload.events);
    }
  };

  ws.onclose = () => {
    setStatus("#eventStreamState", "Event stream: disconnected", "err");
    setTimeout(connectEvents, 2000);
  };

  ws.onerror = () => {
    setStatus("#eventStreamState", "Event stream: error", "err");
    ws.close();
  };
}

async function loadHealth() {
  try {
    const payload = await fetchJson("/api/health");
    setStatus("#healthState", `service: ${payload.service} (${payload.status})`, "ok");
  } catch (err) {
    setStatus("#healthState", `health check failed: ${err.message}`, "err");
  }
}

async function loadSettings() {
  const payload = await fetchJson("/api/settings");
  state.settings = payload.settings || {};
  qs('input[name="heartbeat_interval"]').value = state.settings.heartbeat_interval || "";
  qs('input[name="fallback_budget_per_hour"]').value = state.settings.fallback_budget_per_hour || "";
  qs('input[name="auto_restart"]').checked = Boolean(state.settings.auto_restart);
  updateRoleBadge(payload.role, payload.role_source);
  return payload;
}

async function loadProviders() {
  const filters = getProviderFilterState();
  const params = new URLSearchParams();
  if (filters.kind) {
    params.set("kind", filters.kind);
  }
  if (filters.status) {
    params.set("status", filters.status);
  }
  if (filters.secretStatus) {
    params.set("secret_status", filters.secretStatus);
  }
  if (filters.sortBy) {
    params.set("sort_by", filters.sortBy);
  }
  if (filters.sortDesc) {
    params.set("sort_desc", "true");
  }
  const query = params.toString();
  const payload = await fetchJson(`/api/providers${query ? `?${query}` : ""}`);
  state.providers = payload.providers || [];
  renderCards("#providers", state.providers, buildProviderCard);
}

async function loadProviderChain() {
  const payload = await fetchJson("/api/providers/routing-chain");
  state.providerChain = payload.chain || [];
  renderProviderChain(state.providerChain);
}

async function loadProviderDiagnostics(providerId) {
  const payload = await fetchJson(`/api/providers/${providerId}/metrics`);
  const historyPayload = await fetchJson(`/api/providers/${providerId}/history`);
  const merged = {
    provider_id: providerId,
    metrics: payload.metrics || {},
    last_test: payload.last_test || {},
    test_history: historyPayload.test_history || [],
  };
  state.providerDiagnostics = merged;
  const diagnosticsRoot = qs("#providerDiagnostics");
  if (diagnosticsRoot) {
    diagnosticsRoot.innerHTML = renderProviderDiagnosticsHtml(merged);
  }
}

async function loadAgents() {
  const payload = await fetchJson("/api/agents");
  state.agents = payload.agents || [];
  renderCards("#agents", state.agents, buildAgentCard);
}

async function loadEvents() {
  const payload = await fetchJson("/api/events");
  state.events = payload.events || [];
  renderEvents(state.events);
}

async function loadRuntime() {
  try {
    const payload = await fetchJson("/api/runtime/status");
    const status = payload.runtime || {};
    const source = status.source || "none";
    const connected = Boolean(status.connected);
    setStatus("#runtimeStatus", `runtime (${source}) — connected: ${connected ? "yes" : "no"}`, "ok");

    const servicesPayload = await fetchJson("/api/runtime/services");
    state.runtimeServices = servicesPayload.services || [];
    renderRuntimeServices(state.runtimeServices);
  } catch (err) {
    setStatus("#runtimeStatus", `runtime unavailable: ${err.message}`, "err");
    state.runtimeServices = [];
    renderRuntimeServices([]);
  }
}

async function loadMemory() {
  const payload = await fetchJson("/api/memory");
  renderMemory(payload.memory || []);
}

async function loadChatHistory() {
  const payload = await fetchJson("/api/chat_history");
  renderChat(payload.chat_history || []);
}

async function loadImportJobs() {
  const payload = await fetchJson("/api/agents/imports");
  state.importJobs = payload.imports || [];
  if (state.selectedImportId && !state.importJobs.some((job) => job.id === state.selectedImportId)) {
    state.selectedImportId = "";
  }
  renderImportJobs(state.importJobs);
  if (!state.selectedImportId) {
    const header = qs("#importReviewHeader");
    if (header) {
      header.textContent = "Select an import job to review";
    }
  }
}

async function loadImportItems(importId, reviewState = "") {
  if (!importId) {
    state.importItems = [];
    renderImportItems(state.importItems);
    return;
  }
  const query = reviewState ? `?review_state=${encodeURIComponent(reviewState)}` : "";
  const payload = await fetchJson(`/api/agents/import/${encodeURIComponent(importId)}${query}`);
  state.importItems = payload.items || [];
  renderImportItems(state.importItems);
  const header = qs("#importReviewHeader");
  if (header) {
    const sourceType = payload.job?.source_type ? ` (${payload.job.source_type})` : "";
    header.textContent = `Reviewing import ${importId}${sourceType}`;
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const autoRestart = qs('input[name="auto_restart"]').checked;
  const heartbeat = Number(qs('input[name="heartbeat_interval"]').value);
  const fallback = Number(qs('input[name="fallback_budget_per_hour"]').value);
  const payload = {
    heartbeat_interval: heartbeat || state.settings.heartbeat_interval,
    fallback_budget_per_hour: fallback || state.settings.fallback_budget_per_hour,
    auto_restart: autoRestart,
  };

  setStatus("#settingsStatus", "saving", "");
  try {
    await patchJson("/api/settings", payload);
    setStatus("#settingsStatus", "settings saved", "ok");
    await loadSettings();
  } catch (err) {
    setStatus("#settingsStatus", `save failed: ${err.message}`, "err");
  }
}

async function submitMemory(event) {
  event.preventDefault();
  const scope = qs('input[name="memoryScope"]').value.trim();
  const type = qs('input[name="memoryType"]').value.trim();
  const content = qs('input[name="memoryContent"]').value.trim();
  if (!scope || !type || !content) {
    setStatus("#memoryStatus", "memory form requires scope, type, and content", "err");
    return;
  }

  try {
    await postJson("/api/memory", { scope, type, content });
    setStatus("#memoryStatus", "memory saved", "ok");
    qs('input[name="memoryContent"]').value = "";
    await loadMemory();
  } catch (err) {
    setStatus("#memoryStatus", `save failed: ${err.message}`, "err");
  }
}

async function submitChat(event) {
  event.preventDefault();
  const agent_id = qs('input[name="chatAgentId"]').value.trim();
  const summary = qs('input[name="chatSummary"]').value.trim();
  if (!agent_id || !summary) {
    setStatus("#chatStatus", "chat form requires agent id and summary", "err");
    return;
  }

  try {
    await postJson("/api/chat_history", { agent_id, summary });
    setStatus("#chatStatus", "chat note added", "ok");
    qs('input[name="chatSummary"]').value = "";
    await loadChatHistory();
  } catch (err) {
    setStatus("#chatStatus", `save failed: ${err.message}`, "err");
  }
}

async function submitImport(event) {
  event.preventDefault();
  const sourceType = qs('select[name="sourceType"]')?.value || "openai";
  const fileInput = qs('input[name="sourceFile"]');
  const appendToLatest = Boolean(qs('input[name="appendToLatest"]')?.checked);

  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    setStatus("#importSubmitStatus", "source file is required", "err");
    return;
  }
  const sourceFile = fileInput.files[0];
  if (!sourceFile) {
    setStatus("#importSubmitStatus", "source file is required", "err");
    return;
  }
  const form = new FormData();
  form.append("source_type", sourceType);
  form.append("append_to_latest", String(appendToLatest));
  form.append("source_file", sourceFile);

  setStatus("#importSubmitStatus", "uploading...", "");
  try {
    const payload = await postForm("/api/agents/import", form);
    setStatus("#importSubmitStatus", `import queued: ${payload.import_id}`, "ok");
    await loadImportJobs();
    await loadImportItems(payload.import_id);
  } catch (err) {
    setStatus("#importSubmitStatus", `import failed: ${err.message}`, "err");
  }
}

async function setImportItemState(itemId, stateValue) {
  if (!state.selectedImportId) {
    setStatus("#importSubmitStatus", "select an import job before reviewing", "err");
    return;
  }
  const itemPayload = state.importItems.find((item) => item.id === itemId) || {};
  const metadata = itemPayload.metadata || {};
  const requiresReviewNote = stateValue !== "quarantined" && (metadata.sensitive || metadata.requires_review || itemPayload.review_state === "sensitive");
  let note = "";
  if (requiresReviewNote) {
    note = window.prompt("Review note required for sensitive item:") || "";
    if (!note.trim()) {
      setStatus("#importSubmitStatus", "review note required for sensitive item", "err");
      return;
    }
  } else {
    note = window.prompt("Optional note for this decision:") || "";
  }

  setStatus("#importSubmitStatus", "updating import decision", "");
  try {
    await postJson(`/api/agents/import/${encodeURIComponent(state.selectedImportId)}/decision`, {
      import_item_ids: [itemId],
      review_state: stateValue,
      note: note || undefined,
      actor: state.role,
    });
    setStatus("#importSubmitStatus", `item ${itemId} marked ${stateValue}`, "ok");
    await loadImportItems(state.selectedImportId);
    await loadImportJobs();
  } catch (err) {
    setStatus("#importSubmitStatus", `decision update failed: ${err.message}`, "err");
  }
}

async function submitProvider(event) {
  event.preventDefault();
  const provider_id = qs('input[name="providerId"]').value.trim();
  const name = qs('input[name="providerName"]').value.trim();
  const endpoint = qs('input[name="providerEndpoint"]').value.trim();
  const model = qs('input[name="providerModel"]').value.trim();
  const kind = qs('input[name="providerKind"]').value.trim();
  const priority = Number(qs('input[name="providerPriority"]').value || "100");
  const enabled = qs('input[name="providerEnabled"]').checked;

  if (!provider_id || !name || !endpoint || !model) {
    setStatus("#providerCreateStatus", "provider id, name, endpoint, and model are required", "err");
    return;
  }

  const payload = {
    id: provider_id,
    name,
    endpoint,
    model,
    priority,
    enabled,
    metadata: {},
  };
  if (kind) {
    payload.metadata.kind = kind;
  }

  setStatus("#providerCreateStatus", "creating provider...", "");
  try {
    await postJson("/api/providers", payload);
    setStatus("#providerCreateStatus", `created ${provider_id}`, "ok");
    qs('input[name="providerId"]').value = "";
    qs('input[name="providerName"]').value = "";
    qs('input[name="providerEndpoint"]').value = "";
    qs('input[name="providerModel"]').value = "";
    qs('input[name="providerKind"]').value = "";
    qs('input[name="providerPriority"]').value = "";
    qs('input[name="providerEnabled"]').checked = true;
    await Promise.all([loadProviders(), loadProviderChain()]);
  } catch (err) {
    setStatus("#providerCreateStatus", `create failed: ${err.message}`, "err");
  }
}

async function controlRuntime(action) {
  const service = qs("#runtimeService")?.value || "busy";
  const body = {};
  try {
    await postJson(`/api/runtime/services/${service}/${action}`, body);
    setStatus("#runtimeStatusText", `${action} submitted for ${service}`, "ok");
    await loadRuntime();
  } catch (err) {
    setStatus("#runtimeStatusText", `${action} failed: ${err.message}`, "err");
  }
}

async function onTableChange(event) {
  const target = event.target;
  const action = target?.dataset?.action;
  if (!action) {
    return;
  }

  if (action === "runtime-start") {
    await controlRuntime("start");
    return;
  }

  if (action === "runtime-stop") {
    await controlRuntime("stop");
    return;
  }

  if (action === "runtime-restart") {
    await controlRuntime("restart");
    return;
  }

  if (action === "show-provider-diagnostics") {
    const diagnosticsId = target.dataset.id;
    const diagnosticsRoot = qs("#providerDiagnostics");
    try {
      if (!diagnosticsId) {
        if (diagnosticsRoot) {
          diagnosticsRoot.innerHTML = "<p>Unable to load diagnostics: provider id missing.</p>";
        }
        return;
      }
      setStatus("#providerChainStatus", "loading diagnostics...", "");
      await loadProviderDiagnostics(diagnosticsId);
      setStatus("#providerChainStatus", `Diagnostics loaded for ${diagnosticsId}`, "ok");
    } catch (err) {
      if (diagnosticsRoot) {
        diagnosticsRoot.innerHTML = `<p>Could not load diagnostics for ${escapeHtml(diagnosticsId)}: ${escapeHtml(err.message)}</p>`;
      }
      setStatus("#providerChainStatus", `Could not load diagnostics: ${err.message}`, "err");
    }
    return;
  }

  if (action === "open-import") {
    const importId = target.dataset.id;
    if (!importId) {
      setStatus("#importSubmitStatus", "import id missing", "err");
      return;
    }
    state.selectedImportId = importId;
    await loadImportItems(importId);
    return;
  }

  if (action === "import-item-decision") {
    const itemId = target.dataset.id;
    const decision = target.dataset.state || "approved";
    if (!itemId) {
      setStatus("#importSubmitStatus", "import item id missing", "err");
      return;
    }
    await setImportItemState(itemId, decision);
    return;
  }

  if (state.role === "viewer") {
    setStatus("#healthState", "viewer role: read-only", "err");
    return;
  }

  try {
    if (action === "test-all-providers") {
      setStatus("#providerChainStatus", "running provider checks...", "");
      try {
        const result = await postJson("/api/providers/test-all", {});
        setStatus("#providerChainStatus", `Checks: ${result.pass_count} passed, ${result.fail_count} failed of ${result.checked}`, result.fail_count ? "err" : "ok");
      } catch (err) {
        setStatus("#providerChainStatus", `Provider checks failed: ${err.message}`, "err");
      }
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "toggle-provider") {
      const id = target.dataset.id;
      await patchJson(`/api/providers/${id}`, { enabled: target.checked });
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "set-provider-priority") {
      const id = target.dataset.id;
      await patchJson(`/api/providers/${id}`, { priority: Number(target.value) || 0 });
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "set-provider-model") {
      const id = target.dataset.id;
      await patchJson(`/api/providers/${id}`, { model: target.value.trim() });
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "apply-discovered-model") {
      const id = target.dataset.id;
      const model = target.dataset.model || "";
      await patchJson(`/api/providers/${id}`, { model });
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "discover-provider-models") {
      const id = target.dataset.id;
      const endpointInput = qs(`input[data-action="provider-discovery-endpoint"][data-id="${id}"]`);
      const keyInput = qs(`input[data-action="provider-discovery-key"][data-id="${id}"]`);
      const discoveryPayload = {};
      if (endpointInput?.value?.trim()) {
        discoveryPayload.endpoint = endpointInput.value.trim();
      }
      if (keyInput?.value?.trim()) {
        discoveryPayload.api_key = keyInput.value.trim();
      }
      setStatus("#healthState", "discovering models...", "");
      try {
        const result = await postJson(`/api/providers/${id}/discover-models`, discoveryPayload);
        if (keyInput) {
          keyInput.value = "";
        }
        if (Array.isArray(result?.discovered_models) && result.discovered_models.length > 0) {
          setStatus("#healthState", `Discovered ${result.discovered_models.length} model(s)`, "ok");
        } else if (result?.message) {
          setStatus("#healthState", result.message, "err");
        } else {
          setStatus("#healthState", "Discovery returned no models. Enter model manually.", "err");
        }
      } catch (err) {
        setStatus("#healthState", `Discovery failed: ${err.message}`, "err");
      }
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "test-provider-models") {
      const id = target.dataset.id;
      setStatus("#healthState", "testing provider...", "");
      try {
        const result = await postJson(`/api/providers/${id}/test`, {});
        if (result?.status === "pass") {
          setStatus("#healthState", `Provider test passed in ${result.latency_ms}ms`, "ok");
        } else {
          setStatus("#healthState", `Provider test failed: ${result.error || "see provider card"}`, "err");
        }
      } catch (err) {
        setStatus("#healthState", `Provider test failed: ${err.message}`, "err");
      }
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "set-provider-secret") {
      const id = target.dataset.id;
      const mode = target.dataset.mode || "set";
      const keyInput = qs(`input[data-action="provider-secret-key"][data-id="${id}"]`);
      const value = keyInput?.value?.trim() || "";
      if (!value) {
        setStatus("#healthState", "provider secret requires a key", "err");
        return;
      }
      setStatus("#healthState", "updating provider secret...", "");
      try {
        const result = await postJson(`/api/providers/${id}/secret`, {
          action: mode,
          api_key: value,
        });
        if (keyInput) {
          keyInput.value = "";
        }
        const secret = result?.provider?.metadata?.secret_present;
        setStatus("#healthState", secret ? "Provider secret set" : "Provider secret updated", "ok");
      } catch (err) {
        setStatus("#healthState", `Secret update failed: ${err.message}`, "err");
      }
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "clear-provider-secret") {
      const id = target.dataset.id;
      setStatus("#healthState", "clearing provider secret...", "");
      try {
        const result = await postJson(`/api/providers/${id}/secret`, { action: "clear" });
        const secret = result?.provider?.metadata?.secret_present;
        setStatus("#healthState", secret ? "Provider secret still present" : "Provider secret cleared", secret ? "err" : "ok");
      } catch (err) {
        setStatus("#healthState", `Secret clear failed: ${err.message}`, "err");
      }
      await Promise.all([loadProviders(), loadProviderChain()]);
      return;
    }

    if (action === "toggle-agent") {
      const id = target.dataset.id;
      await patchJson(`/api/agents/${id}`, { enabled: target.checked });
      await loadAgents();
      return;
    }
  } catch (err) {
    setStatus("#healthState", `update failed: ${err.message}`, "err");
  }
}

async function boot() {
  const token = getToken();
  const tokenInput = qs("#apiToken");
  if (tokenInput && token && !tokenInput.value) {
    tokenInput.value = token;
  }

  try {
    await loadSettings();
    await Promise.all([
      loadHealth(),
      loadProviders(),
      loadProviderChain(),
      loadImportJobs(),
      loadAgents(),
      loadEvents(),
      loadMemory(),
      loadChatHistory(),
      loadRuntime(),
    ]);
    connectEvents();
  } catch (err) {
    if (err?.status === 401) {
      updateRoleBadge("unauthorized", "invalid-or-missing-token");
    } else if (err?.status === 403) {
      updateRoleBadge("viewer", "read-token");
    } else if (state.role === "unknown") {
      updateRoleBadge("unknown");
    }
    setStatus("#healthState", `loading failed: ${err.message}`, "err");
  }
}

qs("#saveToken")?.addEventListener("click", saveToken);
qs("#settingsForm")?.addEventListener("submit", saveSettings);
qs("#memoryForm")?.addEventListener("submit", submitMemory);
qs("#chatForm")?.addEventListener("submit", submitChat);
qs("#importForm")?.addEventListener("submit", submitImport);
qs("#providerCreateForm")?.addEventListener("submit", submitProvider);
qs("#providerFilterKind")?.addEventListener("change", loadProviders);
qs("#providerFilterStatus")?.addEventListener("change", loadProviders);
qs("#providerFilterSecret")?.addEventListener("change", loadProviders);
qs("#providerSortBy")?.addEventListener("change", loadProviders);
qs("#providerSortDesc")?.addEventListener("change", loadProviders);
document.body.addEventListener("change", onTableChange);
document.body.addEventListener("click", (event) => {
  const target = event.target;
  const action = target?.dataset?.action;
  if (!action || (
    !action.startsWith("runtime-")
    && action !== "discover-provider-models"
    && action !== "open-import"
    && action !== "apply-discovered-model"
    && action !== "test-provider-models"
    && action !== "test-all-providers"
    && action !== "set-provider-secret"
    && action !== "clear-provider-secret"
    && action !== "import-item-decision"
    && action !== "show-provider-diagnostics"
  )) {
    return;
  }
  onTableChange(event);
});

boot();
setInterval(boot, 15000);
