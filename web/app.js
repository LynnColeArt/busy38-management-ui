const API_BASE = (() => {
  const configured = (window.MANAGEMENT_API_BASE || "").trim();
  return configured !== "" ? configured : "http://127.0.0.1:8031";
})();

const TOKEN_KEY = "busy38-management-token";
const state = {
  settings: {},
  providers: [],
  plugins: [],
  agents: [],
  events: [],
  runtimeServices: [],
  role: "unknown",
  roleSource: "unknown",
  providerChain: [],
  providerDiagnostics: null,
  importJobs: [],
  importItems: [],
  importAudit: null,
  agentDirectory: [],
  agentDirectoryArtifact: null,
  selectedImportId: "",
  rerunImportId: "",
  toolUsage: [],
  toolUsageCount: 0,
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
      const actorId = escapeHtml(job.source_actor_id || "n/a");
      const missionId = escapeHtml(job.source_mission_id || "n/a");
      const schemaVersion = escapeHtml(job.context_schema_version || "2");
      const selected = state.selectedImportId === job.id;
      return `
        <div class="card${selected ? " selected" : ""}">
          <h3>Source ${source}</h3>
          <p><strong>Job:</strong> ${escapeHtml(job.id)}</p>
          <p><strong>Status:</strong> ${status}</p>
          <p><strong>Items:</strong> ${counts}</p>
          <p><strong>Actor:</strong> ${actorId}</p>
          <p><strong>Mission:</strong> ${missionId}</p>
          <p><strong>Context schema:</strong> ${schemaVersion}</p>
          <p>
            <button type="button" data-action="open-import" data-id="${job.id}">
              Open review
            </button>
            <button type="button" data-action="rerun-import" data-id="${job.id}">
              Rerun
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
            <label>
              Reassign scope:
              <input type="text" class="import-item-scope" data-id="${item.id}" value="${scope}" />
            </label>
          </p>
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

function renderImportAudit(audit) {
  const root = qs("#importAudit");
  if (!root) {
    return;
  }
  if (!audit || !audit.job) {
    root.innerHTML = "<p>Select an import job to review its audit trail.</p>";
    return;
  }

  const lineage = audit.lineage || [];
  const timeline = audit.timeline || [];
  const lineageBlock = lineage.length
    ? lineage
        .map((entry, index) => {
          const details = [
            `source: ${escapeHtml(entry.source_type || "unknown")}`,
            `status: ${escapeHtml(entry.status || "unknown")}`,
            `actor: ${escapeHtml(entry.source_actor_id || "n/a")}`,
            `mission: ${escapeHtml(entry.source_mission_id || "n/a")}`,
          ]
            .filter(Boolean)
            .join("<br>");
          const parent = entry.rerun_of_import_id
            ? `<p><strong>Parent rerun:</strong> ${escapeHtml(entry.rerun_of_import_id)}</p>`
            : "";
          return `
            <div class="card">
              <h3>${index + 1}. Import ${escapeHtml(entry.import_id || "unknown")}</h3>
              <p>${details}</p>
              ${parent}
              <p><strong>Created:</strong> ${escapeHtml(entry.created_at || "n/a")}</p>
            </div>
          `;
        })
        .join("")
    : "<p>No lineage found.</p>";

  const eventsBlock = timeline.length
    ? timeline
        .map((event) => {
          const phase = event.phase || "event";
          const created = event.created_at || "n/a";
          const details = event.details || {};
          const detailText = escapeHtml(JSON.stringify(details).replaceAll("\"", "'"));
          return `
            <div class="card">
              <p><strong>${escapeHtml(event.event_type || "event")} · ${escapeHtml(phase)}</strong></p>
              <p><strong>At:</strong> ${escapeHtml(created)}</p>
              <p><strong>Details:</strong> ${detailText}</p>
            </div>
          `;
        })
        .join("")
    : "<p>No events yet.</p>";

  root.innerHTML = `
    <div class="card-list">
      <div class="card">
        <h3>Import ${escapeHtml(audit.job.id || "unknown")}</h3>
        <p><strong>Source:</strong> ${escapeHtml(audit.job.source_type || "unknown")}</p>
        <p><strong>Status:</strong> ${escapeHtml(audit.job.status || "unknown")}</p>
        <p><strong>Actor:</strong> ${escapeHtml(audit.job.source_actor_id || "n/a")}</p>
        <p><strong>Mission:</strong> ${escapeHtml(audit.job.source_mission_id || "n/a")}</p>
      </div>
      <h3>Lineage</h3>
      ${lineageBlock}
      <h3>Timeline</h3>
      ${eventsBlock}
    </div>
  `;
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
  const summary = qs("#providerChainSummary");
  const routingPath = qs("#providerRoutingPath");
  if (!container) {
    return;
  }
  const metadata = chain?.selection_strategy
    ? chain
    : {
      chain: chain || [],
      selection_strategy: "Loading provider routing strategy…",
      selection_rationale: "No chain metadata available yet.",
      active_provider_id: null,
      routing_intent: "loading",
      routing_path: "",
    };
  const nodes = Array.isArray(metadata.chain) ? metadata.chain : [];
  if (!Array.isArray(nodes) || nodes.length === 0) {
    container.innerHTML = "<p>No enabled providers in chain.</p>";
    if (summary) {
      summary.textContent = metadata.selection_strategy || "No enabled providers in chain.";
    }
    if (routingPath) {
      routingPath.textContent = "No routing path configured.";
    }
    return;
  }
  if (summary) {
    const rationale = metadata.selection_rationale || "Fallback routing is based on priority order.";
    const activeProvider = metadata.active_provider_id || nodes[0]?.id || "none";
    const totals = `${metadata.enabled_count ?? nodes.length}/${metadata.total_count ?? nodes.length}`;
    const intent = metadata.routing_intent || "primary then fallback";
    summary.textContent = `${metadata.selection_strategy}. Active provider: ${activeProvider}. Active/total: ${totals}. Intent: ${intent}. ${rationale}`;
  }
  if (routingPath) {
    const pathValue = metadata.routing_path || `${nodes.map((provider) => `${provider.id}`).join(" -> ")} (${nodes.length} nodes)`;
    routingPath.textContent = `Routing path: ${pathValue}`;
  }
  container.innerHTML = nodes
    .map(
      (provider) => `
        <button class="chain-node ${provider.active ? "is-active" : ""}" type="button" data-action="show-provider-diagnostics" data-id="${provider.id}">
          <span>${escapeHtml(provider.name)} (${escapeHtml(provider.id)})</span>
          <span>${provider.active ? "active" : "fallback"}</span>
          <small>${escapeHtml(provider.routing_reason || "fallback candidate")}</small>
          <small>${escapeHtml(provider.routing_behavior || "normal route")}</small>
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
  const displayName = metadata.display_name || provider.name || "Unnamed provider";
  const kind = metadata.kind || "other";
  const fallbackModels = Array.isArray(metadata.fallback_models) ? metadata.fallback_models : [];
  const retries = metadata.retries == null ? "default" : String(metadata.retries);
  const timeout = metadata.timeout_ms == null ? "default" : `${metadata.timeout_ms}ms`;
  const toolTimeout = metadata.tool_timeout_ms == null ? "default" : `${metadata.tool_timeout_ms}ms`;
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
      <h3>${escapeHtml(displayName)}</h3>
      <p><strong>Kind:</strong> ${escapeHtml(kind)}</p>
      <p><strong>Status:</strong> ${formatProviderStatusChip(provider.status)} priority ${provider.priority}</p>
      <p><strong>Model:</strong> ${provider.model}</p>
      <p><strong>Endpoint:</strong> ${provider.endpoint || "n/a"}</p>
      <p><strong>Fallback models:</strong> ${fallbackModels.length ? fallbackModels.map((item) => escapeHtml(item)).join(", ") : "none"}</p>
      <p><strong>Retries:</strong> ${escapeHtml(retries)}</p>
      <p><strong>Timeouts:</strong> request ${escapeHtml(timeout)} • tool ${escapeHtml(toolTimeout)}</p>
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
  const overlay = agent.overlay || {};
  const overlayFound = Boolean(overlay.found);
  const overlayData = (overlay.overlay && typeof overlay.overlay === "object") ? overlay.overlay : {};
  const isAdmin = state.role === "admin";
  const overlayContent = isAdmin ? `${overlayData.content || ""}` : `${overlayData.content_preview || ""}`;
  const overlayCap = overlayData.requested_token_cap != null ? String(overlayData.requested_token_cap) : "";
  const overlaySource = overlayData.source || "runtime";

  return `
    <div class="card">
      <h3>${agent.name}</h3>
      <p><strong>Status:</strong> ${agent.status}</p>
      <p><strong>Role:</strong> ${agent.role}</p>
      <p><strong>Last active:</strong> ${agent.last_active_at || "unknown"}</p>
      <p><strong>Identity overlay:</strong> ${overlayFound ? "configured" : "not configured"} (${overlaySource})</p>
      ${isAdmin ? `
      <p>
        <label>
          Overlay token cap:
          <input
            type="number"
            data-action="agent-overlay-token-cap"
            data-id="${agent.id}"
            value="${escapeHtml(overlayCap)}"
            min="1"
          />
        </label>
      </p>
      <p>
        <label>
          Overlay identity:
          <textarea
            data-action="agent-overlay-content"
            data-id="${agent.id}"
            rows="6"
            placeholder="Optional overlay text for this agent"
          >${escapeHtml(overlayContent)}</textarea>
        </label>
      </p>
      <p>
        <button type="button" data-action="save-agent-overlay" data-id="${agent.id}">
          Save overlay
        </button>
      </p>
      ` : `
      <p><strong>Overlay preview:</strong> ${overlayFound ? escapeHtml(overlayContent || "none") : "none"}</p>
      `}
      <p>
        <label>
          Enabled:
          <input type="checkbox" data-action="toggle-agent" data-id="${agent.id}" ${agent.enabled ? "checked" : ""} />
        </label>
      </p>
      <p>
        <button type="button" data-action="open-agent-tool-usage" data-id="${agent.id}">
          Audit tool usage
        </button>
      </p>
    </div>
  `;
}

function getDirectoryScopeOptions() {
  const seen = new Set(["global"]);
  const options = ["global"];
  for (const agent of state.agents || []) {
    const candidate = String(agent.id || "").trim();
    if (candidate && !seen.has(candidate)) {
      seen.add(candidate);
      options.push(candidate);
    }
  }
  return options;
}

function buildDirectoryScopeSelect(currentScope) {
  const normalizedCurrent = (currentScope || "global").trim() || "global";
  const options = getDirectoryScopeOptions()
    .map((option) => {
      const value = escapeHtml(option);
      const selected = option === normalizedCurrent ? " selected" : "";
      return `<option value="${value}"${selected}>${value}</option>`;
    })
    .join("");
  return `
    <p>
      Reassign all responsibilities:
      <select class="directory-scope-select" data-source-scope="${escapeHtml(normalizedCurrent)}">
        ${options}
      </select>
      <input class="directory-scope-custom" data-source-scope="${escapeHtml(normalizedCurrent)}" type="text" placeholder="or custom scope" />
      <button type="button" data-action="reassign-directory-scope" data-source-scope="${escapeHtml(normalizedCurrent)}">Apply</button>
    </p>
  `;
}

function getDirectoryScopeControlNodes(sourceScope, trigger) {
  const normalizedSource = (sourceScope || "global").trim() || "global";

  if (trigger && typeof trigger.closest === "function") {
    const card = trigger.closest(".card");
    if (card) {
      const scopedSelect = card.querySelector("select.directory-scope-select");
      const scopedCustom = card.querySelector("input.directory-scope-custom");
      if (
        scopedSelect &&
        String(scopedSelect.dataset.sourceScope || "").trim() === normalizedSource
      ) {
        return { select: scopedSelect, custom: scopedCustom };
      }
    }
  }

  const cards = document.querySelectorAll(".card");
  for (const card of cards) {
    const scopedSelect = card.querySelector("select.directory-scope-select");
    if (!scopedSelect) {
      continue;
    }
    const candidate = String(scopedSelect.dataset.sourceScope || "").trim() || "global";
    if (candidate === normalizedSource) {
      return {
        select: scopedSelect,
        custom: card.querySelector("input.directory-scope-custom"),
      };
    }
  }

  return { select: null, custom: null };
}

function buildAgentDirectoryCard(directoryEntry) {
  const scope = escapeHtml(directoryEntry.agent_scope || "global");
  const responsibilityCount = Number(directoryEntry.item_count || 0);
  const responsibilities = Array.isArray(directoryEntry.responsibilities) ? directoryEntry.responsibilities : [];
  const responsibilityHtml = responsibilities
    .map((responsibility) => {
      const title = escapeHtml(responsibility.title || "Responsibility");
      const kind = escapeHtml(responsibility.kind || "memory");
      const summary = escapeHtml(responsibility.summary || "");
      const details = summary ? ` — ${summary}` : "";
      return `<li><strong>${title}</strong> (${kind})${details}</li>`;
    })
    .join("");

  return `
    <div class="card">
      <h3>${scope}</h3>
      <p><strong>Responsibility entries:</strong> ${responsibilityCount}</p>
      <ul>${responsibilityHtml || "<li>No approved items</li>"}</ul>
      ${buildDirectoryScopeSelect(scope)}
    </div>
  `;
}

function renderAgentDirectory() {
  const root = qs("#agentDirectory");
  if (!root) {
    return;
  }

  const cards = [];
  if (state.agentDirectoryArtifact) {
    cards.push(buildDirectoryArtifactCard(state.agentDirectoryArtifact));
  }

  const directories = (state.agentDirectory || []).map((entry) => buildAgentDirectoryCard(entry));
  if (directories.length > 0) {
    cards.push(...directories);
  } else {
    cards.push("<p>No approved agent responsibilities yet.</p>");
  }
  root.innerHTML = cards.join("");
}

function buildDirectoryArtifactCard(artifact) {
  const schemaVersion = escapeHtml(artifact.context_schema_version || "unset");
  const sourceType = escapeHtml(artifact.source_type || "unknown");
  const artifactId = escapeHtml(artifact.artifact_id || "unavailable");
  const importId = escapeHtml(artifact.generated_from_import_id || "-");
  const itemCount = Number(artifact.imported_item_count || 0);
  const sourceStatus = escapeHtml(artifact.import_status || "n/a");
  const metadata = artifact.source_metadata || {};
  const metadataText = Object.keys(metadata).length ? escapeHtml(JSON.stringify(metadata)) : "none";
  const actorId = escapeHtml(artifact.source_actor_id || "n/a");
  const missionId = escapeHtml(artifact.source_mission_id || "n/a");
  const lineage = Array.isArray(artifact.lineage) ? artifact.lineage : [];
  const lineageBlock = lineage.length
    ? `<p><strong>Lineage:</strong> ${lineage
        .map((entry) => {
          const label = escapeHtml(entry.import_id || "unknown");
          return `<span>${label}</span>`;
        })
        .join(" → ")}</p>`
    : "";
  const overlays = artifact.source_metadata && Object.keys(artifact.source_metadata).length
    ? artifact.source_metadata.agent_overlays || artifact.source_metadata.overlays || null
    : null;
  const overlaysLabel = overlays
    ? `<p><strong>Overlays:</strong> ${escapeHtml(
        typeof overlays === "string" ? overlays : JSON.stringify(overlays),
      )}</p>`
    : "";

  return `
    <div class="card">
      <h3>Company Directory Artifact</h3>
      <p><strong>Artifact:</strong> ${artifactId}</p>
      <p><strong>Source:</strong> ${sourceType} (${sourceStatus})</p>
      <p><strong>Generated from:</strong> ${importId}</p>
      <p><strong>Approved entries:</strong> ${itemCount}</p>
      <p><strong>Context schema:</strong> ${schemaVersion}</p>
      <p><strong>Actor ID:</strong> ${actorId}</p>
      <p><strong>Mission ID:</strong> ${missionId}</p>
      ${lineageBlock}
      ${overlaysLabel}
      <p><strong>Metadata:</strong> ${metadataText}</p>
    </div>
  `;
}

function buildPluginCard(plugin) {
  const isViewer = state.role === "viewer";
  const isLocked = isViewer ? "disabled" : "";
  const metadata = plugin.metadata || {};
  const metadataLabel = Object.keys(metadata).length
    ? `<p><strong>Metadata:</strong> ${escapeHtml(JSON.stringify(metadata))}</p>`
    : "";
  return `
    <div class="card">
      <h3>${escapeHtml(plugin.name || plugin.id)}</h3>
      <p><strong>Plugin ID:</strong> ${escapeHtml(plugin.id)}</p>
      <p><strong>Source:</strong> ${escapeHtml(plugin.source || "n/a")}</p>
      <p><strong>Kind:</strong> ${escapeHtml(plugin.kind || "n/a")}</p>
      <p><strong>Status:</strong> ${escapeHtml(plugin.status || "configured")}</p>
      <p><strong>Command:</strong> ${escapeHtml(plugin.command || "n/a")}</p>
      ${metadataLabel}
      <p>
        <label>
          Enabled:
          <input type="checkbox" data-action="toggle-plugin" data-id="${plugin.id}" ${plugin.enabled ? "checked" : ""} ${isLocked} />
        </label>
      </p>
      <p>
        <label>
          Status:
          <input type="text" data-action="set-plugin-status" data-id="${plugin.id}" value="${escapeHtml(plugin.status || "configured")}" ${isLocked} />
        </label>
      </p>
      <p>
        <label>
          Command:
          <input type="text" data-action="set-plugin-command" data-id="${plugin.id}" value="${escapeHtml(plugin.command || "")}" placeholder="plugin command" ${isLocked} />
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
  qs('input[name="proxy_http"]').value = state.settings.proxy_http || "";
  qs('input[name="proxy_https"]').value = state.settings.proxy_https || "";
  qs('input[name="proxy_no_proxy"]').value = state.settings.proxy_no_proxy || "";
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

async function loadPlugins() {
  const payload = await fetchJson("/api/plugins");
  state.plugins = payload.plugins || [];
  renderCards("#plugins", state.plugins, buildPluginCard);
}

async function loadProviderChain() {
  const payload = await fetchJson("/api/providers/routing-chain");
  state.providerChain = payload.chain || [];
  renderProviderChain(payload);
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

async function loadAgentDirectory() {
  const payload = await fetchJson("/api/agents/directory");
  state.agentDirectory = payload.directory || [];
  state.agentDirectoryArtifact = payload.directory_artifact || null;
  renderAgentDirectory();
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

async function loadMemory(scope = "", type = "", itemId = "") {
  const params = new URLSearchParams();
  if (scope) {
    params.set("scope", scope);
  }
  if (type) {
    params.set("type", type);
  }
  if (itemId) {
    params.set("item_id", itemId);
  }
  const query = params.toString();
  const payload = await fetchJson(query ? `/api/memory?${query}` : "/api/memory");
  renderMemory(payload.memory || []);
  if (itemId) {
    setStatus("#memoryStatus", `memory filtered by item ${itemId}`, "ok");
  } else {
    setStatus("#memoryStatus", "", "");
  }
}

async function loadChatHistory(agentId = "", itemId = "") {
  const params = new URLSearchParams();
  if (agentId) {
    params.set("agent_id", agentId);
  }
  if (itemId) {
    params.set("item_id", itemId);
  }
  const query = params.toString();
  const payload = await fetchJson(query ? `/api/chat_history?${query}` : "/api/chat_history");
  renderChat(payload.chat_history || []);
  if (itemId) {
    setStatus("#chatStatus", `chat filtered by item ${itemId}`, "ok");
  } else {
    setStatus("#chatStatus", "", "");
  }
}

function buildToolUsageFilter() {
  const toolId = qs("#toolUsageToolId")?.value.trim() || "";
  const agentId = qs("#toolUsageAgentId")?.value.trim() || "";
  const contextType = qs("#toolUsageContextType")?.value.trim() || "";
  const contextId = qs("#toolUsageContextId")?.value.trim() || "";
  const missionId = qs("#toolUsageMissionId")?.value.trim() || "";
  const sessionId = qs("#toolUsageSessionId")?.value.trim() || "";
  const limit = Math.min(200, Math.max(1, Number(qs("#toolUsageLimit")?.value || 25)));
  const sortDesc = Boolean(qs("#toolUsageSortDesc")?.checked);
  return { toolId, agentId, contextType, contextId, missionId, sessionId, limit, sortDesc };
}

function buildToolUsageParams(filter, overrides = {}) {
  const params = new URLSearchParams();
  const payload = {
    agent_id: overrides.agentId || filter.agentId,
    mission_id: overrides.missionId || filter.missionId,
    context_type: overrides.contextType || filter.contextType,
    context_id: overrides.contextId || filter.contextId,
    session_id: overrides.sessionId || filter.sessionId,
    limit: overrides.limit != null ? overrides.limit : filter.limit,
    sort_desc: overrides.sortDesc != null ? overrides.sortDesc : filter.sortDesc,
  };
  const toolIdOverride = overrides.toolId != null ? overrides.toolId : filter.toolId;
  if (toolIdOverride) {
    params.set("tool_id", toolIdOverride);
  }
  Object.entries(payload).forEach(([key, value]) => {
    if (value !== "" && value !== undefined && value !== null) {
      params.set(key, String(value));
    }
  });
  if (payload.sort_desc === false) {
    params.set("sort_desc", "false");
  }
  return params;
}

function buildToolUsageEndpoint(mode, toolId = "", filter) {
  if (mode === "agent") {
    return `/api/agents/${encodeURIComponent(toolId)}/tool_usage`;
  }
  if (mode === "global_log") {
    return "/api/tool-log";
  }
  if (mode === "tool_log_by_session") {
    return `/api/tool-log/session/${encodeURIComponent(toolId)}`;
  }
  if (toolId) {
    return `/api/tools/${encodeURIComponent(toolId)}/usage`;
  }
  return "/api/tools/usage";
}

async function loadToolUsageRows({ toolId = "", filters = {}, mode = "tool", summarySuffix = "usage" }) {
  const filter = { ...filters };
  const params = buildToolUsageParams(filter);
  if (mode === "global_log") {
    const missionId = filters.missionId || "";
    if (missionId) {
      params.set("mission_id", missionId);
    }
  }
  const query = params.toString();
  const path = `${buildToolUsageEndpoint(mode, toolId, filter)}${query ? `?${query}` : ""}`;
  const payload = await fetchJson(path);
  const usage = payload.usage || [];
  state.toolUsage = usage;
  state.toolUsageCount = Number(payload.count || 0);
  renderToolUsage(usage);
  setStatus("#toolUsageSummary", buildToolUsageSummary(state.toolUsageCount, {
    tool_id: payload.tool_id || toolId || null,
    agent_id: filter.agentId || null,
    context_type: filter.contextType || null,
    context_id: filter.contextId || null,
    mission_id: filter.missionId || null,
    session_id: filter.sessionId || null,
    mode: summarySuffix,
  }), "ok");
  setStatus("#toolUsageStatus", `loaded ${usage.length} row(s)`, "ok");
  if (payload.log_entry) {
    renderToolUsageLogEntry(payload.log_entry);
  }
  if (!usage.length) {
    qs("#toolUsageLogEntry").innerHTML = "";
  }
  return usage;
}

async function loadToolUsage() {
  const filter = buildToolUsageFilter();
  try {
    await loadToolUsageRows({
      toolId: filter.toolId || "",
      filters: filter,
      mode: filter.toolId ? "tool" : "global",
      summarySuffix: "tool usage",
    });
  } catch (err) {
    setStatus("#toolUsageStatus", `load failed: ${err.message}`, "err");
    state.toolUsage = [];
    state.toolUsageCount = 0;
    renderToolUsage([]);
  }
}

async function loadToolLog() {
  const filter = buildToolUsageFilter();
  try {
    await loadToolUsageRows({
      toolId: "",
      filters: filter,
      mode: "global_log",
      summarySuffix: "tool-log",
    });
  } catch (err) {
    setStatus("#toolUsageStatus", `tool log load failed: ${err.message}`, "err");
    state.toolUsage = [];
    state.toolUsageCount = 0;
    renderToolUsage([]);
  }
}

async function loadToolLogBySession(sessionId = "") {
  const filter = buildToolUsageFilter();
  const targetSession = sessionId || filter.sessionId;
  if (!targetSession) {
    setStatus("#toolUsageStatus", "session id is required", "err");
    return;
  }
  try {
    const payload = await fetchJson(`/api/tool-log/session/${encodeURIComponent(targetSession)}`);
    const usage = payload.usage || [];
    state.toolUsage = usage;
    state.toolUsageCount = Number(payload.count || usage.length || 0);
    renderToolUsage(usage);
    setStatus("#toolUsageSummary", `Session ${escapeHtml(targetSession)} log: ${state.toolUsageCount} entries`, "ok");
    setStatus("#toolUsageStatus", `loaded ${usage.length} row(s)`, "ok");
  } catch (err) {
    setStatus("#toolUsageStatus", `session log load failed: ${err.message}`, "err");
    state.toolUsage = [];
    state.toolUsageCount = 0;
    renderToolUsage([]);
  }
}

async function loadAgentToolUsage(agentId) {
  if (!agentId) {
    setStatus("#toolUsageStatus", "agent id is required", "err");
    return;
  }
  const filter = buildToolUsageFilter();
  try {
    const params = buildToolUsageParams(filter, {
      agentId,
      missionId: filter.missionId,
      contextType: filter.contextType,
      contextId: filter.contextId,
      sessionId: filter.sessionId,
      limit: filter.limit,
      sortDesc: filter.sortDesc,
    });
    const query = params.toString();
    const payload = await fetchJson(`/api/agents/${encodeURIComponent(agentId)}/tool_usage?${query}`);
    const usage = payload.usage || [];
    state.toolUsage = usage;
    state.toolUsageCount = Number(payload.count || 0);
    renderToolUsage(usage);
    setStatus("#toolUsageSummary", buildToolUsageSummary(state.toolUsageCount, {
      tool_id: payload.tool_id || null,
      agent_id: agentId,
      context_type: filter.contextType || null,
      context_id: filter.contextId || null,
      mission_id: filter.missionId || null,
      session_id: filter.sessionId || null,
      mode: `agent ${agentId}`,
    }), "ok");
    setStatus("#toolUsageStatus", `loaded ${usage.length} row(s)`, "ok");
  } catch (err) {
    setStatus("#toolUsageStatus", `agent tool usage failed: ${err.message}`, "err");
    state.toolUsage = [];
    state.toolUsageCount = 0;
    renderToolUsage([]);
  }
}

function renderToolUsageLogEntry(entry) {
  const root = qs("#toolUsageLogEntry");
  if (!root) {
    return;
  }
  if (!entry) {
    root.innerHTML = "<p>Select an entry to inspect details.</p>";
    return;
  }
  const metadata = entry.details && typeof entry.details === "object"
    ? `<pre>${escapeHtml(JSON.stringify(entry.details, null, 2))}</pre>`
    : `<pre>${formatToolUsageValue(entry.details)}</pre>`;
  const payload = entry.payload && typeof entry.payload === "object"
    ? `<pre>${escapeHtml(JSON.stringify(entry.payload, null, 2))}</pre>`
    : "";
  root.innerHTML = `
    <div class="card">
      <h3>Tool call ${escapeHtml(entry.id || "n/a")}</h3>
      <p><strong>Tool:</strong> ${escapeHtml(entry.tool_id || "n/a")}</p>
      <p><strong>Agent:</strong> ${escapeHtml(entry.agent_id || "n/a")}</p>
      <p><strong>Session:</strong> ${escapeHtml(entry.session_id || "n/a")}</p>
      <p><strong>Mission:</strong> ${escapeHtml(entry.mission_id || "n/a")}</p>
      <p><strong>Status:</strong> ${escapeHtml(entry.status || "n/a")}</p>
      <p><strong>Result:</strong> ${escapeHtml(entry.result_status || "n/a")}</p>
      <p><strong>Created:</strong> ${escapeHtml(entry.created_at || "n/a")}</p>
      ${payload}
      ${metadata}
    </div>
  `;
}

async function openToolUsageContext(contextType, contextId) {
  if (!contextId) {
    setStatus("#toolUsageStatus", "context id is required for drill-through", "err");
    return;
  }
  try {
    if (contextType === "memory") {
      await loadMemory("", "", contextId);
      return;
    }
    if (contextType === "chat") {
      await loadChatHistory("", contextId);
      return;
    }
    setStatus("#toolUsageStatus", `no drill-through supported for context ${contextType}`, "err");
  } catch (err) {
    setStatus("#toolUsageStatus", `drill-through failed: ${err.message}`, "err");
  }
}

async function openToolLogEntry(toolCallId) {
  if (!toolCallId) {
    setStatus("#toolLogEntryStatus", "tool call id is required", "err");
    return;
  }
  try {
    const payload = await fetchJson(`/api/tool-log/${encodeURIComponent(toolCallId)}`);
    renderToolUsageLogEntry(payload.usage || null);
    setStatus("#toolLogEntryStatus", `loaded tool call ${toolCallId}`, "ok");
  } catch (err) {
    setStatus("#toolLogEntryStatus", `tool call lookup failed: ${err.message}`, "err");
    renderToolUsageLogEntry(null);
  }
}

function formatToolUsageValue(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }
  if (typeof value === "object") {
    return escapeHtml(JSON.stringify(value));
  }
  return escapeHtml(String(value));
}

function renderToolUsage(usageItems) {
  const root = qs("#toolUsage");
  if (!root) {
    return;
  }
  const entries = Array.isArray(usageItems) ? usageItems : [];
  if (!entries.length) {
    root.innerHTML = "<p>No usage records match your filters.</p>";
    return;
  }

  const rows = entries
    .map((entry) => {
      const contextType = (entry.context_type || "n/a");
      const contextId = entry.context_id || "";
      const contextAction = contextType === "memory" || contextType === "chat"
        ? `open-tool-context`
        : "";
      const sessionAction = entry.session_id ? `open-tool-session` : "";
      const sessionButton = sessionAction
        ? `<button
              type="button"
              data-action="${sessionAction}"
              data-session-id="${escapeHtml(entry.session_id)}"
            >
              Open session
            </button>`
        : "";
      const contextButton = contextAction && contextId
        ? `<button
              type="button"
              data-action="${contextAction}"
              data-context-type="${escapeHtml(contextType)}"
              data-context-id="${escapeHtml(contextId)}"
            >
              Open ${escapeHtml(contextType)} context
            </button>`
        : "";
      const entryAction = entry.id
        ? `<button
              type="button"
              data-action="open-tool-log-entry"
              data-tool-call-id="${escapeHtml(entry.id)}"
            >
              View call details
            </button>`
        : "";
      const detail = entry.result_status ? `Result: ${escapeHtml(entry.result_status)} · ` : "";
      const details = entry.details ? `<pre>${formatToolUsageValue(entry.details)}</pre>` : "";
      return `
        <div class="card">
          <h3>${escapeHtml(entry.tool_id || "tool")}</h3>
          <p><strong>Entry:</strong> ${escapeHtml(entry.id || "n/a")}</p>
          <p><strong>Agent:</strong> ${escapeHtml(entry.agent_id || "n/a")}</p>
          <p><strong>Context:</strong> ${escapeHtml(contextType)} ${contextId ? `( ${escapeHtml(contextId)})` : ""}</p>
          <p><strong>Status:</strong> ${escapeHtml(entry.status || "n/a")} · ${detail}duration ${
            typeof entry.duration_ms === "number" ? `${entry.duration_ms}ms` : "n/a"
          }</p>
          <p><strong>Created:</strong> ${escapeHtml(entry.created_at || "n/a")}</p>
          ${details}
          ${contextButton}
          ${sessionButton}
          ${entryAction}
        </div>
      `;
    })
    .join("");

  root.innerHTML = `<div class="card-list">${rows}</div>`;
}

function buildToolUsageSummary(toolUsageCount, toolUsageFilter) {
  const total = Number(toolUsageCount || 0);
  const agentText = toolUsageFilter.agent_id || "all agents";
  const contextText = toolUsageFilter.context_type || "all context types";
  const contextIdText = toolUsageFilter.context_id || "any context";
  const toolText = toolUsageFilter.tool_id || "all tools";
  const missionText = toolUsageFilter.mission_id || "all missions";
  const sessionText = toolUsageFilter.session_id || "all sessions";
  const modeText = toolUsageFilter.mode ? `${toolUsageFilter.mode} ` : "";
  return `Showing ${total} ${modeText}rows for ${toolText}; ${agentText} · ${contextText} · ${contextIdText} · ${missionText} · ${sessionText}`;
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
    renderImportAudit(null);
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
  await loadImportAudit(importId);
}

async function loadImportAudit(importId) {
  if (!importId) {
    state.importAudit = null;
    renderImportAudit(null);
    return;
  }
  try {
    const payload = await fetchJson(`/api/agents/import/${encodeURIComponent(importId)}/audit`);
    state.importAudit = payload;
    renderImportAudit(payload);
  } catch (err) {
    state.importAudit = null;
    renderImportAudit(null);
    setStatus("#importSubmitStatus", `audit load failed: ${err.message}`, "err");
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
    proxy_http: qs('input[name="proxy_http"]').value.trim(),
    proxy_https: qs('input[name="proxy_https"]').value.trim(),
    proxy_no_proxy: qs('input[name="proxy_no_proxy"]').value.trim(),
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
  const actorId = (qs('input[name="actor_id"]')?.value || "").trim();
  const missionId = (qs('input[name="mission_id"]')?.value || "").trim();
  const contextSchemaVersion = (qs('input[name="context_schema_version"]')?.value || "").trim();

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
  if (actorId) {
    form.append("actor_id", actorId);
  }
  if (missionId) {
    form.append("mission_id", missionId);
  }
  if (contextSchemaVersion) {
    form.append("context_schema_version", contextSchemaVersion);
  }

  setStatus("#importSubmitStatus", "uploading...", "");
  try {
    const payload = await postForm("/api/agents/import", form);
    setStatus("#importSubmitStatus", `import queued: ${payload.import_id}`, "ok");
    await loadImportJobs();
    await loadImportItems(payload.import_id);
    await loadImportAudit(payload.import_id);
    await loadAgentDirectory();
  } catch (err) {
    setStatus("#importSubmitStatus", `import failed: ${err.message}`, "err");
  }
}

async function submitImportRerun(event) {
  const fileInput = event.target;
  const file = fileInput?.files?.[0];
  if (!state.rerunImportId) {
    setStatus("#importSubmitStatus", "Select an import first before rerun", "err");
    return;
  }
  if (!file) {
    setStatus("#importSubmitStatus", "Rerun requires a source file", "err");
    return;
  }

  const form = new FormData();
  form.append("source_file", file);

  setStatus("#importSubmitStatus", "rerunning import...", "");
  try {
    const payload = await postForm(`/api/agents/import/${encodeURIComponent(state.rerunImportId)}/rerun`, form);
    setStatus("#importSubmitStatus", `import rerun queued: ${payload.import_id}`, "ok");
    state.rerunImportId = "";
    await loadImportJobs();
    await loadImportItems(payload.import_id);
    await loadImportAudit(payload.import_id);
    await loadAgentDirectory();
  } catch (err) {
    setStatus("#importSubmitStatus", `import rerun failed: ${err.message}`, "err");
  } finally {
    fileInput.value = "";
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
  const scopeInput = document.querySelector(`.import-item-scope[data-id="${itemId}"]`);
  const desiredScope = scopeInput?.value?.trim() || "";

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
      agent_scope: desiredScope || undefined,
    });
    setStatus("#importSubmitStatus", `item ${itemId} marked ${stateValue}`, "ok");
    await loadImportItems(state.selectedImportId);
    await loadImportJobs();
    await loadAgentDirectory();
  } catch (err) {
    setStatus("#importSubmitStatus", `decision update failed: ${err.message}`, "err");
  }
}

async function reassignDirectoryScope(sourceScope, trigger = null) {
  const normalizedSource = (sourceScope || "global").trim() || "global";
  const { select, custom } = getDirectoryScopeControlNodes(normalizedSource, trigger);
  const selectedValue = select?.value?.trim() || "";
  const customValue = custom?.value?.trim() || "";
  const targetScope = customValue || selectedValue;

  if (!targetScope) {
    setStatus("#agentDirectoryStatus", "target scope is required", "err");
    return;
  }
  if (targetScope === normalizedSource) {
    setStatus("#agentDirectoryStatus", "target scope must be different from source scope", "err");
    return;
  }

  setStatus("#agentDirectoryStatus", "reassigning directory scope...", "");
  try {
    await patchJson("/api/agents/directory/reassign", {
      source_scope: normalizedSource,
      target_scope: targetScope,
      actor: state.role,
    });
  if (custom) {
    custom.value = "";
  }
    setStatus("#agentDirectoryStatus", `reassigned ${normalizedSource} -> ${targetScope}`, "ok");
    await Promise.all([
      loadAgentDirectory(),
      loadImportJobs(),
      state.selectedImportId ? loadImportItems(state.selectedImportId) : Promise.resolve(),
    ]);
  } catch (err) {
    setStatus("#agentDirectoryStatus", `reassign failed: ${err.message}`, "err");
  }
}

async function submitProvider(event) {
  event.preventDefault();
  const provider_id = qs('input[name="providerId"]').value.trim();
  const name = qs('input[name="providerName"]').value.trim();
  const endpoint = qs('input[name="providerEndpoint"]').value.trim();
  const model = qs('input[name="providerModel"]').value.trim();
  const kind = qs('input[name="providerKind"]').value.trim();
  const fallbackModels = qs('input[name="providerFallbackModels"]').value.trim();
  const retries = qs('input[name="providerRetries"]').value.trim();
  const timeoutMs = qs('input[name="providerTimeoutMs"]').value.trim();
  const toolTimeoutMs = qs('input[name="providerToolTimeoutMs"]').value.trim();
  const priority = Number(qs('input[name="providerPriority"]').value || "100");
  const enabled = qs('input[name="providerEnabled"]').checked;

  if (!provider_id || !name || !endpoint) {
    setStatus("#providerCreateStatus", "provider id, name, and endpoint are required", "err");
    return;
  }

  const payload = {
    id: provider_id,
    name,
    endpoint,
    model: model || undefined,
    kind: kind || undefined,
    fallback_models: fallbackModels
      ? fallbackModels
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean)
      : [],
    priority,
    enabled,
    metadata: {},
  };
  if (retries) {
    payload.retries = Number(retries);
  }
  if (timeoutMs) {
    payload.timeout_ms = Number(timeoutMs);
  }
  if (toolTimeoutMs) {
    payload.tool_timeout_ms = Number(toolTimeoutMs);
  }
  if (fallbackModels === "") {
    payload.fallback_models = [];
  }

  setStatus("#providerCreateStatus", "creating provider...", "");
  try {
    await postJson("/api/providers", payload);
    setStatus("#providerCreateStatus", `created ${provider_id}`, "ok");
    qs('input[name="providerId"]').value = "";
    qs('input[name="providerName"]').value = "";
    qs('input[name="providerEndpoint"]').value = "";
    qs('input[name="providerModel"]').value = "";
    qs('input[name="providerFallbackModels"]').value = "";
    qs('input[name="providerRetries"]').value = "";
    qs('input[name="providerTimeoutMs"]').value = "";
    qs('input[name="providerToolTimeoutMs"]').value = "";
    qs('input[name="providerKind"]').value = "";
    qs('input[name="providerPriority"]').value = "";
    qs('input[name="providerEnabled"]').checked = true;
    await Promise.all([loadProviders(), loadProviderChain()]);
  } catch (err) {
    setStatus("#providerCreateStatus", `create failed: ${err.message}`, "err");
  }
}

async function submitPlugin(event) {
  event.preventDefault();
  const pluginId = qs('input[name="pluginId"]').value.trim();
  const pluginName = qs('input[name="pluginName"]').value.trim();
  const pluginSource = qs('input[name="pluginSource"]').value.trim();
  const pluginKind = qs('input[name="pluginKind"]').value.trim();
  const pluginCommand = qs('input[name="pluginCommand"]').value.trim();
  const pluginStatus = qs('input[name="pluginStatus"]').value.trim() || "configured";
  const pluginEnabled = qs('input[name="pluginEnabled"]').checked;

  if (!pluginId || !pluginName || !pluginSource || !pluginKind) {
    setStatus("#pluginCreateStatus", "Plugin id, name, source, and kind are required", "err");
    return;
  }

  const payload = {
    id: pluginId,
    name: pluginName,
    source: pluginSource,
    kind: pluginKind,
    command: pluginCommand || undefined,
    status: pluginStatus,
    enabled: pluginEnabled,
  };

  setStatus("#pluginCreateStatus", "creating plugin...", "");
  try {
    const response = await postJson("/api/plugins", payload);
    if (!response?.plugin) {
      throw new Error("Unexpected response from plugin create endpoint");
    }
    setStatus("#pluginCreateStatus", `created ${pluginId}`, "ok");
    qs('input[name="pluginId"]').value = "";
    qs('input[name="pluginName"]').value = "";
    qs('input[name="pluginSource"]').value = "";
    qs('input[name="pluginKind"]').value = "";
    qs('input[name="pluginCommand"]').value = "";
    qs('input[name="pluginStatus"]').value = "";
    qs('input[name="pluginEnabled"]').checked = true;
    await loadPlugins();
  } catch (err) {
    setStatus("#pluginCreateStatus", `create failed: ${err.message}`, "err");
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

  if (action === "refresh-tool-usage") {
    await loadToolUsage();
    return;
  }

  if (action === "refresh-tool-log") {
    await loadToolLog();
    return;
  }

  if (action === "load-tool-log-session") {
    await loadToolLogBySession();
    return;
  }

  if (action === "open-tool-session") {
    const sessionId = target.dataset.sessionId || "";
    await loadToolLogBySession(sessionId);
    return;
  }

  if (action === "open-tool-context") {
    const contextType = target.dataset.contextType || "";
    const contextId = target.dataset.contextId || "";
    await openToolUsageContext(contextType, contextId);
    return;
  }

  if (action === "open-tool-log-entry") {
    const toolCallId = target.dataset.toolCallId || "";
    await openToolLogEntry(toolCallId);
    return;
  }

  if (action === "open-agent-tool-usage") {
    const agentId = target.dataset.id || "";
    await loadAgentToolUsage(agentId);
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
    await loadImportAudit(importId);
    return;
  }

  if (action === "rerun-import") {
    const importId = target.dataset.id;
    const rerunInput = qs("#importRerunFile");
    if (!importId) {
      setStatus("#importSubmitStatus", "import id missing", "err");
      return;
    }
    if (!rerunInput) {
      setStatus("#importSubmitStatus", "rerun input missing", "err");
      return;
    }
    if (state.role === "viewer") {
      setStatus("#importSubmitStatus", "viewer cannot rerun imports", "err");
      return;
    }
    state.rerunImportId = importId;
    rerunInput.click();
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

  if (action === "reassign-directory-scope") {
    const sourceScope = target.dataset.sourceScope || "global";
    await reassignDirectoryScope(sourceScope, target);
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

    if (action === "save-agent-overlay") {
      const id = target.dataset.id;
      if (!id) {
        setStatus("#healthState", "agent id missing", "err");
        return;
      }
      const card = target.closest(".card");
      if (!card) {
        setStatus("#healthState", "agent overlay card missing", "err");
        return;
      }
      const overlayTextarea = card.querySelector(`textarea[data-action="agent-overlay-content"][data-id="${id}"]`);
      const tokenCapInput = card.querySelector(`input[data-action="agent-overlay-token-cap"][data-id="${id}"]`);
      if (!overlayTextarea) {
        setStatus("#healthState", "agent overlay content field missing", "err");
        return;
      }

      const overlayContent = String(overlayTextarea.value || "");
      const capText = tokenCapInput?.value?.trim();
      const payload = { overlay_content: overlayContent };
      if (capText) {
        const normalizedCap = Number(capText);
        if (!Number.isFinite(normalizedCap) || normalizedCap <= 0 || !Number.isInteger(normalizedCap)) {
          setStatus("#healthState", "overlay token cap must be a positive integer", "err");
          return;
        }
        payload.overlay_token_cap = normalizedCap;
      }
      setStatus("#healthState", `saving overlay for ${id}...`, "");
      await patchJson(`/api/agents/${id}`, payload);
      setStatus("#healthState", `overlay saved for ${id}`, "ok");
      await loadAgents();
      return;
    }

    if (action === "toggle-plugin") {
      const id = target.dataset.id;
      await patchJson(`/api/plugins/${id}`, { enabled: target.checked });
      await loadPlugins();
      return;
    }

    if (action === "set-plugin-status") {
      const id = target.dataset.id;
      await patchJson(`/api/plugins/${id}`, { status: target.value.trim() || "configured" });
      await loadPlugins();
      return;
    }

    if (action === "set-plugin-command") {
      const id = target.dataset.id;
      const command = target.value.trim();
      const payload = command ? { command } : { command: null };
      await patchJson(`/api/plugins/${id}`, payload);
      await loadPlugins();
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
      loadPlugins(),
      loadImportJobs(),
      loadAgents(),
      loadAgentDirectory(),
      loadEvents(),
      loadMemory(),
      loadChatHistory(),
      loadToolUsage(),
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
qs("#pluginCreateForm")?.addEventListener("submit", submitPlugin);
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
    && action !== "refresh-tool-usage"
    && action !== "refresh-tool-log"
    && action !== "load-tool-log-session"
    && action !== "rerun-import"
    && action !== "open-tool-context"
    && action !== "open-tool-session"
    && action !== "open-tool-log-entry"
    && action !== "open-agent-tool-usage"
    && action !== "reassign-directory-scope"
    && action !== "apply-discovered-model"
    && action !== "test-provider-models"
    && action !== "test-all-providers"
    && action !== "set-provider-secret"
    && action !== "clear-provider-secret"
    && action !== "import-item-decision"
    && action !== "show-provider-diagnostics"
    && action !== "toggle-plugin"
    && action !== "save-agent-overlay"
    && action !== "set-plugin-status"
    && action !== "set-plugin-command"
  )) {
    return;
  }
  onTableChange(event);
});
qs("#importRerunFile")?.addEventListener("change", submitImportRerun);

boot();
setInterval(boot, 15000);
