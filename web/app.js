const API_BASE = (() => {
  const api = window.busyManagementApiBase;
  if (api && typeof api.resolveManagementApiBase === "function") {
    return api.resolveManagementApiBase(window, document);
  }
  const configured = (window.MANAGEMENT_API_BASE || "").trim();
  if (configured !== "") {
    return configured;
  }
  if (window.location && /^(http|https):$/i.test(window.location.protocol || "")) {
    return window.location.origin;
  }
  return "http://127.0.0.1:8031";
})();

const TOKEN_KEY = "busy38-management-token";
const SEEN_EVENT_STORAGE_FALLBACK = {
  event_ids: [],
  card_reviews: {},
  updated_at: "",
};
const GM_TICKET_STATUS_OPTIONS = [
  "requested",
  "queued",
  "open",
  "in_progress",
  "blocked",
  "complete",
  "resolved",
  "closed",
  "archived",
];
const GM_TICKET_PRIORITY_OPTIONS = ["low", "normal", "high", "critical"];
const GM_TICKET_DISPATCH_ROLES = ["portia", "nora", "mini", "mission_agent"];
const GM_TICKET_MESSAGE_TYPES = ["comment", "status", "request"];
const state = {
  settings: {},
  appearance: {},
  providers: [],
  selectedProviderId: "",
  plugins: [],
  agents: [],
  events: [],
  startupDebugSummary: null,
  corePlugins: [],
  runtimeStatus: null,
  runtimeServices: [],
  visibleEvents: [],
  latestPairingIssue: null,
  mobilePairingState: null,
  role: "unknown",
  roleSource: "unknown",
  providerChain: [],
  providerDiagnostics: null,
  providerActionResults: {},
  importJobs: [],
  importItems: [],
  importAudit: null,
  agentDirectory: [],
  agentDirectoryArtifact: null,
  selectedImportId: "",
  rerunImportId: "",
  selectedAgentAuditId: "",
  selectedAgentOverlayHistoryId: "",
  agentToolAudit: null,
  toolUsage: [],
  toolUsageCount: 0,
  eventFilters: {
    domain: "",
    level: "",
    query: "",
    limit: 50,
  },
  agentOverlayHistory: [],
  agentOverlayHistoryCount: 0,
  gmTickets: [],
  selectedGmTicketId: "",
  selectedGmTicket: null,
  gmTicketMessages: [],
  gmTicketMessageCount: 0,
  gmTicketAudit: null,
  attentionSeen: SEEN_EVENT_STORAGE_FALLBACK,
  dashboardOverview: null,
  selectedAttentionCardId: "",
  attentionHistoryExpandedCards: {},
};
let eventSocket = null;

applyAppearanceTheme({});
loadAttentionSeenState();

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

function pluginUiConsoleLogger() {
  const logger = window.Busy38PluginUiConsole;
  if (!logger || typeof logger !== "object") {
    return null;
  }
  if (typeof logger.logDiagnostics !== "function") {
    return null;
  }
  if (typeof logger.logActionResult !== "function") {
    return null;
  }
  if (typeof logger.logActionRequestFailure !== "function") {
    return null;
  }
  return logger;
}

function mobilePairingQrApi() {
  const api = window.Busy38MobilePairingQr;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.buildPairingQrPayload !== "function") {
    return null;
  }
  if (typeof api.renderPairingQrSvg !== "function") {
    return null;
  }
  return api;
}

function appearanceThemeApi() {
  const api = window.Busy38AppearanceTheme;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.normalizeAppearancePreferences !== "function") {
    return null;
  }
  if (typeof api.effectiveDesktopThemeValue !== "function") {
    return null;
  }
  if (typeof api.applyDocumentTheme !== "function") {
    return null;
  }
  return api;
}

function runtimeUiApi() {
  const api = window.Busy38RuntimeUi;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.buildRuntimeViewModel !== "function") {
    return null;
  }
  if (typeof api.renderServiceCard !== "function") {
    return null;
  }
  if (typeof api.formatActionStatus !== "function") {
    return null;
  }
  return api;
}

function attentionStateApi() {
  const api = window.Busy38AttentionState;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.readSeenState !== "function") {
    return null;
  }
  if (typeof api.markEventsSeen !== "function") {
    return null;
  }
  if (typeof api.markCardReviewed !== "function") {
    return null;
  }
  if (typeof api.getCardReviewedAt !== "function") {
    return null;
  }
  return api;
}

function eventsUiApi() {
  const api = window.Busy38EventsUi;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.buildEventViewModel !== "function") {
    return null;
  }
  if (typeof api.renderEventItem !== "function") {
    return null;
  }
  return api;
}

function dashboardOverviewApi() {
  const api = window.Busy38DashboardOverview;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.buildDashboardViewModel !== "function") {
    return null;
  }
  if (typeof api.renderDashboardCard !== "function") {
    return null;
  }
  return api;
}

function providerRoutingSummaryApi() {
  const api = window.Busy38ProviderRoutingSummary;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.buildProviderRoutingSummary !== "function") {
    return null;
  }
  return api;
}

function providerHealthUiApi() {
  const api = window.Busy38ProviderHealthUi;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.buildProviderHealthSummary !== "function") {
    return null;
  }
  if (typeof api.renderProviderHealthSummary !== "function") {
    return null;
  }
  return api;
}

function providerActionUiApi() {
  const api = window.Busy38ProviderActionUi;
  if (!api || typeof api !== "object") {
    return null;
  }
  if (typeof api.normalizeProviderActionResult !== "function") {
    return null;
  }
  if (typeof api.renderProviderActionResult !== "function") {
    return null;
  }
  return api;
}

function normalizeAppearancePreferences(raw) {
  const api = appearanceThemeApi();
  if (api) {
    return api.normalizeAppearancePreferences(raw);
  }
  const source = raw && typeof raw === "object" ? raw : {};
  return {
    override_enabled: Boolean(source.override_enabled),
    sync_theme_preferences:
      typeof source.sync_theme_preferences === "boolean"
        ? source.sync_theme_preferences
        : true,
    shared_theme_mode: ["system", "light", "dark"].includes(source.shared_theme_mode)
      ? source.shared_theme_mode
      : "system",
    desktop_theme_mode: ["system", "light", "dark"].includes(source.desktop_theme_mode)
      ? source.desktop_theme_mode
      : "system",
    contrast_policy: ["aa", "aaa"].includes(source.contrast_policy)
      ? source.contrast_policy
      : "aa",
    motion_policy: ["default", "reduced"].includes(source.motion_policy)
      ? source.motion_policy
      : "default",
    color_separation_policy: ["default", "stronger"].includes(source.color_separation_policy)
      ? source.color_separation_policy
      : "default",
    text_spacing_policy: ["default", "increased"].includes(source.text_spacing_policy)
      ? source.text_spacing_policy
      : "default",
  };
}

function applyAppearanceTheme(preferences) {
  state.appearance = normalizeAppearancePreferences(preferences);
  const api = appearanceThemeApi();
  if (api) {
    return api.applyDocumentTheme(window, document, state.appearance);
  }
  document.documentElement.dataset.theme = "dark";
  return "dark";
}

function syncAppearanceFormState() {
  const overrideInput = qs('input[name="appearance_override_enabled"]');
  const syncInput = qs('input[name="appearance_sync_enabled"]');
  const themeSelect = qs('select[name="appearance_theme_mode"]');
  const syncRow = qs("#appearanceSyncRow");
  const themeLabel = qs("#appearanceThemeModeLabel");
  const contrastSelect = qs('select[name="appearance_contrast_policy"]');
  const motionSelect = qs('select[name="appearance_motion_policy"]');
  const colorSelect = qs('select[name="appearance_color_separation_policy"]');
  const spacingSelect = qs('select[name="appearance_text_spacing_policy"]');
  const saveButton = qs('#appearanceForm button[type="submit"]');
  if (!overrideInput || !syncInput || !themeSelect) {
    return;
  }
  const isAdmin = state.role === "admin";
  const overrideEnabled = overrideInput.checked;
  const syncEnabled = syncInput.checked;
  overrideInput.disabled = !isAdmin;
  syncInput.disabled = !isAdmin || !overrideEnabled;
  themeSelect.disabled = !isAdmin || !overrideEnabled;
  if (syncRow) {
    syncRow.hidden = !overrideEnabled;
  }
  if (themeLabel) {
    themeLabel.textContent = syncEnabled ? "Theme mode" : "Desktop theme mode";
  }
  for (const control of [contrastSelect, motionSelect, colorSelect, spacingSelect]) {
    if (control) {
      control.disabled = !isAdmin;
    }
  }
  if (saveButton) {
    saveButton.disabled = !isAdmin;
  }
}

function downloadTextFile(filename, content, mimeType = "application/json") {
  const body = typeof content === "string" ? content : JSON.stringify(content, null, 2);
  const blob = new Blob([body], { type: `${mimeType};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
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

function renderAgentToolAudit(audit) {
  const root = qs("#agentToolAudit");
  if (!root) {
    return;
  }
  if (!audit || !audit.agent_id) {
    root.innerHTML = "<p>Select an agent to load tool audit details.</p>";
    return;
  }

  const summary = audit.summary || {};
  const filters = audit.filters || {};
  const toolRows = (audit.tool_breakdown || [])
    .map(
      (row) => `
        <div class="card">
          <h4>${escapeHtml(row.tool_id || "unknown")}</h4>
          <p><strong>Calls:</strong> ${escapeHtml(String(row.call_count || 0))}</p>
          <p><strong>Last used:</strong> ${escapeHtml((row.last_used_at || "").slice(0, 10) || "n/a")}</p>
        </div>
      `,
    )
    .join("");
  const missionRows = (audit.mission_breakdown || [])
    .map((row) => {
      const missionId = row.mission_id || "";
      const drill = missionId
        ? `<button
            type="button"
            data-action="open-tool-mission"
            data-mission-id="${escapeHtml(missionId)}"
            data-agent-id="${escapeHtml(audit.agent_id || "")}"
          >
            Open mission trail
          </button>`
        : "";
      return `
        <div class="card">
          <h4>${escapeHtml(missionId || "n/a")}</h4>
          <p><strong>Calls:</strong> ${escapeHtml(String(row.call_count || 0))}</p>
          <p><strong>Sessions:</strong> ${escapeHtml(String(row.session_count || 0))}</p>
          <p><strong>Last used:</strong> ${escapeHtml((row.last_used_at || "").slice(0, 10) || "n/a")}</p>
          ${drill}
        </div>
      `;
    })
    .join("");
  const sessionRows = (audit.session_breakdown || [])
    .map((row) => {
      const sessionId = row.session_id || "";
      const drill = sessionId
        ? `<button
            type="button"
            data-action="open-tool-session"
            data-session-id="${escapeHtml(sessionId)}"
          >
            Open session
          </button>`
        : "";
      return `
        <div class="card">
          <h4>${escapeHtml(sessionId || "n/a")}</h4>
          <p><strong>Calls:</strong> ${escapeHtml(String(row.call_count || 0))}</p>
          <p><strong>Last used:</strong> ${escapeHtml((row.last_used_at || "").slice(0, 10) || "n/a")}</p>
          ${drill}
        </div>
      `;
    })
    .join("");
  const filterLine = [
    filters.mission_id ? `mission=${escapeHtml(filters.mission_id)}` : "mission=all",
    filters.context_type ? `context=${escapeHtml(filters.context_type)}` : "context=all",
    filters.context_id ? `context_id=${escapeHtml(filters.context_id)}` : "context_id=all",
    filters.session_id ? `session=${escapeHtml(filters.session_id)}` : "session=all",
  ].join(" · ");

  root.innerHTML = `
    <div class="card-list">
      <div class="card">
        <h3>Agent ${escapeHtml(audit.agent_id)} tool audit</h3>
        <p><strong>Total calls:</strong> ${escapeHtml(String(summary.total_tool_calls || 0))}</p>
        <p><strong>Unique tools:</strong> ${escapeHtml(String(summary.unique_tools || 0))}</p>
        <p><strong>Unique missions:</strong> ${escapeHtml(String(summary.unique_missions || 0))}</p>
        <p><strong>Unique sessions:</strong> ${escapeHtml(String(summary.unique_sessions || 0))}</p>
        <p><strong>Filters:</strong> ${filterLine}</p>
        <p>Updated: ${escapeHtml((audit.updated_at || "").slice(0, 10) || "n/a")}</p>
      </div>
      <h3>Top tools</h3>
      ${toolRows || "<p>No tool activity yet.</p>"}
      <h3>Top missions</h3>
      ${missionRows || "<p>No mission activity yet.</p>"}
      <h3>Top sessions</h3>
      ${sessionRows || "<p>No session activity yet.</p>"}
    </div>
  `;
}

function renderAgentOverlayHistory(payload) {
  const root = qs("#agentOverlayHistory");
  if (!root) {
    return;
  }
  if (!payload || !payload.actor_id) {
    root.innerHTML = "<p>Select an agent to load overlay history.</p>";
    return;
  }

  const history = Array.isArray(payload.history) ? payload.history : [];
  const rows = history
    .map((entry) => {
      const createdAt = entry.created_at || "n/a";
      const source = entry.source || "n/a";
      const content = entry.content || "";
      const contentPreview = entry.content_preview || "";
      const preview = content ? content : contentPreview ? contentPreview : "no content captured";
      const reduced = entry.reduced ? "reduced" : "full";
      const createdBy = entry.created_by || "n/a";
      const overlayId = entry.overlay_id || "";
      const overlayVersion = entry.overlay_version != null ? String(entry.overlay_version) : "n/a";
      const tokenCap = entry.requested_token_cap != null ? String(entry.requested_token_cap) : "default";
      return `
        <div class="card">
          <h4>${escapeHtml(overlayId || `${entry.actor_id || payload.actor_id}:v${overlayVersion}`)}</h4>
          <p><strong>Version:</strong> ${escapeHtml(overlayVersion)}</p>
          <p><strong>Source:</strong> ${escapeHtml(source)}</p>
          <p><strong>Created by:</strong> ${escapeHtml(createdBy)}</p>
          <p><strong>Token cap:</strong> ${escapeHtml(tokenCap)} · <strong>Storage:</strong> ${escapeHtml(reduced)}</p>
          <p><strong>Created:</strong> ${escapeHtml(createdAt)}</p>
          <p><strong>Content:</strong> ${escapeHtml(preview)}</p>
        </div>
      `;
    })
    .join("");

  root.innerHTML = `
    <div class="card-list">
      <div class="card">
        <h3>Overlay history for ${escapeHtml(payload.actor_id)}</h3>
        <p><strong>Entries:</strong> ${escapeHtml(String(payload.count || history.length || 0))}</p>
        ${payload.error ? `<p class="status err">Backend warning: ${escapeHtml(payload.error)}</p>` : ""}
      </div>
      ${rows || "<p>No overlay history available.</p>"}
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
  const routingHealth = qs("#providerRoutingHealth");
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
  const routingSummary = providerRoutingSummaryApi()?.buildProviderRoutingSummary(metadata) || null;
  if (!Array.isArray(nodes) || nodes.length === 0) {
    container.innerHTML = "<p>No enabled providers in chain.</p>";
    if (summary) {
      summary.textContent = metadata.selection_strategy || "No enabled providers in chain.";
    }
    if (routingHealth) {
      routingHealth.textContent = routingSummary?.summary || "routing blocked";
      routingHealth.className = `status ${routingSummary?.tone || "err"}`;
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
    const routingState = routingSummary
      ? `${routingSummary.summary}. Available/enabled: ${routingSummary.metric}. ${routingSummary.detail}. `
      : "";
    summary.textContent = `${routingState}${metadata.selection_strategy}. Active provider: ${activeProvider}. Active/total: ${totals}. Intent: ${intent}. ${rationale}`;
  }
  if (routingHealth) {
    routingHealth.textContent = routingSummary?.summary || "routing state unavailable";
    routingHealth.className = `status ${routingSummary?.tone || ""}`;
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
  const healthApi = providerHealthUiApi();
  const healthSummary = healthApi ? healthApi.buildProviderHealthSummary(payload) : null;
  const actionApi = providerActionUiApi();
  const actionSummary = actionApi && payload.last_action_result
    ? actionApi.renderProviderActionResult(payload.last_action_result)
    : "";
  const rows = history
    .map((item) => `<li>${formatDateOrDash(item.tested_at)} — ${escapeHtml(item.status || "unknown")} (${item.models_count || 0} model(s), ${item.latency_ms || 0}ms)</li>`)
    .join("");

  return `
    ${actionSummary}
    ${healthSummary ? healthApi.renderProviderHealthSummary(healthSummary) : ""}
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

function renderPluginDiagnosticsHtml(payload) {
  if (!payload) {
    return "<p>Plugin diagnostics unavailable.</p>";
  }
  const reference = payload.reference || {};
  const ui = payload.ui || {};
  const tools = payload.tools || {};
  const runtime = payload.runtime || {};
  const dependencies = payload.dependencies || {};
  const warnings = Array.isArray(payload.warnings?.entries) ? payload.warnings.entries : [];
  const warningRows = warnings
    .map((warning) => {
      const code = escapeHtml(warning.code || "UNKNOWN");
      const source = escapeHtml(warning.source || "runtime");
      const severity = escapeHtml(String(warning.severity || "warn"));
      const message = escapeHtml(warning.message || "No message provided.");
      return `<li><strong>${source}</strong> (${code}) [${severity}] ${message}</li>`;
    })
    .join("");
  const plugin = payload.plugin || {};
  const status = escapeHtml(payload.status || "ok");
  const warningCount = Number(payload.warnings?.count || 0);
  const errorCount = Number(payload.errors?.count || 0);
  const aliasText = Array.isArray(reference.aliases) && reference.aliases.length
    ? reference.aliases.map(escapeHtml).join(", ")
    : "none";
  const dependsText = Array.isArray(dependencies.declared) && dependencies.declared.length
    ? dependencies.declared.map(escapeHtml).join(", ")
    : "none";

  return `
    <div class="status-grid">
      <div><strong>Plugin:</strong> ${escapeHtml(plugin.id || "unknown")}</div>
      <div><strong>Status:</strong> ${status}</div>
      <div><strong>Warnings:</strong> ${warningCount}</div>
      <div><strong>Errors:</strong> ${errorCount}</div>
      <div><strong>UI actions:</strong> ${ui.action_count || 0}</div>
      <div><strong>Debug action:</strong> ${ui.has_debug_action ? "exposed" : "missing"}</div>
      <div><strong>Tools (active/total):</strong> ${tools.active || 0}/${tools.total || 0}</div>
      <div><strong>Dependencies:</strong> ${dependencies.count || 0}</div>
      <div><strong>Tool conflicts:</strong> ${tools.conflict_count || 0}</div>
    </div>
    <h4>Reference health</h4>
    <ul class="compact-list">
      <li>Required: ${reference.required ? "yes" : "no"}</li>
      <li>Signature required: ${reference.signature_required ? "yes" : "no"}</li>
      <li>Aliases: ${aliasText}</li>
      <li>Matched alias: ${escapeHtml(reference.matched_alias || "unset")}</li>
      <li>Depends on: ${dependsText}</li>
    </ul>
    <h4>Runtime debug</h4>
    <ul class="compact-list">
      <li><strong>Action:</strong> ${escapeHtml(runtime.action || "debug")} (${escapeHtml(runtime.method || "GET")})</li>
      <li><strong>Called:</strong> ${runtime.runtime_called ? "yes" : "no"}</li>
      <li><strong>Success:</strong> ${runtime.success ? "yes" : "no"}</li>
      <li><strong>Message:</strong> ${escapeHtml(runtime.message || "n/a")}</li>
    </ul>
    <h4>Warnings (${warningRows ? warnings.length : 0})</h4>
    <ul class="compact-list">
      ${warningRows || "<li>No warnings.</li>"}
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
  syncMobilePairingAccess();
  syncRuntimeControlAccess();
}

function buildProviderCard(provider) {
  const metadata = parseProviderMetadata(provider.metadata);
  const healthApi = providerHealthUiApi();
  const healthSummary = healthApi ? healthApi.buildProviderHealthSummary({ provider, metadata }) : null;
  const selectedClass = state.selectedProviderId === provider.id ? " provider-card-selected" : "";
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
    <div class="card${selectedClass}" data-provider-card-id="${escapeHtml(provider.id)}">
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
      ${healthSummary ? healthApi.renderProviderHealthSummary(healthSummary) : ""}
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

function focusProviderCard(providerId, options = {}) {
  const normalizedId = `${providerId || ""}`.trim();
  state.selectedProviderId = normalizedId;
  if (!normalizedId || options.scroll === false) {
    return;
  }
  const selectorId = window.CSS && typeof window.CSS.escape === "function"
    ? window.CSS.escape(normalizedId)
    : normalizedId.replace(/["\\]/g, "\\$&");
  window.requestAnimationFrame(() => {
    const card = qs(`[data-provider-card-id="${selectorId}"]`);
    if (card && typeof card.scrollIntoView === "function") {
      card.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
}

function recordProviderActionResult(providerId, payload) {
  const normalizedId = `${providerId || ""}`.trim();
  if (!normalizedId) {
    return null;
  }
  const api = providerActionUiApi();
  if (!api) {
    return null;
  }
  const normalized = api.normalizeProviderActionResult({
    providerId: normalizedId,
    ...payload,
    recordedAt: payload?.recordedAt || new Date().toISOString(),
  });
  state.providerActionResults = {
    ...(state.providerActionResults || {}),
    [normalizedId]: normalized,
  };
  if (state.providerDiagnostics?.provider_id === normalizedId) {
    state.providerDiagnostics = {
      ...state.providerDiagnostics,
      last_action_result: normalized,
    };
    const diagnosticsRoot = qs("#providerDiagnostics");
    if (diagnosticsRoot) {
      diagnosticsRoot.innerHTML = renderProviderDiagnosticsHtml(state.providerDiagnostics);
    }
  }
  return normalized;
}

function buildAgentCard(agent) {
  const overlay = agent.overlay || {};
  const overlayFound = Boolean(overlay.found);
  const overlayData = (overlay.overlay && typeof overlay.overlay === "object") ? overlay.overlay : {};
  const isAdmin = state.role === "admin";
  const lifecycle = agent.lifecycle || "active";
  const overlayContent = isAdmin ? `${overlayData.content || ""}` : `${overlayData.content_preview || ""}`;
  const overlayCap = overlayData.requested_token_cap != null ? String(overlayData.requested_token_cap) : "";
  const overlaySource = overlayData.source || "runtime";
  const lifecycleMeta = lifecycle === "archived" && agent.archived_at
    ? ` · archived ${escapeHtml(agent.archived_at.slice(0, 10))}`
    : "";
  const lifecycleButton = lifecycle === "archived"
    ? `<button type="button" data-action="restore-agent" data-id="${agent.id}">Restore</button>`
    : `<button type="button" data-action="archive-agent" data-id="${agent.id}">Archive</button>`;

  return `
    <div class="card">
      <h3>${agent.name}</h3>
      <p><strong>Status:</strong> ${agent.status}</p>
      <p><strong>Role:</strong> ${agent.role}</p>
      <p><strong>Lifecycle:</strong> ${escapeHtml(lifecycle)}${lifecycleMeta}</p>
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
        <button type="button" data-action="open-agent-overlay-history" data-id="${agent.id}">
          Overlay history
        </button>
        ${isAdmin ? lifecycleButton : ""}
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
  const uiSpec = metadata.ui || {};
  const uiSections = Array.isArray(uiSpec.sections) ? uiSpec.sections : [];
  const metadataLabel = Object.keys(metadata).length
    ? `<p><strong>Metadata:</strong> ${escapeHtml(JSON.stringify(metadata))}</p>`
    : "";
  const uiActionLabel = uiSections
    .filter((section) => section && typeof section === "object")
    .map((section) => {
      const sectionTitle = escapeHtml(section.title || section.id || "Actions");
      const sectionDescription = section.description ? `<p>${escapeHtml(section.description)}</p>` : "";
      const actionCards = Array.isArray(section.actions)
        ? section.actions
            .filter((action) => action && typeof action === "object" && action.id)
            .map((action) => {
              const actionId = String(action.id).trim();
              const actionLabel = escapeHtml(action.label || actionId);
              const actionDesc = action.description ? `<p>${escapeHtml(action.description)}</p>` : "";
              const actionMethod = String(action.method || "POST").trim().toUpperCase() || "POST";
              const fields = Array.isArray(action.fields) ? action.fields : [];
              const fieldRows = fields
                .filter((field) => field && typeof field === "object")
                .map((field) => {
                  const fieldName = String(field.name || "").trim() || String(field.id || "").trim();
                  if (!fieldName) {
                    return "";
                  }
                  const key = escapeHtml(fieldName);
                  const label = escapeHtml(field.label || fieldName);
                  const fieldType = String(field.type || "text").trim().toLowerCase();
                  const defaultValue = field.default === undefined ? "" : field.default;
                  const placeholder = escapeHtml(field.placeholder || "");
                  const isRequired = field.required ? "required" : "";
                  const common = `data-action-id="${escapeHtml(actionId)}" data-plugin-id="${escapeHtml(
                    plugin.id,
                  )}" data-plugin-ui-action-field="1" data-plugin-ui-field="${key}"`;
                  if (fieldType === "textarea") {
                    return `<label>${label}: <textarea ${common} data-plugin-ui-field-type="textarea" ${isLocked} ${isRequired}>${escapeHtml(defaultValue)}</textarea></label>`;
                  }
                  if (fieldType === "checkbox") {
                    const checked = defaultValue ? "checked" : "";
                    return `<label><input type="checkbox" ${checked} ${common} data-plugin-ui-field-type="checkbox" ${isLocked} ${isRequired} /> ${label}</label>`;
                  }
                  if (fieldType === "number") {
                    return `<label>${label}: <input type="number" ${common} data-plugin-ui-field-type="number" value="${escapeHtml(defaultValue)}" placeholder="${placeholder}" ${isLocked} ${isRequired} /></label>`;
                  }
                  if (fieldType === "select" && Array.isArray(field.options)) {
                    const options = field.options
                      .map((option) => {
                        if (option == null) {
                          return "";
                        }
                        const optionValue = typeof option === "string" ? option : option.value;
                        const optionLabel = typeof option === "string" ? option : option.label;
                        if (optionValue == null) {
                          return "";
                        }
                        const optVal = escapeHtml(optionValue);
                        const optLabel = escapeHtml(optionLabel || optionValue);
                        const selected = `${defaultValue}` === `${optionValue}` ? " selected" : "";
                        return `<option value="${optVal}"${selected}>${optLabel}</option>`;
                      })
                      .join("");
                    return `<label>${label}: <select ${common} data-plugin-ui-field-type="select" ${isLocked} ${isRequired}>${options}</select></label>`;
                  }
                  return `<label>${label}: <input type="text" ${common} data-plugin-ui-field-type="text" value="${escapeHtml(
                    defaultValue,
                  )}" placeholder="${placeholder}" ${isLocked} ${isRequired} /></label>`;
                })
                .join("");
              return `
                <div class="plugin-ui-action" data-plugin-id="${escapeHtml(plugin.id)}" data-action-id="${escapeHtml(actionId)}">
                  <p><strong>${actionLabel}</strong> <small>(${escapeHtml(actionMethod)})</small></p>
                  ${actionDesc}
                  ${fieldRows}
                  <p><button type="button" data-action="execute-plugin-ui-action" data-plugin-id="${escapeHtml(
                    plugin.id,
                  )}" data-action-id="${escapeHtml(actionId)}" data-method="${escapeHtml(actionMethod)}" ${isLocked}>Run</button></p>
                  <p class="plugin-ui-action-status status" data-action="plugin-ui-action-status" data-plugin-id="${escapeHtml(
                    plugin.id,
                  )}" data-action-id="${escapeHtml(actionId)}"></p>
                </div>
              `;
            })
            .join("")
        : "";
      if (!actionCards) {
        return "";
      }
      return `
        <div>
          <h4>${sectionTitle}</h4>
          ${sectionDescription}
          <div class="plugin-ui-section-actions">${actionCards}</div>
        </div>
      `;
    })
    .join("");

  const uiSectionBlock = uiActionLabel
    ? `<div class="plugin-ui-sections"><h4>Plugin actions</h4>${uiActionLabel}</div>`
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
      ${uiSectionBlock}
      <p>
        <button type="button" data-action="show-plugin-diagnostics" data-plugin-id="${escapeHtml(plugin.id)}" ${isLocked}>
          Show plugin diagnostics
        </button>
      </p>
      <div class="plugin-ui-sections" data-plugin-debug-root="1" data-plugin-id="${escapeHtml(plugin.id)}"></div>
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
  const runtimeView = runtimeUiApi()?.buildRuntimeViewModel(state.runtimeStatus || {}, {
    services,
    default_service: state.runtimeStatus?.defaultService || "busy",
  }) || {
    summary: {
      defaultService: state.runtimeStatus?.defaultService || "busy",
    },
    services: Array.isArray(services) ? services : [],
  };
  const normalizedServices = runtimeView.services || [];
  const hasServices = normalizedServices.length > 0;
  const select = qs("#runtimeService");
  const previousValue = select?.value || "";
  const defaultService = runtimeView.summary?.defaultService || "busy";
  const options = hasServices
    ? normalizedServices.map((service) => `<option value="${escapeHtml(service.name)}">${escapeHtml(service.name)}</option>`).join("")
    : `<option value="${escapeHtml(defaultService)}">${escapeHtml(defaultService)}</option>`;

  if (select) {
    select.innerHTML = options;
    if (hasServices && normalizedServices.some((service) => service.name === previousValue)) {
      select.value = previousValue;
    } else if (hasServices && normalizedServices.some((service) => service.name === defaultService)) {
      select.value = defaultService;
    } else if (hasServices && normalizedServices[0]) {
      select.value = normalizedServices[0].name;
    } else {
      select.value = defaultService;
    }
  }

  const canControl = state.role === "admin";
  const renderService = (service) => {
    const api = runtimeUiApi();
    if (api) {
      return api.renderServiceCard(service, { canControl });
    }
    return `
      <div class="card">
        <h3>${escapeHtml(service.name)}</h3>
        <p><strong>Running:</strong> ${service.running ? "yes" : "no"}</p>
        <p><strong>PID:</strong> ${service.pid || "-"}</p>
        <p><strong>Log:</strong> ${service.log_file || "-"}</p>
      </div>
    `;
  };

  renderCards("#runtimeServices", normalizedServices, renderService);
  syncRuntimeControlAccess();
}

function syncRuntimeControlAccess() {
  const canControl = state.role === "admin";
  const select = qs("#runtimeService");
  if (select) {
    select.disabled = !canControl;
  }
  document.querySelectorAll('[data-runtime-control="1"]').forEach((node) => {
    node.disabled = !canControl;
  });
  const hint = qs("#runtimeControlHint");
  if (!hint) {
    return;
  }
  if (canControl) {
    hint.textContent = "Admin token can start, stop, and restart runtime services.";
    return;
  }
  if (state.role === "viewer") {
    hint.textContent = "Viewer token can inspect runtime state but cannot control services.";
    return;
  }
  hint.textContent = "Set an admin token to control runtime services.";
}

function setMobilePairingControlsDisabled(disabled) {
  const form = qs("#mobilePairingIssueForm");
  if (form) {
    form.querySelectorAll("input, button").forEach((node) => {
      node.disabled = Boolean(disabled);
    });
  }
  qs('[data-action="refresh-mobile-pairing"]')?.toggleAttribute("disabled", Boolean(disabled));
}

function renderMobilePairingLatest(issue) {
  const root = qs("#mobilePairingLatest");
  if (!root) {
    return;
  }
  if (state.role !== "admin") {
    root.innerHTML = "<p>Admin token required to issue pairing codes.</p>";
    return;
  }
  if (!issue || typeof issue !== "object") {
    root.innerHTML = "<p>No pairing code issued in this browser session.</p>";
    return;
  }
  const qrApi = mobilePairingQrApi();
  let qrMarkup = '<p class="status err">QR renderer unavailable in this browser session.</p>';
  try {
    if (!qrApi) {
      throw new Error("QR renderer unavailable in this browser session.");
    }
    const qrPayload = qrApi.buildPairingQrPayload({
      issue,
      controlPlaneUrl: API_BASE,
      now: new Date(),
    });
    const qrSvg = qrApi.renderPairingQrSvg(qrPayload);
    qrMarkup = `
      <div class="pairing-qr-block">
        <div class="pairing-qr-visual">${qrSvg}</div>
        <p>
          <button
            type="button"
            data-action="copy-mobile-pairing-qr-payload"
            data-payload="${escapeHtml(qrPayload)}"
          >
            Copy QR payload
          </button>
        </p>
      </div>
    `;
  } catch (err) {
    qrMarkup = `<p class="status err">QR unavailable: ${escapeHtml(err.message || String(err))}</p>`;
  }
  root.innerHTML = `
    <div class="card-list">
      <div class="card">
        <h3>Latest issued pairing code</h3>
        <p><strong>Code:</strong> <code>${escapeHtml(issue.pairing_code || "")}</code></p>
        <p><strong>Control plane URL:</strong> <code>${escapeHtml(API_BASE)}</code></p>
        <p><strong>Expires:</strong> ${escapeHtml(issue.expires_at || "n/a")}</p>
        <p><strong>Rooms:</strong> ${Array.isArray(issue.authorized_room_ids) && issue.authorized_room_ids.length ? issue.authorized_room_ids.map(escapeHtml).join(", ") : "n/a"}</p>
        <p><strong>Orchestrators:</strong> ${Array.isArray(issue.orchestrator_scope) && issue.orchestrator_scope.length ? issue.orchestrator_scope.map(escapeHtml).join(", ") : "n/a"}</p>
        <p>
          <button type="button" data-action="copy-mobile-pairing-code" data-code="${escapeHtml(issue.pairing_code || "")}">
            Copy pairing code
          </button>
        </p>
        ${qrMarkup}
        <p class="meta">The raw pairing code is only shown from the live issuance response. Persisted state does not re-display it after refresh.</p>
        <p class="meta">The QR is browser-generated from the live issue response and the active control-plane URL. Reload the page or issue a new code if you need it again.</p>
      </div>
    </div>
  `;
}

function renderMobilePairingState(pairing) {
  const root = qs("#mobilePairingState");
  if (!root) {
    return;
  }
  if (state.role !== "admin") {
    root.innerHTML = "<p>Admin token required to inspect or revoke pairing grants.</p>";
    return;
  }
  if (!pairing || typeof pairing !== "object") {
    root.innerHTML = "<p>No pairing state loaded.</p>";
    return;
  }

  const issued = Array.isArray(pairing.issued) ? pairing.issued : [];
  const revoked = Array.isArray(pairing.revoked) ? pairing.revoked : [];
  const issuedCards = issued.map((entry) => {
    const tokenId = entry.token_id ? escapeHtml(entry.token_id) : "n/a";
    const canRevoke = Boolean(entry.token_id) && entry.status === "active";
    const revokeButton = canRevoke
      ? `<button type="button" data-action="revoke-mobile-pairing" data-token-id="${tokenId}">Revoke grant</button>`
      : "";
    return `
      <div class="card">
        <h3>${formatProviderStatusChip(entry.status || "unknown")} ${escapeHtml(entry.device_label || "unknown device")}</h3>
        <p><strong>Code hash:</strong> ${escapeHtml(entry.pairing_code_hash || "n/a")}</p>
        <p><strong>Rooms:</strong> ${Array.isArray(entry.authorized_room_ids) && entry.authorized_room_ids.length ? entry.authorized_room_ids.map(escapeHtml).join(", ") : "n/a"}</p>
        <p><strong>Orchestrators:</strong> ${Array.isArray(entry.orchestrator_scope) && entry.orchestrator_scope.length ? entry.orchestrator_scope.map(escapeHtml).join(", ") : "n/a"}</p>
        <p><strong>Issued:</strong> ${escapeHtml(entry.issued_at || "n/a")}</p>
        <p><strong>Code expires:</strong> ${escapeHtml(entry.expires_at || "n/a")}</p>
        <p><strong>Consumed:</strong> ${escapeHtml(entry.consumed_at || "n/a")}</p>
        <p><strong>Token ID:</strong> ${tokenId}</p>
        <p><strong>Token expires:</strong> ${escapeHtml(entry.token_expires_at || "n/a")}</p>
        <p><strong>Revoked:</strong> ${escapeHtml(entry.revoked_at || "n/a")}</p>
        ${revokeButton ? `<p>${revokeButton}</p>` : ""}
      </div>
    `;
  }).join("");

  const revokedCards = revoked.map((entry) => `
    <div class="card">
      <h3>${formatProviderStatusChip("revoked")} ${escapeHtml(entry.token_id || "unknown token")}</h3>
      <p><strong>Revoked at:</strong> ${escapeHtml(entry.revoked_at || "n/a")}</p>
      <p><strong>Revoked by:</strong> ${escapeHtml(entry.revoked_by || "n/a")}</p>
    </div>
  `).join("");

  root.innerHTML = `
    <div class="card-list">
      <div class="card">
        <h3>Pairing state</h3>
        <p><strong>Instance:</strong> ${escapeHtml(pairing.instance_id || "n/a")}</p>
        <p><strong>Issued records:</strong> ${escapeHtml(String(issued.length))}</p>
        <p><strong>Revoked tokens:</strong> ${escapeHtml(String(revoked.length))}</p>
      </div>
      ${issuedCards || "<p>No pairing grants recorded.</p>"}
      <h3>Revoked grants</h3>
      ${revokedCards || "<p>No revoked grants.</p>"}
    </div>
  `;
}

function syncMobilePairingAccess() {
  const locked = state.role !== "admin";
  setMobilePairingControlsDisabled(locked);
  renderMobilePairingLatest(state.latestPairingIssue);
  renderMobilePairingState(state.mobilePairingState);
}

function parseCommaSeparatedList(raw) {
  return String(raw || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const helper = document.createElement("textarea");
  helper.value = text;
  helper.setAttribute("readonly", "readonly");
  helper.style.position = "absolute";
  helper.style.left = "-9999px";
  document.body.appendChild(helper);
  helper.select();
  document.execCommand("copy");
  document.body.removeChild(helper);
}

async function submitMobilePairingIssue(event) {
  event.preventDefault();
  if (state.role !== "admin") {
    setStatus("#mobilePairingStatus", "admin token required", "err");
    return;
  }
  const deviceLabel = qs('input[name="mobilePairingDeviceLabel"]')?.value.trim() || "";
  const roomIds = parseCommaSeparatedList(qs('input[name="mobilePairingRoomIds"]')?.value || "");
  const orchestrators = parseCommaSeparatedList(qs('input[name="mobilePairingOrchestrators"]')?.value || "");
  const ttlValue = qs('input[name="mobilePairingTtlSec"]')?.value.trim() || "";
  const ttlSec = Number(ttlValue || "300");

  if (!deviceLabel) {
    setStatus("#mobilePairingStatus", "device label is required", "err");
    return;
  }
  if (!roomIds.length) {
    setStatus("#mobilePairingStatus", "at least one room id is required", "err");
    return;
  }
  if (!orchestrators.length) {
    setStatus("#mobilePairingStatus", "at least one orchestrator id is required", "err");
    return;
  }
  if (!Number.isInteger(ttlSec) || ttlSec <= 0) {
    setStatus("#mobilePairingStatus", "ttl_sec must be a positive integer", "err");
    return;
  }

  setStatus("#mobilePairingStatus", "issuing pairing code...", "");
  try {
    const payload = await postJson("/api/mobile/pairing/issue", {
      device_label: deviceLabel,
      authorized_room_ids: roomIds,
      orchestrator_scope: orchestrators,
      ttl_sec: ttlSec,
    });
    state.latestPairingIssue = payload.pairing || null;
    renderMobilePairingLatest(state.latestPairingIssue);
    setStatus("#mobilePairingStatus", `pairing code issued for ${deviceLabel}`, "ok");
    await loadMobilePairingState();
  } catch (err) {
    setStatus("#mobilePairingStatus", `pairing issue failed: ${err.message}`, "err");
  }
}

async function loadMobilePairingState() {
  if (state.role !== "admin") {
    state.mobilePairingState = null;
    syncMobilePairingAccess();
    return;
  }
  try {
    const payload = await fetchJson("/api/mobile/pairing/state");
    state.mobilePairingState = payload.pairing || null;
    renderMobilePairingState(state.mobilePairingState);
  } catch (err) {
    state.mobilePairingState = null;
    renderMobilePairingState(null);
    setStatus("#mobilePairingStatus", `pairing state failed: ${err.message}`, "err");
  }
}

function buildEventFilterState() {
  return {
    domain: qs("#eventDomainFilter")?.value || "",
    level: qs("#eventLevelFilter")?.value || "",
    query: qs("#eventQueryFilter")?.value.trim() || "",
    limit: Math.min(100, Math.max(1, Number(qs("#eventLimitFilter")?.value || 50))),
  };
}

function updateEventFilterOptions(availableDomains) {
  const select = qs("#eventDomainFilter");
  if (!select) {
    return;
  }
  const currentValue = select.value || "";
  const preferredOrder = ["runtime", "gm_ticket", "import", "agent", "provider", "plugin", "appearance", "mobile", "chat", "memory"];
  const seen = new Set([""]);
  const orderedDomains = [""];
  for (const candidate of preferredOrder) {
    if (Array.isArray(availableDomains) && availableDomains.includes(candidate) && !seen.has(candidate)) {
      seen.add(candidate);
      orderedDomains.push(candidate);
    }
  }
  for (const candidate of availableDomains || []) {
    if (!seen.has(candidate)) {
      seen.add(candidate);
      orderedDomains.push(candidate);
    }
  }
  select.innerHTML = orderedDomains
    .map((domain) => {
      const label = domain || "all";
      return `<option value="${escapeHtml(domain)}">${escapeHtml(label)}</option>`;
    })
    .join("");
  select.value = seen.has(currentValue) ? currentValue : "";
}

function renderDashboardOverview() {
  const root = qs("#dashboardOverview");
  const headline = qs("#dashboardOverviewHeadline");
  if (!root || !headline) {
    return;
  }
  const api = dashboardOverviewApi();
  if (!api) {
    headline.textContent = "Dashboard overview helper unavailable.";
    headline.className = "status err";
    root.innerHTML = "";
    return;
  }
  const view = api.buildDashboardViewModel({
    runtimeStatus: state.runtimeStatus,
    runtimeServices: state.runtimeServices,
    providers: state.providers,
    providerChain: state.providerChain,
    providerActionResults: state.providerActionResults,
    gmTickets: state.gmTickets,
    startupSummaryEvent: state.startupDebugSummary,
    events: state.events,
    seenEventIds: state.attentionSeen?.event_ids || [],
    cardReviewTimestamps: state.attentionSeen?.card_reviews || {},
    now: new Date().toISOString(),
  });
  state.dashboardOverview = view;
  const selectedCardStillPresent = Array.isArray(view.cards)
    && view.cards.some((card) => card.id === state.selectedAttentionCardId);
  if (!selectedCardStillPresent) {
    state.selectedAttentionCardId = view.defaultCardId || "";
  }
  headline.textContent = view.headline || "Control-plane overview unavailable.";
  headline.className = `status ${view.tone || ""}`.trim();
  root.innerHTML = (view.cards || [])
    .map((card) => api.renderDashboardCard({
      ...card,
      selected: card.id === state.selectedAttentionCardId,
    }))
    .join("");
  renderDashboardAttentionHistory();
}

function loadAttentionSeenState() {
  const api = attentionStateApi();
  if (!api) {
    state.attentionSeen = { ...SEEN_EVENT_STORAGE_FALLBACK };
    return state.attentionSeen;
  }
  state.attentionSeen = api.readSeenState(window.localStorage);
  return state.attentionSeen;
}

function markVisibleEventsSeen() {
  const api = attentionStateApi();
  if (!api) {
    return;
  }
  state.attentionSeen = api.markEventsSeen(window.localStorage, state.attentionSeen, state.visibleEvents || []);
}

function renderDashboardAttentionHistory() {
  const root = qs("#dashboardAttentionHistory");
  if (!root) {
    return;
  }
  const overview = state.dashboardOverview;
  const eventsUi = eventsUiApi();
  if (!overview || !Array.isArray(overview.cards)) {
    root.innerHTML = "<p>Attention history unavailable.</p>";
    return;
  }
  const card = overview.cards.find((entry) => entry.id === state.selectedAttentionCardId);
  if (!card) {
    root.innerHTML = "<p>Select a summary card to inspect recent unseen events.</p>";
    return;
  }
  const focus = card.focus || {};
  const remediationItems = Array.isArray(card.remediationItems) ? card.remediationItems : [];
  const primaryProviderId = remediationItems[0]?.providerId || "";
  const freshness = card.freshness && typeof card.freshness === "object" ? card.freshness : {};
  const reviewState = card.reviewState && typeof card.reviewState === "object" ? card.reviewState : null;
  const attentionEventCount = Array.isArray(card.attentionEvents) ? card.attentionEvents.length : 0;
  const seenHistoryCount = Math.max(0, Number(card.seenHistoryCount || 0));
  const showSeenHistory = Boolean(state.attentionHistoryExpandedCards?.[card.id]);
  const seenHistoryEvents = Array.isArray(card.seenHistoryEvents) ? card.seenHistoryEvents : [];
  const seenHistorySummary = `${card.seenHistorySummary || ""}`.trim();
  const seenHistoryShowLabel = `${card.seenHistoryToggleLabel || `Show seen history (${seenHistoryCount})`}`.trim();
  const seenHistoryHideLabel = `${card.seenHistoryToggleHideLabel || "Hide seen history"}`.trim();
  const eventsSeenButton = attentionEventCount > 0
    ? `<button type="button" data-action="mark-dashboard-card-events-seen" data-card-id="${escapeHtml(card.id || "")}">Mark ${attentionEventCount} attached event${attentionEventCount === 1 ? "" : "s"} seen</button>`
    : "";
  const seenHistoryToggle = seenHistoryCount > 0
    ? `
      <p class="control-row dashboard-attention-history-toggle">
        <button type="button" data-action="toggle-dashboard-seen-history" data-card-id="${escapeHtml(card.id || "")}">
          ${showSeenHistory ? escapeHtml(seenHistoryHideLabel) : escapeHtml(seenHistoryShowLabel)}
        </button>
        ${seenHistorySummary ? `<span class="meta">${escapeHtml(seenHistorySummary)}.</span>` : ""}
      </p>
    `
    : "";
  const remediationBlock = remediationItems.length > 0
    ? `
      <div class="provider-health-summary dashboard-attention-recommendations">
        <p><strong>Recommended remediation</strong></p>
        <p class="meta">${escapeHtml(card.remediationHint || "")}</p>
        <ul class="compact-list provider-issue-list">
          ${remediationItems.map((item) => `
            <li>
              <strong>${escapeHtml(item.providerId || "provider")}: ${escapeHtml(item.label || "issue")}</strong>
              <span class="status ${escapeHtml(item.tone || "warn")}">${escapeHtml(item.tone === "err" ? "critical" : "review")}</span>
              <div>${escapeHtml(item.detail || "")}</div>
              <div class="meta">${escapeHtml(item.action || "")}</div>
              <div>
                <button
                  type="button"
                  data-action="open-dashboard-panel"
                  data-panel-id="providersPanel"
                  data-provider-status="${escapeHtml(focus.providerStatus || "")}"
                  data-provider-id="${escapeHtml(item.providerId || "")}"
                >Open diagnostics</button>
              </div>
            </li>
          `).join("")}
        </ul>
      </div>
    `
    : "";
  const actionButtons = `
    <p class="control-row">
      <button
        type="button"
        data-action="open-dashboard-panel"
        data-panel-id="${escapeHtml(focus.panel || "")}"
        data-provider-status="${escapeHtml(focus.providerStatus || "")}"
        data-provider-id="${escapeHtml(primaryProviderId)}"
        data-gm-status="${escapeHtml(focus.gmStatus || "")}"
        data-gm-priority="${escapeHtml(focus.gmPriority || "")}"
      >Open related panel</button>
      <button type="button" data-action="mark-dashboard-card-reviewed" data-card-id="${escapeHtml(card.id || "")}">Mark summary reviewed</button>
      ${eventsSeenButton}
    </p>
  `;
  const eventRows = Array.isArray(card.attentionEvents) && card.attentionEvents.length > 0 && eventsUi
    ? card.attentionEvents.map((event) => eventsUi.renderEventItem(event, {
      actionName: "mark-dashboard-attention-event-seen",
      actionLabel: "Mark event seen",
      eventData: {
        cardId: card.id || "",
      },
    })).join("")
    : '<li class="card event-card"><p>No unseen events currently attached to this summary.</p></li>';
  const seenHistoryRows = showSeenHistory
    ? seenHistoryEvents.length > 0 && eventsUi
      ? seenHistoryEvents.map((event) => eventsUi.renderEventItem(event)).join("")
      : '<li class="card event-card"><p>No seen events are currently available for this summary.</p></li>'
    : "";
  root.innerHTML = `
    <div class="card">
      <h3>${escapeHtml(card.title || "Attention")}</h3>
      <p><strong>${escapeHtml(card.summary || "")}</strong></p>
      <p class="meta">${escapeHtml(card.detail || "")}</p>
      ${freshness.label ? `<p class="meta dashboard-remediation-hint"><strong>Freshness:</strong> ${escapeHtml(freshness.label)} ${escapeHtml(freshness.detail || "")}</p>` : ""}
      ${reviewState?.label ? `<p class="meta dashboard-remediation-hint"><strong>Review:</strong> ${escapeHtml(reviewState.label)}${reviewState.detail ? ` - ${escapeHtml(reviewState.detail)}` : ""}</p>` : ""}
      ${card.remediationHint ? `<p class="meta dashboard-remediation-hint"><strong>Next:</strong> ${escapeHtml(card.remediationHint)}</p>` : ""}
      ${card.actionSummary ? `<p class="meta dashboard-remediation-hint"><strong>Latest:</strong> ${escapeHtml(card.actionSummary)}</p>` : ""}
      ${remediationBlock}
      ${actionButtons}
    </div>
    <ul class="compact-list">
      ${eventRows}
    </ul>
    ${seenHistoryToggle}
    ${showSeenHistory ? `<ul class="compact-list">${seenHistoryRows}</ul>` : ""}
  `;
}

function selectDashboardAttentionCard(target) {
  state.selectedAttentionCardId = target?.dataset?.cardId || "";
  renderDashboardAttentionHistory();
}

function toggleDashboardSeenHistory(target) {
  const cardId = `${target?.dataset?.cardId || ""}`.trim();
  if (!cardId) {
    return;
  }
  state.attentionHistoryExpandedCards = {
    ...state.attentionHistoryExpandedCards,
    [cardId]: !state.attentionHistoryExpandedCards?.[cardId],
  };
  renderDashboardAttentionHistory();
}

function markDashboardCardSeen(target) {
  const cardId = target?.dataset?.cardId || "";
  if (!cardId) {
    return;
  }
  const api = attentionStateApi();
  if (!api) {
    return;
  }
  state.attentionSeen = api.markCardReviewed(window.localStorage, state.attentionSeen, cardId);
  renderDashboardOverview();
}

function markDashboardCardEventsSeen(target) {
  const cardId = target?.dataset?.cardId || "";
  const overview = state.dashboardOverview;
  if (!cardId || !overview || !Array.isArray(overview.cards)) {
    return;
  }
  const card = overview.cards.find((entry) => entry.id === cardId);
  if (!card) {
    return;
  }
  const api = attentionStateApi();
  if (!api) {
    return;
  }
  state.attentionSeen = api.markEventsSeen(window.localStorage, state.attentionSeen, card.attentionEvents || []);
  renderDashboardOverview();
}

function markDashboardAttentionEventSeen(target) {
  const eventId = `${target?.dataset?.eventId || ""}`.trim();
  if (!eventId) {
    return;
  }
  const api = attentionStateApi();
  if (!api) {
    return;
  }
  const event = (Array.isArray(state.events) ? state.events : []).find(
    (entry) => `${entry?.id || ""}`.trim() === eventId,
  );
  if (!event) {
    return;
  }
  state.attentionSeen = api.markEventsSeen(window.localStorage, state.attentionSeen, [event]);
}

async function openDashboardPanel(target) {
  const panelId = target?.dataset?.panelId || "";
  const providerStatus = target?.dataset?.providerStatus || "";
  const providerId = target?.dataset?.providerId || "";
  const gmStatus = target?.dataset?.gmStatus || "";
  const gmPriority = target?.dataset?.gmPriority || "";
  let scrollTargetId = panelId;

  if (panelId === "providersPanel") {
    const statusSelect = qs("#providerFilterStatus");
    if (statusSelect) {
      statusSelect.value = providerStatus;
    }
    await loadProviders();
    if (providerId) {
      focusProviderCard(providerId, { scroll: false });
      await loadProviderDiagnostics(providerId);
      setStatus("#providerChainStatus", `Diagnostics loaded for ${providerId}`, "ok");
      scrollTargetId = "providerDiagnosticsPanel";
    }
  } else if (panelId === "gmTicketsPanel") {
    const statusSelect = qs("#gmTicketFilterStatus");
    const prioritySelect = qs("#gmTicketFilterPriority");
    if (statusSelect) {
      statusSelect.value = gmStatus;
    }
    if (prioritySelect) {
      prioritySelect.value = gmPriority;
    }
    await loadGmTickets();
  } else if (panelId === "runtimePanel") {
    await loadRuntime();
  }

  const panel = scrollTargetId ? qs(`#${scrollTargetId}`) : null;
  if (panel && typeof panel.scrollIntoView === "function") {
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderEvents(items) {
  const list = qs("#events");
  if (!list) {
    return;
  }
  const startupSummary = (items || []).find(
    (event) => String(event?.type || "").trim() === "plugin.startup_debug_summary",
  );
  state.startupDebugSummary = startupSummary || null;
  renderStartupDebugSummary(startupSummary);
  const eventsUi = eventsUiApi();
  if (!eventsUi) {
    list.innerHTML = (items || [])
      .map((event) => `<li>${escapeHtml(event.created_at)} - [${escapeHtml(event.level)}] ${escapeHtml(event.message)}</li>`)
      .join("");
    return;
  }

  state.eventFilters = buildEventFilterState();
  const view = eventsUi.buildEventViewModel(items || [], {
    ...state.eventFilters,
    seenEventIds: state.attentionSeen?.event_ids || [],
  });
  state.visibleEvents = view.events || [];
  updateEventFilterOptions(view.availableDomains);
  const summary = qs("#eventFilterSummary");
  if (summary) {
    const domainParts = Object.entries(view.summary.domains || {})
      .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
      .slice(0, 6)
      .map(([domain, count]) => `${domain}:${count}`);
    const filterParts = [];
    if (view.filters.domain) {
      filterParts.push(`domain=${view.filters.domain}`);
    }
    if (view.filters.level) {
      filterParts.push(`level=${view.filters.level}`);
    }
    if (view.filters.query) {
      filterParts.push(`query=${view.filters.query}`);
    }
    summary.textContent = `${view.filteredCount}/${view.summary.total} shown • ${view.unseenCount || 0} new • ${domainParts.join(" • ") || "no events"}${filterParts.length ? ` • filters: ${filterParts.join(", ")}` : ""}`;
  }
  list.innerHTML = view.events
    .map((event) => eventsUi.renderEventItem(event))
    .join("") || '<li class="card event-card"><p>No events match the current filters.</p></li>';
  renderDashboardOverview();
}

function renderStartupDebugSummary(summaryEvent) {
  const summaryRoot = qs("#startupDebugSummary");
  if (!summaryRoot) {
    return;
  }
  if (!summaryEvent) {
    summaryRoot.innerHTML = "<p>Startup debug summary unavailable yet.</p>";
    return;
  }

  const payload = summaryEvent.payload || {};
  const total = Number(payload.plugin_count || 0);
  const checked = Number(payload.checked || 0);
  const called = Number(payload.runtime_called || 0);
  const passed = Number(payload.runtime_success || 0);
  const missing = Number(payload.missing_debug || 0);
  const requiredTotal = Number(payload.required_total || 0);
  const requiredPresent = Number(payload.required_present || 0);
  const requiredMissing = Number(payload.required_missing || 0);
  const requiredDisabled = Number(payload.required_disabled || 0);
  const errors = Number(payload.error_count || 0);
  const warnings = Number(payload.warn_count || 0);
  const level = String(summaryEvent.level || "info").trim();
  const missingPlugins = Array.isArray(payload.missing_debug_plugins) ? payload.missing_debug_plugins : [];
  const requiredMissingPlugins = Array.isArray(payload.required_missing_plugins) ? payload.required_missing_plugins : [];
  const requiredDisabledPlugins = Array.isArray(payload.required_disabled_plugins) ? payload.required_disabled_plugins : [];
  const warnPlugins = Array.isArray(payload.warn_plugins) ? payload.warn_plugins : [];
  const errorPlugins = Array.isArray(payload.error_plugins) ? payload.error_plugins : [];
  const checkedPlugins = Array.isArray(payload.checked_plugins) ? payload.checked_plugins : [];
  const checkedList = checkedPlugins
    .slice(0, 8)
    .map((pluginId) => `<li>${escapeHtml(String(pluginId))}</li>`)
    .join("");
  const moreChecked = checkedPlugins.length > 8 ? `<li>… and ${checkedPlugins.length - 8} more</li>` : "";
  const warnList = warnPlugins
    .slice(0, 8)
    .map((pluginId) => `<li>${escapeHtml(String(pluginId))}</li>`)
    .join("");
  const moreWarn = warnPlugins.length > 8 ? `<li>… and ${warnPlugins.length - 8} more</li>` : "";
  const errorList = errorPlugins
    .slice(0, 8)
    .map((pluginId) => `<li>${escapeHtml(String(pluginId))}</li>`)
    .join("");
  const moreError = errorPlugins.length > 8 ? `<li>… and ${errorPlugins.length - 8} more</li>` : "";

  const statusLine = level === "error"
    ? "Critical issues detected"
    : level === "warn"
    ? "Warnings detected"
    : "No startup warnings";
  const statusClass = level === "error" ? "status err" : level === "warn" ? "status" : "status ok";

  summaryRoot.className = `${statusClass} card`;
  summaryRoot.innerHTML = `
    <p><strong>${escapeHtml(statusLine)}</strong></p>
    <p>Checked ${checked}/${total} plugins</p>
    <p>Debug executed ${called} times; passed ${passed}</p>
    <p>Required core plugins present: ${requiredPresent}/${requiredTotal}</p>
    <p>Required core plugins missing: ${requiredMissing}</p>
    <p>Required core plugins disabled: ${requiredDisabled}</p>
    <p>Missing debug action on ${missing} plugin(s)</p>
    <p>Warnings: ${warnings} • Errors: ${errors}</p>
    ${warnPlugins.length ? `<p>Warned plugins:</p><ul>${warnList}${moreWarn}</ul>` : "<p>Warned plugins: none</p>"}
    ${missingPlugins.length ? `<p>Missing debug action:</p><ul>${missingPlugins.slice(0, 8).map((pluginId) => `<li>${escapeHtml(String(pluginId))}</li>`).join("")}${missingPlugins.length > 8 ? `<li>… and ${missingPlugins.length - 8} more</li>` : ""}</ul>` : ""}
    ${errorPlugins.length ? `<p>Error plugins:</p><ul>${errorList}${moreError}</ul>` : "<p>Error plugins: none</p>"}
    ${requiredMissingPlugins.length ? `<p>Missing required plugins:</p><ul>${requiredMissingPlugins.slice(0, 8).map((pluginId) => `<li>${escapeHtml(String(pluginId))}</li>`).join("")}${requiredMissingPlugins.length > 8 ? `<li>… and ${requiredMissingPlugins.length - 8} more</li>` : ""}</ul>` : ""}
    ${requiredDisabledPlugins.length ? `<p>Disabled required plugins:</p><ul>${requiredDisabledPlugins.slice(0, 8).map((pluginId) => `<li>${escapeHtml(String(pluginId))}</li>`).join("")}${requiredDisabledPlugins.length > 8 ? `<li>… and ${requiredDisabledPlugins.length - 8} more</li>` : ""}</ul>` : ""}
    <p>Sample checked plugins:</p>
    <ul>${checkedList}${moreChecked}</ul>
  `;
}

function renderCorePluginCoverage(payload) {
  const root = qs("#corePluginCoverage");
  if (!root) {
    return;
  }
  if (!payload) {
    root.innerHTML = "<p>Core plugin coverage unavailable yet.</p>";
    return;
  }

  const requiredSummary = payload.summary || {};
  const requiredTotal = Number(requiredSummary.required_total || 0);
  const requiredPresent = Number(requiredSummary.required_present || 0);
  const requiredCovered = Number(requiredSummary.required_covered || 0);
  const requiredMissing = Number(requiredSummary.required_missing || 0);
  const requiredDisabled = Number(requiredSummary.required_disabled || 0);
  const requiredUiMissing = Number(requiredSummary.required_ui_missing || 0);
  const requiredDebugMissing = Number(requiredSummary.required_debug_missing || 0);
  const requiredState = String(payload.required_state || "UNKNOWN");

  const requiredPlugins = Array.isArray(payload.core_plugins) ? payload.core_plugins : [];
  const rows = requiredPlugins
    .map((entry) => {
      const pluginId = escapeHtml(entry.plugin_id || "unknown");
      const matched = escapeHtml(entry.matched_plugin_id || "unregistered");
      const aliases = Array.isArray(entry.aliases) ? entry.aliases.map(escapeHtml).join(", ") : "none";
      const state = String(entry.coverage_state || "unknown");
      const stateText = state === "covered"
        ? "covered"
        : state === "debug_missing"
        ? "debug missing"
        : state === "ui_contract_missing"
        ? "ui contract missing"
        : state === "disabled"
        ? "disabled"
        : "missing";
      const classes = state === "covered"
        ? "status ok"
        : state === "disabled" ? "status" : "status err";

      return `
        <div class="card">
          <h3>${pluginId}</h3>
          <p><strong>Matched plugin:</strong> ${matched}</p>
          <p><strong>Aliases:</strong> ${aliases}</p>
          <p><strong>Coverage:</strong> <span class="${classes}">${stateText}</span></p>
          <p><strong>UI contract:</strong> ${entry.has_ui_contract ? "yes" : "no"}</p>
          <p><strong>Debug action:</strong> ${entry.has_debug_action ? "yes" : "no"}</p>
          <p><strong>Reason:</strong> ${escapeHtml(entry.reason_code || "n/a")}</p>
        </div>
      `;
    })
    .join("");

  const stateLine = requiredState === "READY"
    ? "All required plugin coverage checks passed."
    : requiredState === "BLOCKED"
    ? "Required plugin coverage is incomplete."
    : "Required plugin coverage state is unknown.";
  root.innerHTML = `
    <p><strong>${escapeHtml(stateLine)}</strong></p>
    <p>Required present/expected: ${requiredPresent}/${requiredTotal}</p>
    <p>Coverage complete: ${requiredCovered}/${requiredTotal}</p>
    <p>Missing: ${requiredMissing} • Disabled: ${requiredDisabled} • UI contract missing: ${requiredUiMissing} • Debug missing: ${requiredDebugMissing}</p>
    <div class="card-list">
      ${rows || "<p>Core plugin coverage unavailable.</p>"}
    </div>
  `;
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
    .map((row) => {
      const sessionText = row.chat_session_id ? ` · session ${escapeHtml(row.chat_session_id)}` : "";
      return `<li><strong>${escapeHtml(row.agent_id)}</strong> (${escapeHtml(row.id)}): ${escapeHtml(row.summary)}${sessionText} — <small>${escapeHtml(row.timestamp)}</small></li>`;
    })
    .join("");
}

function renderGmTickets(tickets) {
  const root = qs("#gmTickets");
  if (!root) {
    return;
  }
  if (!Array.isArray(tickets) || tickets.length === 0) {
    root.innerHTML = "<p>No GM tickets to display.</p>";
    return;
  }

  const cards = tickets
    .map((ticket) => {
      const isSelected = state.selectedGmTicketId === ticket.id;
      const status = escapeHtml(ticket.status || "open");
      const priority = escapeHtml(ticket.priority || "normal");
      const assigned = escapeHtml(ticket.assigned_to || "unassigned");
      const lineage = ticket.lineage || {};
      const requestId = escapeHtml((lineage.request_id || ticket.gm_request_id || "n/a"));
      const currentMission = escapeHtml(lineage.current_mission_id || ticket.phase2_mission || ticket.mission_id || "n/a");
      const buildTicketId = escapeHtml(lineage.build_ticket_id || "n/a");
      const buildBatchId = escapeHtml(lineage.build_batch_id || "n/a");
      const lastMissionEvent = lineage.last_mission_event || {};
      const lastMissionEventSummary = escapeHtml(
        `${lastMissionEvent.phase2_mission || "n/a"} ${lastMissionEvent.event || "no event"} ${lastMissionEvent.status || ""}`.trim(),
      );
      const lastMissionEventAt = escapeHtml(lastMissionEvent.at || lastMissionEvent.timestamp || "n/a");
      const createdAt = escapeHtml(ticket.created_at || ticket.createdAt || "n/a");
      const updatedAt = escapeHtml(ticket.updated_at || ticket.updatedAt || "n/a");
      const requestedBy = escapeHtml(ticket.requested_by || "unknown");
      return `
        <div class="card${isSelected ? " selected" : ""}">
          <h3>${escapeHtml(ticket.title || "untitled")}</h3>
          <p><strong>ID:</strong> ${escapeHtml(ticket.id || "unknown")}</p>
          <p><strong>Request ID:</strong> ${requestId}</p>
          <p><strong>Build ticket ID:</strong> ${buildTicketId}</p>
          <p><strong>Build batch ID:</strong> ${buildBatchId}</p>
          <p><strong>Last mission event:</strong> ${lastMissionEventSummary}</p>
          <p><strong>Last mission event time:</strong> ${lastMissionEventAt}</p>
          <p><strong>Status:</strong> ${status}</p>
          <p><strong>Priority:</strong> ${priority}</p>
          <p><strong>Assigned to:</strong> ${assigned}</p>
          <p><strong>Current mission:</strong> ${currentMission}</p>
          <p><strong>Requested by:</strong> ${requestedBy}</p>
          <p><strong>Phase:</strong> ${escapeHtml(ticket.phase || "n/a")}</p>
          <p><strong>Created:</strong> ${createdAt}</p>
          <p><strong>Updated:</strong> ${updatedAt}</p>
          <p>
            <button type="button" data-action="open-gm-ticket" data-id="${escapeHtml(ticket.id || "")}">
              Open
            </button>
          </p>
        </div>
      `;
    })
    .join("");

  root.innerHTML = `<div class="card-list">${cards}</div>`;
}

function _gmTicketStatusOptions(currentStatus) {
  const statusList = GM_TICKET_STATUS_OPTIONS;
  return statusList
    .map(
      (status) => `<option value="${status}"${status === currentStatus ? " selected" : ""}>${status}</option>`,
    )
    .join("");
}

function _gmTicketPriorityOptions(currentPriority) {
  const priorityList = GM_TICKET_PRIORITY_OPTIONS;
  return priorityList
    .map(
      (priority) => `<option value="${priority}"${priority === currentPriority ? " selected" : ""}>${priority}</option>`,
    )
    .join("");
}

function _gmTicketDispatchRoleOptions(currentRole) {
  const roleList = GM_TICKET_DISPATCH_ROLES;
  const selectedRole = (currentRole || "nora").trim();
  return roleList
    .map(
      (role) => `<option value="${role}"${role === selectedRole ? " selected" : ""}>${role}</option>`,
    )
    .join("");
}

function renderGmTicketDetail(ticket) {
  const root = qs("#gmTicketDetail");
  if (!root) {
    return;
  }
  if (!ticket || !ticket.id) {
    root.innerHTML = "<p>Select a ticket from the GM ticket list to inspect details.</p>";
    return;
  }
  const metadata = ticket.metadata || {};
  const metadataText = Object.keys(metadata).length ? escapeHtml(JSON.stringify(metadata)) : "none";
  const lineage = ticket.lineage || {};
  const assigned = escapeHtml(ticket.assigned_to || "");
  const phase = escapeHtml(ticket.phase || "");
  const scope = escapeHtml(ticket.agent_scope || "global");
  const requestedBy = escapeHtml(ticket.requested_by || "unknown");
  const closedAt = ticket.closed_at ? escapeHtml(ticket.closed_at) : "open";
  const buildTicketId = escapeHtml(lineage.build_ticket_id || "n/a");
  const buildBatchId = escapeHtml(lineage.build_batch_id || "n/a");
  const currentMission = escapeHtml(lineage.current_mission_id || "n/a");
  const requestId = escapeHtml(lineage.request_id || ticket.gm_request_id || "n/a");
  const lastMissionEvent = lineage.last_mission_event || {};
  const lastMissionEventSummary = escapeHtml(
    `${lastMissionEvent.phase2_mission || "n/a"} ${lastMissionEvent.event || "no event"} ${lastMissionEvent.status || ""}`.trim(),
  );
  const lastMissionEventAt = escapeHtml(lastMissionEvent.at || lastMissionEvent.timestamp || "n/a");

  root.innerHTML = `
    <div class="card">
      <h3>${escapeHtml(ticket.title || "untitled")}</h3>
      <p><strong>ID:</strong> ${escapeHtml(ticket.id)}</p>
      <p><strong>Request ID:</strong> ${requestId}</p>
      <p><strong>Build ticket ID:</strong> ${buildTicketId}</p>
      <p><strong>Build batch ID:</strong> ${buildBatchId}</p>
      <p><strong>Current mission:</strong> ${currentMission}</p>
      <p><strong>Last mission event:</strong> ${lastMissionEventSummary}</p>
      <p><strong>Last mission event time:</strong> ${lastMissionEventAt}</p>
      <p><strong>Requested by:</strong> ${requestedBy}</p>
      <p><strong>Current status:</strong> ${escapeHtml(ticket.status || "open")}</p>
      <p><strong>Current priority:</strong> ${escapeHtml(ticket.priority || "normal")}</p>
      <p><strong>Current scope:</strong> ${scope}</p>
      <p><strong>Current phase:</strong> ${escapeHtml(ticket.phase || "n/a")}</p>
      <p><strong>Assigned to:</strong> ${assigned || "unassigned"}</p>
      <p><strong>Metadata:</strong> ${metadataText}</p>
      <p><strong>Closed:</strong> ${closedAt}</p>
      <p><strong>Created:</strong> ${escapeHtml(ticket.created_at || ticket.createdAt || "n/a")}</p>
      <p><strong>Updated:</strong> ${escapeHtml(ticket.updated_at || ticket.updatedAt || "n/a")}</p>
      <div class="control-row">
        <label>
          Status
          <select data-action="gm-ticket-status" data-ticket-id="${escapeHtml(ticket.id)}">
            ${_gmTicketStatusOptions(ticket.status || "open")}
          </select>
        </label>
        <label>
          Priority
          <select data-action="gm-ticket-priority" data-ticket-id="${escapeHtml(ticket.id)}">
            ${_gmTicketPriorityOptions(ticket.priority || "normal")}
          </select>
        </label>
        <label>
          Agent scope
          <input type="text" data-action="gm-ticket-scope" data-ticket-id="${escapeHtml(ticket.id)}" value="${scope}" />
        </label>
        <label>
          Phase
          <input type="text" data-action="gm-ticket-phase" data-ticket-id="${escapeHtml(ticket.id)}" value="${phase}" />
        </label>
      </div>
      <div class="control-row">
        <label>
          Assigned to
          <input type="text" data-action="gm-ticket-assigned-to" data-ticket-id="${escapeHtml(ticket.id)}" value="${assigned}" />
        </label>
        <label>
          Dispatch objective
          <input
            type="text"
            data-action="gm-ticket-dispatch-objective"
            data-ticket-id="${escapeHtml(ticket.id)}"
            value="${escapeHtml(ticket.title || "")}"
            placeholder="Optional objective override"
          />
        </label>
        <label>
          Dispatch role
          <select data-action="gm-ticket-dispatch-role" data-ticket-id="${escapeHtml(ticket.id)}">
            ${_gmTicketDispatchRoleOptions(ticket.dispatch_role || "nora")}
          </select>
        </label>
      </div>
      <div class="control-row">
        <button type="button" data-action="dispatch-gm-ticket" data-ticket-id="${escapeHtml(ticket.id)}">
          Dispatch to phase-2 runtime
        </button>
        <button type="button" data-action="save-gm-ticket-update" data-ticket-id="${escapeHtml(ticket.id)}">Save</button>
        ${ticket.status === "closed" ? "" : `<button type="button" data-action="close-gm-ticket" data-ticket-id="${escapeHtml(ticket.id)}">Close now</button>`}
        <button type="button" data-action="export-gm-ticket-audit" data-ticket-id="${escapeHtml(ticket.id)}">Export audit</button>
      </div>
    </div>
  `;
}

function renderGmTicketMessages(messages) {
  const root = qs("#gmTicketMessages");
  if (!root) {
    return;
  }
  if (!Array.isArray(messages) || messages.length === 0) {
    root.innerHTML = "<p>No messages yet.</p>";
    return;
  }

  const cards = messages
    .map((message) => {
      const metadata = message.metadata || {};
      const metadataText = Object.keys(metadata).length ? `<pre>${escapeHtml(JSON.stringify(metadata, null, 2))}</pre>` : "";
      const responseRequiredText = message.response_required
        ? "<p><strong>Response required:</strong> yes</p>"
        : "";
      return `
        <div class="card">
          <h4>${escapeHtml(message.sender || "sender unknown")}</h4>
          <p><strong>Type:</strong> ${escapeHtml(message.message_type || "comment")}</p>
          ${responseRequiredText}
          <p><strong>Message:</strong> ${escapeHtml(message.content || "")}</p>
          <p><strong>At:</strong> ${escapeHtml(message.created_at || "n/a")}</p>
          ${metadataText}
        </div>
      `;
    })
    .join("");

  root.innerHTML = `<div class="card-list">${cards}</div>`;
}

function renderGmTicketAudit(audit) {
  const root = qs("#gmTicketAudit");
  if (!root) {
    return;
  }
  if (!audit || !audit.ticket || !audit.ticket.id) {
    root.innerHTML = "<p>Open a ticket to inspect its audit trail.</p>";
    return;
  }

  const messages = Array.isArray(audit.messages) ? audit.messages : [];
  const events = Array.isArray(audit.events) ? audit.events : [];
  const summary = audit.summary || {};
  const eventRows = events
    .map(
      (event) => {
        const eventPayload = event.payload || {};
        const payloadText = Object.keys(eventPayload).length
          ? `<pre>${escapeHtml(JSON.stringify(eventPayload, null, 2))}</pre>`
          : "";
        return `
          <div class="card">
            <p><strong>${escapeHtml(event.type || "event")}</strong> <span>(${escapeHtml(event.level || "info")})</span></p>
            <p><strong>At:</strong> ${escapeHtml(event.created_at || "n/a")}</p>
            <p>${escapeHtml(event.message || "")}</p>
            ${payloadText}
          </div>
        `;
      },
    )
    .join("");

  const messageRows = messages
    .map(
      (message) => `
        <div class="card">
          <p><strong>${escapeHtml(message.sender || "sender unknown")}</strong> — ${escapeHtml(message.message_type || "comment")}</p>
          ${message.response_required ? "<p><strong>Response required:</strong> yes</p>" : ""}
          <p>${escapeHtml(message.content || "")}</p>
          <p><strong>At:</strong> ${escapeHtml(message.created_at || "n/a")}</p>
        </div>
      `,
    )
    .join("");

  root.innerHTML = `
    <div class="card-list">
      <div class="card">
        <h3>Ticket ${escapeHtml(audit.ticket.id)}</h3>
        <p><strong>Title:</strong> ${escapeHtml(audit.ticket.title || "untitled")}</p>
        <p><strong>Thread messages:</strong> ${escapeHtml(String(summary.message_count || messages.length || 0))}</p>
        <p><strong>Linked events:</strong> ${escapeHtml(String(summary.event_count || events.length || 0))}</p>
      </div>
      <h3>Ticket messages</h3>
      ${messageRows || "<p>No messages yet.</p>"}
      <h3>Audit events</h3>
      ${eventRows || "<p>No audit events yet.</p>"}
    </div>
  `;
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

async function loadAppearance() {
  const payload = await fetchJson("/api/appearance");
  const preferences = normalizeAppearancePreferences(
    payload.appearance_preferences || {},
  );
  applyAppearanceTheme(preferences);
  const overrideInput = qs('input[name="appearance_override_enabled"]');
  const syncInput = qs('input[name="appearance_sync_enabled"]');
  const themeSelect = qs('select[name="appearance_theme_mode"]');
  const contrastSelect = qs('select[name="appearance_contrast_policy"]');
  const motionSelect = qs('select[name="appearance_motion_policy"]');
  const colorSelect = qs('select[name="appearance_color_separation_policy"]');
  const spacingSelect = qs('select[name="appearance_text_spacing_policy"]');
  if (overrideInput) {
    overrideInput.checked = Boolean(preferences.override_enabled);
  }
  if (syncInput) {
    syncInput.checked = Boolean(preferences.sync_theme_preferences);
  }
  if (themeSelect) {
    themeSelect.value = preferences.override_enabled
      ? (preferences.sync_theme_preferences
        ? preferences.shared_theme_mode
        : preferences.desktop_theme_mode)
      : "system";
  }
  if (contrastSelect) {
    contrastSelect.value = preferences.contrast_policy;
  }
  if (motionSelect) {
    motionSelect.value = preferences.motion_policy;
  }
  if (colorSelect) {
    colorSelect.value = preferences.color_separation_policy;
  }
  if (spacingSelect) {
    spacingSelect.value = preferences.text_spacing_policy;
  }
  updateRoleBadge(payload.role, payload.role_source);
  syncAppearanceFormState();
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
  if (state.selectedProviderId && !state.providers.some((provider) => provider && provider.id === state.selectedProviderId)) {
    state.selectedProviderId = "";
    renderCards("#providers", state.providers, buildProviderCard);
  }
  renderDashboardOverview();
}

async function loadPlugins() {
  const payload = await fetchJson("/api/plugins");
  state.plugins = payload.plugins || [];
  renderCards("#plugins", state.plugins, buildPluginCard);
}

async function loadCorePluginCoverage() {
  const payload = await fetchJson("/api/plugins/core");
  state.corePlugins = payload || {};
  renderCorePluginCoverage(payload);
}

async function loadPluginDiagnostics(pluginId) {
  const normalizedPluginId = String(pluginId || "").trim();
  if (!normalizedPluginId) {
    setStatus("#healthState", "plugin diagnostics missing plugin id", "err");
    return;
  }

  const roots = Array.from(document.querySelectorAll("[data-plugin-debug-root='1']"));
  const root = roots.find((entry) => entry.dataset.pluginId === normalizedPluginId);
  if (!root) {
    setStatus("#healthState", `Unable to find diagnostics container for plugin ${normalizedPluginId}`, "err");
    return;
  }

  root.innerHTML = "<p>Loading plugin diagnostics…</p>";
  try {
    const payload = await fetchJson(`/api/plugins/${encodeURIComponent(normalizedPluginId)}/ui/debug`);
    pluginUiConsoleLogger()?.logDiagnostics(normalizedPluginId, payload);
    root.innerHTML = renderPluginDiagnosticsHtml(payload);
    setStatus("#healthState", `plugin diagnostics loaded for ${normalizedPluginId}`, "ok");
  } catch (err) {
    root.innerHTML = `<p>Plugin diagnostics failed: ${escapeHtml(err.message || "unknown error")}</p>`;
    setStatus("#healthState", `plugin diagnostics failed for ${normalizedPluginId}: ${err.message}`, "err");
    pluginUiConsoleLogger()?.logActionRequestFailure(normalizedPluginId, "debug", err);
  }
}

async function loadProviderChain() {
  const payload = await fetchJson("/api/providers/routing-chain");
  state.providerChain = payload;
  renderProviderChain(payload);
  renderDashboardOverview();
}

async function loadProviderDiagnostics(providerId) {
  focusProviderCard(providerId);
  const provider = Array.isArray(state.providers)
    ? state.providers.find((entry) => entry && entry.id === providerId)
    : null;
  const routingNode = Array.isArray(state.providerChain?.chain)
    ? state.providerChain.chain.find((entry) => entry && entry.id === providerId)
    : null;
  const payload = await fetchJson(`/api/providers/${providerId}/metrics`);
  const historyPayload = await fetchJson(`/api/providers/${providerId}/history`);
  const merged = {
    provider_id: providerId,
    provider,
    routingNode,
    status: provider?.status || routingNode?.status || "",
    metadata: provider?.metadata || {},
    metrics: payload.metrics || {},
    last_test: payload.last_test || {},
    test_history: historyPayload.test_history || [],
    last_action_result: state.providerActionResults?.[providerId] || null,
  };
  state.providerDiagnostics = merged;
  const diagnosticsRoot = qs("#providerDiagnostics");
  if (diagnosticsRoot) {
    diagnosticsRoot.innerHTML = renderProviderDiagnosticsHtml(merged);
  }
}

async function loadAgents() {
  const lifecycle = (qs("#agentLifecycleFilter")?.value || "active").trim() || "active";
  const query = new URLSearchParams({ lifecycle });
  const payload = await fetchJson(`/api/agents?${query.toString()}`);
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
  const filters = buildEventFilterState();
  state.eventFilters = filters;
  const params = new URLSearchParams({ limit: String(filters.limit || 50) });
  const payload = await fetchJson(`/api/events?${params.toString()}`);
  state.events = payload.events || [];
  renderEvents(state.events);
}

async function loadRuntime() {
  try {
    const payload = await fetchJson("/api/runtime/status");
    const status = payload.runtime || {};
    const servicesPayload = await fetchJson("/api/runtime/services");
    const runtimeView = runtimeUiApi()?.buildRuntimeViewModel(status, servicesPayload) || {
      summary: {
        statusLine: `runtime (${status.source || "none"}) - ${status.connected ? "connected" : "unavailable"}`,
        metaLine: "",
        statusKind: status.connected ? "ok" : "err",
        defaultService: servicesPayload.default_service || status.default_service || "busy",
      },
      services: servicesPayload.services || [],
    };
    state.runtimeStatus = runtimeView.summary || null;
    state.runtimeServices = runtimeView.services || [];
    setStatus(
      "#runtimeStatus",
      runtimeView.summary?.statusLine || "runtime status unavailable",
      runtimeView.summary?.statusKind || "",
    );
    const runtimeMeta = qs("#runtimeStatusMeta");
    if (runtimeMeta) {
      runtimeMeta.textContent = runtimeView.summary?.metaLine || "";
    }
    renderRuntimeServices(state.runtimeServices);
    renderDashboardOverview();
  } catch (err) {
    setStatus("#runtimeStatus", `runtime unavailable: ${err.message}`, "err");
    const runtimeMeta = qs("#runtimeStatusMeta");
    if (runtimeMeta) {
      runtimeMeta.textContent = "Unable to load runtime services.";
    }
    state.runtimeStatus = {
      connected: false,
      defaultService: "busy",
      error: err.message,
    };
    state.runtimeServices = [];
    renderRuntimeServices([]);
    renderDashboardOverview();
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

async function loadChatHistory(agentId = "", itemId = "", chatSessionId = "") {
  const params = new URLSearchParams();
  if (agentId) {
    params.set("agent_id", agentId);
  }
  if (itemId) {
    params.set("item_id", itemId);
  }
  if (chatSessionId) {
    params.set("chat_session_id", chatSessionId);
  }
  const query = params.toString();
  const payload = await fetchJson(query ? `/api/chat_history?${query}` : "/api/chat_history");
  renderChat(payload.chat_history || []);
  if (itemId) {
    setStatus("#chatStatus", `chat filtered by item ${itemId}`, "ok");
    return;
  }
  if (chatSessionId) {
    setStatus("#chatStatus", `chat filtered by session ${chatSessionId}`, "ok");
  } else {
    setStatus("#chatStatus", "", "");
  }
}

function buildToolUsageFilter() {
  const toolId = qs("#toolUsageToolId")?.value.trim() || "";
  const agentId = qs("#toolUsageAgentId")?.value.trim() || "";
  const contextType = qs("#toolUsageContextType")?.value.trim() || "";
  const contextId = qs("#toolUsageContextId")?.value.trim() || "";
  const memoryId = qs("#toolUsageMemoryId")?.value.trim() || "";
  const chatMessageId = qs("#toolUsageChatMessageId")?.value.trim() || "";
  const chatSessionId = qs("#toolUsageChatSessionId")?.value.trim() || "";
  const missionId = qs("#toolUsageMissionId")?.value.trim() || "";
  const sessionId = qs("#toolUsageSessionId")?.value.trim() || "";
  const date = qs("#toolUsageDate")?.value.trim() || "";
  const dateFrom = qs("#toolUsageDateFrom")?.value.trim() || "";
  const dateTo = qs("#toolUsageDateTo")?.value.trim() || "";
  const limit = Math.min(200, Math.max(1, Number(qs("#toolUsageLimit")?.value || 25)));
  const sortDesc = Boolean(qs("#toolUsageSortDesc")?.checked);
  return {
    toolId,
    agentId,
    contextType,
    contextId,
    memoryId,
    chatMessageId,
    chatSessionId,
    missionId,
    sessionId,
    date,
    dateFrom,
    dateTo,
    limit,
    sortDesc,
  };
}

function buildToolUsageParams(filter, overrides = {}) {
  const params = new URLSearchParams();
  const payload = {
    agent_id: overrides.agentId || filter.agentId,
    mission_id: overrides.missionId || filter.missionId,
    context_type: overrides.contextType || filter.contextType,
    context_id: overrides.contextId || filter.contextId,
    memory_id: overrides.memoryId || filter.memoryId,
    chat_message_id: overrides.chatMessageId || filter.chatMessageId,
    chat_session_id: overrides.chatSessionId || filter.chatSessionId,
    session_id: overrides.sessionId || filter.sessionId,
    date: overrides.date || filter.date,
    date_from: overrides.dateFrom || filter.dateFrom,
    date_to: overrides.dateTo || filter.dateTo,
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
    memory_id: filter.memoryId || null,
    chat_message_id: filter.chatMessageId || null,
    chat_session_id: filter.chatSessionId || null,
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
    const sessionParams = buildToolUsageParams(filter);
    sessionParams.delete("session_id");
    if (filter.contextType) {
      sessionParams.delete("context_type");
    }
    if (filter.contextId) {
      sessionParams.delete("context_id");
    }
    if (filter.memoryId) {
      sessionParams.delete("memory_id");
    }
    if (filter.chatMessageId) {
      sessionParams.delete("chat_message_id");
    }
    if (filter.chatSessionId) {
      sessionParams.delete("chat_session_id");
    }
    if (filter.agentId) {
      sessionParams.delete("agent_id");
    }
    if (filter.missionId) {
      sessionParams.delete("mission_id");
    }
    if (filter.limit != null) {
      sessionParams.delete("limit");
    }
    if (filter.sortDesc != null) {
      sessionParams.delete("sort_desc");
    }
    if (filter.toolId) {
      sessionParams.delete("tool_id");
    }
    const query = sessionParams.toString();
    const payload = await fetchJson(
      `/api/tool-log/session/${encodeURIComponent(targetSession)}${query ? `?${query}` : ""}`,
    );
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

async function loadToolLogByMission(missionId = "", agentId = "") {
  const filter = buildToolUsageFilter();
  const targetMission = missionId || filter.missionId;
  if (!targetMission) {
    setStatus("#toolUsageStatus", "mission id is required", "err");
    return;
  }
  try {
    const missionParams = buildToolUsageParams(filter, {
      missionId: targetMission,
      agentId: agentId || filter.agentId,
      contextType: filter.contextType,
      contextId: filter.contextId,
      memoryId: filter.memoryId,
      chatMessageId: filter.chatMessageId,
      chatSessionId: filter.chatSessionId,
      sessionId: filter.sessionId,
      date: filter.date,
      dateFrom: filter.dateFrom,
      dateTo: filter.dateTo,
      sortDesc: filter.sortDesc,
    });
    const query = missionParams.toString();
    const payload = await fetchJson(`/api/tool-log${query ? `?${query}` : ""}`);
    state.toolUsage = payload.usage || [];
    state.toolUsageCount = Number(payload.count || state.toolUsage.length || 0);
    renderToolUsage(state.toolUsage);
    const missionSummary = `Mission ${escapeHtml(targetMission)} tool log`;
    const agentText = agentId || filter.agentId ? ` (agent ${escapeHtml(agentId || filter.agentId)})` : "";
    setStatus("#toolUsageSummary", `${missionSummary}${agentText} · ${state.toolUsageCount} row(s)`, "ok");
    setStatus("#toolUsageStatus", `loaded ${state.toolUsage.length} row(s)`, "ok");
    if (payload.log_entry) {
      renderToolUsageLogEntry(payload.log_entry);
    }
  } catch (err) {
    setStatus("#toolUsageStatus", `mission log load failed: ${err.message}`, "err");
    state.toolUsage = [];
    state.toolUsageCount = 0;
    renderToolUsage([]);
    renderToolUsageLogEntry(null);
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
      memoryId: filter.memoryId,
      chatMessageId: filter.chatMessageId,
      chatSessionId: filter.chatSessionId,
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
      memory_id: payload.memory_id || filter.memoryId || null,
      chat_message_id: payload.chat_message_id || filter.chatMessageId || null,
      chat_session_id: payload.chat_session_id || filter.chatSessionId || null,
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

async function loadAgentToolAudit(agentId) {
  if (!agentId) {
    setStatus("#toolUsageStatus", "agent id is required", "err");
    return;
  }
  const filter = buildToolUsageFilter();
  try {
    const params = new URLSearchParams();
    if (filter.missionId) {
      params.set("mission_id", filter.missionId);
    }
    if (filter.contextType) {
      params.set("context_type", filter.contextType);
    }
    if (filter.contextId) {
      params.set("context_id", filter.contextId);
    }
    if (filter.memoryId) {
      params.set("memory_id", filter.memoryId);
    }
    if (filter.chatMessageId) {
      params.set("chat_message_id", filter.chatMessageId);
    }
    if (filter.chatSessionId) {
      params.set("chat_session_id", filter.chatSessionId);
    }
    if (filter.sessionId) {
      params.set("session_id", filter.sessionId);
    }
    params.set("recent_limit", String(Math.min(200, Math.max(1, Number(filter.limit || 25)))));
    params.set("tool_limit", "10");
    params.set("session_limit", "10");

    const query = params.toString();
    const payload = await fetchJson(`/api/agents/${encodeURIComponent(agentId)}/audit${query ? `?${query}` : ""}`);

    state.selectedAgentAuditId = agentId;
    state.agentToolAudit = payload;
    renderAgentToolAudit(payload);

    const recentCalls = Array.isArray(payload.recent_calls) ? payload.recent_calls : [];
    state.toolUsage = recentCalls;
    state.toolUsageCount = Number((payload.summary || {}).total_tool_calls || 0);
    renderToolUsage(recentCalls);
    setStatus("#toolUsageSummary", buildToolUsageSummary(state.toolUsageCount, {
      tool_id: "aggregate",
      agent_id: agentId,
      context_type: filter.contextType || null,
      context_id: filter.contextId || null,
      mission_id: filter.missionId || null,
      session_id: filter.sessionId || null,
      mode: "agent audit",
    }), "ok");
    setStatus("#toolUsageStatus", `loaded audit for ${agentId}`, "ok");
    setStatus("#toolUsageLogEntry", "");
    const targetAgentInput = qs("#toolUsageAgentId");
    if (targetAgentInput) {
      targetAgentInput.value = agentId;
    }
  } catch (err) {
    setStatus("#toolUsageStatus", `agent tool audit failed: ${err.message}`, "err");
    state.toolUsage = [];
    state.toolUsageCount = 0;
    state.selectedAgentAuditId = "";
    state.agentToolAudit = null;
    renderToolUsage([]);
    renderAgentToolAudit(null);
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
  const contextType = (entry.context_type || "n/a");
  const contextId = entry.context_id || "";
  const memoryId = entry.memory_id || "";
  const chatMessageId = entry.chat_message_id || "";
  const chatSessionId = entry.chat_session_id || "";
  const missionId = entry.mission_id || "";
  const detailContextButtons = [];
  if (memoryId) {
    detailContextButtons.push(`
      <button
        type="button"
        data-action="open-tool-context"
        data-context-type="memory"
        data-context-id="${escapeHtml(contextId)}"
        data-memory-id="${escapeHtml(memoryId)}"
        data-chat-message-id=""
        data-chat-session-id=""
      >
        Open linked memory
      </button>
    `);
  }
  if (chatSessionId) {
    detailContextButtons.push(`
      <button
        type="button"
        data-action="open-tool-context"
        data-context-type="chat_session"
        data-context-id=""
        data-memory-id=""
        data-chat-message-id=""
        data-chat-session-id="${escapeHtml(chatSessionId)}"
      >
        Open chat session
      </button>
    `);
  }
  if (contextType === "chat" && (chatMessageId || chatSessionId || contextId)) {
    detailContextButtons.push(`
      <button
        type="button"
        data-action="open-tool-context"
        data-context-type="chat"
        data-context-id="${escapeHtml(contextId)}"
        data-memory-id=""
        data-chat-message-id="${escapeHtml(chatMessageId)}"
        data-chat-session-id="${escapeHtml(chatSessionId)}"
      >
        Open linked chat context
      </button>
    `);
  }
  if (missionId) {
    detailContextButtons.push(`
      <button
        type="button"
        data-action="open-tool-mission"
        data-mission-id="${escapeHtml(missionId)}"
        data-agent-id="${escapeHtml(entry.agent_id || "")}"
      >
        Open mission trail
      </button>
    `);
  }
  if (!memoryId && !chatSessionId && !chatMessageId && contextType !== "n/a" && contextId) {
    detailContextButtons.push(`
      <button
        type="button"
        data-action="open-tool-context"
        data-context-type="${escapeHtml(contextType)}"
        data-context-id="${escapeHtml(contextId)}"
        data-memory-id=""
        data-chat-message-id=""
        data-chat-session-id=""
      >
        Open context
      </button>
    `);
  }
  root.innerHTML = `
    <div class="card">
      <h3>Tool call ${escapeHtml(entry.id || "n/a")}</h3>
      <p><strong>Tool:</strong> ${escapeHtml(entry.tool_id || "n/a")}</p>
      <p><strong>Agent:</strong> ${escapeHtml(entry.agent_id || "n/a")}</p>
      <p><strong>Session:</strong> ${escapeHtml(entry.session_id || "n/a")}</p>
      <p><strong>Mission:</strong> ${escapeHtml(entry.mission_id || "n/a")}</p>
      <p><strong>Memory ID:</strong> ${escapeHtml(entry.memory_id || "n/a")}</p>
      <p><strong>Chat message ID:</strong> ${escapeHtml(entry.chat_message_id || "n/a")}</p>
      <p><strong>Chat session ID:</strong> ${escapeHtml(entry.chat_session_id || "n/a")}</p>
      <p><strong>Status:</strong> ${escapeHtml(entry.status || "n/a")}</p>
      <p><strong>Result:</strong> ${escapeHtml(entry.result_status || "n/a")}</p>
      <p><strong>Created:</strong> ${escapeHtml(entry.created_at || "n/a")}</p>
      ${detailContextButtons.length ? `<p>${detailContextButtons.join("")}</p>` : ""}
      ${payload}
      ${metadata}
    </div>
  `;
}

async function openToolUsageContext(contextType, contextId, memoryId = "", chatMessageId = "", chatSessionId = "") {
  const resolvedMemoryId = memoryId || "";
  const resolvedChatMessageId = chatMessageId || "";
  const resolvedChatSessionId = chatSessionId || "";
  const fallbackContextId = contextId || "";

  if (!contextType && !resolvedMemoryId && !resolvedChatMessageId && !resolvedChatSessionId && !fallbackContextId) {
    setStatus("#toolUsageStatus", "context id is required for drill-through", "err");
    return;
  }
  try {
    if (contextType === "memory") {
      const targetMemoryId = resolvedMemoryId || fallbackContextId;
      if (targetMemoryId) {
        await loadMemory("", "", targetMemoryId);
        return;
      }
      setStatus("#toolUsageStatus", "memory id is required for memory context drill-through", "err");
      return;
    }
    if (contextType === "chat_session") {
      if (resolvedChatSessionId) {
        await loadChatHistory("", "", resolvedChatSessionId);
        return;
      }
      setStatus("#toolUsageStatus", "chat session id is required for chat context drill-through", "err");
      return;
    }
    if (contextType === "chat") {
      if (resolvedChatSessionId) {
        await loadChatHistory("", "", resolvedChatSessionId);
        return;
      }
      const targetChatMessageId = resolvedChatMessageId || fallbackContextId;
      if (targetChatMessageId) {
        await loadChatHistory("", targetChatMessageId);
        return;
      }
      setStatus("#toolUsageStatus", "chat message id is required for chat context drill-through", "err");
      return;
    }
    if (resolvedMemoryId) {
      await loadMemory("", "", resolvedMemoryId);
      return;
    }
    if (resolvedChatMessageId) {
      await loadChatHistory("", resolvedChatMessageId);
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

function _overlayHistoryLimit() {
  const requested = Number(qs("#agentOverlayHistoryLimit")?.value || 20);
  const sanitized = Number.isFinite(requested) ? requested : 20;
  return Math.min(200, Math.max(1, Math.floor(sanitized)));
}

async function loadAgentOverlayHistory(agentId) {
  const targetAgentId = String(agentId || "").trim();
  if (!targetAgentId) {
    setStatus("#agentOverlayHistoryStatus", "agent id is required", "err");
    return;
  }
  const limit = _overlayHistoryLimit();
  try {
    const payload = await fetchJson(
      `/api/agents/${encodeURIComponent(targetAgentId)}/overlay/history?${new URLSearchParams({
        limit: String(limit),
      }).toString()}`,
    );
    state.selectedAgentOverlayHistoryId = targetAgentId;
    state.agentOverlayHistory = payload.history || [];
    state.agentOverlayHistoryCount = Number(payload.count || 0);
    renderAgentOverlayHistory(payload);
    const label = qs("#agentOverlayHistoryHeader");
    if (label) {
      label.textContent = `Overlay history for ${targetAgentId} (last ${Math.min(limit, state.agentOverlayHistoryCount)} entries)`;
    }
    setStatus("#agentOverlayHistoryStatus", `loaded ${state.agentOverlayHistory.length} entry(ies)`, "ok");
  } catch (err) {
    state.selectedAgentOverlayHistoryId = targetAgentId;
    state.agentOverlayHistory = [];
    state.agentOverlayHistoryCount = 0;
    renderAgentOverlayHistory(null);
    setStatus("#agentOverlayHistoryStatus", `load overlay history failed: ${err.message}`, "err");
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
      const memoryId = entry.memory_id || "";
      const chatMessageId = entry.chat_message_id || "";
      const chatSessionId = entry.chat_session_id || "";
      const missionId = entry.mission_id || "";
      const sessionAction = entry.session_id ? `open-tool-session` : "";
      const missionAction = missionId ? `open-tool-mission` : "";
      const sessionButton = sessionAction
        ? `<button
              type="button"
              data-action="${sessionAction}"
              data-session-id="${escapeHtml(entry.session_id)}"
            >
              Open session
            </button>`
        : "";
  const missionButton = missionAction
    ? `<button
              type="button"
              data-action="${missionAction}"
              data-mission-id="${escapeHtml(missionId)}"
              data-agent-id="${escapeHtml(entry.agent_id || "")}"
            >
              Open mission trail
            </button>`
        : "";
      const contextButtons = [];
      if (memoryId) {
        contextButtons.push(`
          <button
            type="button"
            data-action="open-tool-context"
            data-context-type="memory"
            data-context-id="${escapeHtml(contextId)}"
            data-memory-id="${escapeHtml(memoryId)}"
            data-chat-message-id=""
            data-chat-session-id=""
          >
            Open linked memory
          </button>
        `);
      }
      if (chatSessionId) {
        contextButtons.push(`
          <button
            type="button"
            data-action="open-tool-context"
            data-context-type="chat_session"
            data-context-id=""
            data-memory-id=""
            data-chat-message-id=""
            data-chat-session-id="${escapeHtml(chatSessionId)}"
          >
            Open chat session
          </button>
        `);
      }
      if (contextType === "chat" && (chatMessageId || contextId) && !chatSessionId) {
        contextButtons.push(`
          <button
            type="button"
            data-action="open-tool-context"
            data-context-type="chat"
            data-context-id="${escapeHtml(contextId)}"
            data-memory-id=""
            data-chat-message-id="${escapeHtml(chatMessageId)}"
            data-chat-session-id="${escapeHtml(chatSessionId)}"
          >
            Open chat context
          </button>
        `);
      }
      if (
        contextType !== "n/a" &&
        contextType !== "memory" &&
        contextType !== "chat" &&
        contextType !== "chat_session" &&
        contextId &&
        !memoryId &&
        !chatSessionId
      ) {
        contextButtons.push(`
          <button
            type="button"
            data-action="open-tool-context"
            data-context-type="${escapeHtml(contextType)}"
            data-context-id="${escapeHtml(contextId)}"
            data-memory-id=""
            data-chat-message-id=""
            data-chat-session-id=""
          >
            Open context
          </button>
        `);
      }
      const contextButton = contextButtons.join("");
      const resolvedContextButton = contextButton;
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
          ${missionId ? `<p><strong>Mission:</strong> ${escapeHtml(missionId)}</p>` : ""}
          <p><strong>Context:</strong> ${escapeHtml(contextType)} ${contextId ? `( ${escapeHtml(contextId)})` : ""}</p>
          ${memoryId ? `<p><strong>Memory ID:</strong> ${escapeHtml(memoryId)}</p>` : ""}
          ${chatMessageId ? `<p><strong>Chat message:</strong> ${escapeHtml(chatMessageId)}</p>` : ""}
          ${chatSessionId ? `<p><strong>Chat session:</strong> ${escapeHtml(chatSessionId)}</p>` : ""}
          <p><strong>Status:</strong> ${escapeHtml(entry.status || "n/a")} · ${detail}duration ${
            typeof entry.duration_ms === "number" ? `${entry.duration_ms}ms` : "n/a"
          }</p>
          <p><strong>Created:</strong> ${escapeHtml(entry.created_at || "n/a")}</p>
          ${details}
          ${resolvedContextButton}
          ${sessionButton}
          ${missionButton}
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
  const memoryText = toolUsageFilter.memory_id || "all memory ids";
  const chatMessageText = toolUsageFilter.chat_message_id || "all chat messages";
  const chatSessionText = toolUsageFilter.chat_session_id || "all chat sessions";
  const sessionText = toolUsageFilter.session_id || "all sessions";
  const dateText = toolUsageFilter.date
    ? `date=${toolUsageFilter.date}`
    : toolUsageFilter.date_from || toolUsageFilter.date_to
      ? `date range=${toolUsageFilter.date_from || "unbounded"}..${toolUsageFilter.date_to || "unbounded"}`
      : "all dates";
  const modeText = toolUsageFilter.mode ? `${toolUsageFilter.mode} ` : "";
  return `Showing ${total} ${modeText}rows for ${toolText}; ${agentText} · ${contextText} · ${contextIdText} · ${memoryText} · ${chatMessageText} · ${chatSessionText} · ${missionText} · ${sessionText} · ${dateText}`;
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

function _readGmTicketFilters() {
  return {
    status: (qs("#gmTicketFilterStatus")?.value || "").trim(),
    priority: (qs("#gmTicketFilterPriority")?.value || "").trim(),
    assignedTo: (qs("#gmTicketFilterAssignedTo")?.value || "").trim(),
    phase: (qs("#gmTicketFilterPhase")?.value || "").trim(),
    limit: Math.min(200, Math.max(1, Number(qs("#gmTicketFilterLimit")?.value || 20))),
    offset: Math.max(0, Number(qs("#gmTicketFilterOffset")?.value || 0)),
  };
}

function _gmTicketPayloadFromList(filters) {
  const params = new URLSearchParams();
  if (filters.status) {
    params.set("status", filters.status);
  }
  if (filters.priority) {
    params.set("priority", filters.priority);
  }
  if (filters.assignedTo) {
    params.set("assigned_to", filters.assignedTo);
  }
  if (filters.phase) {
    params.set("phase", filters.phase);
  }
  if (filters.limit != null) {
    params.set("limit", String(filters.limit));
  }
  if (filters.offset != null) {
    params.set("offset", String(filters.offset));
  }
  return params;
}

async function loadGmTickets() {
  const filters = _readGmTicketFilters();
  const params = _gmTicketPayloadFromList(filters);
  const query = params.toString();
  setStatus("#gmTicketStatus", "loading GM tickets...", "");
  try {
    const payload = await fetchJson(`/api/gm-tickets${query ? `?${query}` : ""}`);
    state.gmTickets = payload.tickets || payload.items || [];
    if (state.selectedGmTicketId) {
      const selectedTicketStillPresent = state.gmTickets.some((ticket) => ticket.id === state.selectedGmTicketId);
      if (!selectedTicketStillPresent) {
        state.selectedGmTicketId = "";
        state.selectedGmTicket = null;
        state.gmTicketMessages = [];
        state.gmTicketAudit = null;
        renderGmTicketDetail(null);
        renderGmTicketMessages([]);
        renderGmTicketAudit(null);
      }
    }
    renderGmTickets(state.gmTickets);
    setStatus("#gmTicketStatus", `loaded ${state.gmTickets.length} GM ticket(s)`, "ok");
    renderDashboardOverview();
  } catch (err) {
    state.gmTickets = [];
    renderGmTickets([]);
    setStatus("#gmTicketStatus", `failed to load GM tickets: ${err.message}`, "err");
    renderDashboardOverview();
  }
}

async function loadGmTicket(ticketId) {
  if (!ticketId) {
    setStatus("#gmTicketDetailStatus", "ticket id required", "err");
    return;
  }
  setStatus("#gmTicketDetailStatus", "loading ticket details...", "");
  setStatus("#gmTicketAuditStatus", "loading audit trail...", "");
  try {
    const payload = await fetchJson(`/api/gm-tickets/${encodeURIComponent(ticketId)}`);
    const ticket = payload.ticket || payload;
    state.selectedGmTicketId = ticket.id || ticketId;
    state.selectedGmTicket = ticket;
    renderGmTicketDetail(ticket);
    await loadGmTicketMessages(ticket.id || ticketId);
    await loadGmTicketAudit(ticket.id || ticketId);
    setStatus("#gmTicketDetailStatus", "ticket loaded", "ok");
    setStatus("#gmTicketAuditStatus", "audit trail loaded", "ok");
  } catch (err) {
    state.selectedGmTicket = null;
    renderGmTicketDetail(null);
    renderGmTicketMessages([]);
    renderGmTicketAudit(null);
    state.gmTicketAudit = null;
    setStatus("#gmTicketAuditStatus", `failed to load ticket audit: ${err.message}`, "err");
    setStatus("#gmTicketDetailStatus", `failed to load ticket: ${err.message}`, "err");
  }
}

async function loadGmTicketMessages(ticketId) {
  if (!ticketId) {
    state.gmTicketMessages = [];
    renderGmTicketMessages([]);
    return;
  }
  try {
    const payload = await fetchJson(`/api/gm-tickets/${encodeURIComponent(ticketId)}/messages`);
    state.gmTicketMessages = payload.messages || [];
    state.gmTicketMessageCount = Number(payload.count || state.gmTicketMessages.length || 0);
    renderGmTicketMessages(state.gmTicketMessages);
  } catch (err) {
    state.gmTicketMessages = [];
    state.gmTicketMessageCount = 0;
    renderGmTicketMessages([]);
    setStatus("#gmTicketMessageStatus", `failed to load messages: ${err.message}`, "err");
  }
}

async function loadGmTicketAudit(ticketId) {
  if (!ticketId) {
    state.gmTicketAudit = null;
    renderGmTicketAudit(null);
    return;
  }
  try {
    const payload = await fetchJson(`/api/gm-tickets/${encodeURIComponent(ticketId)}/audit`);
    state.gmTicketAudit = payload;
    renderGmTicketAudit(payload);
    setStatus("#gmTicketAuditStatus", `loaded ${payload.summary?.event_count || 0} events`, "ok");
  } catch (err) {
    state.gmTicketAudit = null;
    renderGmTicketAudit(null);
    setStatus("#gmTicketAuditStatus", `failed to load ticket audit: ${err.message}`, "err");
  }
}

async function exportGmTicketAudit(ticketId) {
  const targetId = ticketId || state.selectedGmTicketId;
  if (!targetId) {
    setStatus("#gmTicketAuditStatus", "select a GM ticket before exporting audit", "err");
    return;
  }
  try {
    const payload = await fetchJson(`/api/gm-tickets/${encodeURIComponent(targetId)}/audit/export`);
    const filename = `gm-ticket-${targetId}-audit-${Date.now()}.json`;
    downloadTextFile(filename, payload);
    setStatus("#gmTicketAuditStatus", `exported audit for ${targetId}`, "ok");
  } catch (err) {
    setStatus("#gmTicketAuditStatus", `failed to export ticket audit: ${err.message}`, "err");
  }
}

function _buildGmTicketUpdatePayload() {
  const ticketId = state.selectedGmTicketId;
  if (!ticketId) {
    return { ticketId: "", payload: null };
  }
  const ticket = state.selectedGmTicket || {};
  const root = qs("#gmTicketDetail");
  if (!root) {
    return { ticketId, payload: null };
  }
  const statusNode = root.querySelector(`select[data-action="gm-ticket-status"][data-ticket-id="${ticketId}"]`);
  const priorityNode = root.querySelector(`select[data-action="gm-ticket-priority"][data-ticket-id="${ticketId}"]`);
  const scopeNode = root.querySelector(`input[data-action="gm-ticket-scope"][data-ticket-id="${ticketId}"]`);
  const phaseNode = root.querySelector(`input[data-action="gm-ticket-phase"][data-ticket-id="${ticketId}"]`);
  const assignedNode = root.querySelector(`input[data-action="gm-ticket-assigned-to"][data-ticket-id="${ticketId}"]`);

  const next = {
    status: statusNode?.value?.trim() || ticket.status || "open",
    priority: priorityNode?.value?.trim() || ticket.priority || "normal",
    agent_scope: scopeNode?.value?.trim() || ticket.agent_scope || "global",
    phase: phaseNode?.value?.trim() || ticket.phase || null,
    assigned_to: assignedNode?.value?.trim() || null,
  };

  return { ticketId, payload: next };
}

async function submitGmTicket(event) {
  event.preventDefault();
  const title = qs('input[name="gmTicketTitle"]')?.value.trim() || "";
  const requestedBy = qs('input[name="gmTicketRequestedBy"]')?.value.trim() || "";
  const status = qs('select[name="gmTicketStatus"]')?.value.trim() || undefined;
  const priority = qs('select[name="gmTicketPriority"]')?.value.trim() || undefined;
  const agentScope = qs('input[name="gmTicketAgentScope"]')?.value.trim() || undefined;
  const phase = qs('input[name="gmTicketPhase"]')?.value.trim() || undefined;
  const assignedTo = qs('input[name="gmTicketAssignedTo"]')?.value.trim() || undefined;
  const metadataRaw = qs('textarea[name="gmTicketMetadata"]')?.value.trim() || "";

  if (!title || !requestedBy) {
    setStatus("#gmTicketCreateStatus", "title and requested by are required", "err");
    return;
  }

  let metadata = undefined;
  if (metadataRaw) {
    try {
      metadata = JSON.parse(metadataRaw);
    } catch (err) {
      setStatus("#gmTicketCreateStatus", "metadata must be valid JSON if provided", "err");
      return;
    }
  }

  const payload = {
    title,
    requested_by: requestedBy,
  };
  if (status) {
    payload.status = status;
  }
  if (priority) {
    payload.priority = priority;
  }
  if (agentScope) {
    payload.agent_scope = agentScope;
  }
  if (phase) {
    payload.phase = phase;
  }
  if (assignedTo) {
    payload.assigned_to = assignedTo;
  }
  if (metadata != null) {
    payload.metadata = metadata;
  }

  setStatus("#gmTicketCreateStatus", "creating GM ticket...", "");
  try {
    const created = await postJson("/api/gm-tickets", payload);
    setStatus("#gmTicketCreateStatus", `created ${created.ticket_id || created.id || title}`, "ok");
    if (qs('input[name="gmTicketTitle"]')) qs('input[name="gmTicketTitle"]').value = "";
    if (qs('input[name="gmTicketRequestedBy"]')) qs('input[name="gmTicketRequestedBy"]').value = "";
    if (qs('select[name="gmTicketStatus"]')) qs('select[name="gmTicketStatus"]').value = "";
    if (qs('select[name="gmTicketPriority"]')) qs('select[name="gmTicketPriority"]').value = "normal";
    if (qs('input[name="gmTicketAgentScope"]')) qs('input[name="gmTicketAgentScope"]').value = "";
    if (qs('input[name="gmTicketPhase"]')) qs('input[name="gmTicketPhase"]').value = "";
    if (qs('input[name="gmTicketAssignedTo"]')) qs('input[name="gmTicketAssignedTo"]').value = "";
    if (qs('textarea[name="gmTicketMetadata"]')) qs('textarea[name="gmTicketMetadata"]').value = "";

    await loadGmTickets();
    const createdId = created.id || created.ticket_id || created.ticket?.id;
    if (createdId) {
      await loadGmTicket(createdId);
    }
  } catch (err) {
    setStatus("#gmTicketCreateStatus", `create failed: ${err.message}`, "err");
  }
}

async function submitGmDirectMessage(event) {
  event.preventDefault();
  const sender = qs('input[name="gmDirectSender"]')?.value.trim() || "";
  const messageType = qs('select[name="gmDirectMessageType"]')?.value.trim() || "comment";
  const ticketId = qs('input[name="gmDirectTicketId"]')?.value.trim() || "";
  const title = qs('input[name="gmDirectTitle"]')?.value.trim() || "";
  const content = qs('textarea[name="gmDirectMessageContent"]')?.value.trim() || "";
  const metadataRaw = qs('textarea[name="gmDirectMessageMetadata"]')?.value.trim() || "";

  if (!sender || !content) {
    setStatus("#gmDirectMessageStatus", "sender and message content are required", "err");
    return;
  }

  let metadata = undefined;
  if (metadataRaw) {
    try {
      metadata = JSON.parse(metadataRaw);
    } catch (err) {
      setStatus("#gmDirectMessageStatus", "metadata must be valid JSON if provided", "err");
      return;
    }
  }

  const payload = { sender, content, message_type: messageType };
  if (ticketId) {
    payload.ticket_id = ticketId;
  }
  if (title) {
    payload.title = title;
  }
  if (metadata != null) {
    payload.metadata = metadata;
  }

  setStatus("#gmDirectMessageStatus", "sending direct message...", "");
  try {
    const created = await postJson("/api/gm/message", payload);
    setStatus("#gmDirectMessageStatus", "direct message sent", "ok");
    if (qs('textarea[name="gmDirectMessageContent"]')) {
      qs('textarea[name="gmDirectMessageContent"]').value = "";
    }
    if (qs('textarea[name="gmDirectMessageMetadata"]')) {
      qs('textarea[name="gmDirectMessageMetadata"]').value = "";
    }
    if (qs('input[name="gmDirectTitle"]')) {
      qs('input[name="gmDirectTitle"]').value = "";
    }
    await loadGmTickets();
    const targetTicketId = ticketId || created.ticket?.id || created.ticket?.ticket_id;
    if (targetTicketId) {
      await loadGmTicket(targetTicketId);
    } else if (created.created_ticket) {
      setStatus("#gmDirectMessageStatus", "message sent, open GM tickets to review", "ok");
    }
  } catch (err) {
    setStatus("#gmDirectMessageStatus", `direct message failed: ${err.message}`, "err");
  }
}

async function saveGmTicketUpdate(event) {
  if (event) {
    event.preventDefault();
  }
  const { ticketId, payload } = _buildGmTicketUpdatePayload();
  if (!ticketId || !payload) {
    setStatus("#gmTicketDetailStatus", "select a ticket before saving", "err");
    return;
  }

  setStatus("#gmTicketDetailStatus", "saving ticket update...", "");
  try {
    await patchJson(`/api/gm-tickets/${encodeURIComponent(ticketId)}`, payload);
    setStatus("#gmTicketDetailStatus", "ticket updated", "ok");
    await loadGmTickets();
    await loadGmTicket(ticketId);
  } catch (err) {
    setStatus("#gmTicketDetailStatus", `update failed: ${err.message}`, "err");
  }
}

async function closeGmTicket(ticketId = "") {
  const target = String(ticketId || state.selectedGmTicketId || "").trim();
  if (!target) {
    setStatus("#gmTicketDetailStatus", "select a ticket before closing", "err");
    return;
  }
  setStatus("#gmTicketDetailStatus", "closing ticket...", "");
  try {
    await patchJson(`/api/gm-tickets/${encodeURIComponent(target)}`, {
      status: "closed",
      closed_at: new Date().toISOString(),
    });
    setStatus("#gmTicketDetailStatus", "ticket closed", "ok");
    await loadGmTickets();
    await loadGmTicket(target);
  } catch (err) {
    setStatus("#gmTicketDetailStatus", `close failed: ${err.message}`, "err");
  }
}

async function dispatchGmTicket(ticketId = "") {
  const target = String(ticketId || state.selectedGmTicketId || "").trim();
  if (!target) {
    setStatus("#gmTicketDetailStatus", "select a ticket before dispatch", "err");
    return;
  }

  const root = qs("#gmTicketDetail");
  const objectiveNode = root?.querySelector(
    `input[data-action="gm-ticket-dispatch-objective"][data-ticket-id="${escapeHtml(target)}"]`,
  );
  const roleNode = root?.querySelector(
    `select[data-action="gm-ticket-dispatch-role"][data-ticket-id="${escapeHtml(target)}"]`,
  );
  const objective = objectiveNode?.value?.trim() || "";
  const role = roleNode?.value?.trim() || "";
  const payload = {};
  if (objective) {
    payload.objective = objective;
  }
  if (role) {
    payload.role = role;
  }

  setStatus("#gmTicketDetailStatus", "dispatching ticket to runtime...", "");
  try {
    await postJson(`/api/gm-tickets/${encodeURIComponent(target)}/dispatch`, payload);
    setStatus("#gmTicketDetailStatus", "ticket dispatched", "ok");
    await loadGmTicket(target);
    await loadGmTickets();
    await loadGmTicketMessages(target);
  } catch (err) {
    setStatus("#gmTicketDetailStatus", `dispatch failed: ${err.message}`, "err");
  }
}

async function submitGmTicketMessage(event) {
  event.preventDefault();
  const ticketId = state.selectedGmTicketId;
  if (!ticketId) {
    setStatus("#gmTicketMessageStatus", "select a ticket first", "err");
    return;
  }
  const sender = qs('input[name="gmTicketMessageSender"]')?.value.trim() || "";
  const messageType = qs('select[name="gmTicketMessageType"]')?.value.trim() || GM_TICKET_MESSAGE_TYPES[0];
  const content = qs('textarea[name="gmTicketMessageContent"]')?.value.trim() || "";
  const responseRequired = qs('input[name="gmTicketMessageResponseRequired"]')?.checked === true;
  const metadataRaw = qs('textarea[name="gmTicketMessageMetadata"]')?.value.trim() || "";

  if (!sender || !content) {
    setStatus("#gmTicketMessageStatus", "sender and message content are required", "err");
    return;
  }

  let metadata = undefined;
  if (metadataRaw) {
    try {
      metadata = JSON.parse(metadataRaw);
    } catch (err) {
      setStatus("#gmTicketMessageStatus", "message metadata must be valid JSON if provided", "err");
      return;
    }
  }

  const payload = {
    sender,
    content,
    message_type: messageType,
    response_required: responseRequired,
  };
  if (metadata != null) {
    payload.metadata = metadata;
  }

  setStatus("#gmTicketMessageStatus", "posting message...", "");
  try {
    await postJson(`/api/gm-tickets/${encodeURIComponent(ticketId)}/messages`, payload);
    setStatus("#gmTicketMessageStatus", "message posted", "ok");
    const contentInput = qs('textarea[name="gmTicketMessageContent"]');
    if (contentInput) {
      contentInput.value = "";
    }
  const responseRequiredInput = qs('input[name="gmTicketMessageResponseRequired"]');
  if (responseRequiredInput) {
    responseRequiredInput.checked = false;
  }
  const metadataInput = qs('textarea[name="gmTicketMessageMetadata"]');
  if (metadataInput) {
    metadataInput.value = "";
  }
    await loadGmTicketMessages(ticketId);
    await loadGmTicket(ticketId);
  } catch (err) {
    setStatus("#gmTicketMessageStatus", `message failed: ${err.message}`, "err");
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

async function saveAppearance(event) {
  event.preventDefault();
  if (state.role !== "admin") {
    setStatus("#appearanceStatus", "admin token required", "err");
    return;
  }
  const overrideEnabled = Boolean(
    qs('input[name="appearance_override_enabled"]')?.checked,
  );
  const syncEnabled = Boolean(
    qs('input[name="appearance_sync_enabled"]')?.checked,
  );
  const selectedThemeMode =
    qs('select[name="appearance_theme_mode"]')?.value || "system";
  const contrastPolicy =
    qs('select[name="appearance_contrast_policy"]')?.value || "aa";
  const motionPolicy =
    qs('select[name="appearance_motion_policy"]')?.value || "default";
  const colorSeparationPolicy =
    qs('select[name="appearance_color_separation_policy"]')?.value || "default";
  const textSpacingPolicy =
    qs('select[name="appearance_text_spacing_policy"]')?.value || "default";

  let payload = {
    contrast_policy: contrastPolicy,
    motion_policy: motionPolicy,
    color_separation_policy: colorSeparationPolicy,
    text_spacing_policy: textSpacingPolicy,
  };
  if (!overrideEnabled) {
    payload.override_enabled = false;
  } else if (syncEnabled) {
    payload = {
      ...payload,
      override_enabled: true,
      sync_theme_preferences: true,
      shared_theme_mode: selectedThemeMode,
    };
  } else {
    payload = {
      ...payload,
      override_enabled: true,
      sync_theme_preferences: false,
      desktop_theme_mode: selectedThemeMode,
    };
  }

  setStatus("#appearanceStatus", "saving", "");
  try {
    const response = await patchJson("/api/appearance", payload);
    const preferences = normalizeAppearancePreferences(
      response.appearance_preferences || {},
    );
    applyAppearanceTheme(preferences);
    syncAppearanceFormState();
    setStatus("#appearanceStatus", "appearance saved", "ok");
    await loadAppearance();
  } catch (err) {
    setStatus("#appearanceStatus", `save failed: ${err.message}`, "err");
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
  const chat_session_id = qs('input[name="chatSessionId"]').value.trim();
  if (!agent_id || !summary) {
    setStatus("#chatStatus", "chat form requires agent id and summary", "err");
    return;
  }

  try {
    const payload = { agent_id, summary };
    if (chat_session_id) {
      payload.chat_session_id = chat_session_id;
    }
    await postJson("/api/chat_history", payload);
    setStatus("#chatStatus", "chat note added", "ok");
    qs('input[name="chatSummary"]').value = "";
    if (qs('input[name="chatSessionId"]')) {
      qs('input[name="chatSessionId"]').value = "";
    }
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

async function controlRuntime(action, explicitServiceName = "") {
  const service = explicitServiceName || qs("#runtimeService")?.value || state.runtimeStatus?.defaultService || "busy";
  try {
    const response = await postJson(`/api/runtime/services/${encodeURIComponent(service)}/${action}`, {});
    const message = runtimeUiApi()?.formatActionStatus(action, service, response)
      || response?.message
      || `${action} submitted for ${service}`;
    setStatus("#runtimeStatusText", message, response?.success ? "ok" : "err");
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

  if (action === "refresh-runtime") {
    await loadRuntime();
    setStatus("#runtimeStatusText", "runtime refreshed", "ok");
    return;
  }

  if (action === "runtime-service-start") {
    await controlRuntime("start", target.dataset.serviceName || "");
    return;
  }

  if (action === "runtime-service-stop") {
    await controlRuntime("stop", target.dataset.serviceName || "");
    return;
  }

  if (action === "runtime-service-restart") {
    await controlRuntime("restart", target.dataset.serviceName || "");
    return;
  }

  if (action === "refresh-tool-usage") {
    await loadToolUsage();
    return;
  }

  if (action === "refresh-events") {
    await loadEvents();
    return;
  }

  if (action === "mark-events-seen") {
    markVisibleEventsSeen();
    renderEvents(state.events);
    setStatus("#healthState", "visible events marked seen", "ok");
    return;
  }

  if (action === "refresh-tool-log") {
    await loadToolLog();
    return;
  }

  if (action === "refresh-gm-tickets") {
    await loadGmTickets();
    return;
  }

  if (action === "refresh-mobile-pairing") {
    await loadMobilePairingState();
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
    const memoryId = target.dataset.memoryId || "";
    const chatMessageId = target.dataset.chatMessageId || "";
    const chatSessionId = target.dataset.chatSessionId || "";
    await openToolUsageContext(contextType, contextId, memoryId, chatMessageId, chatSessionId);
    return;
  }

  if (action === "open-tool-mission") {
    const missionId = target.dataset.missionId || "";
    const agentId = target.dataset.agentId || "";
    await loadToolLogByMission(missionId, agentId);
    return;
  }

  if (action === "open-tool-log-entry") {
    const toolCallId = target.dataset.toolCallId || "";
    await openToolLogEntry(toolCallId);
    return;
  }

  if (action === "open-agent-tool-usage") {
    const agentId = target.dataset.id || "";
    await loadAgentToolAudit(agentId);
    return;
  }

  if (action === "open-agent-overlay-history") {
    const agentId = target.dataset.id || "";
    await loadAgentOverlayHistory(agentId);
    return;
  }

  if (action === "refresh-agent-overlay-history") {
    await loadAgentOverlayHistory(state.selectedAgentOverlayHistoryId || "");
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

  if (action === "open-gm-ticket") {
    const ticketId = target.dataset.id;
    if (!ticketId) {
      setStatus("#gmTicketDetailStatus", "ticket id missing", "err");
      return;
    }
    await loadGmTicket(ticketId);
    return;
  }

  if (action === "gm-ticket-status") {
    const ticketId = target.dataset.ticketId;
    if (!state.selectedGmTicket || state.selectedGmTicket.id !== ticketId) {
      setStatus("#gmTicketDetailStatus", "open ticket before updating", "err");
      return;
    }
    state.selectedGmTicket.status = target.value;
    return;
  }

  if (action === "gm-ticket-priority") {
    const ticketId = target.dataset.ticketId;
    if (!state.selectedGmTicket || state.selectedGmTicket.id !== ticketId) {
      setStatus("#gmTicketDetailStatus", "open ticket before updating", "err");
      return;
    }
    state.selectedGmTicket.priority = target.value;
    return;
  }

  if (action === "gm-ticket-scope") {
    const ticketId = target.dataset.ticketId;
    if (!state.selectedGmTicket || state.selectedGmTicket.id !== ticketId) {
      setStatus("#gmTicketDetailStatus", "open ticket before updating", "err");
      return;
    }
    state.selectedGmTicket.agent_scope = target.value;
    return;
  }

  if (action === "gm-ticket-phase") {
    const ticketId = target.dataset.ticketId;
    if (!state.selectedGmTicket || state.selectedGmTicket.id !== ticketId) {
      setStatus("#gmTicketDetailStatus", "open ticket before updating", "err");
      return;
    }
    state.selectedGmTicket.phase = target.value;
    return;
  }

  if (action === "gm-ticket-assigned-to") {
    const ticketId = target.dataset.ticketId;
    if (!state.selectedGmTicket || state.selectedGmTicket.id !== ticketId) {
      setStatus("#gmTicketDetailStatus", "open ticket before updating", "err");
      return;
    }
    state.selectedGmTicket.assigned_to = target.value;
    return;
  }

  if (action === "gm-ticket-dispatch-objective") {
    return;
  }

  if (action === "gm-ticket-dispatch-role") {
    return;
  }

  if (action === "dispatch-gm-ticket") {
    const ticketId = target.dataset.ticketId;
    await dispatchGmTicket(ticketId);
    return;
  }

  if (action === "save-gm-ticket-update") {
    const ticketId = target.dataset.ticketId;
    if (ticketId) {
      state.selectedGmTicketId = ticketId;
    }
    await saveGmTicketUpdate();
    return;
  }

  if (action === "close-gm-ticket") {
    const ticketId = target.dataset.ticketId;
    await closeGmTicket(ticketId);
    return;
  }

  if (action === "export-gm-ticket-audit") {
    const ticketId = target.dataset.ticketId;
    await exportGmTicketAudit(ticketId);
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
          recordProviderActionResult(id, {
            action: "model discovery",
            tone: "ok",
            message: `Discovered ${result.discovered_models.length} model(s)`,
            detail: result.discovered_models.join(", "),
          });
        } else if (result?.message) {
          setStatus("#healthState", result.message, "err");
          recordProviderActionResult(id, {
            action: "model discovery",
            tone: "err",
            message: result.message,
            detail: "Discovery completed without usable model results.",
          });
        } else {
          setStatus("#healthState", "Discovery returned no models. Enter model manually.", "err");
          recordProviderActionResult(id, {
            action: "model discovery",
            tone: "err",
            message: "Discovery returned no models.",
            detail: "Manual model entry is still required.",
          });
        }
      } catch (err) {
        setStatus("#healthState", `Discovery failed: ${err.message}`, "err");
        recordProviderActionResult(id, {
          action: "model discovery",
          tone: "err",
          message: "Discovery failed",
          detail: err.message,
        });
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
          recordProviderActionResult(id, {
            action: "provider test",
            tone: "ok",
            message: `Provider test passed in ${result.latency_ms}ms`,
            detail: `${result.models_count || 0} model(s) returned`,
          });
        } else {
          setStatus("#healthState", `Provider test failed: ${result.error || "see provider card"}`, "err");
          recordProviderActionResult(id, {
            action: "provider test",
            tone: "err",
            message: "Provider test failed",
            detail: result.error || "No error detail returned.",
          });
        }
      } catch (err) {
        setStatus("#healthState", `Provider test failed: ${err.message}`, "err");
        recordProviderActionResult(id, {
          action: "provider test",
          tone: "err",
          message: "Provider test failed",
          detail: err.message,
        });
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
        recordProviderActionResult(id, {
          action: mode === "rotate" ? "secret rotation" : "secret set",
          tone: "err",
          message: "Provider secret requires a key",
          detail: "No key value was provided for the requested secret update.",
        });
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
        recordProviderActionResult(id, {
          action: mode === "rotate" ? "secret rotation" : "secret set",
          tone: "ok",
          message: secret ? "Provider secret available" : "Provider secret updated",
          detail: secret ? "Credential material is now present for this provider." : "The provider metadata updated, but secret presence was not confirmed.",
        });
      } catch (err) {
        setStatus("#healthState", `Secret update failed: ${err.message}`, "err");
        recordProviderActionResult(id, {
          action: mode === "rotate" ? "secret rotation" : "secret set",
          tone: "err",
          message: "Secret update failed",
          detail: err.message,
        });
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
        recordProviderActionResult(id, {
          action: "secret clear",
          tone: secret ? "err" : "ok",
          message: secret ? "Provider secret still present" : "Provider secret cleared",
          detail: secret
            ? "Clear action completed but the provider still reports secret presence."
            : "Credential material is no longer present for this provider.",
        });
      } catch (err) {
        setStatus("#healthState", `Secret clear failed: ${err.message}`, "err");
        recordProviderActionResult(id, {
          action: "secret clear",
          tone: "err",
          message: "Secret clear failed",
          detail: err.message,
        });
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

    if (action === "archive-agent") {
      const id = target.dataset.id;
      const reason = window.prompt(`Reason for archiving ${id}:`)?.trim();
      if (!reason) {
        setStatus("#healthState", "archive cancelled (reason required)", "err");
        return;
      }
      const replacement = window.prompt(`Optional replacement agent id for ${id} responsibilities:`)?.trim();
      const payload = { reason };
      if (replacement) {
        payload.replacement_agent_id = replacement;
      }
      await postJson(`/api/agents/${id}/archive`, payload);
      await loadAgents();
      return;
    }

    if (action === "restore-agent") {
      const id = target.dataset.id;
      const reason = window.prompt(`Reason for restoring ${id}:`)?.trim();
      if (!reason) {
        setStatus("#healthState", "restore cancelled (reason required)", "err");
        return;
      }
      await postJson(`/api/agents/${id}/restore`, { reason });
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

    if (action === "show-plugin-diagnostics") {
      const pluginId = target.dataset.pluginId || "";
      await loadPluginDiagnostics(pluginId);
      return;
    }

    if (action === "revoke-mobile-pairing") {
      const tokenId = target.dataset.tokenId || "";
      if (!tokenId) {
        setStatus("#mobilePairingStatus", "token_id is required", "err");
        return;
      }
      setStatus("#mobilePairingStatus", `revoking ${tokenId}...`, "");
      await postJson("/api/mobile/pairing/revoke", { token_id: tokenId });
      setStatus("#mobilePairingStatus", `revoked ${tokenId}`, "ok");
      await loadMobilePairingState();
      return;
    }

    if (action === "execute-plugin-ui-action") {
      const pluginId = target.dataset.pluginId || "";
      const actionId = target.dataset.actionId || "";
      const method = String(target.dataset.method || "POST").trim().toUpperCase() || "POST";
      const section = target.closest(".plugin-ui-action");
      if (!section) {
        setStatus("#healthState", "plugin action form missing", "err");
        return;
      }

      const statusEl = section.querySelector('[data-action="plugin-ui-action-status"]');
      const fields = section.querySelectorAll("[data-plugin-ui-action-field]");
      const payload = { method };
      for (const field of fields) {
        const fieldName = String(field.dataset.pluginUiField || "").trim();
        if (!fieldName) {
          continue;
        }
        const fieldType = String(field.dataset.pluginUiFieldType || field.type || "text").trim().toLowerCase();
        const isCheckbox = fieldType === "checkbox" || field.type === "checkbox";
        const value = field.value;
        if (isCheckbox) {
          payload[fieldName] = Boolean(field.checked);
          continue;
        }
        if (fieldType === "number") {
          if (value !== "") {
            const normalized = Number(value);
            if (!Number.isFinite(normalized)) {
              setStatus("#healthState", `Invalid number for ${fieldName}`, "err");
              if (statusEl) {
                statusEl.textContent = `Invalid number for ${fieldName}`;
                statusEl.className = "status err";
              }
              return;
            }
            payload[fieldName] = normalized;
          }
          continue;
        }
        if (fieldType === "select") {
          const selected = String(value || "").trim();
          if (field.required && !selected) {
            const message = `${fieldName} is required`;
            setStatus("#healthState", message, "err");
            if (statusEl) {
              statusEl.textContent = message;
              statusEl.className = "status err";
            }
            return;
          }
          if (selected !== "") {
            payload[fieldName] = selected;
          }
          continue;
        }
        if (fieldType === "textarea" || fieldType === "text") {
          const text = String(value || "").trim();
          if (field.required && !text) {
            const message = `${fieldName} is required`;
            setStatus("#healthState", message, "err");
            if (statusEl) {
              statusEl.textContent = message;
              statusEl.className = "status err";
            }
            return;
          }
          if (text !== "") {
            payload[fieldName] = text;
          }
          continue;
        }
      }

      if (statusEl) {
        statusEl.textContent = "running action...";
        statusEl.className = "status";
      }
      let response;
      try {
        response = await postJson(`/api/plugins/${encodeURIComponent(pluginId)}/ui/${encodeURIComponent(actionId)}`, payload);
      } catch (err) {
        const message = `plugin action failed: ${err.message}`;
        setStatus("#healthState", message, "err");
        pluginUiConsoleLogger()?.logActionRequestFailure(pluginId, actionId, err);
        if (statusEl) {
          statusEl.textContent = message;
          statusEl.className = "status err";
        }
        return;
      }

      pluginUiConsoleLogger()?.logActionResult(pluginId, actionId, response);

      if (response?.result?.success) {
        const msg = response.result?.message || "action executed";
        setStatus("#healthState", `plugin action "${actionId}" completed`, "ok");
        if (statusEl) {
          statusEl.textContent = msg;
          statusEl.className = "status ok";
        }
      } else {
        const message = response?.result?.message || "action failed";
        setStatus("#healthState", message, "err");
        if (statusEl) {
          statusEl.textContent = message;
          statusEl.className = "status err";
        }
      }

      await loadPlugins();
      return;
    }
  } catch (err) {
    setStatus("#healthState", `update failed: ${err.message}`, "err");
    if (action === "revoke-mobile-pairing") {
      setStatus("#mobilePairingStatus", `revoke failed: ${err.message}`, "err");
    }
  }
}

async function boot() {
  const token = getToken();
  const tokenInput = qs("#apiToken");
  if (tokenInput && token && !tokenInput.value) {
    tokenInput.value = token;
  }

  try {
    await loadAppearance();
    await loadSettings();
    const loadTasks = [
      loadHealth(),
      loadProviders(),
      loadProviderChain(),
      loadPlugins(),
      loadCorePluginCoverage(),
      loadImportJobs(),
      loadAgents(),
      loadAgentDirectory(),
      loadGmTickets(),
      loadEvents(),
      loadMemory(),
      loadChatHistory(),
      loadToolUsage(),
      loadRuntime(),
    ];
    if (state.role === "admin") {
      loadTasks.push(loadMobilePairingState());
    } else {
      state.latestPairingIssue = null;
      state.mobilePairingState = null;
      syncMobilePairingAccess();
    }
    await Promise.all(loadTasks);
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
qs("#appearanceForm")?.addEventListener("submit", saveAppearance);
qs('input[name="appearance_override_enabled"]')?.addEventListener("change", syncAppearanceFormState);
qs('input[name="appearance_sync_enabled"]')?.addEventListener("change", syncAppearanceFormState);
qs("#settingsForm")?.addEventListener("submit", saveSettings);
qs("#memoryForm")?.addEventListener("submit", submitMemory);
qs("#chatForm")?.addEventListener("submit", submitChat);
qs("#importForm")?.addEventListener("submit", submitImport);
qs("#providerCreateForm")?.addEventListener("submit", submitProvider);
qs("#pluginCreateForm")?.addEventListener("submit", submitPlugin);
qs("#mobilePairingIssueForm")?.addEventListener("submit", submitMobilePairingIssue);
qs("#providerFilterKind")?.addEventListener("change", loadProviders);
qs("#providerFilterStatus")?.addEventListener("change", loadProviders);
qs("#providerFilterSecret")?.addEventListener("change", loadProviders);
qs("#providerSortBy")?.addEventListener("change", loadProviders);
qs("#providerSortDesc")?.addEventListener("change", loadProviders);
qs("#agentLifecycleFilter")?.addEventListener("change", loadAgents);
qs("#eventDomainFilter")?.addEventListener("change", () => renderEvents(state.events));
qs("#eventLevelFilter")?.addEventListener("change", () => renderEvents(state.events));
qs("#eventQueryFilter")?.addEventListener("input", () => renderEvents(state.events));
qs("#eventLimitFilter")?.addEventListener("change", loadEvents);
document.body.addEventListener("change", onTableChange);
document.body.addEventListener("click", (event) => {
  const target = event.target?.closest?.("[data-action]") || event.target;
  const action = target?.dataset?.action;
  if (action === "open-dashboard-focus") {
    selectDashboardAttentionCard(target);
    return;
  }
  if (action === "open-dashboard-panel") {
    openDashboardPanel(target).catch((err) => {
      setStatus("#healthState", `dashboard navigation failed: ${err.message}`, "err");
    });
    return;
  }
  if (action === "mark-dashboard-card-reviewed") {
    markDashboardCardSeen(target);
    setStatus("#healthState", "summary marked reviewed", "ok");
    return;
  }
  if (action === "toggle-dashboard-seen-history") {
    toggleDashboardSeenHistory(target);
    return;
  }
  if (action === "mark-dashboard-card-events-seen") {
    markDashboardCardEventsSeen(target);
    renderEvents(state.events);
    setStatus("#healthState", "attached events marked seen", "ok");
    return;
  }
  if (action === "mark-dashboard-attention-event-seen") {
    markDashboardAttentionEventSeen(target);
    renderEvents(state.events);
    setStatus("#healthState", "attention event marked seen", "ok");
    return;
  }
  if (action === "copy-mobile-pairing-code") {
    const code = target.dataset.code || "";
    if (!code) {
      setStatus("#mobilePairingStatus", "pairing code is unavailable", "err");
      return;
    }
    copyTextToClipboard(code)
      .then(() => setStatus("#mobilePairingStatus", "pairing code copied", "ok"))
      .catch((err) => setStatus("#mobilePairingStatus", `copy failed: ${err.message}`, "err"));
    return;
  }
  if (action === "copy-mobile-pairing-qr-payload") {
    const payload = target.dataset.payload || "";
    if (!payload) {
      setStatus("#mobilePairingStatus", "pairing QR payload is unavailable", "err");
      return;
    }
    copyTextToClipboard(payload)
      .then(() => setStatus("#mobilePairingStatus", "pairing QR payload copied", "ok"))
      .catch((err) => setStatus("#mobilePairingStatus", `copy failed: ${err.message}`, "err"));
    return;
  }
  if (!action || (
    !action.startsWith("runtime-")
    && action !== "discover-provider-models"
    && action !== "open-import"
    && action !== "refresh-tool-usage"
    && action !== "refresh-tool-log"
    && action !== "load-tool-log-session"
    && action !== "refresh-gm-tickets"
    && action !== "refresh-mobile-pairing"
    && action !== "mark-events-seen"
    && action !== "open-dashboard-panel"
    && action !== "mark-dashboard-card-reviewed"
    && action !== "toggle-dashboard-seen-history"
    && action !== "mark-dashboard-card-events-seen"
    && action !== "mark-dashboard-attention-event-seen"
    && action !== "copy-mobile-pairing-qr-payload"
    && action !== "open-gm-ticket"
    && action !== "rerun-import"
    && action !== "open-tool-context"
    && action !== "open-tool-session"
    && action !== "open-tool-log-entry"
    && action !== "open-agent-tool-usage"
    && action !== "open-agent-overlay-history"
    && action !== "refresh-agent-overlay-history"
    && action !== "gm-ticket-status"
    && action !== "gm-ticket-priority"
    && action !== "gm-ticket-scope"
    && action !== "gm-ticket-phase"
    && action !== "gm-ticket-assigned-to"
    && action !== "gm-ticket-dispatch-objective"
    && action !== "gm-ticket-dispatch-role"
    && action !== "dispatch-gm-ticket"
    && action !== "save-gm-ticket-update"
    && action !== "close-gm-ticket"
    && action !== "export-gm-ticket-audit"
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
    && action !== "show-plugin-diagnostics"
    && action !== "execute-plugin-ui-action"
    && action !== "revoke-mobile-pairing"
    && action !== "archive-agent"
    && action !== "restore-agent"
    && action !== "open-dashboard-focus"
  )) {
    return;
  }
  onTableChange(event);
});
qs("#importRerunFile")?.addEventListener("change", submitImportRerun);
qs("#gmTicketCreateForm")?.addEventListener("submit", submitGmTicket);
qs("#gmTicketMessageForm")?.addEventListener("submit", submitGmTicketMessage);
qs("#gmDirectMessageForm")?.addEventListener("submit", submitGmDirectMessage);
qs("#gmTicketFilterStatus")?.addEventListener("change", loadGmTickets);
qs("#gmTicketFilterPriority")?.addEventListener("change", loadGmTickets);
qs("#gmTicketFilterAssignedTo")?.addEventListener("change", loadGmTickets);
qs("#gmTicketFilterPhase")?.addEventListener("change", loadGmTickets);
qs("#gmTicketFilterLimit")?.addEventListener("change", loadGmTickets);
qs("#gmTicketFilterOffset")?.addEventListener("change", loadGmTickets);

boot();
setInterval(boot, 15000);
