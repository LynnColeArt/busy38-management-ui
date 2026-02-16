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
};

const qs = (selector) => document.querySelector(selector);

function getToken() {
  return (qs("#apiToken")?.value || localStorage.getItem(TOKEN_KEY) || "").trim();
}

function saveToken() {
  const token = qs("#apiToken").value.trim();
  if (token === "") {
    localStorage.removeItem(TOKEN_KEY);
    setStatus("#healthState", "token cleared", "");
  } else {
    localStorage.setItem(TOKEN_KEY, token);
    setStatus("#healthState", "token saved", "ok");
  }
  boot();
}

function setStatus(id, text, kind = "") {
  const el = qs(id);
  if (!el) {
    return;
  }
  el.textContent = text;
  el.className = `status ${kind}`;
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

async function apiRequest(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
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

async function postJson(path, body) {
  return apiRequest(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

async function patchJson(path, body) {
  return apiRequest(path, {
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

function buildProviderCard(provider) {
  return `
    <div class="card">
      <h3>${provider.name}</h3>
      <p><strong>Status:</strong> ${provider.status}</p>
      <p><strong>Model:</strong> ${provider.model}</p>
      <p><strong>Endpoint:</strong> ${provider.endpoint}</p>
      <p>
        <label>
          Enabled:
          <input type="checkbox" data-action="toggle-provider" data-id="${provider.id}" ${provider.enabled ? "checked" : ""} />
        </label>
      </p>
      <p>
        <label>
          Priority:
          <input type="number" data-action="set-provider-priority" data-id="${provider.id}" value="${provider.priority}" />
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
}

async function loadProviders() {
  const payload = await fetchJson("/api/providers");
  state.providers = payload.providers || [];
  renderCards("#providers", state.providers, buildProviderCard);
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

async function loadMemory() {
  const payload = await fetchJson("/api/memory");
  renderMemory(payload.memory || []);
}

async function loadChatHistory() {
  const payload = await fetchJson("/api/chat_history");
  renderChat(payload.chat_history || []);
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

  setStatus("#settingsStatus", "saving...", "");
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

async function onTableChange(event) {
  const target = event.target;
  const action = target.dataset.action;
  if (!action) {
    return;
  }

  try {
    if (action === "toggle-provider") {
      const id = target.dataset.id;
      await patchJson(`/api/providers/${id}`, { enabled: target.checked });
      await loadProviders();
      return;
    }

    if (action === "set-provider-priority") {
      const id = target.dataset.id;
      await patchJson(`/api/providers/${id}`, { priority: Number(target.value) || 0 });
      await loadProviders();
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

  await loadHealth();
  try {
    await Promise.all([
      loadSettings(),
      loadProviders(),
      loadAgents(),
      loadEvents(),
      loadMemory(),
      loadChatHistory(),
    ]);
  } catch (err) {
    setStatus("#healthState", `loading failed: ${err.message}`, "err");
  }
}

qs("#saveToken")?.addEventListener("click", saveToken);
qs("#settingsForm")?.addEventListener("submit", saveSettings);
qs("#memoryForm")?.addEventListener("submit", submitMemory);
qs("#chatForm")?.addEventListener("submit", submitChat);
document.body.addEventListener("change", onTableChange);

boot();
setInterval(boot, 15000);
