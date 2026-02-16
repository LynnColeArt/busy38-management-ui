const API_BASE = (() => {
  const configured = (window.MANAGEMENT_API_BASE || "").trim();
  return configured !== "" ? configured : "http://127.0.0.1:8031";
})();

const state = {
  settings: {},
  providers: [],
  agents: [],
  events: [],
};

const qs = (selector) => document.querySelector(selector);

function setStatus(id, text, kind = "") {
  const el = qs(id);
  if (!el) {
    return;
  }
  el.textContent = text;
  el.className = `status ${kind}`;
}

async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json();
}

function renderCards(targetId, items, renderRow) {
  const container = qs(targetId);
  const cards = (items || []).map(renderRow).join("");
  container.innerHTML = `<div class=\"card-list\">${cards || "<p>No data</p>"}</div>`;
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
  list.innerHTML = (items || []).map((event) => `<li>${event.created_at} — [${event.level}] ${event.message}</li>`).join("");
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
  list.innerHTML = (rows || []).map((row) => `<li><strong>${row.agent_id}</strong> (${row.id}): ${row.summary} — <small>${row.timestamp}</small></li>`).join("");
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
  try {
    const payload = await fetchJson("/api/memory");
    renderMemory(payload.memory || []);
  } catch (err) {
    console.warn("memory load failed", err);
  }
}

async function loadChatHistory() {
  try {
    const payload = await fetchJson("/api/chat_history");
    renderChat(payload.chat_history || []);
  } catch (err) {
    console.warn("chat history load failed", err);
  }
}

async function saveSettings(event) {
  event.preventDefault();
  const autoRestart = qs('input[name="auto_restart"]').checked;
  const payload = {
    heartbeat_interval: Number(qs('input[name="heartbeat_interval"]').value) || state.settings.heartbeat_interval,
    fallback_budget_per_hour: Number(qs('input[name="fallback_budget_per_hour"]').value) || state.settings.fallback_budget_per_hour,
    auto_restart: autoRestart,
  };
  setStatus("#settingsStatus", "saving...", "");
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.text();
    setStatus("#settingsStatus", `save failed: ${body}`, "err");
    return;
  }
  setStatus("#settingsStatus", "settings saved", "ok");
  await loadSettings();
}

async function onTableChange(event) {
  const target = event.target;
  const action = target.dataset.action;
  if (!action) {
    return;
  }

  if (action === "toggle-provider") {
    const id = target.dataset.id;
    const payload = { enabled: target.checked };
    await fetch(`${API_BASE}/api/providers/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadProviders();
  }

  if (action === "set-provider-priority") {
    const id = target.dataset.id;
    const payload = { priority: Number(target.value) || 0 };
    await fetch(`${API_BASE}/api/providers/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadProviders();
  }

  if (action === "toggle-agent") {
    const id = target.dataset.id;
    const payload = { enabled: target.checked };
    await fetch(`${API_BASE}/api/agents/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadAgents();
  }
}

async function boot() {
  await loadHealth();
  await Promise.all([loadSettings(), loadProviders(), loadAgents(), loadEvents(), loadMemory(), loadChatHistory()]);
}

qs("#settingsForm").addEventListener("submit", saveSettings);
document.body.addEventListener("change", onTableChange);

boot();
setInterval(boot, 15000);
