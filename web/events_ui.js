(function (global) {
  "use strict";

  function escapeHtml(value) {
    const text = `${value ?? ""}`;
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }

  function toDataAttributeName(value) {
    return `${value || ""}`
      .trim()
      .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
      .replace(/[^a-zA-Z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase();
  }

  function normalizeEvent(event) {
    const payload = event && typeof event === "object" ? event : {};
    const data = payload.payload && typeof payload.payload === "object" ? payload.payload : {};
    return {
      id: `${payload.id || ""}`.trim(),
      type: `${payload.type || payload.event_type || "event"}`.trim() || "event",
      domain: deriveEventDomain(payload.type || payload.event_type || "event"),
      created_at: `${payload.created_at || ""}`.trim(),
      level: `${payload.level || "info"}`.trim() || "info",
      message: `${payload.message || ""}`.trim(),
      payload: data,
      seen: Boolean(payload.seen),
    };
  }

  function deriveEventDomain(eventType) {
    const normalizedType = `${eventType || "event"}`.trim() || "event";
    if (normalizedType.includes(".")) {
      return normalizedType.split(".", 1)[0];
    }
    return normalizedType;
  }

  function matchesQuery(event, query) {
    const normalizedQuery = `${query || ""}`.trim().toLowerCase();
    if (!normalizedQuery) {
      return true;
    }
    const haystack = [
      event.type,
      event.domain,
      event.level,
      event.message,
      JSON.stringify(event.payload || {}),
    ].join(" ").toLowerCase();
    return haystack.includes(normalizedQuery);
  }

  function summarizeEvents(events) {
    const summary = {
      total: 0,
      domains: {},
      levels: {},
    };
    for (const event of events) {
      summary.total += 1;
      summary.domains[event.domain] = (summary.domains[event.domain] || 0) + 1;
      summary.levels[event.level] = (summary.levels[event.level] || 0) + 1;
    }
    return summary;
  }

  function buildEventViewModel(rawEvents, filters) {
    const sourceEvents = Array.isArray(rawEvents) ? rawEvents.map(normalizeEvent) : [];
    const normalizedFilters = filters && typeof filters === "object" ? filters : {};
    const seenEventIds = Array.isArray(normalizedFilters.seenEventIds) ? normalizedFilters.seenEventIds : [];
    const seenIdSet = new Set(seenEventIds.map((value) => `${value || ""}`.trim()).filter(Boolean));
    const domain = `${normalizedFilters.domain || ""}`.trim();
    const level = `${normalizedFilters.level || ""}`.trim();
    const query = `${normalizedFilters.query || ""}`.trim();
    const limit = Math.max(1, Math.min(100, Number(normalizedFilters.limit || 50)));
    const summary = summarizeEvents(sourceEvents);
    const filtered = sourceEvents
      .filter((event) => !domain || event.domain === domain)
      .filter((event) => !level || event.level === level)
      .filter((event) => matchesQuery(event, query))
      .map((event) => ({
        ...event,
        seen: event.id ? seenIdSet.has(event.id) : false,
      }))
      .slice(0, limit);

    return {
      summary,
      filters: {
        domain,
        level,
        query,
        limit,
      },
      filteredCount: filtered.length,
      unseenCount: filtered.filter((event) => !event.seen).length,
      availableDomains: Object.keys(summary.domains).sort(),
      events: filtered,
    };
  }

  function buildRuntimeDetails(event) {
    const payload = event.payload || {};
    if (event.type !== "runtime.service_action") {
      return [];
    }
    return [
      ["Service", payload.service || "busy"],
      ["Action", payload.action || "unknown"],
      ["Result", payload.success ? "success" : "failed"],
      ["Runtime source", payload.runtime_source || "none"],
      ["Connected", payload.runtime_connected ? "yes" : "no"],
      ["Actor", payload.actor || "unknown"],
    ];
  }

  function buildGenericDetails(event) {
    const payload = event.payload || {};
    const rows = [];
    if (payload.gm_ticket_id) {
      rows.push(["GM ticket", payload.gm_ticket_id]);
    }
    if (payload.import_id) {
      rows.push(["Import", payload.import_id]);
    }
    if (payload.agent_id) {
      rows.push(["Agent", payload.agent_id]);
    }
    return rows;
  }

  function renderDetailRows(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return "";
    }
    return `
      <div class="event-meta-grid">
        ${rows.map(([label, value]) => `
          <p><strong>${escapeHtml(label)}:</strong> ${escapeHtml(value)}</p>
        `).join("")}
      </div>
    `;
  }

  function renderEventItem(rawEvent, options) {
    const event = normalizeEvent(rawEvent);
    const normalizedOptions = options && typeof options === "object" ? options : {};
    const typeLabel = escapeHtml(event.type);
    const levelClass = event.level === "error" ? "err" : event.level === "warn" ? "warn" : "ok";
    const seenBadge = event.seen
      ? '<span class="status event-seen-badge">seen</span>'
      : '<span class="status warn event-seen-badge">new</span>';
    const detailRows = [...buildRuntimeDetails(event), ...buildGenericDetails(event)];
    const actionName = `${normalizedOptions.actionName || ""}`.trim();
    const actionLabel = `${normalizedOptions.actionLabel || "Mark event seen"}`.trim() || "Mark event seen";
    const extraEventData = normalizedOptions.eventData && typeof normalizedOptions.eventData === "object"
      ? normalizedOptions.eventData
      : {};
    const actionAttributes = Object.entries(extraEventData)
      .map(([key, value]) => {
        const attrKey = toDataAttributeName(key);
        const attrValue = `${value ?? ""}`.trim();
        if (!attrKey || attrValue === "") {
          return "";
        }
        return ` data-${escapeHtml(attrKey)}="${escapeHtml(attrValue)}"`;
      })
      .join("");
    const actionButton = !event.seen && actionName && event.id
      ? `
        <p class="event-card-actions">
          <button type="button" data-action="${escapeHtml(actionName)}" data-event-id="${escapeHtml(event.id)}"${actionAttributes}>${escapeHtml(actionLabel)}</button>
        </p>
      `
      : "";
    return `
      <li class="card event-card${event.seen ? " is-seen" : " is-unseen"}" data-event-id="${escapeHtml(event.id || "")}">
        <p><strong>${typeLabel}</strong> <span class="status ${levelClass}">${escapeHtml(event.level)}</span> ${seenBadge}</p>
        <p>${escapeHtml(event.message || "No event message.")}</p>
        <p class="meta">${escapeHtml(event.created_at || "unknown time")}</p>
        ${renderDetailRows(detailRows)}
        ${actionButton}
      </li>
    `;
  }

  global.Busy38EventsUi = {
    buildEventViewModel,
    deriveEventDomain,
    renderEventItem,
  };
})(window);
