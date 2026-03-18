(function (global) {
  "use strict";

  function escapeHtml(value) {
    const text = `${value ?? ""}`;
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }

  function normalizeProviderActionResult(raw) {
    const payload = raw && typeof raw === "object" ? raw : {};
    const tone = `${payload.tone || ""}`.trim() || "warn";
    return {
      providerId: `${payload.providerId || ""}`.trim(),
      action: `${payload.action || "provider update"}`.trim() || "provider update",
      tone,
      message: `${payload.message || ""}`.trim() || "No action result recorded.",
      detail: `${payload.detail || ""}`.trim(),
      recordedAt: `${payload.recordedAt || ""}`.trim(),
    };
  }

  function renderProviderActionResult(raw) {
    const view = normalizeProviderActionResult(raw);
    const timestamp = view.recordedAt
      ? `<p class="meta"><strong>Recorded:</strong> ${escapeHtml(view.recordedAt)}</p>`
      : "";
    const detail = view.detail
      ? `<p class="meta">${escapeHtml(view.detail)}</p>`
      : "";
    return `
      <div class="provider-action-result provider-action-result-${escapeHtml(view.tone)}">
        <p><strong>Last action:</strong> ${escapeHtml(view.action)}</p>
        <p><span class="status ${escapeHtml(view.tone)}">${escapeHtml(view.message)}</span></p>
        ${detail}
        ${timestamp}
      </div>
    `;
  }

  global.Busy38ProviderActionUi = {
    normalizeProviderActionResult,
    renderProviderActionResult,
  };
})(window);
