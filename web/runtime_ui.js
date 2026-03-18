(function (global) {
  "use strict";

  function escapeHtml(value) {
    const text = `${value ?? ""}`;
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }

  function normalizeService(service) {
    const payload = service && typeof service === "object" ? service : {};
    return {
      name: `${payload.name || "busy"}`.trim() || "busy",
      running: Boolean(payload.running),
      pid: payload.pid == null || payload.pid === "" ? null : payload.pid,
      pid_file: `${payload.pid_file || ""}`.trim(),
      log_file: `${payload.log_file || ""}`.trim(),
    };
  }

  function coerceServices(statusPayload, servicesPayload) {
    const servicesValue = servicesPayload && typeof servicesPayload === "object" ? servicesPayload.services : null;
    if (Array.isArray(servicesValue)) {
      return servicesValue.map(normalizeService);
    }

    const statusServices = statusPayload && typeof statusPayload === "object" ? statusPayload.services : null;
    if (statusServices && typeof statusServices === "object") {
      return Object.values(statusServices).map(normalizeService);
    }

    return [];
  }

  function summarizeOrchestrator(orchestrator) {
    if (!orchestrator || typeof orchestrator !== "object") {
      return "";
    }
    if (orchestrator.available === false) {
      return `orchestrator unavailable${orchestrator.error ? `: ${orchestrator.error}` : ""}`;
    }
    if (orchestrator.status) {
      return `orchestrator: ${orchestrator.status}`;
    }
    if (typeof orchestrator.running === "boolean") {
      return `orchestrator running: ${orchestrator.running ? "yes" : "no"}`;
    }
    return "orchestrator: connected";
  }

  function buildRuntimeViewModel(statusPayload, servicesPayload) {
    const safeStatus = statusPayload && typeof statusPayload === "object" ? statusPayload : {};
    const safeServicesPayload = servicesPayload && typeof servicesPayload === "object" ? servicesPayload : {};
    const services = coerceServices(safeStatus, safeServicesPayload).sort((a, b) => a.name.localeCompare(b.name));
    const source = `${safeStatus.source || safeServicesPayload.source || "none"}`.trim() || "none";
    const connected = Boolean(
      safeStatus.connected != null ? safeStatus.connected : safeServicesPayload.connected,
    );
    const defaultService = `${safeServicesPayload.default_service || safeStatus.default_service || services[0]?.name || "busy"}`
      .trim() || "busy";
    const serviceCount = services.length;
    const runningCount = services.filter((service) => service.running).length;
    const orchestratorLine = summarizeOrchestrator(safeStatus.orchestrator);
    const error = `${safeStatus.error || safeServicesPayload.error || ""}`.trim();

    const statusLine = `runtime (${source}) - ${connected ? "connected" : "unavailable"} - ${runningCount}/${serviceCount} running`;
    const metaParts = [`default service: ${defaultService}`];
    if (serviceCount > 0) {
      metaParts.push(`services: ${serviceCount}`);
    }
    if (orchestratorLine) {
      metaParts.push(orchestratorLine);
    }
    if (error) {
      metaParts.push(`note: ${error}`);
    }

    return {
      summary: {
        source,
        connected,
        defaultService,
        serviceCount,
        runningCount,
        error,
        orchestratorLine,
        statusLine,
        metaLine: metaParts.join(" • "),
        statusKind: connected ? "ok" : "err",
      },
      services,
    };
  }

  function renderServiceCard(service, options) {
    const normalized = normalizeService(service);
    const canControl = Boolean(options && options.canControl);
    const statusClass = normalized.running ? "running" : "stopped";
    const disabled = canControl ? "" : " disabled";
    const pidText = normalized.pid == null ? "-" : escapeHtml(normalized.pid);
    const pidFileText = normalized.pid_file ? `<code class="runtime-path">${escapeHtml(normalized.pid_file)}</code>` : "-";
    const logFileText = normalized.log_file ? `<code class="runtime-path">${escapeHtml(normalized.log_file)}</code>` : "-";
    const serviceName = escapeHtml(normalized.name);

    return `
      <div class="card">
        <h3>${serviceName}</h3>
        <p><strong>Status:</strong> <span class="status-badge status-${statusClass}">${normalized.running ? "running" : "stopped"}</span></p>
        <p><strong>PID:</strong> ${pidText}</p>
        <p><strong>PID file:</strong> ${pidFileText}</p>
        <p><strong>Log file:</strong> ${logFileText}</p>
        <div class="runtime-service-actions">
          <button type="button" data-action="runtime-service-start" data-runtime-control="1" data-service-name="${serviceName}"${disabled}>Start</button>
          <button type="button" data-action="runtime-service-stop" data-runtime-control="1" data-service-name="${serviceName}"${disabled}>Stop</button>
          <button type="button" data-action="runtime-service-restart" data-runtime-control="1" data-service-name="${serviceName}"${disabled}>Restart</button>
        </div>
      </div>
    `;
  }

  function formatActionStatus(action, serviceName, response) {
    const payload = response && typeof response === "object" ? response : {};
    const message = `${payload.message || `${action} requested for ${serviceName}`}`.trim();
    const updatedAt = `${payload.updated_at || ""}`.trim();
    return updatedAt ? `${message} (${updatedAt})` : message;
  }

  global.Busy38RuntimeUi = {
    buildRuntimeViewModel,
    renderServiceCard,
    formatActionStatus,
  };
})(window);
