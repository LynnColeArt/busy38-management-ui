// Keep plugin UI console telemetry isolated from the DOM app so warning/error
// translation stays testable without booting the full management UI runtime.
(function attachPluginUiConsole(global) {
  "use strict";

  function getConsole(consoleRef) {
    const fallback = global.console || {};
    return consoleRef || fallback;
  }

  function normalizeEntries(value) {
    if (!Array.isArray(value)) {
      return [];
    }
    return value.filter((entry) => entry !== null && entry !== undefined && !(typeof entry === "string" && entry.trim() === ""));
  }

  function normalizeStringArray(value) {
    const seen = new Set();
    const normalized = [];
    for (const entry of normalizeEntries(value)) {
      const text = String(entry || "").trim();
      if (!text || seen.has(text)) {
        continue;
      }
      seen.add(text);
      normalized.push(text);
    }
    return normalized;
  }

  function warningDetailsFromActionResult(result) {
    const payload = result && typeof result.payload === "object" && result.payload ? result.payload : {};
    return {
      warnings: [
        ...normalizeEntries(payload.warnings),
        ...normalizeEntries(payload.warnings?.entries),
      ],
      warningCodes: normalizeStringArray(payload.warning_codes),
      reasonCodes: normalizeStringArray(payload.reason_codes),
    };
  }

  function logDiagnostics(pluginId, payload, consoleRef) {
    const logger = getConsole(consoleRef);
    const warnings = normalizeEntries(payload?.warnings?.entries);
    const errors = normalizeEntries(payload?.errors?.entries);
    const baseRecord = {
      pluginId: String(pluginId || "").trim(),
      status: String(payload?.status || "").trim() || "unknown",
      updatedAt: payload?.updated_at || null,
    };
    if (warnings.length && typeof logger.warn === "function") {
      logger.warn("[plugin-ui] diagnostics warnings", {
        ...baseRecord,
        warnings,
      });
    }
    if (errors.length && typeof logger.error === "function") {
      logger.error("[plugin-ui] diagnostics errors", {
        ...baseRecord,
        errors,
      });
    }
  }

  function logActionResult(pluginId, actionId, response, consoleRef) {
    const logger = getConsole(consoleRef);
    const result = response && typeof response.result === "object" && response.result ? response.result : {};
    const warningDetails = warningDetailsFromActionResult(result);
    const record = {
      pluginId: String(pluginId || "").trim(),
      actionId: String(actionId || "").trim(),
      success: Boolean(result.success),
      message: String(result.message || "").trim() || null,
      updatedAt: response?.updated_at || null,
      payload: result.payload && typeof result.payload === "object" ? result.payload : {},
      reasonCodes: warningDetails.reasonCodes,
      warningCodes: warningDetails.warningCodes,
      warnings: warningDetails.warnings,
    };
    if (result.success === false) {
      if (typeof logger.error === "function") {
        logger.error("[plugin-ui] action failed", record);
      }
      return;
    }
    if ((warningDetails.warnings.length || warningDetails.warningCodes.length || warningDetails.reasonCodes.length) && typeof logger.warn === "function") {
      logger.warn("[plugin-ui] action warnings", record);
    }
  }

  function logActionRequestFailure(pluginId, actionId, error, consoleRef) {
    const logger = getConsole(consoleRef);
    if (typeof logger.error !== "function") {
      return;
    }
    const message = error && typeof error.message === "string" ? error.message : String(error || "unknown error");
    logger.error("[plugin-ui] action request failed", {
      pluginId: String(pluginId || "").trim(),
      actionId: String(actionId || "").trim(),
      message,
    });
  }

  global.Busy38PluginUiConsole = Object.freeze({
    logDiagnostics,
    logActionResult,
    logActionRequestFailure,
  });
})(window);
