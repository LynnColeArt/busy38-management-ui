(function (global) {
  "use strict";

  function normalizeStatus(value) {
    return `${value || "configured"}`.trim().toLowerCase() || "configured";
  }

  function pluralize(count, singular, plural) {
    return `${count} ${count === 1 ? singular : plural}`;
  }

  function summarizeCounts(parts) {
    return parts.filter(Boolean).join(" • ");
  }

  function buildProviderRoutingSummary(raw) {
    const metadata = raw && typeof raw === "object" && !Array.isArray(raw)
      ? raw
      : { chain: Array.isArray(raw) ? raw : [] };
    const chain = Array.isArray(metadata.chain) ? metadata.chain : [];
    const enabled = chain.filter((provider) => provider && provider.enabled !== false);
    const active = enabled.filter((provider) => normalizeStatus(provider.status) === "active");
    const degraded = enabled.filter((provider) => normalizeStatus(provider.status) === "degraded");
    const standby = enabled.filter((provider) => normalizeStatus(provider.status) === "standby");
    const configured = enabled.filter((provider) => normalizeStatus(provider.status) === "configured");
    const unreachable = enabled.filter((provider) => normalizeStatus(provider.status) === "unreachable");
    const available = enabled.filter((provider) => normalizeStatus(provider.status) !== "unreachable");
    const primary = enabled.find((provider) => provider && provider.active) || enabled[0] || null;
    const primaryId = `${primary?.id || metadata.active_provider_id || "none"}`.trim() || "none";
    const primaryStatus = normalizeStatus(primary?.status);

    if (enabled.length === 0) {
      return {
        state: "empty",
        tone: "err",
        metric: "0",
        summary: "no enabled providers",
        detail: "routing chain has nothing to select",
        focusProviderStatus: "",
        availableCount: 0,
        enabledCount: 0,
        primaryProviderId: "",
      };
    }

    if (available.length === 0) {
      return {
        state: "blocked",
        tone: "err",
        metric: `0/${enabled.length}`,
        summary: "routing blocked",
        detail: `all enabled providers are unreachable • ${pluralize(unreachable.length, "provider", "providers")}`,
        focusProviderStatus: "unreachable",
        availableCount: 0,
        enabledCount: enabled.length,
        primaryProviderId: primaryId,
      };
    }

    const degradedParts = [];
    if (active.length > 0) {
      degradedParts.push(`primary ${primaryId} is active`);
    } else {
      degradedParts.push(`no active primary • first candidate ${primaryId} is ${primaryStatus}`);
    }
    if (degraded.length > 0) {
      degradedParts.push(pluralize(degraded.length, "degraded provider", "degraded providers"));
    }
    if (standby.length > 0) {
      degradedParts.push(pluralize(standby.length, "standby fallback", "standby fallbacks"));
    }
    if (configured.length > 0) {
      degradedParts.push(pluralize(configured.length, "configured fallback", "configured fallbacks"));
    }
    if (unreachable.length > 0) {
      degradedParts.push(pluralize(unreachable.length, "unreachable fallback", "unreachable fallbacks"));
    }

    if (active.length === enabled.length) {
      const extraFallbacks = Math.max(0, active.length - 1);
      return {
        state: "healthy",
        tone: "ok",
        metric: `${available.length}/${enabled.length}`,
        summary: "routing healthy",
        detail: extraFallbacks > 0
          ? `primary ${primaryId} is active • ${pluralize(extraFallbacks, "additional active fallback", "additional active fallbacks")}`
          : `primary ${primaryId} is active`,
        focusProviderStatus: "",
        availableCount: available.length,
        enabledCount: enabled.length,
        primaryProviderId: primaryId,
      };
    }

    return {
      state: "degraded",
      tone: "warn",
      metric: `${available.length}/${enabled.length}`,
      summary: "routing available but degraded",
      detail: summarizeCounts(degradedParts),
      focusProviderStatus: active.length === 0
        ? primaryStatus
        : unreachable.length > 0
        ? "unreachable"
        : degraded.length > 0
        ? "degraded"
        : standby.length > 0
        ? "standby"
        : configured.length > 0
        ? "configured"
        : "",
      availableCount: available.length,
      enabledCount: enabled.length,
      primaryProviderId: primaryId,
    };
  }

  global.Busy38ProviderRoutingSummary = {
    buildProviderRoutingSummary,
  };
})(window);
