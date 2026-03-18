(function (global) {
  "use strict";

  function escapeHtml(value) {
    const text = `${value ?? ""}`;
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }

  function normalizeStatus(value) {
    return `${value || "configured"}`.trim().toLowerCase() || "configured";
  }

  function parseMetadata(value) {
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

  function pushIssue(issues, tone, label, detail, action) {
    issues.push({ tone, label, detail, action });
  }

  function buildProviderHealthSummary(input) {
    const payload = input && typeof input === "object" ? input : {};
    const provider = payload.provider && typeof payload.provider === "object" ? payload.provider : {};
    const metadata = parseMetadata(payload.metadata != null ? payload.metadata : provider.metadata);
    const metrics = payload.metrics && typeof payload.metrics === "object" ? payload.metrics : {};
    const lastTest = payload.last_test && typeof payload.last_test === "object"
      ? payload.last_test
      : metadata.last_test && typeof metadata.last_test === "object"
      ? metadata.last_test
      : {};
    const routingNode = payload.routingNode && typeof payload.routingNode === "object" ? payload.routingNode : {};
    const status = normalizeStatus(payload.status || routingNode.status || provider.status);
    const secretPolicy = `${metadata.secret_policy || "required"}`.trim().toLowerCase() || "required";
    const secretPresent = Boolean(metadata.secret_present);
    const issues = [];

    if (secretPolicy === "required" && !secretPresent) {
      pushIssue(
        issues,
        "err",
        "missing required secret",
        "Provider cannot be verified or used against secret-backed endpoints until a key is set.",
        "Set the provider secret, then rerun the provider test."
      );
    }

    if (status === "unreachable") {
      pushIssue(
        issues,
        "err",
        "provider unreachable",
        metrics.last_error_message || lastTest.error || "Transport or endpoint checks are currently failing.",
        "Restore endpoint connectivity or credentials, then rerun the provider test."
      );
    } else if (status === "degraded") {
      pushIssue(
        issues,
        "warn",
        "provider degraded",
        metrics.last_error_message || lastTest.error || "Provider is available, but recent checks indicate errors or reduced quality.",
        "Inspect recent failures and rerun the provider test after the suspected fix."
      );
    } else if (status === "standby") {
      pushIssue(
        issues,
        "warn",
        "standby fallback only",
        "Provider is enabled for fallback but is not carrying primary routing traffic.",
        "Promote priority or mark active if this provider should take live traffic."
      );
    } else if (status === "configured") {
      pushIssue(
        issues,
        "warn",
        "configured but not proven",
        "Provider is configured but has not been promoted into a clearly healthy runtime state.",
        "Run the provider test to confirm endpoint reachability and model discovery."
      );
    }

    if (`${lastTest.status || ""}`.trim().toLowerCase() === "fail") {
      pushIssue(
        issues,
        "err",
        "last provider test failed",
        lastTest.error || "The most recent provider test did not complete successfully.",
        "Fix endpoint or credential issues, then rerun the provider test."
      );
    } else if (!lastTest.tested_at) {
      pushIssue(
        issues,
        "warn",
        "no recorded provider test",
        "There is no recent test result on record for this provider.",
        "Run the provider test to validate the current endpoint and model surface."
      );
    }

    const recentFailures = Number(metrics.failure_count_last_1m || 0);
    if (recentFailures > 0) {
      pushIssue(
        issues,
        "warn",
        "recent request failures",
        `${recentFailures} failure(s) were recorded in the last minute.`,
        "Inspect the last error and health metrics before leaving this provider in the active route."
      );
    }

    if (!issues.length) {
      return {
        tone: "ok",
        summary: "provider ready",
        detail: "No secret, routing, or recent health issues are currently flagged.",
        action: "No immediate operator action required.",
        issues: [],
      };
    }

    const tone = issues.some((issue) => issue.tone === "err") ? "err" : "warn";
    return {
      tone,
      summary: tone === "err" ? "provider needs intervention" : "provider needs review",
      detail: issues.map((issue) => issue.label).join(" • "),
      action: issues[0].action,
      issues,
    };
  }

  function renderProviderHealthSummary(summary) {
    const view = summary && typeof summary === "object" ? summary : buildProviderHealthSummary({});
    const issues = Array.isArray(view.issues) ? view.issues : [];
    const issueRows = issues
      .map((issue) => `
        <li>
          <strong>${escapeHtml(issue.label)}</strong>
          <span class="status ${escapeHtml(issue.tone)}">${escapeHtml(issue.tone === "err" ? "critical" : "review")}</span>
          <div>${escapeHtml(issue.detail || "")}</div>
          <div class="meta">${escapeHtml(issue.action || "")}</div>
        </li>
      `)
      .join("");

    return `
      <div class="provider-health-summary">
        <p><strong>Operator summary:</strong> ${escapeHtml(view.summary || "provider ready")}</p>
        <p class="meta">${escapeHtml(view.detail || "")}</p>
        <p><strong>Next action:</strong> ${escapeHtml(view.action || "No immediate operator action required.")}</p>
        ${issues.length ? `<ul class="compact-list provider-issue-list">${issueRows}</ul>` : ""}
      </div>
    `;
  }

  global.Busy38ProviderHealthUi = {
    buildProviderHealthSummary,
    renderProviderHealthSummary,
  };
})(window);
