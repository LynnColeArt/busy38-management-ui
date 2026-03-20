(function (global) {
  "use strict";

  function escapeHtml(value) {
    const text = `${value ?? ""}`;
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }

  function deriveEventDomain(eventType) {
    const normalizedType = `${eventType || "event"}`.trim() || "event";
    if (normalizedType.includes(".")) {
      return normalizedType.split(".", 1)[0];
    }
    return normalizedType;
  }

  function providerRoutingSummaryApi() {
    const api = global.Busy38ProviderRoutingSummary;
    if (!api || typeof api !== "object") {
      return null;
    }
    if (typeof api.buildProviderRoutingSummary !== "function") {
      return null;
    }
    return api;
  }

  function providerHealthUiApi() {
    const api = global.Busy38ProviderHealthUi;
    if (!api || typeof api !== "object") {
      return null;
    }
    if (typeof api.buildProviderHealthSummary !== "function") {
      return null;
    }
    return api;
  }

  function providerActionUiApi() {
    const api = global.Busy38ProviderActionUi;
    if (!api || typeof api !== "object") {
      return null;
    }
    if (typeof api.normalizeProviderActionResult !== "function") {
      return null;
    }
    return api;
  }

  function timestampValue(value) {
    const millis = Date.parse(`${value || ""}`);
    return Number.isFinite(millis) ? millis : 0;
  }

  function latestTimestamp(values) {
    return (Array.isArray(values) ? values : [])
      .map((value) => ({ value, ts: timestampValue(value) }))
      .filter((entry) => entry.ts > 0)
      .sort((left, right) => right.ts - left.ts)[0]?.value || "";
  }

  function describeFreshness(timestamp, nowValue) {
    const ts = timestampValue(timestamp);
    const nowTs = timestampValue(nowValue) || Date.now();
    if (!ts || ts > nowTs) {
      return { tone: "", label: "", detail: "", timestamp: "" };
    }
    const diffMs = Math.max(0, nowTs - ts);
    const diffMinutes = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    const detail = diffMinutes < 60
      ? `${Math.max(1, diffMinutes)}m ago`
      : diffHours < 24
      ? `${Math.max(1, diffHours)}h ago`
      : `${Math.max(1, diffDays)}d ago`;
    if (diffMinutes <= 15) {
      return { tone: "err", label: "fresh", detail, timestamp: `${timestamp || ""}`.trim() };
    }
    if (diffHours <= 6) {
      return { tone: "warn", label: "recent", detail, timestamp: `${timestamp || ""}`.trim() };
    }
    return { tone: "ok", label: "older", detail, timestamp: `${timestamp || ""}`.trim() };
  }

  function countUnseenEvents(events, seenEventIds, domains) {
    const allowed = new Set(Array.isArray(domains) ? domains : []);
    const seen = new Set(Array.isArray(seenEventIds) ? seenEventIds.map((value) => `${value || ""}`.trim()).filter(Boolean) : []);
    return (Array.isArray(events) ? events : []).filter((event) => {
      const eventId = `${event?.id || ""}`.trim();
      const domain = deriveEventDomain(event?.type || event?.event_type || "event");
      if (!eventId || seen.has(eventId)) {
        return false;
      }
      return allowed.has(domain);
    }).length;
  }

  function countSeenEvents(events, seenEventIds, domains) {
    const allowed = new Set(Array.isArray(domains) ? domains : []);
    const seen = new Set(Array.isArray(seenEventIds) ? seenEventIds.map((value) => `${value || ""}`.trim()).filter(Boolean) : []);
    return (Array.isArray(events) ? events : []).filter((event) => {
      const eventId = `${event?.id || ""}`.trim();
      const domain = deriveEventDomain(event?.type || event?.event_type || "event");
      if (!eventId || !seen.has(eventId)) {
        return false;
      }
      return allowed.has(domain);
    }).length;
  }

  function selectUnseenEvents(events, seenEventIds, domains, limit) {
    const allowed = new Set(Array.isArray(domains) ? domains : []);
    const seen = new Set(Array.isArray(seenEventIds) ? seenEventIds.map((value) => `${value || ""}`.trim()).filter(Boolean) : []);
    const maxItems = Math.max(1, Math.min(10, Number(limit || 5)));
    return (Array.isArray(events) ? events : [])
      .filter((event) => {
        const eventId = `${event?.id || ""}`.trim();
        const domain = deriveEventDomain(event?.type || event?.event_type || "event");
        if (!eventId || seen.has(eventId)) {
          return false;
        }
        return allowed.has(domain);
      })
      .slice(0, maxItems)
      .map((event) => ({ ...event, seen: false }));
  }

  function selectSeenEvents(events, seenEventIds, domains, limit) {
    const allowed = new Set(Array.isArray(domains) ? domains : []);
    const seen = new Set(Array.isArray(seenEventIds) ? seenEventIds.map((value) => `${value || ""}`.trim()).filter(Boolean) : []);
    const maxItems = Math.max(1, Math.min(10, Number(limit || 5)));
    return (Array.isArray(events) ? events : [])
      .filter((event) => {
        const eventId = `${event?.id || ""}`.trim();
        const domain = deriveEventDomain(event?.type || event?.event_type || "event");
        if (!eventId || !seen.has(eventId)) {
          return false;
        }
        return allowed.has(domain);
      })
      .slice()
      .sort((left, right) => timestampValue(right?.created_at) - timestampValue(left?.created_at))
      .slice(0, maxItems)
      .map((event) => ({ ...event, seen: true }));
  }

  function seenHistoryScopeLabel(domains) {
    const primaryDomain = `${Array.isArray(domains) ? domains[0] || "" : ""}`.trim();
    if (primaryDomain === "runtime") {
      return "runtime event";
    }
    if (primaryDomain === "provider") {
      return "provider event";
    }
    if (primaryDomain === "gm_ticket") {
      return "GM event";
    }
    if (primaryDomain === "plugin") {
      return "startup event";
    }
    return "event";
  }

  function oldestTimestamp(events) {
    return (Array.isArray(events) ? events : [])
      .map((event) => ({ value: event?.created_at, ts: timestampValue(event?.created_at) }))
      .filter((entry) => entry.ts > 0)
      .sort((left, right) => left.ts - right.ts)[0]?.value || "";
  }

  function buildSeenHistorySummary(domains, shownEvents, totalCount, nowValue) {
    const events = Array.isArray(shownEvents) ? shownEvents : [];
    const shown = events.length;
    const total = Math.max(0, Number(totalCount || 0));
    if (total <= 0) {
      return "";
    }
    const noun = seenHistoryScopeLabel(domains);
    const nounLabel = shown === 1 ? noun : `${noun}s`;
    const base = `showing ${shown} recent seen ${nounLabel}`;
    const countSuffix = total > shown ? ` of ${total}` : "";
    const oldestShownFreshness = describeFreshness(oldestTimestamp(events), nowValue);
    const ageSuffix = oldestShownFreshness.detail ? `, oldest shown ${oldestShownFreshness.detail}` : "";
    return `${base}${countSuffix}${ageSuffix}`;
  }

  function buildSeenHistoryToggleLabels(card) {
    const count = Math.max(0, Number(card?.seenHistoryCount || 0));
    if (count <= 0) {
      return {
        seenHistoryToggleLabel: "",
        seenHistoryToggleHideLabel: "",
      };
    }
    const hasForegroundAttention = Boolean(
      (card?.tone && card.tone !== "ok")
      && (Number(card?.unseenCount || 0) > 0 || `${card?.freshness?.label || ""}`.trim() === "fresh")
    );
    const base = hasForegroundAttention ? "background seen history" : "seen history";
    return {
      seenHistoryToggleLabel: `Show ${base} (${count})`,
      seenHistoryToggleHideLabel: `Hide ${base}`,
    };
  }

  function latestRuntimeAction(events) {
    const runtimeEvents = (Array.isArray(events) ? events : [])
      .filter((event) => `${event?.type || event?.event_type || ""}`.trim() === "runtime.service_action");
    if (!runtimeEvents.length) {
      return null;
    }
    const sorted = runtimeEvents.slice().sort((left, right) => {
      const leftTs = timestampValue(left?.created_at);
      const rightTs = timestampValue(right?.created_at);
      return rightTs - leftTs;
    });
    const event = sorted[0];
    const payload = event?.payload && typeof event.payload === "object" ? event.payload : {};
    return {
      service: `${payload.service || "service"}`.trim() || "service",
      action: `${payload.action || "action"}`.trim() || "action",
      success: Boolean(payload.success),
      createdAt: `${event?.created_at || ""}`.trim(),
      message: `${event?.message || ""}`.trim(),
    };
  }

  function normalizeRuntime(data) {
    const runtimeStatus = data && typeof data.runtimeStatus === "object" ? data.runtimeStatus : {};
    const runtimeServices = Array.isArray(data?.runtimeServices) ? data.runtimeServices : [];
    const runningCount = runtimeServices.filter((service) => service && service.running).length;
    const serviceCount = runtimeServices.length;
    const connected = Boolean(runtimeStatus.connected);
    const source = `${runtimeStatus.source || "none"}`.trim() || "none";
    const latestAction = latestRuntimeAction(data?.events);
    const freshness = describeFreshness(
      latestTimestamp([
        latestAction?.createdAt,
        ...((Array.isArray(data?.events) ? data.events : [])
          .filter((event) => deriveEventDomain(event?.type || event?.event_type || "event") === "runtime")
          .map((event) => event?.created_at)),
      ]),
      data?.now,
    );
    const actionSummary = latestAction
      ? `Latest action ${latestAction.success ? "succeeded" : "failed"}: ${latestAction.action} ${latestAction.service}`
      : "";
    const remediationHint = latestAction && !latestAction.success
      ? `Inspect ${latestAction.service} logs and retry ${latestAction.action}.`
      : "";
    const seenHistoryEvents = selectSeenEvents(data?.events, data?.seenEventIds, ["runtime"]);
    const seenHistoryCount = countSeenEvents(data?.events, data?.seenEventIds, ["runtime"]);

    if (!connected) {
      const baseCard = {
        id: "runtime",
        title: "Runtime",
        tone: "err",
        metric: "offline",
        summary: `runtime source ${source}`,
        detail: runtimeStatus.error || "runtime control plane is unavailable",
        attentionDomains: ["runtime"],
        focus: {
          panel: "runtimePanel",
        },
        actionSummary,
        remediationHint,
        freshness,
        unseenCount: countUnseenEvents(data?.events, data?.seenEventIds, ["runtime"]),
        seenHistoryCount,
        attentionEvents: selectUnseenEvents(data?.events, data?.seenEventIds, ["runtime"]),
        seenHistoryEvents,
        seenHistorySummary: buildSeenHistorySummary(["runtime"], seenHistoryEvents, seenHistoryCount, data?.now),
      };
      return {
        ...baseCard,
        ...buildSeenHistoryToggleLabels(baseCard),
      };
    }

    const tone = latestAction && !latestAction.success
      ? "warn"
      : serviceCount > 0 && runningCount < serviceCount
      ? "warn"
      : "ok";
    const baseCard = {
      id: "runtime",
      title: "Runtime",
      tone,
      metric: `${runningCount}/${serviceCount || 0}`,
      summary: `services running via ${source}`,
      detail: runtimeStatus.orchestratorLine || `default service ${runtimeStatus.defaultService || "busy"}`,
      attentionDomains: ["runtime"],
      focus: {
        panel: "runtimePanel",
      },
      actionSummary,
      remediationHint,
      freshness,
      unseenCount: countUnseenEvents(data?.events, data?.seenEventIds, ["runtime"]),
      seenHistoryCount,
      attentionEvents: selectUnseenEvents(data?.events, data?.seenEventIds, ["runtime"]),
      seenHistoryEvents,
      seenHistorySummary: buildSeenHistorySummary(["runtime"], seenHistoryEvents, seenHistoryCount, data?.now),
    };
    return {
      ...baseCard,
      ...buildSeenHistoryToggleLabels(baseCard),
    };
  }

  function normalizeProviders(data) {
    const summaryApi = providerRoutingSummaryApi();
    const healthApi = providerHealthUiApi();
    const actionApi = providerActionUiApi();
    const providers = summaryApi
      ? summaryApi.buildProviderRoutingSummary(data?.providerChain)
      : {
        tone: "err",
        metric: "0",
        summary: "no enabled providers",
        detail: "routing chain helper unavailable",
        focusProviderStatus: "",
      };
    const providerRecords = Array.isArray(data?.providers) ? data.providers.filter((provider) => provider && provider.enabled) : [];
    const providersById = new Map(providerRecords.map((provider) => [`${provider.id || ""}`, provider]));
    const actionResultsRaw = data?.providerActionResults && typeof data.providerActionResults === "object"
      ? data.providerActionResults
      : {};
    const actionResults = new Map(
      Object.entries(actionResultsRaw).map(([providerId, raw]) => [
        `${providerId || ""}`,
        actionApi ? actionApi.normalizeProviderActionResult(raw) : raw,
      ])
    );
    const chain = Array.isArray(data?.providerChain?.chain) ? data.providerChain.chain : [];
    const orderedEntries = chain.length > 0
      ? chain
        .map((node, index) => ({
          provider: providersById.get(`${node?.id || ""}`) || null,
          routingNode: node,
          position: Number.isFinite(Number(node?.position)) ? Number(node.position) : index,
        }))
        .filter((entry) => entry.provider)
      : providerRecords.map((provider, index) => ({
        provider,
        routingNode: null,
        position: index,
      }));
    const remediationEntries = healthApi
      ? orderedEntries
        .map((entry) => ({
          ...entry,
          actionResult: actionResults.get(`${entry.provider?.id || entry.routingNode?.id || ""}`) || null,
          summary: healthApi.buildProviderHealthSummary({
            provider: entry.provider,
            metadata: entry.provider?.metadata,
            routingNode: entry.routingNode,
            status: entry.provider?.status || entry.routingNode?.status || "",
          }),
        }))
        .filter((entry) => entry.summary && entry.summary.tone !== "ok")
        .sort((left, right) => {
          const leftActionSeverity = left.actionResult?.tone === "err" ? 0 : left.actionResult?.tone === "warn" ? 1 : 2;
          const rightActionSeverity = right.actionResult?.tone === "err" ? 0 : right.actionResult?.tone === "warn" ? 1 : 2;
          if (leftActionSeverity !== rightActionSeverity) {
            return leftActionSeverity - rightActionSeverity;
          }
          const leftActionTs = timestampValue(left.actionResult?.recordedAt);
          const rightActionTs = timestampValue(right.actionResult?.recordedAt);
          if (leftActionTs !== rightActionTs) {
            return rightActionTs - leftActionTs;
          }
          const leftSeverity = left.summary.tone === "err" ? 0 : 1;
          const rightSeverity = right.summary.tone === "err" ? 0 : 1;
          if (leftSeverity !== rightSeverity) {
            return leftSeverity - rightSeverity;
          }
          const leftActive = left.routingNode?.active ? 0 : 1;
          const rightActive = right.routingNode?.active ? 0 : 1;
          if (leftActive !== rightActive) {
            return leftActive - rightActive;
          }
          return left.position - right.position;
        })
      : [];
    const remediationItems = remediationEntries.slice(0, 3).map((entry) => {
      const primaryIssue = Array.isArray(entry.summary?.issues) && entry.summary.issues.length > 0
        ? entry.summary.issues[0]
        : null;
      const actionResult = entry.actionResult && typeof entry.actionResult === "object"
        ? entry.actionResult
        : null;
      return {
        providerId: `${entry.provider?.id || entry.routingNode?.id || "provider"}`,
        providerStatus: `${entry.provider?.status || entry.routingNode?.status || ""}`.trim().toLowerCase(),
        tone: actionResult?.tone === "err"
          ? "err"
          : primaryIssue?.tone || entry.summary?.tone || actionResult?.tone || "warn",
        label: primaryIssue?.label || entry.summary?.summary || "provider needs review",
        detail: primaryIssue?.detail || entry.summary?.detail || "",
        action: primaryIssue?.action || entry.summary?.action || "",
        latestAction: actionResult ? `${actionResult.action}: ${actionResult.message}` : "",
        latestDetail: actionResult?.detail || "",
        latestTone: actionResult?.tone || "",
        latestRecordedAt: actionResult?.recordedAt || "",
      };
    });
    const latestActionFailure = remediationItems.find((item) => item.latestTone === "err" && item.latestAction);
    const latestActionSuccess = remediationItems.find((item) => item.latestTone === "ok" && item.latestAction);
    const freshness = describeFreshness(
      latestTimestamp([
        ...remediationItems.map((item) => item.latestRecordedAt),
        ...((Array.isArray(data?.events) ? data.events : [])
          .filter((event) => deriveEventDomain(event?.type || event?.event_type || "event") === "provider")
          .map((event) => event?.created_at)),
      ]),
      data?.now,
    );
    const remediationHint = latestActionFailure
      ? `${latestActionFailure.providerId}: latest action failed (${latestActionFailure.latestAction})`
      : remediationItems.length > 0
      ? `${remediationItems[0].providerId}: ${remediationItems[0].action}`
      : latestActionSuccess
      ? `${latestActionSuccess.providerId}: latest verification passed (${latestActionSuccess.latestAction})`
      : "";
    const remediationDetail = remediationItems.length > 0
      ? `${providers.detail} • Next: ${remediationItems[0].providerId} ${remediationItems[0].label}`
      : providers.detail;
    const actionSummary = latestActionFailure
      ? `Latest action failed for ${latestActionFailure.providerId}: ${latestActionFailure.latestAction}`
      : latestActionSuccess
      ? `Latest successful action: ${latestActionSuccess.providerId} ${latestActionSuccess.latestAction}`
      : "";

    const seenHistoryEvents = selectSeenEvents(data?.events, data?.seenEventIds, ["provider"]);
    const seenHistoryCount = countSeenEvents(data?.events, data?.seenEventIds, ["provider"]);
    const baseCard = {
      id: "providers",
      title: "Providers",
      tone: providers.tone,
      metric: providers.metric,
      summary: providers.summary,
      detail: remediationDetail,
      attentionDomains: ["provider"],
      focus: {
        panel: "providersPanel",
        providerStatus: providers.focusProviderStatus || "",
      },
      remediationHint,
      remediationItems,
      actionSummary,
      freshness,
      unseenCount: countUnseenEvents(data?.events, data?.seenEventIds, ["provider"]),
      seenHistoryCount,
      attentionEvents: selectUnseenEvents(data?.events, data?.seenEventIds, ["provider"]),
      seenHistoryEvents,
      seenHistorySummary: buildSeenHistorySummary(["provider"], seenHistoryEvents, seenHistoryCount, data?.now),
    };
    return {
      ...baseCard,
      ...buildSeenHistoryToggleLabels(baseCard),
    };
  }

  function normalizeGmTickets(data) {
    const tickets = Array.isArray(data?.gmTickets) ? data.gmTickets : [];
    const open = tickets.filter((ticket) => !["resolved", "closed", "archived", "complete"].includes(`${ticket.status || ""}`.toLowerCase()));
    const blocked = open.filter((ticket) => `${ticket.status || ""}`.toLowerCase() === "blocked");
    const critical = open.filter((ticket) => `${ticket.priority || ""}`.toLowerCase() === "critical");
    const high = open.filter((ticket) => `${ticket.priority || ""}`.toLowerCase() === "high");

    const tone = critical.length > 0 ? "err" : blocked.length > 0 || high.length > 0 ? "warn" : "ok";
    const seenHistoryEvents = selectSeenEvents(data?.events, data?.seenEventIds, ["gm_ticket"]);
    const seenHistoryCount = countSeenEvents(data?.events, data?.seenEventIds, ["gm_ticket"]);
    const baseCard = {
      id: "gm",
      title: "GM Tickets",
      tone,
      metric: `${open.length}`,
      summary: "open operator threads",
      detail: `${critical.length} critical • ${blocked.length} blocked • ${high.length} high priority`,
      attentionDomains: ["gm_ticket"],
      focus: {
        panel: "gmTicketsPanel",
        gmPriority: critical.length > 0 ? "critical" : "",
        gmStatus: critical.length > 0 ? "" : blocked.length > 0 ? "blocked" : "",
      },
      freshness: describeFreshness(
        latestTimestamp(open.map((ticket) => ticket?.created_at || ticket?.createdAt)),
        data?.now,
      ),
      unseenCount: countUnseenEvents(data?.events, data?.seenEventIds, ["gm_ticket"]),
      seenHistoryCount,
      attentionEvents: selectUnseenEvents(data?.events, data?.seenEventIds, ["gm_ticket"]),
      seenHistoryEvents,
      seenHistorySummary: buildSeenHistorySummary(["gm_ticket"], seenHistoryEvents, seenHistoryCount, data?.now),
    };
    return {
      ...baseCard,
      ...buildSeenHistoryToggleLabels(baseCard),
    };
  }

  function normalizeStartup(data) {
    const summaryEvent = data?.startupSummaryEvent && typeof data.startupSummaryEvent === "object"
      ? data.startupSummaryEvent
      : null;
    const payload = summaryEvent?.payload && typeof summaryEvent.payload === "object"
      ? summaryEvent.payload
      : {};
    const errors = Number(payload.error_count || 0);
    const warnings = Number(payload.warn_count || 0);
    const missingRequired = Number(payload.required_missing || 0);
    const disabledRequired = Number(payload.required_disabled || 0);
    const tone = errors > 0 || missingRequired > 0 ? "err" : warnings > 0 || disabledRequired > 0 ? "warn" : "ok";

    const seenHistoryEvents = selectSeenEvents(data?.events, data?.seenEventIds, ["plugin"]);
    const seenHistoryCount = countSeenEvents(data?.events, data?.seenEventIds, ["plugin"]);
    const baseCard = {
      id: "startup",
      title: "Startup Checks",
      tone,
      metric: `${warnings + errors}`,
      summary: "plugin debug warnings and errors",
      detail: `${errors} errors • ${warnings} warnings • ${missingRequired} missing required`,
      attentionDomains: ["plugin"],
      focus: {
        panel: "startupPanel",
      },
      freshness: describeFreshness(summaryEvent?.created_at, data?.now),
      unseenCount: countUnseenEvents(data?.events, data?.seenEventIds, ["plugin"]),
      seenHistoryCount,
      attentionEvents: selectUnseenEvents(data?.events, data?.seenEventIds, ["plugin"]),
      seenHistoryEvents,
      seenHistorySummary: buildSeenHistorySummary(["plugin"], seenHistoryEvents, seenHistoryCount, data?.now),
    };
    return {
      ...baseCard,
      ...buildSeenHistoryToggleLabels(baseCard),
    };
  }

  function toneRank(tone) {
    if (tone === "err") {
      return 0;
    }
    if (tone === "warn") {
      return 1;
    }
    return 2;
  }

  function freshnessRank(freshness) {
    const label = `${freshness?.label || ""}`.trim();
    if (label === "fresh") {
      return 0;
    }
    if (label === "recent") {
      return 1;
    }
    if (label === "older") {
      return 2;
    }
    return 3;
  }

  function buildReviewState(card, reviewedAt) {
    const reviewTimestamp = `${reviewedAt || ""}`.trim();
    const latestTimestamp = `${card?.freshness?.timestamp || ""}`.trim();
    const needsAttention = Boolean(
      (card?.tone && card.tone !== "ok")
      || Number(card?.unseenCount || 0) > 0
      || `${card?.freshness?.label || ""}`.trim()
    );
    if (!needsAttention) {
      return null;
    }
    if (!reviewTimestamp) {
      return {
        label: "unreviewed",
        tone: "warn",
        detail: "not yet reviewed in this browser",
        reviewedAt: "",
      };
    }
    const reviewedTs = timestampValue(reviewTimestamp);
    const latestTs = timestampValue(latestTimestamp);
    if (reviewedTs > 0 && latestTs > reviewedTs) {
      return {
        label: "updated",
        tone: "warn",
        detail: `newer signal since ${reviewTimestamp}`,
        reviewedAt: reviewTimestamp,
      };
    }
    return {
      label: "reviewed",
      tone: "ok",
      detail: `reviewed ${reviewTimestamp}`,
      reviewedAt: reviewTimestamp,
    };
  }

  function reviewStateRank(reviewState) {
    const label = `${reviewState?.label || ""}`.trim();
    if (label === "updated") {
      return 0;
    }
    if (label === "unreviewed") {
      return 1;
    }
    if (label === "reviewed") {
      return 2;
    }
    return 3;
  }

  function compareDashboardCards(left, right) {
    const toneDiff = toneRank(left?.tone) - toneRank(right?.tone);
    if (toneDiff !== 0) {
      return toneDiff;
    }
    const freshnessDiff = freshnessRank(left?.freshness) - freshnessRank(right?.freshness);
    if (freshnessDiff !== 0) {
      return freshnessDiff;
    }
    const reviewDiff = reviewStateRank(left?.reviewState) - reviewStateRank(right?.reviewState);
    if (reviewDiff !== 0) {
      return reviewDiff;
    }
    const unseenDiff = Number(right?.unseenCount || 0) - Number(left?.unseenCount || 0);
    if (unseenDiff !== 0) {
      return unseenDiff;
    }
    return Number(left?._order || 0) - Number(right?._order || 0);
  }

  function isAttentionCard(card) {
    return Boolean(
      (card?.tone && card.tone !== "ok")
      || Number(card?.unseenCount || 0) > 0
      || `${card?.freshness?.label || ""}`.trim() !== ""
      || `${card?.actionSummary || ""}`.trim() !== ""
    );
  }

  function buildDashboardViewModel(data) {
    const cardReviewTimestamps = data?.cardReviewTimestamps && typeof data.cardReviewTimestamps === "object"
      ? data.cardReviewTimestamps
      : {};
    const cards = [
      normalizeRuntime(data),
      normalizeProviders(data),
      normalizeGmTickets(data),
      normalizeStartup(data),
    ].map((card, index) => ({
      ...card,
      reviewState: buildReviewState(card, cardReviewTimestamps[card.id]),
      _order: index,
    }))
      .sort(compareDashboardCards)
      .map(({ _order, ...card }) => card);
    const errorCount = cards.filter((card) => card.tone === "err").length;
    const warnCount = cards.filter((card) => card.tone === "warn").length;
    const headline = errorCount > 0
      ? `${errorCount} critical area(s) need attention`
      : warnCount > 0
      ? `${warnCount} area(s) need review`
      : "Core control-plane surfaces look stable";
    const tone = errorCount > 0 ? "err" : warnCount > 0 ? "warn" : "ok";
    return {
      tone,
      headline,
      cards,
      defaultCardId: cards.find(isAttentionCard)?.id || cards[0]?.id || "",
    };
  }

  function renderDashboardCard(card) {
    const tone = `${card?.tone || "ok"}`.trim() || "ok";
    const focus = card?.focus && typeof card.focus === "object" ? card.focus : {};
    const unseenCount = Number(card?.unseenCount || 0);
    const unseenBadge = unseenCount > 0
      ? `<span class="status warn dashboard-unseen-count">${escapeHtml(String(unseenCount))} new</span>`
      : '<span class="status event-seen-badge dashboard-unseen-count">seen</span>';
    const selectedClass = card?.selected ? " is-selected" : "";
    const remediationHint = `${card?.remediationHint || ""}`.trim();
    const freshness = card?.freshness && typeof card.freshness === "object" ? card.freshness : {};
    const freshnessBadge = freshness.label
      ? `<span class="status ${escapeHtml(freshness.tone || "")} dashboard-freshness-badge">${escapeHtml(freshness.label)} ${escapeHtml(freshness.detail || "")}</span>`
      : "";
    const reviewState = card?.reviewState && typeof card.reviewState === "object" ? card.reviewState : {};
    const reviewBadge = reviewState.label
      ? `<span class="status ${escapeHtml(reviewState.tone || "")} dashboard-review-badge">${escapeHtml(reviewState.label)}</span>`
      : "";
    return `
      <button
        type="button"
        class="card dashboard-card dashboard-${escapeHtml(tone)}${selectedClass}"
        data-action="open-dashboard-focus"
        data-card-id="${escapeHtml(card?.id || "")}"
        data-panel-id="${escapeHtml(focus.panel || "")}"
        data-provider-status="${escapeHtml(focus.providerStatus || "")}"
        data-gm-status="${escapeHtml(focus.gmStatus || "")}"
        data-gm-priority="${escapeHtml(focus.gmPriority || "")}"
      >
        <p class="dashboard-kicker">${escapeHtml(card?.title || "Summary")} ${unseenBadge} ${freshnessBadge} ${reviewBadge}</p>
        <p class="dashboard-metric">${escapeHtml(card?.metric || "0")}</p>
        <p><strong>${escapeHtml(card?.summary || "")}</strong></p>
        <p class="meta">${escapeHtml(card?.detail || "")}</p>
        ${remediationHint ? `<p class="meta dashboard-remediation-hint"><strong>Next:</strong> ${escapeHtml(remediationHint)}</p>` : ""}
      </button>
    `;
  }

  global.Busy38DashboardOverview = {
    buildDashboardViewModel,
    buildReviewState,
    renderDashboardCard,
  };
})(window);
