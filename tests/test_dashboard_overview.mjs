import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadApi() {
  const providerActionSource = fs.readFileSync(path.resolve("web/provider_action_ui.js"), "utf8");
  const providerHealthSource = fs.readFileSync(path.resolve("web/provider_health_ui.js"), "utf8");
  const providerSource = fs.readFileSync(path.resolve("web/provider_routing_summary.js"), "utf8");
  const source = fs.readFileSync(path.resolve("web/dashboard_overview.js"), "utf8");
  const sandbox = {
    window: {},
  };
  vm.runInNewContext(providerActionSource, sandbox, { filename: "provider_action_ui.js" });
  vm.runInNewContext(providerHealthSource, sandbox, { filename: "provider_health_ui.js" });
  vm.runInNewContext(providerSource, sandbox, { filename: "provider_routing_summary.js" });
  vm.runInNewContext(source, sandbox, { filename: "dashboard_overview.js" });
  return sandbox.window.Busy38DashboardOverview;
}

test("dashboard overview helper highlights critical runtime, provider, and gm pressure", () => {
  const api = loadApi();
  const view = api.buildDashboardViewModel({
    now: "2026-03-15T22:30:00Z",
    runtimeStatus: {
      connected: false,
      source: "none",
      error: "runtime control unavailable",
    },
    runtimeServices: [],
    providers: [
      {
        id: "primary",
        status: "active",
        enabled: true,
        metadata: {
          secret_policy: "required",
          secret_present: false,
          last_test: {
            status: "fail",
            error: "401 unauthorized",
          },
        },
      },
      {
        id: "backup",
        status: "unreachable",
        enabled: true,
        metadata: {
          secret_policy: "none",
          secret_present: false,
          last_test: {
            status: "fail",
            error: "dial tcp timeout",
          },
        },
      },
    ],
    providerChain: {
      chain: [
        { id: "primary", status: "active", enabled: true, active: true, position: 0 },
        { id: "backup", status: "unreachable", enabled: true, active: false, position: 1 },
      ],
    },
    providerActionResults: {
      backup: {
        providerId: "backup",
        action: "secret rotation",
        tone: "err",
        message: "Secret update failed",
        detail: "403 forbidden",
        recordedAt: "2026-03-15T22:20:00Z",
      },
      primary: {
        providerId: "primary",
        action: "provider test",
        tone: "ok",
        message: "Provider test passed in 180ms",
        detail: "2 model(s) returned",
        recordedAt: "2026-03-15T22:19:00Z",
      },
    },
    gmTickets: [
      { id: "gm-1", status: "blocked", priority: "critical" },
      { id: "gm-2", status: "open", priority: "high" },
    ],
    events: [
      {
        id: "evt-runtime",
        type: "runtime.service_action",
        created_at: "2026-03-15T22:21:00Z",
        message: "restart failed for busy",
        payload: {
          service: "busy",
          action: "restart",
          success: false,
        },
      },
      { id: "evt-runtime-seen", type: "runtime.service_action", created_at: "2026-03-15T22:05:00Z" },
      { id: "evt-provider", type: "provider.updated", created_at: "2026-03-15T22:10:00Z" },
      { id: "evt-gm", type: "gm_ticket.updated" },
      { id: "evt-plugin", type: "plugin.startup_debug_summary" },
    ],
    seenEventIds: ["evt-provider", "evt-runtime-seen"],
    startupSummaryEvent: {
      payload: {
        error_count: 1,
        warn_count: 2,
        required_missing: 1,
      },
    },
  });

  assert.equal(view.tone, "err");
  assert.match(view.headline, /critical area/);
  assert.equal(view.defaultCardId, "runtime");
  assert.equal(view.cards[0].id, "runtime");
  const runtimeCard = view.cards.find((card) => card.id === "runtime");
  const providerCard = view.cards.find((card) => card.id === "providers");
  const gmCard = view.cards.find((card) => card.id === "gm");
  const startupCard = view.cards.find((card) => card.id === "startup");
  assert.ok(runtimeCard);
  assert.ok(providerCard);
  assert.ok(gmCard);
  assert.ok(startupCard);
  assert.equal(runtimeCard.title, "Runtime");
  assert.equal(runtimeCard.tone, "err");
  assert.equal(runtimeCard.focus.panel, "runtimePanel");
  assert.match(runtimeCard.actionSummary, /Latest action failed: restart busy/);
  assert.match(runtimeCard.remediationHint, /Inspect busy logs and retry restart/);
  assert.equal(runtimeCard.freshness.label, "fresh");
  assert.equal(runtimeCard.freshness.detail, "9m ago");
  assert.equal(runtimeCard.unseenCount, 1);
  assert.equal(runtimeCard.attentionEvents[0].id, "evt-runtime");
  assert.equal(runtimeCard.seenHistoryCount, 1);
  assert.equal(runtimeCard.seenHistoryToggleLabel, "Show background seen history (1)");
  assert.equal(runtimeCard.seenHistoryToggleHideLabel, "Hide background seen history");
  assert.equal(providerCard.tone, "warn");
  assert.equal(providerCard.summary, "routing available but degraded");
  assert.equal(providerCard.metric, "1/2");
  assert.equal(providerCard.focus.panel, "providersPanel");
  assert.equal(providerCard.focus.providerStatus, "unreachable");
  assert.match(providerCard.detail, /Next: backup provider unreachable/);
  assert.match(providerCard.remediationHint, /backup: latest action failed/);
  assert.match(providerCard.actionSummary, /Latest action failed for backup/);
  assert.equal(providerCard.freshness.label, "fresh");
  assert.equal(providerCard.remediationItems[0].providerId, "backup");
  assert.equal(providerCard.remediationItems[0].providerStatus, "unreachable");
  assert.match(providerCard.remediationItems[0].latestAction, /secret rotation: Secret update failed/);
  assert.equal(providerCard.remediationItems[1].providerId, "primary");
  assert.equal(providerCard.remediationItems[1].providerStatus, "active");
  assert.equal(providerCard.unseenCount, 0);
  assert.equal(providerCard.seenHistoryCount, 1);
  assert.equal(providerCard.seenHistoryEvents[0].id, "evt-provider");
  assert.equal(providerCard.seenHistorySummary, "showing 1 recent seen provider event, oldest shown 20m ago");
  assert.equal(providerCard.seenHistoryToggleLabel, "Show background seen history (1)");
  assert.equal(gmCard.tone, "err");
  assert.equal(gmCard.focus.panel, "gmTicketsPanel");
  assert.equal(gmCard.focus.gmPriority, "critical");
  assert.equal(gmCard.unseenCount, 1);
  assert.equal(startupCard.tone, "err");
  assert.equal(startupCard.unseenCount, 1);
});

test("dashboard overview helper renders summary cards literally", () => {
  const api = loadApi();
  const html = api.renderDashboardCard({
    title: "Providers",
    tone: "warn",
    metric: "2/3",
    summary: "routing available but degraded",
    detail: "1 degraded • 0 unreachable",
    remediationHint: "primary: Set the provider secret, then rerun the provider test.",
    freshness: { tone: "warn", label: "recent", detail: "2h ago" },
    reviewState: { tone: "ok", label: "reviewed" },
    unseenCount: 2,
  });

  assert.match(html, /dashboard-warn/);
  assert.match(html, /Providers/);
  assert.match(html, /routing available but degraded/);
  assert.match(html, /Next:/);
  assert.match(html, /Set the provider secret/);
  assert.match(html, /recent 2h ago/);
  assert.match(html, /reviewed/);
  assert.match(html, /data-action="open-dashboard-focus"/);
  assert.match(html, /data-card-id/);
  assert.match(html, /2 new/);
});

test("dashboard overview helper deprioritizes reviewed cards when severity and freshness tie", () => {
  const api = loadApi();
  const view = api.buildDashboardViewModel({
    now: "2026-03-15T22:30:00Z",
    runtimeStatus: {
      connected: true,
      source: "busy-runtime",
      orchestratorLine: "control plane connected",
    },
    runtimeServices: [
      { name: "busy", running: true },
    ],
    providers: [
      {
        id: "primary",
        status: "active",
        enabled: true,
        metadata: {
          secret_policy: "required",
          secret_present: true,
        },
      },
      {
        id: "backup",
        status: "unreachable",
        enabled: true,
        metadata: {
          secret_policy: "required",
          secret_present: true,
          last_test: {
            status: "fail",
            error: "dial tcp timeout",
          },
        },
      },
    ],
    providerChain: {
      chain: [
        { id: "primary", status: "active", enabled: true, active: true, position: 0 },
        { id: "backup", status: "unreachable", enabled: true, active: false, position: 1 },
      ],
    },
    providerActionResults: {
      backup: {
        providerId: "backup",
        action: "provider test",
        tone: "err",
        message: "Provider test failed",
        detail: "dial tcp timeout",
        recordedAt: "2026-03-15T22:22:00Z",
      },
    },
    events: [
      {
        id: "evt-runtime",
        type: "runtime.service_action",
        created_at: "2026-03-15T22:21:00Z",
        payload: {
          service: "busy",
          action: "restart",
          success: false,
        },
      },
    ],
    seenEventIds: ["evt-runtime"],
    cardReviewTimestamps: {
      runtime: "2026-03-15T22:25:00Z",
    },
  });

  const runtimeCard = view.cards.find((card) => card.id === "runtime");
  const providerCard = view.cards.find((card) => card.id === "providers");

  assert.ok(runtimeCard);
  assert.ok(providerCard);
  assert.equal(runtimeCard.tone, "warn");
  assert.equal(providerCard.tone, "warn");
  assert.equal(runtimeCard.freshness.label, "fresh");
  assert.equal(providerCard.freshness.label, "fresh");
  assert.equal(runtimeCard.reviewState.label, "reviewed");
  assert.equal(providerCard.reviewState.label, "unreviewed");
  assert.equal(view.cards[0].id, "providers");
  assert.equal(view.defaultCardId, "providers");
});

test("index loads the dashboard overview helper before the main app bundle", () => {
  const html = fs.readFileSync(path.resolve("web/index.html"), "utf8");
  const providerActionScript = '<script src="provider_action_ui.js"></script>';
  const providerHealthScript = '<script src="provider_health_ui.js"></script>';
  const providerHelperScript = '<script src="provider_routing_summary.js"></script>';
  const helperScript = '<script src="dashboard_overview.js"></script>';
  const appScript = '<script src="app.js"></script>';

  const providerActionIndex = html.indexOf(providerActionScript);
  const providerHealthIndex = html.indexOf(providerHealthScript);
  const providerHelperIndex = html.indexOf(providerHelperScript);
  const helperIndex = html.indexOf(helperScript);
  const appIndex = html.indexOf(appScript);

  assert.notEqual(providerActionIndex, -1, "expected provider_action_ui.js script tag");
  assert.notEqual(providerHealthIndex, -1, "expected provider_health_ui.js script tag");
  assert.notEqual(providerHelperIndex, -1, "expected provider_routing_summary.js script tag");
  assert.notEqual(helperIndex, -1, "expected dashboard_overview.js script tag");
  assert.notEqual(appIndex, -1, "expected app.js script tag");
  assert.ok(providerActionIndex < helperIndex, "expected provider_action_ui.js before dashboard_overview.js");
  assert.ok(providerHealthIndex < helperIndex, "expected provider_health_ui.js before dashboard_overview.js");
  assert.ok(providerHelperIndex < helperIndex, "expected provider_routing_summary.js before dashboard_overview.js");
  assert.ok(helperIndex < appIndex, "expected dashboard_overview.js to load before app.js");
});

test("dashboard attention remediation buttons preserve item-specific provider status", () => {
  const source = fs.readFileSync(path.resolve("web/app.js"), "utf8");

  assert.match(source, /item\.providerStatus \|\| focus\.providerStatus \|\| ""/);
  assert.match(source, /const primaryProviderStatus = remediationItems\[0\]\?\.providerStatus \|\| focus\.providerStatus \|\| "";/);
  assert.match(source, /data-provider-status="\$\{escapeHtml\(primaryProviderStatus\)\}"/);
});
