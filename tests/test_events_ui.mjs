import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadApi() {
  const source = fs.readFileSync(path.resolve("web/events_ui.js"), "utf8");
  const sandbox = {
    window: {},
  };
  vm.runInNewContext(source, sandbox, { filename: "events_ui.js" });
  return sandbox.window.Busy38EventsUi;
}

test("events ui helper renders runtime service action details", () => {
  const api = loadApi();
  const view = api.buildEventViewModel([{
    id: "evt-runtime-1",
    type: "runtime.service_action",
    created_at: "2026-03-15T19:00:00Z",
    level: "warn",
    message: "service 'missing' not found",
    payload: {
      service: "missing",
      action: "start",
      success: false,
      runtime_source: "mock",
      runtime_connected: true,
      actor: "admin",
    },
  }], { seenEventIds: [] });
  const html = api.renderEventItem(view.events[0]);

  assert.match(html, /runtime\.service_action/);
  assert.match(html, /Service:<\/strong> missing/);
  assert.match(html, /Action:<\/strong> start/);
  assert.match(html, /Result:<\/strong> failed/);
  assert.match(html, /Runtime source:<\/strong> mock/);
  assert.match(html, /new/);
});

test("events ui helper renders a mark-seen action only for unseen events when requested", () => {
  const api = loadApi();
  const unseenView = api.buildEventViewModel([{
    id: "evt-runtime-2",
    type: "runtime.service_action",
    created_at: "2026-03-15T19:05:00Z",
    level: "warn",
    message: "restart failed",
    payload: {
      service: "busy",
      action: "restart",
      success: false,
    },
  }], { seenEventIds: [] });
  const unseenHtml = api.renderEventItem(unseenView.events[0], {
    actionName: "mark-dashboard-attention-event-seen",
    actionLabel: "Mark event seen",
    eventData: {
      cardId: "runtime",
    },
  });
  assert.match(unseenHtml, /data-action="mark-dashboard-attention-event-seen"/);
  assert.match(unseenHtml, /data-event-id="evt-runtime-2"/);
  assert.match(unseenHtml, /data-card-id="runtime"/);
  assert.match(unseenHtml, /Mark event seen/);

  const seenView = api.buildEventViewModel([{
    id: "evt-runtime-2",
    type: "runtime.service_action",
    created_at: "2026-03-15T19:05:00Z",
    level: "warn",
    message: "restart failed",
    payload: {
      service: "busy",
      action: "restart",
      success: false,
    },
  }], { seenEventIds: ["evt-runtime-2"] });
  const seenHtml = api.renderEventItem(seenView.events[0], {
    actionName: "mark-dashboard-attention-event-seen",
    actionLabel: "Mark event seen",
  });
  assert.doesNotMatch(seenHtml, /mark-dashboard-attention-event-seen/);
});

test("events ui helper filters and summarizes event domains", () => {
  const api = loadApi();
  const view = api.buildEventViewModel(
    [
      {
        type: "runtime.service_action",
        level: "info",
        message: "restart queued",
        payload: { service: "busy" },
      },
      {
        type: "gm_ticket.updated",
        level: "warn",
        message: "ticket blocked",
        payload: { gm_ticket_id: "gm-1" },
      },
      {
        type: "runtime.service_action",
        level: "warn",
        message: "service missing",
        payload: { service: "missing" },
      },
    ],
    {
      domain: "runtime",
      level: "warn",
      query: "missing",
      limit: 10,
    },
  );

  assert.equal(api.deriveEventDomain("runtime.service_action"), "runtime");
  assert.equal(view.summary.total, 3);
  assert.equal(view.summary.domains.runtime, 2);
  assert.equal(view.summary.domains.gm_ticket, 1);
  assert.equal(view.filteredCount, 1);
  assert.equal(view.unseenCount, 1);
  assert.equal(view.events[0].message, "service missing");
});

test("index loads the events ui helper before the main app bundle", () => {
  const html = fs.readFileSync(path.resolve("web/index.html"), "utf8");
  const helperScript = '<script src="events_ui.js"></script>';
  const appScript = '<script src="app.js"></script>';

  const helperIndex = html.indexOf(helperScript);
  const appIndex = html.indexOf(appScript);

  assert.notEqual(helperIndex, -1, "expected events_ui.js script tag");
  assert.notEqual(appIndex, -1, "expected app.js script tag");
  assert.ok(helperIndex < appIndex, "expected events_ui.js to load before app.js");
});
