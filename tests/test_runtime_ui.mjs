import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadApi() {
  const source = fs.readFileSync(path.resolve("web/runtime_ui.js"), "utf8");
  const sandbox = {
    window: {},
  };
  vm.runInNewContext(source, sandbox, { filename: "runtime_ui.js" });
  return sandbox.window.Busy38RuntimeUi;
}

test("runtime ui helper builds a source-aware runtime summary", () => {
  const api = loadApi();
  const view = api.buildRuntimeViewModel(
    {
      source: "direct",
      connected: true,
      default_service: "busy",
      orchestrator: { status: "running" },
    },
    {
      services: [
        { name: "busy", running: true, pid: 1001, pid_file: "/tmp/busy.pid", log_file: "/tmp/busy.log" },
        { name: "worker", running: false, pid: null, pid_file: "/tmp/worker.pid", log_file: "/tmp/worker.log" },
      ],
    },
  );

  assert.equal(view.summary.defaultService, "busy");
  assert.equal(view.summary.serviceCount, 2);
  assert.equal(view.summary.runningCount, 1);
  assert.equal(view.summary.statusKind, "ok");
  assert.match(view.summary.statusLine, /runtime \(direct\) - connected - 1\/2 running/);
  assert.match(view.summary.metaLine, /default service: busy/);
  assert.match(view.summary.metaLine, /orchestrator: running/);
});

test("runtime ui helper renders admin-disabled service cards literally", () => {
  const api = loadApi();
  const html = api.renderServiceCard(
    {
      name: "busy<main>",
      running: false,
      pid: null,
      pid_file: "/tmp/busy.pid",
      log_file: "/tmp/busy.log",
    },
    { canControl: false },
  );

  assert.match(html, /status-stopped/);
  assert.match(html, /busy&lt;main&gt;/);
  assert.match(html, /data-action="runtime-service-start"/);
  assert.match(html, /disabled/);
});

test("index loads the runtime ui helper before the main app bundle", () => {
  const html = fs.readFileSync(path.resolve("web/index.html"), "utf8");
  const helperScript = '<script src="runtime_ui.js"></script>';
  const appScript = '<script src="app.js"></script>';

  const helperIndex = html.indexOf(helperScript);
  const appIndex = html.indexOf(appScript);

  assert.notEqual(helperIndex, -1, "expected runtime_ui.js script tag");
  assert.notEqual(appIndex, -1, "expected app.js script tag");
  assert.ok(helperIndex < appIndex, "expected runtime_ui.js to load before app.js");
});
