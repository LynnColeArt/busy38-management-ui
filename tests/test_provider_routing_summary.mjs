import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadApi() {
  const source = fs.readFileSync(path.resolve("web/provider_routing_summary.js"), "utf8");
  const sandbox = {
    window: {},
  };
  vm.runInNewContext(source, sandbox, { filename: "provider_routing_summary.js" });
  return sandbox.window.Busy38ProviderRoutingSummary;
}

test("provider routing summary reports healthy routing when every enabled provider is active", () => {
  const api = loadApi();
  const view = api.buildProviderRoutingSummary({
    active_provider_id: "primary",
    chain: [
      { id: "primary", status: "active", enabled: true, active: true },
      { id: "backup", status: "active", enabled: true, active: false },
    ],
  });

  assert.equal(view.state, "healthy");
  assert.equal(view.tone, "ok");
  assert.equal(view.metric, "2/2");
  assert.equal(view.summary, "routing healthy");
  assert.equal(view.focusProviderStatus, "");
});

test("provider routing summary reports available but degraded routing when fallbacks are impaired", () => {
  const api = loadApi();
  const view = api.buildProviderRoutingSummary({
    active_provider_id: "primary",
    chain: [
      { id: "primary", status: "active", enabled: true, active: true },
      { id: "backup", status: "unreachable", enabled: true, active: false },
      { id: "third", status: "degraded", enabled: true, active: false },
    ],
  });

  assert.equal(view.state, "degraded");
  assert.equal(view.tone, "warn");
  assert.equal(view.metric, "2/3");
  assert.equal(view.summary, "routing available but degraded");
  assert.equal(view.focusProviderStatus, "unreachable");
  assert.match(view.detail, /primary primary is active/);
});

test("provider routing summary reports blocked routing when all enabled providers are unreachable", () => {
  const api = loadApi();
  const view = api.buildProviderRoutingSummary({
    active_provider_id: "primary",
    chain: [
      { id: "primary", status: "unreachable", enabled: true, active: true },
      { id: "backup", status: "unreachable", enabled: true, active: false },
    ],
  });

  assert.equal(view.state, "blocked");
  assert.equal(view.tone, "err");
  assert.equal(view.metric, "0/2");
  assert.equal(view.summary, "routing blocked");
  assert.equal(view.focusProviderStatus, "unreachable");
});
