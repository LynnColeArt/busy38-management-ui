import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadApi() {
  const source = fs.readFileSync(path.resolve("web/provider_health_ui.js"), "utf8");
  const sandbox = {
    window: {},
  };
  vm.runInNewContext(source, sandbox, { filename: "provider_health_ui.js" });
  return sandbox.window.Busy38ProviderHealthUi;
}

test("provider health helper flags missing secrets and failed tests as intervention", () => {
  const api = loadApi();
  const view = api.buildProviderHealthSummary({
    provider: {
      id: "openai-primary",
      status: "active",
      metadata: {
        secret_policy: "required",
        secret_present: false,
      },
    },
    last_test: {
      status: "fail",
      error: "401 unauthorized",
    },
  });

  assert.equal(view.tone, "err");
  assert.equal(view.summary, "provider needs intervention");
  assert.match(view.detail, /missing required secret/);
  assert.match(view.detail, /last provider test failed/);
  assert.match(view.action, /Set the provider secret/);
});

test("provider health helper flags standby and recent failures as review", () => {
  const api = loadApi();
  const view = api.buildProviderHealthSummary({
    provider: {
      id: "ollama-secondary",
      status: "standby",
      metadata: {
        secret_policy: "none",
        secret_present: false,
      },
    },
    last_test: {
      status: "pass",
      tested_at: "2026-03-15T19:00:00Z",
    },
    metrics: {
      failure_count_last_1m: 3,
    },
  });

  assert.equal(view.tone, "warn");
  assert.equal(view.summary, "provider needs review");
  assert.match(view.detail, /standby fallback only/);
  assert.match(view.detail, /recent request failures/);
});

test("provider health helper renders issue list literally", () => {
  const api = loadApi();
  const html = api.renderProviderHealthSummary({
    tone: "warn",
    summary: "provider needs review",
    detail: "standby fallback only",
    action: "Promote priority or mark active if this provider should take live traffic.",
    issues: [
      {
        tone: "warn",
        label: "standby fallback only",
        detail: "Provider is enabled for fallback but is not carrying primary routing traffic.",
        action: "Promote priority or mark active if this provider should take live traffic.",
      },
    ],
  });

  assert.match(html, /Operator summary:/);
  assert.match(html, /provider needs review/);
  assert.match(html, /standby fallback only/);
  assert.match(html, /Next action:/);
});

test("index loads the provider health helper before the main app bundle", () => {
  const html = fs.readFileSync(path.resolve("web/index.html"), "utf8");
  const helperScript = '<script src="provider_health_ui.js"></script>';
  const appScript = '<script src="app.js"></script>';

  const helperIndex = html.indexOf(helperScript);
  const appIndex = html.indexOf(appScript);

  assert.notEqual(helperIndex, -1, "expected provider_health_ui.js script tag");
  assert.notEqual(appIndex, -1, "expected app.js script tag");
  assert.ok(helperIndex < appIndex, "expected provider_health_ui.js to load before app.js");
});
