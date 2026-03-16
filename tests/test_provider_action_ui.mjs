import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadApi() {
  const source = fs.readFileSync(path.resolve("web/provider_action_ui.js"), "utf8");
  const sandbox = {
    window: {},
  };
  vm.runInNewContext(source, sandbox, { filename: "provider_action_ui.js" });
  return sandbox.window.Busy38ProviderActionUi;
}

test("provider action helper normalizes provider action payloads", () => {
  const api = loadApi();
  const view = api.normalizeProviderActionResult({
    providerId: "openai-primary",
    action: "provider test",
    tone: "ok",
    message: "Provider test passed in 210ms",
    detail: "2 model(s) returned",
    recordedAt: "2026-03-15T22:15:00Z",
  });

  assert.equal(view.providerId, "openai-primary");
  assert.equal(view.action, "provider test");
  assert.equal(view.tone, "ok");
  assert.equal(view.message, "Provider test passed in 210ms");
  assert.equal(view.detail, "2 model(s) returned");
});

test("provider action helper renders an action summary strip literally", () => {
  const api = loadApi();
  const html = api.renderProviderActionResult({
    providerId: "openai-primary",
    action: "secret rotation",
    tone: "err",
    message: "Secret update failed",
    detail: "403 forbidden",
    recordedAt: "2026-03-15T22:15:00Z",
  });

  assert.match(html, /Last action:/);
  assert.match(html, /secret rotation/);
  assert.match(html, /Secret update failed/);
  assert.match(html, /403 forbidden/);
  assert.match(html, /Recorded:/);
});

test("index loads the provider action helper before the main app bundle", () => {
  const html = fs.readFileSync(path.resolve("web/index.html"), "utf8");
  const helperScript = '<script src="provider_action_ui.js"></script>';
  const appScript = '<script src="app.js"></script>';

  const helperIndex = html.indexOf(helperScript);
  const appIndex = html.indexOf(appScript);

  assert.notEqual(helperIndex, -1, "expected provider_action_ui.js script tag");
  assert.notEqual(appIndex, -1, "expected app.js script tag");
  assert.ok(helperIndex < appIndex, "expected provider_action_ui.js to load before app.js");
});
