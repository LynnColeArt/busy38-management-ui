import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadApi() {
  const source = fs.readFileSync(path.resolve("web/appearance_theme.js"), "utf8");
  const sandbox = {
    window: {},
    document: { documentElement: { dataset: {} } },
  };
  vm.runInNewContext(source, sandbox, { filename: "appearance_theme.js" });
  return sandbox.window.Busy38AppearanceTheme;
}

test("appearance theme helper resolves desktop preference literally", () => {
  const api = loadApi();
  assert.equal(
    api.effectiveDesktopThemeValue({
      override_enabled: true,
      sync_theme_preferences: true,
      shared_theme_mode: "dark",
    }),
    "dark",
  );
  assert.equal(
    api.effectiveDesktopThemeValue({
      override_enabled: true,
      sync_theme_preferences: false,
      desktop_theme_mode: "light",
    }),
    "light",
  );
  assert.equal(api.effectiveDesktopThemeValue({ override_enabled: false }), "system");
  const normalized = api.normalizeAppearancePreferences({
    contrast_policy: "aaa",
    motion_policy: "reduced",
    color_separation_policy: "stronger",
    text_spacing_policy: "increased",
  });
  assert.equal(normalized.contrast_policy, "aaa");
  assert.equal(normalized.motion_policy, "reduced");
  assert.equal(normalized.color_separation_policy, "stronger");
  assert.equal(normalized.text_spacing_policy, "increased");
});

test("appearance theme helper applies resolved theme to document root", () => {
  const api = loadApi();
  const documentObject = { documentElement: { dataset: {} } };
  const windowObject = {
    matchMedia: () => ({ matches: false }),
  };
  const resolved = api.applyDocumentTheme(windowObject, documentObject, {
    override_enabled: true,
    sync_theme_preferences: true,
    shared_theme_mode: "dark",
  });
  assert.equal(resolved, "dark");
  assert.equal(documentObject.documentElement.dataset.theme, "dark");
  assert.equal(documentObject.documentElement.dataset.themePreference, "dark");
  assert.equal(documentObject.documentElement.dataset.contrastPolicy, "aa");
  assert.equal(documentObject.documentElement.dataset.motionPolicy, "default");
  assert.equal(documentObject.documentElement.dataset.colorSeparation, "default");
  assert.equal(documentObject.documentElement.dataset.textSpacing, "default");
});
