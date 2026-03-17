import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadHelper() {
  const source = fs.readFileSync(path.resolve("web/plugin_ui_console.js"), "utf8");
  const warnCalls = [];
  const errorCalls = [];
  const context = {
    console: {
      warn: (...args) => warnCalls.push(args),
      error: (...args) => errorCalls.push(args),
    },
  };
  context.window = context;
  vm.createContext(context);
  vm.runInContext(source, context, { filename: "web/plugin_ui_console.js" });
  return {
    helper: context.Busy38PluginUiConsole,
    warnCalls,
    errorCalls,
  };
}

test("logDiagnostics emits console warnings and errors from debugger payloads", () => {
  const { helper, warnCalls, errorCalls } = loadHelper();
  helper.logDiagnostics("busy-38-discord", {
    status: "error",
    updated_at: "2026-03-05T02:15:00Z",
    warnings: {
      entries: [{ code: "P_PLUGIN_UI_HANDLER_MISSING", message: "handler missing" }],
    },
    errors: {
      entries: [{ code: "P_PLUGIN_RUNTIME_DEBUG_FAILED", message: "debug failed" }],
    },
  });

  assert.equal(warnCalls.length, 1);
  assert.equal(warnCalls[0][0], "[plugin-ui] diagnostics warnings");
  assert.equal(warnCalls[0][1].pluginId, "busy-38-discord");
  assert.equal(warnCalls[0][1].warnings[0].code, "P_PLUGIN_UI_HANDLER_MISSING");

  assert.equal(errorCalls.length, 1);
  assert.equal(errorCalls[0][0], "[plugin-ui] diagnostics errors");
  assert.equal(errorCalls[0][1].errors[0].code, "P_PLUGIN_RUNTIME_DEBUG_FAILED");
});

test("logActionResult emits warning telemetry for successful actions with warning metadata", () => {
  const { helper, warnCalls, errorCalls } = loadHelper();
  helper.logActionResult("busy-38-discord", "debug", {
    updated_at: "2026-03-05T02:16:00Z",
    result: {
      success: true,
      message: "debug completed with warnings",
      payload: {
        warnings: [{ code: "P_PLUGIN_UI_ASSET_MISSING", message: "ui asset missing" }],
        reason_codes: ["P_PLUGIN_UI_ASSET_MISSING"],
      },
    },
  });

  assert.equal(errorCalls.length, 0);
  assert.equal(warnCalls.length, 1);
  assert.equal(warnCalls[0][0], "[plugin-ui] action warnings");
  assert.equal(warnCalls[0][1].actionId, "debug");
  assert.deepEqual(Array.from(warnCalls[0][1].reasonCodes), ["P_PLUGIN_UI_ASSET_MISSING"]);
});

test("logActionResult emits warning telemetry for top-level reason codes", () => {
  const { helper, warnCalls, errorCalls } = loadHelper();
  helper.logActionResult("busy-38-discord", "validate", {
    updated_at: "2026-03-07T05:16:00Z",
    result: {
      success: true,
      message: "validation completed with warnings",
      reason_codes: ["DISCORD_SCOPE_CUSTOM_EMPTY"],
      warnings: {
        entries: [{ code: "DISCORD_SCOPE_CUSTOM_EMPTY", message: "custom scope is empty" }],
      },
      payload: {
        policy_preview: {},
      },
    },
  });

  assert.equal(errorCalls.length, 0);
  assert.equal(warnCalls.length, 1);
  assert.equal(warnCalls[0][0], "[plugin-ui] action warnings");
  assert.deepEqual(Array.from(warnCalls[0][1].reasonCodes), ["DISCORD_SCOPE_CUSTOM_EMPTY"]);
  assert.equal(warnCalls[0][1].warnings[0].code, "DISCORD_SCOPE_CUSTOM_EMPTY");
});

test("logActionResult emits console errors for handler failures", () => {
  const { helper, warnCalls, errorCalls } = loadHelper();
  helper.logActionResult("busy-38-discord", "debug", {
    result: {
      success: false,
      message: "debug failed",
      errors: ["handler exploded"],
      payload: {
        reason_codes: ["P_PLUGIN_RUNTIME_DEBUG_FAILED"],
      },
    },
  });

  assert.equal(warnCalls.length, 0);
  assert.equal(errorCalls.length, 1);
  assert.equal(errorCalls[0][0], "[plugin-ui] action failed");
  assert.equal(errorCalls[0][1].message, "debug failed");
  assert.deepEqual(Array.from(errorCalls[0][1].reasonCodes), ["P_PLUGIN_RUNTIME_DEBUG_FAILED"]);
  assert.deepEqual(Array.from(errorCalls[0][1].errors), ["handler exploded"]);
});

test("logActionRequestFailure emits console errors for transport failures", () => {
  const { helper, errorCalls } = loadHelper();
  helper.logActionRequestFailure("busy-38-discord", "debug", new Error("500 Internal Server Error"));

  assert.equal(errorCalls.length, 1);
  assert.equal(errorCalls[0][0], "[plugin-ui] action request failed");
  assert.equal(errorCalls[0][1].actionId, "debug");
  assert.equal(errorCalls[0][1].message, "500 Internal Server Error");
});

test("index loads the plugin console helper before the main app bundle", () => {
  const html = fs.readFileSync(path.resolve("web/index.html"), "utf8");
  assert.match(
    html,
    /<script src="plugin_ui_console\.js"><\/script>[\s\S]*<script src="app\.js"><\/script>/,
  );
});
