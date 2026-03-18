import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

function loadApi() {
  const source = fs.readFileSync(path.resolve("web/attention_state.js"), "utf8");
  const sandbox = {
    window: {},
  };
  vm.runInNewContext(source, sandbox, { filename: "attention_state.js" });
  return sandbox.window.Busy38AttentionState;
}

function makeStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, value);
    },
  };
}

test("attention state helper marks event ids as seen and persists them", () => {
  const api = loadApi();
  const storage = makeStorage();
  const next = api.markEventsSeen(storage, null, [
    { id: "evt-1" },
    { id: "evt-2" },
    { id: "evt-2" },
  ]);

  assert.equal(api.isEventSeen(next, "evt-1"), true);
  assert.equal(api.isEventSeen(next, "evt-2"), true);
  assert.equal(api.isEventSeen(next, "evt-3"), false);

  const reloaded = api.readSeenState(storage);
  assert.deepEqual(Array.from(reloaded.event_ids), ["evt-1", "evt-2"]);
});

test("attention state helper stores per-card review timestamps", () => {
  const api = loadApi();
  const storage = makeStorage();
  const next = api.markCardReviewed(storage, null, "runtime", "2026-03-16T01:00:00Z");

  assert.equal(api.getCardReviewedAt(next, "runtime"), "2026-03-16T01:00:00Z");
  assert.equal(api.getCardReviewedAt(next, "providers"), "");

  const reloaded = api.readSeenState(storage);
  assert.equal(api.getCardReviewedAt(reloaded, "runtime"), "2026-03-16T01:00:00Z");
});

test("attention state helper preserves card review timestamps when events are marked seen", () => {
  const api = loadApi();
  const storage = makeStorage();
  const reviewed = api.markCardReviewed(storage, null, "providers", "2026-03-16T02:00:00Z");
  const next = api.markEventsSeen(storage, reviewed, [{ id: "evt-provider-1" }]);

  assert.equal(api.isEventSeen(next, "evt-provider-1"), true);
  assert.equal(api.getCardReviewedAt(next, "providers"), "2026-03-16T02:00:00Z");

  const reloaded = api.readSeenState(storage);
  assert.equal(api.getCardReviewedAt(reloaded, "providers"), "2026-03-16T02:00:00Z");
});

test("attention state helper does not treat seen events as summary review", () => {
  const api = loadApi();
  const storage = makeStorage();
  const next = api.markEventsSeen(storage, null, [{ id: "evt-runtime-1" }]);

  assert.equal(api.isEventSeen(next, "evt-runtime-1"), true);
  assert.equal(api.getCardReviewedAt(next, "runtime"), "");
});

test("index loads the attention state helper before the main app bundle", () => {
  const html = fs.readFileSync(path.resolve("web/index.html"), "utf8");
  const helperScript = '<script src="attention_state.js"></script>';
  const appScript = '<script src="app.js"></script>';

  const helperIndex = html.indexOf(helperScript);
  const appIndex = html.indexOf(appScript);

  assert.notEqual(helperIndex, -1, "expected attention_state.js script tag");
  assert.notEqual(appIndex, -1, "expected app.js script tag");
  assert.ok(helperIndex < appIndex, "expected attention_state.js to load before app.js");
});
