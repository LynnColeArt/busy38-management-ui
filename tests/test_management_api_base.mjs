import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const apiBase = require(path.resolve("web/management_api_base.js"));

test("resolveManagementApiBase honors explicit window override literally", () => {
  const resolved = apiBase.resolveManagementApiBase(
    {
      MANAGEMENT_API_BASE: "https://ops.busy.local",
      location: { protocol: "https:", origin: "https://ignored.example" },
    },
    null,
  );

  assert.equal(resolved, "https://ops.busy.local");
});

test("resolveManagementApiBase honors document-level override literally", () => {
  const resolved = apiBase.resolveManagementApiBase(
    {
      MANAGEMENT_API_BASE: "",
      location: { protocol: "https:", origin: "https://ignored.example" },
    },
    {
      querySelector(selector) {
        assert.equal(selector, 'meta[name="busy38-management-api-base"]');
        return {
          getAttribute(name) {
            assert.equal(name, "content");
            return "https://meta.busy.local";
          },
        };
      },
    },
  );

  assert.equal(resolved, "https://meta.busy.local");
});

test("resolveManagementApiBase defaults to served page origin for http and https", () => {
  assert.equal(
    apiBase.resolveManagementApiBase(
      {
        MANAGEMENT_API_BASE: "",
        location: { protocol: "http:", origin: "http://busy.local:8031" },
      },
      { querySelector() { return null; } },
    ),
    "http://busy.local:8031",
  );
  assert.equal(
    apiBase.resolveManagementApiBase(
      {
        MANAGEMENT_API_BASE: "",
        location: { protocol: "https:", origin: "https://busy.example" },
      },
      { querySelector() { return null; } },
    ),
    "https://busy.example",
  );
});

test("resolveManagementApiBase falls back to loopback for non-http page contexts", () => {
  const resolved = apiBase.resolveManagementApiBase(
    {
      MANAGEMENT_API_BASE: "",
      location: { protocol: "file:", origin: "null" },
    },
    { querySelector() { return null; } },
  );

  assert.equal(resolved, "http://127.0.0.1:8031");
});
