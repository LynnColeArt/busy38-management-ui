import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const qr = require(path.resolve("web/mobile_pairing_qr.js"));

test("buildPairingQrPayload serializes the canonical payload literally", () => {
  const payload = qr.buildPairingQrPayload({
    controlPlaneUrl: "http://busy.local:8031",
    issue: {
      pairing_code: "ABCD-2345",
      instance_id: "busy-local",
      expires_at: "2026-03-07T21:05:00Z",
    },
    now: new Date("2026-03-07T21:00:00Z"),
  });

  assert.equal(
    payload,
    "busy38pair://pair?v=1&control_plane_url=http%3A%2F%2Fbusy.local%3A8031&pairing_code=ABCD-2345&instance_id=busy-local&expires_at=2026-03-07T21%3A05%3A00Z",
  );
});

test("buildPairingQrPayload rejects malformed or expired live issue data", () => {
  assert.throws(
    () =>
      qr.buildPairingQrPayload({
        controlPlaneUrl: "http://busy.local:8031",
        issue: {
          pairing_code: "bad",
          instance_id: "busy-local",
          expires_at: "2026-03-07T21:05:00Z",
        },
        now: new Date("2026-03-07T21:00:00Z"),
      }),
    /AAAA-BBBB/,
  );

  assert.throws(
    () =>
      qr.buildPairingQrPayload({
        controlPlaneUrl: "http://busy.local:8031",
        issue: {
          pairing_code: "ABCD-2345",
          instance_id: "busy-local",
          expires_at: "2026-03-07T20:55:00Z",
        },
        now: new Date("2026-03-07T21:00:00Z"),
      }),
    /expired/,
  );
});

test("renderPairingQrSvg and parsePairingQrPayload stay literal", () => {
  const payload =
    "busy38pair://pair?v=1&control_plane_url=http%3A%2F%2Fbusy.local%3A8031&pairing_code=ABCD-2345&instance_id=busy-local&expires_at=2026-03-07T21%3A05%3A00Z";
  const svg = qr.renderPairingQrSvg(payload);

  assert.match(svg, /^<svg/);
  assert.match(svg, /Busy mobile pairing QR/);

  const parsed = qr.parsePairingQrPayload(payload, {
    now: new Date("2026-03-07T21:00:00Z"),
  });

  assert.deepEqual(parsed, {
    controlPlaneUrl: "http://busy.local:8031",
    pairingCode: "ABCD-2345",
    instanceId: "busy-local",
    expiresAt: "2026-03-07T21:05:00.000Z",
  });
});
