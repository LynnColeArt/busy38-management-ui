import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

test("index exposes the mobile pairing operator panel", () => {
  const html = fs.readFileSync(path.resolve("web/index.html"), "utf8");
  assert.match(html, /id="mobilePairingIssueForm"/);
  assert.match(html, /data-action="refresh-mobile-pairing"/);
  assert.match(html, /id="mobilePairingLatest"/);
  assert.match(html, /id="mobilePairingState"/);
  assert.match(html, /mobile_pairing_qr\.js/);
});

test("app wires mobile pairing issue, QR, state, copy, and revoke actions literally", () => {
  const source = fs.readFileSync(path.resolve("web/app.js"), "utf8");
  assert.match(source, /\/api\/mobile\/pairing\/issue/);
  assert.match(source, /\/api\/mobile\/pairing\/state/);
  assert.match(source, /\/api\/mobile\/pairing\/revoke/);
  assert.match(source, /copy-mobile-pairing-code/);
  assert.match(source, /copy-mobile-pairing-qr-payload/);
  assert.match(source, /revoke-mobile-pairing/);
  assert.match(source, /buildPairingQrPayload/);
  assert.match(source, /renderPairingQrSvg/);
});
