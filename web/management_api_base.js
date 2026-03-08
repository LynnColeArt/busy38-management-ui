(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
    return;
  }
  root.busyManagementApiBase = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  function trimString(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  function resolveManagementApiBase(win, doc) {
    const runtimeWindow = win || (typeof window !== "undefined" ? window : null);
    const runtimeDocument = doc || (typeof document !== "undefined" ? document : null);

    const windowOverride = trimString(runtimeWindow && runtimeWindow.MANAGEMENT_API_BASE);
    if (windowOverride) {
      return windowOverride;
    }

    let documentOverride = "";
    if (runtimeDocument && typeof runtimeDocument.querySelector === "function") {
      const meta = runtimeDocument.querySelector('meta[name="busy38-management-api-base"]');
      documentOverride = trimString(meta && meta.getAttribute("content"));
    }
    if (documentOverride) {
      return documentOverride;
    }

    const protocol = trimString(runtimeWindow && runtimeWindow.location && runtimeWindow.location.protocol).toLowerCase();
    const origin = trimString(runtimeWindow && runtimeWindow.location && runtimeWindow.location.origin);
    if ((protocol === "http:" || protocol === "https:") && origin) {
      return origin;
    }

    return "http://127.0.0.1:8031";
  }

  return {
    resolveManagementApiBase,
  };
});
