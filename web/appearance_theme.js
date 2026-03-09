(() => {
  const THEME_VALUES = new Set(["system", "light", "dark"]);

  function normalizeThemeValue(raw) {
    const value = `${raw || ""}`.trim().toLowerCase();
    return THEME_VALUES.has(value) ? value : "system";
  }

  function coerceBool(raw, fallback) {
    return typeof raw === "boolean" ? raw : fallback;
  }

  function normalizeAppearancePreferences(raw) {
    const source = raw && typeof raw === "object" ? raw : {};
    return {
      override_enabled: coerceBool(source.override_enabled, false),
      sync_theme_preferences: coerceBool(source.sync_theme_preferences, true),
      shared_theme_mode: normalizeThemeValue(source.shared_theme_mode),
      desktop_theme_mode: normalizeThemeValue(source.desktop_theme_mode),
    };
  }

  function effectiveDesktopThemeValue(raw) {
    const prefs = normalizeAppearancePreferences(raw);
    if (!prefs.override_enabled) {
      return "system";
    }
    if (prefs.sync_theme_preferences) {
      return prefs.shared_theme_mode;
    }
    return prefs.desktop_theme_mode;
  }

  function resolveDocumentTheme(windowObject, raw) {
    const effective = effectiveDesktopThemeValue(raw);
    if (effective !== "system") {
      return effective;
    }
    const media = windowObject?.matchMedia?.("(prefers-color-scheme: dark)");
    return media?.matches ? "dark" : "light";
  }

  function applyDocumentTheme(windowObject, documentObject, raw) {
    const resolvedTheme = resolveDocumentTheme(windowObject, raw);
    const root = documentObject?.documentElement;
    if (!root) {
      return resolvedTheme;
    }
    root.dataset.theme = resolvedTheme;
    root.dataset.themePreference = effectiveDesktopThemeValue(raw);
    return resolvedTheme;
  }

  window.Busy38AppearanceTheme = {
    normalizeAppearancePreferences,
    effectiveDesktopThemeValue,
    resolveDocumentTheme,
    applyDocumentTheme,
  };
})();
