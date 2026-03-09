(() => {
  const THEME_VALUES = new Set(["system", "light", "dark"]);
  const CONTRAST_VALUES = new Set(["aa", "aaa"]);
  const MOTION_VALUES = new Set(["default", "reduced"]);
  const COLOR_VALUES = new Set(["default", "stronger"]);
  const SPACING_VALUES = new Set(["default", "increased"]);

  function normalizeThemeValue(raw) {
    const value = `${raw || ""}`.trim().toLowerCase();
    return THEME_VALUES.has(value) ? value : "system";
  }

  function coerceBool(raw, fallback) {
    return typeof raw === "boolean" ? raw : fallback;
  }

  function normalizeLiteral(raw, allowedValues, fallback) {
    const value = `${raw || ""}`.trim().toLowerCase();
    return allowedValues.has(value) ? value : fallback;
  }

  function normalizeAppearancePreferences(raw) {
    const source = raw && typeof raw === "object" ? raw : {};
    return {
      override_enabled: coerceBool(source.override_enabled, false),
      sync_theme_preferences: coerceBool(source.sync_theme_preferences, true),
      shared_theme_mode: normalizeThemeValue(source.shared_theme_mode),
      desktop_theme_mode: normalizeThemeValue(source.desktop_theme_mode),
      contrast_policy: normalizeLiteral(source.contrast_policy, CONTRAST_VALUES, "aa"),
      motion_policy: normalizeLiteral(source.motion_policy, MOTION_VALUES, "default"),
      color_separation_policy: normalizeLiteral(
        source.color_separation_policy,
        COLOR_VALUES,
        "default",
      ),
      text_spacing_policy: normalizeLiteral(
        source.text_spacing_policy,
        SPACING_VALUES,
        "default",
      ),
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
    const prefs = normalizeAppearancePreferences(raw);
    const resolvedTheme = resolveDocumentTheme(windowObject, raw);
    const root = documentObject?.documentElement;
    if (!root) {
      return resolvedTheme;
    }
    root.dataset.theme = resolvedTheme;
    root.dataset.themePreference = effectiveDesktopThemeValue(prefs);
    root.dataset.contrastPolicy = prefs.contrast_policy;
    root.dataset.motionPolicy = prefs.motion_policy;
    root.dataset.colorSeparation = prefs.color_separation_policy;
    root.dataset.textSpacing = prefs.text_spacing_policy;
    return resolvedTheme;
  }

  window.Busy38AppearanceTheme = {
    normalizeAppearancePreferences,
    effectiveDesktopThemeValue,
    resolveDocumentTheme,
    applyDocumentTheme,
  };
})();
