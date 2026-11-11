"use strict";
(() => {
  const root = document.documentElement;
  const storageKey = "healthcaredl-theme";
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
  const isTheme = (value) => value === "light" || value === "dark";
  let preference = null;

  function readPreference() {
    try {
      const saved = localStorage.getItem(storageKey);
      preference = isTheme(saved) ? saved : null;
    } catch {
      // Theme switching still works when browser storage is unavailable.
    }
  }

  function applyTheme() {
    const theme = preference || (systemTheme.matches ? "dark" : "light");
    root.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').content =
      theme === "dark" ? "#0e1922" : "#f7f9fb";
    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
      toggle.setAttribute("aria-checked", String(theme === "dark"));
      toggle.title = `Switch to ${theme === "dark" ? "light" : "dark"} theme`;
      document.getElementById("theme-label").textContent =
        theme === "dark" ? "Dark" : "Light";
    }
  }

  // Run before the stylesheet to avoid flashing the wrong theme on navigation.
  readPreference();
  applyTheme();

  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("theme-toggle");
    applyTheme();
    toggle.hidden = false;
    toggle.addEventListener("click", () => {
      preference = root.dataset.theme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(storageKey, preference);
      } catch {
        // Keep the selection for this page even if it cannot be persisted.
      }
      applyTheme();
    });
  });

  systemTheme.addEventListener("change", () => {
    if (preference === null) applyTheme();
  });
  window.addEventListener("storage", (event) => {
    if (event.key === storageKey || event.key === null) {
      readPreference();
      applyTheme();
    }
  });
  window.addEventListener("pageshow", () => {
    readPreference();
    applyTheme();
  });
})();
