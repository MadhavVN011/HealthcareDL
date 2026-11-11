"use strict";
(() => {
  document.documentElement.classList.add("js-ready");
  const toggle = document.getElementById("mobile-toggle");
  toggle?.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    document.body.classList.toggle("navigation-open", open);
  });
  const input = document.getElementById("symptoms");
  if (!input) return;
  const picker = document.getElementById("symptom-picker");
  const status = document.getElementById("transcription");
  const speechButton = document.getElementById("startSpeechRecognition");
  const search = document.getElementById("symptom-search");
  const results = document.getElementById("symptom-search-results");
  const manual = document.getElementById("manual-entry");
  const selected = document.getElementById("selected-symptoms");
  const normalize = (value) =>
    value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  const names = () => {
    const unique = new Map();
    input.value
      .split(/[,;\n]/)
      .map((s) => s.trim())
      .filter(Boolean)
      .forEach((value) => {
        const key = normalize(value);
        if (!unique.has(key)) unique.set(key, value);
      });
    return [...unique.values()];
  };
  function renderSelection() {
    const values = names();
    selected.replaceChildren();
    document.getElementById("selected-count").textContent = String(
      values.length,
    );
    if (!values.length) {
      const empty = document.createElement("p");
      empty.className = "selection-empty";
      empty.textContent = "No symptoms selected yet";
      selected.append(empty);
    }
    values.forEach((value) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "selected-chip";
      chip.setAttribute("aria-label", `Remove ${value.replaceAll("_", " ")}`);
      chip.textContent = value.replaceAll("_", " ") + " ×";
      chip.addEventListener("click", () => {
        input.value = names()
          .filter((name) => name !== value)
          .join(", ");
        renderSelection();
        search.focus();
      });
      selected.append(chip);
    });
    document.querySelectorAll("[data-symptom]").forEach((button) => {
      button.classList.toggle(
        "is-selected",
        values.some((v) => normalize(v) === button.dataset.symptom),
      );
    });
  }
  function addSymptom(value) {
    const values = names();
    if (!values.some((name) => normalize(name) === normalize(value)))
      values.push(value.replaceAll("_", " "));
    const combined = values.join(", ");
    if (combined.length > input.maxLength) {
      status.textContent =
        "The symptom list is too long. Remove some entries first.";
      return;
    }
    input.value = combined;
    renderSelection();
    status.textContent = "Symptom added to your list.";
  }
  document.getElementById("search-panel").hidden = false;
  manual.open = Boolean(document.querySelector(".error"));
  input.addEventListener("input", renderSelection);
  document.querySelectorAll("[data-symptom]").forEach((button) => {
    button.addEventListener("click", () => {
      addSymptom(button.dataset.symptom);
      search.value = "";
      results.hidden = true;
      search.focus();
    });
  });
  search.addEventListener("input", () => {
    const query = normalize(search.value).replaceAll("_", " ");
    results.hidden = !query;
    let matches = 0;
    results.querySelectorAll("[data-symptom]").forEach((button) => {
      const match = button.dataset.symptom.replaceAll("_", " ").includes(query);
      button.hidden = !match;
      if (match) matches++;
    });
    document.getElementById("no-search-results").hidden = matches > 0;
  });
  search.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      results.hidden = true;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      if (!results.hidden)
        results.querySelector("[data-symptom]:not([hidden])")?.click();
    }
  });
  document.getElementById("clear-symptoms").addEventListener("click", () => {
    input.value = "";
    renderSelection();
    status.textContent = "Symptom list cleared.";
    search.focus();
  });
  document.getElementById("add-symptom").addEventListener("click", () => {
    if (!picker.value) {
      picker.focus();
      return;
    }
    addSymptom(picker.value);
    picker.value = "";
  });
  renderSelection();
  const exploreButton = document.getElementById("explore-button");
  const buttonContent = [...exploreButton.childNodes].map((node) =>
    node.cloneNode(true),
  );
  window.addEventListener("pageshow", () => {
    exploreButton.disabled = false;
    exploreButton.replaceChildren(
      ...buttonContent.map((node) => node.cloneNode(true)),
    );
  });
  document.getElementById("explorer-form").addEventListener("submit", () => {
    const button = document.getElementById("explore-button");
    button.disabled = true;
    button.textContent = "Exploring…";
  });
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    status.textContent =
      "Voice input is unavailable in this browser. You can type or choose symptoms above.";
    return;
  }
  speechButton.hidden = false;
  status.textContent =
    "Microphone is off. Review the privacy note before starting.";
  let recognition = null;
  speechButton.addEventListener("click", () => {
    if (recognition) return;
    recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    speechButton.disabled = true;
    status.textContent = "Listening…";
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
        .replace(/\bcomma\b/gi, ",")
        .trim();
      const combined = [input.value.trim(), transcript]
        .filter(Boolean)
        .join(", ");
      if (combined.length > input.maxLength) {
        status.textContent =
          "The symptom list is too long. Shorten it before adding more.";
        return;
      }
      input.value = combined;
      renderSelection();
      manual.open = true;
      status.textContent =
        "Text added to your symptom list. Check the names before submitting.";
      input.focus();
    };
    recognition.onerror = () => {
      status.textContent =
        "Voice input could not finish. Check microphone permissions or type your symptoms.";
    };
    recognition.onend = () => {
      recognition = null;
      speechButton.disabled = false;
      if (status.textContent === "Listening…")
        status.textContent =
          "No speech was captured. You can try again or type.";
    };
    try {
      recognition.start();
    } catch {
      recognition = null;
      speechButton.disabled = false;
      status.textContent =
        "Voice input could not start. You can type or choose symptoms above.";
    }
  });
})();
