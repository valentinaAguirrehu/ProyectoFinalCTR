import { FEATURE_NAMES } from "./presets.js";

const CATEGORY_MAX = 2147483647;

export function readForm() {
  const data = {};
  const errors = {};
  for (const name of FEATURE_NAMES) {
    const input = document.getElementById(name);
    const raw = input.value.trim();
    if (raw === "") {
      errors[name] = "Obligatorio.";
      continue;
    }
    if (!/^\d+$/.test(raw)) {
      errors[name] = "Escribe un entero mayor o igual que 0, sin puntos ni comas.";
      continue;
    }
    const value = Number(raw);
    if (!Number.isSafeInteger(value)) {
      errors[name] = "El número es demasiado grande.";
      continue;
    }
    if (name !== "timestamp" && value > CATEGORY_MAX) {
      errors[name] = "El máximo permitido es 2.147.483.647.";
      continue;
    }
    data[name] = value;
  }
  return { data, errors };
}

export function fillForm(features) {
  for (const name of FEATURE_NAMES) {
    const input = document.getElementById(name);
    if (input && features[name] !== undefined) input.value = String(features[name]);
  }
}

export function clearForm() {
  for (const name of FEATURE_NAMES) {
    document.getElementById(name).value = "";
  }
}

export function paintErrors(errors) {
  document.querySelectorAll("[data-error-for]").forEach((element) => {
    const message = errors[element.dataset.errorFor] || "";
    element.textContent = message;
    const input = document.getElementById(element.dataset.errorFor);
    if (input) input.setAttribute("aria-invalid", message ? "true" : "false");
  });
}

export function focusFirstInvalid(errors) {
  const name = FEATURE_NAMES.find((feature) => errors[feature]);
  if (name) document.getElementById(name).focus();
}

export function selectedModel() {
  return document.getElementById("model-select").value;
}
