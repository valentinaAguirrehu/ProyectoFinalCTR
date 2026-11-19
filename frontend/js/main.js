import { api, errorMessage, getBase, setBase } from "./api.js";
import { mountCompare } from "./compare.js";
import { mountDashboard } from "./dashboard.js";
import { mountSimulator } from "./simulator.js";

const VIEWS = {
  simulator: mountSimulator,
  compare: mountCompare,
  dashboard: mountDashboard,
};

let current = "simulator";

function show(name) {
  if (!VIEWS[name]) return;
  current = name;
  document.querySelectorAll("[data-view]").forEach((section) => {
    section.hidden = section.dataset.view !== name;
  });
  document.querySelectorAll("[data-nav]").forEach((button) => {
    const active = button.dataset.nav === name;
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  VIEWS[name]();
}

async function checkHealth() {
  const status = document.getElementById("api-status");
  status.dataset.state = "pending";
  status.textContent = "Conectando…";
  try {
    const health = await api("/health");
    const ok = health.status === "ok";
    status.dataset.state = ok ? "ok" : "warn";
    status.textContent = ok
      ? `API activa · ${health.models_loaded} modelo${health.models_loaded === 1 ? "" : "s"}`
      : "API sin modelos";
  } catch (error) {
    status.dataset.state = "bad";
    status.textContent = "API no disponible";
    status.title = errorMessage(error);
  }
}

function bind() {
  document.querySelectorAll("[data-nav]").forEach((button) => {
    button.addEventListener("click", () => show(button.dataset.nav));
  });
  document.addEventListener("ctr-show", (event) => show(event.detail));

  const input = document.getElementById("api-base");
  input.value = getBase();
  document.getElementById("api-form").addEventListener("submit", (event) => {
    event.preventDefault();
    setBase(input.value);
    input.value = getBase();
    checkHealth();
    show(current);
  });
}

if (window.Chart) {
  window.Chart.defaults.font.family = '"Segoe UI", "Helvetica Neue", sans-serif';
  window.Chart.defaults.color = "#6f685e";
  window.Chart.defaults.borderColor = "#d9d0c3";
}

bind();
show("simulator");
checkHealth();
