import { api, errorMessage } from "./api.js";
import { formatDate, formatInt, formatMetric, formatPercent, formatUptime } from "./format.js";
import { modelLabel } from "./presets.js";

const CHART_METRICS = [
  ["roc_auc", "ROC-AUC"],
  ["pr_auc", "PR-AUC"],
  ["accuracy", "Accuracy"],
  ["precision", "Precision"],
  ["recall", "Recall"],
  ["f1", "F1"],
];

const TABLE_METRICS = [
  ...CHART_METRICS,
  ["log_loss", "Log loss"],
  ["brier", "Brier"],
];

const COLORS = ["#c4491d", "#1e3a5f", "#146c54", "#8a5a12"];

let charts = {};
let bound = false;
let split = "test";
let importanceModel = "";
let requestToken = 0;
let importanceToken = 0;

function showStatus(text, kind = "") {
  const box = document.getElementById("dashboard-status");
  box.hidden = !text;
  box.dataset.kind = kind;
  box.textContent = text || "";
}

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

function drawChart(id, config) {
  destroyChart(id);
  if (!window.Chart) return false;
  charts[id] = new window.Chart(document.getElementById(id), config);
  return true;
}

function metricValue(row, key) {
  const value = row[key];
  return value === null || value === undefined ? null : Number(value);
}

function renderComparison(payload) {
  const rows = payload?.models || [];
  const best = payload?.best_model_by_roc_auc;
  const callout = document.getElementById("best-model");
  if (!rows.length) {
    callout.hidden = true;
    document.getElementById("metrics-table").replaceChildren();
    destroyChart("metrics-chart");
    document.getElementById("metrics-chart-box").hidden = true;
    document.getElementById("metrics-empty").hidden = false;
    return;
  }
  document.getElementById("metrics-empty").hidden = true;
  document.getElementById("metrics-chart-box").hidden = false;

  const winner = rows.find((row) => row.model === best) || rows[0];
  callout.hidden = false;
  callout.textContent = `Mejor ROC-AUC en ${payload.split === "validation" ? "validación" : "prueba"}: ${modelLabel(winner.model)} (${formatMetric(winner.roc_auc)}).`;

  const table = document.getElementById("metrics-table");
  table.replaceChildren();
  const head = document.createElement("tr");
  for (const label of ["Modelo", "Umbral", ...TABLE_METRICS.map(([, label]) => label)]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.append(cell);
  }
  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    if (row.model === best) tr.className = "best";
    const cells = [
      modelLabel(row.model),
      formatPercent(row.threshold),
      ...TABLE_METRICS.map(([key]) => formatMetric(metricValue(row, key))),
    ];
    cells.forEach((text, index) => {
      const cell = document.createElement("td");
      cell.textContent = text;
      if (index === 0) cell.dataset.label = "Modelo";
      tr.append(cell);
    });
    tbody.append(tr);
  }
  table.append(thead, tbody);

  const drawn = drawChart("metrics-chart", {
    type: "bar",
    data: {
      labels: CHART_METRICS.map(([, label]) => label),
      datasets: rows.map((row, index) => ({
        label: modelLabel(row.model),
        data: CHART_METRICS.map(([key]) => metricValue(row, key)),
        backgroundColor: COLORS[index % COLORS.length],
        borderRadius: 6,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { min: 0, max: 1 } },
      plugins: { legend: { position: "bottom" } },
    },
  });
  document.getElementById("metrics-chart-note").hidden = drawn;
}

function renderImportance(payload) {
  const rows = payload?.feature_importance || [];
  const frame = document.getElementById("importance-chart").parentElement;
  const note = document.getElementById("importance-empty");
  note.hidden = rows.length > 0;
  frame.hidden = rows.length === 0;
  if (!rows.length) {
    note.textContent = "Este modelo no devolvió importancias.";
    destroyChart("importance-chart");
    return;
  }
  const max = Math.max(...rows.map((row) => Number(row.importance) || 0), 1);
  const drawn = drawChart("importance-chart", {
    type: "bar",
    data: {
      labels: rows.map((row) => row.feature),
      datasets: [{
        label: "Importancia",
        data: rows.map((row) => Number(row.importance)),
        backgroundColor: "#1e3a5f",
        borderRadius: 6,
        maxBarThickness: 22,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { min: 0, suggestedMax: max } },
    },
  });
  document.getElementById("importance-chart-note").hidden = drawn;
  if (!drawn) {
    const note = document.getElementById("importance-chart-note");
    note.hidden = false;
    note.textContent = rows.map((row) => `${row.feature}: ${formatMetric(row.importance)}`).join(" · ");
  }
}

function renderDataset(payload) {
  const box = document.getElementById("dataset-panel");
  box.replaceChildren();
  if (!payload) {
    box.textContent = "No se encontró el resumen del dataset. Revisa que exista data/processed/ctr_phase3/manifest.json.";
    return;
  }

  const intro = document.createElement("p");
  intro.className = "muted";
  intro.textContent = `Objetivo: ${payload.target}. División: ${payload.split_method}.`;
  box.append(intro);

  const cards = document.createElement("div");
  cards.className = "stat-grid";
  for (const splitRow of payload.splits || []) {
    const card = document.createElement("article");
    card.className = "stat-card";
    const title = document.createElement("h3");
    title.textContent = splitLabel(splitRow.split);
    const rows = document.createElement("p");
    rows.textContent = `${formatInt(splitRow.filas)} filas`;
    const rate = document.createElement("p");
    rate.className = "stat-value";
    const clickRate = Number(splitRow.click_1_porcentaje);
    rate.textContent = Number.isNaN(clickRate) ? "—" : `${formatMetric(clickRate)} %`;
    const caption = document.createElement("p");
    caption.className = "muted";
    caption.textContent = "CTR observado (clics)";
    card.append(title, rows, rate, caption);
    cards.append(card);
  }
  box.append(cards);

  const excluded = payload.excluded_features || {};
  const names = Object.keys(excluded);
  if (names.length) {
    const title = document.createElement("h3");
    title.textContent = "Variables que el formulario no envía";
    const list = document.createElement("ul");
    list.className = "excluded-list";
    for (const name of names) {
      const item = document.createElement("li");
      const strong = document.createElement("strong");
      strong.textContent = name;
      item.append(strong, document.createTextNode(`: ${excluded[name]}`));
      list.append(item);
    }
    box.append(title, list);
  }
}

function splitLabel(name) {
  if (name === "train") return "Entrenamiento";
  if (name === "validation") return "Validación";
  if (name === "test") return "Prueba";
  return name;
}

function renderUsage(stats, recent, health) {
  const cards = document.getElementById("usage-cards");
  cards.replaceChildren();
  const items = [
    ["Estado", health ? (health.status === "ok" ? "Operativa" : "Sin modelos") : "—"],
    ["Modelos cargados", health ? formatInt(health.models_loaded) : "—"],
    ["Tiempo activa", health ? formatUptime(health.uptime_seconds) : "—"],
    ["Predicciones servidas", stats ? formatInt(stats.total_predictions) : "—"],
  ];
  for (const [label, value] of items) {
    const card = document.createElement("article");
    card.className = "stat-card";
    const title = document.createElement("h3");
    title.textContent = label;
    const number = document.createElement("p");
    number.className = "stat-value";
    number.textContent = value;
    card.append(title, number);
    cards.append(card);
  }

  const byModel = document.getElementById("usage-by-model");
  byModel.replaceChildren();
  const entries = Object.entries(stats?.by_model || {});
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Todavía no hay predicciones en esta sesión del servidor. Estima un anuncio en el simulador y actualiza el tablero.";
    byModel.append(empty);
  } else {
    for (const [name, row] of entries) {
      const item = document.createElement("li");
      item.textContent = `${modelLabel(name)}: ${formatInt(row.predictions)} predicciones, CTR medio ${formatPercent(row.mean_ctr_probability)}, tasa de clic esperado ${formatPercent(row.predicted_click_rate)}.`;
      byModel.append(item);
    }
  }

  const table = document.getElementById("recent-table");
  const recentEmpty = document.getElementById("recent-empty");
  table.replaceChildren();
  if (recentEmpty) recentEmpty.hidden = Boolean(recent?.length);
  if (!recent?.length) return;
  const head = document.createElement("tr");
  for (const label of ["Hora", "Modelo", "Tipo", "Anuncios", "CTR medio", "Clics"]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.append(cell);
  }
  const thead = document.createElement("thead");
  thead.append(head);
  const tbody = document.createElement("tbody");
  for (const row of recent) {
    const tr = document.createElement("tr");
    for (const value of [
      formatDate(row.created_at),
      modelLabel(row.model),
      row.kind === "batch" ? "Lote" : "Individual",
      formatInt(row.count),
      formatPercent(row.mean_ctr_probability),
      formatInt(row.predicted_clicks),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      tr.append(cell);
    }
    tbody.append(tr);
  }
  table.append(thead, tbody);
}

function fillImportanceSelect(models) {
  const select = document.getElementById("importance-model");
  const previous = importanceModel || select.value;
  select.replaceChildren();
  const preferred = models.find((model) => model.name !== "logistic_regression") || models[0];
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = modelLabel(model.name);
    select.append(option);
  }
  const keep = [...select.options].some((option) => option.value === previous);
  importanceModel = keep ? previous : (preferred?.name || "");
  if (importanceModel) select.value = importanceModel;
}

async function loadImportance() {
  const token = ++importanceToken;
  const note = document.getElementById("importance-empty");
  const modelName = importanceModel;
  if (!modelName) {
    note.hidden = false;
    note.textContent = "No hay un modelo seleccionado.";
    destroyChart("importance-chart");
    return;
  }
  try {
    const payload = await api(`/api/v1/metrics/models/${encodeURIComponent(modelName)}/feature-importance`);
    if (token !== importanceToken) return;
    renderImportance(payload);
  } catch (error) {
    if (token !== importanceToken) return;
    destroyChart("importance-chart");
    document.getElementById("importance-chart").parentElement.hidden = true;
    note.hidden = false;
    note.textContent = errorMessage(error);
  }
}

async function loadDashboard() {
  const token = ++requestToken;
  showStatus("Actualizando indicadores…", "info");
  const [health, models, comparison, dataset, usage, recent] = await Promise.allSettled([
    api("/health"),
    api("/api/v1/models"),
    api(`/api/v1/metrics/models?split=${split}`),
    api("/api/v1/metrics/dataset"),
    api("/api/v1/metrics/predictions"),
    api("/api/v1/metrics/predictions/recent?limit=8"),
  ]);

  if (token !== requestToken) return;

  if (health.status === "rejected" && models.status === "rejected") {
    showStatus(errorMessage(health.reason), "error");
  } else if (health.status === "fulfilled" && health.value.status !== "ok") {
    showStatus("La API está en marcha, pero no tiene modelos cargados. El simulador responderá cuando exista la carpeta models/.", "error");
  } else {
    showStatus("");
  }

  const catalog = models.status === "fulfilled" ? models.value.models : [];
  fillImportanceSelect(catalog);
  renderComparison(comparison.status === "fulfilled" ? comparison.value : { models: [] });
  if (comparison.status === "rejected") {
    document.getElementById("metrics-empty").hidden = false;
    document.getElementById("metrics-empty").textContent = errorMessage(comparison.reason);
  }
  if (dataset.status === "fulfilled") renderDataset(dataset.value);
  else document.getElementById("dataset-panel").textContent = errorMessage(dataset.reason);
  renderUsage(
    usage.status === "fulfilled" ? usage.value : null,
    recent.status === "fulfilled" ? recent.value : [],
    health.status === "fulfilled" ? health.value : null,
  );
  await loadImportance();
}

export function mountDashboard() {
  if (!bound) {
    bound = true;
    document.getElementById("refresh-dashboard").addEventListener("click", () => {
      loadDashboard();
    });
    document.querySelectorAll("[data-split]").forEach((button) => {
      button.addEventListener("click", () => {
        split = button.dataset.split;
        document.querySelectorAll("[data-split]").forEach((item) => {
          item.setAttribute("aria-pressed", item === button ? "true" : "false");
        });
        loadDashboard();
      });
    });
    document.getElementById("importance-model").addEventListener("change", (event) => {
      importanceModel = event.target.value;
      loadImportance();
    });
  }
  loadDashboard();
}
