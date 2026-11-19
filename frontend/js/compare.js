import { api, errorMessage, predictOne } from "./api.js";
import { formatPercent } from "./format.js";
import { focusFirstInvalid, paintErrors, readForm } from "./form.js";
import { FEATURE_NAMES, modelLabel } from "./presets.js";

let chart;
let bound = false;

const COLORS = {
  catboost: "#c4491d",
  xgboost: "#1e3a5f",
  logistic_regression: "#146c54",
};

function colorFor(name, index) {
  return COLORS[name] || ["#8a5a12", "#3f6212", "#7c3a4a"][index % 3];
}

function showMessage(text) {
  const box = document.getElementById("compare-message");
  box.hidden = !text;
  box.textContent = text || "";
}

function renderSummary(data) {
  const list = document.getElementById("compare-codes");
  list.replaceChildren();
  for (const name of FEATURE_NAMES) {
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = name;
    const value = document.createElement("strong");
    value.textContent = String(data[name]);
    item.append(label, value);
    list.append(item);
  }
}

function renderCards(rows) {
  const wrap = document.getElementById("compare-cards");
  wrap.replaceChildren();
  const scored = rows.filter((row) => row.prediction);
  const best = scored.reduce((top, row) => (
    !top || row.prediction.ctr_probability > top.prediction.ctr_probability ? row : top
  ), null);

  for (const row of rows) {
    const card = document.createElement("article");
    card.className = "model-card";
    if (best && row.model.name === best.model.name) card.classList.add("winner");

    const title = document.createElement("h3");
    title.textContent = modelLabel(row.model.name);

    const algo = document.createElement("p");
    algo.className = "muted";
    algo.textContent = row.model.algorithm || row.model.name;

    card.append(title, algo);

    if (!row.prediction) {
      const failure = document.createElement("p");
      failure.className = "inline-error";
      failure.textContent = errorMessage(row.error);
      card.append(failure);
    } else {
      const score = document.createElement("p");
      score.className = "model-score";
      score.textContent = formatPercent(row.prediction.ctr_probability);

      const decision = document.createElement("p");
      decision.className = "rank-decision";
      decision.dataset.click = row.prediction.predicted_click ? "yes" : "no";
      decision.textContent = row.prediction.predicted_click ? "Clic esperado" : "Sin clic esperado";

      const threshold = document.createElement("p");
      threshold.className = "muted";
      threshold.textContent = `Umbral ${formatPercent(row.prediction.threshold)} · versión ${row.prediction.model_version}`;
      card.append(score, decision, threshold);
    }
    wrap.append(card);
  }
}

function renderChart(rows) {
  const canvas = document.getElementById("compare-chart");
  const note = document.getElementById("compare-chart-note");
  if (!window.Chart) {
    note.hidden = false;
    note.textContent = "La gráfica no está disponible, pero las tarjetas de arriba sí muestran la probabilidad.";
    return;
  }
  note.hidden = true;
  const usable = rows.filter((row) => row.prediction);
  if (chart) chart.destroy();
  chart = new window.Chart(canvas, {
    type: "bar",
    data: {
      labels: usable.map((row) => modelLabel(row.model.name)),
      datasets: [{
        label: "Probabilidad de clic",
        data: usable.map((row) => row.prediction.ctr_probability),
        backgroundColor: usable.map((row, index) => colorFor(row.model.name, index)),
        borderRadius: 8,
        maxBarThickness: 72,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => formatPercent(context.parsed.y),
          },
        },
      },
      scales: {
        y: {
          min: 0,
          max: 1,
          ticks: { callback: (value) => formatPercent(value) },
        },
      },
    },
  });
}

async function onCompare() {
  const { data, errors } = readForm();
  paintErrors(errors);
  if (Object.keys(errors).length) {
    focusFirstInvalid(errors);
    const simMessage = document.getElementById("sim-message");
    simMessage.hidden = false;
    simMessage.dataset.kind = "error";
    simMessage.textContent = "Revisa los campos marcados antes de comparar los modelos.";
    document.dispatchEvent(new CustomEvent("ctr-show", { detail: "simulator" }));
    return;
  }

  const button = document.getElementById("run-compare");
  button.disabled = true;
  showMessage("");
  renderSummary(data);
  try {
    const catalog = await api("/api/v1/models");
    if (!catalog.models.length) {
      showMessage("No hay modelos cargados. Entrena con python -m src.ml.train y vuelve a intentar.");
      document.getElementById("compare-results").hidden = true;
      return;
    }
    const rows = await Promise.all(catalog.models.map(async (model) => {
      try {
        const prediction = await predictOne(data, model.name);
        return { model, prediction, error: null };
      } catch (error) {
        return { model, prediction: null, error };
      }
    }));
    document.getElementById("compare-results").hidden = false;
    renderCards(rows);
    renderChart(rows);
  } catch (error) {
    showMessage(errorMessage(error));
  } finally {
    button.disabled = false;
  }
}

export function mountCompare() {
  if (bound) return;
  bound = true;
  document.getElementById("run-compare").addEventListener("click", onCompare);
}
