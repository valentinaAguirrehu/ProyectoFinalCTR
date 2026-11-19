import { api, errorMessage, fieldErrors, predictBatch, predictOne } from "./api.js";
import { formatPercent } from "./format.js";
import { clearForm, fillForm, focusFirstInvalid, paintErrors, readForm, selectedModel } from "./form.js";
import { PRESETS, creativeFrom, modelLabel } from "./presets.js";

let bound = false;
let hasResult = false;

function renderCreative(creative) {
  const visual = document.getElementById("ad-visual");
  visual.dataset.tone = creative.tone;
  document.getElementById("ad-brand").textContent = creative.brand;
  document.getElementById("ad-kicker").textContent = creative.kicker;
  document.getElementById("ad-headline").textContent = creative.headline;
  document.getElementById("ad-copy").textContent = creative.copy;
  document.getElementById("ad-cta").textContent = creative.cta;
}

function syncCreative() {
  const { data, errors } = readForm();
  if (Object.keys(errors).length === 0) renderCreative(creativeFrom(data));
}

function markStale() {
  if (!hasResult) return;
  document.getElementById("result-stale").hidden = false;
}

function showMessage(text, kind = "error") {
  const box = document.getElementById("sim-message");
  box.hidden = !text;
  box.dataset.kind = kind;
  box.textContent = text || "";
}

function renderResult(prediction) {
  hasResult = true;
  document.getElementById("result-empty").hidden = true;
  document.getElementById("result-stale").hidden = true;
  const panel = document.getElementById("result-panel");
  panel.hidden = false;

  const probability = Number(prediction.ctr_probability);
  const percent = Math.max(0, Math.min(100, probability * 100));
  const gauge = document.getElementById("result-gauge");
  gauge.style.setProperty("--p", String(percent));
  gauge.querySelector("span").textContent = formatPercent(probability);

  const decision = document.getElementById("result-decision");
  decision.textContent = prediction.predicted_click ? "Clic esperado" : "Sin clic esperado";
  decision.dataset.click = prediction.predicted_click ? "yes" : "no";

  document.getElementById("result-model").textContent = modelLabel(prediction.model);
  document.getElementById("result-threshold").textContent = formatPercent(prediction.threshold);
  document.getElementById("result-version").textContent = prediction.model_version || "—";
  document.getElementById("result-request").textContent = prediction.request_id || "—";

  const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  panel.scrollIntoView({ behavior, block: "nearest" });
}

async function loadModels() {
  const select = document.getElementById("model-select");
  const previous = select.value;
  const note = document.getElementById("model-load-errors");
  try {
    const payload = await api("/api/v1/models");
    const options = [
      {
        value: "",
        label: payload.default_model
          ? `Predeterminado (${modelLabel(payload.default_model)})`
          : "Predeterminado",
      },
      ...payload.models.map((model) => ({
        value: model.name,
        label: model.is_default ? `${modelLabel(model.name)} · predeterminado` : modelLabel(model.name),
      })),
    ];
    select.replaceChildren();
    for (const option of options) {
      const element = document.createElement("option");
      element.value = option.value;
      element.textContent = option.label;
      select.append(element);
    }
    if ([...select.options].some((option) => option.value === previous)) select.value = previous;
    const failed = Object.entries(payload.load_errors || {});
    note.textContent = failed.length
      ? `No se pudieron cargar: ${failed.map(([name]) => modelLabel(name)).join(", ")}.`
      : "";
  } catch (error) {
    select.replaceChildren(new Option("Predeterminado", ""));
    note.textContent = errorMessage(error);
  }
}

function renderRanking(presets, response) {
  const list = document.getElementById("scenario-results");
  const ranked = presets
    .map((preset, index) => ({ preset, prediction: response.predictions[index] }))
    .sort((a, b) => b.prediction.ctr_probability - a.prediction.ctr_probability);

  list.replaceChildren();
  ranked.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "rank-card";
    if (index === 0) card.classList.add("winner");

    const place = document.createElement("p");
    place.className = "rank-place";
    place.textContent = index === 0 ? "Mayor probabilidad" : `Puesto ${index + 1}`;

    const title = document.createElement("h3");
    title.textContent = item.preset.brand;

    const headline = document.createElement("p");
    headline.textContent = item.preset.headline;

    const score = document.createElement("p");
    score.className = "rank-score";
    score.textContent = formatPercent(item.prediction.ctr_probability);

    const decision = document.createElement("p");
    decision.className = "rank-decision";
    decision.textContent = item.prediction.predicted_click ? "Clic esperado" : "Sin clic esperado";

    card.append(place, title, headline, score, decision);
    list.append(card);
  });

  const summary = document.getElementById("scenario-summary");
  summary.hidden = false;
  summary.textContent = `${response.count} anuncios con ${modelLabel(response.model)}. CTR medio ${formatPercent(response.mean_ctr_probability)}. Clics esperados: ${response.predicted_clicks}.`;
}

async function onSubmit(event) {
  event.preventDefault();
  const { data, errors } = readForm();
  paintErrors(errors);
  if (Object.keys(errors).length) {
    focusFirstInvalid(errors);
    showMessage("Revisa los campos marcados. Los once valores son enteros obligatorios.");
    return;
  }

  const button = document.getElementById("estimate");
  button.disabled = true;
  showMessage("");
  try {
    const prediction = await predictOne(data, selectedModel());
    renderCreative(creativeFrom(data));
    renderResult(prediction);
  } catch (error) {
    paintErrors(fieldErrors(error));
    showMessage(errorMessage(error));
  } finally {
    button.disabled = false;
  }
}

async function onRank() {
  const button = document.getElementById("rank-scenarios");
  button.disabled = true;
  showMessage("");
  try {
    const response = await predictBatch(PRESETS.map((preset) => preset.features), selectedModel());
    renderRanking(PRESETS, response);
  } catch (error) {
    showMessage(errorMessage(error));
  } finally {
    button.disabled = false;
  }
}

function bind() {
  const form = document.getElementById("ad-form");
  form.addEventListener("submit", onSubmit);
  form.addEventListener("input", () => {
    syncCreative();
    markStale();
  });

  document.getElementById("preset-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-preset]");
    if (!button) return;
    const preset = PRESETS.find((item) => item.id === button.dataset.preset);
    if (!preset) return;
    fillForm(preset.features);
    paintErrors({});
    renderCreative(preset);
    markStale();
    showMessage("");
  });

  document.getElementById("clear-form").addEventListener("click", () => {
    clearForm();
    paintErrors({});
    showMessage("");
    markStale();
    renderCreative({
      brand: "Sin anuncio",
      kicker: "Simulación",
      headline: "Completa el formulario",
      copy: "Los once códigos se envían al modelo cuando estimas la probabilidad.",
      cta: "Estimar",
      tone: "custom",
    });
  });

  document.getElementById("rank-scenarios").addEventListener("click", onRank);
  document.getElementById("model-select").addEventListener("change", markStale);
}

export async function mountSimulator() {
  if (!bound) {
    bind();
    bound = true;
    syncCreative();
  }
  await loadModels();
}
