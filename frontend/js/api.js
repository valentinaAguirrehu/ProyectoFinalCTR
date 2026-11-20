const STORAGE_KEY = "ctr.apiBase";
const LOCAL_BASE = "http://127.0.0.1:8000";
const IS_LOCAL = ["localhost", "127.0.0.1", ""].includes(location.hostname);
// En producción (Vercel) se usa la URL de la API publicada en <meta name="ctr-api-base">.
const DEPLOYED_BASE = document.querySelector('meta[name="ctr-api-base"]')?.content.trim() || "";
const DEFAULT_BASE = IS_LOCAL || !DEPLOYED_BASE ? LOCAL_BASE : DEPLOYED_BASE;

export class ApiError extends Error {
  constructor(status, body) {
    super(status === 0 ? "No se pudo conectar con la API." : `La API respondió ${status}.`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function readStorage(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorage(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* La interfaz sigue funcionando si el navegador bloquea el almacenamiento. */
  }
}

export function getBase() {
  return (readStorage(STORAGE_KEY) || DEFAULT_BASE).replace(/\/$/, "");
}

export function setBase(url) {
  const clean = (url || DEFAULT_BASE).trim().replace(/\/$/, "");
  writeStorage(STORAGE_KEY, clean);
  return clean;
}

function fieldName(field) {
  if (!field) return "dato";
  return String(field).split(".").filter(Boolean).pop();
}

export function errorMessage(error) {
  if (!(error instanceof ApiError)) return "Ocurrió un error inesperado.";
  if (error.status === 0) {
    return "No se pudo conectar con la API. Comprueba que uvicorn esté en marcha y que la dirección sea la correcta.";
  }
  const detail = error.body?.detail;
  if (error.status === 422 && Array.isArray(detail)) {
    return detail.map((item) => `${fieldName(item.field)}: ${item.message}`).join(" ");
  }
  if (typeof detail === "string") return detail;
  if (detail && typeof detail.message === "string") {
    const available = Array.isArray(detail.available) && detail.available.length
      ? ` Disponibles: ${detail.available.join(", ")}.`
      : "";
    return `${detail.message}.${available}`;
  }
  if (error.status === 503) {
    return "No hay modelos cargados. En la raíz del proyecto ejecuta python -m src.ml.train y vuelve a conectar.";
  }
  return "La API rechazó la solicitud.";
}

export function fieldErrors(error) {
  const detail = error instanceof ApiError ? error.body?.detail : null;
  if (!Array.isArray(detail)) return {};
  const errors = {};
  for (const item of detail) {
    const name = fieldName(item.field);
    if (name && item.message) errors[name] = item.message;
  }
  return errors;
}

export async function api(path, { method = "GET", json } = {}) {
  let response;
  try {
    response = await fetch(`${getBase()}${path}`, {
      method,
      headers: json ? { "Content-Type": "application/json" } : undefined,
      body: json ? JSON.stringify(json) : undefined,
    });
  } catch {
    throw new ApiError(0, { error: "network_error" });
  }

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: "invalid_json", detail: "La API no devolvió JSON." };
    }
  }
  if (!response.ok) throw new ApiError(response.status, data || { error: "http_error" });
  return data;
}

export function predictOne(ad, model) {
  const query = model ? `?model=${encodeURIComponent(model)}` : "";
  return api(`/api/v1/predict${query}`, { method: "POST", json: ad });
}

export function predictBatch(ads, model) {
  const query = model ? `?model=${encodeURIComponent(model)}` : "";
  return api(`/api/v1/predict/batch${query}`, { method: "POST", json: { ads } });
}
