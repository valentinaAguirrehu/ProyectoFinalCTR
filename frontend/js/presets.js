/** Características que acepta la API y escenarios de demostración del simulador. */

export const FEATURES = [
  { name: "timestamp", label: "Tiempo", hint: "Segundos desde el inicio del periodo" },
  { name: "campaign", label: "Campaña", hint: "Código anonimizado" },
  { name: "cat1", label: "Categoría 1", hint: "Contexto categórico" },
  { name: "cat2", label: "Categoría 2", hint: "Contexto categórico" },
  { name: "cat3", label: "Categoría 3", hint: "Suele pesar en CatBoost" },
  { name: "cat4", label: "Categoría 4", hint: "Contexto categórico" },
  { name: "cat5", label: "Categoría 5", hint: "Contexto categórico" },
  { name: "cat6", label: "Categoría 6", hint: "Suele pesar en CatBoost" },
  { name: "cat7", label: "Categoría 7", hint: "Contexto categórico" },
  { name: "cat8", label: "Categoría 8", hint: "Suele pesar en CatBoost" },
  { name: "cat9", label: "Categoría 9", hint: "Suele pesar en CatBoost" },
];

export const FEATURE_NAMES = FEATURES.map((feature) => feature.name);

/** La tarjeta visual es una simulación. El modelo solo recibe estos enteros. */
export const PRESETS = [
  {
    id: "retail",
    brand: "Casa Norte",
    kicker: "Hogar · patrocinado",
    headline: "Renueva tu sala esta semana",
    copy: "Selección de temporada con envío a tu ciudad.",
    cta: "Ver la colección",
    tone: "warm",
    features: {
      timestamp: 2288076,
      campaign: 787943,
      cat1: 30763035,
      cat2: 32440040,
      cat3: 11426249,
      cat4: 29196072,
      cat5: 32440044,
      cat6: 1973606,
      cat7: 25459503,
      cat8: 32440044,
      cat9: 29196072,
    },
  },
  {
    id: "tech",
    brand: "Nodo Lab",
    kicker: "Tecnología · patrocinado",
    headline: "Auriculares con 40 horas de batería",
    copy: "Un anuncio de producto listo para estimar su CTR.",
    cta: "Comprar ahora",
    tone: "cool",
    features: {
      timestamp: 2100000,
      campaign: 512004,
      cat1: 12004410,
      cat2: 8801203,
      cat3: 4401922,
      cat4: 29196072,
      cat5: 10293847,
      cat6: 558821,
      cat7: 25459503,
      cat8: 9004412,
      cat9: 12004410,
    },
  },
  {
    id: "travel",
    brand: "Ruta Clara",
    kicker: "Viajes · patrocinado",
    headline: "Tres noches frente al mar",
    copy: "Escapada de fin de semana. Cupos limitados.",
    cta: "Reservar",
    tone: "sea",
    features: {
      timestamp: 2500000,
      campaign: 990112,
      cat1: 30763035,
      cat2: 150022,
      cat3: 88001123,
      cat4: 44012,
      cat5: 32440044,
      cat6: 7722110,
      cat7: 100293,
      cat8: 6612094,
      cat9: 88001123,
    },
  },
];

const MODEL_LABELS = {
  catboost: "CatBoost",
  xgboost: "XGBoost",
  logistic_regression: "Regresión logística",
};

export function modelLabel(name) {
  return MODEL_LABELS[name] || name;
}

export function matchPreset(features) {
  return PRESETS.find((preset) =>
    FEATURE_NAMES.every((name) => preset.features[name] === features[name]),
  ) || null;
}

export function creativeFrom(features) {
  const preset = matchPreset(features);
  if (preset) return preset;
  return {
    id: "custom",
    brand: "Anuncio personalizado",
    kicker: "Simulación · patrocinado",
    headline: `Campaña ${features.campaign ?? ""}`.trim(),
    copy: "Vista simulada. El modelo solo recibe los códigos del formulario.",
    cta: "Ver anuncio",
    tone: "custom",
    features,
  };
}
