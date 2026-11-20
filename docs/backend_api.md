# Backend e integración del modelo: API de predicción de CTR

Este documento describe el backend que expone los modelos de predicción de CTR (fase de **Despliegue** de CRISP-DM). Está construido con Python y FastAPI.

## 1. Arquitectura

```
Cliente (frontend, Postman, curl)
        │  JSON / HTTP
        ▼
┌──────────────────────────── src/api ────────────────────────────┐
│ main.py        App, middleware (X-Request-ID, tiempos), CORS    │
│ routers/       predictions · models · metrics   (capa HTTP)     │
│ schemas.py     Validación de entrada/salida con Pydantic        │
│ errors.py      Errores de dominio → respuestas JSON uniformes   │
│ services/      PredictionService · MetricsService · PredictionLog│
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌──────────────────────────── src/ml ─────────────────────────────┐
│ registry.py    Descubre y carga los modelos de models/<nombre>/ │
│ models.py      Interfaz común CTRModel: CatBoost, XGBoost, LogReg│
│ evaluation.py  Métricas y umbral de decisión                     │
│ train.py       Entrena y genera los artefactos                   │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
src/data/preprocessing.py   Contrato de características (Fase 3)
data/processed/ctr_phase3/  Particiones Parquet y manifest.json
models/<nombre>/            Modelo + metadata.json (umbral, métricas)
```

Decisiones principales:

- **Separación por capas.** Los routers solo traducen HTTP; la lógica está en servicios y los modelos se acceden a través del registro. Así se pueden cambiar o añadir modelos sin tocar los endpoints.
- **Contrato de características único.** `src/data/preprocessing.py` define `FEATURES` y `to_model_frame()`, usados tanto en entrenamiento como en inferencia, lo que evita diferencias entre ambos (*training/serving skew*).
- **Modelos intercambiables.** Cada modelo implementa `CTRModel.predict_proba()`. Para integrar un modelo nuevo de la fase de modelado basta con crear una subclase, registrarla en `ALGORITHMS` y guardar el artefacto con `model.save(...)`.
- **Carga al arrancar.** Los modelos se cargan una vez (evento *lifespan*) y se mantienen en memoria. `POST /api/v1/models/reload` permite cargar modelos reentrenados sin reiniciar el servidor. Si un artefacto está dañado, la API arranca igual y lo informa en `load_errors`.
- **Umbral de decisión por modelo.** Se ajusta en la partición de validación (máximo F1) y se guarda en `metadata.json`. La partición de prueba solo se usa para reportar métricas.

## 2. Puesta en marcha

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Generar los artefactos de modelos (requiere data/processed/ctr_phase3/)
python -m src.ml.train                          # CatBoost + regresión logística
python -m src.ml.train --models xgboost         # opcional; en macOS: brew install libomp

# 2) Levantar la API
uvicorn src.api.main:app --reload

# 3) Pruebas
pytest
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Esquema OpenAPI: <http://127.0.0.1:8000/openapi.json>

Opciones del entrenamiento: `--train-rows` (por defecto 1.000.000; `0` = partición completa), `--eval-rows` (500.000), `--seed`, `--models-dir`, `--data-dir`.

### Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `CTR_MODELS_DIR` | `models/` | Carpeta de artefactos. |
| `CTR_DATA_DIR` | `data/processed/ctr_phase3/` | Carpeta con `manifest.json`. |
| `CTR_DEFAULT_MODEL` | mejor AUC de validación | Modelo usado cuando no se indica `?model=`. |
| `CTR_MAX_BATCH_SIZE` | `1000` | Máximo de anuncios por lote. |
| `CTR_PREDICTION_LOG_SIZE` | `500` | Predicciones recientes guardadas en memoria. |
| `CTR_CORS_ORIGINS` | `*` | Orígenes permitidos, separados por comas. |

## Despliegue recomendado: frontend en Vercel y API en Render

La función de Vercel para la API superó el límite de tamaño del bundle (500 MB) por las dependencias de CatBoost y los datos del proyecto. Mantén en Vercel solo el frontend estático y ejecuta esta API como un servicio web de Python en Render.

### API en Render

1. Crea un **Web Service** desde el repositorio `valentinaAguirrehu/ProyectoFinalCTR`, rama `main`, dejando el directorio raíz vacío.
2. Configura:
   - **Build Command:** `pip install -r requirements-api.txt`
   - **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
3. En las variables de entorno, define `CTR_CORS_ORIGINS` con el dominio público del frontend de Vercel, por ejemplo `https://<tu-dominio>.vercel.app`.
4. Cuando Render entregue la URL del servicio, comprueba `/health`, `/api/v1/models` y `/docs`.

`requirements-api.txt` incluye solo las dependencias usadas para servir la API y cargar el modelo CatBoost. El directorio `models/catboost/` versionado contiene `model.cbm` y `metadata.json`.

### Frontend estático en Vercel

En **Project Settings → Build and Deployment**, cambia **Root Directory** a `frontend`, el preset a **Other**, deja vacío el comando de compilación y usa `.` como directorio de salida si Vercel lo solicita. Después de desplegar, abre la web, pega la URL del servicio Render (sin `/api/v1`) en **Dirección de la API** y pulsa **Conectar**.

## 3. Endpoints

Prefijo: `/api/v1` (salvo `/health`).

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio (`ok` / `degraded`), modelos cargados. |
| POST | `/predict` | CTR de un anuncio. `?model=` opcional. |
| POST | `/predict/batch` | CTR de un lote `{"ads": [...]}` (mismo orden que la entrada). |
| GET | `/models` | Modelos cargados, modelo por defecto y errores de carga. |
| GET | `/models/{name}` | Detalle: características, entrenamiento y métricas. |
| POST | `/models/reload` | Recarga los artefactos desde disco. |
| GET | `/metrics/dataset` | Resumen de la Fase 3: particiones, cortes temporales y variables excluidas. |
| GET | `/metrics/models?split=test` | Comparativa de modelos (`validation` o `test`), ordenada por ROC-AUC. |
| GET | `/metrics/models/{name}` | Métricas de validación y prueba de un modelo. |
| GET | `/metrics/models/{name}/feature-importance` | Importancia de variables (CatBoost / XGBoost). |
| GET | `/metrics/predictions` | Uso de la API: peticiones, predicciones y CTR medio predicho por modelo. |
| GET | `/metrics/predictions/recent?limit=20` | Últimas predicciones servidas. |

### Entrada: características del anuncio

Los once campos son obligatorios y enteros. Las categorías son los códigos anonimizados del dataset Criteo; una categoría no vista en entrenamiento es válida y el modelo la trata como desconocida.

| Campo | Regla |
|---|---|
| `timestamp` | entero ≥ 0 (segundos desde el inicio del periodo del dataset) |
| `campaign`, `cat1` … `cat9` | entero entre 0 y 2.147.483.647 |

No se aceptan campos adicionales (p. ej. `uid`, `cost`, `conversion`), ya que se excluyeron en la Fase 3 por fuga de información o disponibilidad no confirmada.

### Ejemplo: predicción individual

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict?model=catboost" \
  -H "Content-Type: application/json" \
  -d '{"timestamp": 2288076, "campaign": 787943, "cat1": 30763035, "cat2": 32440040,
       "cat3": 11426249, "cat4": 29196072, "cat5": 32440044, "cat6": 1973606,
       "cat7": 25459503, "cat8": 32440044, "cat9": 29196072}'
```

```json
{
  "ctr_probability": 0.2050,
  "predicted_click": false,
  "threshold": 0.2857,
  "request_id": "9ac6cf9bc95b43b8ac4a8625c50612e5",
  "model": "catboost",
  "model_version": "1.0.0",
  "created_at": "2026-11-15T04:37:51.104487Z"
}
```

`predicted_click` es `ctr_probability >= threshold`. Para priorizar o pujar por anuncios conviene usar directamente `ctr_probability`.

### Ejemplo: lote

```json
POST /api/v1/predict/batch
{"ads": [{...anuncio 1...}, {...anuncio 2...}]}
```

Respuesta: `count`, `mean_ctr_probability`, `predicted_clicks` y la lista `predictions`.

## 4. Manejo de errores

Todas las respuestas de error tienen el mismo formato; `request_id` también se devuelve en la cabecera `X-Request-ID` (se puede enviar una propia para trazar la petición).

```json
{"error": "validation_error",
 "detail": [{"field": "cat1", "message": "Input should be greater than or equal to 0", "type": "greater_than_equal"}],
 "request_id": "..."}
```

| HTTP | `error` | Causa |
|---|---|---|
| 404 | `model_not_found` | `?model=` no existe; `detail.available` lista los disponibles. |
| 404 | `resource_not_found` | Falta `manifest.json` o el modelo no ofrece importancia de variables. |
| 404 | `not_found` | Ruta inexistente. |
| 413 | `batch_too_large` | El lote supera `CTR_MAX_BATCH_SIZE`. |
| 422 | `validation_error` | Campo faltante, tipo incorrecto, valor negativo, campo no permitido, JSON mal formado o lote vacío. |
| 500 | `prediction_failed` | El modelo falló o devolvió valores no válidos. |
| 503 | `no_models_available` | No hay modelos cargados (entrenar y llamar a `/models/reload`). |

## 5. Modelos integrados

Resultados del entrenamiento de referencia (`python -m src.ml.train`: 1.000.000 filas de entrenamiento y 500.000 de validación/prueba, muestreadas con semilla 42). Umbral ajustado en validación.

| Modelo | ROC-AUC prueba | PR-AUC | Log loss | F1 | Umbral |
|---|---:|---:|---:|---:|---:|
| `catboost` (por defecto) | 0,6926 | 0,5667 | 0,6002 | 0,5785 | 0,286 |
| `logistic_regression` (línea base) | 0,6807 | 0,5527 | 0,6080 | 0,5707 | 0,291 |

La tabla muestra resultados de referencia, no los del artefacto de demostración desplegado. Este último se entrenó con 250.000 filas de entrenamiento y 100.000 de validación y prueba: AUC de validación **0,6793**, AUC de prueba **0,6823**, F1 de prueba **0,5705** y umbral **0,2886**.

Las variables más importantes para CatBoost son `cat3`, `cat9`, `cat6` y `cat8`. Las cifras se pueden consultar en vivo en `/api/v1/metrics/models`. La optimización de hiperparámetros corresponde a la fase de Modelado; la API cargará automáticamente cualquier modelo mejorado que se guarde en `models/`.

### Formato de artefacto

```
models/<nombre>/
├── model.cbm | model.json | model.joblib   # según el algoritmo
├── categories.json                         # solo XGBoost
└── metadata.json                           # name, version, algorithm, threshold, features, training, metrics
```

`models/` se excluye normalmente de Git para evitar subir artefactos grandes. El modelo CatBoost de demostración en `models/catboost/` está versionado explícitamente para que Render lo cargue; si se genera otro modelo, hay que integrarlo de forma deliberada y revisar su tamaño.

## 6. Pruebas

`tests/test_api.py` (32 casos) entrena modelos pequeños con datos sintéticos, por lo que no necesita el dataset. Cubre:

- Salud del servicio, cabeceras de trazabilidad y documentación OpenAPI.
- Predicción individual y por lotes con cada modelo, determinismo, consistencia entre lote e individual, categorías no vistas y que el modelo aprende la señal sintética.
- Validación: campos faltantes, nulos, negativos, tipos incorrectos, campos extra, JSON mal formado, lote vacío y lote demasiado grande.
- Errores: modelo inexistente (404), API sin modelos (503), artefacto dañado reportado sin detener el servicio.
- Métricas: comparativa ordenada, métricas por modelo, importancia de variables, resumen del dataset y estadísticas de uso.
