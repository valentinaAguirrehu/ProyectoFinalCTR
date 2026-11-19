"""Aplicacion FastAPI del servicio de prediccion de CTR.

Ejecutar desde la raiz del proyecto:

    uvicorn src.api.main:app --reload

Documentacion interactiva: http://127.0.0.1:8000/docs (Swagger) y /redoc.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import Settings
from src.api.errors import register_exception_handlers
from src.api.routers import metrics, models, predictions
from src.api.schemas import HealthResponse
from src.api.services.metrics_service import MetricsService
from src.api.services.prediction_log import PredictionLog
from src.api.services.prediction_service import PredictionService
from src.ml.registry import ModelRegistry

API_PREFIX = "/api/v1"
DESCRIPTION = """
API del proyecto **Prediccion de CTR en Anuncios Digitales** (dataset Criteo Attribution Modeling,
metodologia CRISP-DM, fase de despliegue).

Recibe las caracteristicas de una impresion de anuncio (`timestamp`, `campaign`, `cat1`-`cat9`) y
devuelve la probabilidad estimada de clic usando los modelos entrenados en `models/`.

* **Prediccion**: individual y por lotes, con seleccion de modelo.
* **Modelos**: listado, detalle y recarga en caliente de los artefactos.
* **Metricas y resultados**: dataset, comparativa de modelos, importancia de variables y uso de la API.

Todos los errores siguen el formato `{"error": ..., "detail": ..., "request_id": ...}`.
"""

log = logging.getLogger("ctr.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = ModelRegistry(settings.models_dir, settings.default_model).load_all()
        if not registry.models:
            log.warning("La API arranco sin modelos; /predict respondera 503 hasta entrenar y recargar.")
        prediction_log = PredictionLog(settings.prediction_log_size)
        app.state.settings = settings
        app.state.registry = registry
        app.state.prediction_service = PredictionService(registry, prediction_log, settings.max_batch_size)
        app.state.metrics_service = MetricsService(registry, prediction_log, settings.data_dir)
        app.state.started_at = time.monotonic()
        yield

    app = FastAPI(
        title="CTR Prediction API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Prediccion", "description": "Estimacion de la probabilidad de clic."},
            {"name": "Modelos", "description": "Modelos de Machine Learning integrados."},
            {"name": "Metricas y resultados", "description": "Desempeno de modelos, dataset y uso."},
            {"name": "Sistema", "description": "Estado del servicio."},
        ],
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
                       allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        log.info("%s %s -> %s (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    register_exception_handlers(app)

    @app.get("/health", response_model=HealthResponse, tags=["Sistema"], summary="Estado del servicio")
    def health(request: Request) -> HealthResponse:
        registry: ModelRegistry = request.app.state.registry
        return HealthResponse(
            status="ok" if registry.models else "degraded",
            models_loaded=len(registry.models),
            default_model=registry.default_name,
            uptime_seconds=round(time.monotonic() - request.app.state.started_at, 2),
        )

    api = APIRouter(prefix=API_PREFIX)
    api.include_router(predictions.router)
    api.include_router(models.router)
    api.include_router(metrics.router)
    app.include_router(api)
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
