"""Endpoints de consulta de metricas y resultados."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_metrics_service
from src.api.routers.models import get_model_or_404
from src.api.schemas import ErrorResponse
from src.api.services.metrics_service import MetricsService
from src.ml.registry import LoadedModel

router = APIRouter(prefix="/metrics", tags=["Metricas y resultados"])
Service = Annotated[MetricsService, Depends(get_metrics_service)]
Model = Annotated[LoadedModel, Depends(get_model_or_404)]
_404 = {404: {"model": ErrorResponse}}


@router.get("/dataset", responses=_404, summary="Resumen del dataset preparado (Fase 3)")
def dataset_summary(service: Service) -> dict[str, Any]:
    """Particiones temporales, distribucion de ``click`` y variables usadas/excluidas."""
    return service.dataset_summary()


@router.get("/models", summary="Comparar metricas de todos los modelos")
def compare_models(
    service: Service,
    split: Annotated[Literal["validation", "test"], Query(description="Particion de evaluacion")] = "test",
) -> dict[str, Any]:
    """Tabla comparativa ordenada por ROC-AUC (AUC, PR-AUC, log loss, Brier, accuracy,
    precision, recall y F1 con el umbral de cada modelo)."""
    return service.model_comparison(split)


@router.get("/models/{name}", responses=_404, summary="Metricas de un modelo")
def model_metrics(model: Model, service: Service) -> dict[str, Any]:
    return service.model_metrics(model)


@router.get("/models/{name}/feature-importance", responses=_404, summary="Importancia de variables")
def feature_importance(model: Model, service: Service) -> dict[str, Any]:
    return service.feature_importance(model)


@router.get("/predictions", summary="Estadisticas de las predicciones servidas")
def prediction_stats(service: Service) -> dict[str, Any]:
    """Conteos y CTR medio predicho por modelo desde que arranco el servidor."""
    return service.prediction_stats()


@router.get("/predictions/recent", summary="Ultimas predicciones servidas")
def recent_predictions(
    service: Service, limit: Annotated[int, Query(ge=1, le=500)] = 20
) -> list[dict[str, Any]]:
    return service.recent_predictions(limit)
