"""Endpoints de prediccion de CTR."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_prediction_service
from src.api.schemas import AdFeatures, BatchPredictionResponse, BatchRequest, ErrorResponse, PredictionResponse
from src.api.services.prediction_service import PredictionService

router = APIRouter(prefix="/predict", tags=["Prediccion"])

ModelQuery = Annotated[
    str | None,
    Query(description="Nombre del modelo a usar. Si se omite, se usa el modelo por defecto.",
          examples=["catboost"]),
]
Service = Annotated[PredictionService, Depends(get_prediction_service)]

_ERRORS = {
    404: {"model": ErrorResponse, "description": "Modelo no encontrado"},
    422: {"model": ErrorResponse, "description": "Datos de entrada no validos"},
    503: {"model": ErrorResponse, "description": "No hay modelos cargados"},
}


@router.post("", response_model=PredictionResponse, responses=_ERRORS,
             summary="Predecir el CTR de un anuncio")
def predict(ad: AdFeatures, service: Service, model: ModelQuery = None) -> PredictionResponse:
    """Devuelve la probabilidad de clic de una impresion y la decision binaria segun el
    umbral del modelo (ajustado para maximizar F1 en la particion de validacion)."""
    return service.predict_one(ad, model)


@router.post("/batch", response_model=BatchPredictionResponse,
             responses={**_ERRORS, 413: {"model": ErrorResponse, "description": "Lote demasiado grande"}},
             summary="Predecir el CTR de un lote de anuncios")
def predict_batch(body: BatchRequest, service: Service, model: ModelQuery = None) -> BatchPredictionResponse:
    """Evalua varios anuncios en una sola llamada. Las predicciones se devuelven en el
    mismo orden que la entrada. El tamano maximo se configura con ``CTR_MAX_BATCH_SIZE``."""
    return service.predict_batch(body.ads, model)
