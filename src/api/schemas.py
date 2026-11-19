"""Esquemas Pydantic de entrada y salida de la API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_CAT = dict(ge=0, le=2_147_483_647)

# Registro real de la particion de prueba (sin la etiqueta).
AD_EXAMPLE = {
    "timestamp": 2288076,
    "campaign": 787943,
    "cat1": 30763035,
    "cat2": 32440040,
    "cat3": 11426249,
    "cat4": 29196072,
    "cat5": 32440044,
    "cat6": 1973606,
    "cat7": 25459503,
    "cat8": 32440044,
    "cat9": 29196072,
}


class AdFeatures(BaseModel):
    """Caracteristicas de una impresion de anuncio (codigos anonimizados de Criteo)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": AD_EXAMPLE})

    timestamp: int = Field(..., ge=0, description="Segundos desde el inicio del periodo del dataset")
    campaign: int = Field(..., description="Codigo de la campana", **_CAT)
    cat1: int = Field(..., description="Caracteristica contextual categorica 1", **_CAT)
    cat2: int = Field(..., description="Caracteristica contextual categorica 2", **_CAT)
    cat3: int = Field(..., description="Caracteristica contextual categorica 3", **_CAT)
    cat4: int = Field(..., description="Caracteristica contextual categorica 4", **_CAT)
    cat5: int = Field(..., description="Caracteristica contextual categorica 5", **_CAT)
    cat6: int = Field(..., description="Caracteristica contextual categorica 6", **_CAT)
    cat7: int = Field(..., description="Caracteristica contextual categorica 7", **_CAT)
    cat8: int = Field(..., description="Caracteristica contextual categorica 8", **_CAT)
    cat9: int = Field(..., description="Caracteristica contextual categorica 9", **_CAT)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"ads": [AD_EXAMPLE]}})

    ads: list[AdFeatures] = Field(..., min_length=1, description="Lista de anuncios a evaluar")


class Prediction(BaseModel):
    ctr_probability: float = Field(..., ge=0, le=1, description="Probabilidad estimada de clic")
    predicted_click: bool = Field(..., description="ctr_probability >= threshold")
    threshold: float = Field(..., description="Umbral de decision ajustado en validacion")


class PredictionResponse(Prediction):
    request_id: str
    model: str
    model_version: str
    created_at: datetime


class BatchPredictionResponse(BaseModel):
    request_id: str
    model: str
    model_version: str
    created_at: datetime
    count: int
    mean_ctr_probability: float
    predicted_clicks: int
    predictions: list[Prediction]


class ModelSummary(BaseModel):
    name: str
    algorithm: str
    version: str
    trained_at: str | None
    threshold: float
    is_default: bool
    validation_roc_auc: float | None


class ModelDetail(ModelSummary):
    features: list[str]
    categorical_features: list[str]
    training: dict[str, Any]
    metrics: dict[str, dict[str, float]]


class ModelsResponse(BaseModel):
    default_model: str | None
    models: list[ModelSummary]
    load_errors: dict[str, str]


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' si hay al menos un modelo cargado, 'degraded' si no")
    models_loaded: int
    default_model: str | None
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    request_id: str | None = None
