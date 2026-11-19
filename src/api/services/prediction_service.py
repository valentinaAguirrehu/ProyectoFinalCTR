"""Servicio de prediccion de CTR: valida el modelo, ejecuta la inferencia y la registra."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np

from src.api.errors import BatchTooLargeError, ModelNotFoundError, NoModelsAvailableError, PredictionError
from src.api.schemas import AdFeatures, BatchPredictionResponse, Prediction, PredictionResponse
from src.api.services.prediction_log import PredictionLog
from src.data.preprocessing import to_model_frame
from src.ml.registry import LoadedModel, ModelRegistry

log = logging.getLogger("ctr.prediction")


class PredictionService:
    def __init__(self, registry: ModelRegistry, prediction_log: PredictionLog, max_batch_size: int):
        self.registry = registry
        self.prediction_log = prediction_log
        self.max_batch_size = max_batch_size

    def resolve_model(self, name: str | None) -> LoadedModel:
        if not self.registry.models:
            raise NoModelsAvailableError(
                "No hay modelos cargados. Entrene con 'python -m src.ml.train' y reinicie la API "
                "o llame a POST /api/v1/models/reload.")
        model = self.registry.get(name)
        if model is None:
            raise ModelNotFoundError(
                {"message": f"Modelo '{name}' no encontrado", "available": sorted(self.registry.models)})
        return model

    def _infer(self, model: LoadedModel, ads: list[AdFeatures]) -> np.ndarray:
        try:
            frame = to_model_frame([ad.model_dump() for ad in ads])
            proba = np.asarray(model.model.predict_proba(frame), dtype=float)
        except Exception as exc:
            log.exception("Fallo la inferencia con el modelo %s", model.name)
            raise PredictionError(f"Fallo la inferencia con el modelo '{model.name}'") from exc
        if not np.all(np.isfinite(proba)):
            raise PredictionError("El modelo devolvio probabilidades no validas")
        return np.clip(proba, 0.0, 1.0)

    def predict_one(self, ad: AdFeatures, model_name: str | None = None) -> PredictionResponse:
        model = self.resolve_model(model_name)
        proba = float(self._infer(model, [ad])[0])
        response = PredictionResponse(
            request_id=uuid.uuid4().hex,
            model=model.name,
            model_version=str(model.metadata.get("version", "unknown")),
            created_at=datetime.now(timezone.utc),
            ctr_probability=proba,
            predicted_click=proba >= model.threshold,
            threshold=model.threshold,
        )
        self.prediction_log.record(model.name, [proba], model.threshold, response.request_id, kind="single")
        return response

    def predict_batch(self, ads: list[AdFeatures], model_name: str | None = None) -> BatchPredictionResponse:
        if len(ads) > self.max_batch_size:
            raise BatchTooLargeError(
                f"El lote tiene {len(ads)} anuncios; el maximo permitido es {self.max_batch_size}")
        model = self.resolve_model(model_name)
        proba = self._infer(model, ads)
        predictions = [
            Prediction(ctr_probability=float(p), predicted_click=bool(p >= model.threshold),
                       threshold=model.threshold)
            for p in proba
        ]
        response = BatchPredictionResponse(
            request_id=uuid.uuid4().hex,
            model=model.name,
            model_version=str(model.metadata.get("version", "unknown")),
            created_at=datetime.now(timezone.utc),
            count=len(predictions),
            mean_ctr_probability=float(proba.mean()),
            predicted_clicks=int(sum(p.predicted_click for p in predictions)),
            predictions=predictions,
        )
        self.prediction_log.record(model.name, proba.tolist(), model.threshold, response.request_id, kind="batch")
        return response
