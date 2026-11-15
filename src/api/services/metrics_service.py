"""Servicio de consulta de metricas y resultados: dataset, modelos y uso de la API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.api.errors import ResourceNotFoundError
from src.api.services.prediction_log import PredictionLog
from src.ml.registry import LoadedModel, ModelRegistry

COMPARISON_METRICS = ("roc_auc", "pr_auc", "log_loss", "brier", "accuracy", "precision", "recall", "f1")


class MetricsService:
    def __init__(self, registry: ModelRegistry, prediction_log: PredictionLog, data_dir: Path):
        self.registry = registry
        self.prediction_log = prediction_log
        self.data_dir = data_dir

    def dataset_summary(self) -> dict[str, Any]:
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ResourceNotFoundError(f"No se encontro el manifiesto de datos en {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        return {
            "target": manifest["config"]["target"],
            "features": manifest["config"]["features"],
            "categorical_features": manifest["config"]["categorical_features"],
            "excluded_features": manifest["config"]["excluded_reasons"],
            "split_method": manifest["config"]["split_method"],
            "temporal_cutoffs": manifest["temporal_cutoffs"],
            "splits": manifest["split_summary"],
        }

    def model_comparison(self, split: str) -> dict[str, Any]:
        rows = []
        for model in self.registry.models.values():
            metrics = model.metadata.get("metrics", {}).get(split)
            if metrics:
                rows.append({"model": model.name, "algorithm": model.metadata.get("algorithm"),
                             "threshold": model.threshold,
                             **{k: metrics.get(k) for k in COMPARISON_METRICS}})
        rows.sort(key=lambda r: r["roc_auc"] or 0.0, reverse=True)
        return {"split": split, "default_model": self.registry.default_name,
                "best_model_by_roc_auc": rows[0]["model"] if rows else None, "models": rows}

    def model_metrics(self, model: LoadedModel) -> dict[str, Any]:
        return {"model": model.name, "threshold": model.threshold,
                "metrics": model.metadata.get("metrics", {}),
                "training": model.metadata.get("training", {})}

    def feature_importance(self, model: LoadedModel) -> dict[str, Any]:
        importance = model.model.feature_importance()
        if importance is None:
            raise ResourceNotFoundError(
                f"El modelo '{model.name}' ({model.metadata.get('algorithm')}) no expone importancia de variables")
        ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        return {"model": model.name,
                "feature_importance": [{"feature": f, "importance": v} for f, v in ranked]}

    def prediction_stats(self) -> dict[str, Any]:
        return self.prediction_log.summary()

    def recent_predictions(self, limit: int) -> list[dict[str, Any]]:
        return self.prediction_log.recent(limit)
