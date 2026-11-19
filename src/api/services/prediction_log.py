"""Registro en memoria de las predicciones servidas (para metricas operativas)."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class _ModelStats:
    requests: int = 0
    predictions: int = 0
    predicted_clicks: int = 0
    probability_sum: float = 0.0
    last_prediction_at: str | None = None


@dataclass
class PredictionLog:
    max_entries: int = 500
    _entries: deque = field(init=False)
    _stats: dict[str, _ModelStats] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._entries = deque(maxlen=self.max_entries)

    def record(self, model: str, probabilities: list[float], threshold: float, request_id: str,
               kind: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        clicks = sum(p >= threshold for p in probabilities)
        with self._lock:
            stats = self._stats.setdefault(model, _ModelStats())
            stats.requests += 1
            stats.predictions += len(probabilities)
            stats.predicted_clicks += clicks
            stats.probability_sum += sum(probabilities)
            stats.last_prediction_at = now
            self._entries.append({
                "request_id": request_id, "created_at": now, "model": model, "kind": kind,
                "count": len(probabilities), "mean_ctr_probability": sum(probabilities) / len(probabilities),
                "predicted_clicks": clicks,
            })

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_model = {
                name: {
                    "requests": s.requests,
                    "predictions": s.predictions,
                    "predicted_clicks": s.predicted_clicks,
                    "predicted_click_rate": s.predicted_clicks / s.predictions if s.predictions else 0.0,
                    "mean_ctr_probability": s.probability_sum / s.predictions if s.predictions else 0.0,
                    "last_prediction_at": s.last_prediction_at,
                }
                for name, s in self._stats.items()
            }
            return {
                "total_requests": sum(s.requests for s in self._stats.values()),
                "total_predictions": sum(s.predictions for s in self._stats.values()),
                "by_model": by_model,
            }

    def recent(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)[-limit:][::-1]
