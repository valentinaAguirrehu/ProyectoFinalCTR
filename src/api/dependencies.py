"""Inyeccion de dependencias: los servicios viven en ``app.state`` y se crean al arrancar."""

from __future__ import annotations

from fastapi import Request

from src.api.services.metrics_service import MetricsService
from src.api.services.prediction_service import PredictionService
from src.ml.registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


def get_metrics_service(request: Request) -> MetricsService:
    return request.app.state.metrics_service
