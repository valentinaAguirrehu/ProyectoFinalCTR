"""Endpoints para consultar y recargar los modelos integrados."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_registry
from src.api.errors import ModelNotFoundError
from src.api.schemas import ErrorResponse, ModelDetail, ModelsResponse, ModelSummary
from src.ml.registry import LoadedModel, ModelRegistry

router = APIRouter(prefix="/models", tags=["Modelos"])
Registry = Annotated[ModelRegistry, Depends(get_registry)]


def _summary(model: LoadedModel, default: str | None) -> dict:
    md = model.metadata
    return dict(name=model.name, algorithm=md.get("algorithm", "unknown"), version=str(md.get("version", "unknown")),
                trained_at=md.get("trained_at"), threshold=model.threshold, is_default=model.name == default,
                validation_roc_auc=md.get("metrics", {}).get("validation", {}).get("roc_auc"))


def get_model_or_404(name: str, registry: Registry) -> LoadedModel:
    model = registry.models.get(name)
    if model is None:
        raise ModelNotFoundError({"message": f"Modelo '{name}' no encontrado", "available": sorted(registry.models)})
    return model


@router.get("", response_model=ModelsResponse, summary="Listar modelos cargados")
def list_models(registry: Registry) -> ModelsResponse:
    default = registry.default_name
    return ModelsResponse(default_model=default,
                          models=[ModelSummary(**_summary(m, default)) for m in registry.models.values()],
                          load_errors=registry.load_errors)


@router.get("/{name}", response_model=ModelDetail, responses={404: {"model": ErrorResponse}},
            summary="Detalle de un modelo")
def get_model(model: Annotated[LoadedModel, Depends(get_model_or_404)], registry: Registry) -> ModelDetail:
    md = model.metadata
    return ModelDetail(**_summary(model, registry.default_name), features=md.get("features", []),
                       categorical_features=md.get("categorical_features", []),
                       training=md.get("training", {}), metrics=md.get("metrics", {}))


@router.post("/reload", response_model=ModelsResponse, summary="Recargar modelos desde disco")
def reload_models(registry: Registry) -> ModelsResponse:
    """Vuelve a leer ``models/`` sin reiniciar el servidor (p. ej. tras reentrenar)."""
    registry.load_all()
    return list_models(registry)
