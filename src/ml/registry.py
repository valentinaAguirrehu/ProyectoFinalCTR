"""Registro de modelos: descubre y carga los artefactos de ``models/<nombre>/``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.ml.models import METADATA_FILE, CTRModel

log = logging.getLogger("ctr.registry")


@dataclass
class LoadedModel:
    name: str
    model: CTRModel
    metadata: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    @property
    def validation_auc(self) -> float:
        return float(self.metadata.get("metrics", {}).get("validation", {}).get("roc_auc", 0.0))


@dataclass
class ModelRegistry:
    models_dir: Path
    preferred_default: str | None = None
    models: dict[str, LoadedModel] = field(default_factory=dict)
    load_errors: dict[str, str] = field(default_factory=dict)

    def load_all(self) -> "ModelRegistry":
        """Carga todos los artefactos. Se construye un diccionario nuevo y se reemplaza
        al final, para que una recarga no deje la API sin modelos a mitad de camino."""
        models: dict[str, LoadedModel] = {}
        errors: dict[str, str] = {}
        if not self.models_dir.is_dir():
            log.warning("No existe el directorio de modelos %s", self.models_dir)
        else:
            for directory in sorted(p for p in self.models_dir.iterdir() if (p / METADATA_FILE).is_file()):
                try:
                    model, metadata = CTRModel.load(directory)
                    name = metadata.get("name", directory.name)
                    models[name] = LoadedModel(name, model, metadata)
                    log.info("Modelo cargado: %s (%s)", name, metadata.get("algorithm"))
                except Exception as exc:  # un artefacto danado no debe tumbar la API
                    errors[directory.name] = f"{type(exc).__name__}: {exc}"
                    log.error("No se pudo cargar %s: %s", directory.name, exc)
        self.models, self.load_errors = models, errors
        return self

    @property
    def default_name(self) -> str | None:
        """Modelo por defecto: el configurado o, si no, el de mayor AUC en validacion."""
        if self.preferred_default in self.models:
            return self.preferred_default
        if not self.models:
            return None
        return max(self.models.values(), key=lambda m: m.validation_auc).name

    def get(self, name: str | None = None) -> LoadedModel | None:
        name = name or self.default_name
        return self.models.get(name) if name else None
