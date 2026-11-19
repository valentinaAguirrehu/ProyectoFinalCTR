"""Configuracion de la API mediante variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.data.preprocessing import DEFAULT_DATA_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    data_dir: Path
    default_model: str | None
    max_batch_size: int
    prediction_log_size: int
    cors_origins: list[str]

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            models_dir=Path(os.getenv("CTR_MODELS_DIR", PROJECT_ROOT / "models")),
            data_dir=Path(os.getenv("CTR_DATA_DIR", DEFAULT_DATA_DIR)),
            default_model=os.getenv("CTR_DEFAULT_MODEL") or None,
            max_batch_size=int(os.getenv("CTR_MAX_BATCH_SIZE", "1000")),
            prediction_log_size=int(os.getenv("CTR_PREDICTION_LOG_SIZE", "500")),
            cors_origins=[o.strip() for o in os.getenv("CTR_CORS_ORIGINS", "*").split(",") if o.strip()],
        )
