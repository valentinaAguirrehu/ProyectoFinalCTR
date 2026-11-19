"""Contrato de caracteristicas compartido entre el entrenamiento y la API.

Refleja las decisiones de la Fase 3 (ver docs/fase_3_preparacion_datos.md):
- Objetivo: ``click``.
- Predictores: ``timestamp``, ``campaign`` y ``cat1``-``cat9``.
- ``campaign`` y ``cat1``-``cat9`` son codigos categoricos enteros (sin one-hot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

TARGET = "click"
TEMPORAL_FEATURE = "timestamp"
CATEGORICAL_FEATURES: list[str] = ["campaign"] + [f"cat{i}" for i in range(1, 10)]
FEATURES: list[str] = [TEMPORAL_FEATURE] + CATEGORICAL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ctr_phase3"
SPLITS = ("train", "validation", "test")


def to_model_frame(records: Iterable[Mapping[str, int]] | pd.DataFrame) -> pd.DataFrame:
    """Construye un DataFrame con las columnas y tipos que esperan los modelos.

    Acepta un DataFrame o una secuencia de diccionarios. Lanza ``KeyError`` si
    falta alguna caracteristica.
    """
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    missing = [col for col in FEATURES if col not in frame.columns]
    if missing:
        raise KeyError(f"Faltan caracteristicas: {missing}")
    frame = frame[FEATURES].copy()
    frame[TEMPORAL_FEATURE] = frame[TEMPORAL_FEATURE].astype("int64")
    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("int64")
    return frame


def split_path(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> str:
    """Patron glob de los Parquet de una particion."""
    if split not in SPLITS:
        raise ValueError(f"Particion desconocida: {split}")
    return str(data_dir / f"split={split}" / "*.parquet")


def load_split(
    split: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    sample_rows: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Lee una particion con DuckDB (opcionalmente muestreada) y separa X / y."""
    import duckdb

    columns = ", ".join(FEATURES + [TARGET])
    query = f"SELECT {columns} FROM read_parquet('{split_path(split, data_dir)}')"
    if sample_rows:
        query += f" USING SAMPLE reservoir({int(sample_rows)} ROWS) REPEATABLE ({int(seed)})"
    with duckdb.connect() as con:
        df = con.sql(query).df()
    return to_model_frame(df), df[TARGET].astype("int8")
