"""Entrena los modelos de CTR y genera los artefactos que consume la API.

Uso (desde la raiz del proyecto):

    python -m src.ml.train                         # catboost + regresion logistica
    python -m src.ml.train --models catboost xgboost --train-rows 2000000

Cada modelo se guarda en ``models/<nombre>/`` con su ``metadata.json``. El umbral
de decision se ajusta en validacion; la particion de prueba solo se usa para
reportar metricas finales.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.preprocessing import DEFAULT_DATA_DIR, PROJECT_ROOT, load_split
from src.ml.evaluation import best_f1_threshold, evaluate
from src.ml.models import ALGORITHMS, CTRModel

log = logging.getLogger("ctr.train")
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def train_and_save(
    algorithm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame | None,
    y_test: pd.Series | None,
    models_dir: Path,
    name: str | None = None,
    version: str = "1.0.0",
    **params,
) -> dict:
    """Entrena un modelo, calcula umbral y metricas, y guarda el artefacto."""
    name = name or algorithm
    model: CTRModel = ALGORITHMS[algorithm]()
    start = time.perf_counter()
    model.fit(X_train, y_train, X_val, y_val, **params)
    elapsed = time.perf_counter() - start

    val_proba = model.predict_proba(X_val)
    threshold = best_f1_threshold(y_val.to_numpy(), val_proba)
    metrics = {"validation": evaluate(y_val.to_numpy(), val_proba, threshold)}
    if X_test is not None and y_test is not None:
        metrics["test"] = evaluate(y_test.to_numpy(), model.predict_proba(X_test), threshold)

    metadata = {
        "name": name,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "training": {"train_rows": int(len(X_train)), "validation_rows": int(len(X_val)),
                     "train_ctr": float(y_train.mean()), "fit_seconds": round(elapsed, 2),
                     "params": {k: v for k, v in params.items()}},
        "metrics": metrics,
    }
    model.save(models_dir / name, metadata)
    log.info("Modelo %s guardado. AUC val=%.4f", name, metrics["validation"]["roc_auc"])
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["catboost", "logistic_regression"],
                        choices=sorted(ALGORITHMS))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--train-rows", type=int, default=1_000_000,
                        help="Muestra de entrenamiento (0 = particion completa)")
    parser.add_argument("--eval-rows", type=int, default=500_000,
                        help="Muestra de validacion y prueba (0 = completas)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Cargando datos desde %s", args.data_dir)
    X_tr, y_tr = load_split("train", args.data_dir, args.train_rows or None, args.seed)
    X_va, y_va = load_split("validation", args.data_dir, args.eval_rows or None, args.seed)
    X_te, y_te = load_split("test", args.data_dir, args.eval_rows or None, args.seed)
    log.info("Filas: train=%d validation=%d test=%d", len(X_tr), len(X_va), len(X_te))

    for algorithm in args.models:
        log.info("Entrenando %s", algorithm)
        try:
            train_and_save(algorithm, X_tr, y_tr, X_va, y_va, X_te, y_te, args.models_dir)
        except ImportError as exc:
            log.error("No se pudo entrenar %s: %s", algorithm, exc)
        except Exception as exc:  # p. ej. XGBoost sin libomp en macOS
            log.error("Fallo el entrenamiento de %s: %s", algorithm, exc)


if __name__ == "__main__":
    main()
