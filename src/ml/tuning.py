"""Optimizacion de hiperparametros con Optuna para XGBoost y CatBoost.

Uso (desde la raiz del proyecto):

    python -m src.ml.tuning --algorithm xgboost --train-rows 200000 --eval-rows 50000 --n-trials 20
    python -m src.ml.tuning --algorithm catboost --train-rows 200000 --eval-rows 50000 --n-trials 20

El objetivo es el ROC-AUC sobre la particion de validacion. El umbral de
decision NO se optimiza aqui (se ajusta por F1 en ``src.ml.train`` y
``src.ml.evaluation``). La particion de prueba nunca se usa en esta fase.

Diseno de los espacios de busqueda
----------------------------------
* ``scale_pos_weight`` de XGBoost se fija a neg/pos del entrenamiento en cada
  estudio (no se muestrea) para compensar el desbalance (~36 % de clics).
* CatBoost recibe las categoricas por nombre (ver Fase 3) y usa
  ``early_stopping`` sobre validacion para evitar sobreajuste y recortar
  el costo de cada trial.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.preprocessing import DEFAULT_DATA_DIR, PROJECT_ROOT, load_split

log = logging.getLogger("ctr.tuning")
DEFAULT_TUNING_DIR = PROJECT_ROOT / "reports" / "tuning"


# ---------------------------------------------------------------- espacios
def suggest_xgboost(trial: Any) -> dict[str, Any]:
    """Espacio de busqueda para XGBoost (tree_method=hist, soporte categorico)."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def suggest_catboost(trial: Any) -> dict[str, Any]:
    """Espacio de busqueda para CatBoost (manejo nativo de categoricas)."""
    return {
        "iterations": trial.suggest_int("iterations", 300, 900, step=100),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
    }


SUGGESTERS = {"xgboost": suggest_xgboost, "catboost": suggest_catboost}


# ---------------------------------------------------------------- estudio
def _fit_and_score(
    algorithm: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Entrena con ``params`` y devuelve el ROC-AUC en validacion."""
    from src.ml.models import ALGORITHMS

    model = ALGORITHMS[algorithm]()
    model.fit(X_train, y_train, X_val, y_val, **params)
    proba = model.predict_proba(X_val)
    return float(roc_auc_score(y_val.to_numpy(), np.asarray(proba)))


def run_study(
    algorithm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 20,
    seed: int = 42,
    output: Path | None = None,
) -> dict[str, Any]:
    """Ejecuta el estudio Optuna y devuelve ``best_params``, ``best_score`` e historial."""
    import optuna

    if algorithm not in SUGGESTERS:
        raise ValueError(f"Algoritmo no soportado para tuning: {algorithm!r}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    base_extra: dict[str, Any] = {}
    if algorithm == "xgboost":
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        base_extra["scale_pos_weight"] = (neg / pos) if pos else 1.0
        log.info("scale_pos_weight fijado a %.3f (neg=%d pos=%d)", base_extra["scale_pos_weight"], neg, pos)

    def objective(trial: Any) -> float:
        params = {**SUGGESTERS[algorithm](trial), **base_extra}
        try:
            return _fit_and_score(algorithm, params, X_train, y_train, X_val, y_val)
        except Exception as exc:  # un trial fallido no debe tumbar el estudio
            log.warning("Trial %d fallido (%s): %s", trial.number, algorithm, exc)
            raise optuna.TrialPruned() from exc

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    result = {
        "algorithm": algorithm,
        "seed": seed,
        "n_trials": n_trials,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "best_score": float(study.best_value),
        "best_params": {**study.best_params, **base_extra},
        "trials": [
            {"number": t.number, "value": float(t.value) if t.value is not None else None,
             "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log.info("Mejor %s: AUC=%.4f -> %s", algorithm, result["best_score"], output)
    else:
        log.info("Mejor %s: AUC=%.4f params=%s", algorithm, result["best_score"], result["best_params"])
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", required=True, choices=sorted(SUGGESTERS))
    parser.add_argument("--train-rows", type=int, default=200_000)
    parser.add_argument("--eval-rows", type=int, default=50_000)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Ruta del JSON de salida (por defecto reports/tuning/<algo>.json)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output = args.output or (DEFAULT_TUNING_DIR / f"{args.algorithm}.json")
    log.info("Cargando datos: train=%s val=%s (semilla %d)", args.train_rows, args.eval_rows, args.seed)
    X_tr, y_tr = load_split("train", args.data_dir, args.train_rows or None, args.seed)
    X_va, y_va = load_split("validation", args.data_dir, args.eval_rows or None, args.seed)
    return run_study(args.algorithm, X_tr, y_tr, X_va, y_va, args.n_trials, args.seed, output)


if __name__ == "__main__":
    main()
