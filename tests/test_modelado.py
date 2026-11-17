"""Pruebas de la fase de modelado (Jesus): tuning, comparacion y figuras.

Usan datos sinteticos: no requieren el dataset Criteo ni entrenan modelos
pesados (solo XGBoost/CatBoost diminutos y logistic regression).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import CATEGORICAL_FEATURES, to_model_frame
from src.ml.compare import select_best
from src.ml.evaluation import best_f1_threshold, evaluate
from src.ml.plots import (
    plot_confusion_matrix,
    plot_metric_bars,
    plot_roc_curves,
)
from src.ml.tuning import SUGGESTERS, suggest_catboost, suggest_xgboost


def _synthetic(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    data = {"timestamp": rng.integers(0, 2_000_000, n)}
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.integers(0, 5, n) * 1000
    df = pd.DataFrame(data)
    logit = -0.5 + 1.5 * (df["campaign"] == 3000) + 0.2 * rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8"))
    return to_model_frame(df), y


class _FakeTrial:
    """Trial minimo compatible con los suggest_* (valores medios del rango)."""

    def suggest_int(self, name, low, high, step=1):
        return (low + high) // 2 if step == 1 else low + ((high - low) // (2 * step)) * step

    def suggest_float(self, name, low, high, log=False):
        return float(np.sqrt(low * high)) if log else (low + high) / 2


# ------------------------------------------------------------- evaluation
def test_evaluate_returns_all_required_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8])
    m = evaluate(y_true, proba, threshold=0.5)
    for key in ["roc_auc", "pr_auc", "log_loss", "brier", "accuracy",
                "precision", "recall", "f1"]:
        assert key in m
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_best_f1_threshold_maximizes_f1_on_validation():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    proba = rng.random(500)
    thr = best_f1_threshold(y_true, proba)
    assert 0.0 < thr < 1.0
    from sklearn.metrics import f1_score

    grid = np.linspace(0.05, 0.95, 19)
    best_grid = max(f1_score(y_true, proba >= t) for t in grid)
    assert f1_score(y_true, proba >= thr) >= best_grid - 1e-9


# ----------------------------------------------------------------- tuning
def test_suggesters_cover_both_algorithms():
    assert set(SUGGESTERS) == {"xgboost", "catboost"}


def test_suggest_xgboost_keys():
    params = suggest_xgboost(_FakeTrial())
    assert {"n_estimators", "max_depth", "learning_rate", "min_child_weight",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"} <= set(params)


def test_suggest_catboost_keys():
    params = suggest_catboost(_FakeTrial())
    assert {"iterations", "depth", "learning_rate", "l2_leaf_reg",
            "bagging_temperature", "random_strength"} <= set(params)


# ----------------------------------------------------------------- compare
def test_select_best_uses_validation_auc_then_pr_auc():
    results = {
        "xgboost": {"metrics": {"validation": {"roc_auc": 0.70, "pr_auc": 0.55, "f1": 0.60}}},
        "catboost": {"metrics": {"validation": {"roc_auc": 0.72, "pr_auc": 0.50, "f1": 0.50}}},
    }
    assert select_best(results) == "catboost"
    tie = {
        "xgboost": {"metrics": {"validation": {"roc_auc": 0.72, "pr_auc": 0.50, "f1": 0.50}}},
        "catboost": {"metrics": {"validation": {"roc_auc": 0.72, "pr_auc": 0.58, "f1": 0.50}}},
    }
    assert select_best(tie) == "catboost"


# ------------------------------------------------------------------- plots
def test_figures_are_saved(tmp_path):
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8, 0.3, 0.7])
    fpr = np.array([0.0, 0.25, 1.0])
    tpr = np.array([0.0, 0.75, 1.0])
    p1 = plot_roc_curves({"xgboost": (fpr, tpr, 0.8)}, tmp_path / "roc.png")
    p2 = plot_confusion_matrix(y_true, proba, 0.5, "xgboost", tmp_path / "cm.png")
    p3 = plot_metric_bars({"xgboost": {"roc_auc": 0.8, "f1": 0.7}},
                          ["roc_auc", "f1"], tmp_path / "bars.png")
    for p in (p1, p2, p3):
        assert p.is_file() and p.stat().st_size > 0
