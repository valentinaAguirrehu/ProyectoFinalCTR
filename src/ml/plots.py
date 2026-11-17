"""Figuras de evaluacion de los modelos de CTR (ROC, PR, matrices de confusion).

Todas las funciones reciben probabilidades ya calculadas y guardan PNG en
``reports/figures/``. Estilo sobrio y apto para el informe academico.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", font_scale=1.05)
MODEL_COLORS = {"xgboost": "#1f77b4", "catboost": "#d62728", "logistic_regression": "#2ca02c"}


def _prepare_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path) -> Path:
    """Curvas ROC superpuestas. ``curves``: nombre -> (fpr, tpr, auc)."""
    path = _prepare_dir(path)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC = 0.500)")
    for name, (fpr, tpr, score) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})",
                color=MODEL_COLORS.get(name, None), linewidth=2)
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Curvas ROC — XGBoost vs CatBoost (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
                   baseline: float, path: Path) -> Path:
    """Curvas Precision-Recall superpuestas con la prevalencia como referencia."""
    path = _prepare_dir(path)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Prevalencia = {baseline:.3f}")
    for name, (precision, recall, ap) in curves.items():
        ax.plot(recall, precision, label=f"{name} (AP = {ap:.4f})",
                color=MODEL_COLORS.get(name, None), linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall (test)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                          model_name: str, path: Path) -> Path:
    """Matriz de confusion normalizada por fila + conteos, con el umbral del modelo."""
    path = _prepare_dir(path)
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{v:,d}\n({r:.1%})" for v, r in zip(row, nrow)]
                      for row, nrow in zip(cm, np.nan_to_num(cm_norm))])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicho 0", "Predicho 1"], yticklabels=["Real 0", "Real 1"])
    ax.set_title(f"Matriz de confusion — {model_name}\n(umbral = {threshold:.3f}, test)")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_bars(metrics: dict[str, dict[str, float]], metric_names: list[str], path: Path) -> Path:
    """Barras agrupadas por metrica para comparar modelos lado a lado."""
    path = _prepare_dir(path)
    names = list(metrics)
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * width,
               [metrics[name].get(m, 0.0) for m in metric_names], width=width,
               label=name, color=MODEL_COLORS.get(name, None))
    ax.set_xticks(x, metric_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor")
    ax.set_title("Comparacion de metricas (test, umbral ajustado en validacion)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(importances: dict[str, dict[str, float]], path: Path,
                            top_n: int = 11) -> Path:
    """Barras horizontales de importancia (ya normalizada a suma 1 en models.py)."""
    path = _prepare_dir(path)
    fig, axes = plt.subplots(1, len(importances), figsize=(5 * len(importances), 4.5),
                             sharey=True, squeeze=False)
    for ax, (name, imp) in zip(axes.flat, importances.items()):
        items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        feats, vals = [k for k, _ in items][::-1], [v for _, v in items][::-1]
        ax.barh(feats, vals, color=MODEL_COLORS.get(name, None))
        ax.set_title(f"Importancia — {name}")
        ax.set_xlabel("Importancia normalizada")
    axes.flat[0].set_ylabel("Variable")
    fig.suptitle("Importancia de variables por modelo")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_curve_points(y_true: np.ndarray, proba: np.ndarray,
                     max_points: int = 500) -> tuple[list[float], list[float], float]:
    """Puntos ROC (diezmados) + AUC, listos para serializar a JSON."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    score = float(auc(fpr, tpr))
    idx = np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int)
    return [float(v) for v in fpr[idx]], [float(v) for v in tpr[idx]], score


def pr_curve_points(y_true: np.ndarray, proba: np.ndarray,
                    max_points: int = 500) -> tuple[list[float], list[float], float]:
    """Puntos Precision-Recall (diezmados) + AP, listos para serializar a JSON."""
    from sklearn.metrics import average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = float(average_precision_score(y_true, proba))
    idx = np.linspace(0, len(precision) - 1, min(max_points, len(precision))).astype(int)
    return [float(v) for v in recall[idx]], [float(v) for v in precision[idx]], ap
