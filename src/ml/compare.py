"""Comparacion final XGBoost vs CatBoost sobre la particion de prueba.

Uso (desde la raiz del proyecto):

    python -m src.ml.compare --models-dir models --eval-rows 200000
    python -m src.ml.compare --models xgboost catboost --eval-rows 0

Lee los artefactos de ``models/<nombre>/`` (v. ``src.ml.train``), calcula en
prueba Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Log-Loss y Brier con
el umbral ajustado en validacion, dibuja ROC/PR/matrices de confusion y elige
el mejor modelo por ROC-AUC de validacion (desempate: PR-AUC, luego F1).

Salidas (versionables en Git, sin datos personales)::

    reports/metrics/comparison.json
    reports/figures/roc_curves.png
    reports/figures/pr_curves.png
    reports/figures/confusion_matrix_<modelo>.png
    reports/figures/metrics_bars.png
    reports/figures/feature_importance.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.data.preprocessing import DEFAULT_DATA_DIR, PROJECT_ROOT, load_split
from src.ml.evaluation import evaluate
from src.ml.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metric_bars,
    plot_pr_curves,
    plot_roc_curves,
    pr_curve_points,
    roc_curve_points,
)
from src.ml.registry import ModelRegistry

log = logging.getLogger("ctr.compare")
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
BAR_METRICS = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]


def compare_models(
    models_dir: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    models: list[str] | None = None,
    eval_rows: int | None = 200_000,
    seed: int = 42,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict:
    """Evalua los modelos en prueba, genera figuras y devuelve el comparativo."""
    registry = ModelRegistry(models_dir).load_all()
    if not registry.models:
        raise FileNotFoundError(f"No hay modelos en {models_dir}. Entrene con 'python -m src.ml.train'.")

    wanted = models or sorted(registry.models)
    missing = [m for m in wanted if m not in registry.models]
    if missing:
        raise KeyError(f"Modelos no encontrados en {models_dir}: {missing}. Disponibles: {sorted(registry.models)}")

    X_te, y_te = load_split("test", data_dir, eval_rows or None, seed)
    y_true = y_te.to_numpy()
    log.info("Evaluando %s en prueba: %d filas", wanted, len(X_te))

    results: dict[str, dict] = {}
    roc_curves, pr_curves, importances, bar_metrics = {}, {}, {}, {}
    for name in wanted:
        loaded = registry.models[name]
        proba = np.asarray(loaded.model.predict_proba(X_te), dtype=float)
        metrics = evaluate(y_true, proba, loaded.threshold)
        pred = (proba >= loaded.threshold).astype(int)
        tn, fp, fn, tp = (((pred == 0) & (y_true == 0)).sum(), ((pred == 1) & (y_true == 0)).sum(),
                          ((pred == 0) & (y_true == 1)).sum(), ((pred == 1) & (y_true == 1)).sum())
        fpr, tpr, _ = roc_curve_points(y_true, proba)
        rec, prec, _ = pr_curve_points(y_true, proba)
        results[name] = {
            "algorithm": loaded.metadata.get("algorithm", name),
            "threshold": loaded.threshold,
            "metrics": {"validation": loaded.metadata.get("metrics", {}).get("validation", {}),
                        "test": metrics},
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc_curve": {"fpr": fpr, "tpr": tpr},
            "pr_curve": {"recall": rec, "precision": prec},
        }
        roc_curves[name] = (np.array(fpr), np.array(tpr), metrics["roc_auc"])
        pr_curves[name] = (np.array(prec), np.array(rec), metrics["pr_auc"])
        bar_metrics[name] = metrics
        importance = loaded.model.feature_importance()
        if importance:
            importances[name] = importance
        log.info("%s test: AUC=%.4f F1=%.4f acc=%.4f", name, metrics["roc_auc"], metrics["f1"], metrics["accuracy"])

    best = select_best(results)
    summary = {
        "test_rows": int(len(X_te)),
        "seed": seed,
        "best_model": best,
        "criterion": "roc_auc de validacion (desempate: pr_auc, f1)",
        "models": results,
    }

    figures_dir = reports_dir / "figures"
    plot_roc_curves(roc_curves, figures_dir / "roc_curves.png")
    plot_pr_curves(pr_curves, float(y_true.mean()), figures_dir / "pr_curves.png")
    plot_metric_bars(bar_metrics, BAR_METRICS, figures_dir / "metrics_bars.png")
    for name in wanted:
        loaded = registry.models[name]
        plot_confusion_matrix(y_true, np.asarray(loaded.model.predict_proba(X_te), dtype=float),
                              loaded.threshold, name, figures_dir / f"confusion_matrix_{name}.png")
    if importances:
        plot_feature_importance(importances, figures_dir / "feature_importance.png")

    metrics_path = reports_dir / "metrics" / "comparison.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("Mejor modelo: %s. Comparativo en %s", best, metrics_path)
    return summary


def select_best(results: dict[str, dict]) -> str:
    """Mejor modelo por ROC-AUC de validacion; desempata por PR-AUC y F1."""

    def key(name: str) -> tuple[float, float, float]:
        val = results[name]["metrics"]["validation"]
        return (float(val.get("roc_auc", 0.0)), float(val.get("pr_auc", 0.0)), float(val.get("f1", 0.0)))

    return max(results, key=key)


def print_markdown_table(summary: dict) -> str:
    """Tabla Markdown con las metricas de prueba (lista para el informe)."""
    header = "| Modelo | ROC-AUC | PR-AUC | Log-Loss | Acc | Prec | Rec | F1 | Umbral |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for name, info in sorted(summary["models"].items(),
                             key=lambda kv: kv[1]["metrics"]["test"]["roc_auc"], reverse=True):
        m = info["metrics"]["test"]
        star = " (mejor)" if name == summary["best_model"] else ""
        rows.append(f"| {name}{star} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['log_loss']:.4f} "
                    f"| {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} "
                    f"| {m['f1']:.4f} | {info['threshold']:.3f} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--eval-rows", type=int, default=200_000,
                        help="Filas de prueba (0 = particion completa)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = compare_models(args.models_dir, args.data_dir, args.models,
                             args.eval_rows or None, args.seed, args.reports_dir)
    print(print_markdown_table(summary))
    return summary


if __name__ == "__main__":
    main()
