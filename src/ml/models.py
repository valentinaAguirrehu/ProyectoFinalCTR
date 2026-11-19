"""Envoltorios de modelos de CTR con una interfaz comun.

Cada modelo se guarda en su propia carpeta ``models/<nombre>/`` con:
- el archivo del modelo (formato nativo de cada libreria), y
- ``metadata.json`` con algoritmo, umbral de decision, caracteristicas y metricas.

La API solo depende de esta interfaz (``predict_proba``), por lo que cualquier
modelo nuevo de la fase de modelado puede integrarse implementando una subclase
de ``CTRModel`` y registrandola en ``ALGORITHMS``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.data.preprocessing import CATEGORICAL_FEATURES, FEATURES, TEMPORAL_FEATURE

METADATA_FILE = "metadata.json"


class CTRModel(ABC):
    """Interfaz comun: recibe un DataFrame con ``FEATURES`` y devuelve P(click=1)."""

    algorithm: ClassVar[str]
    artifact_file: ClassVar[str]

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None, **params: Any) -> "CTRModel":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def feature_importance(self) -> dict[str, float] | None:
        """Importancia normalizada (suma 1) por caracteristica, si el modelo la ofrece."""
        return None

    @staticmethod
    def _normalize(values: Any) -> dict[str, float]:
        values = np.asarray(values, dtype=float)
        total = values.sum() or 1.0
        return {f: float(v / total) for f, v in zip(FEATURES, values)}

    @abstractmethod
    def _save_estimator(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def _load_estimator(cls, path: Path) -> "CTRModel":
        ...

    def save(self, directory: Path, metadata: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._save_estimator(directory / self.artifact_file)
        metadata = {**metadata, "algorithm": self.algorithm, "artifact": self.artifact_file,
                    "features": FEATURES, "categorical_features": CATEGORICAL_FEATURES}
        (directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    @staticmethod
    def load(directory: Path) -> tuple["CTRModel", dict[str, Any]]:
        metadata = json.loads((directory / METADATA_FILE).read_text())
        algorithm = metadata.get("algorithm")
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Algoritmo no soportado: {algorithm!r}")
        model_cls = ALGORITHMS[algorithm]
        return model_cls._load_estimator(directory / metadata["artifact"]), metadata


class CatBoostCTR(CTRModel):
    algorithm = "catboost"
    artifact_file = "model.cbm"

    def fit(self, X, y, X_val=None, y_val=None, **params):
        from catboost import CatBoostClassifier

        defaults = dict(iterations=400, learning_rate=0.1, depth=6, loss_function="Logloss",
                        eval_metric="AUC", random_seed=42, verbose=100, allow_writing_files=False)
        self.estimator = CatBoostClassifier(**{**defaults, **params})
        eval_set = (X_val, y_val) if X_val is not None else None
        self.estimator.fit(X, y, cat_features=CATEGORICAL_FEATURES, eval_set=eval_set,
                           early_stopping_rounds=50 if eval_set else None)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[FEATURES])[:, 1]

    def feature_importance(self):
        return self._normalize(self.estimator.get_feature_importance())

    def _save_estimator(self, path):
        self.estimator.save_model(str(path))

    @classmethod
    def _load_estimator(cls, path):
        from catboost import CatBoostClassifier

        estimator = CatBoostClassifier()
        estimator.load_model(str(path))
        return cls(estimator)


class XGBoostCTR(CTRModel):
    """XGBoost con soporte categorico nativo. Guarda las categorias vistas en
    entrenamiento para codificar igual en inferencia (las no vistas pasan a NaN)."""

    algorithm = "xgboost"
    artifact_file = "model.json"
    categories_file = "categories.json"

    def __init__(self, estimator: Any = None, categories: dict[str, list[int]] | None = None):
        super().__init__(estimator)
        self.categories = categories or {}

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = pd.Categorical(X[col], categories=self.categories[col])
        return X

    def fit(self, X, y, X_val=None, y_val=None, **params):
        from xgboost import XGBClassifier

        self.categories = {col: sorted(int(v) for v in X[col].unique()) for col in CATEGORICAL_FEATURES}
        defaults = dict(n_estimators=400, learning_rate=0.1, max_depth=8, tree_method="hist",
                        enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
                        random_state=42, n_jobs=-1)
        if X_val is not None:
            defaults["early_stopping_rounds"] = 50
        self.estimator = XGBClassifier(**{**defaults, **params})
        eval_set = [(self._encode(X_val), y_val)] if X_val is not None else None
        self.estimator.fit(self._encode(X), y, eval_set=eval_set, verbose=100)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._encode(X))[:, 1]

    def feature_importance(self):
        return self._normalize(self.estimator.feature_importances_)

    def _save_estimator(self, path):
        self.estimator.save_model(str(path))
        (path.parent / self.categories_file).write_text(json.dumps(self.categories))

    @classmethod
    def _load_estimator(cls, path):
        from xgboost import XGBClassifier

        estimator = XGBClassifier()
        estimator.load_model(str(path))
        categories = json.loads((path.parent / cls.categories_file).read_text())
        return cls(estimator, categories)


class LogisticCTR(CTRModel):
    """Linea base: regresion logistica con one-hot de las categorias frecuentes."""

    algorithm = "logistic_regression"
    artifact_file = "model.joblib"

    def fit(self, X, y, X_val=None, y_val=None, **params):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        preprocess = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=params.pop("min_frequency", 50)), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), [TEMPORAL_FEATURE]),
        ])
        clf = LogisticRegression(max_iter=params.pop("max_iter", 300), C=params.pop("C", 1.0), **params)
        self.estimator = Pipeline([("preprocess", preprocess), ("clf", clf)]).fit(X[FEATURES], y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X[FEATURES])[:, 1]

    def _save_estimator(self, path):
        import joblib

        joblib.dump(self.estimator, path)

    @classmethod
    def _load_estimator(cls, path):
        import joblib

        return cls(joblib.load(path))


ALGORITHMS: dict[str, type[CTRModel]] = {
    m.algorithm: m for m in (CatBoostCTR, XGBoostCTR, LogisticCTR)
}
