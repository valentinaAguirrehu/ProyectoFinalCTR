"""Fixtures: modelos pequenos entrenados con datos sinteticos, independientes del dataset real."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.config import Settings
from src.api.main import create_app
from src.data.preprocessing import CATEGORICAL_FEATURES, to_model_frame
from src.ml.train import train_and_save


def synthetic_data(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    data = {"timestamp": rng.integers(0, 2_000_000, n)}
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.integers(0, 8, n) * 1000
    df = pd.DataFrame(data)
    logit = -0.6 + 1.5 * (df["campaign"] == 3000) - 1.0 * (df["cat1"] == 0) + 0.2 * rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8"))
    return to_model_frame(df), y


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("artifacts")
    models_dir, data_dir = root / "models", root / "data"
    X_tr, y_tr = synthetic_data(3000, 1)
    X_va, y_va = synthetic_data(1000, 2)
    X_te, y_te = synthetic_data(1000, 3)
    train_and_save("catboost", X_tr, y_tr, X_va, y_va, X_te, y_te, models_dir, iterations=50, verbose=0)
    train_and_save("logistic_regression", X_tr, y_tr, X_va, y_va, X_te, y_te, models_dir, min_frequency=1)

    broken = models_dir / "broken"
    broken.mkdir()
    (broken / "metadata.json").write_text(json.dumps({"name": "broken", "algorithm": "desconocido"}))

    data_dir.mkdir()
    manifest = Path(__file__).resolve().parents[1] / "data/processed/ctr_phase3/manifest.json"
    if manifest.is_file():
        (data_dir / "manifest.json").write_text(manifest.read_text())
    else:
        (data_dir / "manifest.json").write_text(json.dumps({
            "config": {"target": "click", "features": [], "categorical_features": [], "excluded_reasons": {},
                       "split_method": "test"},
            "temporal_cutoffs": {}, "split_summary": []}))
    return root


def make_settings(root: Path, **overrides) -> Settings:
    values = dict(models_dir=root / "models", data_dir=root / "data", default_model=None,
                  max_batch_size=5, prediction_log_size=50, cors_origins=["*"])
    return Settings(**{**values, **overrides})


@pytest.fixture()
def client(artifacts_dir) -> TestClient:
    with TestClient(create_app(make_settings(artifacts_dir))) as c:
        yield c


@pytest.fixture()
def empty_client(tmp_path) -> TestClient:
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


@pytest.fixture()
def ad() -> dict:
    return {"timestamp": 1_500_000, "campaign": 3000, "cat1": 1000, "cat2": 2000, "cat3": 0, "cat4": 4000,
            "cat5": 5000, "cat6": 6000, "cat7": 7000, "cat8": 0, "cat9": 1000}
