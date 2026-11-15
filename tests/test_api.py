"""Pruebas de funcionamiento de la API de prediccion de CTR."""

from __future__ import annotations

import pytest

API = "/api/v1"


# ---------------------------------------------------------------- Sistema
def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == 2
    assert body["default_model"] in {"catboost", "logistic_regression"}


def test_health_degraded_without_models(empty_client):
    body = empty_client.get("/health").json()
    assert body == {**body, "status": "degraded", "models_loaded": 0, "default_model": None}


def test_request_id_header_is_propagated(client):
    response = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"
    assert "X-Process-Time-Ms" in response.headers


def test_openapi_documents_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ["/api/v1/predict", "/api/v1/predict/batch", "/api/v1/models", "/api/v1/metrics/models",
                 "/api/v1/metrics/dataset", "/api/v1/metrics/predictions"]:
        assert path in paths
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------- Prediccion
@pytest.mark.parametrize("model", [None, "catboost", "logistic_regression"])
def test_predict_single(client, ad, model):
    params = {"model": model} if model else {}
    response = client.post(f"{API}/predict", json=ad, params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["ctr_probability"] <= 1.0
    assert body["predicted_click"] == (body["ctr_probability"] >= body["threshold"])
    if model:
        assert body["model"] == model


def test_predict_is_deterministic(client, ad):
    first = client.post(f"{API}/predict", json=ad).json()["ctr_probability"]
    second = client.post(f"{API}/predict", json=ad).json()["ctr_probability"]
    assert first == pytest.approx(second)


def test_model_learned_signal(client, ad):
    """El generador sintetico aumenta el CTR cuando campaign == 3000."""
    high = client.post(f"{API}/predict", json=ad, params={"model": "catboost"}).json()["ctr_probability"]
    low = client.post(f"{API}/predict", json={**ad, "campaign": 5000},
                      params={"model": "catboost"}).json()["ctr_probability"]
    assert high > low


def test_predict_unseen_categories(client, ad):
    unseen = {**ad, "campaign": 999_999_999, "cat3": 123_456_789}
    for model in ["catboost", "logistic_regression"]:
        response = client.post(f"{API}/predict", json=unseen, params={"model": model})
        assert response.status_code == 200, response.text


def test_predict_batch_preserves_order_and_counts(client, ad):
    ads = [ad, {**ad, "campaign": 5000}, {**ad, "cat1": 0}]
    response = client.post(f"{API}/predict/batch", json={"ads": ads}, params={"model": "catboost"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 3 and len(body["predictions"]) == 3
    single = [client.post(f"{API}/predict", json=a, params={"model": "catboost"}).json()["ctr_probability"]
              for a in ads]
    assert [p["ctr_probability"] for p in body["predictions"]] == pytest.approx(single)
    assert body["predicted_clicks"] == sum(p["predicted_click"] for p in body["predictions"])


# ---------------------------------------------------------------- Validacion y errores
@pytest.mark.parametrize("mutation, field", [
    ({"campaign": None}, "campaign"),
    ({"cat1": -1}, "cat1"),
    ({"timestamp": "ayer"}, "timestamp"),
    ({"cat2": 1.5}, "cat2"),
    ({"extra_field": 1}, "extra_field"),
])
def test_predict_rejects_invalid_input(client, ad, mutation, field):
    response = client.post(f"{API}/predict", json={**ad, **mutation})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert any(field in err["field"] for err in body["detail"])
    assert body["request_id"]


def test_predict_rejects_missing_field(client, ad):
    ad.pop("cat9")
    response = client.post(f"{API}/predict", json=ad)
    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "cat9"


def test_predict_rejects_malformed_json(client):
    response = client.post(f"{API}/predict", content="{no es json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_predict_unknown_model(client, ad):
    response = client.post(f"{API}/predict", json=ad, params={"model": "no_existe"})
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "model_not_found"
    assert "catboost" in body["detail"]["available"]


def test_predict_without_models_returns_503(empty_client, ad):
    response = empty_client.post(f"{API}/predict", json=ad)
    assert response.status_code == 503
    assert response.json()["error"] == "no_models_available"


def test_batch_empty_and_too_large(client, ad):
    assert client.post(f"{API}/predict/batch", json={"ads": []}).status_code == 422
    response = client.post(f"{API}/predict/batch", json={"ads": [ad] * 6})  # max_batch_size=5
    assert response.status_code == 413
    assert response.json()["error"] == "batch_too_large"


def test_unknown_route_uses_error_format(client):
    response = client.get(f"{API}/no-existe")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


# ---------------------------------------------------------------- Modelos
def test_list_models_reports_broken_artifact(client):
    body = client.get(f"{API}/models").json()
    assert {m["name"] for m in body["models"]} == {"catboost", "logistic_regression"}
    assert sum(m["is_default"] for m in body["models"]) == 1
    assert "broken" in body["load_errors"]


def test_model_detail(client):
    body = client.get(f"{API}/models/catboost").json()
    assert body["algorithm"] == "catboost"
    assert body["features"][0] == "timestamp"
    assert {"validation", "test"} <= set(body["metrics"])
    assert client.get(f"{API}/models/no_existe").status_code == 404


def test_reload_models(client):
    response = client.post(f"{API}/models/reload")
    assert response.status_code == 200
    assert len(response.json()["models"]) == 2


def test_default_model_from_settings(artifacts_dir, ad):
    from fastapi.testclient import TestClient

    from src.api.main import create_app
    from tests.conftest import make_settings

    with TestClient(create_app(make_settings(artifacts_dir, default_model="logistic_regression"))) as c:
        assert c.post(f"{API}/predict", json=ad).json()["model"] == "logistic_regression"


# ---------------------------------------------------------------- Metricas
def test_metrics_comparison(client):
    body = client.get(f"{API}/metrics/models", params={"split": "validation"}).json()
    assert body["split"] == "validation"
    aucs = [m["roc_auc"] for m in body["models"]]
    assert aucs == sorted(aucs, reverse=True)
    assert body["best_model_by_roc_auc"] == body["models"][0]["model"]
    assert client.get(f"{API}/metrics/models", params={"split": "train"}).status_code == 422


def test_metrics_single_model(client):
    body = client.get(f"{API}/metrics/models/logistic_regression").json()
    assert 0.5 < body["metrics"]["test"]["roc_auc"] <= 1.0


def test_feature_importance(client):
    body = client.get(f"{API}/metrics/models/catboost/feature-importance").json()
    total = sum(item["importance"] for item in body["feature_importance"])
    assert total == pytest.approx(1.0)
    assert body["feature_importance"][0]["feature"] in {"campaign", "cat1"}
    # la regresion logistica no expone importancia
    assert client.get(f"{API}/metrics/models/logistic_regression/feature-importance").status_code == 404


def test_dataset_summary(client):
    body = client.get(f"{API}/metrics/dataset").json()
    assert body["target"] == "click"


def test_dataset_summary_missing_manifest(empty_client):
    response = empty_client.get(f"{API}/metrics/dataset")
    assert response.status_code == 404
    assert response.json()["error"] == "resource_not_found"


def test_prediction_stats_and_recent(client, ad):
    client.post(f"{API}/predict", json=ad, params={"model": "catboost"})
    client.post(f"{API}/predict/batch", json={"ads": [ad, ad]}, params={"model": "catboost"})
    stats = client.get(f"{API}/metrics/predictions").json()
    assert stats["total_requests"] == 2
    assert stats["total_predictions"] == 3
    assert stats["by_model"]["catboost"]["predictions"] == 3
    recent = client.get(f"{API}/metrics/predictions/recent", params={"limit": 5}).json()
    assert [r["kind"] for r in recent] == ["batch", "single"]
