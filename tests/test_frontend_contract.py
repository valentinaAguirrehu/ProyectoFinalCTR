"""Contrato de la interfaz: pantallas, campos y endpoints, sin levantar la API."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
FEATURES = ["timestamp", "campaign", *[f"cat{i}" for i in range(1, 10)]]
ENDPOINTS = [
    "/health",
    "/api/v1/predict",
    "/api/v1/predict/batch",
    "/api/v1/models",
    "/api/v1/metrics/models",
    "/api/v1/metrics/dataset",
    "/api/v1/metrics/predictions",
    "/api/v1/metrics/predictions/recent",
]


def test_pagina_tiene_las_tres_secciones_y_los_once_campos():
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert 'lang="es"' in html
    assert 'name="viewport"' in html
    assert 'id="ad-form"' in html
    for section in ("simulator", "compare", "dashboard"):
        assert f'data-view="{section}"' in html
        assert f'data-nav="{section}"' in html
    for name in FEATURES:
        assert f'id="{name}"' in html
    assert "uid" not in html
    assert "conversion" not in html


def test_cliente_usa_los_endpoints_del_backend():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (FRONTEND / "js").glob("*.js")
    )
    for endpoint in ENDPOINTS:
        assert endpoint in source
    for name in FEATURES:
        assert name in source


def test_el_estilo_es_responsive_y_la_grafica_esta_vendida():
    css = (FRONTEND / "css" / "styles.css").read_text(encoding="utf-8")
    assert "@media (max-width: 680px)" in css
    assert "@media (max-width: 960px)" in css
    chart = FRONTEND / "vendor" / "chart.umd.min.js"
    assert chart.is_file()
    assert chart.stat().st_size > 100_000
