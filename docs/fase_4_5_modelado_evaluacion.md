# Fases 4 y 5. Modelado y Evaluación — XGBoost vs CatBoost

**Autor:** Jesús David Villota Arteaga — Área: desarrollo y entrenamiento de modelos de ML.

## 1. Objetivo y técnica seleccionada

Predecir `click` (1 = el usuario hizo clic en el anuncio, 0 = no) a partir de las 11 variables de la Fase 3
(`timestamp` + `campaign` + `cat1`–`cat9`). Se implementan y comparan dos algoritmos de *gradient boosting*,
ambos con manejo nativo de variables categóricas (sin one-hot, coherente con la decisión de la Fase 3):

| Algoritmo | Por qué se eligió | Tratamiento de categóricas |
|---|---|---|
| **XGBoost** (`tree_method="hist"`, `enable_categorical=True`) | Estado del arte en tabular, rápido, con `scale_pos_weight` para el desbalance (~36 % de clics) | Codificación por categorías vistas en entrenamiento; lo no visto pasa a `NaN` (`src/ml/models.py: XGBoostCTR`) |
| **CatBoost** | Diseñado para categóricas de alta cardinalidad (`cat7` tiene 51 365 valores) mediante *ordered target statistics* | Categóricas por nombre, sin preprocesamiento |

La regresión logística con one-hot (`LogisticCTR`) se conserva solo como línea base del contrato de la API; no forma
parte de esta comparación.

## 2. Diseño de la prueba (CRISP-DM fase 5)

- **Partición temporal** (sin fuga): train `timestamp <= 1848196` < validation `<= 2269328` < test. Los cortes y la
  distribución por partición están en `data/processed/ctr_phase3/manifest.json`.
- **El umbral de decisión se ajusta en validación** (máximo F1, `src/ml/evaluation.py: best_f1_threshold`) y **prueba
  se reserva** para el reporte final. Ningún hiperparámetro se tocó mirando prueba.
- **Métricas:** ROC-AUC (criterio principal, umbral-independiente), PR-AUC, Log-Loss, Brier, Accuracy, Precision,
  Recall y F1, más matrices de confusión y curvas ROC/PR (`reports/figures/`).
- **Reproducibilidad:** semilla 42 en todos los muestreos (`load_split`), tuning (`TPESampler`) y modelos; versiones
  de librerías en `requirements.txt`.

## 3. Optimización de hiperparámetros (Optuna, `src/ml/tuning.py`)

Búsqueda bayesiana TPE con poda por mediana, **10 trials por algoritmo** sobre 100 000 filas de train / 50 000 de
validación. Objetivo: ROC-AUC en validación. `scale_pos_weight = neg/pos = 1.781` se fijó (no se muestreó).

| Algoritmo | Espacio explorado | Mejor configuración | AUC val (tuning) |
|---|---|---|---|
| XGBoost | `n_estimators` 200–800, `max_depth` 4–10, `learning_rate` 0.03–0.3 (log), `min_child_weight` 1–10, `subsample`/`colsample_bytree` 0.6–1.0, `reg_alpha`/`reg_lambda` 1e-4–10 (log) | `n_estimators=600, max_depth=8, learning_rate=0.0315, min_child_weight=9.73, subsample=0.933, colsample_bytree=0.685, reg_alpha=0.00081, reg_lambda=0.00083` | **0.6731** |
| CatBoost | `iterations` 300–900, `depth` 4–8, `learning_rate` 0.03–0.2 (log), `l2_leaf_reg` 1–10, `bagging_temperature` 0–1, `random_strength` 0–10 | `iterations=800, depth=5, learning_rate=0.0424, l2_leaf_reg=2.65, bagging_temperature=0.304, random_strength=5.25` | **0.6761** |

Comando reproducible: `python -m src.ml.tuning --algorithm {xgboost,catboost} --train-rows 100000 --eval-rows 50000 --n-trials 10`
(salida: `reports/tuning/<algoritmo>.json`).

## 4. Entrenamiento final y artefactos

Reentrenamiento con los mejores hiperparámetros sobre **500 000 filas de train / 200 000 de validación**
(`python -m src.ml.train` / `train_and_save`), con *early stopping* a 50 rondas sobre validación. Cada modelo queda en
`models/<nombre>/` con su `metadata.json` (algoritmo, umbral, características, métricas de validación y prueba), formato
que la API carga sin cambios (`src/ml/registry.py`). `models/` está excluido de Git por tamaño (CatBoost ~82 MB,
XGBoost ~157 MB) y se regenera con el comando anterior.

## 5. Resultados en prueba (200 000 filas, CTR observado = 0.365)

| Modelo | ROC-AUC | PR-AUC | Log-Loss | Acc | Prec | Rec | F1 | Umbral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **catboost (mejor)** | **0.6876** | **0.5594** | **0.6032** | 0.5408 | 0.4336 | 0.8461 | **0.5734** | 0.282 |
| xgboost | 0.6846 | 0.5572 | 0.6435 | 0.5563 | 0.4412 | 0.8131 | 0.5720 | 0.424 |

Comparativo máquina: `reports/metrics/comparison.json` — regenerar con `python -m src.ml.compare --eval-rows 200000`.

### 5.1 Matrices de confusión (umbral de validación)

| | CatBoost (test) | XGBoost (test) |
|---|---|---|
| TN / FP | 46 455 / 80 606 | 51 954 / 75 107 |
| FN / TP | 11 228 / 61 711 | 13 633 / 59 306 |

Lectura: ambos modelos priorizan **recall** (~0.81–0.85) sobre precisión (~0.43–0.44): capturan la gran mayoría de los
clics a costa de falsos positivos. Es el comportamiento esperado con el umbral de máximo F1 y es razonable para CTR,
donde perder un clic potencial cuesta más que mostrar un anuncio de más. CatBoost recupera **+2 405 clics** (TP) con
2 405 FN menos que XGBoost.

### 5.2 Curvas ROC/PR y calibración

- Las curvas ROC (`reports/figures/roc_curves.png`) confirman la ventaja de CatBoost en casi todo el rango de FPR.
- En PR (`reports/figures/pr_curves.png`) ambos superan con holgura la prevalencia (0.365).
- **Calibración:** CTR medio predicho vs observado (0.365): CatBoost **0.381** (bien calibrado) frente a XGBoost
  **0.497** (sobreestima la probabilidad media; su umbral óptimo sube a 0.424 para compensarlo). Si el frontend va a
  mostrar la probabilidad como “% de clic”, la de CatBoost es más honesta; XGBoost requeriría recalibración
  (p. ej. isotónica) antes de exhibirse.

### 5.3 Importancia de variables (normalizada, suma 1)

- **CatBoost:** `cat9` (0.234) > `cat8` (0.152) > `cat6` (0.135) ≈ `cat3` (0.135) > `campaign` (0.128).
- **XGBoost:** `cat9` (0.214) > `cat3` (0.192) > `campaign` (0.185) > `cat7` (0.119) > `cat8` (0.106).
- Coinciden en el top (`cat9`, `cat3`, `campaign`, `cat8`); la diferencia está en `cat7` (la de mayor cardinalidad):
  XGBoost la aprovecha más y CatBoost la regulariza mejor, lo que explica parte de su mejor generalización.

## 6. Selección del mejor modelo y despliegue

**Criterio:** mayor ROC-AUC en **validación** (desempate: PR-AUC, luego F1) — implementado en `select_best`
(`src/ml/compare.py`). Resultado:

- Validación: CatBoost 0.6839 > XGBoost 0.6813 → **ganador: `catboost`**.
- Prueba confirma la generalización: 0.6876 > 0.6846 (diferencia pequeña pero consistente en PR-AUC, Log-Loss, Brier y F1).

La API ya expone ambos (`GET /api/v1/metrics/models`, `?model=xgboost|catboost`) y usa por defecto el de mayor AUC de
validación (`ModelRegistry.default_name`), por lo que **el despliegue sirve `catboost` automáticamente** sin cambios de
código. El frontend/dashboard debe etiquetar las métricas con modelo, partición y umbral (disponibles en cada
`metadata.json`).

## 7. Conclusiones y trabajo futuro

1. Ambos modelos superan el azar y a la línea base; **CatBoost es el mejor**: +0.003 en ROC-AUC de prueba, mejor
   Log-Loss (0.603 vs 0.644) y mejor calibración, gracias a su tratamiento nativo de categóricas de alta cardinalidad.
2. El recall alto / precisión moderada es una decisión consciente (umbral de máximo F1); si el negocio prefiere menos
   falsos positivos, el umbral puede subirse consultando la curva PR.
3. Futuro: calibración isotónica de XGBoost, entrenamiento sobre la partición completa, validación cruzada temporal y
   monitoreo de deriva (los cortes temporales están en el manifiesto para repetir el protocolo).

## 8. Archivos entregados (rama `Jesus`)

| Archivo | Contenido |
|---|---|
| `src/ml/tuning.py` | Optimización Optuna (CLI `python -m src.ml.tuning`) |
| `src/ml/plots.py` | Figuras ROC/PR/confusión/barras/importancia |
| `src/ml/compare.py` | Evaluación en prueba, comparativo y selección del mejor (CLI `python -m src.ml.compare`) |
| `notebooks/04_modelado_evaluacion.ipynb` | Notebook reproducible fases 4–5 |
| `tests/test_modelado.py` | 7 pruebas unitarias de modelado |
| `reports/metrics/comparison.json` + `reports/figures/*.png` | Resultados y figuras versionadas |
| `docs/fase_4_5_modelado_evaluacion.md` | Este documento |
