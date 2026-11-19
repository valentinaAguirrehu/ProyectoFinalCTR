# Frontend: simulador de anuncios y dashboard

Módulo de interfaz de la fase de **Despliegue** de CRISP-DM. La aplicación vive en `frontend/` y consume la API de FastAPI que ya expone los modelos entrenados. No reentrena modelos ni modifica el backend.

## 1. Qué permite hacer

1. **Simulador.** Un formulario recoge las once características de una impresión (`timestamp`, `campaign`, `cat1`–`cat9`) y llama a `POST /api/v1/predict`. La respuesta se muestra como probabilidad de clic, decisión (clic esperado o no) y umbral del modelo. La tarjeta de la derecha es una vista simulada del anuncio: el texto creativo no entra al modelo.
2. **Escenarios.** Tres anuncios de demostración se evalúan juntos con `POST /api/v1/predict/batch` y se ordenan por probabilidad.
3. **Comparación.** El mismo anuncio se envía a cada modelo cargado (CatBoost, XGBoost y la regresión logística, si existen) para ver lado a lado la probabilidad.
4. **Dashboard.** Consulta el estado del servicio, la comparativa de métricas en prueba o validación, la importancia de variables, el resumen del dataset preparado y las predicciones recientes.

La interfaz es responsive: en pantallas angostas el formulario, la vista previa y las tarjetas pasan a una sola columna.

## 2. Cómo ejecutarla

Desde la raíz del proyecto, con el entorno ya instalado (`pip install -r requirements.txt`):

```bash
python -m src.ml.train
uvicorn src.api.main:app --reload
```

En otra terminal:

```bash
python -m http.server 5500 --directory frontend
```

Abrir http://127.0.0.1:5500. Si la API usa otro host o puerto, se cambia en el campo «Dirección de la API» y se pulsa **Conectar**. El valor queda guardado en el navegador.

Sin la carpeta `models/`, la API arranca en estado degradado y el simulador muestra el error 503. El dashboard sigue indicando que no hay modelos.

## 3. Contrato con el backend

| Pantalla | Método y ruta |
|---|---|
| Estado de la conexión | `GET /health` |
| Lista de modelos del selector | `GET /api/v1/models` |
| Estimación de un anuncio | `POST /api/v1/predict?model=` |
| Ordenar los tres escenarios | `POST /api/v1/predict/batch?model=` |
| Tabla y gráfica de métricas | `GET /api/v1/metrics/models?split=test` o `validation` |
| Importancia de variables | `GET /api/v1/metrics/models/{nombre}/feature-importance` |
| Resumen del dataset | `GET /api/v1/metrics/dataset` |
| Uso y predicciones recientes | `GET /api/v1/metrics/predictions` y `/predictions/recent` |

El cuerpo de una predicción solo incluye los once enteros. El navegador valida antes de enviar: campos vacíos, texto, negativos y categorías por encima de 2.147.483.647. Si la API responde 422, el mensaje se muestra en el formulario.

`predicted_click` se interpreta tal como lo define la API: la probabilidad es mayor o igual que el umbral guardado en el modelo (el umbral se ajustó en validación para maximizar F1). Para decidir entre anuncios se usa la probabilidad, no solo el sí o el no.

Las gráficas usan Chart.js 4.4.6 (licencia MIT), guardado en `frontend/vendor/chart.umd.min.js`, para que el dashboard no dependa de una red externa el día de la sustentación.

## 4. Decisiones de diseño

- **HTML, CSS y JavaScript sin paso de compilación.** Se puede abrir con el servidor de Python y no hace falta instalar Node.
- **Una sola página con tres secciones.** Simulador, comparación y dashboard comparten la dirección de la API y el formulario.
- **El anuncio visual es una simulación.** Criteo entrega códigos anonimizados. Inventar un texto de anuncio ayuda a explicar el caso de negocio, y la interfaz dice de forma explícita que ese texto no se envía al modelo.
- **Los errores no tapan el resto de la pantalla.** Si el dataset o un modelo no responden, las otras tarjetas del dashboard siguen cargando.
- **La matriz de confusión no se dibuja aquí.** La API publica Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, log loss y Brier. Esas son las cifras del tablero. Las matrices quedan en los notebooks de modelado.

## 5. Pruebas de interfaz

Automatizadas, sin levantar el servidor:

```bash
pytest tests/test_frontend_contract.py
```



