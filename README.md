
# Predicción de CTR en Anuncios Digitales

## Descripción del proyecto

Proyecto académico enfocado en el desarrollo de un modelo predictivo para estimar la probabilidad de que un usuario haga clic en un anuncio digital (Click Through Rate - CTR), utilizando técnicas de Machine Learning y la metodología CRISP-DM.

## Objetivo general

Desarrollar un modelo de clasificación que permita predecir la probabilidad de clic en anuncios digitales a partir del análisis de datos históricos.

## Metodología

Se utilizará CRISP-DM, desarrollando las siguientes fases:

1. Comprensión del negocio.
2. Comprensión de los datos.
3. Preparación de los datos.
4. Modelado.
5. Evaluación.
6. Despliegue.

## Dataset

Criteo Attribution Modeling.

Fuente: https://www.kaggle.com/datasets/sharatsachin/criteo-attribution-modeling

## Tecnologías

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- FastAPI y Uvicorn
- Pytest
- Git y GitHub

## Estructura del proyecto

- data/: almacenamiento de datos.
- notebooks/: análisis y experimentación.
- src/: código fuente.
  - src/data/: contrato de características compartido.
  - src/ml/: modelos, entrenamiento, evaluación y registro.
  - src/api/: backend FastAPI (endpoints, servicios, validación).
- models/: artefactos entrenados (generados localmente, no versionados).
- tests/: pruebas de la API y del contrato de la interfaz.
- docs/: documentación de CRISP-DM, de la API y del frontend.
- frontend/: simulador de anuncios, comparación de modelos y dashboard.

## Backend y API de predicción

```bash
pip install -r requirements.txt
python -m src.ml.train            # entrena y guarda los modelos en models/
uvicorn src.api.main:app --reload # API en http://127.0.0.1:8000/docs
pytest                            # pruebas
```

Documentación completa: [docs/backend_api.md](docs/backend_api.md).

## Frontend: simulador y dashboard

Interfaz web en HTML, CSS y JavaScript. No requiere Node. Hay que abrirla con un servidor local: si se abre el archivo directamente, el navegador puede bloquear los módulos.

```bash
# Terminal 1, en la raíz del proyecto
uvicorn src.api.main:app --reload

# Terminal 2
python -m http.server 5500 --directory frontend
```

Abrir <http://127.0.0.1:5500>. La pantalla permite cambiar la dirección de la API si no está en `http://127.0.0.1:8000`.

Documentación del módulo: [docs/frontend.md](docs/frontend.md).

## Despliegue

Frontend en Vercel y API en Render: [docs/despliegue.md](docs/despliegue.md).

## Integrantes
Leidy valentina aguirre 
Angie Fernanda Benavides
Jesús David Villota
Camila Isabel Bastidas
Proyecto académico de Minería de Datos.
