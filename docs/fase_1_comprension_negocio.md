# Fase 1. Comprensión del negocio

## Contexto

La publicidad digital requiere estimar la posibilidad de interacción con los anuncios. En este proyecto se predice si una impresión publicitaria recibe un clic. El CTR relaciona clics e impresiones, pero las probabilidades del modelo no deben presentarse automáticamente como el CTR real de una campaña.

## Problema y objetivos

**Problema:** identificar patrones históricos que permitan anticipar si una impresión recibirá un clic, usando únicamente información disponible al momento de mostrar el anuncio.

**Objetivo de negocio:** estimar la probabilidad de clic por impresión para apoyar el análisis de efectividad de anuncios digitales.

**Objetivo de minería de datos:** construir un clasificador binario para predecir `click` (0 = sin clic, 1 = clic).

### Objetivos específicos

- Comprender y explorar la estructura y calidad del dataset.
- Identificar características candidatas y prevenir fuga de información.
- Limpiar, transformar y dividir los datos en entrenamiento, validación y prueba.
- Entrenar XGBoost y CatBoost y compararlos con una línea base.
- Evaluar con AUC, matriz de confusión, precisión, recall, F1-score y métricas de probabilidades.
- Desplegar la predicción en una aplicación web con simulador de anuncios y dashboard.

## Criterios de éxito

### Técnicos

- Comparar XGBoost y CatBoost con un modelo base usando una partición de prueba reservada.
- Reportar AUC, matriz de confusión, precisión, recall, F1-score, AUC-PR y log loss.
- Justificar el umbral de la matriz de confusión y revisar la diferencia de desempeño entre validación y prueba.
- Como referencia inicial, se propone AUC de prueba >= 0,70. El equipo debe ratificar o ajustar el valor después de medir la línea base; el conjunto de prueba no debe usarse para ajustar el modelo.

### Negocio y despliegue

- La aplicación presenta de forma comprensible la probabilidad para un anuncio simulado.
- El dashboard muestra métricas del modelo y resultados por campaña, si los datos lo permiten.

## Metodología CRISP-DM

1. Comprensión del negocio.
2. Comprensión de los datos.
3. Preparación de los datos.
4. Modelado.
5. Evaluación.
6. Despliegue.

## Datos, alcance y supuestos

- **Dataset:** Criteo Attribution Modeling for Bidding, disponible en [Kaggle](https://www.kaggle.com/datasets/sharatsachin/criteo-attribution-modeling). La Fase 2 documenta la estructura y calidad del archivo `pcb_dataset_final.tsv`.
- **Alcance:** predicción de clic a nivel de impresión y su integración posterior en una aplicación web.
- **Supuesto:** los patrones históricos del dataset aportan información útil sobre impresiones futuras; debe revisarse su estabilidad temporal.
- **Fuera de alcance:** optimización de pujas, atribución de conversiones y predicción de conversión.

## Riesgos y limitaciones

| Riesgo o limitación | Mitigación |
|---|---|
| Fuga de información por variables de conversión, atribución o posteriores al clic | Revisar su disponibilidad temporal y excluirlas antes del modelado si revelan resultados futuros. |
| Distribución de clics distinta a campañas reales | Documentar esta limitación y no presentar la probabilidad como CTR real sin calibración y validación externa. |
| Características categóricas anonimizadas | Describirlas como códigos y evaluar una representación adecuada. |
| Dataset de gran volumen | Procesar por bloques y controlar el uso de memoria. |
| Identificadores y riesgo de sobreajuste | Evaluar su uso y diseñar particiones que eviten fuga temporal. |

## Recursos y plan

- **Herramientas:** Python, pandas, DuckDB, XGBoost, CatBoost y una herramienta de desarrollo web.
- **Plan:** Fase 1 → Fase 2 → Fase 3 (análisis y preparación) → Fase 4 (modelado) → Fase 5 (evaluación) → Fase 6 (despliegue).
- **Entregables de análisis:** documentación de las fases 1 a 3, notebooks de negocio/datos/preparación y dataset procesado con particiones de entrenamiento, validación y prueba.

## Distribución de responsabilidades

| Integrante | Área | Fases principales |
|---|---|---|
| Leidy Valentina Aguirre | Análisis y preparación de datos | 1, 2 y 3 |
| Jesús David Villota Arteaga | Modelos de Machine Learning | 4 y 5 |
| Camila Isabel Benavides | Backend e integración del modelo | 6 (API) |
| Angie Fernanda Benavides | Frontend, simulador y dashboard | 6 (interfaz) |

## Glosario

- **Impresión:** ocasión en que se muestra un anuncio a un usuario.
- **Clic:** interacción del usuario con el anuncio.
- **CTR:** relación entre clics e impresiones.
- **Conversión:** acción posterior a la interacción, por ejemplo, una compra.
- **AUC:** capacidad del modelo para ordenar ejemplos positivos por encima de los negativos.
