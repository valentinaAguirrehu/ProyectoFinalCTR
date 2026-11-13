# Fase 2. Comprensión de los datos

## Objetivo

Comprender la estructura, el contenido, la calidad y las limitaciones del conjunto de datos antes de transformarlo. Esta fase corresponde a la segunda etapa de CRISP-DM y sus resultados se documentan y exploran en [`notebooks/02_comprension_datos.ipynb`](../notebooks/02_comprension_datos.ipynb).

## Fuente y alcance

- **Dataset:** Criteo Attribution Modeling (`pcb_dataset_final.tsv`).
- **Fuente:** [Kaggle: Criteo Attribution Modeling](https://www.kaggle.com/datasets/sharatsachin/criteo-attribution-modeling).
- **Formato:** TSV (separado por tabulaciones), con encabezado.
- **Alcance del notebook:** inspección inicial de una muestra de 1.000 filas y análisis por bloques del conjunto completo. La muestra corresponde a las primeras filas del archivo y se usa para inspección y gráficos exploratorios; no se interpreta como una muestra aleatoria.

## Hallazgos principales

La exploración registrada en el notebook identificó **16.468.027 registros y 22 columnas**, con un archivo de aproximadamente **2,41 GB**. Por su tamaño, las métricas globales se calculan leyendo bloques para limitar el uso de memoria.

La variable objetivo es `click` (0 = sin clic, 1 = clic). En la ejecución validada se contaron 10.520.464 registros de clase 0 (63,88 %) y 5.947.563 de clase 1 (36,12 %). No se encontraron valores nulos ni valores de `click` distintos de 0 y 1.

El análisis global encontró `-1` en `conversion_timestamp`, `conversion_id`, `click_pos` y `click_nb` (15.661.831 filas, 95,10 % en cada columna), y en `time_since_last_click` (8.770.820 filas, 53,26 %). Estos valores requieren interpretación por variable; no deben reemplazarse automáticamente como si todos tuvieran el mismo significado. La comparación global de las 22 columnas reportó 0 duplicados exactos.

Las columnas `cat1` a `cat9` son códigos categóricos anonimizados aunque se lean como números. `uid`, `campaign` y `conversion_id` son identificadores; `timestamp` y `conversion_timestamp` son temporales; `click` es la variable objetivo.

## Calidad y limitaciones

El notebook muestra por separado los resultados de la muestra y los cálculos globales. También contiene una comprobación de duplicados exactos sobre todas las columnas mediante DuckDB. Sus celdas deben ejecutarse en orden para actualizar los resultados con el archivo local.

La relación entre `click` y `conversion` observada en la muestra no se generaliza al conjunto completo. Las variables de conversión y las que describen el recorrido posterior al clic se consideran candidatas a fuga de información para una predicción hecha al momento de la impresión.

## Decisiones para la siguiente fase

La fase 2 no modifica el dataset ni crea particiones. En la fase 3 se deberá:

1. Definir el tratamiento de `-1` según el significado de cada variable.
2. Seleccionar únicamente información disponible al momento de predecir el clic y documentar exclusiones por fuga de información.
3. Decidir cómo representar las variables categóricas de alta cardinalidad y los identificadores.
4. Preparar las particiones de entrenamiento, validación y prueba con una estrategia que considere la dimensión temporal del problema.

## Reproducibilidad

Descarga `pcb_dataset_final.tsv` desde [Kaggle: Criteo Attribution Modeling](https://www.kaggle.com/datasets/sharatsachin/criteo-attribution-modeling) y ubícalo en `data/raw/pcb_dataset_final.tsv`. El archivo original no se incluye en Git por su tamaño. Instala las dependencias desde la raíz con `pip install -r requirements.txt` y ejecuta [`notebooks/02_comprension_datos.ipynb`](../notebooks/02_comprension_datos.ipynb) de arriba hacia abajo. El análisis global y la comprobación de duplicados pueden tardar por el tamaño del archivo.