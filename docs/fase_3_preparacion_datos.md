# Fase 3. Preparación de los datos

## Objetivo y alcance

Preparar una tabla analítica para modelar `click` sin modificar el TSV original. El flujo está implementado en [`notebooks/03_preparacion_datos.ipynb`](../notebooks/03_preparacion_datos.ipynb) y usa DuckDB para leer y transformar el archivo por debajo del límite de memoria configurado.

Esta fase conserva la etiqueta y las características acordadas, valida el resultado y genera particiones cronológicas de entrenamiento, validación y prueba. No entrena modelos ni ajusta sus hiperparámetros.

## Entrada y salida

- **Entrada:** `data/raw/pcb_dataset_final.tsv` (16.468.027 registros, 22 columnas).
- **Salida:** `data/processed/ctr_phase3/`.
- **Formato:** Parquet con compresión ZSTD, particionado en `split=train/`, `split=validation/` y `split=test/`.
- **Archivos de auditoría:** `manifest.json` y `split_summary.csv`, generados junto a las particiones.

## Selección y transformación de variables

### Predictores y objetivo

- **Objetivo:** `click`, entero de 8 bits con valores 0 y 1.
- **Predictores:** `timestamp`, `campaign` y `cat1`–`cat9`.
- **Categóricas:** `campaign` y `cat1`–`cat9`; se preservan sus códigos como enteros de 32 bits, sin one-hot ni codificación ordinal nueva.
- **Temporal:** `timestamp`, entero de 64 bits, se conserva como predictor y se usa para formar las particiones.

### Variables excluidas

| Variable(s) | Motivo |
|---|---|
| `uid` | Identificador de usuario; riesgo de memorización y sobreajuste. |
| `conversion`, `conversion_timestamp`, `conversion_id`, `attribution` | Resultados de conversión/atribución posteriores a la impresión; riesgo de fuga. |
| `click_pos`, `click_nb` | Información del recorrido relacionada con clics posteriores. |
| `cpo` | Costo por pedido, asociado con un resultado posterior. |
| `cost`, `time_since_last_click` | No se confirmó que estén disponibles al momento de predecir la impresión; se excluyen conservadoramente hasta validar su semántica. |

La Fase 2 no encontró nulos. Los valores `-1` aparecieron en campos excluidos; por ello no se imputan ni se reemplazan en esta versión. El notebook verifica que las variables conservadas y el objetivo no tengan nulos, valores `-1` ni etiquetas distintas de 0 y 1. No aplica balanceo, para conservar la distribución natural en validación y prueba.

## Partición temporal

Se usan los percentiles discretos 70 y 85 de `timestamp` calculados sobre el archivo completo. En la ejecución realizada, los cortes fueron **1.848.196** y **2.269.328**:

- Entrenamiento: `timestamp <= 1.848.196`.
- Validación: `1.848.196 < timestamp <= 2.269.328`.
- Prueba: `timestamp > 2.269.328`.

Los registros con el mismo `timestamp` permanecen en la misma partición. Por este motivo, las proporciones efectivas pueden diferir ligeramente del 70/15/15 esperado; el notebook registra el tamaño, el rango temporal y la distribución de `click` de cada partición y valida que no exista solapamiento temporal.

### Resultado de la ejecución

Se procesaron las 16.468.027 filas. Las particiones contienen ambas clases y sus rangos temporales no se solapan:

| Partición | Filas | Proporción | `click=0` | `click=1` | Rango de `timestamp` |
|---|---:|---:|---:|---:|---:|
| Entrenamiento | 11.527.626 | 70,0000 % | 7.397.325 | 4.130.301 | 0–1.848.196 |
| Validación | 2.470.203 | 15,0000 % | 1.556.926 | 913.277 | 1.848.197–2.269.328 |
| Prueba | 2.470.198 | 15,0000 % | 1.566.213 | 903.985 | 2.269.329–2.671.199 |

El tamaño combinado de los tres Parquet es aproximadamente **133 MiB**, frente a los 2,41 GiB del TSV original. El notebook verificó el total de filas, el esquema, la ausencia de nulos y valores `-1` en las columnas conservadas, ambas clases por partición y el orden temporal.

## Uso en el modelado

- CatBoost debe recibir `campaign` y `cat1`–`cat9` como características categóricas.
- Para XGBoost, al leer datos con pandas, convertir dichas columnas a `category` y habilitar el soporte categórico (`enable_categorical=True`, `tree_method="hist"`).
- Ajustar los modelos y el umbral usando entrenamiento y validación. Reservar prueba para la evaluación final.
- No cargar toda la partición de entrenamiento en memoria si el equipo no tiene capacidad suficiente; consultar Parquet por lotes con DuckDB.

## Reproducibilidad y entrega

1. Instalar dependencias desde la raíz con `pip install -r requirements.txt`.
2. Confirmar que `data/raw/pcb_dataset_final.tsv` está disponible.
3. Ejecutar [`notebooks/03_preparacion_datos.ipynb`](../notebooks/03_preparacion_datos.ipynb) de principio a fin.
4. Revisar `split_summary.csv`, `manifest.json` y las tres carpetas Parquet antes de entrenar.

La carpeta `data/processed/ctr_phase3/`, que contiene las tres particiones Parquet, `manifest.json` y `split_summary.csv`, está incluida en este repositorio. El TSV original no se incluye en Git por su tamaño; descárgalo desde [Kaggle: Criteo Attribution Modeling](https://www.kaggle.com/datasets/sharatsachin/criteo-attribution-modeling) y guárdalo como `data/raw/pcb_dataset_final.tsv` si necesitas reproducir el procesamiento. El notebook no sobrescribe salidas existentes con manifiestos diferentes.