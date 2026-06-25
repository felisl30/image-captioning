# Handoff técnico — D1: diagnóstico de mode collapse por checkpoint

Este documento explica, para otro modelo de lenguaje o asistente de código, qué hacen los archivos relacionados con **D1** dentro del proyecto de fine-tuning de BLIP sobre radiografías.

## 1. Contexto del proyecto

El proyecto usa `Salesforce/blip-image-captioning-base` para generar captions de radiografías de tórax del dataset MIMIC-CXR. El modelo fue fine-tuneado sobre `impression` médicas. Después del fine-tuning apareció un problema de **mode collapse**: el modelo tiende a repetir una o dos frases dominantes, por ejemplo:

```text
no acute cardiopulmonary process.
no significant interval change.
```

D1 es el primer diagnóstico liviano para entender **cuándo aparece ese collapse** durante el entrenamiento.

La pregunta concreta de D1 es:

```text
¿El modelo ya colapsa desde epoch_1, o el collapse aparece progresivamente hasta best?
```

## 2. Archivos involucrados en D1

Los archivos relevantes son:

```text
scripts/run_d1_checkpoint_collapse.py
notebooks/06_debug_mode_collapse.ipynb
```

Y los outputs esperados se guardan en:

```text
outputs/mode_collapse_debug/
```

Estructura esperada dentro del repo:

```text
image-captioning/
├── scripts/
│   └── run_d1_checkpoint_collapse.py
├── notebooks/
│   └── 06_debug_mode_collapse.ipynb
├── data/
│   └── selected_indices.json
├── models/
│   └── blip_finetuned_5k/
│       ├── epoch_1/
│       ├── epoch_2/
│       ├── epoch_3/
│       └── best/
└── outputs/
    └── mode_collapse_debug/
```

## 3. `scripts/run_d1_checkpoint_collapse.py`

### Propósito

Este script es el **motor real de D1**. Hace inferencia greedy sobre un conjunto fijo de radiografías usando varios checkpoints del fine-tuning, y mide qué tan colapsadas están las captions en cada checkpoint.

No entrena, no modifica pesos y no escribe nada dentro de `models/`.

### Checkpoints que compara

El script busca checkpoints con estos nombres:

```text
epoch_1
epoch_2
epoch_3
best
```

Por defecto busca en:

```text
models/blip_finetuned_5k/
models/blip_finetuned/
```

También puede usar checkpoints debug si se pasa:

```bash
--allow-debug
```

En ese caso busca en:

```text
models/blip_finetuned_debug_save/
models/blip_finetuned_debug/
models/blip_finetuned_notebook_debug/
```

### Qué considera un checkpoint usable

Una carpeta de checkpoint se considera usable si existe y contiene, como mínimo:

```text
config.json
model.safetensors
```

o alternativamente:

```text
config.json
pytorch_model.bin
```

También diagnostica si existen tokenizer/processor, por ejemplo:

```text
tokenizer.json
tokenizer_config.json
vocab.txt
processor_config.json
preprocessor_config.json
```

### Imágenes usadas

Por defecto usa:

```text
data/selected_indices.json
```

con:

```bash
--max-images 30
```

La idea es evaluar siempre las mismas 30 radiografías para que la comparación entre checkpoints sea justa.

### Decoding usado

D1 usa generación greedy:

```python
model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    num_beams=1,
    do_sample=False,
)
```

Esto es intencional. D1 no prueba sampling ni estrategias alternativas de decoding. Solo mide el comportamiento natural del checkpoint bajo greedy decoding.

### Métricas que calcula

Para cada checkpoint calcula:

```text
n_total
n_unique
unique_ratio
top_caption_norm
top_count
top_pct
```

Significado:

- `n_total`: cantidad de captions generadas.
- `n_unique`: cantidad de captions únicas.
- `unique_ratio`: `n_unique / n_total`.
- `top_caption_norm`: caption normalizada más repetida.
- `top_count`: cantidad de veces que aparece la caption dominante.
- `top_pct`: proporción de imágenes que recibieron la caption dominante.

La métrica más importante es `top_pct`.

Ejemplo de lectura:

```text
top_pct = 0.90
```

significa que el 90% de las imágenes generaron la misma caption, lo cual indica collapse fuerte.

### Outputs que genera

El script escribe:

```text
outputs/mode_collapse_debug/checkpoint_captions.csv
outputs/mode_collapse_debug/checkpoint_collapse_summary.csv
outputs/mode_collapse_debug/checkpoint_collapse_summary.json
outputs/mode_collapse_debug/checkpoint_examples.csv
```

#### `checkpoint_captions.csv`

Una fila por checkpoint e imagen.

Columnas principales:

```text
checkpoint
checkpoint_path
idx
reference
caption
caption_norm
```

Uso: análisis detallado caso por caso.

#### `checkpoint_collapse_summary.csv`

Una fila por checkpoint.

Columnas principales:

```text
checkpoint
n_total
n_unique
unique_ratio
top_caption_norm
top_count
top_pct
```

Uso: tabla central para decidir cuándo aparece el collapse.

#### `checkpoint_collapse_summary.json`

Mismo contenido que el CSV resumen, pero en formato JSON.

Uso: consumo programático por otros scripts o asistentes.

#### `checkpoint_examples.csv`

Ejemplos cualitativos por checkpoint.

Contiene dos tipos de ejemplos:

```text
first_examples
top_caption_examples
```

Uso: mostrar ejemplos concretos en notebook, informe o discusión con otro modelo.

## 4. Comandos principales del script D1

### Verificar sin correr inferencia

Usar cuando todavía no están los checkpoints reales:

```bash
python scripts/run_d1_checkpoint_collapse.py --dry-run
```

### Corrida real cuando estén los checkpoints

```bash
python scripts/run_d1_checkpoint_collapse.py \
  --checkpoint-root models/blip_finetuned_5k \
  --indices data/selected_indices.json \
  --max-images 30 \
  --device cpu
```

### Smoke test con checkpoints debug

No sirve como resultado final, pero sirve para validar que el script corre:

```bash
python scripts/run_d1_checkpoint_collapse.py \
  --allow-debug \
  --indices data/selected_indices.json \
  --max-images 3 \
  --device cpu
```

### Incluir BLIP base como referencia opcional

```bash
python scripts/run_d1_checkpoint_collapse.py \
  --checkpoint-root models/blip_finetuned_5k \
  --indices data/selected_indices.json \
  --max-images 30 \
  --device cpu \
  --include-base
```

## 5. `notebooks/06_debug_mode_collapse.ipynb`

### Propósito

Este notebook es el **lector y presentador de resultados de D1**. No debería contener la lógica pesada de inferencia. Esa lógica vive en:

```text
scripts/run_d1_checkpoint_collapse.py
```

El notebook sirve para:

1. mostrar el comando D1 que debe ejecutarse;
2. leer los CSVs generados;
3. mostrar la tabla de collapse por checkpoint;
4. graficar `unique_ratio` y `top_pct`;
5. mostrar ejemplos cualitativos.

### Celdas esperadas

El notebook subido contiene estas secciones:

```text
1. Setup
2. Comando D1 que se correrá cuando estén los checkpoints
3. Leer resultados
4. Plot simple de collapse
5. Ejemplos cualitativos
```

### Output visual principal

El notebook genera un gráfico simple con:

```text
unique_ratio
top_pct
```

por checkpoint.

Interpretación rápida:

- `unique_ratio` alto → más diversidad.
- `top_pct` alto → más collapse.
- `top_pct` cerca de 1.0 → casi todas las imágenes reciben la misma caption.

## 6. Cómo interpretar D1

### Caso A — Collapse desde `epoch_1`

Ejemplo:

```text
epoch_1 top_pct = 0.90
epoch_2 top_pct = 0.92
epoch_3 top_pct = 0.93
best    top_pct = 0.93
```

Interpretación:

```text
El collapse aparece desde el comienzo. El problema probablemente viene del incentivo de entrenamiento, la distribución del dataset o la loss, no solo de sobreentrenamiento tardío.
```

Siguiente paso recomendado:

```text
D2: analizar top-1/top-2 token a token para ver si greedy decoding está ocultando alternativas.
```

### Caso B — Collapse progresivo

Ejemplo:

```text
epoch_1 top_pct = 0.40
epoch_2 top_pct = 0.65
epoch_3 top_pct = 0.85
best    top_pct = 0.90
```

Interpretación:

```text
El modelo se vuelve progresivamente más rígido. Puede haber overfitting al modo textual dominante.
```

Siguiente paso recomendado:

```text
Revisar si epoch_1 o epoch_2 son mejores para análisis de heatmaps, aunque best tenga menor validation loss.
```

### Caso C — `best/` tiene peor diversidad que un checkpoint intermedio

Ejemplo:

```text
epoch_1 unique_ratio = 0.50
epoch_2 unique_ratio = 0.35
epoch_3 unique_ratio = 0.20
best    unique_ratio = 0.07
```

Interpretación:

```text
El checkpoint con mejor val loss no necesariamente es el mejor para análisis interpretativo. Puede ser útil usar un checkpoint intermedio para D3.
```

## 7. Qué no debe hacer otro modelo al tocar D1

No debe:

- convertir D1 en entrenamiento;
- modificar `src/models/finetuner.py`;
- cambiar los checkpoints;
- regenerar `selected_indices.json`;
- mezclar sampling en D1;
- usar `do_sample=True` dentro de D1;
- guardar modelos nuevos;
- reemplazar D1 por métricas BLEU/CIDEr.

D1 es solo un diagnóstico de diversidad/collapse bajo greedy decoding por checkpoint.

## 8. Dependencias técnicas

D1 necesita:

```text
torch
transformers
datasets
pandas
Pillow
```

No necesita:

```text
grad-cam
pytorch_grad_cam
matplotlib
```

`matplotlib` solo es necesario para el notebook si se quiere graficar.

## 9. Relación con D2 y D3

D1 no resuelve el collapse. Solo dice **cuándo aparece**.

Luego:

```text
D2 → mide qué tan puntiaguda está la distribución token a token.
D3 → mira si los heatmaps ya cambiaron aunque las captions colapsen.
```

Relación práctica:

```text
D1 encuentra el checkpoint más interesante.
D2 explica si greedy decoding está amplificando el problema.
D3 decide si el collapse textual bloquea o no el análisis de interpretabilidad.
```

## 10. Estado esperado antes de correr D1 real

Para que D1 sea válido, deberían existir:

```text
models/blip_finetuned_5k/epoch_1/
models/blip_finetuned_5k/epoch_2/
models/blip_finetuned_5k/epoch_3/
models/blip_finetuned_5k/best/
```

o la misma estructura bajo:

```text
models/blip_finetuned/
```

Si solo existen checkpoints debug, D1 puede correrse como smoke test, pero no debe usarse como conclusión del TP.

## 11. Resumen ejecutivo para otro LLM

D1 es un diagnóstico liviano de mode collapse. Compara captions greedy generadas por checkpoints intermedios (`epoch_1`, `epoch_2`, `epoch_3`, `best`) sobre las mismas 30 radiografías (`data/selected_indices.json`). El script `scripts/run_d1_checkpoint_collapse.py` carga checkpoints, genera captions, normaliza textos y calcula diversidad (`unique_ratio`) y dominancia de la caption más repetida (`top_pct`). El notebook `notebooks/06_debug_mode_collapse.ipynb` no hace inferencia pesada: lee los CSVs, grafica y muestra ejemplos. D1 debe correrse solo cuando estén los checkpoints reales; mientras tanto puede validarse con `--dry-run` o con `--allow-debug` como smoke test.
