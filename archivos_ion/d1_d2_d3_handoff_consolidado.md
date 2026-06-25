# Handoff técnico consolidado — D1, D2 y D3

Este documento explica, para otro modelo de lenguaje o asistente de código, qué son y para qué sirven los tres diagnósticos agregados al proyecto:

```text
D1 — Collapse por checkpoint
D2 — Probe de distribución token a token
D3 — Heatmap probe base vs fine-tuned
```

Estos tres pasos forman una fase de diagnóstico previa a decidir si conviene reentrenar, cambiar decoding o avanzar con el análisis de interpretabilidad.

---

## 1. Contexto general del proyecto

El proyecto usa `Salesforce/blip-image-captioning-base` para generar captions de radiografías de tórax del dataset MIMIC-CXR.

El modelo fue fine-tuneado usando el campo `impression` como target textual. Después del fine-tuning apareció un problema de **mode collapse**: el modelo tiende a repetir una o dos frases dominantes para muchas imágenes distintas, por ejemplo:

```text
no acute cardiopulmonary process.
no significant interval change.
```

Este problema afecta la legibilidad de las captions, pero no necesariamente invalida el objetivo central del TP.

La pregunta central del proyecto no es solamente si BLIP aprende a generar texto médico, sino si después del fine-tuning aprende también a **mirar regiones clínicamente relevantes** de la radiografía.

Por eso se diseñaron tres diagnósticos:

```text
D1 → cuándo aparece el collapse
D2 → qué tan segura es la distribución textual colapsada
D3 → si los heatmaps cambiaron aunque las captions colapsen
```

---

## 2. Resumen ejecutivo

### D1 — Collapse por checkpoint

D1 compara `epoch_1`, `epoch_2`, `epoch_3` y `best` sobre las mismas radiografías.

Pregunta principal:

```text
¿El collapse ya aparece desde epoch_1 o aparece progresivamente durante el entrenamiento?
```

Archivos:

```text
scripts/run_d1_checkpoint_collapse.py
notebooks/06_debug_mode_collapse.ipynb
```

Output principal:

```text
outputs/mode_collapse_debug/checkpoint_collapse_summary.csv
```

Métricas clave:

```text
unique_ratio
top_pct
top_caption_norm
```

---

### D2 — Probe de distribución token a token

D2 mira la distribución interna del decoder durante generación greedy.

Pregunta principal:

```text
¿El modelo está realmente muy seguro del token top-1 o greedy decoding está ocultando alternativas cercanas?
```

Archivos:

```text
scripts/run_d2_token_probe.py
notebooks/07_debug_token_distribution.ipynb
```

Output principal:

```text
outputs/mode_collapse_debug/d2_checkpoint_summary.csv
```

Métricas clave:

```text
mean_p_top1
mean_p_top2
mean_gap
mean_entropy
pct_steps_top1_gt_090
pct_steps_gap_lt_010
```

---

### D3 — Heatmap probe base vs fine-tuned

D3 compara mapas visuales del modelo base y del modelo fine-tuneado.

Pregunta principal:

```text
Aunque las captions estén colapsadas, ¿los heatmaps del fine-tuned cambiaron respecto de BLIP base?
```

Archivos:

```text
scripts/run_d3_heatmap_probe.py
notebooks/08_debug_heatmaps_probe.ipynb
```

Output principal:

```text
outputs/mode_collapse_debug/d3_heatmaps/d3_heatmap_summary.csv
```

Figuras clave:

```text
cross_att_logits_grid.png
gradcam_grid.png
cross_vs_gradcam.png
```

---

## 3. Estructura de archivos esperada

Dentro del repo:

```text
image-captioning/
├── scripts/
│   ├── run_d1_checkpoint_collapse.py
│   ├── run_d2_token_probe.py
│   └── run_d3_heatmap_probe.py
│
├── notebooks/
│   ├── 06_debug_mode_collapse.ipynb
│   ├── 07_debug_token_distribution.ipynb
│   └── 08_debug_heatmaps_probe.ipynb
│
├── data/
│   └── selected_indices.json
│
├── models/
│   ├── blip_base/
│   └── blip_finetuned_5k/
│       ├── epoch_1/
│       ├── epoch_2/
│       ├── epoch_3/
│       └── best/
│
└── outputs/
    └── mode_collapse_debug/
        ├── checkpoint_captions.csv
        ├── checkpoint_collapse_summary.csv
        ├── checkpoint_examples.csv
        ├── d2_token_probe_steps.csv
        ├── d2_image_summary.csv
        ├── d2_checkpoint_summary.csv
        ├── d2_high_confidence_examples.csv
        ├── d2_low_margin_examples.csv
        └── d3_heatmaps/
            ├── d3_heatmap_summary.csv
            ├── d3_heatmap_summary.json
            └── idx_<IDX>/
                ├── base/
                └── finetuned/
```

---

## 4. D1 — Collapse por checkpoint

## 4.1. Objetivo

D1 busca responder:

```text
¿Cuándo aparece el mode collapse durante el entrenamiento?
```

Para eso carga los checkpoints intermedios:

```text
epoch_1
epoch_2
epoch_3
best
```

y genera captions greedy sobre las mismas imágenes fijas de:

```text
data/selected_indices.json
```

Luego mide cuántas captions únicas aparecen y qué porcentaje ocupa la caption dominante.

---

## 4.2. Archivos de D1

```text
scripts/run_d1_checkpoint_collapse.py
notebooks/06_debug_mode_collapse.ipynb
```

### `scripts/run_d1_checkpoint_collapse.py`

Es el motor de D1.

Hace:

1. detecta la raíz del proyecto;
2. busca checkpoints usables;
3. carga el dataset;
4. carga cada checkpoint;
5. genera captions greedy;
6. normaliza captions;
7. calcula métricas de collapse;
8. guarda CSVs y JSONs.

No hace:

```text
- entrenamiento
- sampling
- heatmaps
- BLEU/CIDEr/METEOR
- modificación de checkpoints
```

### `notebooks/06_debug_mode_collapse.ipynb`

Es el lector y visualizador de D1.

Hace:

1. muestra comandos de ejecución;
2. carga outputs de D1;
3. muestra tabla de collapse;
4. grafica `unique_ratio` y `top_pct`;
5. muestra ejemplos por checkpoint.

No debería contener lógica pesada de inferencia.

---

## 4.3. Outputs de D1

### `checkpoint_captions.csv`

Una fila por:

```text
checkpoint × imagen
```

Columnas típicas:

```text
checkpoint
checkpoint_path
idx
reference
caption
caption_norm
```

Uso:

```text
Análisis detallado de captions por imagen.
```

### `checkpoint_collapse_summary.csv`

Una fila por checkpoint.

Columnas clave:

```text
checkpoint
n_total
n_unique
unique_ratio
top_caption_norm
top_count
top_pct
```

Uso:

```text
Tabla principal de D1.
```

### `checkpoint_examples.csv`

Ejemplos cualitativos por checkpoint.

Uso:

```text
Mostrar ejemplos de captions generadas y captions dominantes.
```

---

## 4.4. Métricas de D1

### `unique_ratio`

```text
unique_ratio = n_unique / n_total
```

Interpretación:

```text
Alto → más diversidad textual.
Bajo → más collapse.
```

### `top_pct`

```text
top_pct = top_count / n_total
```

Interpretación:

```text
Alto → una caption domina muchas imágenes.
Cerca de 1.0 → collapse extremo.
```

### `top_caption_norm`

Caption dominante normalizada.

Ejemplo:

```text
no acute cardiopulmonary process
```

---

## 4.5. Comandos de D1

### Dry-run

```bash
python scripts/run_d1_checkpoint_collapse.py --dry-run
```

### Corrida real

```bash
python scripts/run_d1_checkpoint_collapse.py \
  --checkpoint-root models/blip_finetuned_5k \
  --indices data/selected_indices.json \
  --max-images 30 \
  --device cpu
```

### Smoke test con debug

```bash
python scripts/run_d1_checkpoint_collapse.py \
  --allow-debug \
  --indices data/selected_indices.json \
  --max-images 3 \
  --device cpu
```

---

## 4.6. Interpretación de D1

### Caso A — Collapse desde `epoch_1`

```text
epoch_1 top_pct alto
epoch_2 top_pct alto
epoch_3 top_pct alto
best    top_pct alto
```

Interpretación:

```text
El collapse aparece desde el inicio. Probablemente el problema viene del incentivo del entrenamiento, la distribución semántica del dataset o la loss.
```

Siguiente paso:

```text
Mirar D2.
```

### Caso B — Collapse progresivo

```text
epoch_1 top_pct bajo/medio
epoch_2 top_pct mayor
epoch_3 top_pct alto
best    top_pct alto
```

Interpretación:

```text
El modelo se vuelve cada vez más rígido. Puede haber overfitting al modo textual dominante.
```

Siguiente paso:

```text
Considerar checkpoint intermedio para D3 o ajustar entrenamiento.
```

### Caso C — `best/` colapsa más que un checkpoint intermedio

Interpretación:

```text
El menor validation loss no necesariamente produce el checkpoint más útil para interpretabilidad.
```

---

# 5. D2 — Probe de distribución token a token

## 5.1. Objetivo

D2 busca responder:

```text
¿El modelo está realmente muy seguro del token elegido, o greedy decoding está ocultando alternativas cercanas?
```

Para eso usa:

```python
generate(output_scores=True, return_dict_in_generate=True)
```

y analiza la distribución token a token.

---

## 5.2. Archivos de D2

```text
scripts/run_d2_token_probe.py
notebooks/07_debug_token_distribution.ipynb
```

### `scripts/run_d2_token_probe.py`

Es el motor de D2.

Hace:

1. carga checkpoints;
2. genera captions greedy;
3. obtiene scores/logits por token;
4. aplica softmax;
5. calcula top-1, top-2, gap, ratio y entropía;
6. guarda CSVs detallados y agregados.

No hace:

```text
- entrenamiento
- sampling
- heatmaps
- Grad-CAM
- modificación de checkpoints
```

### `notebooks/07_debug_token_distribution.ipynb`

Es el lector y visualizador de D2.

Hace:

1. muestra comandos;
2. carga CSVs;
3. grafica concentración de distribución;
4. muestra ejemplos de alta confianza;
5. muestra ejemplos de bajo margen;
6. permite inspeccionar tokens paso a paso;
7. ayuda a escribir una conclusión.

---

## 5.3. Outputs de D2

### `d2_token_probe_steps.csv`

Una fila por:

```text
checkpoint × imagen × token
```

Columnas principales:

```text
checkpoint
idx
caption
reference
step
generated_token
generated_text_piece
p_generated
generated_is_top1
top1_token
top1_text_piece
p_top1
top2_token
top2_text_piece
p_top2
gap_top1_top2
ratio_top1_top2
logit_gap_top1_top2
entropy
topk_mass
```

Uso:

```text
Debug fino token por token.
```

### `d2_image_summary.csv`

Una fila por:

```text
checkpoint × imagen
```

Uso:

```text
Seleccionar casos de alta confianza o bajo margen.
```

### `d2_checkpoint_summary.csv`

Una fila por checkpoint.

Columnas clave:

```text
checkpoint
n_images
n_steps_total
mean_p_top1
mean_p_top2
mean_gap
mean_entropy
pct_steps_top1_gt_090
pct_steps_top1_gt_095
pct_steps_gap_lt_010
```

Uso:

```text
Tabla principal de D2.
```

### `d2_high_confidence_examples.csv`

Casos donde:

```text
mean_p_top1 alto
mean_gap alto
```

Uso:

```text
Ejemplos donde el modelo está muy seguro.
```

### `d2_low_margin_examples.csv`

Casos donde:

```text
mean_gap bajo
```

Uso:

```text
Ejemplos donde greedy puede estar ocultando alternativas.
```

---

## 5.4. Métricas de D2

### `p_top1`

Probabilidad del token más probable en un paso.

```text
Alta → modelo muy seguro del token.
Media/baja → distribución menos rígida.
```

### `p_top2`

Probabilidad del segundo token más probable.

```text
Si está cerca de p_top1, hay alternativa plausible.
```

### `gap_top1_top2`

```text
gap = p_top1 - p_top2
```

Interpretación:

```text
Gap alto → top-1 domina claramente.
Gap bajo → top-1 y top-2 están cerca.
```

### `ratio_top1_top2`

```text
ratio = p_top1 / p_top2
```

Interpretación:

```text
Ratio alto → top-1 domina mucho.
Ratio cercano a 1 → alternativas muy cercanas.
```

### `entropy`

Mide la dispersión de la distribución.

```text
Entropía baja → distribución concentrada.
Entropía alta → distribución más abierta.
```

### `pct_steps_top1_gt_090`

Proporción de pasos donde:

```text
p_top1 > 0.90
```

Interpretación:

```text
Alto → el modelo decide muchos tokens con confianza extrema.
```

### `pct_steps_gap_lt_010`

Proporción de pasos donde:

```text
gap_top1_top2 < 0.10
```

Interpretación:

```text
Alto → muchas decisiones token a token son ambiguas.
```

---

## 5.5. Comandos de D2

### Dry-run

```bash
python scripts/run_d2_token_probe.py --dry-run
```

### Corrida real

```bash
python scripts/run_d2_token_probe.py \
  --checkpoint-root models/blip_finetuned_5k \
  --indices data/selected_indices.json \
  --max-images 30 \
  --device cpu
```

### Smoke test con debug

```bash
python scripts/run_d2_token_probe.py \
  --allow-debug \
  --indices data/selected_indices.json \
  --max-images 1 \
  --device cpu
```

---

## 5.6. Interpretación de D2

### Caso A — Distribución muy puntiaguda

Patrón:

```text
mean_p_top1 > 0.90
mean_gap > 0.50
pct_steps_top1_gt_090 alto
pct_steps_gap_lt_010 bajo
```

Interpretación:

```text
El modelo está realmente muy seguro. Cambiar decoding probablemente no alcance.
```

Siguiente paso:

```text
Considerar reentrenamiento, regularización, balanceo o cambios de loss.
```

### Caso B — Top-1 y top-2 cercanos

Patrón:

```text
mean_p_top1 entre 0.40 y 0.70
mean_gap < 0.10
pct_steps_gap_lt_010 alto
```

Interpretación:

```text
Greedy decoding puede estar amplificando diferencias pequeñas.
```

Siguiente paso:

```text
Probar sampling, temperature o nucleus decoding sin reentrenar.
```

### Caso C — Mixto

Interpretación:

```text
Algunos tokens son seguros y otros ambiguos. Conviene mirar ejemplos cualitativos.
```

---

# 6. D3 — Heatmap probe base vs fine-tuned

## 6.1. Objetivo

D3 busca responder:

```text
Aunque las captions estén colapsadas, ¿el fine-tuning cambió los heatmaps del modelo?
```

Compara:

```text
BLIP base
BLIP fine-tuned best/
```

sobre pocas radiografías fijas.

---

## 6.2. Archivos de D3

```text
scripts/run_d3_heatmap_probe.py
notebooks/08_debug_heatmaps_probe.ipynb
```

### `scripts/run_d3_heatmap_probe.py`

Es el motor de D3.

Hace:

1. carga BLIP base;
2. carga BLIP fine-tuneado;
3. carga pocas imágenes de `selected_indices.json`;
4. extrae QK logits / cross-attention por palabra;
5. extrae Grad-CAM por palabra;
6. guarda grillas de heatmaps;
7. guarda comparaciones cross-attention vs Grad-CAM;
8. guarda CSV/JSON resumen.

No hace:

```text
- entrenamiento
- métricas BLEU/CIDEr
- cambios en checkpoints
- regeneración de splits
```

### `notebooks/08_debug_heatmaps_probe.ipynb`

Es el lector y visualizador de D3.

Hace:

1. muestra comandos;
2. verifica outputs;
3. carga `d3_heatmap_summary.csv`;
4. muestra tablas;
5. muestra galerías base vs fine-tuned;
6. ayuda a escribir conclusión.

---

## 6.3. Señales visuales de D3

### QK logits / cross-attention

Se extraen con:

```python
from src.interpretability.cross_att_logits import extract_cross_att_logits
```

Qué representa:

```text
Afinidad espacial Q·K entre cada token generado y cada patch visual, antes del softmax.
```

Output conceptual:

```python
{
    "caption": "...",
    "maps": [
        ("word_1", array_24x24),
        ("word_2", array_24x24),
        ...
    ]
}
```

### Grad-CAM

Se extrae con:

```python
from src.interpretability.gradcam import compute_gradcam
```

Qué representa:

```text
Regiones del encoder visual que más influyen en el logit de un token generado.
```

Output conceptual:

```python
{
    "caption": "...",
    "maps": [
        ("word_1", array_24x24),
        ("word_2", array_24x24),
        ...
    ]
}
```

---

## 6.4. Outputs de D3

### `d3_heatmap_summary.csv`

Una fila por:

```text
modelo × imagen
```

Columnas clave:

```text
model_tag
model_path
idx
reference
original_path
cross_caption
gradcam_caption
n_cross_maps
n_gradcam_maps
cross_grid_path
gradcam_grid_path
comparison_path
cross_status
gradcam_status
error
```

### `d3_heatmap_summary.json`

Mismo resumen en JSON.

### `original.png`

Imagen original.

### `cross_att_logits_grid.png`

Grilla de QK logits / cross-attention por palabra.

### `gradcam_grid.png`

Grilla de Grad-CAM por palabra.

### `cross_vs_gradcam.png`

Comparación entre ambos métodos visuales.

---

## 6.5. Comandos de D3

### Dry-run

```bash
python scripts/run_d3_heatmap_probe.py --dry-run
```

### Corrida real

```bash
python scripts/run_d3_heatmap_probe.py \
  --ft-model-dir models/blip_finetuned_5k/best \
  --indices data/selected_indices.json \
  --max-images 5 \
  --device cpu
```

### Smoke test con debug

```bash
python scripts/run_d3_heatmap_probe.py \
  --allow-debug \
  --max-images 1 \
  --device cpu
```

### Solo QK logits, sin Grad-CAM

```bash
python scripts/run_d3_heatmap_probe.py \
  --ft-model-dir models/blip_finetuned_5k/best \
  --indices data/selected_indices.json \
  --max-images 2 \
  --device cpu \
  --skip-gradcam
```

---

## 6.6. Interpretación de D3

### Caso A — Heatmaps fine-tuned más coherentes

Patrón:

```text
base: mapas difusos o fuera de regiones clínicas
finetuned: mapas más concentrados en pulmones, corazón, dispositivos o hallazgos
```

Interpretación:

```text
El fine-tuning modificó la representación visual.
```

Resultado reportable:

```text
El modelo aprendió a mirar mejor.
```

### Caso B — Captions colapsadas pero heatmaps cambian

Interpretación:

```text
El modelo pudo haber aprendido a ver distinto aunque el decoder no exprese esa diversidad.
```

Resultado reportable:

```text
El encoder / puente visual-lingüístico cambió; el decoder textual colapsó.
```

### Caso C — Heatmaps iguales entre base y fine-tuned

Interpretación:

```text
El fine-tuning no modificó significativamente la mirada del modelo.
```

### Caso D — Cross-attention y Grad-CAM divergen

Interpretación:

```text
Las señales de interpretabilidad no están alineadas.
```

Debe reportarse como limitación o resultado negativo parcial.

---

# 7. Orden recomendado de uso

Orden lógico:

```text
1. D1
2. D2
3. D3
```

## 7.1. Por qué D1 primero

D1 dice si el collapse:

```text
- ya aparece en epoch_1
- aparece progresivamente
- empeora en best
- deja algún checkpoint intermedio más útil
```

## 7.2. Por qué D2 después

D2 explica si el collapse:

```text
- está internalizado en una distribución muy confiada
- o está amplificado por greedy decoding
```

## 7.3. Por qué D3 al final

D3 decide si el problema textual bloquea o no el análisis principal.

Pregunta clave:

```text
¿Los heatmaps ya cambiaron de forma útil aunque las captions sean malas?
```

---

# 8. Decisiones que se pueden tomar después de D1-D2-D3

## 8.1. Si D1 muestra checkpoint intermedio mejor

Usar ese checkpoint para D3 o compararlo contra `best`.

## 8.2. Si D2 muestra distribución poco puntiaguda

Probar S1:

```text
temperature
top-p / nucleus sampling
top-k sampling
```

## 8.3. Si D2 muestra distribución muy puntiaguda

Decoding no alcanza.

Opciones:

```text
reentrenar
bajar LR
label smoothing
balanceo semántico
más datos
```

## 8.4. Si D3 muestra heatmaps útiles

No hace falta resolver perfectamente el collapse para que el TP sea válido.

El collapse pasa a ser:

```text
limitación metodológica + hallazgo interpretativo
```

## 8.5. Si D3 no muestra cambios visuales

Puede ser necesario:

```text
revisar entrenamiento
usar más datos
probar checkpoint intermedio
analizar si el fine-tuning solo adaptó lenguaje
```

---

# 9. Qué no debe hacer otro modelo con estos archivos

Otro modelo no debe:

```text
- entrenar modelos dentro de D1/D2/D3
- modificar checkpoints
- borrar outputs sin consultar
- regenerar selected_indices.json
- mezclar D1 con sampling
- mezclar D2 con Grad-CAM
- mezclar D3 con BLEU/CIDEr/METEOR
- cambiar BLIP por otro modelo
- tocar src/interpretability sin necesidad
- convertir notebooks en scripts pesados
```

Regla general:

```text
Los scripts hacen cómputo.
Los notebooks leen, grafican e interpretan.
```

---

# 10. Dependencias por diagnóstico

## D1

Necesita:

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
```

## D2

Necesita:

```text
torch
transformers
datasets
pandas
Pillow
```

El notebook necesita:

```text
matplotlib
jupyter
```

No necesita:

```text
grad-cam
```

## D3

Necesita:

```text
torch
transformers
datasets
pandas
Pillow
matplotlib
grad-cam
```

El import de Grad-CAM es:

```python
pytorch_grad_cam
```

pero se instala con:

```bash
pip install grad-cam
```

---

# 11. Resumen final para otro LLM

D1, D2 y D3 son diagnósticos complementarios para estudiar el mode collapse del BLIP fine-tuneado en radiografías.

D1 mide diversidad de captions por checkpoint. Usa greedy decoding sobre las mismas imágenes y calcula `unique_ratio` y `top_pct` para saber cuándo aparece el collapse.

D2 mide la distribución token a token durante generación. Usa `generate(output_scores=True, return_dict_in_generate=True)` y calcula `p_top1`, `p_top2`, `gap`, `ratio` y `entropy` para decidir si el modelo está realmente seguro o si greedy decoding oculta alternativas.

D3 mira el lado visual. Compara BLIP base contra BLIP fine-tuned `best/` sobre pocas radiografías y guarda mapas QK logits / cross-attention y Grad-CAM. Sirve para decidir si el fine-tuning cambió la forma en que el modelo mira la radiografía aunque el decoder textual esté colapsado.

Los tres pasos no entrenan, no modifican checkpoints y no reemplazan el análisis principal. Son una fase de diagnóstico para decidir cómo seguir y cómo reportar el problema.
