# Handoff técnico — D2: probe de distribución token a token

Este documento explica, para otro modelo de lenguaje o asistente de código, qué hacen los archivos relacionados con **D2** dentro del proyecto de fine-tuning de BLIP sobre radiografías.

## 1. Contexto general

El proyecto fine-tunea `Salesforce/blip-image-captioning-base` sobre radiografías de tórax del dataset MIMIC-CXR usando el campo `impression` como target textual.

Después del fine-tuning aparece un problema de **mode collapse**: el modelo tiende a repetir una o dos captions dominantes, por ejemplo:

```text
no acute cardiopulmonary process.
no significant interval change.
```

D1 responde **cuándo aparece** ese collapse comparando checkpoints.

D2 responde otra pregunta:

```text
¿El modelo está realmente seguro de la frase colapsada, o greedy decoding está ocultando alternativas cercanas?
```

En otras palabras: D2 mira la distribución de probabilidad del decoder **por token generado**.

## 2. Archivos involucrados en D2

Los archivos relevantes son:

```text
scripts/run_d2_token_probe.py
notebooks/07_debug_token_distribution.ipynb
```

Y los outputs esperados se guardan en:

```text
outputs/mode_collapse_debug/
```

Estructura esperada:

```text
image-captioning/
├── scripts/
│   └── run_d2_token_probe.py
├── notebooks/
│   └── 07_debug_token_distribution.ipynb
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
        ├── d2_token_probe_steps.csv
        ├── d2_image_summary.csv
        ├── d2_checkpoint_summary.csv
        ├── d2_checkpoint_summary.json
        ├── d2_high_confidence_examples.csv
        └── d2_low_margin_examples.csv
```

## 3. Qué es D2

D2 es un diagnóstico de distribución interna durante generación.

El script genera captions con greedy decoding, pero además pide a HuggingFace que devuelva los logits/scores de cada paso:

```python
out = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    num_beams=1,
    do_sample=False,
    output_scores=True,
    return_dict_in_generate=True,
)
```

Luego, para cada token generado, calcula:

```text
p_top1
p_top2
gap_top1_top2
ratio_top1_top2
logit_gap_top1_top2
entropy
topk_mass
```

Esto permite saber si el token elegido por greedy era una decisión obvia o si había alternativas muy cercanas.

## 4. `scripts/run_d2_token_probe.py`

### Propósito

Este script es el **motor real de D2**.

Carga checkpoints de BLIP fine-tuneado, genera captions greedy sobre radiografías fijas y guarda métricas token a token sobre la distribución de probabilidad del decoder.

No entrena, no modifica checkpoints y no calcula heatmaps.

### Checkpoints que compara

Busca estos checkpoints:

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

Una carpeta es usable si contiene:

```text
config.json
```

y además alguno de estos archivos de pesos:

```text
model.safetensors
pytorch_model.bin
```

El script también reporta si encuentra archivos de tokenizer/processor:

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

La idea es usar exactamente las mismas imágenes fijas que D1 y D3.

### Función central: `generate_with_token_probe`

Esta función:

1. Convierte la radiografía a RGB.
2. Procesa la imagen con `BlipProcessor`.
3. Genera un caption con greedy decoding.
4. Recupera `out.scores`.
5. Para cada paso de generación:
   - aplica softmax a los logits;
   - obtiene top-k tokens;
   - identifica top-1 y top-2;
   - calcula probabilidades, gaps, ratio, entropía y masa top-k;
   - guarda el token generado y su texto decodificado.

El output de esta función es:

```python
caption, step_rows
```

donde:

```text
caption   → caption final decodificada
step_rows → lista de filas, una por token generado
```

### Funciones de resumen

El script tiene dos niveles de agregación.

#### `summarize_by_image`

Agrupa por:

```text
checkpoint
idx
```

y genera métricas promedio por imagen:

```text
mean_p_top1
median_p_top1
mean_p_top2
median_p_top2
mean_gap
median_gap
mean_ratio
median_ratio
mean_entropy
mean_topk_mass
pct_steps_top1_gt_090
pct_steps_top1_gt_095
pct_steps_top1_gt_099
pct_steps_gap_lt_010
pct_steps_gap_lt_005
pct_generated_is_top1
```

#### `summarize_by_checkpoint`

Agrupa por:

```text
checkpoint
```

y resume el comportamiento promedio de cada checkpoint.

Este archivo es el más importante para interpretación global:

```text
d2_checkpoint_summary.csv
```

## 5. Outputs de D2

### `d2_token_probe_steps.csv`

Archivo más detallado. Una fila por:

```text
checkpoint
imagen
token generado
```

Columnas principales:

```text
checkpoint
checkpoint_path
idx
caption
reference
step
generated_token_id
generated_token
generated_text_piece
p_generated
generated_is_top1
top1_id
top1_token
top1_text_piece
p_top1
top2_id
top2_token
top2_text_piece
p_top2
gap_top1_top2
ratio_top1_top2
logit_top1
logit_top2
logit_gap_top1_top2
entropy
topk_mass
topk_ids_json
topk_tokens_json
topk_probs_json
```

Uso:

```text
Debug fino token por token.
```

Ejemplo de lectura:

```text
step=3
top1_text_piece="acute"
p_top1=0.96
top2_text_piece="significant"
p_top2=0.02
gap_top1_top2=0.94
```

Interpretación:

```text
El modelo estaba muy seguro de elegir "acute"; sampling probablemente no cambiaría mucho ese token.
```

### `d2_image_summary.csv`

Una fila por checkpoint e imagen.

Uso:

```text
Encontrar imágenes donde el modelo está muy confiado o muy ambiguo.
```

Sirve para seleccionar ejemplos cualitativos.

### `d2_checkpoint_summary.csv`

Una fila por checkpoint.

Uso:

```text
Tabla principal de D2.
```

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
pct_steps_top1_gt_099
pct_steps_gap_lt_010
pct_steps_gap_lt_005
```

### `d2_checkpoint_summary.json`

Mismo contenido que el CSV resumen, pero en JSON.

Uso:

```text
Consumo programático por scripts, notebooks o modelos de lenguaje.
```

### `d2_high_confidence_examples.csv`

Imágenes donde el modelo fue más confiado.

Ordena por:

```text
mean_p_top1 alto
mean_gap alto
```

Uso:

```text
Ejemplos de distribución muy puntiaguda.
```

### `d2_low_margin_examples.csv`

Imágenes donde top-1 y top-2 estuvieron más cerca.

Ordena por:

```text
mean_gap bajo
mean_p_top1 bajo
```

Uso:

```text
Ejemplos donde greedy decoding podría estar ocultando alternativas plausibles.
```

## 6. `notebooks/07_debug_token_distribution.ipynb`

### Propósito

Este notebook es el **lector y visualizador de D2**.

No debería hacer inferencia pesada. El trabajo pesado vive en:

```text
scripts/run_d2_token_probe.py
```

El notebook sirve para:

1. mostrar comandos D2;
2. leer CSVs generados por el script;
3. mostrar resumen por checkpoint;
4. graficar concentración de distribución;
5. mostrar ejemplos de alta confianza;
6. mostrar ejemplos de bajo margen;
7. inspeccionar una imagen token a token;
8. escribir una conclusión para el informe.

### Secciones del notebook

El notebook contiene estas secciones:

```text
1. Comandos D2
2. Carga de resultados
3. Resumen por checkpoint
4. Gráfico principal
5. Porcentaje de decisiones muy confiadas o ambiguas
6. Ejemplos de alta confianza
7. Ejemplos de bajo margen
8. Inspección token a token
9. Interpretación automática preliminar
10. Conclusión para informe
```

### Gráficos principales

El notebook genera gráficos con:

```text
mean_p_top1
mean_p_top2
mean_gap
```

y también:

```text
pct_steps_top1_gt_090
pct_steps_top1_gt_095
pct_steps_gap_lt_010
```

Estos gráficos permiten ver si un checkpoint se vuelve más rígido o más ambiguo durante el entrenamiento.

## 7. Comandos principales de D2

### Verificar sin correr inferencia

```bash
python scripts/run_d2_token_probe.py --dry-run
```

### Corrida real cuando estén los checkpoints

```bash
python scripts/run_d2_token_probe.py \
  --checkpoint-root models/blip_finetuned_5k \
  --indices data/selected_indices.json \
  --max-images 30 \
  --device cpu
```

### Smoke test con checkpoints debug

```bash
python scripts/run_d2_token_probe.py \
  --allow-debug \
  --indices data/selected_indices.json \
  --max-images 1 \
  --device cpu
```

### Incluir BLIP base como referencia

```bash
python scripts/run_d2_token_probe.py \
  --checkpoint-root models/blip_finetuned_5k \
  --indices data/selected_indices.json \
  --max-images 30 \
  --device cpu \
  --include-base
```

### Cambiar top-k guardado

```bash
python scripts/run_d2_token_probe.py \
  --checkpoint-root models/blip_finetuned_5k \
  --indices data/selected_indices.json \
  --max-images 30 \
  --device cpu \
  --top-k 10
```

### Incluir tokens especiales

Por defecto, D2 puede omitir tokens especiales como EOS. Si se quieren incluir:

```bash
python scripts/run_d2_token_probe.py \
  --checkpoint-root models/blip_finetuned_5k \
  --include-special-tokens
```

## 8. Cómo interpretar D2

### Caso A — Distribución muy confiada

Patrón:

```text
mean_p_top1 > 0.90
mean_gap > 0.50
pct_steps_top1_gt_090 alto
pct_steps_gap_lt_010 bajo
```

Interpretación:

```text
El modelo realmente está muy seguro de sus tokens. Greedy decoding no es el único problema.
```

Consecuencia:

```text
Cambiar a sampling puede no alcanzar. Habría que considerar reentrenamiento, regularización, balanceo o cambios de loss.
```

### Caso B — Top-1 y top-2 cercanos

Patrón:

```text
mean_p_top1 entre 0.40 y 0.70
mean_gap bajo, por ejemplo < 0.10
pct_steps_gap_lt_010 alto
```

Interpretación:

```text
El modelo conoce alternativas cercanas, pero greedy siempre elige la más probable.
```

Consecuencia:

```text
S1, es decir sampling / temperature / nucleus decoding, probablemente pueda mejorar diversidad sin reentrenar.
```

### Caso C — Distribución mixta

Patrón:

```text
mean_p_top1 medio
mean_gap medio
algunos tokens muy seguros
otros tokens ambiguos
```

Interpretación:

```text
El collapse puede venir de una combinación entre distribución semántica dominante y efecto greedy.
```

Consecuencia:

```text
Conviene mirar ejemplos token a token y comparar con D1.
```

## 9. Relación con D1

D1 mide diversidad textual por checkpoint:

```text
¿cuántas captions únicas genera cada checkpoint?
```

D2 mide la forma de la distribución interna:

```text
¿qué tan seguro estaba el modelo en cada token?
```

Relación práctica:

```text
D1 detecta el síntoma.
D2 ayuda a inferir la causa.
```

Ejemplo:

```text
D1: best tiene top_pct = 0.90
D2: best tiene mean_p_top1 = 0.97 y mean_gap = 0.85
```

Lectura:

```text
El collapse no parece ser solo un artefacto de greedy decoding. El modelo está muy confiado.
```

Otro ejemplo:

```text
D1: best tiene top_pct = 0.90
D2: best tiene mean_p_top1 = 0.55 y mean_gap = 0.05
```

Lectura:

```text
Greedy decoding probablemente está amplificando diferencias chicas. Probar sampling puede tener sentido.
```

## 10. Relación con S1

S1 es probar estrategias de decoding sin reentrenar:

```text
temperature
top-p / nucleus sampling
top-k sampling
beam search diverso
```

D2 decide si S1 tiene chances.

Regla práctica:

```text
Si D2 muestra gaps bajos → probar S1.
Si D2 muestra gaps altos → S1 probablemente no alcanza.
```

## 11. Relación con D3

D3 mira heatmaps base vs fine-tuned.

D2 no mira atención ni Grad-CAM. Solo mira distribución textual.

Relación:

```text
D2 dice si el decoder está rígido.
D3 dice si, pese a eso, la parte visual cambió de forma interesante.
```

Puede pasar que:

```text
D2 diga: el decoder está colapsado.
D3 diga: los heatmaps sí cambiaron.
```

Ese es un resultado reportable:

```text
El fine-tuning pudo haber modificado la representación visual aunque el decoder textual colapse.
```

## 12. Qué no debe hacer otro modelo al tocar D2

No debe:

- convertir D2 en entrenamiento;
- modificar `src/models/finetuner.py`;
- cambiar los checkpoints;
- regenerar `selected_indices.json`;
- usar sampling dentro de D2;
- usar `do_sample=True`;
- calcular BLEU/CIDEr/METEOR acá;
- calcular Grad-CAM o cross-attention acá;
- guardar modelos nuevos;
- borrar outputs previos sin consultar.

D2 debe mantenerse como diagnóstico token a token bajo greedy decoding.

## 13. Dependencias técnicas

D2 necesita:

```text
torch
transformers
datasets
pandas
Pillow
```

El notebook necesita además:

```text
matplotlib
jupyter
ipykernel
```

D2 no necesita:

```text
grad-cam
pytorch_grad_cam
pycocoevalcap
```

## 14. Costos computacionales

D2 es más liviano que D3, porque no hace backward ni heatmaps.

Pero es más pesado que D1 porque guarda métricas por token y calcula softmax completo sobre el vocabulario en cada paso.

Costo aproximado:

```text
n_checkpoints × n_images × n_tokens
```

Ejemplo:

```text
4 checkpoints × 30 imágenes × 20 tokens ≈ 2400 pasos de análisis
```

En CPU puede tardar, pero es razonable para 30 imágenes.

## 15. Estado esperado antes de correr D2 real

Para correr D2 de forma válida deberían existir:

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

También debería existir:

```text
data/selected_indices.json
data/hf_cache/
models/blip_base/
```

Si solo existen checkpoints debug, D2 puede correrse con `--allow-debug`, pero los resultados no deben usarse como conclusión final.

## 16. Checklist para otro LLM antes de modificar algo

Antes de tocar código de D2, verificar:

```text
[ ] El script sigue usando greedy decoding.
[ ] El script sigue usando output_scores=True.
[ ] El script sigue guardando d2_token_probe_steps.csv.
[ ] El script sigue guardando d2_checkpoint_summary.csv.
[ ] El notebook sigue siendo liviano y solo lee outputs.
[ ] No se mezcló D2 con D3.
[ ] No se modificó selected_indices.json.
[ ] No se cambió la estructura de checkpoints.
```

## 17. Resumen ejecutivo para otro LLM

D2 es un diagnóstico de distribución token a token para explicar el mode collapse. El script `scripts/run_d2_token_probe.py` carga checkpoints (`epoch_1`, `epoch_2`, `epoch_3`, `best`), genera captions greedy sobre las radiografías fijas de `data/selected_indices.json` y usa `generate(output_scores=True, return_dict_in_generate=True)` para obtener scores por paso. Para cada token calcula probabilidades top-1/top-2, gap, ratio, entropía y masa top-k. Guarda CSVs detallados por token, agregados por imagen y agregados por checkpoint. El notebook `notebooks/07_debug_token_distribution.ipynb` lee esos CSVs, grafica concentración de distribución y muestra ejemplos de alta confianza y bajo margen. D2 sirve para decidir si el collapse es una distribución realmente puntiaguda o si greedy decoding está ocultando alternativas cercanas. No entrena, no modifica checkpoints y no calcula heatmaps.
