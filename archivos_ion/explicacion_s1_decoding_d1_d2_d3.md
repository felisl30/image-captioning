# Explicación de S1 — Decoding strategies y relación con D1/D2/D3

Este documento explica, para otra LLM o para un asistente de código, qué hacen los archivos del paso **S1** dentro del proyecto de fine-tuning de BLIP sobre radiografías MIMIC-CXR, y cómo se relacionan con los diagnósticos previos **D1**, **D2** y **D3**.

---

## 1. Contexto general del problema

El proyecto fine-tunea `Salesforce/blip-image-captioning-base` sobre radiografías de tórax usando el campo `impression` como target textual.

Después del fine-tuning inicial sobre un subconjunto de datos, el modelo mostró **mode collapse**: en vez de generar captions variados y específicos, tendía a repetir una o dos frases genéricas, por ejemplo:

```text
no acute cardiopulmonary process.
no significant interval change.
```

Este comportamiento es problemático porque puede ocultar si el modelo realmente aprendió información clínica visual o si solo aprendió a emitir una frase común.

Para estudiar este problema se dividieron las tareas en diagnósticos `D1`, `D2`, `D3` y una primera estrategia de solución `S1`.

---

## 2. Ubicación de los archivos S1

Los archivos de S1 son:

```text
scripts/run_s1_decoding_experiment.py
notebooks/09_debug_decoding_strategies.ipynb
```

La estructura esperada dentro del proyecto es:

```text
image-captioning/
├── scripts/
│   └── run_s1_decoding_experiment.py
│
├── notebooks/
│   └── 09_debug_decoding_strategies.ipynb
│
└── outputs/
    └── decoding_sampling/
        └── s1_selected_30/
```

---

## 3. Qué intenta responder S1

S1 responde esta pregunta:

> ¿El mode collapse se debe principalmente al modo de generación greedy, o el modelo realmente internalizó una distribución colapsada?

En otras palabras:

- Si con `greedy` el modelo repite siempre lo mismo, pero con sampling genera captions más diversos y razonables, entonces el problema puede estar parcialmente en el decoding.
- Si todas las estrategias generan casi las mismas frases, entonces el collapse está internalizado en el modelo y no se arregla solo con inferencia.

S1 es una estrategia de **bajo costo**, porque:

- no requiere GPU;
- no reentrena;
- no modifica checkpoints;
- solo cambia parámetros de `model.generate()`.

---

## 4. Script S1: `scripts/run_s1_decoding_experiment.py`

### 4.1 Propósito

Este script es el motor computacional de S1.

Carga un checkpoint fine-tuneado, toma las imágenes fijas de `data/selected_indices.json`, genera captions con distintas estrategias de decoding y guarda resultados en CSV/JSON.

### 4.2 Inputs principales

El script espera:

```text
models/blip_finetuned_5k/best/
```

o alternativamente:

```text
models/blip_finetuned/best/
```

También usa:

```text
data/selected_indices.json
data/hf_cache/
```

`selected_indices.json` contiene las 30 radiografías fijas que se vienen usando para comparar antes/después del fine-tuning. No debe modificarse.

### 4.3 Estrategias de decoding que prueba

El script compara varias estrategias:

#### 1. Greedy

```text
num_beams=1
do_sample=False
```

Es el baseline. Elige siempre el token más probable.

Si greedy colapsa, eso no alcanza para concluir que el modelo no conoce alternativas; puede ser que simplemente el argmax sea demasiado dominante.

#### 2. Temperature + nucleus sampling suave

```text
temperature=1.2
top_p=0.95
```

Busca un aumento leve de diversidad.

#### 3. Temperature + nucleus sampling recomendado

```text
temperature=1.3
top_p=0.90
```

Es el punto medio razonable: más diversidad, pero intentando evitar demasiado ruido.

#### 4. Temperature + nucleus sampling agresivo

```text
temperature=1.5
top_p=0.85
```

Genera más variedad, pero con mayor riesgo de captions incoherentes.

#### 5. Diverse beam search

```text
num_beams=8
num_beam_groups=4
diversity_penalty=0.5
```

Fuerza diversidad entre beams. Sirve para ver si el modelo tiene varias salidas plausibles escondidas.

#### 6. Contrastive decoding

```text
penalty_alpha=0.6
top_k=4
```

Intenta favorecer generaciones más informativas y menos repetitivas.

Puede depender de la versión de `transformers`; por eso el script captura errores por estrategia.

---

## 5. Outputs del script S1

El script guarda todo en:

```text
outputs/decoding_sampling/s1_selected_30/
```

Archivos principales:

```text
s1_all_captions.csv
s1_decoding_summary.csv
s1_image_strategy_summary.csv
s1_examples.csv
s1_errors.csv
s1_decoding_summary.json
s1_strategy_comparison.png
```

### 5.1 `s1_all_captions.csv`

Contiene una fila por caption generado.

Columnas importantes:

```text
model_tag
strategy
idx
sample_id
reference
caption
caption_norm
len_words
categories
```

Sirve para inspección cualitativa.

### 5.2 `s1_decoding_summary.csv`

Resumen agregado por estrategia.

Columnas importantes:

```text
strategy
n_images
n_captions
n_unique
unique_ratio
top_caption_norm
top_count
top_pct
mean_len_words
pct_normal
pct_specific_clinical
```

Interpretación:

- `unique_ratio` alto: más diversidad.
- `top_pct` bajo: menos repetición.
- `pct_specific_clinical` alto: mayor presencia de términos clínicos específicos.
- `pct_normal` alto: mayor tendencia a captions normales o genéricas.

### 5.3 `s1_image_strategy_summary.csv`

Resumen por imagen y estrategia.

Sirve para detectar casos donde una estrategia mejora una imagen concreta o donde el collapse persiste.

### 5.4 `s1_examples.csv`

Ejemplos cualitativos por estrategia.

Incluye captions iniciales y ejemplos de la caption dominante.

### 5.5 `s1_errors.csv`

Errores por estrategia, si los hay.

Puede ser útil especialmente para `contrastive` o `diverse_beam`, porque algunas versiones de `transformers` pueden manejar esos parámetros de manera distinta.

### 5.6 `s1_strategy_comparison.png`

Gráfico que compara:

```text
unique_ratio
top_pct
```

por estrategia.

---

## 6. Notebook S1: `notebooks/09_debug_decoding_strategies.ipynb`

### 6.1 Propósito

Este notebook es liviano. No implementa generación ni lógica pesada.

Su función es:

1. mostrar los comandos para correr S1;
2. cargar los CSVs generados por el script;
3. mostrar tablas comparativas;
4. graficar diversidad vs repetición;
5. graficar contenido clínico;
6. permitir inspección cualitativa;
7. dejar una conclusión preliminar para el informe.

### 6.2 Secciones principales

El notebook está organizado así:

```text
0. Setup
1. Comandos para correr S1
2. Cargar outputs
3. Resumen principal por estrategia
4. Gráfico: diversidad vs repetición
5. Gráfico: contenido clínico
6. Ranking simple de estrategias
7. Inspección cualitativa por estrategia
8. Inspección por imagen
9. Casos donde S1 mejora diversidad
10. Casos donde sigue el collapse
11. Errores
12. Lectura automática preliminar
13. Conclusión para informe
```

### 6.3 Qué NO debe hacer el notebook

El notebook no debería:

- cargar el modelo;
- generar captions;
- implementar loops sobre imágenes;
- repetir la lógica del script;
- modificar checkpoints;
- modificar `selected_indices.json`.

La regla del proyecto es que los notebooks sean delgados y que la lógica reutilizable viva en scripts o módulos.

---

## 7. Comandos esperados para S1

### 7.1 Dry-run

Verifica paths y configuración sin correr inferencia:

```bash
python scripts/run_s1_decoding_experiment.py \
  --model-dir models/blip_finetuned_5k/best \
  --indices data/selected_indices.json \
  --max-images 30 \
  --device cpu \
  --dry-run
```

### 7.2 Smoke test con checkpoint debug

No sirve para resultado final, pero permite probar que el script corre:

```bash
python scripts/run_s1_decoding_experiment.py \
  --allow-debug \
  --indices data/selected_indices.json \
  --max-images 2 \
  --samples-per-image 2 \
  --device cpu
```

### 7.3 Corrida real

Cuando esté el checkpoint real:

```bash
python scripts/run_s1_decoding_experiment.py \
  --model-dir models/blip_finetuned_5k/best \
  --indices data/selected_indices.json \
  --max-images 30 \
  --samples-per-image 4 \
  --device cpu
```

---

## 8. Cómo interpretar S1

### Caso A — Sampling mejora diversidad y mantiene coherencia

Ejemplo:

```text
greedy: unique_ratio bajo, top_pct alto
sampling: unique_ratio alto, top_pct bajo, captions razonables
```

Interpretación:

> El modelo tenía alternativas plausibles, pero greedy amplificaba el collapse.

En este caso, se puede usar una estrategia de sampling como generación final y reportar que el modo greedy era demasiado restrictivo.

### Caso B — Sampling mejora diversidad pero introduce ruido

Ejemplo:

```text
sampling genera captions distintas, pero clínicamente incoherentes
```

Interpretación:

> Hay un trade-off entre diversidad y confiabilidad clínica.

En este caso, S1 sirve como análisis, pero no necesariamente como solución final.

### Caso C — Ninguna estrategia mejora

Ejemplo:

```text
greedy, sampling, diverse beam y contrastive repiten la misma frase
```

Interpretación:

> El collapse está internalizado en el modelo, no es solo un problema del decoding.

En ese caso, las soluciones siguientes serían:

- entrenar con más datos;
- balanceo semántico;
- ajustar learning rate;
- label smoothing;
- mejorar exposición a hallazgos raros.

---

## 9. Relación entre S1 y D1

### D1: `run_d1_checkpoint_collapse.py` + `06_debug_mode_collapse.ipynb`

D1 analiza **cuándo** aparece el collapse durante el fine-tuning.

Compara captions generadas por:

```text
epoch_1
epoch_2
epoch_3
best
```

sobre las mismas imágenes.

D1 responde:

> ¿El modelo ya colapsa desde la primera época o el collapse aparece progresivamente?

### Cómo se conecta con S1

Si D1 muestra que:

#### El collapse aparece recién en épocas posteriores

Entonces puede haber overfitting o convergencia hacia frases dominantes. S1 puede probar si `best/` todavía conserva alternativas plausibles que greedy oculta.

#### El collapse aparece desde `epoch_1`

Entonces S1 ayuda a separar dos posibilidades:

- greedy está amplificando un sesgo temprano;
- o el fine-tuning ya internalizó el patrón colapsado desde el inicio.

### Lectura combinada

```text
D1 dice cuándo aparece el collapse.
S1 dice si el collapse se puede mitigar cambiando decoding.
```

---

## 10. Relación entre S1 y D2

### D2: `run_d2_token_probe.py` + `07_debug_token_distribution.ipynb`

D2 estudia la distribución token a token durante generación.

Mide:

```text
p_top1
p_top2
gap_top1_top2
entropy
```

D2 responde:

> ¿El modelo está realmente muy confiado en el token dominante o hay alternativas cercanas?

### Cómo se conecta con S1

Si D2 muestra:

#### Alta confianza y gran gap

```text
p_top1 alto
p_top2 bajo
gap grande
```

Entonces es probable que S1 no mejore mucho. El modelo está seguro de la frase dominante.

#### Margen bajo entre top-1 y top-2

```text
p_top1 y p_top2 cercanos
gap pequeño
entropy alta
```

Entonces S1 tiene más chances de mejorar: sampling puede elegir alternativas razonables.

### Lectura combinada

```text
D2 dice si hay alternativas token a token.
S1 prueba si esas alternativas producen captions completas más diversas.
```

---

## 11. Relación entre S1 y D3

### D3: `run_d3_heatmap_probe.py` + `08_debug_heatmaps_probe.ipynb`

D3 compara heatmaps de BLIP base vs fine-tuned:

```text
cross-attention / QK logits
Grad-CAM
```

D3 responde:

> Aunque el caption colapse, ¿los mapas visuales cambiaron después del fine-tuning?

### Cómo se conecta con S1

S1 trabaja sobre el texto generado. D3 trabaja sobre la explicación visual.

Puede ocurrir:

#### S1 mejora captions y D3 mejora heatmaps

Resultado fuerte: el fine-tuning mejoró tanto lenguaje como atención visual.

#### S1 no mejora captions, pero D3 sí muestra cambios visuales

Resultado interesante: el modelo reorganizó su representación visual, pero el decoder quedó colapsado lingüísticamente.

Esto se puede reportar como:

> El fine-tuning parece afectar la mirada del modelo, aunque el componente textual no expresa bien la diversidad clínica.

#### S1 mejora captions, pero D3 no cambia heatmaps

Resultado posible: el fine-tuning mejoró sobre todo el lenguaje, no la mirada.

### Lectura combinada

```text
S1 evalúa si el output textual se puede rescatar por decoding.
D3 evalúa si la representación visual cambió aunque el texto siga colapsado.
```

---

## 12. Mapa conceptual D1/D2/D3/S1

```text
D1 — Evolución por checkpoint
    Pregunta:
        ¿cuándo aparece el collapse?
    Necesita:
        epoch_1, epoch_2, epoch_3, best

D2 — Distribución token a token
    Pregunta:
        ¿hay alternativas cercanas al top-1?
    Necesita:
        checkpoint(s), idealmente best y/o epochs

D3 — Heatmaps base vs fine-tuned
    Pregunta:
        ¿cambió la mirada visual del modelo?
    Necesita:
        BLIP base + best fine-tuned

S1 — Decoding strategies
    Pregunta:
        ¿se puede mitigar el collapse sin reentrenar?
    Necesita:
        best fine-tuned
```

---

## 13. Dependencias y prerequisitos

S1 necesita:

```text
torch
transformers
datasets
pandas
matplotlib
Pillow
```

No necesita:

```text
grad-cam
CUDA
GPU
epoch_1/epoch_2/epoch_3
```

A diferencia de D3, S1 no usa interpretabilidad visual ni Grad-CAM.

---

## 14. Estado esperado antes de correr S1

Para corrida real:

```text
models/blip_finetuned_5k/best/
```

o:

```text
models/blip_finetuned/best/
```

debe contener al menos:

```text
config.json
model.safetensors o pytorch_model.bin
archivos del processor/tokenizer
```

También debe existir:

```text
data/selected_indices.json
data/hf_cache/
```

---

## 15. Qué debe mirar otra LLM al revisar S1

Una LLM que revise este módulo debería verificar:

1. Que `run_s1_decoding_experiment.py` no entrena ni modifica modelos.
2. Que usa `selected_indices.json` y no genera nuevos índices.
3. Que compara estrategias sobre las mismas imágenes.
4. Que guarda outputs reproducibles en `outputs/decoding_sampling/s1_selected_30/`.
5. Que el notebook solo lee outputs y no duplica la lógica del script.
6. Que la interpretación no se base solo en `unique_ratio`, sino también en revisión cualitativa.
7. Que sampling no sea considerado automáticamente “mejor” si genera captions clínicamente absurdas.
8. Que S1 se interprete junto con D1, D2 y D3.

---

## 16. Conclusión

S1 es la primera estrategia de mitigación del mode collapse porque es barata, rápida y no requiere reentrenar.

Su rol dentro del pipeline es:

```text
D1 diagnostica cuándo aparece el collapse.
D2 diagnostica si hay alternativas token a token.
D3 evalúa si los heatmaps siguen siendo informativos.
S1 prueba si cambiando decoding se recupera diversidad textual.
```

Si S1 funciona, puede usarse como método de generación final o como análisis adicional en el informe.

Si S1 no funciona, el diagnóstico apunta a que el modelo necesita una solución más profunda: más datos, balanceo semántico o cambios en el entrenamiento.
