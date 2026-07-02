# Handoff — Análisis para presentación preliminar del TP

**Fecha:** 2026-07-02  
**Estado:** pipeline corrido en VM, outputs descargados localmente, VM apagada.  
**Próximo paso:** análisis y figuras en CPU local para presentación preliminar.

---

## 1. Qué se hizo hasta acá

### Pipeline completado en VM (GPU L4)

Se corrió `notebooks/07_explicabilidad_comparada_v0.ipynb` en la VM con:

```
25 radiografías × 3 modelos × 3 métodos = 225 combinaciones
```

- **Modelos:** `base`, `ft5k` (`../output_5k/best`), `ft10k` (`models/blip_finetuned_10k/best`)
- **Métodos:** `post_softmax`, `qk_logits`, `gradcam`
- **Generación:** best-of-3 con seeds `[42, 43, 44]`, T=1.2, top_p=0.95, max 40 tokens
- **Imágenes:** `data/visual_test_indices.json` (25 índices fijos del set B)

### Outputs descargados localmente

```
outputs/notebook_comparativo/
├── arrays/
│   └── idx_<NNN>/                  ← 25 carpetas, una por imagen
│       ├── idx_<NNN>__base__post_softmax.npz
│       ├── idx_<NNN>__base__qk_logits.npz
│       ├── idx_<NNN>__base__gradcam.npz
│       ├── idx_<NNN>__ft5k__post_softmax.npz
│       ├── idx_<NNN>__ft5k__qk_logits.npz
│       ├── idx_<NNN>__ft5k__gradcam.npz
│       ├── idx_<NNN>__ft10k__post_softmax.npz
│       ├── idx_<NNN>__ft10k__qk_logits.npz
│       └── idx_<NNN>__ft10k__gradcam.npz
├── captions/
│   └── captions_bestof3.json       ← captions de los 3 modelos para las 25 imágenes
├── metrics/
│   ├── spatial_per_token.csv       ← métricas token a token (~2100 filas)
│   └── spatial_summary.csv        ← métricas por (idx, modelo, par de métodos)
└── summary.csv                     ← un renglón por (idx, modelo), 75 filas
```

Cada `.npz` contiene:
- `words`: tokens filtrados (sin stopwords ni puntuación)
- `heatmaps`: array `(n_tokens, 24, 24)` float32
- `caption`: la caption completa generada

Para cargar imágenes en el análisis local: usar el dataset desde `data/hf_cache/`
(no se guardaron las imágenes en los outputs — están en el dataset original).

---

## 2. Primeros hallazgos de las métricas

Métricas promedio sobre las 25 imágenes por modelo y par de métodos:

| Modelo | Par | Pearson | IoU top-10% |
|---|---|---|---|
| base  | post_softmax vs qk_logits | +0.52 | 0.667 |
| ft5k  | post_softmax vs qk_logits | +0.57 | 0.815 |
| ft10k | post_softmax vs qk_logits | +0.58 | 0.827 |
| base  | post_softmax vs gradcam   | -0.09 | 0.047 |
| ft5k  | post_softmax vs gradcam   | -0.09 | 0.038 |
| ft10k | post_softmax vs gradcam   | -0.09 | 0.059 |
| base  | qk_logits vs gradcam      | -0.09 | 0.053 |
| ft5k  | qk_logits vs gradcam      | +0.00 | 0.047 |
| ft10k | qk_logits vs gradcam      | +0.04 | 0.063 |

**Lecturas preliminares:**
- Post-softmax y QK-logits son consistentes entre sí y se vuelven **más parecidos
  después del finetuning** (IoU sube de 0.67 → 0.83). El finetuning consolida la
  atención hacia las mismas zonas.
- Grad-CAM está desacoplado de los dos métodos de atención en todos los modelos
  (IoU < 0.06 siempre). Son mecanismos de naturaleza distinta.
- El finetuning no cambia la relación entre QK-logits y Grad-CAM — siguen sin
  coincidir.

**Sobre las captions:**
- `base` genera captions genéricas con errores médicos graves
  ("a male lung with an pneumonia in the stomach").
- `ft5k` y `ft10k` generan vocabulario médico real (edema, atelectasis, effusion,
  pleural) aunque a veces con alguna incoherencia sintáctica.
- El número de tokens relevantes (médicos) por caption aumenta notablemente
  con el finetuning: `base` ≈ 4–9 tokens, `ft5k/ft10k` ≈ 6–12 tokens.

---

## 3. Objetivo — análisis para presentación preliminar

El objetivo es armar un análisis visual y cuantitativo que responda la pregunta
central del TP:

> ¿El fine-tuning hace que BLIP no solo aprenda a hablar el lenguaje médico
> sino también a mirar las regiones clínicamente relevantes?

Los tres resultados posibles son:
- **A:** captions mejoran Y heatmaps cambian hacia zonas relevantes
- **B:** captions mejoran pero heatmaps no cambian
- **C:** heatmaps cambian pero no hacia zonas relevantes

Para la presentación preliminar, el análisis debería cubrir:

### 3.1 Comparación de captions (base vs ft5k vs ft10k)
- Mostrar ejemplos side-by-side para 4–6 radiografías representativas
- Tabla con n_tokens_relevant por modelo (del `summary.csv`)
- Destacar diferencias cualitativas: vocabulario médico, coherencia, especificidad

### 3.2 Heatmaps comparativos por token médico
- Para las mismas radiografías: mostrar el heatmap del mismo token en base, ft5k
  y ft10k usando los `.npz`
- Foco en tokens de alta relevancia clínica: "effusion", "edema", "atelectasis",
  "pneumonia", "consolidation"
- Usar la función `compare_models_for_token` de `docs/analisis_post_vm.md`

### 3.3 Métricas cuantitativas
- Tabla resumen del punto 2 (ya calculada)
- Gráfico de barras: IoU post_softmax vs qk_logits por modelo
- Interpretación de qué significa que el IoU suba con el finetuning

### 3.4 Conclusión preliminar
Con los datos actuales, el escenario más probable es **B parcial hacia A**:
- Las captions mejoran claramente (vocabulario médico, especificidad)
- Los métodos de atención se vuelven más consistentes entre sí post-finetuning
  (IoU 0.67 → 0.83), lo que sugiere que la atención se concentra más
- Si los heatmaps muestran visualmente zonas médicamente plausibles (pulmones,
  pleura, mediastino) en ft5k/ft10k vs dispersión en base, es resultado A

---

## 4. Cómo proceder — paso a paso

### Paso 1 — Cargar datos
```python
import numpy as np, json, csv
from pathlib import Path

ROOT = Path(".")  # desde image-captioning/
ARRAYS = ROOT / "outputs/notebook_comparativo/arrays"

# Cargar una imagen del dataset para visualización
from datasets import load_dataset
ds = load_dataset("itsanmolgupta/mimic-cxr-dataset",
                  cache_dir=str(ROOT / "data/hf_cache"))

with open(ROOT / "data/visual_test_indices.json") as f:
    visual_indices = json.load(f)

# Mapeo idx_str → índice numérico del dataset
idx_map = {f"idx_{i:04d}": i for i in visual_indices}
```

### Paso 2 — Función de visualización comparativa
Ver `docs/analisis_post_vm.md` secciones 4 y 5 para el código completo.
Las dos funciones clave son:
- `plot_medical_heatmaps(image_path, npz_path)` — solo tokens médicos de un modelo
- `compare_models_for_token(image_path, idx_str, token)` — mismo token, 3 modelos

### Paso 3 — Selección de ejemplos para la presentación
Elegir 4–6 imágenes donde:
1. `ft10k` genera un token médico específico que `base` no genera
2. El heatmap de ese token tiene estructura espacial (no es uniforme)
3. La zona activada coincide visualmente con la zona clínica (pulmón, pleura)

### Paso 4 — Figuras con calidad de presentación
```python
import matplotlib.pyplot as plt
# DPI alto para poster/presentación
fig.savefig("outputs/figures_paper/figura_comparativa.png", dpi=200, bbox_inches="tight")
```

---

## 5. Archivos de referencia

| Archivo | Qué tiene |
|---|---|
| `docs/analisis_post_vm.md` | Código para cargar `.npz` y visualizar |
| `docs/metricas_espaciales.md` | Explicación de las 4 métricas |
| `docs/teacher_forcing_interpretabilidad.md` | Explicación del pipeline técnico |
| `src/interpretability/token_filter.py` | Vocabulario médico (`MEDICAL`) y stopwords |
| `outputs/notebook_comparativo/metrics/spatial_summary.csv` | Métricas ya calculadas |
| `outputs/notebook_comparativo/captions/captions_bestof3.json` | Captions generadas |
| `outputs/notebook_comparativo/summary.csv` | Resumen por (imagen, modelo) |
| `CLAUDE.md` §1 | Pregunta de investigación y resultados posibles A/B/C |

---

## 6. Lo que NO hay que hacer en este punto

- **No volver a correr la VM** — todos los datos están en local.
- **No modificar** `data/visual_test_indices.json` ni los checkpoints.
- **No regenerar** los `.npz` — llevaría varias horas en CPU.
- **No usar greedy decoding** — el mode collapse está resuelto con T=1.2.
