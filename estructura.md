# Estructura del Proyecto — BLIP Fine-tuning e Interpretabilidad Visual
**I308 Visión Artificial — Otoño 2026**
**Autor:** Felipe
**Tema:** Fine-tuning de BLIP sobre radiografías y análisis de cross-attention + Grad-CAM

> **Estado del documento (2026-06-28):** refleja el repo **real**, no el plan original. Los
> archivos marcados **🧪 transitorio** son de debug/experimentación de sesiones puntuales: no son
> parte del pipeline final y se pueden borrar sin afectar la reproducibilidad del resultado.
> Los marcados **📦 desactualizado** describen el plan viejo y se conservan solo como referencia.

---

## Dataset utilizado

**`itsanmolgupta/mimic-cxr-dataset`** en Hugging Face
- 30.600 imágenes de tórax (512×512 px) con campos `image`, `findings` e `impression`
- Formato Parquet, ~800 MB, split único: `train`
- Sin necesidad de registro en PhysioNet

```python
from datasets import load_dataset
ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")
```

Se usa `impression` como target del fine-tuning (corta, estilo caption) y como referencia para
métricas. `findings` es demasiado larga para target. Ver nota al final.

---

## Árbol real del repositorio (lo versionado)

```
image-captioning/
│
├── CLAUDE.md                       ← Contexto del proyecto para Claude Code (leer primero)
├── PLAN.md                         ← 📦 desactualizado: plan original de notebooks 00–05
├── estructura.md                   ← Este archivo
├── HOJA_DE_RUTA.md                 ← Plan vigente: notebook comparativo + análisis del paper
├── NOTEBOOK_COMPARATIVO.md         ← Diseño detallado del notebook de explicabilidad (07)
├── soluciones_extra.md             ← Soluciones al mode collapse acotadas al scope
├── README.md
├── requirements.txt
├── prueba.ipynb                    ← 🧪 transitorio: scratchpad de pruebas sueltas
│
├── data/                           ← (versionado salvo hf_cache/) — índices y assets livianos
│   ├── splits/
│   │   ├── train_indices.json          ← ~10k índices de entrenamiento
│   │   ├── val_indices.json            ← índices de validación
│   │   ├── test_indices.json           ← índices de test
│   │   ├── test_sub_indices.json       ← subconjunto de test (600 imgs) para evaluaciones rápidas
│   │   ├── train_sub_indices.json      ← 🧪 subconjunto de smoke test
│   │   ├── val_sub_indices.json        ← 🧪 subconjunto de smoke test
│   │   └── _selected_unused.json       ← 🧪 descarte de una regeneración de splits
│   ├── selected_indices.json           ← 30 radiografías fijas (set A, usado por S1/captions)
│   ├── visual_test_indices.json        ← 25 radiografías fijas (set B, usado por los HEATMAPS)
│   ├── img_prueba/                     ← 🧪 imágenes sueltas de prueba (perro.jpg, prueba1.jpeg)
│   └── hf_cache/                       ← cache de HuggingFace (gitignoreado)
│
├── src/                            ← Lógica reutilizable (los notebooks llaman, no implementan)
│   ├── data/
│   │   ├── utils.py                    ← load_mimic_dataset y helpers de I/O
│   │   ├── dataset.py                  ← MimicCXRDataset (envuelve el HF dataset)
│   │   ├── dataloader.py               ← DataLoaders train/val/test desde los índices
│   │   └── split_generator.py          ← Genera los splits JSON (una sola vez, seed fija)
│   │
│   ├── models/
│   │   ├── blip_loader.py              ← Carga BLIP + processor desde disco o HF; sanity check
│   │   └── finetuner.py               ← Loop de fine-tuning (AdamW, scheduler, checkpoints)
│   │
│   ├── interpretability/
│   │   ├── cross_att_logits.py        ← PRINCIPAL: logits Q·K pre-softmax por palabra (⚠️ ver nota)
│   │   ├── cross_attention.py         ← Pesos post-softmax (referencia; mapas planos s/576 tokens)
│   │   └── gradcam.py                 ← Grad-CAM para ViT (reshape transform 24×24 custom)
│   │
│   └── visualization/
│       ├── heatmap.py                 ← overlay_heatmap, grillas por palabra, comparación
│       └── plots.py                   ← Plots auxiliares (curvas, distribuciones)
│
├── notebooks/                      ← Ver tabla §"notebooks" — algunos son transitorios
│
├── docs/                           ← Documentación técnica de apoyo (ver tabla)
│
├── archivos_ion/                   ← 🧪 Scripts y notebooks de los experimentos D1–D3/S1 (debug)
│
├── analisis/                       ← Hallazgos consolidados para el paper (01–04)
│
├── models/                         ← (gitignoreado) — ver §"Contenido gitignoreado"
└── outputs/                        ← (gitignoreado) — ver §"Contenido gitignoreado"
```

> **Módulos del plan original que NO existen** (los menciona `PLAN.md`/`CLAUDE.md` pero no están
> en disco): `src/data/preprocessing.py`, `src/interpretability/encoder_attention.py`,
> `src/visualization/comparison_grid.py`, y todo `src/metrics/` (`nlg_metrics.py`,
> `spatial_metrics.py`). Estos dos últimos hay que **crearlos** para el notebook comparativo
> (ver `NOTEBOOK_COMPARATIVO.md`).

---

## Descripción de componentes

### `src/data/`
| Archivo | Descripción |
|---|---|
| `utils.py` | `load_mimic_dataset(cache_dir)` y helpers de carga/guardado de índices JSON. |
| `dataset.py` | `MimicCXRDataset(Dataset)`: recibe el HF dataset + lista de índices, devuelve `pixel_values`/`input_ids` para BLIP. Target = `impression`. |
| `dataloader.py` | Crea los `DataLoader` train/val/test con `collate_fn` para texto de largo variable. |
| `split_generator.py` | Genera `data/splits/*.json` con seed fija. Se corre **una sola vez**. |

### `src/models/`
| Archivo | Descripción |
|---|---|
| `blip_loader.py` | `load_model_and_processor(model_dir, device)` — carga desde disco o HF. Incluye sanity check. |
| `finetuner.py` | Loop completo: AdamW `lr=1e-5`, scheduler con warmup, `train_one_epoch`/`eval_one_epoch`, guardado de `epoch_N/` y `best/`, early stopping. |

### `src/interpretability/`
> **⚠️ Estado:** `cross_att_logits.py` (logits Q·K) es la señal principal pero está **pendiente de
> validación del profesor**. Si la rechaza → degradar a `cross_attention.py` (post-softmax) +
> `gradcam.py`. Ver `docs/cross_att_logits_integracion.md`.

| Archivo | Descripción |
|---|---|
| `cross_att_logits.py` | Captura Q y K vía hooks, calcula `Q·K^T/√d` sin softmax → mapas distintos por palabra. `layer_idx=9` por defecto. |
| `cross_attention.py` | Pesos post-softmax. Limitación: ≈1/576 para todos los patches → mapas planos. Se mantiene como referencia y para la figura metodológica. |
| `gradcam.py` | Grad-CAM sobre el encoder ViT con `blip_vit_reshape_transform` (577→24×24, no 224×224). |

### `src/visualization/`
| Archivo | Descripción |
|---|---|
| `heatmap.py` | `overlay_heatmap`, `plot_word_heatmaps`, `save_heatmap_grid`, `plot_comparison_heatmaps`. Normalización **per-heatmap** (comparar patrón, no intensidad). |
| `plots.py` | Plots auxiliares (curvas de loss, distribuciones de captions, etc.). |

### `notebooks/`
| Notebook | Estado | Descripción |
|---|---|---|
| `00_exploracion_dataset.ipynb` | activo | Exploración del HF dataset, filtros, generación de splits. |
| `005_regenerar_splits.ipynb` | activo | Regeneración de splits con el dataset completo. |
| `01_calibracion_coco.ipynb` | activo | Parte 1: validar interpretabilidad en imágenes naturales (COCO). |
| `02_baseline_radiografias.ipynb` | activo | Parte 2: BLIP base sobre las radiografías. |
| `03_gcp_finetuner.ipynb` | activo | Fine-tuning real en GPU (GCP). `MAX_TRAIN_SAMPLES` define el output dir. |
| `06_analisis_captions.ipynb` | activo | Análisis de calidad de captions (overlap, diversidad). |
| `03_finetuning.ipynb` | 📦 viejo | Versión previa del finetuning, reemplazada por `03_gcp_finetuner.ipynb`. |
| `debug_cross_attention.ipynb` | 🧪 transitorio | Debug del post-softmax aplanado. |
| `prueba_finetuning.ipynb` | 🧪 transitorio | Pruebas sueltas de finetuning. |
| `00_analisis_resultados.md` | nota | Notas sueltas de resultados (no es notebook). |
| `07_explicabilidad_comparada.ipynb` | **a crear** | Pipeline comparativo (ver `NOTEBOOK_COMPARATIVO.md`). |

### `archivos_ion/` — 🧪 experimentos de mode collapse (debug)
Scripts y notebooks de las sesiones D1–D3 y S1. Útiles como referencia, **no** parte del pipeline
final. Generan outputs en `outputs/mode_collapse_debug/`, `outputs/decoding_sampling/`, etc.

| Archivo | Qué hace |
|---|---|
| `run_s1_decoding_experiment.py` | S1: 6 estrategias de decoding (greedy, temperatura, etc.). |
| `run_d1_checkpoint_collapse.py` | D1: collapse por checkpoint (epoch_1/2/3/best). |
| `run_d2_token_probe.py` | D2: distribuciones token a token (entropía, top1/top2). |
| `run_d3_heatmap_probe.py` | D3: heatmaps cross-att + Grad-CAM, base vs FT. **Base del notebook 07.** |
| `06–09_debug_*.ipynb` | Versiones notebook de los experimentos D/S. |
| `comandos.md`, `*_handoff_*.md`, `explicacion_*.md` | Comandos y explicaciones de cada experimento. |

### `analisis/` — hallazgos para el paper (vigente)
| Archivo | Contenido |
|---|---|
| `01_mode_collapse_s1_d2.md` | Diagnóstico y solución del collapse (T=1.2). |
| `02_captions_10k.md` | Calidad de captions del modelo 10k. |
| `03_hallazgos_paper.md` | Síntesis para el informe IEEE (figuras, referencias). |
| `04_analisis_pendiente.md` | Qué falta (post-softmax vs logits, D3, por token). |

### `docs/` — documentación técnica
`blip_modelo.md`, `interpretabilidad.md`, `finetuning_tecnico.md`, `guia_finetuning_gcp.md`,
`mejoras_mode_collapse.md`, `cross_att_logits_integracion.md`, `reporte_ion.md`.

### `../gcp/` (fuera de image-captioning) — infraestructura de la VM
`1_setup_gcloud.sh`, `2_create_vm.py`, `3_setup_vm.sh`, `4_guia_vm.md`, `run_temp_sampling.py`
(genera captions T=1.2/T=1.5). Necesario para reentrenar; no para el análisis local.

---

## Contenido gitignoreado — paths que NO están en el repo

> **Para quien clona el repo:** `models/` y `outputs/` están **gitignoreados** (pesos = cientos
> de MB, outputs = regenerables). Acá quedan documentados los paths y qué va en cada uno, para
> poder ubicarse sin tener los archivos. Los pesos se regeneran corriendo el fine-tuning
> (`notebooks/03_gcp_finetuner.ipynb`) o se copian aparte; los outputs se regeneran corriendo
> los notebooks. **No hace falta pushear nada de esto.**

Cada **checkpoint** (carpeta `best/`, `epoch_N/`) contiene los 6 archivos estándar de HuggingFace:
```
<checkpoint>/
├── config.json
├── generation_config.json
├── model.safetensors          ← los pesos (~900 MB, gitignoreado)
├── processor_config.json
├── tokenizer_config.json
└── tokenizer.json
```

#### `models/` (gitignoreado)
```
models/
├── blip_base/                      ← BLIP preentrenado (Salesforce/blip-image-captioning-base)
├── blip_finetuned_10k/             ← fine-tuning con 10.000 pares (el modelo principal)
│   ├── best/                       ← mejor val loss (= epoch_3). El que se usa en el análisis.
│   ├── epoch_1/
│   ├── epoch_2/
│   └── epoch_3/
└── blip_finetuned/best/            ← 🧪 LEGACY / vacío. No usar.

# El modelo de 5.000 pares NO está dentro de models/. Vive FUERA del repo en:
../output_5k/best/                  ← checkpoint 5k (ruta relativa a image-captioning/)
```

> **Nota de nombres:** el plan original hablaba de `models/blip_finetuned/best`; en la práctica
> los modelos quedaron en `models/blip_finetuned_10k/` (10k) y `../output_5k/best` (5k, externo).
> Quien reentrene puede consolidar el 5k en `models/blip_finetuned_5k/best` si quiere todo junto.

#### `outputs/` (gitignoreado)
```
outputs/
├── parte1_coco/                    ← Parte 1: captions + figura de calibración (COCO)
├── parte2_baseline/                ← Parte 2: captions + comparacion_<idx>.png (BLIP base, radiografías)
├── captions/                       ← captions base vs ft 5k (greedy) + distribución
├── captions_10k/captions/          ← captions base vs ft 10k (greedy)
├── prueba_mas_temp/                ← captions T=1.2 y T=1.5 (600 imgs) — JSON {idx, generated, reference}
├── decoding_sampling/              ← experimento S1 (6 estrategias de decoding)
│   ├── s1_selected_30/             ← S1 sobre selected_indices (csv resúmenes + png)
│   └── s1_10k/                     ← S1 sobre el modelo 10k
├── mode_collapse_debug/            ← experimento D2 (distribuciones token a token, csv/json)
├── finetuning/                     ← curva de loss del entrenamiento (loss_curve_5k.png)
├── dataset_checks/                 ← verificación de dataset/dataloader
└── notebook_comparativo/           ← (A GENERAR) pipeline de explicabilidad — ver NOTEBOOK_COMPARATIVO.md
    ├── captions/captions_bestof3.json
    ├── heatmaps/idx_<NNN>/<modelo>/{original,post_softmax_grid,gradcam_grid,qk_logits_grid,explanation}.png
    ├── arrays/idx_<NNN>__<modelo>__<metodo>.npz
    ├── metrics/{spatial_per_token,spatial_summary}.csv
    ├── figures_paper/{postsoftmax_vs_logits,comparativa_modelos_idx<NNN>}.png
    └── summary.csv
```

> **Índices del análisis visual:** los heatmaps usan `data/visual_test_indices.json` (25
> radiografías fijas, sí versionado). `selected_indices.json` (30 imgs) es un set distinto, sin
> overlap. Ver `HOJA_DE_RUTA.md` y `NOTEBOOK_COMPARATIVO.md` para el detalle del pipeline nuevo.

---

## Estado actual del trabajo (resumen)

| Fase | Estado |
|---|---|
| Pipeline de datos + splits | ✅ completo |
| Carga de modelo | ✅ completo |
| Fine-tuning 5k y 10k | ✅ hecho (collapse resuelto con T=1.2, ver `analisis/01`) |
| Interpretabilidad (3 métodos) | ✅ implementada (logits Q·K pendiente de profesor) |
| Notebook comparativo (07) | ⏳ a crear — ver `NOTEBOOK_COMPARATIVO.md` |
| Métricas (`src/metrics/`) | ⏳ a crear (espaciales + calidad de caption) |
| Redacción del paper | ⏳ pendiente — base en `analisis/03` |

---

## Nota sobre el campo de texto a usar

| Campo | Contenido | Largo | Uso |
|---|---|---|---|
| `findings` | Hallazgo por hallazgo | Largo (≤1.5k chars) | No usar como target — demasiado largo para BLIP |
| `impression` | Conclusión clínica | Corto (1–3 oraciones) | **Target del fine-tuning** y referencia de métricas |

---

## Resultados esperados (tres escenarios, todos válidos)

- **A** — Captions mejoran *y* heatmaps se vuelven médicamente coherentes → lenguaje y visión se adaptan juntos.
- **B** — Captions mejoran *pero* heatmaps no cambian → solo se adapta el lenguaje.
- **C** — Heatmaps cambian pero no hacia zonas relevantes → la atención se reorganiza sin coherencia clínica.

> Hallazgo provisional (ver `analisis/03`): **tipo A parcial** — el FT adapta visión y lenguaje;
> el collapse con greedy es un artefacto de decodificación, no un fallo de aprendizaje.
