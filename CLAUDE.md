# CLAUDE.md

Contexto del proyecto para Claude Code. Leer antes de tocar cualquier archivo.

---

## 1. Qué es este proyecto

Trabajo Práctico Final de **I308 Visión Artificial — Otoño 2026** (Universidad de San Andrés).

**Pregunta de investigación:** ¿Es suficiente el fine-tuning para que un modelo de captioning generalista (BLIP) no solo aprenda a *hablar* el lenguaje médico sino también a *mirar* las regiones clínicamente relevantes de una radiografía?

Para responderla, se comparan **mapas de cross-attention** y **Grad-CAM** sobre las mismas radiografías, **antes** y **después** del fine-tuning. La comparación es la contribución central del trabajo.

El proyecto tiene tres partes secuenciales:

| Parte | Qué se hace | Para qué |
|---|---|---|
| **1. Calibración** | BLIP base sobre 20–30 imágenes naturales de MS-COCO | Validar que las herramientas de interpretabilidad funcionan en un dominio donde el resultado correcto es obvio |
| **2. Baseline médico** | BLIP base sobre radiografías fijas | Documentar captions genéricos y heatmaps incoherentes médicamente — establece el punto de partida |
| **3. Post fine-tuning** | BLIP fine-tuneado sobre las *mismas* radiografías | Comparar antes/después y responder la pregunta de investigación |

Hay tres resultados posibles, **todos válidos**:
- **A:** captions mejoran *y* heatmaps se vuelven médicamente coherentes → lenguaje y visión se adaptan juntos
- **B:** captions mejoran pero heatmaps no cambian → solo se adapta el lenguaje
- **C:** heatmaps cambian pero no hacia zonas relevantes → la atención se reorganiza sin coherencia clínica

No hay una hipótesis a defender. El objetivo es **medir y reportar** lo que pasa.

> **Estado (2026-07-02):** fine-tuning **hecho** (5k y 10k pares); mode collapse con greedy
> **resuelto con sampling T=1.2** (ver §8.7). El **notebook comparativo de explicabilidad ya se
> corrió en la VM**: 25 radiografías × 3 modelos (base/ft5k/ft10k) × 3 métodos (post-softmax /
> QK-logits / Grad-CAM). Outputs en `outputs/notebook_comparativo/` (arrays `.npz`, captions,
> métricas, summary). Las **métricas de captions** ya están calculadas (recall médico ×3.75 con
> el fine-tuning; ver `IMPORTANTE/analisis_resultados_captions.md`). La etapa actual es el
> **análisis de resultados y armado de la presentación** (plan: `docs/plan_analisis_presentacion.md`,
> pendientes: `IMPORTANTE/pendientes_analisis_presentacion.md`). Planes previos: `HOJA_DE_RUTA.md`
> y `NOTEBOOK_COMPARATIVO.md`. Hallazgos: `analisis/01–04`. Estructura real: `estructura.md`.

---

## 2. Stack técnico

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10+ |
| DL framework | PyTorch |
| Modelo base | `Salesforce/blip-image-captioning-base` (HuggingFace) |
| Tokenizer / processor | `BlipProcessor` (HuggingFace) |
| Dataset | `itsanmolgupta/mimic-cxr-dataset` (HuggingFace Datasets, Parquet, ~800 MB) |
| Interpretabilidad — atención | `output_attentions=True` + extracción manual con forward hooks |
| Interpretabilidad — gradientes | `pytorch-grad-cam` con `vit_reshape_transform` |
| Métricas NLG | `pycocoevalcap` (CIDEr) + métricas propias en `src/metrics/caption_metrics.py` (BLEU, ROUGE-L, recall médico). METEOR descartado (JAR de Java incompatible) |
| Visualización | `matplotlib`, `PIL` |
| Hosting de cómputo | Local para desarrollo, Kaggle / Google Cloud (GPU L4 o T4) para fine-tuning |

**No usar:**
- TensorFlow / Keras (el proyecto es 100% PyTorch)
- Modelos de captioning distintos a BLIP base (no BLIP-2, no GIT, no otros)
- Datasets distintos al de HuggingFace para el fine-tuning (no descargar MIMIC-CXR de PhysioNet)

---

## 3. Estructura del repositorio (real, 2026-07-02)

```
image-captioning/
├── data/
│   ├── hf_cache/                ← cache de HuggingFace (gitignoreado)
│   ├── img_prueba/              ← imágenes locales de smoke test (prueba1.jpeg, perro.jpg)
│   ├── splits/                  ← train/val/test_indices.json (+ *_sub_indices de smoke test)
│   ├── selected_indices.json    ← 30 radiografías fijas (set A: captions/S1)
│   └── visual_test_indices.json ← 25 radiografías fijas (set B: HEATMAPS). Sin overlap con A.
├── models/                      ← (gitignoreado, ver §3.1)
│   ├── blip_base/
│   ├── blip_finetuned/          ← 🧪 legacy/vacío, no usar
│   └── blip_finetuned_10k/{best,epoch_1,2,3}
│   # el 5k vive FUERA del repo en ../output_5k/best
├── src/
│   ├── data/                    ← utils, dataset, dataloader, split_generator
│   ├── models/                  ← blip_loader, finetuner, generation (best-of-N, §8.8)
│   ├── interpretability/        ← compare (orquestador), cross_att_logits, cross_attention,
│   │                              gradcam, token_filter (§8.8)
│   ├── visualization/           ← heatmap, plots
│   └── metrics/                 ← spatial_metrics (heatmaps) + caption_metrics (calidad captions)
├── scripts/                     ← runners: run_single_image_compare, plot_single_image_compare,
│                                  run_caption_metrics, reorganize_arrays, smoke_*
├── notebooks/                   ← ver sección 6 (algunos son transitorios)
├── archivos_ion/                ← 🧪 scripts/notebooks de experimentos D1–D3/S1 (debug)
├── analisis/                    ← 01–04: hallazgos consolidados para el paper
├── IMPORTANTE/                  ← análisis de resultados y pendientes para la presentación
├── docs/                        ← documentación técnica de apoyo
├── outputs/                     ← figuras y CSVs (gitignoreado)
│   └── notebook_comparativo/    ← producto del notebook 07: arrays/, captions/, metrics/, summary.csv
├── HOJA_DE_RUTA.md              ← plan (previo al notebook comparativo)
├── NOTEBOOK_COMPARATIVO.md      ← diseño del notebook 07
├── requirements.txt
└── .gitignore
```

**Módulos del plan original que NO existen** (no asumir que están): `src/data/preprocessing.py`,
`src/interpretability/encoder_attention.py`, `src/visualization/comparison_grid.py`.
`src/metrics/` **ya existe** (`spatial_metrics.py` + `caption_metrics.py`).

Ver **`estructura.md`** para el árbol completo, qué es transitorio y los paths gitignoreados.

### 3.1 Checkpoints (gitignoreados)
- `models/blip_base/` — BLIP preentrenado.
- `models/blip_finetuned_10k/{best,epoch_1,2,3}` — modelo principal (best = epoch_3).
- `../output_5k/best` — modelo 5k, **fuera del repo** (ruta relativa a `image-captioning/`).
- `models/blip_finetuned/best` — 🧪 legacy/vacío, no usar.
Cada checkpoint son los 6 archivos HF estándar (`model.safetensors` ~900 MB).

---

## 4. Dataset — detalles críticos

```python
from datasets import load_dataset
ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")
# ds["train"] tiene 30.600 filas con columnas: image, findings, impression
```

**Campos:**
- `image`: PIL Image en escala de grises, 512×512 px (a redimensionar a 384×384 para BLIP — el processor lo hace automáticamente)
- `findings`: descripción detallada hallazgo por hallazgo (hasta ~1.5k chars). No usar como target — es demasiado larga para el estilo de caption corto que BLIP genera.
- `impression`: conclusión clínica resumida (1–3 oraciones). **Usar como target del fine-tuning** y como referencia para evaluación con BLEU/CIDEr/METEOR. Ver sección 8.5.

**Gotchas del dataset:**
- Algunas filas tienen `impression` vacío o `None` — filtrar antes de splitear
- Las imágenes son grayscale; BLIP espera RGB → convertir con `image.convert("RGB")` antes de pasar al processor
- No hay un split nativo train/val/test — se genera con `src/data/split_generator.py` con seed fijo (42) para reproducibilidad
- El cache de HF puede ocupar varios GB en disco — apuntar `HF_DATASETS_CACHE` a `data/hf_cache/` si hace falta controlarlo

**Splits (ya generados):**
- `train_indices.json` / `val_indices.json` / `test_indices.json` — splits principales.
- `test_sub_indices.json` → subconjunto de 600 imgs para evaluaciones rápidas.
- `selected_indices.json` → **30** radiografías fijas (set A: captions/S1).
- `visual_test_indices.json` → **25** radiografías fijas (set B: **heatmaps**). **No comparten
  índices con A** — usar el set correcto según la tarea (los heatmaps van con `visual_test`).

> El fine-tuning se hizo con **5k** y **10k** pares (no 15k). El salto 5k→10k fue marginal en
> calidad (ver `analisis/02`).

---

## 5. Modelo BLIP — detalles arquitectónicos relevantes

**Identificador HF:** `Salesforce/blip-image-captioning-base`

**Componentes que importan para interpretabilidad:**
- Encoder visual: **ViT-Base**, patches de 16×16, input 384×384 → **576 patch tokens** (24×24 grid) de 768 dim
- Decoder de texto: Transformer con **cross-attention** sobre los 576 patch tokens
- Para extraer atención del decoder al encoder: el shape del tensor de cross-attention es `(batch, n_heads, T_caption, 576)`

**Cómo cargar:**
```python
from transformers import BlipForConditionalGeneration, BlipProcessor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

**Cómo generar captions con atención:**
```python
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs, output_attentions=True, return_dict_in_generate=True)
caption = processor.decode(out.sequences[0], skip_special_tokens=True)
# out.cross_attentions es una tupla (tuple por step, tuple por capa)
```

---

## 6. Notebooks — orden y rol (real)

| Notebook | Estado | Rol | Dónde |
|---|---|---|---|
| `00_exploracion_dataset.ipynb` | activo | Inspeccionar dataset, filtrar inválidas, generar splits | Local |
| `005_regenerar_splits.ipynb` | activo | Regenerar splits con dataset completo | Local |
| `01_calibracion_coco.ipynb` | activo | **Parte 1:** validar interpretabilidad en COCO | Local |
| `02_baseline_radiografias.ipynb` | activo | **Parte 2:** BLIP base sobre radiografías | Local |
| `03_gcp_finetuner.ipynb` | activo | Fine-tuning real (`MAX_TRAIN_SAMPLES` define el output dir) | **GPU (GCP)** |
| `06_analisis_captions.ipynb` | activo | Análisis de calidad de captions | Local |
| `07_explicabilidad_comparada_v0.ipynb` | **hecho (corrido en VM)** | Pipeline comparativo 3 métodos × 3 modelos × 25 imgs. Notebook delgado que orquesta `scripts/run_single_image_compare.py`. Cambiar `RUN_MODE`/`MODELS`/`DEVICE` para el run real | **GPU** |
| `09_single_image_compare_minimo.ipynb` | activo | Smoke test local del pipeline comparativo (1 imagen, base, CPU) | Local |
| `03_finetuning.ipynb` | 📦 viejo | Reemplazado por `03_gcp_finetuner.ipynb` | — |
| `debug_cross_attention.ipynb`, `prueba_finetuning.ipynb` | 🧪 transitorio | Scratchpads de debug | — |

Los experimentos de mode collapse (D1–D3, S1) viven en `archivos_ion/` como scripts `run_*.py`
+ notebooks `0[6-9]_debug_*.ipynb`. Son debug, no pipeline final.

**Regla crítica:** los notebooks deben ser **delgados**. Toda la lógica reutilizable va en `src/`. Los notebooks llaman, no implementan. Si hay que escribir más de ~20 líneas de lógica en un notebook, es señal de que debería ir a un módulo en `src/`.

---

## 7. Convenciones de código

- **Imports absolutos desde `src/`**: `from src.interpretability.gradcam import compute_gradcam`
- **Type hints** en funciones públicas (no obligatorio en helpers privados de 3 líneas)
- **Docstrings estilo Google** en funciones que se usan desde notebooks
- **Nombres en inglés** para código (`compute_gradcam`, `extract_cross_attention`), **textos en español** para informe y comentarios largos
- **Seeds fijas** (`seed=42`) en cualquier cosa que involucre aleatoriedad: splits, shuffling de DataLoader, sampling
- **No usar `print` para debug en `src/`**; usar `logging` con `logger = logging.getLogger(__name__)`. En notebooks `print` está bien.
- **Guardar figuras con DPI alto** (`dpi=200`) — terminan en el poster A0 y se ven los pixels si son bajas
- **Paths como `pathlib.Path`**, no strings concatenados

**Estructura típica de un script en `src/`:**

```python
"""Descripción de una línea de qué hace este módulo."""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def funcion_publica(arg: TipoX) -> TipoY:
    """Una línea de qué hace.

    Args:
        arg: ...

    Returns:
        ...
    """
    ...
```

---

## 8. Trampas conocidas e implementación

### 8.1. Grad-CAM en ViT requiere `reshape_transform`

ViT no tiene mapas de activación 2D nativos como una CNN. El truco estándar es:

```python
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
from pytorch_grad_cam import GradCAM

cam = GradCAM(
    model=model.vision_model,           # solo el encoder ViT de BLIP
    target_layers=[model.vision_model.encoder.layers[-1].layer_norm1],
    reshape_transform=vit_reshape_transform,
)
```

El `reshape_transform` por defecto asume **197 tokens** (CLS + 196 patches para 224×224). **BLIP usa 384×384 → 577 tokens (CLS + 576 patches, grilla 24×24)**. Hay que usar un transform custom que descarte el CLS y haga reshape a 24×24 en lugar de 14×14.

### 8.2. Cross-attention por palabra

**En transformers 5.x el enfoque cambió.** `generate(output_attentions=True)` ya no expone `out.cross_attentions`. La solución usa `register_forward_hook` sobre la capa de cross-attention del decoder.

**Limitación conocida — pesos post-softmax:** el softmax sobre 576 patch tokens aplana las diferencias semánticas entre palabras hasta hacerlas visualmente indistinguibles. Los pesos post-softmax son ≈ 1/576 para todos los patches, independientemente del token generado. Esto fue diagnosticado exhaustivamente y no tiene fix dentro del mecanismo de atención estándar.

**Solución implementada — logits Q·K (`cross_att_logits.py`):** capturar los vectores Q y K proyectados via hooks sobre `crossattention.self.query` y `crossattention.self.key`, y calcular `Q·K^T / √d` sin aplicar softmax. Produce mapas distintos y coherentes por palabra.

```python
from src.interpretability.cross_att_logits import extract_cross_att_logits

results = extract_cross_att_logits(model, processor, inputs, num_batch=1, layer_idx=9)
# results[0] = {"caption": str, "maps": [(palabra, array(24,24)), ...]}
```

**⚠️ Pendiente de validación del profesor:** se está esperando confirmación de que los logits Q·K son aceptables como señal de interpretabilidad para el informe. Si no lo son, la alternativa es **Grad-CAM + cross-attention post-softmax**. Ver `docs/cross_att_logits_integracion.md`.

**Enfoque legacy (transformers ≤ 4.x):** documentado en `cross_attention.py` al final del archivo.

> **Nota:** `encoder_attention.py` (auto-atención CLS→patches del encoder) figura en docs viejos
> pero **no existe** en `src/`. Si se necesita, hay que crearlo.

### 8.3. Las radiografías de análisis son fijas

`data/selected_indices.json` (30, captions) y `data/visual_test_indices.json` (25, heatmaps) se
generan UNA VEZ y nunca se modifican. Si se regeneran, la comparación antes/después pierde
validez. **Los heatmaps usan `visual_test_indices.json`** — no confundir con `selected_indices`.

### 8.4. El fine-tuning consume memoria

BLIP-base + batch 8 + radiografías 384×384 ronda ~10–12 GB de VRAM. En una T4 (16 GB) entra. En GPUs locales más chicas usar gradient accumulation con batch efectivo 8.

### 8.5. `findings` vs `impression` — no mezclar

- Fine-tuning: usar `impression` como target (`findings` es demasiado largo y descriptivo — no matchea con el estilo de caption corto que BLIP genera)
- Evaluación con BLEU/CIDEr/METEOR: usar `impression` como referencia
- Documentar esta elección en el informe

### 8.6. PIL Image en grayscale → RGB

```python
image_rgb = image.convert("RGB")  # antes de pasar al processor
```

Olvidar esto da un error críptico de shape en BLIP.

### 8.7. Mode collapse con greedy → usar T=1.2 (NO greedy para el análisis)

Tras el fine-tuning, **greedy decoding colapsa**: 65–80% de las imágenes reciben la misma
caption ("no acute cardiopulmonary process"). Diagnóstico (D2): es un **sesgo estadístico de
MIMIC-CXR amplificado por greedy**, no un fallo de aprendizaje — el modelo tiene vocabulario
médico diverso en sus pesos.

**Solución:** sampling con `do_sample=True, temperature=1.2, top_p=0.95`. Rompe el collapse
(unique_ratio ~0.93), 83% vocabulario médico, 15% overlap con la referencia. Detalle en
`analisis/01` y `analisis/02`.

```python
out = model.generate(**inputs, do_sample=True, temperature=1.2, top_p=0.95, max_new_tokens=40)
```

- **Nunca usar greedy** para el análisis final (colapsa).
- **No subir T más allá de 1.2** (T=1.5 produce incoherencias en ~38/600 casos).
- El sampling es estocástico → fijar seed y/o usar **best-of-N** (ver §8.8 y `NOTEBOOK_COMPARATIVO.md`).

### 8.8. Pipeline comparativo con `generated_ids` (teacher forcing)

Cambio arquitectónico central del notebook comparativo. **Antes** cada método de
interpretabilidad llamaba internamente a `model.generate()`, así que con sampling
estocástico cada uno explicaba una caption distinta → comparación inválida.

**Ahora** la caption se genera **una sola vez** y los tres métodos explican esa misma
secuencia vía forward con teacher forcing (les pasás `generated_ids`, no regeneran):

```python
from src.models.generation import generate_caption_best_of_n
from src.interpretability.compare import extract_all_methods

# 1. genera la caption una vez (best-of-N, T=1.2)
result = generate_caption_best_of_n(model, processor, image, seeds=[42, 43, 44])
generated_ids = result["generated_ids"]

# 2. los 3 métodos explican esa misma secuencia
maps = extract_all_methods(model, processor, image, generated_ids, layer_idx=9)
# maps = {"post_softmax": {...}, "qk_logits": {...}, "gradcam": {...}}
```

Módulos involucrados:
- `src/models/generation.py` — `generate_caption_best_of_n`: samplea N captions y elige
  la mejor por score (riqueza médica, sin repetición/genericidad).
- `src/interpretability/compare.py` — `extract_all_methods`: orquesta los 3 métodos sobre
  `generated_ids` y alinea los mapas token a token.
- `src/interpretability/token_filter.py` — descarta stopwords/puntuación; `MEDICAL` marca
  vocabulario clínico. Se usa tanto en interpretabilidad como en `caption_metrics.py`.
- Los 3 extractores (`cross_attention`, `cross_att_logits`, `gradcam`) aceptan
  `generated_ids`. Grad-CAM hace N forwards+backwards (uno por token); los de atención
  hacen un solo forward con hooks. Detalle: `docs/teacher_forcing_interpretabilidad.md`.

Comparación entre métodos: `src/metrics/spatial_metrics.py` (Pearson, coseno, MSE, top-k
IoU). Ver `docs/metricas_espaciales.md`.

### 8.9. Métricas de calidad de captions

`src/metrics/caption_metrics.py` evalúa la caption generada vs la `impression` de referencia.
Dos capas: sin dependencias (BLEU, ROUGE-L, recall médico, categorización clínica) y opcional
(CIDEr vía `pycocoevalcap`, requiere `conda activate tp_vision`). La métrica más informativa
en este dominio es el **recall médico** (fracción del vocabulario clínico de la referencia
capturado), no BLEU/ROUGE que subestiman la mejora. Detalle: `docs/METRICAS_CAPTIONS.md`;
resultados: `IMPORTANTE/analisis_resultados_captions.md`.

---

## 9. Comandos frecuentes

```bash
# Setup inicial
pip install -r requirements.txt

# Generar splits (una sola vez, ejecutar desde la raíz del repo)
python -m src.data.split_generator --seed 42

# Smoke test de carga del modelo
python -m src.models.blip_loader --sanity-check

# Fine-tuning: se hace desde notebooks/03_gcp_finetuner.ipynb en GPU (GCP).
# MAX_TRAIN_SAMPLES define el nº de pares y el output dir (5k / 10k).
# Infra de la VM: ../gcp/ (1_setup_gcloud.sh, 2_create_vm.py, 3_setup_vm.sh, 4_guia_vm.md)

# Pipeline comparativo sobre una imagen (genera caption + 3 métodos + métricas espaciales):
python scripts/run_single_image_compare.py \
    --image-path data/img_prueba/prueba1.jpeg \
    --model-dir models/blip_base \
    --output-dir outputs/single_image_compare/prueba1 \
    --device cuda --seeds 42 43 44 --max-new-tokens 40

# Reorganizar los .npz del notebook comparativo por imagen (idx_NNN/...):
python scripts/reorganize_arrays.py

# Métricas de calidad de captions (requiere el entorno conda tp_vision para CIDEr):
conda activate tp_vision
python scripts/run_caption_metrics.py --coco
# → outputs/notebook_comparativo/metrics/caption_metrics_{per_item,summary}.csv

# Experimento D3 (heatmaps base vs FT) — antecedente del notebook 07:
python archivos_ion/run_d3_heatmap_probe.py \
    --base-model-dir models/blip_base \
    --ft-model-dir models/blip_finetuned_10k/best \
    --indices data/visual_test_indices.json \
    --max-images 25 --device cuda --output-dir outputs/d3_full_10k
```

---

## 10. Restricciones del entorno académico

Este es un trabajo evaluado bajo el código de honor de UdeSA. Importante:

- El proyecto se entrega como informe IEEE de 8 páginas + póster A0 + código fuente
- Está permitido usar implementaciones existentes (HuggingFace BLIP, `pytorch-grad-cam`) **citándolas**
- Se requiere **al menos una modificación o aporte propio**: el aporte de este proyecto es la **comparación sistemática cross-attention vs Grad-CAM antes/después del fine-tuning sobre las mismas imágenes** — no copiar este análisis de ningún paper existente
- Todas las funciones que vengan de papers o repos externos deben tener un comentario `# Source: <cita>` en el código

---

## 11. Cuándo preguntar antes de actuar

Claude Code debe **preguntar al usuario** antes de:

- Borrar archivos en `data/`, `models/`, o `outputs/`
- Modificar `selected_indices.json`, `visual_test_indices.json` o cualquier archivo en `data/splits/` después de su generación inicial
- Cambiar la versión del modelo BLIP (no usar BLIP-2 sin discutir)
- **Reentrenar** los modelos 5k o 10k (ya están y funcionan — pisarlos pierde el análisis hecho)
- Cambiar hiperparámetros centrales del fine-tuning si ya hay checkpoint (`models/blip_finetuned_10k/best`, `../output_5k/best`)
- Reescribir grandes porciones de los notebooks ya ejecutados

Para todo lo demás (escribir funciones nuevas en `src/`, agregar tests, refactorizar internals, documentar), avanzar sin consultar.

---

## 12. Referencias clave

| Tema | Paper | Link |
|---|---|---|
| Modelo base | Li et al., BLIP, ICML 2022 | https://arxiv.org/abs/2201.12086 |
| Encoder visual | Dosovitskiy et al., ViT, ICLR 2021 | https://arxiv.org/abs/2010.11929 |
| Grad-CAM | Selvaraju et al., ICCV 2017 | https://arxiv.org/abs/1610.02391 |
| Grad-CAM en Transformers | Chefer et al., ICCV 2021 | https://arxiv.org/abs/2103.15679 |
| Crítica a la atención como explicación | Jain & Wallace, NAACL 2019 | https://arxiv.org/abs/1902.10186 |
| Attention Rollout | Abnar & Zuidema, ACL 2020 | https://arxiv.org/abs/2005.00928 |
| Fine-tuning de captioning médico | Nicolson et al., AIIM 2023 | https://arxiv.org/abs/2201.09405 |