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
| **2. Baseline médico** | BLIP base sobre 20–30 radiografías fijas | Documentar captions genéricos y heatmaps incoherentes médicamente — establece el punto de partida |
| **3. Post fine-tuning** | BLIP fine-tuneado sobre las *mismas* 20–30 radiografías | Comparar antes/después y responder la pregunta de investigación |

Hay tres resultados posibles, **todos válidos**:
- **A:** captions mejoran *y* heatmaps se vuelven médicamente coherentes → lenguaje y visión se adaptan juntos
- **B:** captions mejoran pero heatmaps no cambian → solo se adapta el lenguaje
- **C:** heatmaps cambian pero no hacia zonas relevantes → la atención se reorganiza sin coherencia clínica

No hay una hipótesis a defender. El objetivo es **medir y reportar** lo que pasa.

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
| Métricas NLG | `pycocoevalcap` (BLEU, CIDEr, METEOR) |
| Visualización | `matplotlib`, `PIL` |
| Hosting de cómputo | Local para desarrollo, Kaggle / Google Cloud (GPU L4 o T4) para fine-tuning |

**No usar:**
- TensorFlow / Keras (el proyecto es 100% PyTorch)
- Modelos de captioning distintos a BLIP base (no BLIP-2, no GIT, no otros)
- Datasets distintos al de HuggingFace para el fine-tuning (no descargar MIMIC-CXR de PhysioNet)

---

## 3. Estructura del repositorio

```
blip-interpretabilidad/
├── data/
│   ├── hf_cache/                ← cache de HuggingFace (NO versionar)
│   ├── splits/                  ← train/val/test_indices.json
│   ├── selected_indices.json    ← 20–30 índices fijos para Partes 2 y 3
│   └── coco/selected/           ← 20–30 imágenes de COCO para Parte 1
├── models/
│   ├── blip_base/               ← cache del modelo HF (NO versionar)
│   └── blip_finetuned/
│       ├── epoch_N/
│       └── best/                ← checkpoint con mejor val loss
├── src/
│   ├── data/                    ← dataset, dataloader, preprocessing, split_generator
│   ├── models/                  ← blip_loader, finetuner
│   ├── interpretability/        ← cross_att_logits (principal), cross_attention, encoder_attention, gradcam
│   ├── visualization/           ← heatmap, comparison_grid
│   └── metrics/                 ← nlg_metrics, spatial_metrics
├── notebooks/                   ← 00 a 05, ver sección 6
├── outputs/                     ← figuras y CSVs (NO versionar)
├── informe/
├── poster/
├── requirements.txt
└── .gitignore
```

Ver `estructura_proyecto.md` para descripción detallada de cada archivo.

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

**Splits a generar (una sola vez):**
- `train_indices.json` → ~15.000 índices
- `val_indices.json` → ~1.500 índices
- `test_indices.json` → ~1.000 índices
- `selected_indices.json` → 20–30 índices del split de test (las imágenes "estrella" del análisis)

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

## 6. Notebooks — orden y rol

| # | Notebook | Rol | Dónde correr |
|---|---|---|---|
| 00 | `exploracion_dataset.ipynb` | Inspeccionar dataset, filtrar filas inválidas, generar splits y `selected_indices.json` | Local |
| 01 | `calibracion_coco.ipynb` | **Parte 1:** validar interpretabilidad en COCO | Local |
| 02 | `baseline_radiografias.ipynb` | **Parte 2:** BLIP base sobre radiografías seleccionadas | Local |
| 03 | `finetuning.ipynb` | Fine-tuning de BLIP sobre 10k pares | **GPU (Kaggle / GCP)** |
| 04 | `analisis_postft.ipynb` | **Parte 3:** mismo pipeline post fine-tuning, sobre las mismas imágenes | Local |
| 05 | `resultados_y_metricas.ipynb` | Métricas, tablas, figura central 3×3 | Local |

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

**⚠️ Pendiente de validación del profesor:** se está esperando confirmación de que los logits Q·K son aceptables como señal de interpretabilidad para el informe. Si no lo son, la alternativa es Grad-CAM + encoder self-attention. Ver `docs/respuesta_profesor_cross_attention.txt`.

**Encoder self-attention (`encoder_attention.py`):** alternativa no word-specific. Extrae la atención CLS→patches del encoder ViT, útil para comparar foco visual global antes/después del fine-tuning.

**Enfoque legacy (transformers ≤ 4.x):** documentado en `cross_attention.py` al final del archivo.

### 8.3. Las 20–30 radiografías son fijas

`data/selected_indices.json` se genera UNA VEZ y nunca se modifica. Las Partes 2 y 3 usan exactamente los mismos índices. Si se regenera, la comparación antes/después pierde validez.

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

---

## 9. Comandos frecuentes

```bash
# Setup inicial
pip install -r requirements.txt

# Generar splits (una sola vez, ejecutar desde la raíz del repo)
python -m src.data.split_generator --seed 42

# Smoke test de carga del modelo
python -m src.models.blip_loader --sanity-check

# Lanzar fine-tuning desde CLI (alternativa al notebook 03)
python -m src.models.finetuner \
    --train-indices data/splits/train_indices.json \
    --val-indices data/splits/val_indices.json \
    --epochs 3 \
    --batch-size 8 \
    --lr 1e-5 \
    --output-dir models/blip_finetuned/

# Correr métricas NLG sobre el test set
python -m src.metrics.nlg_metrics \
    --captions-base outputs/parte2_baseline/captions.json \
    --captions-ft outputs/parte3_finetuned/captions.json \
    --references data/splits/test_indices.json
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
- Modificar `selected_indices.json` o cualquier archivo en `data/splits/` después de su generación inicial
- Cambiar la versión del modelo BLIP (no usar BLIP-2 sin discutir)
- Cambiar hiperparámetros centrales del fine-tuning (lr, epochs, batch size) si ya hay un checkpoint en `models/blip_finetuned/best/`
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