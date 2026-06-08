# Estructura del Proyecto — BLIP Fine-tuning e Interpretabilidad Visual
**I308 Visión Artificial — Otoño 2026**  
**Autor:** Felipe  
**Tema:** Fine-tuning de BLIP sobre radiografías y análisis de cross-attention + Grad-CAM

---

## Dataset utilizado

**`itsanmolgupta/mimic-cxr-dataset`** en Hugging Face  
- 30.600 imágenes de tórax (512×512 px) con campos `image`, `findings` e `impression`
- Formato Parquet, ~800 MB, split único: `train`
- Sin necesidad de registro en PhysioNet
- Descarga con un comando:

```python
from datasets import load_dataset
ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")
```

Los campos `findings` (descripción detallada) e `impression` (conclusión resumida) se pueden usar
indistintamente como texto de referencia. Se recomienda usar `findings` para el fine-tuning
y `impression` para evaluación cualitativa, dado que es más concisa y comparable a lo que BLIP generaría.

---

## Árbol de directorios

```
blip-interpretabilidad/
│
├── README.md
│
├── data/
│   ├── hf_cache/                      ← Cache local de HuggingFace (Parquet descargado)
│   │   └── .gitkeep                   ← No se sube al repo (demasiado pesado)
│   │
│   ├── splits/
│   │   ├── train_indices.json         ← Índices del split de entrenamiento (10.000 muestras)
│   │   ├── val_indices.json           ← Índices del split de validación (1.000 muestras)
│   │   └── test_indices.json          ← Índices del split de test (1.000 muestras)
│   │
│   ├── selected_indices.json          ← Índices de las 20–30 radiografías fijas para análisis
│   │
│   └── coco/
│       ├── raw/                       ← Imágenes descargadas de MS-COCO
│       └── selected/                  ← 20–30 imágenes seleccionadas para calibración (Parte 1)
│
├── models/
│   ├── blip_base/                     ← Pesos BLIP preentrenado (Salesforce/blip-image-captioning-base)
│   │   └── .gitkeep                   ← No se sube al repo
│   │
│   └── blip_finetuned/
│       ├── epoch_1/
│       ├── epoch_2/
│       ├── epoch_3/
│       └── best/                      ← Checkpoint con mejor loss de validación
│
├── src/
│   ├── data/
│   │   ├── dataset.py                 ← Clase MimicCXRDataset que envuelve el HF dataset
│   │   ├── dataloader.py              ← DataLoaders de train/val/test usando los índices en splits/
│   │   ├── preprocessing.py           ← Transforms de imagen y tokenización de findings/impression
│   │   └── split_generator.py         ← Script para generar y guardar los splits (una sola vez)
│   │
│   ├── models/
│   │   ├── blip_loader.py             ← Carga BlipForConditionalGeneration desde HuggingFace
│   │   └── finetuner.py               ← Loop de entrenamiento, AdamW lr=1e-5, checkpoints
│   │
│   ├── interpretability/
│   │   ├── cross_attention.py         ← Extracción de mapas de cross-attention via forward hooks
│   │   ├── gradcam.py                 ← Grad-CAM para ViT usando pytorch-grad-cam + vit_reshape_transform
│   │   └── rollout.py                 ← (Opcional) Attention Rollout para combinar capas
│   │
│   ├── visualization/
│   │   ├── heatmap.py                 ← Superposición de heatmap sobre imagen
│   │   └── comparison_grid.py         ← Figura central 3×3 (la "figura de nueve celdas")
│   │
│   └── metrics/
│       ├── nlg_metrics.py             ← BLEU-4, CIDEr, METEOR
│       └── spatial_metrics.py         ← Correlación espacial cross-attention vs Grad-CAM
│
├── notebooks/
│   ├── 00_exploracion_dataset.ipynb   ← Exploración del HF dataset: distribución, ejemplos, campos
│   ├── 01_calibracion_coco.ipynb      ← Parte 1: validar herramientas en imágenes naturales
│   ├── 02_baseline_radiografias.ipynb ← Parte 2: BLIP base sobre las radiografías seleccionadas
│   ├── 03_finetuning.ipynb            ← Fine-tuning de BLIP sobre los 10k pares de MIMIC-CXR
│   ├── 04_analisis_postft.ipynb       ← Parte 3: análisis post fine-tuning sobre las mismas imágenes
│   └── 05_resultados_y_metricas.ipynb ← Métricas cuantitativas, tablas y figuras para informe/poster
│
├── outputs/
│   ├── parte1_coco/                   ← Figuras de calibración (imágenes naturales)
│   │   └── {img_id}_{palabra}/
│   │
│   ├── parte2_baseline/               ← Heatmaps antes del fine-tuning (radiografías)
│   │   └── {idx}_{palabra}/
│   │
│   ├── parte3_finetuned/              ← Heatmaps después del fine-tuning (mismas radiografías)
│   │   └── {idx}_{palabra}/
│   │
│   ├── figura_central/
│   │   └── grid_9celdas.png           ← La figura 3×3 que resume el trabajo completo
│   │
│   └── metricas/
│       ├── nlg_antes_despues.csv      ← BLEU-4, CIDEr, METEOR antes y después
│       └── correlacion_espacial.csv   ← Correlación cross-attention vs Grad-CAM antes y después
│
├── informe/
│   ├── informe.pdf
│   └── informe.tex                    ← (Opcional) Fuente LaTeX
│
├── poster/
│   ├── poster.pdf
│   └── poster_source/
│
├── requirements.txt
└── .gitignore
```

---

## Descripción de cada componente

### `data/`

| Archivo / Carpeta | Descripción |
|---|---|
| `hf_cache/` | Directorio donde HuggingFace guarda el Parquet descargado (`~/.cache/huggingface` por defecto, o se redirige aquí con `HF_DATASETS_CACHE`). No se versiona. |
| `splits/train_indices.json` | Lista de 10.000 índices enteros del split `ds["train"]` reservados para fine-tuning. Se genera una sola vez con `split_generator.py` para garantizar reproducibilidad. |
| `splits/val_indices.json` | 1.000 índices para validación durante el fine-tuning. |
| `splits/test_indices.json` | 1.000 índices para evaluación final de métricas NLG. |
| `selected_indices.json` | Los 20–30 índices de las radiografías que se usan en las Partes 2 y 3. **Son fijos y no cambian.** Permiten la comparación antes/después sobre exactamente las mismas imágenes. |
| `coco/selected/` | 20–30 imágenes de MS-COCO donde BLIP genera captions correctos y verificables visualmente (p.ej. `"a dog running in a park"`). Se usan solo en la Parte 1. |

---

### `src/data/`

| Archivo | Descripción |
|---|---|
| `split_generator.py` | Script que carga `ds["train"]` (30.6k muestras), filtra filas con `findings` vacío o `None`, y genera los tres archivos JSON en `data/splits/` con índices aleatorios pero con semilla fija para reproducibilidad. Se ejecuta **una sola vez**. |
| `dataset.py` | Clase `MimicCXRDataset(Dataset)` de PyTorch. Recibe el HF dataset y una lista de índices. Para cada ítem devuelve `{"pixel_values": tensor, "input_ids": tensor}` listo para BLIP. El texto de entrenamiento es `findings`; opcionalmente se puede configurar para usar `impression`. |
| `dataloader.py` | Crea los tres `DataLoader` (train/val/test) usando `MimicCXRDataset` con los índices de `splits/`. Configura `batch_size`, `num_workers`, `pin_memory` y `collate_fn` para manejar textos de longitud variable. |
| `preprocessing.py` | Transformaciones de imagen: resize a 224×224, normalización con la media/std de BLIP. Tokenización: usa `BlipProcessor` para convertir el texto en `input_ids` con padding y truncado a 512 tokens. |

---

### `src/models/`

| Archivo | Descripción |
|---|---|
| `blip_loader.py` | Carga `BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")` y `BlipProcessor`. Incluye una función `sanity_check(image)` que genera un caption sobre una imagen de prueba para verificar que el modelo cargó correctamente antes de cualquier modificación. |
| `finetuner.py` | Loop de entrenamiento completo: optimizador AdamW con `lr=1e-5`, scheduler lineal con warmup, cálculo del loss de generación por época, guardado de checkpoints en `models/blip_finetuned/epoch_N/`, early stopping si el val loss no mejora en 2 épocas consecutivas. Adapta tanto el encoder visual como el decoder de texto. |

---

### `src/interpretability/`

| Archivo | Descripción |
|---|---|
| `cross_attention.py` | Registra un `register_forward_hook` sobre la capa de cross-attention de la última capa del decoder de BLIP (o usa `output_attentions=True` en `generate()`). Extrae tensores de shape `(batch, n_heads, T_caption, 196)`, promedia sobre heads, hace reshape a 14×14 y upscale bilineal a 224×224. Devuelve un dict `{palabra: heatmap_224x224}`. |
| `gradcam.py` | Usa `pytorch-grad-cam` con `GradCAM(model, target_layers=[...])` y `vit_reshape_transform` aplicado al encoder ViT. Para cada palabra del caption, define un `ClassifierOutputTarget` custom que apunta al logit de ese token y corre `cam(input_tensor, targets=[...])` para obtener el mapa 224×224. |
| `rollout.py` | *(Opcional)* Implementa Attention Rollout (Abnar & Zuidema, ACL 2020) combinando los mapas de atención de todas las capas del decoder mediante multiplicación matricial recursiva. Útil si los heatmaps de la última capa resultan difusos. |

---

### `src/visualization/`

| Archivo | Descripción |
|---|---|
| `heatmap.py` | Función `overlay_heatmap(image, heatmap, alpha=0.5)` que normaliza el heatmap a [0,1], aplica colormap `jet` y lo superpone sobre la imagen original en escala de grises. Devuelve una imagen RGB lista para guardar o mostrar. |
| `comparison_grid.py` | Genera la **figura central de 9 celdas** (3 columnas × 3 filas). Columnas: imagen natural con BLIP base / radiografía con BLIP base / radiografía con BLIP fine-tuneado. Filas: caption generado (texto) / cross-attention heatmap / Grad-CAM heatmap. Guarda el resultado en `outputs/figura_central/grid_9celdas.png`. |

---

### `src/metrics/`

| Archivo | Descripción |
|---|---|
| `nlg_metrics.py` | Calcula BLEU-4, CIDEr y METEOR comparando los captions generados (usando `findings` o `impression` como referencia) antes y después del fine-tuning. Usa `pycocoevalcap` o `nltk`. Guarda resultados en `outputs/metricas/nlg_antes_despues.csv`. |
| `spatial_metrics.py` | Para cada imagen y cada palabra, calcula la correlación de Pearson entre el mapa de cross-attention (196 valores) y el mapa de Grad-CAM (196 valores). Agrega por imagen. Guarda en `outputs/metricas/correlacion_espacial.csv`. Si correlación media aumenta post fine-tuning → los dos mecanismos se alinean más. |

---

### `notebooks/`

| Notebook | Descripción |
|---|---|
| `00_exploracion_dataset.ipynb` | Carga el HF dataset, inspecciona distribución de longitud de `findings` e `impression`, muestra ejemplos de imagen+texto, filtra filas con campos vacíos, y llama a `split_generator.py` para crear los splits. Punto de entrada del proyecto. |
| `01_calibracion_coco.ipynb` | **Parte 1.** Carga BLIP base, corre cross-attention + Grad-CAM sobre las 20–30 imágenes de COCO, genera figuras y verifica visualmente que los heatmaps apuntan a los objetos correctos para cada palabra del caption. |
| `02_baseline_radiografias.ipynb` | **Parte 2.** Carga BLIP base, corre el mismo pipeline sobre las 20–30 radiografías de `selected_indices.json`. Documenta captions generados y (in)coherencia médica de los heatmaps. Establece el baseline. |
| `03_finetuning.ipynb` | Fine-tuning de BLIP sobre los 10.000 pares de `splits/train_indices.json`. Se ejecuta en la nube (Kaggle con GPU T4 o Google Cloud con L4). Monitorea val loss por época y guarda checkpoints. |
| `04_analisis_postft.ipynb` | **Parte 3.** Carga `models/blip_finetuned/best/`, corre el pipeline sobre las *mismas* radiografías de `selected_indices.json`. Genera la figura comparativa antes/después para cada imagen. |
| `05_resultados_y_metricas.ipynb` | Calcula BLEU-4, CIDEr, METEOR y correlación espacial. Genera tablas y figuras listas para el informe. Genera `grid_9celdas.png`. |

---

### `outputs/`

| Carpeta / Archivo | Descripción |
|---|---|
| `parte1_coco/` | Figuras de calibración: imagen original + heatmaps de cross-attention y Grad-CAM por palabra, en imágenes naturales. |
| `parte2_baseline/` | Las mismas figuras sobre las 20–30 radiografías seleccionadas con BLIP base. |
| `parte3_finetuned/` | Las mismas figuras sobre las *mismas* radiografías pero con BLIP fine-tuneado. Se comparan directamente con `parte2_baseline/`. |
| `figura_central/grid_9celdas.png` | La figura 3×3 que resume visualmente todo el trabajo. Es la pieza central del poster. |
| `metricas/nlg_antes_despues.csv` | BLEU-4, CIDEr, METEOR antes y después del fine-tuning. |
| `metricas/correlacion_espacial.csv` | Correlación cross-attention vs Grad-CAM por imagen y por palabra, antes y después del fine-tuning. |

---

### Archivos raíz

| Archivo | Descripción |
|---|---|
| `README.md` | Descripción del proyecto e instrucciones para reproducir todos los experimentos. Incluye el comando `load_dataset(...)` para descargar los datos. |
| `requirements.txt` | `transformers`, `datasets`, `torch`, `torchvision`, `pytorch-grad-cam`, `nltk`, `pycocoevalcap`, `matplotlib`, `numpy`, `pandas`, `Pillow` |
| `.gitignore` | Excluye: `data/hf_cache/`, `models/blip_base/`, `models/blip_finetuned/`, `outputs/`, `__pycache__/`, `.ipynb_checkpoints/` |

---

## Orden de ejecución recomendado

```
1. Instalar dependencias               →  pip install -r requirements.txt
2. Explorar y preparar datos           →  notebooks/00_exploracion_dataset.ipynb
                                          (genera splits/ y selected_indices.json)
3. Descargar BLIP base                 →  src/models/blip_loader.py (auto al primer import)
4. Calibrar herramientas               →  notebooks/01_calibracion_coco.ipynb  (local)
5. Baseline en radiografías            →  notebooks/02_baseline_radiografias.ipynb  (local)
6. Fine-tuning                         →  notebooks/03_finetuning.ipynb  (Kaggle / Google Cloud)
7. Análisis post fine-tuning           →  notebooks/04_analisis_postft.ipynb  (local)
8. Métricas y figuras finales          →  notebooks/05_resultados_y_metricas.ipynb  (local)
9. Redactar informe y poster           →  informe/ y poster/
```

---

## Nota sobre el campo de texto a usar

El dataset tiene dos campos de texto:

| Campo | Contenido | Longitud típica | Uso recomendado |
|---|---|---|---|
| `findings` | Descripción detallada hallazgo por hallazgo | Larga (hasta 1.5k chars) | Fine-tuning (target de generación) |
| `impression` | Conclusión clínica resumida | Corta (1–3 oraciones) | Evaluación cualitativa, comparación con captions generados |

Para el fine-tuning se recomienda usar `findings` como texto objetivo, ya que es más descriptivo y entrena al modelo a generar lenguaje médico detallado. Para evaluar si el modelo generó algo médicamente coherente, comparar el caption generado con `impression` es más intuitivo.

---

## Resultados esperados

El trabajo puede concluir en uno de tres escenarios, todos igualmente válidos como contribución:

- **Resultado A** — Captions mejoran *y* heatmaps se vuelven médicamente coherentes → el fine-tuning adapta lenguaje y visión simultáneamente.
- **Resultado B** — Captions mejoran *pero* heatmaps no cambian significativamente → el modelo aprendió vocabulario médico pero sigue mirando las mismas zonas.
- **Resultado C** — Heatmaps cambian pero no hacia zonas médicamente relevantes → el fine-tuning reorganiza la atención sin coherencia clínica.