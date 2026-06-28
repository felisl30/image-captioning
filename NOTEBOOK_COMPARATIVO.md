# Diseño del notebook comparativo de explicabilidad — `07_explicabilidad_comparada.ipynb`

**Deriva de:** `HOJA_DE_RUTA.md` (decisiones del 2026-06-28).
**Objetivo:** pipeline de evaluación sobre las **25 radiografías fijas** de
`data/visual_test_indices.json`, que para cada modelo (base / 5k / 10k) genere su caption
(best-of-3, T=1.2) y muestre **a dónde miró** con los **tres métodos** de explicabilidad
(cross-attention post-softmax, Grad-CAM, cross-attention logits Q·K). Todos los outputs se
guardan en **`outputs/notebook_comparativo/`** para poder revisarlos y re-graficarlos después
sin recomputar.

> Regla heredada: notebook **delgado**. Toda la lógica vive en `src/`; el notebook orquesta.
> Todo corre en **GPU**. No reentrenar, no tocar los índices fijos, nunca greedy para el análisis.

---

## 1. Estructura de outputs (`outputs/notebook_comparativo/`)

Todo lo que el notebook produce queda acá, organizado para acceso posterior:

```
outputs/notebook_comparativo/
├── captions/
│   └── captions_bestof3.json        # por (idx, modelo): 3 candidatas + scores + elegida + token_ids
├── heatmaps/
│   └── idx_<NNN>/
│       ├── base/
│       │   ├── original.png
│       │   ├── post_softmax_grid.png    # grilla por palabra, método post-softmax
│       │   ├── gradcam_grid.png         # grilla por palabra, método Grad-CAM
│       │   ├── qk_logits_grid.png       # grilla por palabra, método logits Q·K
│       │   └── explanation.png          # figura principal: tokens × 3 métodos + caption + ref
│       ├── ft5k/   (idem)
│       └── ft10k/  (idem)
├── arrays/                              # mapas crudos (24×24) para re-graficar sin recomputar
│   └── idx_<NNN>__<modelo>__<metodo>.npz   # claves = tokens; valores = array(24,24)
├── metrics/
│   ├── spatial_per_token.csv           # concentración + COM por (idx, modelo, metodo, token)
│   └── spatial_summary.csv             # promedio por (modelo, metodo) sobre tokens médicos
├── figures_paper/
│   ├── postsoftmax_vs_logits.png       # figura metodológica (mapa plano vs estructurado)
│   └── comparativa_modelos_idx<NNN>.png# base|5k|10k apilados para una imagen
└── summary.csv                         # índice maestro de toda la corrida (ver §5)
```

**Por qué este layout:**
- `captions/` separa la generación (cara, sampleada) del resto → se cachea una vez.
- `arrays/` guarda los mapas crudos 24×24 → permite re-hacer figuras con otro colormap/alpha
  sin volver a correr los modelos.
- `heatmaps/idx_<NNN>/<modelo>/` agrupa por imagen y modelo → fácil de inspeccionar a mano.
- `metrics/` y `figures_paper/` son los entregables del análisis.
- `summary.csv` es el punto de entrada para cargar todo en cualquier notebook posterior.

---

## 2. Funciones que faltan — qué escribir y dónde

Orden de dependencia. Cada función dice **archivo**, **firma** y **comportamiento**.

### 2.1 `src/models/generation.py` (NUEVO) — generación best-of-3

```python
def generate_caption_best_of_n(
    model, processor, image,
    seeds=(42, 43, 44),
    temperature=1.2, top_p=0.95, max_new_tokens=40,
    medical_vocab: set[str] | None = None,
    device="cuda",
) -> dict:
    """Genera N captions T=1.2 (una por seed) y elige la mejor por score automático.

    Returns:
        {
          "caption": str,                 # la elegida
          "token_ids": Tensor,            # ids de la elegida (para los extractores)
          "tokens": list[str],
          "chosen_seed": int,
          "candidates": [                 # las 3, para transparencia/cache
             {"caption": str, "score": float, "seed": int,
              "medical_richness": int, "rep_penalty": float, "len_penalty": float},
             ...
          ],
        }
    """
```

Helpers privados en el mismo archivo:

```python
def _medical_richness(tokens: list[str], medical_vocab: set[str]) -> int:
    """Nº de tokens médicos DISTINTOS (tras merge de subwords). Término principal del score."""

def _repetition_penalty(tokens: list[str]) -> float:
    """Fracción de bigramas repetidos. Castiga degeneración ('effusion effusion effusion')."""

def _length_penalty(tokens: list[str]) -> float:
    """Penalización fuerte si < 3 palabras de contenido o largo anómalo/truncado."""

def _score_caption(tokens, medical_vocab) -> float:
    """score = medical_richness - λ_rep*rep_penalty - λ_len*len_penalty.
       Desempate determinista por seed más bajo lo resuelve el caller."""
```

**Detalle de selección:** generar con cada seed (fijar `torch.manual_seed(seed)` antes de cada
`generate`), scorear, quedarse con el máximo; ante empate, menor seed. `medical_vocab` viene de
`token_filter.MEDICAL` (§2.4).

### 2.2 `src/interpretability/cross_att_logits.py` (MODIFICAR) — aceptar caption fija

`extract_cross_att_logits` hoy regenera con greedy. Agregar parámetro **opcional**:

```python
def extract_cross_att_logits(
    model, processor, inputs, num_batch,
    layer_idx=9, head_reduction="max",
    generated_ids: "Tensor | None" = None,   # NUEVO
) -> list[dict]:
    # Si generated_ids is not None -> NO generar; explicar esa secuencia.
    # Si es None -> comportamiento actual (retrocompatible).
```

**Riesgo a resolver al codear (de la hoja de ruta):** el hook de K depende del KV-cache (dispara
en el paso con shape 577). Si re-inyectar ids rompe la captura, la alternativa segura es generar
internamente con `do_sample=True, temperature=1.2, top_p=0.95` y el `chosen_seed` de C2 — pero
preferir re-inyección para garantizar que los 3 métodos expliquen exactamente la misma caption.

### 2.3 `src/interpretability/cross_attention.py` (MODIFICAR) — post-softmax con caption fija

Mismo cambio en `eval_and_extract_cross_att` (el método post-softmax):

```python
def eval_and_extract_cross_att(..., generated_ids=None):
    # idem: si se pasa, explica esa secuencia; si no, comportamiento actual.
```

### 2.4 `src/interpretability/gradcam.py` (MODIFICAR) — inyectar ids

`compute_gradcam` ya genera internamente con greedy (`model.generate(max_new_tokens=40)`).
Exponer un parámetro para pasar los ids ya generados:

```python
def compute_gradcam(model, processor, images, device="cpu",
                    generated_ids_list: "list[Tensor] | None" = None):  # NUEVO
    # Si se pasa, usa esos ids en _gradcam_single en vez de regenerar.
```

### 2.5 `src/interpretability/token_filter.py` (NUEVO) — blacklist + resaltado

```python
STOPWORDS = { ... }   # ver blacklist completa en HOJA_DE_RUTA.md §C3
PUNCT     = set(".,;:'\"()/-")
SPECIAL   = {"[CLS]", "[SEP]", "[PAD]"}
MEDICAL   = { "effusion", "edema", "atelectasis", ... }  # solo para resaltar, NO filtra

def is_blacklisted(token: str) -> bool:
    """True si es stopword / puntuación / token especial / resto de subword (##...)."""

def is_medical(token: str) -> bool:
    """True si el token está en MEDICAL (para colorear en la figura)."""

def filter_relevant_tokens(maps: list[tuple[str, "np.ndarray"]]
                           ) -> list[tuple[str, "np.ndarray"]]:
    """Descarta de la lista (palabra, mapa) los tokens blacklisted. Conserva el resto."""
```

### 2.6 `src/interpretability/compare.py` (NUEVO) — orquesta los 3 métodos

Para no repetir lógica en el notebook, una función que devuelve los 3 métodos sobre la **misma**
caption:

```python
def extract_all_methods(
    model, processor, image, generated_ids, device="cuda",
    layer_idx=9, head_reduction="max",
) -> dict:
    """Corre los 3 métodos sobre la caption ya generada (generated_ids).

    Returns:
        {
          "post_softmax": {"caption": str, "maps": [(palabra, array24x24), ...]},
          "gradcam":      {"caption": str, "maps": [...]},
          "qk_logits":    {"caption": str, "maps": [...]},
        }
    """
```

Internamente llama a §2.2/2.3/2.4 pasándoles `generated_ids`, y filtra cada `maps` con
`filter_relevant_tokens` (§2.5). La caption debe ser idéntica en los tres (misma secuencia).

### 2.7 `src/visualization/heatmap.py` (MODIFICAR) — figura principal por modelo

```python
def plot_model_explanation(
    image, results_by_method: dict, reference: str, model_tag: str,
    generated_caption: str, alpha=0.55, colormap="jet", dpi=200,
) -> "plt.Figure":
    """Figura para UN modelo y UNA imagen.

    Layout: una FILA por token relevante (tras blacklist), una COLUMNA por método
    (original | post-softmax | grad-cam | qk-logits). Tokens médicos resaltados.
    Título: model_tag + caption generada. Leyenda: impression de referencia.
    Rótulo fijo: 'normalización per-heatmap → comparar patrón, no intensidad'.
    """
```

(Se reusa `overlay_heatmap` ya existente. `save_heatmap_grid` ya sirve para los *_grid.png por
método.)

### 2.8 `src/metrics/spatial_metrics.py` (NUEVO) — métricas espaciales

```python
def mass_concentration(heatmap: "np.ndarray") -> float:
    """Qué tan enfocado: entropía espacial normalizada (o fracción de masa en top-k% de parches).
       Bajo = difuso, alto = concentrado."""

def center_of_mass(heatmap: "np.ndarray") -> tuple[float, float]:
    """(row, col) del foco de atención, en coords de la grilla 24x24."""

def com_shift(h_a, h_b) -> float:
    """Distancia euclídea entre centros de masa de dos heatmaps (mismo método, distinto modelo)."""

def summarize_per_model(rows) -> "pd.DataFrame":
    """Promedia concentración sobre los tokens médicos por (modelo, método).
       Comparar SIEMPRE dentro de un método entre modelos, nunca entre métodos (escalas distintas)."""
```

---

## 3. Estructura del notebook (celdas)

El notebook solo **orquesta y cachea**. Pseudocódigo por celda:

**Celda 1 — Config y paths**
```python
INDICES   = json.load(open("data/visual_test_indices.json"))     # 25 imágenes
MODELS    = {"base":  "models/blip_base",
             "ft5k":  "../output_5k/best",                        # fuera del repo
             "ft10k": "models/blip_finetuned_10k/best"}
OUT       = Path("outputs/notebook_comparativo")
SEEDS     = (42, 43, 44); T = 1.2; TOP_P = 0.95
LAYER_IDX = 9; HEAD_REDUCTION = "max"; DEVICE = "cuda"
# crear subcarpetas de §1
```

**Celda 2 — Dataset y referencias**
Cargar split HF (`load_mimic_dataset`), levantar las 25 imágenes y sus `impression`.

**Celda 3 — Generación best-of-3 (cachea `captions/captions_bestof3.json`)**
Para cada modelo y cada idx: `generate_caption_best_of_n` [2.1]. Si el JSON ya existe, **cargarlo**
en vez de regenerar (la generación es lo más caro/no determinista de re-hacer).

**Celda 4 — Extracción de los 3 métodos + figuras (loop principal)**
```python
for tag, path in MODELS.items():
    model, processor = load_model_and_processor(path, device=DEVICE)
    for idx in INDICES:
        gen = captions[tag][idx]                      # del cache de celda 3
        res = extract_all_methods(model, processor, img, gen["token_ids"], ...)  # [2.6]
        # guardar arrays crudos -> arrays/idx_<idx>__<tag>__<metodo>.npz
        # guardar *_grid.png por método -> heatmaps/idx_<idx>/<tag>/
        # plot_model_explanation -> heatmaps/idx_<idx>/<tag>/explanation.png  [2.7]
    del model; torch.cuda.empty_cache()
```

**Celda 5 — Métricas espaciales (cachea `metrics/*.csv`)**
Recorrer los arrays crudos, calcular `mass_concentration` y `center_of_mass` por
(idx, modelo, método, token médico), guardar `spatial_per_token.csv` y `summarize_per_model`
→ `spatial_summary.csv` [2.8].

**Celda 6 — Figuras del paper**
- `comparativa_modelos_idx<NNN>.png`: apilar `explanation.png` de base|5k|10k para 2-3 imágenes
  con hallazgos claros.
- `postsoftmax_vs_logits.png`: para un token médico, mapa post-softmax (plano) vs logits Q·K
  (estructurado).

**Celda 7 — `summary.csv`**
Escribir el índice maestro (§5).

---

## 4. Reproducibilidad y caché

- **Generación** (celda 3): seeds fijos (42,43,44) + cache en JSON. Si existe, no regenerar.
- **Arrays crudos** (celda 4): permiten re-graficar (otro colormap/alpha) sin tocar los modelos.
- **Idempotencia:** cada celda chequea si su output ya existe y ofrece `FORCE=True` para rehacer.
- **Orden de carga de modelos:** uno por vez, liberar VRAM entre modelos (`del` + `empty_cache`).

---

## 5. `summary.csv` — columnas

Un renglón por (idx, modelo):

| columna | qué |
|---|---|
| `idx` | índice de la imagen (visual_test) |
| `model_tag` | base / ft5k / ft10k |
| `model_path` | ruta del checkpoint |
| `reference` | `impression` del radiólogo |
| `caption` | caption elegida (best-of-3) |
| `chosen_seed` | seed ganador |
| `n_candidates` | 3 |
| `n_tokens_total` | tokens de la caption |
| `n_tokens_relevant` | tokens tras blacklist |
| `n_tokens_medical` | tokens médicos (resaltados) |
| `explanation_path` | ruta a `explanation.png` |
| `arrays_prefix` | prefijo de los `.npz` de esa (idx, modelo) |
| `mean_concentration_qklogits` | concentración media (tokens médicos) |
| `status` / `error` | ok / detalle de fallo |

Este CSV es el punto de entrada para cualquier análisis posterior: se carga en pandas y desde
ahí se navegan figuras, arrays y métricas.

---

## 6. Checklist de implementación (orden)

1. `token_filter.py` [2.5] — sin dependencias, base de todo.
2. `generation.py` [2.1] — usa `MEDICAL` de token_filter.
3. Parámetro `generated_ids` en los 3 extractores [2.2, 2.3, 2.4].
4. `compare.py` [2.6] — pega 2 y 3.
5. `plot_model_explanation` en `heatmap.py` [2.7].
6. `spatial_metrics.py` [2.8].
7. Notebook `07_explicabilidad_comparada.ipynb` que orquesta (celdas §3).

**Caveats a dejar escritos en el notebook (para el paper):** normalización per-heatmap (patrón,
no intensidad); logits Q·K pendientes de validación del profesor (degradar a post-softmax +
Grad-CAM si los rechaza); comparación cross-modelo cualitativa (tokens no alineados); el base
produce poco vocabulario médico y eso es un resultado, no un bug.
