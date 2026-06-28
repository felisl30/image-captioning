# Hoja de ruta — Comparación de explicabilidad (base / 5k / 10k) y análisis del paper

**Fecha:** 2026-06-28
**Autor del plan:** sesión de planificación (no se ejecutó ni modificó código).
**Objetivo de esta etapa:** notebook que compare los tres métodos de explicabilidad
(cross-attention post-softmax, logits Q·K, Grad-CAM) sobre las radiografías fijas,
con el modelo 10k + T=1.2, y analizar cómo cambian las regiones que mira el modelo
**sin fine-tuning → 5k de FT → 10k de FT**. Luego, redacción del análisis del paper.

> Regla de esta etapa: **primero se cierra el código (solo diseño en este doc, sin ejecutar),
> después se corre, después se escribe el análisis.** Este documento es solo el diseño.

---

## 0. Estado verificado del repo (FASE 1 — exploración)

Lo que existe y funciona:

| Componente | Estado |
|---|---|
| `src/interpretability/cross_attention.py` | `eval_and_extract_cross_att` (post-softmax) y `eval_and_extract_qk_logits`. Ambos regeneran con greedy internamente. |
| `src/interpretability/cross_att_logits.py` | `extract_cross_att_logits` (Q·K, `layer_idx=9`, `head_reduction="max"`). Regenera con greedy. |
| `src/interpretability/gradcam.py` | `compute_gradcam` + `blip_vit_reshape_transform` (577→24×24). Regenera con greedy. |
| `src/visualization/heatmap.py` | `overlay_heatmap`, `plot_word_heatmaps`, `save_heatmap_grid`, `plot_comparison_heatmaps` (solo 2 métodos). Normalización **per-heatmap**. |
| `archivos_ion/run_d3_heatmap_probe.py` | Pipeline base vs **un** FT. Buen esqueleto pero compara 2 modelos / 2 métodos. |
| Checkpoints locales | `models/blip_base`, `models/blip_finetuned_10k/{best,epoch_1,epoch_2,epoch_3}`. |
| **Checkpoint 5k** | **NO está en `models/`** (`models/blip_finetuned/best` vacío). **Sí existe** en `../output_5k/best/` (fuera del repo, ruta absoluta `tp_final/output_5k/best`). |
| Imágenes del análisis | `data/visual_test_indices.json` → **25 radiografías fijas** (idx 731, 2296, …). Son las de todos los heatmaps previos. |
| Captions T=1.2 existentes | `outputs/prueba_mas_temp/captions_temp12.json` → son de `selected_indices` (idx 15399…), **no** de las 25 visuales. Hay que generar frescas. |

Lo que **NO existe** (y CLAUDE.md/PLAN.md asumen que sí):

- `src/interpretability/encoder_attention.py` — mencionado, no está.
- `src/metrics/` (nlg_metrics, spatial_metrics) — no está.

**Supuesto de interpretación** (confirmar si es incorrecto): *"las imágenes visuales"* = las
**25 radiografías de `visual_test_indices.json`**, no imágenes naturales de COCO. Todo el
trabajo previo de heatmaps usa esos índices, así que el plan asume eso.

---

## 1. Diagnóstico crítico (FASE 2) — por qué el código actual no alcanza

1. **Greedy colapsa en los FT.** Las tres funciones de extracción hacen `model.generate(num_beams=1)`
   por dentro. En 5k/10k eso da "no acute cardiopulmonary process" (4 tokens, sin vocabulario
   médico). El pedido "10k + T=1.2" es **incumplible** sin tocar estas funciones.

2. **Cada método explica una caption distinta.** Como cada uno regenera por su cuenta, el
   post-softmax, el logit Q·K y el Grad-CAM pueden estar explicando captions diferentes → la
   comparación entre métodos no es válida tal cual.

3. **No se puede alinear el mismo token entre modelos.** El base no dice "effusion"; el 10k sí.
   Si cada modelo genera lo suyo, la pregunta central *"¿a dónde mira al decir 'effusion',
   sin-ft vs 5k vs 10k?"* no tiene respuesta porque el token no aparece en los tres.

4. **Falta filtro de stopwords/puntuación** (pedido explícito).

5. **El comparador es de 2 métodos / 2 modelos.** Se necesitan 3 métodos × 3 modelos.

6. **Todo cualitativo.** Sin métrica espacial no se puede *cuantificar* el cambio de foco.

7. **Normalización per-heatmap.** `overlay_heatmap` hace min-max individual → comparables los
   **patrones** espaciales, **no** las intensidades absolutas entre modelos. Caveat a documentar.

8. **Sampling no determinista.** T=1.2 cambia entre corridas → sin seed fijo + cache, las
   figuras del paper no son reproducibles.

9. **Cómputo.** **Decisión (2026-06-28): todo corre en GPU**, así que Grad-CAM sobre las 25
   imágenes × 3 modelos × 3 métodos no es problema. Se procesan las **25 radiografías completas**.

10. **Dependencia del profesor.** Los logits Q·K siguen pendientes de validación. El diseño debe
    **degradar con gracia**: si el profesor los rechaza, post-softmax + Grad-CAM siguen sosteniendo
    el análisis.

---

## 2. Decisión metodológica central (FASE 3 + FASE 4) — DEFINIDA 2026-06-28

### Diseño final: un pipeline por modelo, cada modelo se explica a sí mismo

**No hay teacher forcing ni caption forzada entre modelos.** Para **cada** modelo, por separado:

1. El modelo **genera su propia predicción** sobre la radiografía con **T=1.2** (seed fijo).
2. Sobre **esa** caption se extraen los **tres métodos**: cross-attention post-softmax,
   Grad-CAM, y cross-attention logits Q·K — todos explicando la misma caption que el modelo
   acaba de generar.
3. La **impression de referencia se muestra como texto** (título/leyenda) para que el lector
   compare contra la verdad clínica, pero **no se fuerza ni se analiza**.

El mismo pipeline se corre para **base → 5k → 10k**. Cada modelo produce su propia figura
(predicción + 3 métodos). La pregunta *"cómo cambian los lugares a donde mira sin-ft / 5k / 10k"*
se responde **cualitativamente** poniendo las figuras de los tres modelos una al lado de la otra
(no hay alineación forzada de tokens, porque cada modelo dice cosas distintas).

### Qué muestra cada figura (por imagen × por modelo)

```
[ idx=731 | modelo=10k ]   REF (impression): "Acute asymmetric pulmonary edema..."
GENERADO (T=1.2): "moderate pulmonary edema with small bilateral effusion"

  token "edema"      → [post-softmax] [grad-cam] [logits Q·K]
  token "effusion"   → [post-softmax] [grad-cam] [logits Q·K]
  token "bilateral"  → [post-softmax] [grad-cam] [logits Q·K]
  ...                  (solo tokens que pasan la blacklist; médicos resaltados)
```

### Por qué este diseño (FASE 4 — crítica)

- **Honesto:** se analiza lo que el modelo realmente vio al generar lo que realmente dijo. Sin
  artificios contrafácticos (el problema del teacher forcing sobre el base, que descartamos).
- **Comparación cross-modelo es cualitativa por diseño:** como cada modelo genera captions
  distintas, no se comparan los mismos tokens. Se compara el **comportamiento agregado**
  ("¿el 10k mira más el campo pulmonar que el base?"). Esto es exactamente lo que pidió el
  usuario. Limitación a declarar en el paper: no es una comparación token-a-token controlada.
- **El base casi no produce vocabulario médico** → sus figuras tendrán pocos tokens tras la
  blacklist. Eso **es en sí un resultado** (el base no tiene a qué "mirar médicamente porque no
  habla médico"). Reportarlo, no esconderlo.

---

## 3. Trabajo de código a cerrar (SOLO DISEÑO — no ejecutar todavía)

Ordenado por dependencia. Cada item dice qué archivo toca y qué firma se propone.

### C1. Refactor: separar generación de extracción  ⬅ habilita todo lo demás
**Archivos:** `cross_att_logits.py`, `cross_attention.py`, `gradcam.py`.
**Qué:** que las tres funciones acepten `generated_ids` (o `caption_ids`) **opcional**. Si se
pasa, no regeneran: explican esa caption. Si no, mantienen el comportamiento actual
(retrocompatible). El objetivo acá es **que los 3 métodos expliquen la misma caption T=1.2 que
el modelo generó** (no que cada uno regenere con greedy y colapse). NO se necesita teacher
forcing entre modelos (cada modelo usa su propia caption).
- `extract_cross_att_logits(..., generated_ids: Tensor | None = None)`
- `eval_and_extract_cross_att(..., generated_ids=None)` (post-softmax)
- `compute_gradcam` ya pasa ids internamente → exponer un parámetro para inyectarlos.
**Por qué primero:** sin esto no hay T=1.2 (greedy colapsa) ni los 3 métodos comparten caption.
**Riesgo:** los hooks de Q/K dependen del KV-cache (capturan K en el paso con 577 tokens). Si se
re-alimenta la caption en un solo forward, el patrón de captura cambia — **revisar que el hook
siga disparando** una vez con shape 577. Alternativa más segura: mantener la generación con
`do_sample=True, temperature=1.2, top_p=0.95` (mismo seed) dentro de cada extractor en vez de
`num_beams=1`, en lugar de re-inyectar ids. Evaluar ambas en C1; la de menor riesgo gana.

### C2. Helper de generación T=1.2 con BEST-OF-3 (decidido 2026-06-28)
**Archivo nuevo:** `src/models/generation.py` (o celda del notebook si se prefiere fino).
**Decoding:** `do_sample=True, temperature=1.2, top_p=0.95, max_new_tokens=40` para **los TRES
modelos** (base incluido — el base no colapsa, pero se usa el mismo decoding para que la
comparación sea consistente).

**Best-of-3 — cómo se elige (automático y reproducible):**
1. **Generar 3 candidatas** por imagen con **3 seeds fijos** (42, 43, 44). Seeds fijos ⇒ las 3
   candidatas son siempre las mismas ⇒ reproducible para el paper.
2. **Score por candidata** (gana el mayor):
   - `+` **riqueza médica**: nº de tokens **médicos distintos** (lista médica de C3). Término
     principal — más vocabulario clínico = más tokens analizables en los heatmaps.
   - `−` **repetición**: fracción de bigramas repetidos (castiga degeneración tipo
     "effusion effusion effusion").
   - `−` **longitud anómala**: penalización fuerte si < 3 palabras de contenido (caption
     vacía/colapsada) o excesivamente larga/truncada.
3. **Desempate determinista:** gana el de **seed más bajo**.
4. **Transparencia:** cachear **las 3 candidatas + sus scores + cuál ganó** (no solo la elegida).

**Firma:** `generate_caption_best_of_n(model, processor, image, seeds=(42,43,44),
temperature=1.2, top_p=0.95, max_new_tokens=40) -> dict{caption, token_ids, tokens,
candidates: [{caption, score, seed}], chosen_seed}`.
**Cache:** `outputs/explicabilidad/captions_t12_bestof3_visual25.json` por (idx, modelo).
**Por qué:** reproducibilidad + reuso entre los 3 métodos + evita que una muestra mala arruine
la figura, sin selección manual.

### C3. Filtro de tokens — BLACKLIST DEFINIDA (2026-06-28)
**Archivo nuevo:** `src/interpretability/token_filter.py`.
**Firma:** `filter_relevant_tokens(maps: list[(str, array)]) -> list[(str, array)]`
y `is_medical(token: str) -> bool` (para resaltar, no para filtrar).
**Decisión:** **blacklist** (descartar) + **lista médica solo para resaltar en color** (no filtra).

**BLACKLIST (tokens que se descartan del heatmap):**
```python
STOPWORDS = {
    # artículos / determinantes
    "a", "an", "the", "this", "that", "these", "those",
    # preposiciones / conjunciones
    "of", "in", "on", "at", "to", "with", "and", "or", "for",
    "from", "by", "as", "into", "than", "then", "but",
    # pronombres / expletivos
    "it", "its", "there", "which", "who",
    # auxiliares / cópulas
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    # negaciones / relleno vago (clínicos en el texto, pero su heatmap no apunta a nada)
    "no", "not", "without", "change", "evidence", "process", "acute",
}
PUNCT = set(".,;:'\"()/-")           # + tokens que son solo símbolos
SPECIAL = {"[CLS]", "[SEP]", "[PAD]"}  # + restos de subword "##..."
```

**LISTA MÉDICA (solo para resaltar, NUNCA filtra):** hallazgos, dispositivos y modificadores
espaciales/severidad —
`effusion, edema, atelectasis, consolidation, opacity, opacification, cardiomegaly,
pneumothorax, pneumonia, congestion, infiltrate, nodule, mass, fracture, tube, catheter,
line, picc, port, device, left, right, upper, lower, bilateral, basilar, apical, mild,
moderate, severe, small, large, patchy, ...` (extender desde la lista de keywords que ya usa
el overlap de `analisis/02` / notebook 06 para no duplicar criterios).

**Por qué blacklist y no whitelist:** una whitelist médica pura escondería tokens interesantes
no anticipados ("hernia", "devices", abreviaturas). La blacklist conserva todo lo que no sea
relleno y deja el juicio clínico al análisis; el color solo guía la lectura.

### C4. Figura por modelo: predicción + 3 métodos
**Archivo:** extender `src/visualization/heatmap.py`.
**Función nueva (la principal):**
- `plot_model_explanation(image, results_by_method: dict[str, dict], reference: str, model_tag: str, ...)`
  → para **un** modelo y una imagen: una **fila por token relevante** (tras blacklist) y una
  **columna por método** (post-softmax | Grad-CAM | logits Q·K). Título con `model_tag`,
  caption generada, y la **impression de referencia** como leyenda. Tokens médicos resaltados [C3].
**Cómo se compara base/5k/10k:** se corre `plot_model_explanation` una vez por modelo y se
**apilan las tres figuras** (o se arma una grilla 3-modelos × ... para una imagen). No hay
alineación forzada de tokens.
**Caveat a codear visible en la figura:** rótulo "normalización per-heatmap → comparar patrón,
no intensidad absoluta".

### C5. Métricas espaciales (INCLUIDA — Bloque A, decidido 2026-06-28)
**Archivo nuevo:** `src/metrics/spatial_metrics.py`.
Cuantifican el cambio de foco base→5k→10k (los heatmaps, no el texto).
- `mass_concentration(heatmap)` → entropía espacial normalizada o fracción de masa en top-k%.
  Baja = atención difusa; alta = atención enfocada.
- `center_of_mass(heatmap)` → (row, col) del foco.
**Uso compatible con el diseño:** como los modelos generan tokens distintos, NO se compara
token-a-token. Se reporta el **promedio de concentración sobre los tokens médicos** por modelo
→ tabla "concentración media base / 5k / 10k". Responde "¿el FT concentra más la atención?".
**Crítica:** la escala difiere entre métodos (logits vs Grad-CAM) → comparar **dentro** de un
método entre modelos, nunca entre métodos.

### C7. Métricas de calidad de caption (OPCIONALES — a charlar con el profesor)
**Archivo:** `src/metrics/nlg_metrics.py` (lo que aplique) + reuso de los keyword-sets que ya
usa `analisis/02`. **Base ya medida:** overlap de keywords médicos = 0% (base) → 15.2% (10k+T=1.2),
17% alucinaciones, 14% misses. Estas cuatro lo enriquecen sin pasarse:

1. **Precision + Recall + F1 de keywords médicos** *(casi gratis, reusa los sets).*
   El overlap actual es ~recall. Agregar **precision** (de las palabras médicas que el modelo
   dijo, cuántas estaban en la referencia → mide alucinación) y **F1**. Cuenta la historia
   "más recall pero menos precision = habla más médico pero inventa más". *Recomendada.*

2. **Clasificación binaria normal vs. anormal** *(liviana, clínica).*
   Etiquetar caption y referencia como "normal / con hallazgos" por keywords y medir
   accuracy/precision/recall del binario. Responde lo más básico: "¿distingue sano de enfermo?".
   Conecta directo con las 100 alucinaciones (normal→inventa) y 85 misses (anormal→normal).
   *Recomendada.*

3. **CIDEr** *(un número estándar de paper).*
   Métrica de captioning que mejor correlaciona con juicio humano. Requiere `pycocoevalcap`.
   Reportar con greedy (determinista) y enmarcar como referencia: será bajo, esperable en
   radiología. Preferible a BLEU (que el sampling castiga más). *Puede ir.*

4. **BERTScore (similitud semántica)** *(un poco más pesada).*
   Compara por significado, no por palabra exacta → captura sinónimos ("effusion"≈"fluid",
   "opacity"≈"consolidation") que el overlap exacto pierde. Agrega una dependencia y carga un
   modelo, pero es un solo número. *Opcional.*

**Encuadre para el paper (todas):** el objetivo NO es captioning SOTA. Es mostrar que el FT
aprendió lenguaje médico (0%→15%) y, con honestidad, que la fiabilidad clínica es limitada
(alucinaciones/misses). Las métricas documentan ambas cosas. **Decisión final: pendiente de
charla con el profesor.**

### C6. Notebook nuevo `07_explicabilidad_comparada.ipynb`  (delgado, llama a `src/`)
Estructura de celdas:
1. Config: índices = `visual_test_indices.json` (**25 completas**, todo en GPU); modelos =
   {base, 5k(`../output_5k/best`), 10k}; T=1.2 + top_p=0.95 para los **tres** modelos;
   best-of-3 seeds (42,43,44); layer_idx=9; head_reduction.
2. Cargar dataset HF + las imágenes + sus `impression` de referencia.
3. **Por cada modelo** (base → 5k → 10k):
   a. Generar caption **best-of-3** T=1.2 y cachear las 3 candidatas + scores + elegida [C2].
   b. Extraer los **3 métodos** sobre la caption elegida [C1].
   c. Filtrar tokens con la blacklist, resaltar médicos [C3].
   d. `plot_model_explanation` → figura (predicción + 3 métodos), con la referencia en leyenda [C4].
4. Apilar las figuras de los 3 modelos por imagen para el análisis cualitativo cross-modelo.
5. **Métricas espaciales** [C5]: tabla de concentración media (tokens médicos) por modelo.
6. **Fig. post-softmax vs logits** (pendiente histórico): un token, mapa plano vs estructurado.

---

## 4. Priorización (FASE 5)

### Bloque A — imprescindible para que el notebook tenga sentido
1. **C1** que los 3 métodos expliquen la caption T=1.2 (no greedy). ⬅ bloquea todo.
2. **C2** generación T=1.2 cacheada (seed fijo).
3. **C3** filtro de tokens (blacklist ya definida).
4. **C4** `plot_model_explanation` (predicción + 3 métodos por modelo).
5. **C5** métricas espaciales (concentración + centro de masa). ⬅ incluida por pedido.
6. **C6** notebook que orquesta todo, los 3 modelos.

### Bloque B — redacción del paper (recién cuando A corrió)
6. Análisis visual: por modelo, qué generó y a dónde miró (3 métodos); luego cross-modelo
   cualitativo base→5k→10k, cruzando con los supuestos y con `analisis/01–04`.
7. Caveats: normalización per-heatmap (patrón, no intensidad); logits Q·K sin validar; el base
   produce poco vocabulario médico (resultado, no bug); comparación cross-modelo cualitativa.
8. Integrar las nuevas figuras a las ya planeadas en `analisis/03_hallazgos_paper.md` (Fig. 1–5).
9. Cerrar la narrativa "tipo A parcial": el FT adapta visión y lenguaje; el collapse es de
   decodificación.

### Bloque C — opcionales (no bloquean el paper)
- **C7** métricas de calidad de caption (precision/recall/F1, binario normal/anormal, CIDEr,
  BERTScore) — a charlar con el profesor cuáles entran.
- Copiar el 5k a `models/blip_finetuned_5k/best` para no depender de ruta externa.
- Sensibilidad a `layer_idx` y `head_reduction` (1 figura de apoyo metodológico).

---

## 5. Decisiones tomadas (2026-06-28) y riesgos vivos

**Decididas con el usuario:**

| Tema | Decisión |
|---|---|
| "Imágenes visuales" | Las **25 radiografías** de `visual_test_indices.json`. |
| Diseño de comparación | **Sin teacher forcing.** Cada modelo genera su caption T=1.2 y se explica con los 3 métodos. Referencia mostrada como texto, no forzada. Cross-modelo = cualitativo. |
| Generación de caption | **Best-of-3** (seeds 42/43/44), elegida por score automático (riqueza médica − repetición − longitud anómala), candidatas cacheadas. |
| Decoding del base | **T=1.2 igual que los FT** (decoding uniforme entre los 3 modelos). |
| Filtro de tokens | **Blacklist** (definida en C3) + lista médica solo para **resaltar**. |
| Cómputo | **Todo en GPU** → se procesan las **25 radiografías completas** (sin subset). |
| Métricas espaciales | **Incluidas** (Bloque A): concentración + centro de masa por modelo. |
| Métricas de calidad de caption | **Opcionales, a charlar con el profesor** (C7): precision/recall/F1, binario normal/anormal, CIDEr, BERTScore. |

**Riesgos vivos (resolver al codear):**

| Riesgo | Mitigación |
|---|---|
| Hooks Q/K si se re-inyectan ids | En C1, comparar re-inyección vs mantener `do_sample` con seed; elegir la de menor riesgo. |
| Logits Q·K pendientes del profesor | Diseño degrada a post-softmax + Grad-CAM si los rechaza. |
| Normalización per-heatmap | Solo comparar patrón espacial, no intensidad; declararlo en figura y paper. |
| 5k fuera del repo (`../output_5k/best`) | Referenciar por ruta; opcional copiar a `models/` (Bloque C). |

---

## 6. Qué NO hacer (heredado de handoff 2026-06-25)

- No reentrenar 5k ni 10k.
- No tocar `selected_indices.json` ni `visual_test_indices.json`.
- No usar greedy para el análisis final (colapsa).
- No subir T más allá de 1.2.
