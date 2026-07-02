# Métricas de calidad de captions

Este documento describe el pipeline de métricas que evalúa **si el modelo genera
buenas captions** comparando la caption generada contra la impresión clínica de
referencia de MIMIC-CXR.

> **No confundir con `docs/metricas_espaciales.md`.** Aquel compara heatmaps entre
> métodos de interpretabilidad (validación metodológica). Este evalúa la calidad
> del texto generado por el modelo (evaluación del modelo).

---

## Archivos

| Archivo | Rol |
|---|---|
| `src/metrics/caption_metrics.py` | Funciones de métricas (una capa sin dependencias + una opcional) |
| `scripts/run_caption_metrics.py` | Runner: lee `summary.csv`, calcula y guarda resultados |

**Entrada:** `outputs/notebook_comparativo/summary.csv` (columnas `reference`, `caption`, `model_tag`, `idx`).

**Salida:**
```
outputs/notebook_comparativo/metrics/caption_metrics_per_item.csv    # una fila por (imagen, modelo)
outputs/notebook_comparativo/metrics/caption_metrics_summary.csv     # agregados por modelo
```

**Uso:**
```bash
python scripts/run_caption_metrics.py           # métricas sin dependencias
python scripts/run_caption_metrics.py --coco    # agrega CIDEr (necesita pycocoevalcap)
```

> **Entorno:** CIDEr necesita el entorno conda `tp_vision`
> (`conda activate tp_vision`), que ya tiene `pycocoevalcap`. El `python3` del
> sistema solo tiene numpy y corre únicamente la capa sin dependencias.

---

## Diseño en dos capas

El entorno local solo tiene numpy. Por eso el pipeline funciona en dos niveles:

1. **Sin dependencias** (stdlib + numpy): siempre corre. BLEU, ROUGE-L, overlap
   médico, categorización clínica, diversidad.
2. **Opcional** (`pycocoevalcap`): CIDEr. Se activa con `--coco` y se omite
   silenciosamente si la librería no está.

---

## Las métricas — qué mide cada una y por qué

### BLEU-1..4 (`sentence_bleu`)

Mide solapamiento de n-gramas entre la caption y la referencia, con penalización
por brevedad (brevity penalty). BLEU-1 son unigramas (palabras sueltas), BLEU-4
llega hasta secuencias de 4 palabras.

**Por qué está:** es la métrica clásica de generación de texto. BLEU alto indica
que la caption usa las mismas palabras y frases que la referencia.

**Limitación en este dominio:** las impresiones clínicas son cortas y usan
fraseo muy variable ("pulmonary edema" vs "edema"), así que BLEU-4 tiende a ser
bajo aunque la caption sea clínicamente correcta. Se implementa con smoothing +1
para evitar ceros en textos cortos. **No sobre-interpretar BLEU-4 acá.**

### ROUGE-L (`rouge_l`)

Basada en la subsecuencia común más larga (LCS). No exige que las palabras sean
contiguas, solo que aparezcan en el mismo orden. Devuelve precisión, recall y
F-measure.

**Por qué está:** más tolerante que BLEU al reordenamiento. Captura si la caption
sigue la misma estructura de la referencia aunque intercale otras palabras.
Es estándar en resumen de texto y complementa a BLEU.

### Overlap de keywords médicos (`medical_overlap`)

Usa el vocabulario médico de `token_filter.MEDICAL` (hallazgos, anatomía,
dispositivos, severidad). Calcula:

- **recall médico**: cuánto del vocabulario médico de la *referencia* capturó la
  caption. Es la métrica más importante — responde "¿el modelo nombró los
  hallazgos que estaban en el informe real?"
- **precision médica**: cuánto del vocabulario médico de la *caption* está en la
  referencia. Detecta invención de términos.
- **Jaccard** y **F1**: combinaciones de las dos.

**Por qué está:** es la métrica más interpretable clínicamente y no depende de
fraseo exacto. Es la que mejor muestra el salto del fine-tuning. Es la versión
formalizada del "overlap de keywords" que aparece en `analisis/02`.

### Categorización clínica (`clinical_category`)

Clasifica cada par (referencia, caption) en una categoría, replicando el análisis
de `analisis/02_captions_10k.md`:

| Categoría | Significado |
|---|---|
| `correct_negative` | referencia normal → caption normal ✓ |
| `hallucination` | referencia normal → caption inventa un hallazgo ✗ |
| `miss` | referencia con hallazgo → caption dice normal / no lo menciona ✗ |
| `good_overlap` | recall médico ≥ 0.5 |
| `partial_overlap` | recall médico entre 0 y 0.5 |
| `uncategorized` | vocabulario médico presente pero sin overlap con la referencia |

Para decidir si un texto es "normal" usa dos señales: presencia de frases como
"no acute", "no evidence", "unremarkable", y ausencia de hallazgos positivos
(subconjunto `FINDINGS` del vocabulario médico).

**Por qué está:** traduce las métricas numéricas a categorías que se entienden
sin contexto técnico. Es lo mejor para una presentación: "el modelo base falla
por omisión (56% miss), el fine-tuneado empieza a capturar hallazgos".

### Diversidad — unique ratio (`unique_ratio`)

Proporción de captions distintas sobre el total. Detecta mode collapse.

**Por qué está:** un modelo colapsado genera la misma caption siempre. Un unique
ratio bajo es señal de alerta. Con T=1.2 debería ser alto.

### CIDEr (opcional, `compute_coco_metrics`)

- **CIDEr** (`--coco`): pondera n-gramas por TF-IDF, dando más peso a los términos
  informativos y menos a las palabras comunes. Es el estándar de facto en
  image captioning. Muy apropiado acá porque prioriza los términos médicos raros.
  Es Python puro dentro de `pycocoevalcap`, así que corre confiablemente y no
  necesita Java.

> **Nota:** se evaluó METEOR pero se descartó — su JAR de Java no es compatible
> con el `openjdk` disponible en el entorno y no devuelve un score válido. CIDEr
> + recall médico cubren el objetivo de evaluación de captions.

---

## Resultados actuales (25 imágenes de `visual_test_indices.json`)

Corrida sin `--coco` sobre las captions generadas con T=1.2:

| Modelo | BLEU-4 | ROUGE-L F | CIDEr | Recall médico | F1 médico | Unique ratio | % good overlap | % hallucination | % miss |
|---|---|---|---|---|---|---|---|---|---|
| base  | 0.071 | 0.066 | 0.033 | 0.085 | 0.068 | 0.840 | 0%  | 0%  | 56% |
| ft5k  | 0.075 | 0.100 | 0.052 | 0.231 | 0.183 | 0.640 | 12% | 12% | 0%  |
| ft10k | 0.077 | 0.105 | 0.052 | 0.319 | 0.226 | 0.760 | 20% | 12% | 8%  |

**Lecturas:**

- **El recall médico casi se cuadruplica** con el fine-tuning (0.085 → 0.319).
  Es el hallazgo más fuerte: el modelo fine-tuneado nombra los hallazgos reales
  del informe mucho más seguido.
- **El modelo base falla por omisión** (56% miss): genera captions no médicas
  ("a male lung with a pneumonia in the stomach"), así que casi nunca captura el
  hallazgo real.
- **`good_overlap` sube 0% → 12% → 20%**: el fine-tuning aumenta progresivamente
  los casos donde la caption captura ≥50% de los hallazgos de la referencia.
- **BLEU-4 y ROUGE-L quedan bajos y casi planos** entre modelos. Confirma que las
  métricas de n-gramas no capturan bien la mejora clínica en este dominio — el
  fraseo cambia aunque el contenido médico mejore. Por eso el recall médico es
  más informativo acá.
- **CIDEr mejora modestamente** (0.033 → 0.052, +59%): al ponderar por TF-IDF
  detecta algo de la mejora que BLEU no ve, pero sigue siendo baja en términos
  absolutos por la misma razón (fraseo variable, referencias cortas).
- **Aparecen alucinaciones** (12%) con el fine-tuning: el modelo empieza a
  inventar hallazgos en estudios normales. Es el costo de que hable en médico.

---

## Idea futura — métrica semántica con BERT (BERTScore)

Todas las métricas actuales son **léxicas**: comparan palabras o n-gramas. Ninguna
entiende que "pulmonary edema" y "fluid in the lungs" significan lo mismo. Ahí es
donde una métrica basada en embeddings ayudaría.

**BERTScore** (Zhang et al., ICLR 2020) compara los embeddings contextuales de
cada token (de un modelo tipo BERT) entre caption y referencia, y calcula
precisión/recall/F1 sobre la similitud coseno de esos embeddings. Captura
similitud semántica aunque las palabras no coincidan.

**Por qué encajaría bien acá:**
- Las impresiones clínicas usan mucho fraseo alternativo para lo mismo.
- Un modelo BERT del dominio médico daría embeddings mucho mejores que uno
  genérico. Candidatos: **BioClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`),
  **PubMedBERT**, o **CXR-BERT** (específico de radiografías de tórax).

**Bosquejo de implementación:**
```python
# pip install bert-score
from bert_score import score

P, R, F1 = score(
    cands=hypotheses,
    refs=references,
    model_type="emilyalsentzer/Bio_ClinicalBERT",  # BERT clínico
    lang="en",
)
# F1.mean() → BERTScore agregado
```

Se agregaría como una tercera capa opcional en `caption_metrics.py`
(`compute_bertscore(references, hypotheses)`), análoga a `compute_coco_metrics`,
que devuelva `None` si `bert-score` no está instalado. Idealmente comparar el
BERTScore genérico vs el clínico para mostrar el efecto del dominio.

**Cuidado:** requiere descargar el modelo (~400 MB) y correr forward passes, así
que conviene hacerlo una sola vez y cachear. No necesita GPU para 25 captions
pero acelera bastante si hay.

---

## Referencias

| Métrica | Referencia |
|---|---|
| BLEU | Papineni et al., ACL 2002 |
| ROUGE | Lin, ACL 2004 |
| CIDEr | Vedantam et al., CVPR 2015 |
| BERTScore | Zhang et al., ICLR 2020 |
| BioClinicalBERT | Alsentzer et al., ClinicalNLP 2019 |
