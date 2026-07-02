# Análisis de resultados — Calidad de captions (base vs ft5k vs ft10k)

**Fecha:** 2026-07-02
**Datos:** 25 radiografías de `data/visual_test_indices.json`, captions generadas con
T=1.2 / top_p=0.95 / best-of-3 seeds.
**Fuente:** `outputs/notebook_comparativo/metrics/caption_metrics_summary.csv` y
`caption_metrics_per_item.csv`, generados por `scripts/run_caption_metrics.py --coco`.

> Este análisis evalúa **la calidad del texto generado** (¿el modelo describe bien
> la radiografía?). Para el análisis de heatmaps ver `docs/plan_analisis_presentacion.md`.

---

## 1. Tabla de resultados

| Modelo | BLEU-4 | ROUGE-L F | CIDEr | Recall médico | F1 médico | Unique ratio |
|---|---|---|---|---|---|---|
| base  | 0.071 | 0.066 | 0.033 | 0.085 | 0.068 | 0.840 |
| ft5k  | 0.075 | 0.100 | 0.052 | 0.231 | 0.183 | 0.640 |
| ft10k | 0.077 | 0.105 | 0.052 | **0.319** | 0.226 | 0.760 |

### Distribución de categorías clínicas (% de las 25 imágenes)

| Modelo | correct_neg | good_overlap | partial_overlap | miss | hallucination | uncategorized |
|---|---|---|---|---|---|---|
| base  | 12% | 0%  | 16% | **56%** | 0%  | 16% |
| ft5k  | 0%  | 12% | 40% | 0%  | 12% | 36% |
| ft10k | 0%  | **20%** | 36% | 8%  | 12% | 24% |

---

## 2. Hallazgo central

**El recall médico casi se cuadruplica con el fine-tuning: 0.085 → 0.319 (×3.75).**

El recall médico mide qué fracción de los términos clínicos de la referencia real
aparecen en la caption generada. Es la métrica más informativa en este dominio
porque no depende del fraseo exacto, solo de si el hallazgo correcto fue nombrado.

- El modelo **base falla mayormente por omisión** (56% miss): genera captions no
  médicas, así que casi nunca menciona el hallazgo real.
- El fine-tuning **elimina el problema de omisión** (ft5k: 0% miss) y aumenta
  progresivamente los aciertos sustanciales (`good_overlap`: 0% → 12% → 20%).
- El costo: aparecen **alucinaciones** (12%) — el modelo empieza a inventar
  hallazgos en estudios normales. Es el precio de que hable en lenguaje médico.

---

## 3. Ejemplos concretos (referencia vs generada)

### 3.1 Progresión clara del fine-tuning — idx_26073

```
REF   : Status post right thoracentesis with improvement in right pleural effusion
        with residual small right and moderate left pleural effusion...

base  : "thoroidus on an uncute chest ct scan with meta in the upper half"
        → miss (recall 0.00): palabras inventadas, sin contenido médico real

ft5k  : "no appended for acute pneumonia or aspiration given clinical diagnosis.
        bibasilar opacities..."
        → uncategorized (recall 0.00): vocabulario médico pero no el de la referencia

ft10k : "findings suggest small right pleural effusion and increase in extent of
        left lower lobe mass..."
        → good_overlap (recall 0.71): captura "pleural effusion", "right", "left"
```

Este caso ilustra el escenario ideal: base genera ruido, ft5k ya habla en médico
pero desalineado, ft10k acierta el hallazgo principal (derrame pleural).

### 3.2 Alucinación — idx_2838

```
REF   : No acute intrathoracic abnormality.
ft10k : "minimal left-sided consolidation. no signs of pneumonia."
        → hallucination: la referencia es normal, la caption inventa una consolidación
```

Muestra el modo de falla del fine-tuning: sobre-reporte en estudios normales.

### 3.3 Falla del modelo base — idx_7252

```
REF   : Interval repositioning of the right PICC... Persistent extensive
        heterogeneous bilateral opacities...
base  : "thoroidus on thoroidus related to the thoroidus, showing that both
        thoroidus and thoroidus are usually due to..."
        → miss: repetición incoherente, cero contenido clínico
```

El baseline confirma que BLIP sin fine-tuning no cubre el dominio radiológico.

---

## 4. Lectura de las métricas de n-gramas (BLEU / ROUGE / CIDEr)

- **BLEU-4 queda plano** (0.071 → 0.077) y bajo. No capta la mejora clínica: el
  fraseo cambia (referencias telegráficas vs captions más largas) aunque el
  contenido médico mejore mucho.
- **ROUGE-L sube modestamente** (0.066 → 0.105): detecta algo más de estructura
  compartida que BLEU.
- **CIDEr mejora +59%** (0.033 → 0.052): al ponderar por TF-IDF da más peso a los
  términos médicos raros, así que ve parte de la mejora. Sigue baja en absoluto
  por las referencias cortas y el fraseo variable.

**Conclusión metodológica:** en captioning médico las métricas de n-gramas
subestiman la mejora. El **recall médico** y la **categorización clínica** son
las señales que hay que reportar como principales; BLEU/ROUGE/CIDEr van como
complemento estándar.

---

## 5. Relación con la pregunta de investigación

La pregunta del TP (ver `CLAUDE.md` §1) es si el fine-tuning adapta el lenguaje
**y** la visión. Este análisis cubre la parte **lenguaje**:

- **El lenguaje mejora claramente:** recall médico ×3.75, desaparición del modo
  de falla por omisión, aparición de aciertos sustanciales.
- **Con un límite:** ~20% de `good_overlap` y 12% de alucinaciones marcan un techo
  de precisión diagnóstica con BLIP-base y fine-tuning parcial (consistente con
  `analisis/02`, que reporta ~15-25% de overlap como ceiling arquitectónico).

Falta cruzar esto con el análisis de heatmaps para responder si la **visión**
también se adaptó (escenario A vs B del CLAUDE.md).

---

## 6. Métricas usadas — referencias bibliográficas

| Métrica | Qué mide | Referencia |
|---|---|---|
| BLEU | Solapamiento de n-gramas con brevity penalty | Papineni et al., ACL 2002 |
| ROUGE-L | F-measure sobre subsecuencia común más larga (LCS) | Lin, ACL 2004 |
| CIDEr | N-gramas ponderados por TF-IDF (estándar en captioning) | Vedantam et al., CVPR 2015 |
| Recall médico | Fracción del vocabulario clínico de la ref. capturado | Aporte propio (vocab de `token_filter.py`) |
| Categorización clínica | hallucination / miss / overlap | Adaptado de `analisis/02_captions_10k.md` |
| BERTScore (futuro) | Similitud semántica vía embeddings BERT | Zhang et al., ICLR 2020 |

**Nota sobre METEOR:** se evaluó pero se descartó — su implementación en
`pycocoevalcap` depende de un JAR de Java incompatible con el entorno. CIDEr
cubre el rol de métrica COCO estándar.

**Referencias del dominio (para el paper):**
- Li et al., *BLIP*, ICML 2022 — modelo base.
- Nicolson et al., *Fine-tuning de captioning médico*, AIIM 2023 — antecedente directo.
- Jain & Wallace, NAACL 2019 — por qué las métricas superficiales no bastan.

---

## 7. Cómo reproducir

```bash
conda activate tp_vision
python scripts/run_caption_metrics.py --coco
# genera:
#   outputs/notebook_comparativo/metrics/caption_metrics_per_item.csv
#   outputs/notebook_comparativo/metrics/caption_metrics_summary.csv
```

Detalle de qué hace cada métrica: `docs/METRICAS_CAPTIONS.md`.
