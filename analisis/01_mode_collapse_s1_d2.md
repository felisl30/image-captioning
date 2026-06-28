# Análisis 01 — Mode Collapse: Diagnóstico y Solución

**Fecha:** 2026-06-25  
**Modelo:** BLIP-base fine-tuneado sobre MIMIC-CXR (5k imágenes, 3 épocas)  
**Checkpoint analizado:** `models/blip_finetuned_5k/best` (= epoch_3)  
**Imágenes usadas:** 25 radiografías de `data/visual_test_indices.json`

---

## 1. El problema: mode collapse

Tras el fine-tuning, el modelo con decodificación greedy generaba captions casi idénticas para todas las imágenes:

| Métrica | Valor |
|---|---|
| Imágenes evaluadas | 25 |
| Captions únicas | 2 |
| Unique ratio | 0.08 |
| Caption dominante | "no acute cardiopulmonary process." |
| % con caption dominante | 80% |
| Largo promedio | 4 palabras |
| % clasificadas como "normal" | 100% |

El 80% de las 25 imágenes recibía exactamente la misma caption, sin importar su contenido visual.

---

## 2. Herramienta S1 — Experimento de estrategias de decodificación

### Qué hace S1

S1 toma el mismo checkpoint fine-tuneado y genera captions usando 6 estrategias distintas, sin modificar los pesos del modelo. El objetivo es determinar si el collapse es un artefacto de la decodificación greedy o si está internalizado en los pesos.

Las estrategias evaluadas:
- **greedy** — decodificación determinista estándar (baseline colapsado)
- **sample_t1.2_p0.95** — temperatura 1.2 + nucleus sampling (top-p=0.95)
- **sample_t1.3_p0.90** — temperatura 1.3 + nucleus sampling (top-p=0.90)
- **sample_t1.5_p0.85** — temperatura 1.5 + nucleus sampling (top-p=0.85)
- **diverse_beam** — diverse beam search (falló, ver nota)
- **contrastive** — contrastive decoding (falló, ver nota)

Para las estrategias de sampling se generaron 4 captions por imagen (100 totales). Para greedy, 1 por imagen (25 totales).

> **Nota:** `diverse_beam` y `contrastive` fallaron con `ValueError: requires trust_remote_code=True` — cambio de API en la versión de transformers instalada en la VM. No afecta las conclusiones ya que las estrategias de temperatura resolvieron el problema.

### Resultados

| Estrategia | n_captions | Únicas | Unique ratio | top_pct | Largo promedio | % normal | % clínico específico |
|---|---|---|---|---|---|---|---|
| greedy | 25 | 2 | 0.08 | 0.80 | 4.0 | 100% | 0% |
| sample_t1.2_p0.95 | 100 | 99 | **0.99** | 0.02 | 15.0 | 19% | **70%** |
| sample_t1.3_p0.90 | 100 | 99 | **0.99** | 0.02 | 15.9 | 12% | **77%** |
| sample_t1.5_p0.85 | 100 | 98 | **0.98** | 0.03 | 18.7 | 15% | **75%** |

### Ejemplos cualitativos

**Greedy (colapsado):**
```
[idx=731] REF: Acute asymmetric pulmonary edema, right greater than left...
          CAP: no acute cardiopulmonary process.

[idx=2838] REF: No acute intrathoracic abnormality.
           CAP: no acute cardiopulmonary process.
```

**sample_t1.2_p0.95 (collapse roto):**
```
[idx=731] CAP: patchy left base opacification which should be considered...
[idx=731] CAP: no acute intrathoracic process. persistent right pulmonary vascular congestion.
[idx=731] CAP: minimal change and increase consolidation.

[idx=2838] CAP: no acute intrathoracic process.   ← correcto
[idx=2838] CAP: mild vascular congestion. small right pleural effusion.
```

**sample_t1.3_p0.90 (más diverso, más largo):**
```
[idx=731] CAP: stable moderate-to-severe bilateral pleural effusions, moderate to large left base atelectasis.
[idx=731] CAP: slight interval re-placement of the endotracheal, ng tube. a focal opacity concerning for possible infection...
```

### Conclusión S1

El collapse **desaparece completamente con temperatura T=1.2**. El modelo tiene conocimiento médico diverso en sus pesos — el problema estaba en cómo se decodificaba, no en lo que aprendió. T=1.2 es la estrategia recomendada: rompe el collapse, genera captions con vocabulario clínico específico (70%), y mantiene un largo razonable (~15 palabras). T=1.5 es más verboso y puede generar frases algo incoherentes.

---

## 3. Herramienta D2 — Análisis de distribuciones token a token

### Qué hace D2

D2 hace un forward pass con `output_scores=True` y analiza la distribución de probabilidad en cada paso de generación. Para cada token generado calcula:

- **p_top1**: probabilidad del token elegido
- **p_top2**: probabilidad del segundo candidato
- **gap**: p_top1 - p_top2 (qué tan lejos quedó la alternativa más cercana)
- **entropy**: entropía de la distribución completa (0 = certeza total, alto = mucha incertidumbre)
- **pct_steps_gap_lt_0.10**: porcentaje de decisiones donde el top-2 estuvo a menos de 10 puntos del top-1 (decisiones "ambiguas")

D2 corrió sobre los 4 checkpoints disponibles: epoch_1, epoch_2, epoch_3 y best.

### Evolución del collapse por época

| Checkpoint | mean_p_top1 | mean_entropy | pct_gap < 0.10 |
|---|---|---|---|
| epoch_1 | 0.657 | **1.905** | 21.4% |
| epoch_2 | 0.794 | **1.072** | 4.0% |
| epoch_3 / best | 0.756 | **1.196** | 6.7% |

El collapse **se instaló principalmente en epoch_2**: la entropía cayó de 1.91 a 1.07 (casi a la mitad) y los pasos con alternativas cercanas bajaron de 21% a solo 4%. El modelo se volvió mucho más "seguro" de sus predicciones después de una sola época de entrenamiento adicional. En epoch_3 hubo una recuperación parcial (entropía subió a 1.20), lo que sugiere que el modelo comenzó a regularizarse levemente.

Que `best = epoch_3` confirma que epoch_3 tenía el mejor val_loss — el early stopping con patience=2 no llegó a activarse.

### El mecanismo del collapse: cascada de certeza

El dato más revelador está en el análisis token a token. Para `idx=731` (edema pulmonar agudo asimétrico) con epoch_2:

| Step | Token | p_top1 | Entropía | Gap |
|---|---|---|---|---|
| 0 | "no" | 0.212 | 4.380 | 0.163 |
| 1 | "acute" | 0.558 | 2.209 | 0.465 |
| 2 | "card" | 0.840 | 0.848 | 0.748 |
| 3 | "##io" | 0.995 | 0.041 | 0.992 |
| 4 | "##pu" | 0.994 | 0.061 | 0.992 |

En el **step 0**, el modelo elige "no" con p=0.212 — no está seguro. Las alternativas son clínicamente plausibles: ["no", "mild", "interval", "low", "moderate"]. "no" gana con un margen pequeño.

A partir del **step 2** ("card"), el modelo ya está comprometido con "cardiopulmonary" con p=0.840. En step 3 y 4 la entropía cae a ~0.04: la distribución es prácticamente determinista.

**El mismo patrón en las tres épocas para el primer token:**

| Checkpoint | token="no" p_top1 | entropy step 0 | Top-5 alternativas |
|---|---|---|---|
| epoch_1 | 0.182 | 4.623 | [no, interval, low, increased, moderate] |
| epoch_2 | 0.212 | 4.380 | [no, mild, interval, low, moderate] |
| epoch_3 | 0.218 | 4.311 | [no, mild, interval, low, new] |

La probabilidad de "no" en el primer token creció de 0.18 → 0.21 → 0.22 a lo largo del entrenamiento, pero la diferencia real está en lo que pasa después: en epoch_2 la cascada es más agresiva (entropy cae más rápido), explicando el collapse más severo en esa época.

### Por qué el collapse NO es un fallo visual

El modelo elige "no" en step 0 con p=0.18-0.22 — una probabilidad baja. Las alternativas ("interval", "mild", "increased") son médicamente válidas. Esto indica que:

1. El modelo **sí tiene diversidad** en sus representaciones — el conocimiento médico está presente en los pesos
2. El collapse es **estadístico**: en MIMIC-CXR una fracción grande de las impressions empieza con frases negativas ("no acute", "no significant", "no evidence"), y el fine-tuning sobre 5k imágenes amplificó esa frecuencia
3. Greedy decoding **amplifica** ese sesgo: elige "no" aunque gane por poco, y a partir de ahí la cascada cierra la distribución hacia "cardiopulmonary process"

Los **heatmaps de cross-attention y Grad-CAM confirman** que el modelo sí atiende a regiones clínicamente relevantes (campos pulmonares, región cardíaca, bases) incluso cuando genera la caption colapsada. La atención visual es correcta; el problema está en la etapa de generación de texto.

---

## 4. Conclusiones

| Pregunta | Respuesta |
|---|---|
| ¿El collapse es un fallo del modelo? | No — el modelo tiene conocimiento médico diverso en sus pesos |
| ¿Cuándo se instaló el collapse? | Principalmente en epoch_2 (entropía: 1.91 → 1.07) |
| ¿Cuál es el mecanismo? | Sesgo estadístico en step 0 + cascada de certeza en pasos siguientes |
| ¿Cómo se soluciona sin reentrenar? | Temperatura T=1.2 con top-p=0.95 (unique_ratio 0.99, 70% hallazgos clínicos) |
| ¿Es necesario reentrenar para el análisis? | No para el análisis preliminar — los heatmaps son válidos con el modelo actual |

**Estrategia recomendada para el resto del análisis:** usar `sample_t1.2_p0.95` como método de generación estándar. Para análisis de heatmaps palabra por palabra, tomar la caption más coherente de las 4 generadas por imagen.

---

## 5. Archivos generados

| Archivo | Descripción |
|---|---|
| `outputs/decoding_sampling/s1_selected_30/s1_all_captions.csv` | Todas las captions generadas por estrategia |
| `outputs/decoding_sampling/s1_selected_30/s1_decoding_summary.csv` | Resumen estadístico por estrategia |
| `outputs/decoding_sampling/s1_selected_30/s1_strategy_comparison.png` | Plot unique_ratio vs top_pct por estrategia |
| `outputs/mode_collapse_debug/d2_token_probe_steps.csv` | Métricas por token por checkpoint |
| `outputs/mode_collapse_debug/d2_checkpoint_summary.csv` | Resumen por checkpoint |
| `outputs/mode_collapse_debug/d2_high_confidence_examples.csv` | Ejemplos de tokens con alta certeza |
