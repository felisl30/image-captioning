# Análisis 02 — Calidad de Captions: Modelo 10k

**Fecha:** 2026-06-25  
**Modelo:** BLIP-base fine-tuneado sobre MIMIC-CXR (10k imágenes, 3 épocas)  
**Checkpoint:** `models/blip_finetuned_10k/best`  
**Imágenes evaluadas:** 600 (primeros 600 de `data/splits/test_sub_indices.json`)  
**Archivos:** `outputs/captions_10k/captions/`, `outputs/prueba_mas_temp/`

---

## 1. Comparativa de estrategias de decodificación

| | BASE greedy | FT 10k greedy | FT 10k T=1.2 | FT 10k T=1.5 |
|---|---|---|---|---|
| Captions únicas | 93/600 (15%) | 17/600 (3%) | 558/600 (93%) | 583/600 (97%) |
| Caption dominante | 24% | 65% | 2.3% | 1.7% |
| Largo promedio | 9.1 pal. | 4.0 pal. | 15.0 pal. | 18.0 pal. |
| % vocabulario médico | 0% | 9% | 83% | 86% |
| Overlap keywords con ref | 0% | 2.5% | **15.2%** | 14.5% |
| Captions muy cortas (<3 pal.) | 0% | 7.5% | 5.3% | 2.2% |

**Conclusión de estrategia:** T=1.2 gana sobre T=1.5 en overlap (15.2% vs 14.5%) y produce captions más coherentes. T=1.5 genera frases truncadas, con repeticiones y estructura incoherente en ~38/600 casos. **Estrategia recomendada: `sample_t1.2_p0.95`.**

---

## 2. Calidad diagnóstica real (T=1.2, 600 imágenes)

Clasificación de cada par referencia vs caption generada:

| Categoría | N | % | Descripción |
|---|---|---|---|
| Correctas negativas | 51 | 8.5% | Ref normal → cap normal ✓ |
| Buen overlap (≥0.5) | 51 | 8.5% | Captura ≥50% de keywords médicos de la ref |
| Overlap parcial | 55 | 9.2% | Captura algo pero <50% |
| **Alucinaciones** | **100** | **16.7%** | Ref normal → cap inventa findings ✗ |
| **Misses** | **85** | **14.2%** | Ref tiene findings → cap dice normal ✗ |
| Sin categorizar | 258 | 43.0% | Vocab médico pero diferente al de la ref |

### Ejemplos de buenos casos (overlap ≥ 0.5)

```
REF: Findings most consistent with pulmonary edema.
CAP: moderate pulmonary edema with right apex catheter and mediastinum.

REF: Bibasilar atelectasis without definite acute cardiopulmonary process.
CAP: interval advancement of the right base opacity which may represent atelectasis...

REF: Cardiomegaly without superimposed acute cardiopulmonary process.
CAP: moderate cardiomegaly with a small left pleural effusion.
```

### Ejemplos de alucinaciones

```
REF: No evidence of acute disease.
CAP: no pneumothorax.   ← técnicamente inofensivo pero inventado

REF: No acute cardiopulmonary process.
CAP: enteric tube tip not in use. enteric tube in the body of stomach...  ← inventa dispositivos
```

### Ejemplos de misses

```
REF: Basilar atelectasis without definite focal consolidation.
CAP: no change.

REF: Worsening left upper lobe atelectasis.
CAP: no acute cardiopulmonary process.
```

---

## 3. Comparativa 5k vs 10k

| Métrica | 5k + T=1.2 (25 imgs) | 10k + T=1.2 (600 imgs) |
|---|---|---|
| Unique ratio | 0.99 | 0.93 |
| % vocabulario médico | 70% | 83% |
| Overlap con referencia | ~15% | 15.2% |
| Largo promedio | 15.0 pal. | 15.0 pal. |
| % clínico específico (S1) | 70% | 75% |

El salto de 5k a 10k es marginal en overlap y calidad diagnóstica. La mejora principal vino de usar temperatura, no de más datos. Con 10k el modelo aprendió más vocabulario específico (83% vs 70%) pero no mejoró sustancialmente la precisión.

---

## 4. Limitaciones observadas

1. **Sesgo hacia negativos:** el dataset MIMIC-CXR tiene alta proporción de estudios con impresiones negativas ("no acute", "no significant change"). El modelo internalizó ese sesgo.

2. **Hallazgos específicos difíciles:** el modelo falla sistemáticamente en devices (ET tubes, PICC lines, NG tubes) y hallazgos poco frecuentes (pneumothorax, metástasis). Necesitaría más ejemplos de esas clases.

3. **Falta de contexto clínico:** MIMIC-CXR está escrito para médicos con acceso a historia clínica y estudios previos. El modelo solo ve la imagen.

4. **Techo arquitectónico:** con BLIP-base y fine-tuning parcial, el overlap estimado máximo es ~20-25%. Para mejoras significativas se necesitaría encoder médico preentrenado o dataset 10x mayor.

---

## 5. Conclusión para el TP

Las captions con T=1.2 son **médicamente coherentes en vocabulario** pero **no son clínicamente fiables** (17% alucinaciones, 14% misses). Esto es el resultado esperado para un modelo generalista con 10k imágenes de entrenamiento.

Para el análisis de interpretabilidad del TP, este nivel de calidad es suficiente: las captions tienen vocabulario médico real que permite analizar qué partes de la imagen activa cada término (cross-attention por token). La limitación diagnóstica es en sí misma un resultado reportable.
