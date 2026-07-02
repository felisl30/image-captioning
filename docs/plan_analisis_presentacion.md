# Plan de análisis — Presentación preliminar del TP

**Fecha:** 2026-07-02  
**Contexto:** todos los datos están en local. No se necesita GPU ni VM para nada de lo que sigue.

---

## Estructura sugerida de la presentación

```
1. Introducción y pregunta de investigación
2. Metodología
3. Resultados — análisis general (descriptivo)
4. Resultados — análisis específico (cuantitativo y visual)
5. Conclusiones y trabajo futuro
```

---

## Sección 1 — Introducción y pregunta de investigación

**Qué explicar:**

El punto de partida es que BLIP es un modelo de captioning generalista entrenado sobre imágenes naturales (COCO, etc.). No sabe nada de radiología. La pregunta es si con fine-tuning sobre pares (radiografía, impresión clínica) de MIMIC-CXR el modelo no solo aprende a hablar el lenguaje médico sino también a *mirar* las regiones clínicamente relevantes.

**Los tres resultados posibles** (todos válidos, ninguno es "malo"):

- **A:** captions mejoran Y la atención visual se dirige a zonas clínicas relevantes
- **B:** captions mejoran pero la atención no cambia
- **C:** la atención cambia pero no hacia zonas relevantes

**Por qué es interesante:** separa aprendizaje lingüístico de aprendizaje visual en un único modelo encoder-decoder. No es obvio que el fine-tuning sobre texto adapte también la visión.

---

## Sección 2 — Metodología

**Qué explicar:**

### Modelo y datos
- BLIP-base (`Salesforce/blip-image-captioning-base`)
- Dataset: MIMIC-CXR (`itsanmolgupta/mimic-cxr-dataset`), 30.600 pares imagen-impresión
- Fine-tuning con 5k y 10k pares, 3 épocas, target: campo `impression`
- Evaluación sobre 25 radiografías fijas (`data/visual_test_indices.json`)

### Métodos de interpretabilidad
Tres métodos, todos sobre la misma caption generada (teacher forcing):

| Método | Qué captura | Naturaleza |
|---|---|---|
| Cross-attention post-softmax | Pesos de atención normalizados | Atención |
| QK-logits (pre-softmax) | Logits Q·K^T/√d sin normalizar | Atención |
| Grad-CAM | Gradientes respecto a activaciones ViT | Gradientes |

Punto clave metodológico: los tres métodos reciben los mismos `generated_ids` — explican exactamente la misma secuencia de tokens, lo que hace la comparación entre métodos válida.

### Decodificación
- Greedy decoding colapsa post fine-tuning (65-80% captions idénticas)
- Solución: sampling con T=1.2, top_p=0.95, best-of-3 seeds
- Los análisis de interpretabilidad usan siempre T=1.2

---

## Sección 3 — Análisis general (descriptivo)

Esta sección es **descriptiva**. No entrar en análisis específicos ni métricas detalladas. El objetivo es dar una imagen de conjunto de qué pasó.

### 3.1 Qué cambió en las captions

- **Base:** captions genéricas, vocabulario no médico, errores anatómicos
  - Ejemplo: *"a male lung with an pneumonia in the stomach"*
- **ft5k:** vocabulario médico real, sintaxis a veces incoherente
  - Ejemplo: *"no appendedemental pneumonia or pap given clinical correlation. pulmonary edema with anomary indicator."*
- **ft10k:** vocabulario médico más limpio, mayor especificidad
  - Ejemplo: *"no appendedemental pneumonia or pulmonary edema with bibasilar atelectasis."*

El número de tokens médicamente relevantes por caption aumenta con el fine-tuning.

### 3.2 Qué cambió en los heatmaps

**Descripción visual general** (sin métricas aquí):

- En el modelo base, los tres métodos producen mapas con poca estructura — la atención no se concentra en zonas específicas.
- Post fine-tuning, los mapas de QK-logits y Grad-CAM muestran foco más claro en campos pulmonares, región cardíaca y bases.
- Los tres métodos no coinciden entre sí: cross-attention post-softmax es visualmente plano (limitación conocida del softmax sobre 576 patches), QK-logits muestra más contraste, Grad-CAM muestra activaciones más difusas.

### 3.3 El mode collapse

- Tras el fine-tuning, greedy decoding produce la misma caption para el 65-80% de las imágenes.
- Caption dominante: *"no acute cardiopulmonary process."*
- Causa: sesgo estadístico de MIMIC-CXR amplificado por greedy (ver `analisis/01`).
- Solución: temperatura T=1.2 → 93% captions únicas, 83% vocabulario médico.
- **Punto importante:** el collapse es un artefacto de decodificación, no de aprendizaje. Los heatmaps post fine-tuning son coherentes incluso cuando las captions colapsan.

---

## Sección 4 — Análisis específico (cuantitativo y visual)

Esta sección tiene el análisis concreto. Requiere generar las figuras y tablas desde los datos locales.

### 4.1 Comparación de captions — tabla cuantitativa

**Qué mostrar:** tabla con métricas de calidad de captions por modelo.

| Métrica | base | ft5k | ft10k |
|---|---|---|---|
| % vocabulario médico | ~0% | ~75% | ~83% |
| Unique ratio | ~15% | ~93% | ~95% |
| Overlap con referencia (keywords) | ~0% | ~12% | ~15% |
| Tokens relevantes promedio por caption | ~4 | ~9 | ~8 |

Los datos de tokens relevantes están en `outputs/notebook_comparativo/summary.csv`
(columna `n_tokens_relevant`). El resto viene de `analisis/02_captions_10k.md`.

**Cómo generarlo:**
```python
import csv
rows = list(csv.DictReader(open("outputs/notebook_comparativo/summary.csv")))
# agrupar por model_tag, promediar n_tokens_relevant
```

### 4.2 Comparación de heatmaps — figura central

**Qué mostrar:** grilla con la misma imagen y el mismo token médico para los tres modelos.

```
        base          ft5k          ft10k
img     [original]
token   [heatmap]   [heatmap]    [heatmap]
```

**Tokens a mostrar:** "effusion", "edema", "atelectasis", "pneumonia" — los que aparezcan en ft5k y ft10k pero no en base (o con heatmap completamente distinto).

**Cómo generarlo:** usar `compare_models_for_token()` de `docs/analisis_post_vm.md`.
Elegir 3-4 imágenes donde el contraste base vs ft sea visualmente claro.

**Fuente de datos:** `outputs/notebook_comparativo/arrays/idx_<NNN>/`

### 4.3 Post-softmax vs QK-logits — por qué usamos logits

**Qué mostrar:** figura lado a lado del mismo token con los dos métodos de atención.

- post_softmax → mapa casi uniforme (≈1/576 por patch)
- qk_logits → mapa con zonas de alta activación delimitadas

**Mensaje:** los pesos post-softmax no son informativos sobre 576 patches (Abnar & Zuidema, 2020). Los logits Q·K revelan la estructura de atención real.

**Cómo generarlo:**
```python
z = np.load("outputs/notebook_comparativo/arrays/idx_0731__ft10k__post_softmax.npz", allow_pickle=True)
z2 = np.load("outputs/notebook_comparativo/arrays/idx_0731__ft10k__qk_logits.npz", allow_pickle=True)
# graficar el mismo índice de token en ambos
```

### 4.4 Métricas de concordancia entre métodos

**Qué mostrar:** tabla o gráfico de barras del IoU top-10% por modelo y par de métodos.

| Par | base | ft5k | ft10k |
|---|---|---|---|
| post_softmax vs qk_logits | 0.667 | 0.815 | 0.827 |
| post_softmax vs gradcam | 0.047 | 0.038 | 0.059 |
| qk_logits vs gradcam | 0.053 | 0.047 | 0.063 |

**Mensaje:**
- Post-softmax y QK-logits se vuelven más consistentes con el fine-tuning (IoU +24%). La atención se consolida.
- Grad-CAM permanece desacoplado de los métodos de atención en todos los modelos — son mecanismos de naturaleza distinta y no se espera que coincidan.

**Fuente:** `outputs/notebook_comparativo/metrics/spatial_summary.csv` (ya calculado).

### 4.5 Análisis por token — semántica espacial

**Qué mostrar:** para una misma caption con ft10k, mostrar que tokens distintos activan zonas distintas.

Ejemplo esperado para *"bilateral pleural effusions with atelectasis"*:
- "pleural" → bases pulmonares laterales
- "effusion" → ángulos costofrénicos
- "atelectasis" → bases posteriores

**Por qué importa:** demuestra que la atención es semánticamente específica, no un mapa difuso uniforme. Justifica el pipeline de interpretabilidad token a token.

**Cómo generarlo:** usar `plot_medical_heatmaps()` de `docs/analisis_post_vm.md` con un caso de ft10k que tenga ≥3 tokens médicos distintos.

---

## Sección 5 — Conclusiones y trabajo futuro

### Conclusión preliminar

Con los datos actuales el resultado se acerca a **escenario A parcial**:

- Las captions mejoran claramente post fine-tuning (vocabulario médico, especificidad, unique ratio).
- Los métodos de atención se vuelven más consistentes entre sí (IoU +24%), lo que indica que la atención se concentra más después del fine-tuning.
- **Pendiente de confirmar visualmente:** si esa mayor concentración apunta efectivamente a zonas clínicas relevantes (pulmones, pleura, mediastino) vs zonas irrelevantes (fondo, bordes).

### Trabajo futuro

1. **Validar clínicamente los heatmaps** — comparar las zonas activadas con anotaciones de hallazgos (no disponibles en MIMIC-CXR público, pero se pueden hacer manualmente sobre los 25 casos).
2. **Métricas NLG formales** — BLEU, CIDEr, METEOR sobre el test set completo.
3. **Arquitecturas más grandes** — BLIP-2, BioViL-T (específico de radiología), para comparar con el baseline.
4. **Fine-tuning más largo** — 10k imágenes fueron suficientes para el vocabulario pero no para la precisión diagnóstica.
5. **Attention Rollout** (Abnar & Zuidema, 2020) — alternativa a los logits Q·K para atención en transformers multicapa.

---

## Lista de tareas para la presentación

```
[ ] Tabla de captions: n_tokens_relevant por modelo (summary.csv)
[ ] Figura central: base vs ft5k vs ft10k para mismo token médico (arrays/)
[ ] Figura post-softmax vs qk_logits (arrays/)
[ ] Gráfico de barras IoU por modelo (spatial_summary.csv — ya calculado)
[ ] Figura análisis por token: distintos tokens, distintas zonas (arrays/ ft10k)
[ ] Selección de 3-4 imágenes representativas de los 25 casos
```

Todo se genera en CPU local desde `outputs/notebook_comparativo/`. No se necesita GPU.
