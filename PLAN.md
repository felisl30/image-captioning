# Plan de trabajo

## Estado actual

### Completado
- `src/data/` — utils, split_generator, dataset, dataloader implementados y verificados
- `src/models/blip_loader.py` — carga desde disco o HuggingFace, sanity check
- `src/interpretability/cross_attention.py` — `eval_and_extract_cross_att` (via forward hook, compatible con transformers 5.x), `merge_subword_attentions`. Limitación documentada: pesos post-softmax son uniformes sobre 576 tokens.
- `src/interpretability/cross_att_logits.py` — `extract_cross_att_logits` con logits Q·K pre-softmax. Pipeline principal de interpretabilidad. ⚠️ Pendiente validación del profesor.
- `src/interpretability/encoder_attention.py` — auto-atención CLS→patches del encoder ViT, saliencia global no word-specific.
- `src/interpretability/gradcam.py` — `compute_gradcam`, helpers, wrapper, TokenTarget
- `src/visualization/heatmap.py` — `overlay_heatmap`, superposición de heatmap sobre imagen
- `notebooks/00_exploracion_dataset.ipynb` — exploración, splits generados, forward pass verificado
- `models/blip_base/` — modelo descargado y cacheado en disco

### Bloqueado — esperando al profesor
El uso de logits Q·K (`cross_att_logits.py`) como señal de interpretabilidad para el informe está pendiente de confirmación del profesor. Hasta tener respuesta, los notebooks 02 y 04 no pueden cerrarse. Si el profesor aprueba → integrar con `docs/cross_att_logits_integracion.md`. Si no → usar Grad-CAM + encoder attention.

### Pendiente antes de continuar
Los splits en `data/splits/` fueron generados con **200 muestras** de smoke test, no con el dataset completo (30.633 filas). Hay que regenerarlos con el dataset completo para obtener train≈15.000, val≈1.500, test≈1.000. El notebook 00 tiene el código — descomentar `load_mimic_dataset` y volver a correr `generate_splits`. **No tocar `selected_indices.json`.**

---

## Pasos siguientes en orden

### 1. Regenerar splits con dataset completo
**Dónde:** `notebooks/00_exploracion_dataset.ipynb`  
**Qué hacer:** descomentar la celda de `load_mimic_dataset` (carga completa), comentar la celda de streaming con N_SAMPLES=200, correr `generate_splits` de nuevo.  
**Resultado esperado:** `data/splits/train_indices.json` con 10.000 índices, `data/selected_indices.json` con 30 índices del test set real.  
**Importante:** una vez regenerados, no volver a tocarlos.

---

### 2. Notebook 02 — baseline radiografías (Parte 2)
**Archivo a crear:** `notebooks/02_baseline_radiografias.ipynb`  
**Qué hace:** corre BLIP base sobre las 30 radiografías de `selected_indices.json` y guarda captions y heatmaps baseline.

Pasos dentro del notebook:
1. Cargar modelo base desde `models/blip_base/`
2. Para cada imagen en `selected_indices.json`:
   - Generar caption con `model.generate()`
   - Extraer mapas de interpretabilidad (señal a definir según respuesta del profesor: `extract_cross_att_logits` o Grad-CAM)
3. Guardar en `outputs/parte2_baseline/`

**⚠️ No arrancar hasta tener respuesta del profesor** sobre qué señal de cross-attention usar.

---

### 3. ~~Implementar `src/visualization/heatmap.py`~~ ✓ Completo

`overlay_heatmap` ya está implementado en `src/visualization/heatmap.py`.

---

### 4. Implementar `src/models/finetuner.py`
**Qué hace:** fine-tuning de BLIP sobre el split de train con `findings` como target.

Funciones mínimas:
- `train_epoch(model, dataloader, optimizer, device)` — un epoch de entrenamiento
- `evaluate(model, dataloader, device)` — val loss
- `finetune(model, processor, train_loader, val_loader, epochs, lr, output_dir)` — loop completo, guarda checkpoint del mejor val loss

**Dónde se corre:** en Kaggle o GCP con GPU. El notebook 03 llama a este módulo.

---

### 5. Notebook 03 — fine-tuning (GPU)
**Archivo a crear:** `notebooks/03_finetuning.ipynb`  
**Qué hace:** llama a `finetuner.py` con los splits reales.

Parámetros sugeridos:
- `text_col="findings"` (target más rico)
- `epochs=3`, `lr=1e-5`, `batch_size=8`
- Guardar checkpoint en `models/blip_finetuned/best/`

**Este notebook se sube a Kaggle/GCP tal cual — no requiere cambios locales.**

---

### 6. Notebook 04 — análisis post fine-tuning (Parte 3)
**Archivo a crear:** `notebooks/04_analisis_postft.ipynb`  
**Qué hace:** exactamente lo mismo que el notebook 02 pero cargando el modelo desde `models/blip_finetuned/best/`.

Las mismas 30 radiografías, el mismo pipeline. Guarda en `outputs/parte3_finetuned/`.

---

### 7. Implementar `src/metrics/nlg_metrics.py`
**Qué hace:** calcula BLEU, CIDEr, METEOR sobre los captions generados.

Función principal:
- `compute_nlg_metrics(captions_dict, references_dict) -> dict` — recibe dicts índice→texto, devuelve scores

Usa `pycocoevalcap`. Los captions se comparan contra `impression` (no `findings`) como referencia.

---

### 8. Notebook 05 — resultados y métricas
**Archivo a crear:** `notebooks/05_resultados_y_metricas.ipynb`  
**Qué hace:** tabla comparativa base vs fine-tuneado, figura central 3×3 (imagen / cross-att / gradcam) para casos representativos.

---

## Orden recomendado

```
1. Regenerar splits            ← prerequisito de todo
2. heatmap.py                  ← necesario para visualizar en notebook 02
3. Notebook 02 (baseline)      ← Parte 2 del trabajo
4. finetuner.py                ← prerequisito del notebook 03
5. Notebook 03 (GPU)           ← fine-tuning
6. Notebook 04 (post-FT)       ← Parte 3 del trabajo
7. nlg_metrics.py              ← prerequisito del notebook 05
8. Notebook 05 (resultados)    ← cierre del análisis
```

## Lo que NO falta
- Pipeline de datos: completo y verificado
- Carga del modelo: completa
- Interpretabilidad: `cross_att_logits.py`, `encoder_attention.py`, `gradcam.py` implementados (decisión sobre cuál usar en informe: pendiente de profesor)
- Visualización: `heatmap.py` completo
- Splits: generados (necesitan regenerarse con dataset completo antes del fine-tuning)
