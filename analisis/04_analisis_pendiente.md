# Análisis 04 — Pendientes

**Fecha:** 2026-06-25  
**Estado:** estos análisis están por hacer. La VM está cerrada.

---

## 1. Cross-attention: antes vs después del softmax

**Por qué importa:**  
El paper de Jain & Wallace (2019) y Abnar & Zuidema (2020) argumentan que los pesos post-softmax no son explicativos porque el softmax aplana las diferencias. Con 576 patches, post-softmax cada patch recibe ≈1/576 ≈ 0.0017 de peso — visualmente plano.

**Qué mostrar:**
- Mapa post-softmax para un token dado → imagen casi uniforme, sin estructura
- Mapa Q·K pre-softmax (logits) para el mismo token → zonas de alta activación claramente delimitadas
- Figura lado a lado: post-softmax vs logits Q·K para el token "edema" o "effusion"

**Cómo generarlo:**  
`src/interpretability/cross_att_logits.py` ya extrae los logits Q·K.  
Para post-softmax usar `output_attentions=True` en `model.generate()` (o forward hook sobre la capa de atención antes de la aplicación de softmax).

**Mensaje para el paper:**  
> "Los pesos de atención post-softmax son visualmente uniformes debido al aplaneamiento del softmax sobre 576 patches (Fig. X izquierda). Los logits Q·K^T/√d sin softmax revelan la estructura de atención real del modelo (Fig. X derecha), consistente con hallazgos de Abnar & Zuidema (2020) sobre las limitaciones del rollout de atención."

---

## 2. Comparativa antes/después del fine-tuning

**Por qué es la figura central del TP:**  
Es la respuesta directa a la pregunta de investigación. Misma imagen, mismo token → atención del modelo base vs modelo fine-tuneado.

**Qué mostrar (figura 3×2 o 3×3):**
- Fila 1: imagen original
- Fila 2: cross-attention logits Q·K — modelo BASE
- Fila 3: cross-attention logits Q·K — modelo FT
- (Opcional) Fila 4: Grad-CAM BASE vs Grad-CAM FT

Para tokens clínicamente informativos: "edema", "effusion", "atelectasis", "consolidation".

**Imágenes a usar:** `data/visual_test_indices.json` (25 imágenes fijas).  
Elegir 3-4 que tengan hallazgos claros (no estudios normales) para que el contraste sea visible.

**Cómo generarlo:**  
`archivos_ion/run_d3_heatmap_probe.py` ya hace esto — outputs en `outputs/` de la VM.  
Necesita la VM con GPU para Grad-CAM completo.

---

## 3. Grad-CAM — análisis completo

**Estado actual:** se corrió D3 con `--skip-gradcam` en sesión anterior. Los heatmaps de Grad-CAM que tenemos son de una corrida manual previa.

**Qué falta:**
- Correr D3 completo (con gradcam) sobre las 25 imágenes con ambos modelos (base + FT 10k)
- Comparar las activaciones: ¿el FT concentra más la activación en zonas pulmonares? ¿o la dispersa?

**Comando en VM:**
```
python archivos_ion/run_d3_heatmap_probe.py \
  --base-model-dir models/blip_base \
  --ft-model-dir models/blip_finetuned_10k/best \
  --indices data/visual_test_indices.json \
  --max-images 25 \
  --device cuda \
  --output-dir outputs/d3_full_10k
```

**Para el paper:**  
Grad-CAM y cross-attention logits deben aparecer como métodos complementarios:
- Grad-CAM: basado en gradientes, independiente del mecanismo de atención, más comparable con literatura CNN
- Cross-attention logits: específico de transformers, varía por token, permite análisis semántico más fino

Citar Chefer et al. (2021) para justificar el uso de Grad-CAM en ViT.

---

## 4. Análisis por token (semántico)

**Idea:** mostrar que distintos tokens de un mismo caption activan distintas regiones.

Ejemplo esperado para caption "bilateral pleural effusions with atelectasis":
- "bilateral" → ambos lados del tórax
- "pleural" → bases pulmonares laterales
- "effusion" → ángulos costofrénicos
- "atelectasis" → bases posteriores

**Cómo generarlo:**  
`extract_cross_att_logits()` ya devuelve `[(palabra, array_24x24), ...]` por caption.  
Armar una figura con subplots: una columna por token relevante.

**Nota:** solo posible con captions T=1.2 (los captions greedy tienen muy pocos tokens distintos para analizar).

---

## 5. Orden de ejecución sugerido (próxima sesión VM)

1. **D3 completo con 10k** (GPU, ~20 min):
   ```
   python archivos_ion/run_d3_heatmap_probe.py --ft-model-dir models/blip_finetuned_10k/best --indices data/visual_test_indices.json --max-images 25 --device cuda --output-dir outputs/d3_full_10k
   ```

2. **Figura post-softmax vs logits Q·K** (puede hacerse en notebook local, no necesita GPU):
   - Tomar una imagen de `visual_test_indices.json`
   - Correr forward pass con base y FT
   - Graficar ambos mapas lado a lado

3. **Figura comparativa base vs FT** (usar outputs de D3):
   - Seleccionar 3-4 imágenes con hallazgos claros
   - Armar grilla: original | cross-att BASE | cross-att FT | gradcam BASE | gradcam FT

4. **Análisis por token** (notebook local):
   - Elegir 2-3 captions con T=1.2 que tengan ≥3 tokens médicos distintos
   - Graficar atención por token
