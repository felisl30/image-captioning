# Análisis 03 — Hallazgos para el Paper

**Fecha:** 2026-06-25  
**Scope:** Síntesis de todos los hallazgos hasta ahora, organizados para el informe IEEE.

---

## 1. Resultado principal del TP

El experimento responde la pregunta de investigación: *¿es suficiente el fine-tuning para que BLIP no solo aprenda a hablar el lenguaje médico sino también a mirar las regiones clínicamente relevantes?*

**Resultado observado: tipo A parcial.**  
- Los heatmaps de cross-attention y Grad-CAM post fine-tuning muestran foco en regiones clínicamente relevantes (campos pulmonares, región cardíaca, bases) → la atención visual mejoró.  
- Las captions con greedy decoding colapsaron → el lenguaje no mejoró con decodificación estándar.  
- Con temperatura T=1.2 el lenguaje también mejora → el modelo internalizó vocabulario médico, el problema era de decodificación.

**Argumento central:** el fine-tuning adaptó tanto la visión como el lenguaje. El collapse es un artefacto de decodificación, no un fallo de aprendizaje.

---

## 2. Hallazgos por módulo

### 2.1 Mode collapse (D1 + D2)

- **Magnitud:** con greedy, 65-80% de captions son idénticas dependiendo del checkpoint.
- **Época de instalación:** epoch_2 es donde el collapse se vuelve severo (entropía promedio: 1.91 → 1.07).
- **Mecanismo (D2):** cascada de certeza. En step 0 el token "no" gana con p=0.18-0.22 sobre alternativas plausibles (interval, mild, increased). A partir de step 2 ("card") la entropía cae a <1.0 y el modelo queda comprometido.

```
step=0  "no"    p=0.21  entropy=4.38  (distribución diversa, "mild"/"interval" competitivos)
step=1  "acute" p=0.56  entropy=2.21
step=2  "card"  p=0.84  entropy=0.85
step=3  "##io"  p=0.995 entropy=0.04  (prácticamente determinista)
```

- **Causa raíz:** sesgo estadístico del corpus. MIMIC-CXR tiene alta frecuencia de impresiones negativas. Fine-tuning con 5-10k ejemplos amplificó ese sesgo. Greedy siempre elige el ganador marginal y la cascada hace el resto.
- **Solución:** temperatura T=1.2 con top-p=0.95 → unique_ratio 0.93-0.99, 75-83% vocabulario clínico específico.

**Referencia:** Jain & Wallace (NAACL 2019) — crítica a la atención como explicación; análogo aquí: el modelo no está "equivocado", el mecanismo de decodificación amplifica un sesgo marginal.

### 2.2 Atención visual (D3 — cross-attention Q·K logits)

- Los mapas de cross-attention post-softmax son planos (≈1/576 para todos los patches) — confirmado por diagnóstico exhaustivo.
- Los logits Q·K pre-softmax producen mapas distintos y coherentes por token.
- Post fine-tuning: los mapas muestran foco en campos pulmonares, región cardíaca y bases pulmonares — regiones clínicamente relevantes para radiografía de tórax.
- La atención cambia por token generado: tokens como "edema", "effusion", "atelectasis" activan zonas distintas de la imagen.

**Referencia:**  
- Abnar & Zuidema (ACL 2020) — Attention Rollout: limitaciones de la atención post-softmax como explicación.  
- Jain & Wallace (NAACL 2019) — por qué los pesos post-softmax no son explicativos.

### 2.3 Grad-CAM

- Funciona correctamente con `blip_vit_reshape_transform` (577 tokens → descarta CLS → grilla 24×24).
- Post fine-tuning: activa regiones de pulmón y área cardíaca de forma coherente.
- Comparado con cross-attention logits: Grad-CAM muestra activaciones más difusas (región amplia), cross-attention logits es más esparsa y específica por token.

**Referencia:**  
- Selvaraju et al. (ICCV 2017) — Grad-CAM.  
- Chefer et al. (ICCV 2021) — Grad-CAM en Transformers: por qué se necesita el reshape transform.

### 2.4 Calidad de captions (10k + T=1.2)

- Overlap de keywords médicos con referencia: 15.2% (vs 0% del modelo base).
- Alucinaciones: 17% (inventa findings en estudios normales).
- Misses: 14% (dice normal cuando hay findings).
- Limitación: ceiling arquitectónico de ~20-25% con BLIP-base y fine-tuning parcial.

---

## 3. Figuras clave para el paper

| Figura | Descripción | Dónde está |
|---|---|---|
| **Fig. 1** | Comparativa cross-attention (base vs fine-tuned) por token, misma imagen | pendiente — ver `04_analisis_pendiente.md` |
| **Fig. 2** | Grad-CAM (base vs fine-tuned), misma imagen | pendiente |
| **Fig. 3** | Cross-attention logits vs post-softmax — ilustrar por qué se usan logits | pendiente |
| **Fig. 4** | Curva de entropía por step (D2) — cascada de certeza | datos en `outputs/mode_collapse_debug/` |
| **Fig. 5** | Tabla S1: unique_ratio vs top_pct por estrategia de decoding | datos en `outputs/decoding_sampling/` |

---

## 4. Argumento para el paper (estructura sugerida)

**Sección de resultados:**

1. *Baseline (BLIP base):* captions no médicas ("a chest xray with a large open chest"), heatmaps sin coherencia clínica. Confirma que el dominio no está cubierto por el preentrenamiento.

2. *Post fine-tuning — lenguaje:* collapse con greedy (65-80%). Análisis D2 explica el mecanismo. Temperatura T=1.2 lo resuelve: captions con vocabulario clínico real, 15% overlap con referencias clínicas.

3. *Post fine-tuning — visión:* cross-attention logits y Grad-CAM muestran foco en campos pulmonares y región cardíaca. El modelo aprendió a "mirar" radiología, incluso cuando las captions colapsan con greedy.

4. *Interpretación:* el collapse no implica que el modelo no aprendió. La atención visual es médicamente coherente independientemente de la estrategia de decodificación. Esto sugiere que el componente visual del fine-tuning fue más robusto que el lingüístico.

**Limitaciones a reportar:**
- Precisión diagnóstica limitada (10k imágenes, fine-tuning parcial).
- Collapse resuelto con heurística de decodificación, no con mejora del entrenamiento.
- Los logits Q·K como señal de interpretabilidad son una aproximación — ver pendiente de validación del profesor (`docs/respuesta_profesor_cross_attention.txt`).

---

## 5. Referencias a citar

| Tema | Referencia |
|---|---|
| BLIP | Li et al., ICML 2022 |
| ViT | Dosovitskiy et al., ICLR 2021 |
| Grad-CAM | Selvaraju et al., ICCV 2017 |
| Grad-CAM en Transformers | Chefer et al., ICCV 2021 |
| Atención como explicación (crítica) | Jain & Wallace, NAACL 2019 |
| Attention Rollout | Abnar & Zuidema, ACL 2020 |
| Fine-tuning captioning médico | Nicolson et al., AIIM 2023 |
