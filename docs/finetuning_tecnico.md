# Fine-tuning técnico — BLIP sobre MIMIC-CXR

Documento de referencia para entender las decisiones de implementación del fine-tuning.
Para la guía de ejecución paso a paso ver `guia_finetuning_gcp.md`.

---

## 1. Qué se fine-tunea y por qué

BLIP-base tiene dos componentes: un **encoder visual ViT-B/16** y un **decoder de texto BERT-like**. El modelo base fue preentrenado sobre imágenes naturales (COCO, conceptual captions), no radiografías.

### Estrategia de congelamiento

```
Encoder ViT (12 capas):
  - Capas 0–7 → CONGELADAS  (features visuales genéricas: bordes, texturas)
  - Capas 8–11 → entrenables (features de alto nivel, dominio-específicas)

Decoder BERT (12 capas):
  - Capas 0–3 → CONGELADAS  (sintaxis, tokens comunes)
  - Capas 4–11 → entrenables (semántica, vocabulario médico)
  - LM head → entrenable    (mapeo a tokens del vocabulario)
```

**Resultado:** 128M / 224M parámetros entrenables (~57%).

**Por qué no entrenar todo:** con un dataset de 5k–15k imágenes, entrenar los 224M parámetros produciría overfitting. Las capas bajas del encoder ya aprendieron features visuales útiles (detección de bordes, estructuras) que son transferibles desde imágenes naturales.

**Por qué no entrenar solo el LM head:** el encoder necesita aprender a prestarle atención a las zonas clínicamente relevantes (campos pulmonares, silueta cardíaca). Si solo entrenamos el decoder, el encoder sigue "mirando" donde miraría en fotos de perros.

---

## 2. Optimizador — AdamW con LR diferencial

```python
optimizer = AdamW([
    {"params": encoder_params, "lr": 5e-6},   # encoder más conservador
    {"params": decoder_params, "lr": 1e-5},   # decoder más agresivo
], weight_decay=0.01)
```

El encoder recibe un LR más bajo porque sus pesos ya son buenos (pre-entrenados) y solo necesitan ajuste fino. El decoder parte de una inicialización más lejana del dominio médico y necesita aprender más.

**Weight decay 0.01:** regularización estándar de AdamW. Penaliza pesos grandes sin afectar el bias.

---

## 3. Scheduler — warmup lineal

```
Steps 0 → N_warmup : LR sube linealmente de 0 al LR máximo
Steps N_warmup → N_total : LR baja linealmente a 0
```

Con `warmup_ratio=0.05` y 5k/batch_8 = 625 steps por época × 3 épocas = 1875 steps totales:
- Warmup: primeros ~94 steps (~7% de la primera época)
- Decay: el resto del entrenamiento

**Por qué warmup:** al inicio los gradientes son grandes y ruidosos. Empezar con LR bajo evita que el modelo "destruya" las representaciones preentrenadas en los primeros batches.

---

## 4. Gradient accumulation

Permite simular un batch size grande cuando la VRAM no alcanza:

```
batch_size=4, grad_accum_steps=2 → batch efectivo = 8
```

En cada micro-batch, la loss se divide por `grad_accum_steps` antes del `.backward()`. Los gradientes se acumulan en los tensores `.grad` de los parámetros. El `optimizer.step()` solo se llama cada N micro-batches.

**Impacto en el scheduler:** el número de optimizer steps es `batches_por_época / grad_accum_steps`, no `batches_por_época`. El scheduler se inicializa con el número correcto de optimizer steps para que el warmup y el decay estén bien calibrados.

Con L4 y batch_size=8, gradient accumulation no es necesario (`GRAD_ACCUM_STEPS=1`). Solo activarlo si hay OOM.

---

## 5. Mixed Precision (AMP)

**Problema:** FP32 es preciso pero lento. FP16 es 2–4x más rápido en Tensor Cores pero los gradientes pueden hacerse tan pequeños que se redondean a 0 (*underflow*).

**Solución AMP:**

```
Forward pass  → FP16  (rápido, Tensor Cores)
Loss backward → loss × scale_factor → FP16 backward
               → gradientes × (1/scale_factor) → FP32
Optimizer step → FP32 (preciso)
Pesos         → FP32 (siempre)
```

El `GradScaler` ajusta el `scale_factor` automáticamente:
- Si detecta inf/nan en gradientes → reduce el factor y **saltea** el optimizer step
- Si hay muchos steps sin problemas → aumenta el factor

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    loss = model(**inputs).loss / accum_steps

scaler.scale(loss).backward()

if step % accum_steps == 0:
    scaler.unscale_(optimizer)          # convierte gradientes a FP32
    clip_grad_norm_(params, max_norm=1.0)
    scaler.step(optimizer)              # solo ejecuta si no hay inf/nan
    scaler.update()                     # ajusta scale_factor
    scheduler.step()
    optimizer.zero_grad()
```

**Ganancia en L4:** ~1.5–2x throughput, ~40% menos VRAM.
**En CPU o MPS:** AMP se deshabilita automáticamente (GradScaler solo funciona con CUDA).

---

## 6. Métricas de entrenamiento

### 6.1 Val loss (cross-entropy)

```
loss = -mean( log P(token_t | imagen, tokens_0..t-1) )
```

Se calcula con teacher forcing: el decoder recibe los tokens correctos como input. Mide qué tan bien el modelo predice el siguiente token dado el prefijo correcto. Rango típico:
- BLIP base sobre radiografías (sin fine-tuning): ~6.5–7.5
- Después de 3 épocas con 5k imágenes: ~3.5–5.0
- Después de 3 épocas con 15k imágenes: ~2.5–4.0

### 6.2 Perplexidad

```
perplexity = exp(val_loss)
```

Interpretación: "el modelo está tan confundido como si eligiera al azar entre N tokens en cada posición".

| Etapa | Val loss típico | Perplexidad |
|---|---|---|
| BLIP base sin FT | 6.8 | ~900 |
| Después de FT 5k | 4.2 | ~67 |
| Después de FT 15k | 3.5 | ~33 |

La perplexidad baja mucho más dramáticamente que la loss, lo que la hace más intuitiva para reportar.

### 6.3 Token accuracy

```python
preds = logits.argmax(dim=-1)     # (batch, seq_len)
mask  = labels != -100            # excluir padding
acc   = (preds[mask] == labels[mask]).float().mean()
```

Mide el porcentaje de tokens donde el argmax del logit coincide con el token correcto, bajo teacher forcing.

**Limitación:** no es la accuracy de generación real. Durante `generate()` el modelo usa sus propias predicciones previas (no el ground truth), así que el error se propaga. La token accuracy bajo teacher forcing es siempre mayor que la calidad real de los captions generados.

**Para qué sirve:** detectar si el modelo está aprendiendo vocabulario médico. Una subida de 20% → 55% en token accuracy indica que el modelo dejó de predecir tokens genéricos ("the", "a", "of") y empezó a predecir términos como "cardiomegaly", "no acute", "pulmonary".

Rango típico: 15–25% (base sin FT) → 50–65% (después de FT).

### 6.4 BLEU / CIDEr / METEOR (evaluación final)

Requieren generación real (no teacher forcing). Se calculan **después** del entrenamiento en el notebook 05. No se computan durante el training loop porque son 10–50x más lentos que calcular la loss.

Se usan las referencias `impression` del split de test.

---

## 7. Early stopping

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    guardar checkpoint "best/"
    epochs_sin_mejora = 0
else:
    epochs_sin_mejora += 1
    if epochs_sin_mejora >= patience:
        break  # early stopping
```

Con `patience=2` y `epochs=3`, el entrenamiento puede cortarse en la época 2 si la época 1 fue mejor. Esto es normal — no significa que el modelo sea malo, sino que ya convergió.

---

## 8. Checkpoints guardados

```
models/blip_finetuned_{n_tag}/
├── epoch_1/          ← checkpoint al final de la época 1
├── epoch_2/          ← checkpoint al final de la época 2 (si llegó)
├── epoch_3/          ← checkpoint al final de la época 3 (si llegó)
├── best/             ← copia del epoch con menor val_loss
└── training_history.json   ← métricas por época
```

El `training_history.json` tiene por época:
- `train_loss`, `val_loss`, `best_val_loss`
- `val_token_acc`, `val_perplexity`
- `trainable_params`, `total_params`, `trainable_pct`
- `grad_accum_steps`

Solo `best/` es necesario para el notebook 04 (análisis post fine-tuning). Los `epoch_N/` se pueden borrar para ahorrar espacio.

---

## 9. Posibles resultados y su interpretación

| Escenario | train_loss | val_loss | Interpretación |
|---|---|---|---|
| Normal | baja | baja | El modelo está aprendiendo y generaliza |
| Overfitting | baja | sube desde época 1 | Dataset demasiado chico o LR alto — probar con más datos o más regularización |
| No aprende | estable ~6.5 | estable ~6.5 | LR demasiado bajo, freeze incorrecto o dataset corrupto |
| Loss = NaN | NaN | — | LR demasiado alto o imagen corrupta — verificar con `--max-train-batches 1` |

Con 5k imágenes y 3 épocas, lo más probable es el escenario **normal** con alguna tendencia leve a overfitting en la época 3. El early stopping debería manejarlo.
