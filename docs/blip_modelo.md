# BLIP base — estructura del modelo

**Identificador HuggingFace:** `Salesforce/blip-image-captioning-base`  
**Clase:** `BlipForConditionalGeneration`  
**Guardado en:** `models/blip_base/`  
**Tamaño en disco:** ~855 MB (`model.safetensors`)

---

## Archivos en esta carpeta

| Archivo | Qué contiene |
|---|---|
| `model.safetensors` | Pesos del modelo (~473 tensores, 855 MB) |
| `config.json` | Arquitectura completa: dimensiones, capas, vocab size |
| `generation_config.json` | Parámetros por defecto de generación (max tokens, etc.) |
| `processor_config.json` | Configuración del preprocesador de imagen (resolución, normalización) |
| `tokenizer.json` | Vocabulario y reglas de tokenización (WordPiece, 30.524 tokens) |
| `tokenizer_config.json` | Metadatos del tokenizador (tokens especiales, clase) |

Para cargar: `BlipForConditionalGeneration.from_pretrained("models/blip_base")`.  
Para cargar el processor: `BlipProcessor.from_pretrained("models/blip_base")`.

---

## Arquitectura general

BLIP es un modelo encoder-decoder visual-lingüístico:

```
Imagen (384×384 RGB)
    ↓
[ Encoder visual: ViT-B/16 ]
    ↓ 576 vectores de patch (768 dim cada uno)
[ Decoder de texto: Transformer causal con cross-attention ]
    ↓
Caption generado token a token
```

---

## Encoder visual — ViT-B/16

| Parámetro | Valor |
|---|---|
| Tipo | Vision Transformer Base, patch 16×16 |
| Resolución de entrada | **384×384 px** |
| Tamaño de patch | 16×16 px |
| Patches por lado | 384 / 16 = **24** |
| Total de patch tokens | 24 × 24 = **576** |
| Token CLS | 1 (posición 0 de la secuencia) |
| Secuencia total al encoder | 577 tokens (CLS + 576 patches) |
| Dimensión de embedding | 768 |
| Número de bloques transformer | 12 |
| Cabezas de atención por bloque | 12 |
| Dimensión FFN interna | 3.072 |
| Dropout | 0.0 (sin dropout en inferencia ni fine-tuning) |

Cada bloque de atención del encoder procesa los 577 tokens (self-attention pura, sin cross-attention).

### Lo que importa para la interpretabilidad

Los 576 patch tokens de salida son los que el decoder va a "mirar" mediante cross-attention. Cada patch token corresponde a una región de 16×16 px en la imagen original, organizados en una grilla de 24×24. El token CLS **no** se pasa al decoder.

Para Grad-CAM: la capa target es `model.vision_model.encoder.layers[-1].layer_norm1`. Al aplicar `vit_reshape_transform`, los 576 tokens se reorganizan como una grilla (24, 24).

---

## Decoder de texto — Transformer causal

| Parámetro | Valor |
|---|---|
| Tipo | Transformer con self-attention causal + cross-attention |
| Número de bloques | 12 |
| Dimensión de embedding | 768 |
| Cabezas de atención | 12 (self-attention y cross-attention) |
| Dimensión FFN interna | 3.072 |
| Longitud máxima de secuencia | 512 tokens |
| Vocabulario | 30.524 tokens (WordPiece, igual que BERT) |
| `encoder_hidden_size` | 768 (dimensión de los patch tokens que recibe por cross-attention) |

Cada bloque del decoder tiene tres sublayers en orden:
1. **Self-attention causal** — cada token generado atiende a los tokens previos del caption.
2. **Cross-attention** — cada token generado atiende a los 576 patch tokens del encoder. Este es el mecanismo que mapea "qué miró el modelo al generar cada palabra".
3. **FFN** — red feed-forward posición a posición.

### Lo que importa para la interpretabilidad

El tensor de cross-attention en cada bloque del decoder tiene shape:

```
(batch, n_heads, T_generado, 576)
```

Donde `T_generado` es el número de tokens generados hasta ese punto. Al extraer la atención del **último bloque** (capa 11) para el paso de generación `t`:

```
out.cross_attentions[t][11]  →  shape: (1, 12, 1, 576)
```

Promediando sobre las 12 cabezas y haciendo reshape:

```
→ shape: (576,)  →  reshape(24, 24)  →  upscale bilineal  →  (384, 384)
```

---

## Estrategia de descongelamiento para fine-tuning

Ver `docs/finetuning.md` para el razonamiento completo. Resumen:

| Componente | Bloques | Estado | LR |
|---|---|---|---|
| ViT patch embedding + pos. emb | — | Congelado | — |
| ViT encoder | Bloques 1–8 | Congelado | — |
| ViT encoder | Bloques 9–12 | Descongelado | `5e-6` |
| Decoder | Bloques 1–4 | Congelado | — |
| Decoder | Bloques 5–12 + LM head | Descongelado | `1e-5` |

---

## Parámetros totales (aproximados)

| Componente | Parámetros |
|---|---|
| Encoder ViT (12 bloques) | ~86M |
| Decoder (12 bloques + embeddings) | ~110M |
| **Total** | **~196M** |

Con la estrategia de descongelamiento (4 bloques encoder + 8 bloques decoder), se entrenan aproximadamente **80–90M parámetros** (~45% del total).

---

## Tokens especiales del tokenizador

| Token | ID |
|---|---|
| `[PAD]` | 0 |
| `[BOS]` | 30522 |
| `[EOS]` | 2 |
| `[SEP]` | 102 |

El decoder genera tokens hasta encontrar `[EOS]` (id=2) o alcanzar `max_new_tokens`. En `MimicCXRDataset`, los tokens `[PAD]` se enmascaran con `-100` en `labels` para que no contribuyan al loss.
