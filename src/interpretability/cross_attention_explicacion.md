# Explicación de `cross_attention.py`

## El problema que resuelve

Cross-attention responde la pregunta: **¿a qué zonas de la imagen estaba "mirando" el decoder cuando generó una palabra específica?**

Durante la generación, el decoder de BLIP atiende a los patch tokens del encoder visual mediante capas de cross-attention. Los pesos de esa atención son directamente interpretables como un mapa espacial: un peso alto en el patch `(i, j)` significa que ese parche de la imagen contribuyó fuertemente a la generación de ese token.

---

## Por qué no se puede leer directo de `generate()` en transformers 5.x

En transformers ≤ 4.x, llamar `generate(output_attentions=True)` devolvía un objeto `GenerateEncoderDecoderOutput` con un campo `cross_attentions`:

```python
out.cross_attentions[paso][capa]  # → (batch, n_heads, 1, 576)
```

A partir de **transformers 5.x**, el output unificado de `generate()` es `GenerateDecoderOnlyOutput` — incluso para modelos encoder-decoder como BLIP — y el campo `cross_attentions` ya no existe. El campo `attentions` que sí existe contiene las self-attentions del decoder con shape `(batch, n_heads, 1, 1)`, no los pesos de cross-attention al encoder visual.

La solución es enganchar el módulo directamente con un **forward hook**.

---

## La arquitectura del decoder de BLIP

El decoder de texto de BLIP está basado en BERT-base. Li et al. (2022) tomaron los pesos preentrenados de BERT y le agregaron capas de cross-attention en cada bloque transformer para inyectar las features del encoder visual (ViT). Por eso el módulo se llama `text_decoder.bert`.

Estructura relevante:

```
model.text_decoder
  └── bert
        └── encoder
              └── layer[0..11]           ← 12 bloques transformer
                    ├── attention         ← self-attention del texto
                    ├── crossattention    ← cross-attention con el encoder visual  ← nos interesa
                    └── intermediate / output
```

La última capa (`layer[-1]`) tiene la representación más rica semánticamente, igual que en Grad-CAM.

---

## Componente 1: el forward hook

```python
target = model.text_decoder.bert.encoder.layer[-1].crossattention.self
hook = target.register_forward_hook(hook_fn)
```

`register_forward_hook` registra una función que se llama automáticamente **después** de cada llamada al `forward()` de ese módulo. El hook recibe `(module, input, output)`.

El output de `BlipTextSelfAttention.forward()` cuando `output_attentions=True` es una tupla:

```
(context_layer, attention_probs)
```

donde:
- `context_layer`: shape `(batch, 1, 768)` — la representación del token después de atender al encoder
- `attention_probs`: shape `(batch, n_heads, 1, 577)` — los pesos de atención ← **esto es lo que queremos**

El shape `577 = 1 CLS + 576 patches`. En transformers 5.x el CLS del encoder visual se incluye en las keys del cross-attention.

El hook se registra antes de `generate()` y se elimina con `hook.remove()` en un bloque `finally` para garantizar limpieza aunque falle la generación.

---

## Componente 2: la captura y el reshape

```python
def hook_fn(module, input, output):
    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
        attn = output[1]
        if attn.shape[-1] == 577:   # confirma que es cross-att al encoder, no self-att
            captured.append(attn.detach())
```

El filtro `shape[-1] == 577` es importante: durante la generación, el mismo módulo podría capturar self-attentions (shape variable) si los hooks se propagan. Solo nos quedamos con los tensores que tienen exactamente 577 keys (el encoder visual completo).

**Alineación token ↔ captura:** `generate()` llama al decoder una vez por token generado. Cada llamada dispara el hook exactamente una vez. Por lo tanto `captured[i]` corresponde al token `i` de la secuencia generada — sin necesidad de índice adicional.

Para procesar:

```python
attn = captured[i]              # (batch, n_heads, 1, 577)
attn_patches = attn[..., 1:]    # descarto CLS → (batch, n_heads, 1, 576)
attn_map = attn_patches[batch].mean(0)[0].numpy().reshape(24, 24)
```

1. `attn[..., 1:]` — descarta la posición 0 (CLS), queda con los 576 patches
2. `.mean(0)` sobre `n_heads` — promedio de las 12 cabezas de atención
3. `[0]` — quita la dimensión de query (siempre 1, el token actual)
4. `.reshape(24, 24)` — los 576 patches en la grilla 24×24 que corresponde a la imagen 384×384

---

## Componente 3: `merge_subword_attentions`

BERT tokeniza con WordPiece: las palabras largas se dividen en subwords marcados con `##`. Por ejemplo, `"cardiomegaly"` se tokeniza como `["cardiac", "##omegaly"]`.

```python
def merge_subword_attentions(tokens, attention_maps):
    for token, attn_map in zip(tokens, attention_maps):
        if token.startswith("##"):
            merged_maps[-1].append(attn_map)   # agrega al grupo anterior
        else:
            merged_tokens.append(token)
            merged_maps.append([attn_map])
    
    maps_list = [(token, np.mean(maps, axis=0)) for token, maps in zip(...)]
```

Los mapas de los subwords se promedian para obtener un único mapa `(24, 24)` por palabra visible. El output usa **lista de tuplas** (no dict) para preservar el orden y permitir palabras repetidas (e.g., "no" puede aparecer varias veces en un caption médico).

---

## Flujo completo

```
imagen PIL
    └─► processor → pixel_values (1, 3, 384, 384)

hook registrado en layer[-1].crossattention.self

model.generate(**inputs, output_attentions=True, max_new_tokens=40)
    ├── paso 0: decoder genera token t0
    │     └── hook captura attn_weights (1, 12, 1, 577) → captured[0]
    ├── paso 1: decoder genera token t1
    │     └── hook captura attn_weights (1, 12, 1, 577) → captured[1]
    └── ...

hook removido

para cada i en range(n_tokens):
    attn = captured[i][..., 1:]         # (1, 12, 1, 576) — sin CLS
    mapa = attn[0].mean(0)[0]           # (576,)
    mapa_2d = mapa.reshape(24, 24)      # (24, 24)

merge_subword_attentions(tokens, mapas)
    → {"caption": "no acute cardiopulmonary process",
       "maps": [("no", (24,24)), ("acute", (24,24)), ...]}
```

---

## Diferencia con Grad-CAM

| | Cross-attention | Grad-CAM |
|---|---|---|
| Qué mide | Pesos de atención del decoder sobre los patches | Gradiente del logit del token respecto al encoder |
| Dónde se engancha | Decoder (última capa de cross-attention) | Encoder ViT (última `layer_norm1`) |
| Interpretación | "A qué patches miró el decoder" | "Qué patches causaron ese logit" |
| Costo | Barato — una captura por token dentro de `generate()` | Caro — un forward + backward por token |
| Normalización | Valores crudos, normaliza `heatmap.py` | pytorch-grad-cam normaliza a [0,1] |

---

## Referencia

La extracción via forward hook es una práctica estándar para acceder a representaciones intermedias cuando la API de alto nivel no las expone. El cambio en transformers 5.x que motivó este enfoque se discute en el issue de HuggingFace relacionado con la unificación de outputs de `generate()`.
