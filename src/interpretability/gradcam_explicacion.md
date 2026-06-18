# Explicación de `gradcam.py`

## El problema que resuelve

Grad-CAM responde la pregunta: **¿qué regiones de la imagen causaron que el modelo generara una palabra específica?**

A diferencia de cross-attention (que lee los pesos internos del decoder), Grad-CAM mide influencia real: si una región de la imagen desapareciera, ¿cuánto cambiaría el logit de ese token? Lo mide con gradientes.

---

## Por qué Grad-CAM no funciona directo en BLIP

Grad-CAM fue diseñado para clasificación: hay un forward pass, hay un logit de clase, se calcula el gradiente, listo.

BLIP tiene dos problemas:

**Problema 1 — generación autoregresiva:** `model.generate()` genera tokens uno a uno en un loop interno y no expone los logits intermedios de forma que Grad-CAM pueda engancharse.

**Problema 2 — ViT no tiene feature maps 2D:** las CNNs producen tensores `(B, C, H, W)` que Grad-CAM puede interpretar espacialmente. ViT produce una secuencia de tokens `(B, 577, 768)`. Grad-CAM no sabe cómo interpretar eso como un mapa espacial.

Los tres componentes del archivo resuelven estos dos problemas.

---

## Componente 1: `blip_vit_reshape_transform` (línea 159)

Resuelve el **Problema 2**.

```python
def blip_vit_reshape_transform(tensor, height=24, width=24):
    result = tensor[:, 1:, :]                                     # (B, 576, 768)
    result = result.reshape(result.size(0), height, width, result.size(2))  # (B, 24, 24, 768)
    result = result.transpose(2, 3).transpose(1, 2)               # (B, 768, 24, 24)
    return result
```

La capa target del ViT (última `layer_norm1`) produce un tensor `(B, 577, 768)`:
- posición `0` → token CLS (resumen global, no tiene posición espacial)
- posiciones `1..576` → los 576 patch tokens, cada uno representa un parche 16×16 de la imagen

El transform:
1. Descarta el CLS (`[:, 1:, :]`) → `(B, 576, 768)`
2. Reordena los 576 tokens en una grilla 24×24 → `(B, 24, 24, 768)`
3. Transpone a formato `(B, C, H, W)` que espera pytorch-grad-cam → `(B, 768, 24, 24)`

**Por qué `height=24` y no `14`:** el transform estándar de pytorch-grad-cam asume imágenes 224×224 (14×14 patches). BLIP usa 384×384 → 24×24 patches. Por eso hay un transform custom.

---

## Componente 2: `TokenTarget` y `BLIPGradCAMWrapper` (líneas 167–190)

Resuelven el **Problema 1** juntos.

### `BLIPGradCAMWrapper`

```python
class BLIPGradCAMWrapper(torch.nn.Module):
    def __init__(self, model, input_ids):
        self.model = model
        self.input_ids = input_ids

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values, input_ids=self.input_ids)
        return outputs.logits  # (batch, seq_len, vocab_size)
```

pytorch-grad-cam necesita un modelo con una interfaz simple: recibe un tensor, devuelve un tensor. `BlipForConditionalGeneration` devuelve un dataclass, no un tensor plano. El wrapper lo resuelve.

El truco clave está en `input_ids`: en lugar de usar `model.generate()` (que hace decodificación autoregresiva), pasamos los tokens **ya generados** directamente. Esto se llama **teacher forcing** — le decimos al modelo "asumí que hasta acá generaste estos tokens, ahora hacé un forward pass normal". El modelo devuelve los logits para todas las posiciones en un solo forward pass.

Cada instancia del wrapper tiene `input_ids` fijo correspondiente a los tokens hasta el paso `t`. Por eso se crea un wrapper nuevo por cada token en el loop de `_gradcam_single`.

### `TokenTarget`

```python
class TokenTarget:
    def __init__(self, token_id):
        self.token_id = token_id

    def __call__(self, model_output):
        if model_output.dim() == 2:
            return model_output[-1, self.token_id]
        return model_output[:, -1, self.token_id]
```

pytorch-grad-cam necesita saber **qué escalar optimizar** para calcular los gradientes. Nos interesa el logit del token `token_id` en la **última posición** (`-1`).

¿Por qué `-1`? Porque en el wrapper, `input_ids` contiene `[BOS, t0, t1, ..., t]`. El modelo predice el siguiente token en cada posición; la última posición predice el token `t+1`, que es exactamente el token que queremos explicar.

**Gotcha con pytorch-grad-cam 1.5.x:** al computar el loss internamente, la librería hace:

```python
loss = sum([target(output) for target, output in zip(targets, outputs)])
```

`zip(targets, outputs)` itera sobre `outputs` **a lo largo de la dimensión de batch**. Si `outputs` tiene shape `(1, seq_len, vocab_size)`, en cada iteración `output = outputs[0]` — un tensor **2D** `(seq_len, vocab_size)`, ya sin la dimensión de batch.

Si `TokenTarget` asume siempre shape 3D y hace `model_output[:, -1, token_id]` sobre un tensor 2D, falla con `IndexError: too many indices for tensor of dimension 2`. Este error es suprimido silenciosamente por el context manager de `GradCAM` (su `__exit__` retorna `True`), lo que produce un `UnboundLocalError` posterior al no haberse asignado el resultado.

El fix detecta el número de dimensiones: `model_output[-1, token_id]` si es 2D, `model_output[:, -1, token_id]` si es 3D.

---

## Componente 3: `_gradcam_single` y `compute_gradcam` (líneas 193–267)

### `_gradcam_single` — el loop por token

```python
for t in range(n):
    input_ids_t = generated_ids[:, :t + 2]          # BOS + tokens 0..t
    wrapper = BLIPGradCAMWrapper(model, input_ids_t)
    target = TokenTarget(token_id=token_ids[t].item())

    with GradCAM(model=wrapper, target_layers=target_layer, ...) as cam:
        grayscale_cam = cam(input_tensor=pixel_values, targets=[target])
```

Por cada token generado `t`:
1. Construye un wrapper con los tokens hasta `t` como teacher forcing
2. Crea un `TokenTarget` para el token `t`
3. Corre Grad-CAM: hace un forward pass a través del wrapper, calcula el gradiente de `logits[:, -1, token_id]` respecto a las activaciones de `layer_norm1`, pondera los canales de activación por esos gradientes, y produce un mapa 24×24

El resultado `grayscale_cam` tiene shape `(1, 384, 384)` — pytorch-grad-cam lo upscalea automáticamente al tamaño de la imagen de entrada. Lo reducimos de vuelta a `(24, 24)` para que sea comparable con los mapas de cross-attention:

```python
heatmap = torch.tensor(grayscale_cam[0]).unsqueeze(0).unsqueeze(0)  # (1, 1, 384, 384)
heatmap = F.interpolate(heatmap, size=(24, 24), ...).squeeze().numpy()  # (24, 24)
```

El uso de `with GradCAM(...) as cam:` (context manager) es importante: registra hooks en la capa target al entrar y los elimina al salir. Sin esto, los hooks se acumulan en cada iteración del loop y los gradientes quedan contaminados.

Al final del loop, `subword_maps` es una lista de arrays `(24, 24)`, uno por subword token. Se pasa a `merge_subword_attentions` (importada de `cross_attention.py`) para promediar los subwords que forman la misma palabra.

### `compute_gradcam` — el loop por imagen

Itera sobre las imágenes del batch y llama a `_gradcam_single` para cada una. Procesa de a una imagen a la vez porque Grad-CAM requiere un backward pass por token — hacer batching real de Grad-CAM sería mucho más complejo y el beneficio en este contexto es mínimo.

---

## Resumen del flujo completo

```
imagen PIL
    └─► processor → pixel_values (1, 3, 384, 384)
    └─► model.generate() → generated_ids [BOS, t0, t1, ..., tN, EOS]

para cada token t en [t0..tN]:
    input_ids_t = [BOS, t0, ..., t]         ← teacher forcing hasta t
    wrapper.forward(pixel_values)
        → logits (1, len(input_ids_t), vocab_size)
    TokenTarget extrae logits[:, -1, token_id_t]
    Grad-CAM calcula gradiente de ese escalar
        respecto a layer_norm1 del ViT
    reshape_transform convierte (1, 577, 768) → (1, 768, 24, 24)
    Grad-CAM produce mapa (24, 24)

merge_subword_attentions agrupa ## tokens
    → { "cardiomegaly": (24,24), "no": (24,24), ... }
```

---

## Diferencia con cross-attention

| | Cross-attention | Grad-CAM |
|---|---|---|
| Qué mide | Pesos de atención del decoder sobre los patches | Gradiente del logit del token respecto al encoder |
| Dónde ocurre | En el decoder, capturado via forward hook | En el encoder ViT, via backward pass |
| Interpretación | "A qué patches miró el decoder al generar este token" | "Qué patches del encoder más influyeron en ese logit" |
| Cómo se extrae | Hook en `crossattention.self` de la última capa del decoder | Teacher forcing + GradCAM sobre `layer_norm1` del ViT |
| Costo | Barato (una captura por token dentro de `generate()`) | Caro (un forward + backward por token) |
