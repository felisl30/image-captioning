# Teacher Forcing para Interpretabilidad en BLIP

Este documento explica cómo y por qué se usa teacher forcing en la pipeline de
interpretabilidad comparada, qué lo diferencia del entrenamiento y de la inferencia
normal, y cómo garantiza que las activaciones capturadas son las mismas que se
produjeron durante la generación original.

---

## El problema de base: métodos que generaban su propia caption

Antes del refactor (commit `a0682b8`), cada método de interpretabilidad llamaba
internamente a `model.generate()`:

```python
# Cross-attention
out = model.generate(**inputs, output_attentions=True, ...)

# QK-logits
out = model.generate(**inputs, ...)

# Grad-CAM
generated_ids = model.generate(**inputs, max_new_tokens=40)
```

Con sampling estocástico (T=1.2, top-p), cada llamada a `generate()` puede
producir una caption distinta. Entonces:

- Cross-attention explicaba la caption A: *"a chest x-ray with bilateral effusion"*
- QK-logits explicaba la caption B: *"chest radiograph showing pleural effusion"*
- Grad-CAM explicaba la caption C: *"a chest x-ray showing bilateral pleural"*

Comparar los heatmaps entre métodos era inválido: no se aislaba la diferencia
entre métodos, sino que se mezclaba con la diferencia entre captions. Cualquier
métrica (Pearson, IoU, coseno) medía dos cosas a la vez.

---

## La solución: separar generación de explicación

El flujo nuevo tiene dos pasos explícitos:

```
PASO 1 — generar la caption una sola vez
  generated_ids = model.generate(image)
  # = [BOS, "a", "chest", "x-ray", "showing", "effusion", EOS]

PASO 2 — todos los métodos explican ESA caption fija
  cross_att  = eval_and_extract_cross_att(..., generated_ids=generated_ids)
  qk_logits  = extract_cross_att_logits(...,  generated_ids=generated_ids)
  gradcam    = compute_gradcam(...,            generated_ids_list=[generated_ids])
```

Resultado: todos los heatmaps corresponden a los mismos tokens, token a token,
y las métricas de comparación miden solo la diferencia entre métodos.

---

## Qué es teacher forcing

**Teacher forcing** es dar al decoder los tokens ya conocidos como input, en vez
de dejar que los genere él mismo.

En generación autoregresiva normal, el modelo produce los tokens uno a uno:

```
Paso 1: imagen + [BOS]                         → predice "a"
Paso 2: imagen + [BOS, "a"]                    → predice "chest"
Paso 3: imagen + [BOS, "a", "chest"]           → predice "x-ray"
Paso 4: imagen + [BOS, "a", "chest", "x-ray"] → predice "showing"
...
```

Son N forwards secuenciales. Cada uno depende del token generado en el paso
anterior.

Con teacher forcing se da toda la secuencia de una vez y el transformer la
procesa en paralelo:

```python
decoder_input = generated_ids[:, :-1]
# = [BOS, "a", "chest", "x-ray", "showing", "effusion"]
# (sin el EOS final, porque el decoder predice "qué sigue" en cada posición)

model(
    pixel_values=image,
    input_ids=decoder_input,
    output_attentions=True,
)
```

El **causal mask** garantiza que la posición t solo puede ver las posiciones
anteriores, igual que en generación:

```
Posición 2 ("chest"):
  Puede ver:    [BOS, "a", "chest"]       ← idéntico al paso 3 de generación
  No puede ver: ["x-ray", "showing", ...] ← bloqueado por causal mask
```

Por eso las activaciones en la posición t son matemáticamente idénticas a las
que se produjeron durante generación. No es una aproximación.

---

## Cómo BLIP puede recibir imagen y texto a la vez

BLIP es un modelo encoder-decoder:

- El **encoder** (ViT) procesa la imagen → produce 577 vectores, uno por patch.
- El **decoder** (transformer de texto) genera la caption token a token y, en
  cada paso, atiende a los features de la imagen vía **cross-attention**.

El decoder puede recibir imagen y texto simultáneamente. En teacher forcing le
das los dos:

```python
model(
    pixel_values=...,    # imagen: lo que el encoder ya procesó
    input_ids=...,       # caption fija: input del decoder
)
```

El decoder no "genera" nada en este forward — procesa la secuencia dada y
produce logits y activaciones para cada posición.

---

## Los tres modos de uso del forward (y cómo se combinan)

En PyTorch hay dos dimensiones completamente independientes:

**`model.train()` vs `model.eval()`** — controla comportamientos estocásticos
internos: Dropout (¿se apagan neuronas al azar?) y BatchNorm. No tiene nada
que ver con si los pesos se actualizan.

**`torch.no_grad()` vs gradiente activo** — controla si PyTorch construye el
grafo computacional para poder hacer backward. No tiene nada que ver con
train/eval.

Combinadas:

| | `no_grad` | gradiente activo |
|---|---|---|
| **`eval()`** | inferencia normal | Grad-CAM |
| **`train()`** | raro | entrenamiento normal |

Para los tres métodos de interpretabilidad:

| Método | Modo del modelo | Gradientes | Por qué |
|---|---|---|---|
| Cross-attention | `eval()` | `no_grad` | Solo captura activaciones via hook |
| QK-logits | `eval()` | `no_grad` | Solo captura Q y K via hook |
| Grad-CAM | `eval()` | activos | Necesita `d(logit_t)/d(activaciones_ViT)` |

---

## Cómo captura activaciones cada método

### Cross-attention y QK-logits: hooks de PyTorch

Antes de correr el forward, se registra una función que se ejecuta
automáticamente cuando la activación pasa por una capa:

```python
def hook_attn(module, module_input, output):
    captured_attn.append(output[1])   # [valores, pesos_att] → captura pesos

attn_hook = layer.crossattention.register_forward_hook(hook_attn)
```

Luego un solo `model(...)` procesa toda la secuencia. La capa de
cross-attention emite un tensor `(B, heads, T, 577)` — T posiciones de la
caption × 577 patches de la imagen — que el hook captura. Se re-parte por token:

```python
attn_steps = [attn_all[:, :, i:i+1, :] for i in range(T)]
```

Un forward, T heatmaps.

### Grad-CAM: N forwards con gradiente activo

Grad-CAM necesita `d(logit_token_t) / d(activaciones_ViT)`, lo que requiere
un backward por cada token. No se puede colapsar en un solo forward:

```python
for t in range(len(token_ids)):
    # prefix hasta el token t (teacher forcing token a token)
    input_ids_t = generated_ids[:, :t + 2]

    with GradCAM(model=wrapper, target_layers=[...]) as cam:
        grayscale_cam = cam(input_tensor=pixel_values, targets=[TokenTarget(token_id=t)])
        #  ↑ internamente: forward → backward → gradientes → heatmap
```

Para una caption de N tokens visibles, Grad-CAM hace N forwards + N backwards.

---

## Por qué no hay mismatch con el teacher forcing

En entrenamiento, el teacher forcing usa **ground truth del dataset**. El modelo
puede predecir tokens distintos a los que recibe como input:

```
Input del decoder (teacher): [BOS, "a", "chest", "x-ray"]    ← caption humana
Output del modelo:           ["a",  "scan", "of",  "the"]    ← lo que predijo
                                       ↑
                              mismatch — exposure bias
```

Este es el problema conocido como **exposure bias**: el modelo aprende
condicionado a inputs perfectos y en inferencia nunca los ve.

Acá **no hay mismatch**, y es por diseño. El "teacher" no es una caption
humana — son los tokens que el propio modelo generó:

```python
# Paso 1: el modelo genera su propia caption
generated_ids = model.generate(image)

# Paso 2: esos mismos tokens se usan como input
model(pixel_values=image, input_ids=generated_ids[:, :-1])
```

El contexto que ve el modelo en cada posición t durante teacher forcing es
`[BOS, t1, ..., t_{t-1}]` — exactamente el mismo que habría visto durante
generación en el paso t. Las activaciones son idénticas por construcción.

---

## El token en cada posición es input, no output

Este es el punto más importante para entender por qué el mecanismo es válido.

Intuitivamente parece raro: "¿cómo puedo dar el token t si el modelo no lo
generó todavía?" La respuesta es que en el decoder el token t es el **input**
en la posición t, no el output. El output en la posición t es el logit para
el token t+1.

```
Posición t en el decoder:
  Input:   token t   (dado por teacher forcing)
  Proceso: self-attention sobre [t0..tt] + cross-attention sobre imagen
  Output:  logit para t+1
```

El modelo no "sabe" si el token vino de generación o de teacher forcing. Solo
procesa el input que recibe y produce activaciones. Las activaciones en la
posición t dependen del contexto `[t0, ..., tt]`, que es el mismo en ambos
casos porque se usan los tokens generados originalmente.

---

## Resumen del flujo completo

```
1. generate_caption_best_of_n(model, image)
   └── samplea N captions con T=1.2 / top-p
   └── elige la mejor por score (riqueza médica, sin repetición)
   └── devuelve generated_ids = [BOS, t1, t2, ..., tN, EOS]

2. extract_all_methods(model, processor, image, generated_ids)
   ├── cross_attention:
   │     registra hook → 1 forward con [BOS..tN] → captura (T,577) → split por token
   ├── qk_logits:
   │     registra hooks Q y K → 1 forward → captura Q·Kᵀ/√d → split por token
   └── gradcam:
         loop t=0..N:
           forward con [BOS..tt] + backward → d(logit_t)/d(ViT_feats) → heatmap_t

3. compute_spatial_metrics(results)
   └── alinea mapas token a token entre métodos
   └── calcula Pearson, coseno, MSE, top-k IoU por par de métodos
   └── devuelve tabla de comparación
```

Los tres métodos explican la misma secuencia de tokens, generada una sola vez,
con el modelo congelado (`model.eval()`, sin `optimizer.step()`). Las
activaciones capturadas son las mismas que se produjeron durante la generación
original.
