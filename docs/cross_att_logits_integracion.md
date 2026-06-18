# Integración de `cross_att_logits.py` en el pipeline

## Qué hace

`src/interpretability/cross_att_logits.py` extrae mapas de atención por palabra
usando los **logits pre-softmax** (Q·K^T / √d) en lugar de los pesos post-softmax.

La función principal es `extract_cross_att_logits`, que tiene la misma firma de
entrada y salida que `eval_and_extract_cross_att` de `cross_attention.py`, por
lo que se puede usar como drop-in replacement en cualquier notebook o pipeline.

---

## Uso básico

```python
from src.interpretability.cross_att_logits import extract_cross_att_logits

results = extract_cross_att_logits(
    model, processor, inputs, num_batch=1,
    layer_idx=9,        # capa con mayor variabilidad entre palabras (diagnóstico)
    head_reduction="max",
)

# Misma estructura que eval_and_extract_cross_att:
caption = results[0]["caption"]          # "a group of people sitting at a table eating"
maps    = results[0]["maps"]             # [(palabra, array(24,24)), ...]
```

---

## Dónde integrarlo en los notebooks del proyecto

### Notebook 02 — `baseline_radiografias.ipynb` (Parte 2)

```python
from src.interpretability.cross_att_logits import extract_cross_att_logits
from src.visualization.heatmap import overlay_heatmap

# Reemplazar la llamada a eval_and_extract_cross_att:
results = extract_cross_att_logits(model, processor, inputs, num_batch=len(images))

for result in results:
    for word, heatmap in result["maps"]:
        overlay = overlay_heatmap(image, heatmap, alpha=0.55, colormap="jet")
        # guardar / mostrar overlay
```

### Notebook 04 — `analisis_postft.ipynb` (Parte 3, post fine-tuning)

Misma llamada que arriba pero sobre `model_finetuned`. La comparación antes/después
queda:

```python
maps_base = extract_cross_att_logits(model_base, processor, inputs, num_batch=1)
maps_ft   = extract_cross_att_logits(model_ft,   processor, inputs, num_batch=1)
```

---

## Parámetros relevantes

| Parámetro | Default | Cuándo cambiar |
|---|---|---|
| `layer_idx` | 9 | Si los mapas se ven uniformes, probar 8 o 7 |
| `head_reduction` | `"max"` | `"mean"` si los mapas son demasiado ruidosos |

---

## Qué reportar al comparar con softmax

| | Softmax (`eval_and_extract_cross_att`) | Q·K logits (`extract_cross_att_logits`) |
|---|---|---|
| Qué captura | Pesos post-softmax (prob. que suman 1) | Afinidad semántica Q·K antes de normalizar |
| Interpretación | Distribución de atención real del modelo | Proxy de "intención" espacial por palabra |
| Problema | Aplasta diferencias sobre 576 tokens | No es el cómputo que el modelo ejecuta |
| Cuándo usar | Para describir el mecanismo interno | Para interpretabilidad espacial por palabra |

**Qué decir en el informe:** los mapas de Q·K logits muestran que el decoder de
BLIP sí desarrolla preferencias espaciales diferenciadas por palabra a nivel de
similitud Q·K, pero el softmax sobre 576 patches comprime esas diferencias hasta
hacerlas visualmente indistinguibles en los pesos de atención. Los mapas de logits
son proxies de interpretabilidad, no la distribución de atención exacta.

---

## Relación con `cross_attention.py`

`cross_att_logits.py` reutiliza `merge_subword_attentions` de `cross_attention.py`
para agrupar subwords con `##`. No duplica esa lógica.

`eval_and_extract_qk_logits` en `cross_attention.py` es la versión de exploración
usada en el notebook de debug. `extract_cross_att_logits` en `cross_att_logits.py`
es la versión limpia para el pipeline de producción (notebooks 02 y 04).
