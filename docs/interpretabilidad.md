# Interpretabilidad en BLIP — Cross-Attention y Grad-CAM

## Por qué dos métodos

Existe un debate activo en la literatura sobre si los pesos de atención son explicaciones fiables de lo que un modelo "mira" (Jain & Wallace, 2019). La respuesta que da este proyecto es pragmática: en lugar de adoptar uno solo, se usan dos mecanismos de distinta naturaleza y se mide si coinciden.

- **Cross-attention** viene de dentro del modelo, es parte de su computación normal.
- **Grad-CAM** viene de afuera, usa los gradientes para inferir qué zonas importan para una predicción concreta.

Si ambos apuntan a las mismas regiones de la radiografía, hay más confianza en la explicación. Si divergen, hay que reportarlo y discutirlo. La correlación espacial entre los dos mapas es una de las métricas cuantitativas del trabajo.

---

## Cross-Attention

### Qué es

En el decoder de BLIP, cada capa tiene un mecanismo de **cross-attention**: cuando el modelo genera un token (una palabra del caption), ese token "hace una pregunta" (query) a los 576 vectores de patch del encoder visual (keys y values). Los pesos de atención resultantes, uno por patch, indican cuánto contribuyó cada región visual a generar ese token.

El shape del tensor de cross-attention para un paso de generación es:

```
(batch, n_heads, 1, 576)
```

Donde:
- `n_heads = 12` en ViT-Base
- `576 = 24 × 24` patches de 16×16 px sobre una imagen de 384×384

### Qué representa

Un valor alto en la posición `[i]` del mapa significa que el patch `i` fue muy relevante para generar la palabra actual. Si el modelo genera "consolidation" y el mapa de atención tiene los pesos altos concentrados en la zona inferior derecha del pulmón, eso sugiere que el modelo está mirando una región anatómicamente coherente.

### Limitaciones conocidas

- Los pesos de atención son distribuciones de probabilidad que **no necesariamente reflejan causalidad**. Un patch puede recibir atención alta simplemente porque es el "más parecido" a la query, no porque sea causalmente responsable de la predicción.
- Las capas intermedias del decoder agregan información de formas no lineales. Los pesos de la última capa son los más directamente interpretables, pero pueden perder información de capas anteriores. Attention Rollout (módulo `rollout.py`) intenta combinar todas las capas.
- En decodificación autoregresiva, el shape real de la salida de `generate()` es una tupla anidada `(pasos, capas, batch, heads, 1, 576)`. Hay que saber indexarla correctamente.

### Cómo se extrae (conceptualmente)

HuggingFace permite pasar `output_attentions=True` a `generate()`. El resultado incluye los tensores de cross-attention de todas las capas y todos los pasos de generación. Para obtener el mapa de la palabra `t` de la capa `L`:

1. Indexar por paso `t` y capa `L` para obtener un tensor `(batch, heads, 1, 576)`.
2. Promediar sobre los `heads` para obtener `(1, 576)`.
3. Hacer reshape a `(24, 24)`.
4. Upscale bilineal a `(384, 384)` para superponerlo sobre la imagen original.

La función `extract_cross_attention` en `src/interpretability/cross_attention.py` encapsula este proceso y devuelve un diccionario `{palabra: heatmap_384x384}`.

---

## Grad-CAM

### Qué es

Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al., 2017) es una técnica de explicación por gradientes. La idea base es: si se quiere saber qué zonas de la imagen contribuyeron a una predicción concreta, se puede mirar el gradiente de esa predicción con respecto a las activaciones de una capa intermedia. Las zonas donde el gradiente es alto son las que más importan.

Originalmente fue diseñado para CNNs, donde la última capa convolucional produce un mapa espacial de activaciones. En ViT no hay convoluciones, pero el principio se adapta.

### Adaptación a ViT

El encoder visual de BLIP es un ViT. En lugar de un mapa de feature maps 2D, produce una secuencia de 576 vectores de patch (más el token CLS). La librería `pytorch-grad-cam` incluye un `vit_reshape_transform` genérico, pero **BLIP requiere un transform custom** porque usa 384×384 (no 224×224):

1. Toma la secuencia de tokens de salida de una capa del encoder `(batch, 577, 768)`.
2. Descarta el token CLS (posición 0).
3. Hace reshape de los 576 tokens restantes a `(batch, 768, 24, 24)` — una grilla 2D de activaciones.

Esto permite usar el algoritmo estándar de Grad-CAM como si fuera una CNN.

### Qué capa usar como target

La capa target es la última capa del encoder ViT — típicamente la `LayerNorm` final de la última capa de atención. Es la representación más elaborada de la imagen antes de que los vectores pasen al decoder. Las capas tempranas capturan bordes y texturas; las últimas, semántica de alto nivel.

### Qué se computa como "clase objetivo"

Grad-CAM fue diseñado para clasificación, donde la "clase" es un logit de salida. En generación de texto, la clase objetivo es el **logit correspondiente al token que se quiere explicar**. Para cada palabra del caption, se computa un Grad-CAM distinto usando ese logit como señal.

Este ajuste requiere definir un target custom en `pytorch-grad-cam` que apunte al logit del token correcto en el vocabulario.

### Qué representa

Un valor alto en una zona significa que las activaciones del encoder en esa región tenían un gradiente positivo grande respecto al logit de la palabra objetivo. En otras palabras: si esa zona se activa más, el modelo es más propenso a predecir esa palabra.

### Limitaciones conocidas

- Grad-CAM es **posthoc** y aproximado. El gradiente local puede ser engañoso si la función de loss tiene regiones planas o la activación es saturada.
- Combinar Grad-CAM con decodificación autoregresiva requiere cuidado: para obtener el gradiente del token `t`, se necesita hacer un forward pass específico que llegue a ese token, no un `generate()` completo.
- El `vit_reshape_transform` estándar de `pytorch-grad-cam` asume 197 tokens (CLS + 196 patches, para 224×224). **BLIP usa 577 tokens (CLS + 576 patches)**. Hay que usar el transform custom `blip_vit_reshape_transform` definido en `gradcam.py` que reorganiza a 24×24.

---

## Comparación entre los dos métodos

| Dimensión | Cross-Attention | Grad-CAM |
|---|---|---|
| Origen | Interno al modelo (pesos de atención) | Externo (gradientes posthoc) |
| Qué mide | Cuánto atendió el decoder a cada patch al generar una palabra | Qué patches del encoder tienen mayor influencia causal sobre una predicción |
| Granularidad | Por palabra, por capa, por head | Por palabra, por capa target |
| Costo computacional | Bajo — sale gratis del forward pass | Medio — requiere backward pass por palabra |
| Sensibilidad al fine-tuning | Alta — los pesos de atención cambian directamente | Alta — el encoder cambia y los gradientes también |
| Referencia | BLIP paper (Li et al., 2022) | Selvaraju et al. (2017), Chefer et al. (2021) para ViT |

---

## Métricas de comparación entre mapas

Para cuantificar si los dos métodos coinciden y si cambian con el fine-tuning, se usa la **correlación de Pearson** entre los dos mapas aplanados a 576 valores (resolución de patches, no de píxeles). Se calcula por imagen y por palabra, y se agrega como media y desviación estándar.

- **Correlación alta** (~0.7–1.0): los dos mecanismos explican las mismas regiones.
- **Correlación baja o negativa**: los mecanismos divergen — hay que discutir por qué.

Si la correlación media aumenta después del fine-tuning, sugiere que el ajuste alinea la atención del decoder con las activaciones del encoder en regiones médicamente relevantes.

---

## Attention Rollout (módulo opcional)

Attention Rollout (Abnar & Zuidema, ACL 2020) es una alternativa para agregar la información de todas las capas del decoder en lugar de usar solo la última. La idea es que la atención de la capa `L+1` pasa a través de la atención de la capa `L`, y se puede propagar hacia atrás mediante multiplicación matricial recursiva.

Es útil si los heatmaps de la última capa resultan muy difusos o con poco contraste. No reemplaza a cross-attention ni a Grad-CAM: es una forma distinta de agregar la señal de atención ya disponible.

---

## Orden de implementación recomendado

1. Implementar `cross_attention.py` primero — es más directo y sirve para calibrar el pipeline visual con COCO (Parte 1).
2. Implementar `gradcam.py` — más complejo por la necesidad del target custom para generación.
3. Implementar `rollout.py` solo si los heatmaps de cross-attention resultan difusos en las primeras pruebas.
4. Implementar `spatial_metrics.py` — depende de que ambos mapas estén funcionando.
