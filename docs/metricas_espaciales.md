# Métricas espaciales para comparación de métodos de interpretabilidad

Este documento explica las 4 métricas implementadas en `src/metrics/spatial_metrics.py`
para comparar los heatmaps producidos por los tres métodos de explicabilidad:
post-softmax cross-attention, QK-logits y Grad-CAM.

---

## Qué se compara

Cada método produce un heatmap **24×24** por token de la caption (576 patches en total).
Las métricas comparan dos heatmaps del mismo token, generados por dos métodos distintos.
Para calcularlas, los mapas se aplanan a vectores de 576 valores.

Las comparaciones posibles son tres pares:

```
post_softmax  vs  qk_logits
post_softmax  vs  gradcam
qk_logits     vs  gradcam
```

---

## Las 4 métricas

### 1. Pearson (`pearson`)

```python
x = x - x.mean()
y = y - y.mean()
return dot(x, y) / (norm(x) * norm(y))
```

Mide si la **estructura relativa** de los dos mapas es similar: si los patches que
uno activa más que su promedio son los mismos que activa el otro. Rango [-1, 1].

**Por qué está:** es invariante a escala y offset. Si cross-attention produce valores
en [0, 1] y QK-logits produce valores en [0, 100], Pearson no se ve afectado.
Responde a la pregunta: "¿los dos métodos señalan la misma región aunque sus
magnitudes sean completamente distintas?"

**Limitación:** puede ser alto aunque los picos sean débiles y dispersos. Es sensible
a la forma global del mapa, no solo a las zonas con mayor activación.

---

### 2. Coseno (`cosine`)

```python
return dot(x, y) / (norm(x) * norm(y))
```

Mide el ángulo entre los dos vectores heatmap. Rango [0, 1] para valores no negativos
(que es el caso de heatmaps).

**Por qué está:** similar a Pearson pero sin restar la media. Eso lo hace sensible
a dónde están concentradas las activaciones en términos absolutos. Si un mapa tiene
activaciones difusas y el otro las concentra en pocos patches, Coseno lo detecta
mejor que Pearson. Trata el mapa como un vector desde el origen, no centrado.

**Diferencia con Pearson:** Pearson centra primero y mide correlación lineal.
Coseno mide dirección en el espacio vectorial original. En la práctica suelen
moverse juntos pero pueden divergir cuando hay diferencias de distribución.

---

### 3. MSE (`mse`)

```python
return mean((x - y) ** 2)
```

Error cuadrático medio entre los dos mapas, valor a valor. Sin rango acotado.

**Por qué está:** mide la **distancia absoluta** entre los mapas. A diferencia de
Pearson y Coseno, penaliza diferencias de magnitud. Dos mapas que señalan la misma
zona pero con intensidades muy distintas (0.1 vs 0.9) pueden tener Pearson alto
pero MSE alto también. Es un control de que los mapas no solo tienen la misma forma
sino valores comparables.

**Interpretación invertida:** MSE más bajo = más parecidos. Al contrario que las
otras tres métricas.

---

### 4. Top-k IoU (`top10_iou`) — la métrica de overlap

```python
k = max(1, int(round(0.10 * 576)))   # = 58 patches
idx_x = set(top_k_indices(x))
idx_y = set(top_k_indices(y))
return len(idx_x & idx_y) / len(idx_x | idx_y)
```

Toma el **top 10% de patches más activados** de cada mapa (los 58 con mayor valor
de 576) y calcula el IoU clásico entre esos dos conjuntos:

```
IoU = |intersección| / |unión|
```

Rango [0, 1]. 1 significa que los dos métodos coinciden exactamente en qué patches
consideran importantes.

**Por qué está:** es la métrica más directa para responder la pregunta central del
TP: "¿los métodos coinciden en qué región de la imagen mirar?"

**Por qué top 10% y no todo el mapa:** los heatmaps tienen mucho ruido de fondo —
la mayoría de los 576 patches tiene activaciones bajas. Comparar todo el mapa mezcla
señal con ruido. Quedarse con el top 10% fuerza a mirar solo dónde cada método
pone su atención real.

---

## Cómo se usan juntas

Cada métrica responde una pregunta distinta:

| Métrica | Pregunta que responde |
|---|---|
| Pearson | ¿La distribución espacial tiene la misma forma? |
| Coseno | ¿Los vectores apuntan en la misma dirección? |
| MSE | ¿Las magnitudes son similares además de la forma? |
| Top-k IoU | ¿Los patches que cada método considera "importantes" se solapan? |

Usarlas juntas permite distinguir casos sutiles: dos métodos pueden tener Pearson
alto (misma forma) pero IoU bajo (los picos exactos no coinciden), o Coseno alto
pero MSE alto (misma dirección pero escalas muy distintas).

---

## Resultados de las pruebas locales

Se corrió el pipeline con `models/blip_base` sobre una imagen local (`prueba1.jpeg`,
una imagen de comida) y se obtuvieron estos valores de referencia:

```
post_softmax vs qk_logits:
  pearson_mean = 0.38
  cosine_mean  = 0.28
  top10_iou    = 0.62   ← alto: comparten ~62% de los patches top

post_softmax vs gradcam:
  pearson_mean = -0.11
  cosine_mean  = 0.03
  top10_iou    = 0.009  ← casi ningún overlap

qk_logits vs gradcam:
  pearson_mean = -0.28
  cosine_mean  = 0.55
  top10_iou    = 0.006  ← casi ningún overlap
```

**Interpretación:** cross-attention y QK-logits se solapan bastante (IoU 0.62),
lo que tiene sentido porque ambos son variantes del mismo mecanismo de atención.
Grad-CAM está casi completamente desacoplado de los otros dos (IoU < 0.01), lo
que también es esperable: es un mecanismo de naturaleza diferente (gradientes
respecto a la imagen vs. pesos de atención del decoder).

Estos valores son sobre una imagen de prueba, no sobre radiografías. Los valores
reales sobre MIMIC-CXR pueden diferir.

---

## Qué no está implementado todavía

Las métricas actuales comparan **métodos entre sí** para el mismo modelo. No hay
todavía métricas que comparen el **mismo método entre modelos distintos**: por
ejemplo, "¿el finetuning hace que cross-attention mire más la zona de derrame
pleural en ft10k que en base?". Eso requeriría comparar heatmaps de `base` vs
`ft5k` vs `ft10k` para la misma imagen y el mismo token, usando las mismas
funciones de `spatial_metrics.py`. No es difícil de agregar pero no está
implementado en el pipeline actual.
