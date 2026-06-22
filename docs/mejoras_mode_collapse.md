# Mejoras al fine-tuning: mode collapse y estrategias de solución

Contexto: el modelo finetuneado con 5k imágenes colapsa a ~2 frases para el 90%+ de los casos
(`"no acute cardiopulmonary process."` y `"no significant interval change."`).
Este documento analiza las causas y las estrategias de mejora ordenadas por esfuerzo.

---

## Por qué ocurre el mode collapse

El análisis del notebook `06_analisis_captions.ipynb` mostró que el dataset de impressions
**ya es diverso textualmente** (entropía 13.32 bits, top-1 solo 4.7%). El collapse no viene
del desbalance de datos sino de dos factores:

1. **Clustering semántico**: frases textualmente distintas ("no acute cardiopulmonary process",
   "no acute intrathoracic process", "no evidence of acute disease") son semánticamente
   equivalentes. El modelo aprende que todas convergen al mismo territorio y colapsa al
   representante más frecuente de ese cluster.

2. **Decoding greedy**: el modo de generación por defecto elige siempre el token más probable.
   Si la distribución aprendida tiene aunque sea un ligero sesgo hacia la frase más común,
   greedy la amplifica hasta el 90%.

3. **Poca exposición a hallazgos raros**: con 5k ejemplos de train, patologías como neumotórax
   o consolidación focal aparecen 1-2 veces — insuficiente para aprender su representación visual.

---

## Estrategia 1 — Temperature / nucleus sampling en inferencia

**Esfuerzo:** ninguno (no requiere reentrenar)
**Probabilidad de mejora:** media-alta
**Cuándo probarlo:** primero, antes de reentrenar nada

Si el modelo aprendió la distribución correcta pero greedy la aplana, cambiar el modo de
generación ya produce captions más variadas.

```python
out = model.generate(
    **inputs,
    do_sample=True,
    temperature=1.3,   # > 1 suaviza la distribución, más variedad
    top_p=0.9,         # nucleus sampling: considera solo los tokens del 90% acumulado
    max_new_tokens=50,
)
```

**Cómo evaluar si funcionó:** comparar el conteo de captions únicas antes y después.
Si con temperatura el modelo genera hallazgos específicos (efusión, atelectasia, consolidación)
aunque sea en algunos casos, el modelo sabe más de lo que greedy muestra.

**Valores a explorar:**

| temperature | top_p | Efecto esperado |
|---|---|---|
| 1.0 | 1.0 | Igual que greedy (casi) |
| 1.2 | 0.95 | Ligera variedad |
| 1.3 | 0.9 | Balance razonable |
| 1.5 | 0.85 | Más variedad, más riesgo de incoherencia |

---

## Estrategia 2 — Entrenar con más datos (15k–24k)

**Esfuerzo:** ~2-3h de GPU (una corrida más en la VM)
**Probabilidad de mejora:** media
**Cuándo probarlo:** si la estrategia 1 mejora pero no alcanza

Más datos no cambia la distribución relativa del dataset, pero sí aumenta la exposición
a hallazgos raros. Con 5k, una patología que ocurre en el 2% del dataset aparece ~100 veces
en train. Con 24k aparece ~480 veces — suficiente para aprender su representación visual.

**Cómo implementarlo:** el finetuner ya soporta esto, solo cambiar los índices:

```bash
python -m src.models.finetuner \
    --train-indices data/splits/train_indices.json \  # usar todos los 15k índices
    --val-indices data/splits/val_indices.json \
    --epochs 3 \
    --batch-size 8 \
    --lr 1e-5 \
    --output-dir models/blip_finetuned_15k/
```

**Qué esperar:** reducción del collapse pero no eliminación. La distribución semántica
es la misma — el modelo sigue viendo mayoritariamente radiografías "normales".

---

## Estrategia 3 — Balanceo semántico del dataset

**Esfuerzo:** alto (requiere clustering + reentrenar)
**Probabilidad de mejora:** alta
**Cuándo probarlo:** si las estrategias 1 y 2 no son suficientes

El balanceo por string exacto no ayuda (demostrado en el notebook: el top-1 ya es 4.7%).
El balanceo que sí ayuda es **semántico**: agrupar impressions por categoría clínica y
limitar cuántos ejemplos de cada categoría entran al training.

**Categorías sugeridas** (basadas en los hallazgos más frecuentes en MIMIC-CXR):

```python
CATEGORIAS = {
    "normal":           ["no acute", "no evidence", "unremarkable", "no significant"],
    "atelectasia":      ["atelectasis", "atelectatic", "volume loss"],
    "efusion":          ["pleural effusion", "effusion"],
    "edema":            ["pulmonary edema", "vascular congestion", "interstitial edema"],
    "consolidacion":    ["consolidation", "pneumonia", "opacity", "infiltrate"],
    "neumo":            ["pneumothorax"],
    "dispositivos":     ["picc", "endotracheal", "enteric tube", "ng tube", "catheter"],
    "otros":            [],  # todo lo que no matchea
}

def categorizar(impression: str) -> str:
    imp = impression.lower()
    for cat, keywords in CATEGORIAS.items():
        if any(kw in imp for kw in keywords):
            return cat
    return "otros"
```

Luego al armar el dataset de train, limitar a `max_per_cat` ejemplos por categoría:

```python
from collections import defaultdict
import random

def balancear(indices, dataset, max_per_cat=800, seed=42):
    random.seed(seed)
    por_cat = defaultdict(list)
    for idx in indices:
        imp = dataset[idx]["impression"] or ""
        cat = categorizar(imp)
        por_cat[cat].append(idx)
    
    balanceados = []
    for cat, idxs in por_cat.items():
        sample = random.sample(idxs, min(len(idxs), max_per_cat))
        balanceados.extend(sample)
        print(f"  {cat:<20}: {len(idxs):>5} → {len(sample)}")
    
    random.shuffle(balanceados)
    return balanceados
```

Con `max_per_cat=800` y las categorías de arriba se obtiene un dataset de ~5k-7k ejemplos
con distribución mucho más uniforme entre hallazgos normales y patológicos.

---

## Estrategia 4 — Ajuste de hiperparámetros

**Esfuerzo:** bajo-medio (cambios en el comando de entrenamiento)
**Probabilidad de mejora:** baja-media en aislamiento, alta en combinación con otras

Algunos ajustes que pueden reducir el collapse:

### Learning rate más bajo
Con `lr=1e-5` el modelo puede converger demasiado rápido a la moda antes de aprender
representaciones específicas. Probar `lr=5e-6` con más épocas.

```bash
--lr 5e-6 --epochs 5
```

### Label smoothing
Evita que el modelo se vuelva demasiado confiado en el token más probable.

```python
# En el loop de training, reemplazar CrossEntropyLoss por:
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## Orden recomendado de experimentación

```
1. Temperature sampling (minutos, sin GPU)
   └── ¿mejora la variedad? 
       ├── Sí → reportar en el informe como finding + usar para generar captions finales
       └── No → el modelo aprendió realmente poco, ir a 2

2. Reentrenar con 15k-24k datos (3h GPU)
   └── ¿reduce el collapse?
       ├── Sí, suficiente → cerrar aquí
       └── No → ir a 3

3. Balanceo semántico + reentrenar (1 día de trabajo + 3h GPU)
   └── Resultado esperado: captions con hallazgos específicos en 30-50% de los casos
```

---

## Impacto en la pregunta de investigación

El mode collapse **no invalida el trabajo**. Incluso con captions degeneradas, la comparación
de heatmaps antes/después del fine-tuning sigue siendo válida e interesante:

- Si los heatmaps cambian pese al collapse textual → el encoder sí aprendió a ver diferente,
  aunque el decoder no sepa expresarlo (resultado más rico que el esperado)
- Si los heatmaps no cambian → el fine-tuning solo afectó el registro lingüístico del decoder,
  no la representación visual (Resultado B del CLAUDE.md)

El mode collapse se reporta como **limitación metodológica** en el informe, con las estrategias
de este documento como trabajo futuro.
