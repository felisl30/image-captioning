# Soluciones extra al mode collapse — versión acotada al scope del TP

Este documento reemplaza la versión anterior. Complementa `docs/mejoras_mode_collapse.md`
manteniendo disciplina de scope:

> Este es un TP de **interpretabilidad y análisis** sobre BLIP fine-tuneado.
> El fine-tuning es un medio, no el fin. Las captions necesitan ser **suficientemente
> buenas para sostener un análisis de heatmaps** — no perfectas, no estado del arte.
> Cualquier solución que convierta el trabajo en "ingeniería sobre el dataset" o
> "ingeniería sobre la arquitectura" queda fuera de scope.

## Lo que queda descartado y por qué

- **Cambiar el target a `findings`** → `findings` es 10× más largo y de estilo descriptivo,
  no del estilo de caption corto para el que BLIP fue preentrenado. Pasaría de "transfer de
  dominio" (radiología) a "transfer de dominio + de estilo" simultáneo. Multiplica el riesgo
  sin ser necesario para responder la pregunta del TP.
- **Cabezal auxiliar de clasificación (CheXpert labels) + multitask** → cambia la arquitectura
  del modelo, requiere instalar/correr el CheXpert labeler, requiere reescribir el train loop.
  Alta probabilidad de éxito pero costo desproporcionado para el rol que tiene el
  fine-tuning en este TP.
- **SCST / RL con CIDEr** → días de implementación, fuera de scope.
- **Curriculum learning en dos etapas** → duplica el costo de GPU sin garantía de mejora
  proporcional.

Quedan tres familias de soluciones — todas dentro del flujo "BLIP estándar + cambio mínimo".

---

## Fase 1 — Debug y diagnóstico

Antes de tocar nada, entender qué está pasando. Es trabajo barato (1 día como mucho) y
puede ahorrar reentrenamientos enteros.

### D1. Inspeccionar checkpoints intermedios — ¿cuándo aparece el collapse?

El finetuner ya guarda `epoch_1/`, `epoch_2/`, `epoch_3/` y `best/`. Cargar cada uno y
generar captions sobre las mismas 30 imágenes de `visual_test_indices.json`. Reportar:
captions únicas, top-1%, 5 ejemplos.

**Qué responde:**
- Si el collapse ya está en `epoch_1/` → la causa es el **incentivo** (loss + sampling).
  Saltar a S2 (loss reweighting).
- Si aparece progresivo entre épocas → es **overfitting al modo dominante**. Saltar a S2
  + reducir LR / early stop más agresivo.
- Si `epoch_1/` es diverso pero malo y `best/` es colapsado pero gramatical → hay un punto
  intermedio que sirve para el análisis. **Posiblemente ese checkpoint sea más útil para
  los heatmaps que el `best/`.**

Trabajo: ~2h. Salida: una tabla y 3 captions por checkpoint en el notebook 06.

### D2. Sondear la distribución bajo el capó — ¿qué tan puntiagudo es?

Generar con `output_scores=True, return_dict_in_generate=True`. Para cada caption,
calcular el log-prob del token greedy y el del segundo mejor en cada posición.

```python
out = model.generate(**inputs, max_new_tokens=30, output_scores=True,
                     return_dict_in_generate=True)
for step, score in enumerate(out.scores):
    probs = score.softmax(-1)
    top2 = probs.topk(2, dim=-1).values[0]
    print(f"step {step}: p(top1)={top2[0]:.3f}, p(top2)={top2[1]:.3f}, "
          f"ratio={top2[0]/top2[1]:.1f}")
```

**Qué responde:**
- Si `p(top1) ≈ 0.99` consistentemente → el modelo está **muy confiado** en una sola
  respuesta. Cambiar decoding no va a alcanzar; necesita reentrenamiento (S2).
- Si `p(top1)` está en `0.4–0.7` con `p(top2)` cercano → el modelo **conoce más de lo que
  greedy muestra**. S1 (decoding) probablemente alcance.

Esta sola medición decide si Fase 2 empieza en S1 o salta a S2. Trabajo: ~1h.

### D3. Mirar los heatmaps antes de "arreglar" nada

Esto es **literalmente el análisis central del TP**. Vale la pena adelantarlo aunque las
captions sean malas.

Correr el pipeline de cross-attention + Grad-CAM sobre 5–10 imágenes con el modelo `best/`
actual y compararlo con el modelo base. Si los heatmaps **ya muestran cambios médicamente
coherentes** (zonas de patología vs aleatoriedad), entonces:

- El collapse textual pasa a ser un **hallazgo reportable** (resultado tipo D: el encoder
  aprendió a ver, el decoder no aprendió a hablar), no un problema a resolver.
- Las soluciones de Fase 2 se vuelven mejoras opcionales para tener captions más legibles
  en el informe, no requisito metodológico.

Trabajo: ~2h (ya está toda la infra hecha). Esta es la pregunta más importante del proyecto.

---

## Fase 2 — Soluciones, ordenadas de "mejor a peor para este TP"

Criterio del orden: máxima mejora de captions / mínimo cambio al pipeline existente.

### S1. (Recomendado primero) Estrategias de decoding sin reentrenar

**Costo:** cero entrenamiento. Cambia solo cómo se llama a `model.generate(...)`.

Si D2 muestra una distribución no totalmente puntiaguda, esto solo puede resolver el problema.
Probar las tres variantes en orden y elegir la que dé mejor `unique_ratio` con captions
clínicamente plausibles a ojo.

```python
# (a) temperature + nucleus — ya en doc original, dejarlo como baseline
out_a = model.generate(**inputs, do_sample=True, temperature=1.2, top_p=0.9,
                       max_new_tokens=40)

# (b) diverse beam search — determinístico, devuelve k hipótesis distintas
out_b = model.generate(**inputs, num_beams=8, num_beam_groups=4,
                       diversity_penalty=0.5, do_sample=False,
                       num_return_sequences=4, max_new_tokens=40)

# (c) contrastive search — combate degeneración sin sacrificar coherencia
out_c = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=40)
```

**Cuál usar para el informe:** la que mejor balancee diversidad y plausibilidad clínica.
Probablemente (b) diverse beam — es determinístico (reproducible para el informe) y devuelve
varias hipótesis que enriquecen la figura de comparación.

**Por qué es la primera apuesta:** cero riesgo, cero costo de GPU, aplica al checkpoint que
ya existe. Si esto alcanza, el trabajo de fine-tuning queda como está y el TP avanza al
análisis de heatmaps inmediatamente.

### S2. Reentrenar con loss reweighting por frecuencia inversa

**Costo:** ~30 líneas de código + 1 corrida de GPU (~3h con los mismos hiperparámetros).

Si S1 no alcanza, el cambio más quirúrgico al training loop. No cambia el dataset, no cambia
el target, no cambia la arquitectura. Solo cambia cuánto pesa cada ejemplo en el gradiente.

**Idea:** "no acute cardiopulmonary process" aparece 1141 veces sobre 24k; las patologías
raras 1–3 veces. Para el optimizador es trivialmente óptimo memorizar la frase frecuente.
Reponderar cada ejemplo por `1 / frecuencia(impression)` elimina ese atajo.

**Implementación:**

```python
# En src/data/dataset.py — precalcular pesos una vez
from collections import Counter

def build_example_weights(dataset, indices, text_col="impression", smooth=10):
    imps = [(dataset[i][text_col] or "").strip().lower() for i in indices]
    counts = Counter(imps)
    raw = {imp: 1.0 / (counts[imp] + smooth) for imp in counts}
    mean = sum(raw[imp] for imp in imps) / len(imps)
    return [raw[imp] / mean for imp in imps]  # promedio normalizado a 1

# En el Dataset, devolver el peso junto con cada ejemplo:
# item["weight"] = weights[idx]

# En src/models/finetuner.py — modificar train_one_epoch:
loss_per_example = outputs.loss   # ya promediada sobre tokens
weighted_loss = (loss_per_example * batch["weight"]).mean()
weighted_loss.backward()
```

**Tradeoff:** sube la varianza del gradiente. Con `smooth=10` y `grad_accum_steps=4` se
controla. Si el train loss se vuelve inestable, subir `smooth` a 20.

**Resultado esperado:** top-1 de 83% → ~40–50% con la misma cantidad de datos. Es la mejora
de fondo más grande que se puede conseguir sin cambiar nada estructural.

### S3. Filtrar las frases "normales" triviales del train (combinable con S2)

**Costo:** 10 líneas + la misma corrida de GPU que S2.

```python
NORMAL_TEMPLATES = {
    "no acute cardiopulmonary process.",
    "no acute cardiopulmonary abnormality.",
    "no acute intrathoracic process.",
    "no significant interval change.",
    "no change.",
    "no evidence of acute disease.",
    "as above.",
}

def filter_trivial_normals(dataset, indices, text_col="impression"):
    return [
        i for i in indices
        if (dataset[i][text_col] or "").strip().lower() not in NORMAL_TEMPLATES
    ]
```

Saca ~3000–4000 ejemplos sobre 24k. Combinado con S2, el efecto se multiplica: ya no hay
ni siquiera la opción de aprender la frase fácil. Reportable en el informe en una línea
("descartamos del train las 7 frases que abarcaban el 13% del corpus y eran clínicamente
vacías; el conjunto resultante es de ~21k pares").

**No usar S3 solo** sin S2 — descartar normales redistribuye el desbalance pero no lo elimina
(otras frases ahora son las más frecuentes y el modelo colapsará a ellas).

---

## Fase 3 — Último recurso, solo si Fase 2 no alcanza

Una sola idea reservada para este caso, porque tiene mejor relación costo/beneficio que
las descartadas pero sigue agregando complejidad:

### S4. Unlikelihood loss sobre las 2–3 frases residuales del collapse

Si después de S2+S3 el modelo sigue colapsando a alguna frase específica (por ejemplo,
ahora "mild pulmonary edema" pasó a ser el nuevo modo dominante con 60%), agregar una
penalización quirúrgica `-log(1 - p(frase_dominante))` en posiciones donde el target real
es distinto. ~50 líneas, decisión hiperparamétrica de un `alpha`.

No detallo el código aquí porque solo tiene sentido si Fase 2 falla; en ese caso, vale la
pena escribirlo con el comportamiento residual a la vista.

---

## Plan recomendado de implementación gradual

```
DÍA 1 — Debug (Fase 1)
  D1  Captions a 4 checkpoints                       (~2h)
  D2  Probe de distribución (top1/top2)              (~1h)
  D3  Heatmaps con el modelo actual                  (~2h)
       └── Decisión: ¿el collapse es bloqueante o reportable?

DÍA 2 — Solución mínima (Fase 2, S1)
  S1  Probar (a), (b), (c) de decoding               (~3h)
       └── Si mejora suficiente → cerrar, pasar a análisis del TP
       └── Si no → continuar

DÍA 3 — Solución estructural (Fase 2, S2+S3)
  S2  Implementar weighted loss + corrida GPU        (~6h código + 3h GPU)
  S3  Aplicar filtro de normales en la misma corrida (~30min adicional)
       └── Reevaluar con D1/D2 sobre el nuevo checkpoint

DÍA 4 — (Opcional) Fase 3 si todavía no alcanza
  S4  Unlikelihood quirúrgica                        (~4h)
```

El estado "óptimo para este TP" es: captions con suficiente vocabulario médico variado para
que la cross-attention por palabra produzca mapas distinguibles y comparables entre el
modelo base y el fine-tuneado. **No** es captions perfectas ni métricas BLEU/CIDEr
competitivas con el estado del arte.

## Recordatorio del scope

Lo central del TP es comparar heatmaps antes/después. Cada hora gastada en mejorar captions
más allá del umbral "suficiente para análisis" es una hora no gastada en el análisis mismo.
Si en cualquier punto del plan D3 confirma que los heatmaps ya cambiaron de manera
significativa, **detener la Fase 2 y empezar el análisis** — el collapse pasa a ser un
hallazgo metodológico del informe, no un obstáculo.
