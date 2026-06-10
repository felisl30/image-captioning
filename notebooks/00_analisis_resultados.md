# Análisis del notebook 00 — Exploración del dataset

Este documento explica qué hizo el notebook, qué significan los resultados, qué cosas hay que tener en cuenta antes de continuar, y dónde hay decisiones pendientes.

---

## Lo que hizo el notebook

1. Configuró el `PROJECT_ROOT` y los paths de importación.
2. Creó la estructura de carpetas y archivos `__init__.py`.
3. Escribió (sobreescribió) los módulos de `src/` con `%%writefile` (ver advertencia abajo).
4. Cargó **200 muestras** en modo streaming como smoke test.
5. Verificó columnas y ausencia de vacíos.
6. Graficó distribución de longitudes de `findings` e `impression`.
7. Generó splits con `generate_splits()`.
8. Verificó que los splits no se solapan y que `selected ⊆ test`.
9. Creó un `DataLoader` y obtuvo un batch.
10. Ejecutó un forward pass completo con `BlipForConditionalGeneration`.
11. Generó un caption de prueba sobre una radiografía seleccionada.
12. Guardó un resumen JSON en `outputs/dataset_checks/`.

---

## Cosas que pueden resultar confusas

### 1. Los splits son 140 / 30 / 30, no 24k / 4.5k / 1.5k

**Por qué:** el notebook carga solo `N_SAMPLES = 200` en modo smoke test (streaming). Con solo 200 muestras válidas, `auto_shrink=True` en `generate_splits()` calcula splits proporcionales: 70% train = 140, 15% val = 30, el resto test = 30.

**Esto es correcto para probar el pipeline.** Los splits reales (24k/4.5k/1.5k) se generan cuando se corre el notebook con el dataset completo, descomentando la celda de `load_mimic_dataset` y comentando la de streaming.

**Acción necesaria antes del fine-tuning:** re-ejecutar la celda de `generate_splits()` con el dataset completo cargado.

---

### 2. `selected_indices.json` tiene 30 ítems y es idéntico a `test_indices.json`

**Por qué:** en smoke test con solo 30 muestras de test y `selected_size=30`, `selected` termina siendo todo el test set. Esto es un artefacto del smoke test.

**Con el dataset completo:** test tendrá ~1.500 muestras y selected será un subconjunto de 20–30 de ellas — correctamente distintos.

**Importante:** `selected_indices.json` se genera una sola vez con el dataset completo y no se vuelve a tocar. Es el corazón del experimento comparativo.

---

### 3. `pixel_values` tiene shape `(2, 3, 384, 384)`, no `224×224`

**Este es el hallazgo más importante del notebook para la interpretabilidad.**

BLIP base para captioning usa imágenes de **384×384 px**, no 224×224. El `BlipProcessor` redimensiona automáticamente a 384×384 antes de pasarlas al modelo.

Consecuencias directas para el análisis de interpretabilidad:

| Dato | Lo que dice CLAUDE.md | Lo que es realmente |
|---|---|---|
| Resolución de entrada | 224×224 | **384×384** |
| Patches por lado | 14 (224/16) | **24 (384/16)** |
| Total de patch tokens | 196 (14×14) | **576 (24×24)** |
| Shape de cross-attention | `(B, heads, T, 196)` | **`(B, heads, T, 576)`** |
| Tokens para reshape en Grad-CAM | 197 (CLS+196) | **577 (CLS+576)** |
| Grid para visualización | 14×14 | **24×24** |

Esto afecta tanto `cross_attention.py` como `gradcam.py`. El `vit_reshape_transform` estándar de `pytorch-grad-cam` asume 197 tokens — hay que pasarle el parámetro correcto o ajustarlo para 577.

---

### 4. La loss es 6.55 — ¿es alta o baja?

La loss es una cross-entropy sobre el vocabulario del decoder (≈30.000 tokens). Para un modelo completamente aleatorio sería `ln(30000) ≈ 10.3`. Que el modelo base dé 6.55 sobre una radiografía (dominio distinto) es razonable: el modelo ya sabe generar texto coherente, aunque no sepa hablar de radiografías.

Después del fine-tuning se espera que baje a ~3–5 sobre impression de radiografías.

---

### 5. El caption generado: `"a chest xray with a large, open chest"`

Esto es exactamente el baseline esperado. BLIP base identifica que es una radiografía de tórax (correcto) pero describe la imagen en términos generales e incorrectos médicamente. La impression real era `"Port-A-Cath tip over mid SVC. No acute pulmonary process identified."` — el modelo no menciona el catéter ni el proceso pulmonar.

Este resultado documenta el **Resultado B** esperado del trabajo: antes del fine-tuning, el lenguaje es genérico.

---

### 6. Una impression en el batch tiene texto incompleto: `"...increased since . with Dr."`

Esto es una muestra con datos de calidad baja en el dataset. La impression aparentemente fue truncada o tiene artefactos del procesamiento del dataset original. Con 30k muestras habrá algunas así. No es un bug — el filtro de vacíos que ya está implementado (`is_empty_text`) no los captura porque el texto no está vacío, solo está incompleto.

No es necesario filtrarlos explícitamente; son una minoría y el fine-tuning es robusto a algo de ruido.

---

### 7. ⚠️ Las celdas `%%writefile` sobreescriben los archivos de `src/`

Las celdas con `%%writefile src/data/split_generator.py` (y otros módulos) reescriben los archivos de `src/` cada vez que se ejecutan. Esto es un problema porque:

- `split_generator.py` fue actualizado para usar los defaults correctos (24k/4.5k/1.5k, ratios 80/15).
- Si se vuelve a ejecutar esa celda del notebook, se sobreescribe con los valores viejos (10k/1k/1k, ratio 70%).

**Acción recomendada:** en la próxima ejecución del notebook, borrar o comentar todas las celdas `%%writefile`. Los módulos de `src/` ya están correctos en disco y no necesitan regenerarse desde el notebook.

---

## Imágenes de demostración — TODO

Se quieren extraer **5 imágenes de demostración** completamente fuera de cualquier split (train, val, test, selected). El objetivo es tener un set fijo de radiografías para mostrar visualmente que el modelo base no funciona en el dominio médico, como punto de partida cualitativo del trabajo.

### Dónde sacarlas

Con el dataset completo (~30.633 filas válidas totales) y splits de 24k + 4.5k + 1.5k = 30.000 muestras, quedan ~633 índices sin usar. Las 5 imágenes de demo se sacan de ese pool de sobrantes.

### Cuándo hacerlo

Después de generar los splits reales con el dataset completo. El flujo es:

```
1. Cargar dataset completo (load_mimic_dataset)
2. Generar splits reales → train/val/test/selected quedan fijos
3. Calcular el pool de sobrantes:
      used = set(train + val + test)
      all_valid = set(get_valid_indices(train_split))
      leftover = sorted(all_valid - used)
4. Elegir los primeros 5 del pool ordenado (determinista, reproducible)
5. Guardar en data/demo_indices.json
```

### Qué hacer con ellas

- Correr BLIP base sobre las 5 imágenes y guardar los captions en `outputs/demo/captions_base.json`.
- En el informe/poster se pueden mostrar como "estas son radiografías reales, esto es lo que dice el modelo antes del fine-tuning".
- No se usan para entrenamiento ni evaluación cuantitativa.

**TODO en el notebook:** agregar una celda al final de `00_exploracion_dataset.ipynb` que:
1. Calcule el pool de sobrantes tras generar los splits reales.
2. Elija los primeros 5 índices del pool (seed no necesaria, es determinista por orden).
3. Los guarde en `data/demo_indices.json`.

---

## Checklist antes de continuar al notebook 01

- [ ] **Re-ejecutar con dataset completo**: descomentar `load_mimic_dataset`, comentar el bloque de streaming.
- [ ] **Comentar / borrar las celdas `%%writefile`**: los módulos ya están correctos en `src/`.
- [ ] **Actualizar el call a `generate_splits`** con los nuevos defaults: `train_size=24000, val_size=4500, test_size=1500`.
- [ ] **Verificar que `pixel_values` sigue siendo `384×384`** después de recargar con el dataset completo.
- [ ] **Generar `demo_indices.json`** con las 5 imágenes de demostración (ver sección anterior).
- [ ] **No volver a tocar `selected_indices.json`** una vez generado con el dataset completo.
- [ ] Corregir en `CLAUDE.md` la mención a 224×224 / 196 patches → 384×384 / 576 patches.
