# Reporte de auditoría — diagnósticos D1/D2/D3 y solución S1 (archivos_ion)

Este reporte audita el material entregado en `archivos_ion/`:

- 4 scripts: `run_d1_checkpoint_collapse.py`, `run_d2_token_probe.py`,
  `run_d3_heatmap_probe.py`, `run_s1_decoding_experiment.py`.
- 4 notebooks: `06_debug_mode_collapse.ipynb`, `07_debug_token_distribution.ipynb`,
  `08_debug_heatmaps_probe.ipynb`, `09_debug_decoding_strategies.ipynb`.
- 5 documentos de handoff en Markdown.

El reporte audita la coherencia interna del código, su alineación con la API real
de `src/` y su compatibilidad con el estado actual del repositorio. **No se
modificó nada** — este documento es solo un diagnóstico.

---

## Sección 1 — Errores a corregir

Se ordenan en tres bloques: errores **bloqueantes** (impiden correr el flujo
tal cual está), **defectos funcionales** (no rompen pero degradan los
resultados o el análisis) y **inconsistencias menores** (estéticas o
defensivas).

### 1.1. Errores bloqueantes (impiden ejecutar el flujo "out of the box")

#### B1. La carpeta `scripts/` no existe en el repo

Todos los handoffs y notebooks asumen `scripts/run_d1_checkpoint_collapse.py`,
etc. La estructura actual del repo no tiene una carpeta `scripts/`. Los
notebooks 06–09 imprimen comandos que apuntan a esa carpeta, y el comando
fallará con `python: can't open file 'scripts/run_d1_checkpoint_collapse.py'`.

**Corrección:** crear `image-captioning/scripts/` y copiar los 4 `.py` ahí.
(Decisión de integración — ver mi mensaje de chat posterior.)

#### B2. Los checkpoints `epoch_1/2/3` que D1 y D2 requieren no existen

D1 y D2 dependen de comparar `epoch_1/`, `epoch_2/`, `epoch_3/`, `best/`
dentro de `models/blip_finetuned_5k/` o `models/blip_finetuned/`. El estado
real del repo es:

```
models/blip_base/                ← existe, OK
models/blip_finetuned/best/      ← carpeta VACÍA (0 archivos)
../output_5k/best/               ← el único checkpoint real (fuera de image-captioning/)
```

Consecuencias concretas:
- D1 falla con `ERROR: hay menos de dos checkpoints usables` (línea 429–435
  de `run_d1_checkpoint_collapse.py`).
- D2 emite la advertencia "menos de dos checkpoints usables" pero puede correr
  con un solo checkpoint. Sin embargo, no encuentra ninguno con los paths
  default — hay que pasar `--checkpoint-root` apuntando manualmente.

**Corrección (dos opciones):**
- (a) **Re-correr el finetuner y guardar todas las épocas.** El `finetuner.py`
  ya guarda `epoch_N/` por defecto (línea 555–558). En la corrida que generó
  `output_5k/best/` se pasó probablemente `--skip-checkpoint-save` o se borró
  todo menos `best/`. Una nueva corrida con los flags default debería dejar
  los 4 checkpoints.
- (b) **Reducir el alcance de D1/D2 a un único checkpoint** (`best/`) y
  reportar D1 como "no aplicable" hasta tener checkpoints intermedios. D2 sí
  corre con uno solo y aporta información útil.

#### B3. El checkpoint real está fuera del project root

El finetuner volcó a `tp_final/output_5k/best/`, que está **fuera** de
`image-captioning/`. Los scripts auto-buscan en:

```
models/blip_finetuned_5k/best
models/blip_finetuned/best
```

Ninguno existe en imagen-captioning. Auto-discovery falla.

**Corrección (dos opciones):**
- (a) **Symlink:** `ln -s ../output_5k models/blip_finetuned_5k` (relativo
  desde `image-captioning/models/`). No mueve archivos, los scripts encuentran
  el path esperado.
- (b) **Pasar explícito:** `--ft-model-dir ../output_5k/best` o
  `--checkpoint-root ../output_5k` en cada invocación. Más verboso pero menos
  estado mágico.

#### B4. Divergencia de índices: `selected_indices.json` (30) vs `visual_test_indices.json` (25)

El proyecto tiene **dos** archivos de índices en `data/`:

```
data/selected_indices.json     ← 30 índices, los que usan TODOS los scripts nuevos
data/visual_test_indices.json  ← 25 índices, los que usa prueba_finetuning.ipynb
```

**Las listas no se intersectan en el primer índice.** `selected_indices`
empieza con `[15399, 2454, 5062, ...]` (mismas imágenes con las que se
generó `captions_ft_5k.json`). `visual_test_indices` empieza con `[731, ...]`.

Esto significa que el análisis "antes vs después del finetuning" que ya
existe en `prueba_finetuning.ipynb` usa imágenes distintas a las que usarán
D1/D2/D3/S1. **No se podrá hacer comparación directa con el material previo
si no se unifica esto.**

**Corrección:** decidir cuál es la fuente de verdad y borrar/renombrar la
otra. Mi recomendación: **mantener `selected_indices.json`** porque es la
que ya tiene captions generadas para el modelo fine-tuneado actual (los 300
ejemplos de `captions_ft_5k.json` ÷ 10 muestreos por imagen = 30 imágenes
únicas que coinciden con este archivo) y porque es la que los nuevos scripts
usan. Actualizar `prueba_finetuning.ipynb` para usar el mismo archivo
cuando se reescriba.

#### B5. `pytorch_grad_cam` no está garantizado en el entorno

D3 lo importa dentro de `run_gradcam` (línea 197). Si falta, el script lo
detecta (`dependency_available("pytorch_grad_cam")`, línea 470) y aborta a
menos que se pase `--skip-gradcam`. No es bug del script — es un prerrequisito
de instalación a verificar antes de correr D3.

**Corrección:** `pip install grad-cam` (nombre del paquete pip; import name
es `pytorch_grad_cam`). Confirmar también que está en `requirements.txt`.

---

### 1.2. Defectos funcionales (corren pero generan resultados subóptimos)

#### F1. D1 — `make_examples` crashea si la caption greedy está vacía

`make_examples` (línea 226–260) llama
`Counter(group["caption_norm"]).most_common(1)[0][0]`. Si TODOS los captions
de un checkpoint son strings vacíos (caso edge: checkpoint corrupto), el
Counter tendrá una sola entrada y todo OK. Pero si el grupo está vacío
(que no debería pasar dada la lógica del flujo), crashea con `IndexError`.
Probabilidad baja, pero no hay guarda.

**Corrección:** envolver en
`if not group.empty:` o filtrar `caption_norm` no vacío antes.

#### F2. D1 — Carga el dataset HF entero (30k filas) aunque max-images=30

Línea 464: `ds = load_mimic_dataset(cache_dir=...)`. Es necesario porque
luego accede `hf_split[int(idx)]`. No es un bug, pero la primera corrida
descarga ~800 MB de Parquet si no está en cache. Esperable. Solo flag
para el usuario que la primera ejecución sea lenta.

#### F3. D2 — `summarize_strategy` y `pct_specific_clinical` en S1 dan métrica engañosa

(Este punto es de S1, lo agrupo aquí.) Líneas 231–233 de `run_s1_decoding_experiment.py`:

```python
"pct_other":             float((group["categories"].str.contains("other")).mean()),
"pct_normal":            float((group["categories"].str.contains("normal")).mean()),
"pct_specific_clinical": float((~group["categories"].isin(["other", "normal"])).mean()),
```

El campo `categories` contiene strings pipe-separados como `"normal|effusion"`
(porque `classify_caption` devuelve TODAS las categorías que matchean, no una
sola). `str.contains("normal")` matchea `"normal|effusion"`, así que un caption
que detecta efusión Y es normal cuenta como ambas. Por otro lado,
`isin(["other","normal"])` exige match exacto, así que la misma caption
`"normal|effusion"` queda fuera del set "normal puro" y `~isin` la cuenta
como `pct_specific_clinical=True`. **Resultado:** `pct_normal +
pct_specific_clinical > 1` es posible y normal, sin que esté documentado.

**Corrección sugerida:** definir explícitamente "es solamente normal" como
`categories == "normal"` (string exacto) y dejar `pct_specific_clinical` como
"al menos una categoría clínica específica". O simplemente documentarlo en
la interpretación del notebook.

#### F4. S1 — `diverse_beam` rompe si `--samples-per-image < 4`

Línea 180: `"num_return_sequences": min(4, max(1, samples_per_image))`.
Combinado con `num_beams=8, num_beam_groups=4`, transformers requiere que
`num_return_sequences` sea divisible por `num_beam_groups`. Si el usuario
pasa `--samples-per-image 1`, el resultado es `min(4,1)=1`, no divisible por
4. La estrategia falla con `ValueError`. El try/except la captura y la
escribe a `s1_errors.csv`, así que no rompe el flujo, pero la columna
`diverse_beam` queda vacía silenciosamente.

**Corrección:** forzar `num_return_sequences` a ser múltiplo de
`num_beam_groups` (por ejemplo `((samples_per_image + 3) // 4) * 4`), o
saltar la estrategia si `samples_per_image < 4`.

#### F5. S1 — `contrastive_search` puede no estar disponible en transformers viejas

Línea 187: `"penalty_alpha": 0.6, "top_k": 4`. Contrastive search requiere
transformers ≥ 4.24 aprox. CLAUDE.md sec 8.2 dice "transformers 5.x". OK.
Pero la combinación con `do_sample=False` y sin `num_beams` es la sintaxis
estándar. Igualmente el try/except absorbe cualquier error. Solo verificar
que la versión instalada efectivamente sea ≥ 4.24.

#### F6. S1 — La semilla solo se fija una vez al inicio

`set_seed(args.seed)` en línea 326. Las estrategias de sampling
(`sample_t1.2_p0.95`, etc.) consumen estado random secuencialmente. Si se
re-corre el script con misma seed sin cambiar nada, da el mismo resultado.
Pero si se cambia `--max-images` o el orden de las estrategias, los
resultados de cada estrategia cambian. Esto es esperable para sampling,
pero merece quedar documentado. No es bug.

#### F7. D2 — La variable `checkpoint_path` se reusa fuera del loop

Línea 747: `checkpoint_path = output_dir / "d2_checkpoint_summary.csv"`.
Esto re-usa el nombre que dentro del loop (línea 643) apunta al
`Path` del checkpoint en disco. Como el loop ya terminó, funciona, pero es
shadowing confuso. Renombrar a `checkpoint_summary_csv_path` o similar.

#### F8. D3 — `cross_grid_path` / `gradcam_grid_path` se setean antes de guardar

Líneas 332–333 y 366–367:

```python
row["cross_status"] = "ok"
row["cross_grid_path"] = str(cross_path)
save_heatmap_grid(...)  # esto podría tirar
```

Si `save_heatmap_grid` (matplotlib) tira excepción, el outer
`except Exception as e:` cambia `cross_status` a `"error"` pero
`cross_grid_path` queda con un path que no existe en disco. El notebook 08
celda 23 muestra esos paths como referencia, podría confundir.

**Corrección:** setear `row["cross_grid_path"] = str(cross_path)` solo
**después** del `save_heatmap_grid` exitoso.

#### F9. D3 — `extract_cross_att_logits` hardcodea `max_new_tokens=40`

En `src/interpretability/cross_att_logits.py` línea 83. D1/D2/S1 también
usan 40 por default. Coherente. Solo flag: si se decide aumentar
`max_new_tokens` en algún script, hay que sincronizar también con
`extract_cross_att_logits` (que no acepta parámetro) editándolo a mano.
No bloqueante.

#### F10. D3 — Si `extract_cross_att_logits` devuelve lista vacía, el wrapper la convierte en error opaco

`run_cross_att_logits` (línea 191) raisea `RuntimeError("extract_cross_att_logits
devolvió lista vacía.")`. Esto pasa cuando el hook de `K` no captura nada,
típicamente por mismatch de versión de transformers o cambio en la arquitectura
interna. El error no explica la causa subyacente.

**Corrección sugerida:** añadir info de diagnóstico (capa, head_reduction,
estructura interna detectada) al mensaje, o al menos un `logger.warning` antes
del raise.

---

### 1.3. Inconsistencias menores

#### M1. `find_project_root` en scripts vs notebooks

Scripts: chequean `src/` AND `requirements.txt` (líneas 50–62 de cada `.py`).
Notebooks: solo chequean `src/`. Diferencia poco relevante (no creo que haya
otros directorios con `src/` en el cwd típico), pero la inconsistencia podría
sorprender. Unificar a la versión más estricta (la de los scripts).

#### M2. `DEBUG_CANDIDATE_ROOTS` apunta a paths que nadie usa

```
models/blip_finetuned_debug_save/
models/blip_finetuned_debug/
models/blip_finetuned_notebook_debug/
```

Estos paths no existen en el repo. La flag `--allow-debug` no encontrará nada
y los scripts caerán al error "no hay checkpoints usables". El flag tampoco
es necesario en este punto del proyecto — los debug checkpoints fueron una
herramienta del desarrollo del finetuner. Se puede dejar la flag pero
documentar que en este repo no aplica, o sacarla.

#### M3. Notebooks generan figuras `.png` en el OUTPUT_DIR de los CSVs

Notebooks 06 y 07 guardan
`checkpoint_collapse_plot.png` / `d2_checkpoint_distribution_summary.png` /
`d2_confident_vs_ambiguous_steps.png` en `outputs/mode_collapse_debug/`,
junto a los CSVs que los scripts producen. No es bug — solo flag organizativo.
Si se quieren separar los artefactos del informe de los CSVs intermedios,
crear `outputs/figures/` o similar.

#### M4. D2 imprime "Cantidad usada: {len(indices)}" pero el dataset puede tener menos filas

Si el usuario pasa `--max-images 100` y el dataset solo tiene 30 índices
en `selected_indices.json`, `load_indices` cap a 30 y el mensaje dice
"Cantidad usada: 30". Correcto, pero el contraste con `--max-images 100`
puede confundir. Trivial.

#### M5. Ningún script usa `tqdm`

Solo `print` por iteración. En CPU con 30 imágenes y 4 checkpoints
(D1: ~120 iteraciones; D2 igual; S1: ~120 imágenes × 6 estrategias = 720;
D3: 10 modelo×imagen con backward por token), el log se ensucia bastante.
No es bug. Si se quiere, agregar `tqdm.tqdm` al outer loop.

#### M6. `--include-base` en D1/D2/S1 tiene comportamiento sutil

D1 lo prepende a `checkpoints`, D2 igual, S1 lo prepende a `model_jobs`.
En todos los casos, la métrica del modelo base se grafica junto a las del
fine-tuneado. Para D1 y D2 esto puede ser engañoso porque el modelo base
genera captions de imágenes naturales ("a chest x-ray with a large open
chest..."), no clínicas — comparar `unique_ratio` o `mean_p_top1` con
fine-tuneado es comparar peras con manzanas. Para S1 es más legítimo:
permite ver si las estrategias de decoding mejoran el modelo base también.
Documentar.

#### M7. Los notebooks usan `display(...)` sin importarlo

Notebook 06 cell 4, notebook 07 cells 12/18/20, notebook 08 cells 10/12/14/16/23.
En Jupyter funciona porque está inyectado, pero `flake8`/`pyright` lo marca
como error. Importar `from IPython.display import display` (notebook 08 sí
lo hace).

#### M8. El handoff `d3_handoff_explicacion.md` dice "alpha=0.55" pero el script default es 0.55

Consistente. Solo flag: cualquier cambio de default debe actualizarse en
ambos lados.

---

## Sección 2 — Explicación del código

### 2.1. Visión general del bundle

`archivos_ion/` introduce una capa de **diagnóstico** sobre el modelo
fine-tuneado, en respuesta al mode collapse documentado en
`docs/mejoras_mode_collapse.md` y `soluciones_extra.md`. La arquitectura
siguiente:

```
Cada paso (D1, D2, D3, S1) = 1 script .py + 1 notebook .ipynb
  - El script HACE el cómputo (greedy, scores, heatmaps, sampling).
  - El notebook LEE el output, grafica y permite interpretación cualitativa.
```

Esta separación es buena: respeta la regla del CLAUDE.md sec 6 ("notebooks
delgados, lógica en `src/` o scripts"). Los scripts son auto-contenidos
(no dependen de Jupyter), reproducibles vía CLI, y los notebooks solo leen
CSVs/PNGs.

### 2.2. D1 — `run_d1_checkpoint_collapse.py` + `06_debug_mode_collapse.ipynb`

**Pregunta que responde:** ¿el collapse aparece desde la primera época o
progresivamente?

**Cómo:**
1. Descubre checkpoints `epoch_1`, `epoch_2`, `epoch_3`, `best` en
   `models/blip_finetuned_5k/` (con fallback a `models/blip_finetuned/`).
2. Para cada checkpoint, carga el modelo, genera **greedy** sobre las 30
   imágenes de `selected_indices.json`.
3. Calcula `unique_ratio = n_unique / n_total` y
   `top_pct = top_count / n_total` — las dos métricas centrales del collapse
   textual.
4. Guarda 4 archivos: `checkpoint_captions.csv` (raw), `checkpoint_collapse_summary.csv`
   (1 fila por checkpoint, la tabla del informe), su .json equivalente, y
   `checkpoint_examples.csv` (5 primeros + 5 de la caption dominante por
   checkpoint).

**El notebook** lee esos CSVs y muestra: la tabla resumen, un gráfico de
barras con `unique_ratio` y `top_pct`, y ejemplos cualitativos. Es de ~5
celdas de código.

**Aporte de D1 al pipeline:** define si el collapse es un problema
**temprano** (incentivo de entrenamiento) o **progresivo** (overfitting).
Esa información es input necesario para decidir si vale la pena buscar un
checkpoint intermedio o si hay que cambiar la receta de entrenamiento.

### 2.3. D2 — `run_d2_token_probe.py` + `07_debug_token_distribution.ipynb`

**Pregunta que responde:** ¿el modelo está muy confiado o greedy decoding
oculta alternativas cercanas?

**Cómo (más complejo que D1):**
1. Misma estructura de descubrimiento de checkpoints.
2. Para cada imagen y checkpoint, llama a:
   ```python
   model.generate(..., output_scores=True, return_dict_in_generate=True)
   ```
3. Para cada paso de generación, aplica softmax a los logits, extrae
   top-k tokens y calcula:
   - `p_top1`, `p_top2`: probabilidades.
   - `gap_top1_top2 = p_top1 - p_top2`: cuánto domina top-1.
   - `ratio_top1_top2 = p_top1 / p_top2`: ídem proporcional.
   - `logit_gap`: misma diferencia pero en logits (antes de softmax).
   - `entropy`: dispersión total de la distribución.
   - `topk_mass`: cuánta probabilidad está concentrada en top-k.
4. Filtra (por default) tokens especiales (EOS, PAD) para no contaminar
   las métricas. Esa flag es importante: EOS suele tener `p_top1` muy alto
   y inflaría la confianza media.
5. Tres niveles de agregación:
   - `d2_token_probe_steps.csv` (raw, una fila por token).
   - `d2_image_summary.csv` (una fila por checkpoint×imagen).
   - `d2_checkpoint_summary.csv` (una fila por checkpoint — la tabla del informe).
6. Genera además `d2_high_confidence_examples.csv` (10 imágenes donde el
   modelo más se planta) y `d2_low_margin_examples.csv` (10 imágenes donde
   top-1 y top-2 estuvieron empatados — candidatas para que sampling ayude).

**El notebook** grafica `mean_p_top1`, `mean_p_top2`, `mean_gap` y los
porcentajes "high confidence" / "low margin" por checkpoint. Incluye una
celda de "interpretación automática preliminar" que clasifica el escenario
en Caso A (distribución puntiaguda → reentrenar) o Caso B (gap chico →
probar sampling).

**Aporte de D2 al pipeline:** es la decisión bisagra entre S1 (sampling
sin reentrenar) y soluciones más caras (S2 loss reweighting, S3 filter).
**Sin D2, S1 sería un experimento a ciegas.**

### 2.4. D3 — `run_d3_heatmap_probe.py` + `08_debug_heatmaps_probe.ipynb`

**Pregunta que responde:** aunque las captions colapsen, ¿los heatmaps
del modelo fine-tuneado cambiaron respecto del base?

**Cómo:**
1. Carga `models/blip_base/` y `models/blip_finetuned_5k/best/` (con la
   misma búsqueda fallback).
2. Para 5 imágenes (default) de `selected_indices.json`, corre **dos**
   pipelines de interpretabilidad usando los módulos existentes en `src/`:
   - `extract_cross_att_logits` (cross-attention pre-softmax Q·K^T/√d).
   - `compute_gradcam` (con `pytorch_grad_cam` y el `reshape_transform`
     de BLIP).
3. Guarda figuras en `outputs/mode_collapse_debug/d3_heatmaps/idx_<IDX>/<model>/`:
   - `original.png` (radiografía RGB).
   - `cross_att_logits_grid.png` (un heatmap por palabra del caption).
   - `gradcam_grid.png` (idem).
   - `cross_vs_gradcam.png` (comparación lado a lado).
4. Acumula un `d3_heatmap_summary.csv` con captions, paths a figuras y
   estado de ejecución (`ok`/`error`/`skipped`/`not_run`) por método.

**Características clave:**
- Captura excepciones por método (cross-att y gradcam por separado), así que
  si Grad-CAM falla, cross-attention sí se guarda. Aceptable dado que
  Grad-CAM es el componente más frágil (~5–10 min/imagen en CPU, dependencias
  externas).
- Tiene `--skip-gradcam` para correr solo la parte rápida.
- Argumentos `--layer-idx 9` y `--head-reduction max` son los defaults que
  CLAUDE.md sec 8.2 documenta como los que dan mapas más interpretables.

**El notebook** muestra una galería automática por imagen×modelo, con
side-by-side de las dos figuras (cross vs gradcam) para base y fine-tuneado.
No hace inferencia.

**Aporte de D3 al pipeline:** este es el experimento más cercano a la
pregunta de investigación central del TP. Si D3 muestra heatmaps cambiados
clínicamente coherentes, el mode collapse pasa de problema a **hallazgo
reportable** (resultado tipo D del análisis en `soluciones_extra.md`).
**Es la decisión "el TP es viable como está o requiere reentrenamiento"**.

### 2.5. S1 — `run_s1_decoding_experiment.py` + `09_debug_decoding_strategies.ipynb`

**Pregunta que responde:** ¿cambiando solo el decoding (sin reentrenar) se
puede recuperar diversidad?

**Cómo:**
1. Carga `models/blip_finetuned_5k/best/` (un solo modelo, salvo
   `--include-base`).
2. Para cada imagen, prueba 6 estrategias en orden:
   - `greedy` (baseline).
   - `sample_t1.2_p0.95`, `sample_t1.3_p0.90`, `sample_t1.5_p0.85`
     (temperature + nucleus, agresividad creciente).
   - `diverse_beam` (8 beams, 4 grupos, diversity_penalty=0.5).
   - `contrastive` (penalty_alpha=0.6, top_k=4).
3. Para sampling, genera `samples_per_image` (default 4) captions por imagen
   para cuantificar la variabilidad. Para greedy, 1; para contrastive, 1;
   para diverse_beam, 4.
4. Captura excepciones por estrategia/imagen y las escribe a `s1_errors.csv`,
   para que un fallo de `contrastive` en una versión vieja de transformers
   no rompa todo el experimento.
5. Clasifica cada caption con un keyword matcher simple (`CLINICAL_KEYWORDS`
   en líneas 43–52: normal, atelectasis, effusion, edema, consolidation,
   pneumothorax, devices, cardiac). Esto permite medir si una estrategia
   no solo es más diversa sino también más **clínicamente específica**.
6. Genera 5 archivos: `s1_all_captions.csv` (raw), `s1_decoding_summary.csv`
   (1 fila por estrategia — la tabla del informe), `s1_image_strategy_summary.csv`
   (estrategia×imagen), `s1_examples.csv`, `s1_errors.csv`, y un PNG de
   comparación `s1_strategy_comparison.png`.

**El notebook** lee esos CSVs y muestra: ranking simple por estrategia,
gráficos de `unique_ratio` y `top_pct`, gráficos de `pct_normal` vs
`pct_specific_clinical`, ejemplos cualitativos por estrategia, casos donde
S1 mejora y casos donde no.

**Aporte de S1 al pipeline:** es la solución más barata posible. Si D2
muestra `gap` chico y S1 efectivamente recupera diversidad sin sacrificar
coherencia, **el problema se resuelve sin tocar el modelo**. Si S1 no
ayuda, queda confirmado que el collapse es de fondo y hay que ir a
soluciones más caras (S2 loss reweighting).

### 2.6. Cómo se conectan los 4 pasos

```
D1 ──────►  ¿cuándo aparece el collapse?
              │
              ├── temprano    → causa: incentivo / dataset
              └── progresivo  → causa: overfitting

D2 ──────►  ¿la distribución es puntiaguda?
              │
              ├── sí (mean_p_top1 > 0.90, gap > 0.5) → S1 no alcanza, ir a S2
              └── no (gap < 0.1)                      → S1 vale la pena

D3 ──────►  ¿los heatmaps cambiaron?
              │
              ├── sí → el TP es viable AUNQUE las captions colapsen
              └── no → necesitamos S1+S2 sí o sí para salvar el análisis

S1 ──────►  ¿decoding alternativo recupera diversidad textual?
              │
              ├── sí → usarlo como modo de generación final del TP
              └── no → confirma collapse internalizado → reentrenar
```

D1, D2, D3 son **independientes entre sí**. Pueden correrse en cualquier
orden o en paralelo. S1 depende conceptualmente de D2 (S1 solo tiene sentido
si D2 sugiere que hay alternativas), pero técnicamente también se puede
correr sin D2 — los resultados se interpretan después.

### 2.7. Calidad técnica general

Aspectos positivos:
- Separación script/notebook respeta CLAUDE.md.
- Buena ergonomía CLI: `--dry-run`, `--allow-debug`, `--num-threads`,
  `--device auto`, paths absolutos y relativos.
- Captura defensiva de excepciones (D3, S1) — un componente que falle no
  rompe el experimento entero.
- Outputs estructurados (CSV + JSON + figuras) que sirven tanto para
  notebooks como para incorporar al informe directamente.
- No tocan checkpoints, splits, ni archivos canónicos del proyecto.
- Reutilizan las funciones de `src/interpretability/` y `src/models/` sin
  reescribirlas.

Aspectos débiles:
- Sin tests unitarios (esperable en código de exploración, pero
  `summarize_collapse`, `summarize_by_image`, `classify_caption` se prestan).
- Algunos shadowings y guardas defensivas redundantes (M2, F1, F7, F8).
- Las advertencias sobre paths/checkpoints son print-based; sería mejor un
  `logger` para poder filtrar/parsear desde otros scripts.
- La documentación handoff es exhaustiva pero **asume una estructura de
  repo que no coincide al 100% con la actual** (B1, B2, B3).

Sin embargo, los problemas son de **integración**, no de **diseño**. El
código está bien escrito; solo necesita ajustes de paths y verificación de
prerequisitos antes de la primera corrida.

---

## Veredicto resumido

| Dimensión | Estado |
|---|---|
| Diseño conceptual | Bueno. D1/D2/D3/S1 cubren bien el espacio de preguntas. |
| Implementación | Buena. Código limpio, separación clara, manejo defensivo de errores. |
| Compatibilidad con `src/` actual | OK. Firmas y formatos de retorno son correctos. |
| Compatibilidad con estado del repo | **Mala**. B1–B4 requieren acción antes de poder correr. |
| Reproducibilidad | OK. Seeds fijas, índices fijos, outputs versionables. |
| Documentación | Muy buena. Cinco handoffs sumamente claros (quizás demasiado largos). |

**Recomendación:** resolver B1–B4 (todos son trabajo organizativo, ningún
cambio en el código de archivos_ion). Después, correr en este orden:
**D3 primero** (responde si el TP es viable como está), después
**D2** (decide si S1 vale la pena), después **S1** (la solución más
barata), y dejar **D1 al final** o saltearlo si los checkpoints
intermedios no se rescatan.
