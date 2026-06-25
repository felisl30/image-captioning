# Handoff técnico — D3: heatmap probe base vs fine-tuned

Este documento explica, para otro modelo de lenguaje o asistente de código, qué hacen los archivos relacionados con **D3** dentro del proyecto de fine-tuning de BLIP sobre radiografías.

## 1. Contexto general

El proyecto fine-tunea `Salesforce/blip-image-captioning-base` sobre radiografías de tórax del dataset MIMIC-CXR usando el campo `impression` como target textual.

Después del fine-tuning apareció un problema de **mode collapse**: el modelo tiende a generar captions repetidas y dominantes, por ejemplo:

```text
no acute cardiopulmonary process.
no significant interval change.
```

D1 y D2 diagnostican ese collapse desde el lado textual:

```text
D1 → ¿cuándo aparece el collapse entre epoch_1, epoch_2, epoch_3 y best?
D2 → ¿el decoder está realmente muy seguro token a token, o greedy decoding oculta alternativas?
```

D3 responde una pregunta distinta y más cercana a la pregunta central del TP:

```text
Aunque las captions estén colapsadas, ¿los heatmaps del modelo fine-tuneado cambiaron respecto de BLIP base?
```

La idea es mirar si el fine-tuning modificó la parte visual / interpretativa del modelo, incluso si el decoder textual sigue generando frases repetidas.

## 2. Archivos involucrados en D3

Los archivos relevantes son:

```text
scripts/run_d3_heatmap_probe.py
notebooks/08_debug_heatmaps_probe.ipynb
```

Y los outputs esperados se guardan en:

```text
outputs/mode_collapse_debug/d3_heatmaps/
```

Estructura esperada:

```text
image-captioning/
├── scripts/
│   └── run_d3_heatmap_probe.py
├── notebooks/
│   └── 08_debug_heatmaps_probe.ipynb
├── data/
│   └── selected_indices.json
├── models/
│   ├── blip_base/
│   └── blip_finetuned_5k/
│       └── best/
└── outputs/
    └── mode_collapse_debug/
        └── d3_heatmaps/
            ├── d3_heatmap_summary.csv
            ├── d3_heatmap_summary.json
            ├── idx_<IDX>/
            │   ├── base/
            │   │   ├── original.png
            │   │   ├── cross_att_logits_grid.png
            │   │   ├── gradcam_grid.png
            │   │   └── cross_vs_gradcam.png
            │   └── finetuned/
            │       ├── original.png
            │       ├── cross_att_logits_grid.png
            │       ├── gradcam_grid.png
            │       └── cross_vs_gradcam.png
            └── ...
```

## 3. Qué es D3

D3 es un diagnóstico visual de interpretabilidad.

Compara dos modelos:

```text
base       → models/blip_base/
finetuned  → models/blip_finetuned_5k/best/ o models/blip_finetuned/best/
```

sobre las mismas radiografías fijas:

```text
data/selected_indices.json
```

Para cada imagen y cada modelo, extrae dos señales visuales:

```text
1. QK logits / cross-attention por palabra
2. Grad-CAM por palabra
```

Y guarda figuras para comparar:

```text
base vs finetuned
cross-attention vs Grad-CAM
```

D3 no entrena, no modifica pesos y no escribe dentro de `models/`.

## 4. `scripts/run_d3_heatmap_probe.py`

### Propósito

Este script es el **motor real de D3**.

Carga BLIP base y BLIP fine-tuneado, corre ambos sobre pocas radiografías, extrae mapas de interpretabilidad por palabra y guarda figuras + CSV/JSON resumen.

El docstring del script resume su propósito así:

```text
Compara BLIP base contra BLIP fine-tuneado best/ en pocas radiografías fijas.

Genera:
- captions base y fine-tuned;
- grillas de QK logits / cross-attention por palabra;
- grillas de Grad-CAM por palabra;
- comparación cross-attention vs Grad-CAM;
- CSV/JSON resumen.

No entrena. No modifica checkpoints.
```

### Modelos usados

Por defecto, el modelo base se espera en:

```text
models/blip_base/
```

El modelo fine-tuneado se busca en:

```text
models/blip_finetuned_5k/best/
models/blip_finetuned/best/
```

También puede usar checkpoints debug si se pasa:

```bash
--allow-debug
```

En ese caso busca en:

```text
models/blip_finetuned_debug_save/best/
models/blip_finetuned_debug/best/
models/blip_finetuned_notebook_debug/best/
```

### Qué considera un checkpoint usable

Una carpeta se considera checkpoint usable si contiene:

```text
config.json
```

y además alguno de estos pesos:

```text
model.safetensors
pytorch_model.bin
```

El script también reporta si encuentra:

```text
processor_config.json
preprocessor_config.json
tokenizer.json
generation_config.json
```

### Imágenes usadas

Por defecto usa:

```text
data/selected_indices.json
```

con:

```bash
--max-images 5
```

D3 usa menos imágenes que D1/D2 porque Grad-CAM es mucho más costoso: requiere backward pass por token.

### Función `run_cross_att_logits`

Esta función llama a:

```python
from src.interpretability.cross_att_logits import extract_cross_att_logits
```

y extrae mapas QK logits / cross-attention por palabra.

Conceptualmente:

```text
imagen → BLIP processor → pixel_values → extract_cross_att_logits
```

La salida esperada de `extract_cross_att_logits` es una lista de diccionarios con esta forma:

```python
{
    "caption": "...",
    "maps": [
        ("word_1", array_24x24),
        ("word_2", array_24x24),
        ...
    ]
}
```

En este proyecto, `cross_att_logits.py` usa logits pre-softmax `Q·K^T / sqrt(d)` como proxy de afinidad espacial por palabra.

### Función `run_gradcam`

Esta función llama a:

```python
from src.interpretability.gradcam import compute_gradcam
```

y extrae mapas Grad-CAM por palabra.

Conceptualmente:

```text
imagen → BLIP processor → model.generate() → tokens generados
      → teacher forcing token a token
      → Grad-CAM sobre encoder ViT
```

La salida tiene el mismo formato esperado que cross-attention:

```python
{
    "caption": "...",
    "maps": [
        ("word_1", array_24x24),
        ("word_2", array_24x24),
        ...
    ]
}
```

### Función `save_heatmap_grid`

Esta función usa:

```python
from src.visualization.heatmap import save_heatmap_grid
```

para guardar una grilla de mapas por palabra.

Archivos generados:

```text
cross_att_logits_grid.png
gradcam_grid.png
```

### Función `save_comparison`

Esta función usa:

```python
from src.visualization.heatmap import plot_comparison_heatmaps
```

para guardar una comparación entre cross-attention y Grad-CAM.

Archivo generado:

```text
cross_vs_gradcam.png
```

### Función `run_for_model`

Esta es la función central del script.

Para un modelo específico (`base` o `finetuned`):

1. carga modelo y processor con `load_model_and_processor`;
2. recorre los índices de `selected_indices.json`;
3. carga cada radiografía del dataset;
4. guarda la imagen original;
5. corre QK logits / cross-attention;
6. corre Grad-CAM, salvo que se haya pasado `--skip-gradcam`;
7. guarda figuras;
8. acumula una fila de resumen.

La fila de resumen contiene:

```text
model_tag
model_path
idx
reference
original_path
cross_caption
gradcam_caption
n_cross_maps
n_gradcam_maps
cross_grid_path
gradcam_grid_path
comparison_path
cross_status
gradcam_status
error
```

## 5. Outputs de D3

### `d3_heatmap_summary.csv`

Archivo principal de resumen.

Una fila por:

```text
modelo × imagen
```

Ejemplo:

```text
base       × idx_12345
finetuned  × idx_12345
base       × idx_67890
finetuned  × idx_67890
```

Columnas clave:

```text
model_tag
idx
reference
cross_caption
gradcam_caption
cross_status
gradcam_status
n_cross_maps
n_gradcam_maps
cross_grid_path
gradcam_grid_path
comparison_path
error
```

### `d3_heatmap_summary.json`

Mismo contenido que el CSV, pero en JSON.

Uso:

```text
Consumo programático por notebooks, scripts o modelos de lenguaje.
```

### `original.png`

Imagen radiográfica original convertida a RGB.

Se guarda dentro de:

```text
outputs/mode_collapse_debug/d3_heatmaps/idx_<IDX>/<model_tag>/original.png
```

Aunque la imagen original sea igual para `base` y `finetuned`, se guarda en ambas carpetas para que cada caso sea autocontenido.

### `cross_att_logits_grid.png`

Grilla de mapas QK logits / cross-attention por palabra.

Se guarda por imagen y modelo:

```text
idx_<IDX>/base/cross_att_logits_grid.png
idx_<IDX>/finetuned/cross_att_logits_grid.png
```

Sirve para ver qué regiones del espacio visual tienen mayor afinidad QK con cada token generado.

### `gradcam_grid.png`

Grilla de mapas Grad-CAM por palabra.

Se guarda por imagen y modelo:

```text
idx_<IDX>/base/gradcam_grid.png
idx_<IDX>/finetuned/gradcam_grid.png
```

Sirve para ver qué regiones del encoder visual influyen más en el logit de cada token.

### `cross_vs_gradcam.png`

Comparación visual entre QK logits / cross-attention y Grad-CAM.

Se genera solo si ambos métodos corrieron correctamente.

Se guarda en:

```text
idx_<IDX>/<model_tag>/cross_vs_gradcam.png
```

## 6. `notebooks/08_debug_heatmaps_probe.ipynb`

### Propósito

Este notebook es el **lector y visualizador de D3**.

No debería extraer heatmaps directamente. La extracción pesada vive en:

```text
scripts/run_d3_heatmap_probe.py
```

El notebook sirve para:

1. mostrar comandos D3;
2. verificar si existen los outputs;
3. cargar `d3_heatmap_summary.csv`;
4. mostrar la tabla resumen;
5. comparar captions base vs fine-tuned;
6. revisar qué métodos fallaron o funcionaron;
7. mostrar galerías de imágenes;
8. ayudar a escribir una conclusión para el informe.

### Secciones del notebook

El notebook contiene estas secciones:

```text
1. Comandos D3
2. Verificación de outputs
3. Resumen compacto
4. Comparación de captions base vs fine-tuned
5. Estado de ejecución por modelo
6. Galería de una imagen
7. Galería automática de todos los casos
8. Tabla de archivos generados
9. Interpretación preliminar por caso
10. Conclusión para informe
```

### Qué muestra visualmente

Para cada índice disponible, el notebook muestra:

```text
base/original.png
base/cross_att_logits_grid.png
base/gradcam_grid.png
base/cross_vs_gradcam.png

finetuned/original.png
finetuned/cross_att_logits_grid.png
finetuned/gradcam_grid.png
finetuned/cross_vs_gradcam.png
```

Si un archivo no existe, el notebook lo marca como faltante en lugar de romper.

## 7. Comandos principales de D3

### Verificar sin correr nada pesado

```bash
python scripts/run_d3_heatmap_probe.py --dry-run
```

### Corrida real con checkpoint fine-tuneado

```bash
python scripts/run_d3_heatmap_probe.py \
  --ft-model-dir models/blip_finetuned_5k/best \
  --indices data/selected_indices.json \
  --max-images 5 \
  --device cpu
```

### Corrida real usando búsqueda automática del checkpoint

```bash
python scripts/run_d3_heatmap_probe.py \
  --indices data/selected_indices.json \
  --max-images 5 \
  --device cpu
```

El script buscará automáticamente:

```text
models/blip_finetuned_5k/best
models/blip_finetuned/best
```

### Smoke test con checkpoint debug

```bash
python scripts/run_d3_heatmap_probe.py \
  --allow-debug \
  --max-images 1 \
  --device cpu
```

### Probar solo cross-attention/QK logits, sin Grad-CAM

Útil si falta `grad-cam` o si se quiere validar primero la parte más liviana:

```bash
python scripts/run_d3_heatmap_probe.py \
  --ft-model-dir models/blip_finetuned_5k/best \
  --indices data/selected_indices.json \
  --max-images 2 \
  --device cpu \
  --skip-gradcam
```

### Cambiar capa de QK logits

Por defecto:

```bash
--layer-idx 9
```

Para probar otra capa:

```bash
python scripts/run_d3_heatmap_probe.py \
  --ft-model-dir models/blip_finetuned_5k/best \
  --layer-idx 8
```

### Cambiar agregación de heads

Por defecto:

```bash
--head-reduction max
```

Alternativa:

```bash
--head-reduction mean
```

`max` tiende a dar mapas más nítidos. `mean` tiende a dar mapas más suaves.

## 8. Dependencias de D3

D3 necesita:

```text
torch
transformers
datasets
pandas
Pillow
matplotlib
grad-cam
```

El import real del paquete de Grad-CAM es:

```python
pytorch_grad_cam
```

Pero se instala con:

```bash
pip install grad-cam
```

Si falta, el script avisa:

```text
ADVERTENCIA: falta pytorch_grad_cam.
Para D3 completo instalá:
  pip install grad-cam
```

Para correr sin Grad-CAM:

```bash
--skip-gradcam
```

## 9. Cómo interpretar D3

### Caso A — Fine-tuned cambia heatmaps de forma coherente

Patrón:

```text
base: mapas difusos, externos al pulmón, poco interpretables
finetuned: mapas más concentrados en campos pulmonares, silueta cardíaca, dispositivos o zonas patológicas
```

Interpretación:

```text
El fine-tuning modificó la representación visual del modelo. Aunque las captions estén colapsadas, hay evidencia de adaptación visual.
```

Este es un resultado reportable.

### Caso B — Captions colapsan, pero heatmaps cambian

Patrón:

```text
cross_caption / gradcam_caption repetidos
pero mapas fine-tuned claramente distintos de base
```

Interpretación:

```text
El encoder o el puente visual-lingüístico pudo haber aprendido a mirar distinto, aunque el decoder no exprese bien esa diversidad.
```

Este es un resultado muy interesante para el informe:

```text
El modelo aprendió parcialmente a ver, pero no aprendió bien a hablar.
```

### Caso C — Heatmaps base y fine-tuned son casi iguales

Patrón:

```text
base y finetuned miran regiones similares
no hay mejora visual clara
```

Interpretación:

```text
El fine-tuning puede haber afectado principalmente el lenguaje o haber sido insuficiente.
```

Siguiente paso:

```text
Revisar D1/D2, usar checkpoint intermedio, o considerar una corrida de fine-tuning mejor.
```

### Caso D — Cross-attention y Grad-CAM divergen

Patrón:

```text
cross_att_logits_grid mira una zona
gradcam_grid mira otra zona
```

Interpretación:

```text
Las dos señales de interpretabilidad no están alineadas. Hay que reportarlo como limitación o como resultado negativo parcial.
```

Dado que cross-attention/QK logits y Grad-CAM miden cosas distintas, no se espera coincidencia perfecta.

## 10. Relación con D1 y D2

D1 y D2 miran el lado textual del problema.

```text
D1 → diversidad de captions por checkpoint.
D2 → distribución top-1/top-2 por token.
```

D3 mira el lado visual / interpretativo:

```text
D3 → heatmaps base vs fine-tuned.
```

Relación práctica:

```text
D1 puede decir que best está colapsado.
D2 puede decir que el decoder está muy confiado.
D3 puede decir si, pese a eso, los heatmaps cambiaron de forma útil.
```

La conclusión más importante puede surgir de D3:

```text
Si los heatmaps mejoran aunque las captions colapsen, el mode collapse no invalida el TP; pasa a ser una limitación metodológica y un hallazgo.
```

## 11. Relación con el objetivo central del TP

La pregunta central del proyecto es si el fine-tuning de BLIP sobre radiografías cambia no solo el lenguaje, sino también la forma en que el modelo mira regiones clínicamente relevantes.

D3 es el diagnóstico más cercano a esa pregunta.

Puede sostener tres tipos de conclusión:

```text
A. Captions mejoran y heatmaps mejoran.
B. Captions no mejoran bien, pero heatmaps sí cambian.
C. Ni captions ni heatmaps muestran mejora clara.
```

Todos son resultados válidos si están bien documentados.

## 12. Qué no debe hacer otro modelo al tocar D3

No debe:

- entrenar modelos;
- modificar checkpoints;
- borrar `models/blip_base/`;
- borrar `models/blip_finetuned_5k/best/`;
- regenerar `selected_indices.json`;
- cambiar el dataset;
- mezclar D3 con métricas BLEU/CIDEr;
- reescribir `src/interpretability/` sin necesidad;
- convertir el notebook en un script pesado;
- correr Grad-CAM sobre 30 imágenes sin advertir el costo;
- usar modelos distintos a BLIP base sin autorización.

D3 debe mantenerse como un probe visual acotado.

## 13. Costos computacionales

D3 es el más pesado de los diagnósticos D1-D2-D3.

Motivo:

```text
Grad-CAM requiere backward pass por token.
```

Costo aproximado:

```text
n_modelos × n_imágenes × n_tokens × backward
```

Con:

```text
2 modelos × 5 imágenes × 15-25 tokens
```

ya puede tardar bastante en CPU.

Por eso el default es:

```bash
--max-images 5
```

y existe:

```bash
--skip-gradcam
```

para probar primero solo la parte de QK logits / cross-attention.

## 14. Checklist antes de correr D3 real

Antes de correr D3, verificar:

```text
[ ] Existe models/blip_base/
[ ] Existe models/blip_finetuned_5k/best/ o models/blip_finetuned/best/
[ ] Existe data/selected_indices.json
[ ] Existe data/hf_cache/ o hay conexión para descargar dataset
[ ] Está instalado grad-cam, salvo que se use --skip-gradcam
[ ] El entorno tiene suficiente RAM/VRAM
[ ] Se corre primero con --max-images 1 o --max-images 2 como smoke test
```

## 15. Checklist para otro LLM antes de modificar algo

Antes de tocar código de D3, verificar:

```text
[ ] El script sigue comparando base vs finetuned.
[ ] El script sigue usando las mismas imágenes de selected_indices.json.
[ ] El script sigue guardando d3_heatmap_summary.csv.
[ ] El script sigue guardando figuras por idx/model_tag.
[ ] El notebook sigue siendo liviano y solo lee outputs.
[ ] No se mezcló D3 con entrenamiento.
[ ] No se modificó selected_indices.json.
[ ] No se cambió BLIP por otro modelo.
[ ] No se eliminó --skip-gradcam.
[ ] No se eliminó --dry-run.
```

## 16. Resumen ejecutivo para otro LLM

D3 es el probe visual del proyecto. El script `scripts/run_d3_heatmap_probe.py` compara BLIP base contra BLIP fine-tuneado `best/` sobre pocas radiografías fijas de `data/selected_indices.json`. Para cada imagen y modelo extrae mapas QK logits / cross-attention por palabra mediante `extract_cross_att_logits`, mapas Grad-CAM por palabra mediante `compute_gradcam`, y guarda grillas visuales y comparaciones en `outputs/mode_collapse_debug/d3_heatmaps/`. También guarda `d3_heatmap_summary.csv` y `.json` con captions, paths de figuras, estados de ejecución y errores. El notebook `notebooks/08_debug_heatmaps_probe.ipynb` no extrae heatmaps; solo lee el summary, muestra tablas y despliega galerías de imágenes para comparar `base` vs `finetuned`. D3 sirve para decidir si el fine-tuning modificó la mirada del modelo aunque el decoder textual esté colapsado. No entrena, no modifica checkpoints y debe correrse con pocas imágenes, especialmente si Grad-CAM está activado.
