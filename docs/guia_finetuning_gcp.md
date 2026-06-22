# Guía de ejecución del fine-tuning en GCP

## Qué hace la última celda del notebook (Sección 6 — Sanity check del checkpoint)

Después de terminar el entrenamiento completo, la última celda verifica que el modelo realmente aprendió algo útil:

1. Carga `models/blip_finetuned/best/` desde disco.
2. Toma el primer índice de `data/selected_indices.json` — una de las 30 radiografías fijas que se usan para la comparación antes/después.
3. Genera un caption con el modelo fine-tuneado.
4. Imprime el caption generado junto al `impression` de referencia del dataset.
5. Muestra la imagen.

**Por qué importa:** es la primera verificación cualitativa del modelo. Si el caption generado tiene sentido clínico (aunque sea parcial) y se parece al `impression`, el entrenamiento funcionó. Si el output es genérico ("a chest x-ray of a patient") o absurdo, algo salió mal y conviene revisar el historial de loss antes de bajar el checkpoint.

En la VM local esta celda no hace nada porque `models/blip_finetuned/best/` no existe todavía. En GCP, ejecutarla al final del entrenamiento es obligatorio.

---

## Qué transferir a la VM antes de empezar

### Archivos que van en git (ya están)

```
data/splits/train_indices.json
data/splits/val_indices.json
data/splits/test_indices.json
data/selected_indices.json
src/
notebooks/03_finetuning.ipynb
requirements.txt
```

### Archivos que NO van en git y hay que transferir manualmente

| Qué | Tamaño aprox | Cómo |
|---|---|---|
| `data/hf_cache/` | ~800 MB | `gcloud storage cp -r gs://BUCKET/hf_cache data/` o dejar que se descargue en la VM |
| `models/blip_base/` | ~1 GB | Ídem — o descargar desde HuggingFace en la VM |

Si la VM tiene buena conectividad a HuggingFace, lo más simple es **no transferir nada** y dejar que el script descargue el dataset y el modelo base directamente. Solo es obligatorio transferir los splits JSON (que ya están en git).

---

## Setup de la VM

```bash
# 1. Clonar el repo
git clone <url-del-repo> image-captioning
cd image-captioning

# 2. Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verificar CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Esperado: True  NVIDIA L4 (o T4, según la VM)

# 4. Opcional pero recomendado: setear HF_TOKEN para evitar rate-limiting
export HF_TOKEN="hf_..."
# O agregar al .bashrc de la VM para que persista entre sesiones

# 5. Verificar que los splits existen
ls data/splits/
# Debe mostrar: train_indices.json  val_indices.json  test_indices.json

ls data/selected_indices.json
# Debe existir
```

---

## Ejecución del notebook 03

Abrir Jupyter en la VM y ejecutar las celdas en orden. Lo importante de cada sección:

### Sección 0 — Setup del entorno
Detecta la raíz del repo, imprime la versión de PyTorch y confirma si hay CUDA.
**Verificar que diga `CUDA disponible: True` y muestre el nombre de la GPU antes de continuar.**

### Sección 1 — Paths
Define todas las rutas. Si el dataset o el modelo base no están en disco, los va a descargar cuando el finetuner los pida.

### Sección 2 — Verificación de splits
Comprueba que no hay solapamiento entre train/val/test y que `selected` está contenido en `test`.
**No continuar si algún chequeo falla.**

### Sección 3 — Smoke test CPU
Corre 1 batch de train + 1 de val en CPU para verificar que el pipeline funciona.
**En la VM dejar `RUN_LOCAL_SMOKE_TEST = True` y ejecutar igual** — es rápido y confirma que no hay errores de importación ni de paths antes de tocar la GPU.

### Sección 3.1 — Verificación del historial de smoke test
Confirma que `training_history.json` se escribió correctamente. No saltar.

### Sección 4 — Smoke test GPU
**Cambiar `RUN_GPU_SMOKE_TEST = False` → `True`.**

Corre 2 batches train + 1 val en CUDA con `batch_size=2`. Objetivo: confirmar que:
- No hay `CUDA OOM`
- El loss es un número finito
- El pipeline de forward+backward funciona en GPU

Si hay OOM acá con `batch_size=2`, la VM no tiene suficiente VRAM (poco probable en T4 o L4).

### Sección 4 — Fine-tuning completo
**Cambiar `RUN_FULL_TRAINING = False` → `True`.**

El comando que corre es:
```bash
python -m src.models.finetuner \
    --train-indices data/splits/train_indices.json \
    --val-indices data/splits/val_indices.json \
    --output-dir models/blip_finetuned \
    --epochs 3 \
    --batch-size 4 \
    --device cuda \
    --text-col impression
```

**Parámetros a ajustar según la GPU:**

| GPU | batch_size recomendado | Tiempo estimado por época |
|---|---|---|
| T4 (16 GB) | 4 | ~2.5 h |
| L4 (24 GB) | 8 | ~1.5 h |
| A100 (40 GB) | 16 | ~45 min |

Para cambiar el batch size en la VM, editar la celda y cambiar `"--batch-size", "4"` al valor correspondiente.

**Qué mirar durante el entrenamiento:**
- El loss de train debe bajar entre épocas. Si sube, el LR es demasiado alto (no debería pasar con los defaults).
- El loss de val debe bajar también. Si train baja pero val sube desde la época 1, hay overfitting — con 3 épocas es poco probable.
- El early stopping se activa si val no mejora 2 épocas seguidas (`patience=2`).
- Los checkpoints se guardan en `models/blip_finetuned/epoch_N/` y el mejor en `models/blip_finetuned/best/`.

### Sección 5 — Verificación del checkpoint
Comprueba que `models/blip_finetuned/best/` existe y muestra el historial de loss.
Si `best/` no existe, el entrenamiento no terminó o hubo un error.

### Sección 5 — Curva de entrenamiento
Grafica `train_loss` y `val_loss` por época. Guardar la figura — va al informe.

### Sección 6 — Sanity check del checkpoint fine-tuneado
Carga `best/`, genera un caption sobre la primera radiografía de `selected_indices.json` y muestra la imagen.

**Qué esperar:**
- Caption antes del fine-tuning (BLIP base): algo genérico como *"a chest x-ray of a patient"*
- Caption después del fine-tuning: debería mencionar hallazgos específicos (*"no acute cardiopulmonary disease"*, *"mild cardiomegaly"*, etc.)

Si el caption sigue siendo genérico, el modelo no aprendió suficiente — revisar el historial de loss.

---

## Descargar el checkpoint al finalizar

```bash
# Desde la VM, comprimir el checkpoint
tar -czf blip_finetuned_best.tar.gz models/blip_finetuned/best/

# Descargar a la máquina local con gcloud o scp
gcloud compute scp VM_NAME:~/image-captioning/blip_finetuned_best.tar.gz .

# También descargar el historial y la figura
gcloud compute scp VM_NAME:~/image-captioning/models/blip_finetuned/training_history.json .
gcloud compute scp VM_NAME:~/image-captioning/outputs/finetuning/loss_curve.png .
```

El checkpoint `best/` es lo único necesario para el notebook 04 (`analisis_postft.ipynb`). Los checkpoints `epoch_N/` se pueden borrar de la VM para ahorrar espacio.

---

## Checklist antes de apagar la VM

- [ ] `models/blip_finetuned/best/` contiene `config.json`, `model.safetensors` y archivos del processor
- [ ] `models/blip_finetuned/training_history.json` descargado
- [ ] `outputs/finetuning/loss_curve.png` descargado
- [ ] Sanity check ejecutado y caption generado tiene sentido clínico
- [ ] Checkpoint `best/` descargado y descomprimido en la máquina local

---

## Errores comunes en GCP

| Error | Causa | Fix |
|---|---|---|
| `CUDA out of memory` | batch_size demasiado grande | Bajar de 8 a 4 |
| `FileNotFoundError: data/splits/...` | Cloné el repo pero los splits no están en git | Copiarlos manualmente o regenerar con `python -m src.data.split_generator --seed 42` |
| `KeyError: 'impression'` | dataset con schema distinto al esperado | Verificar `ds["train"].column_names` |
| `RuntimeError: Expected all tensors on same device` | algún tensor no se movió a cuda | No debería pasar — `prepare_batch` mueve todo |
| Warning de rate-limiting de HuggingFace | Sin HF_TOKEN | `export HF_TOKEN="hf_..."` |
| Loss = NaN desde el primer batch | Imagen corrupta o LR demasiado alto | Verificar con `--max-train-batches 1 --batch-size 1`; los defaults de LR son seguros |
