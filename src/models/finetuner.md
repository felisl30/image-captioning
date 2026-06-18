# Guía para implementar `finetuner.py`

Este archivo es tu hoja de ruta para escribir el loop de fine-tuning de BLIP.
Cada sección dice **qué hacer** y **por qué**, sin darte el código terminado.

---

## Estructura general del archivo

```
finetuner.py
├── imports
├── freeze_model()          ← congela/descongela capas según la estrategia
├── build_optimizer()       ← AdamW con LR diferencial por grupo de parámetros
├── build_scheduler()       ← lineal con warmup
├── train_one_epoch()       ← un paso completo de entrenamiento
├── eval_one_epoch()        ← validación sin gradientes
├── run_finetuning()        ← función principal que orquesta todo
└── bloque __main__         ← CLI con argparse
```

---

## Paso 1 — `freeze_model(model)`

**Qué hacer:** congelar todo el modelo primero, luego descongelar selectivamente.

```python
# 1. Congela todo
for param in model.parameters():
    param.requires_grad = False

# 2. Descongela encoder ViT — solo los últimos 4 bloques (índices 8, 9, 10, 11)
for layer in model.vision_model.encoder.layers[8:]:
    for param in layer.parameters():
        param.requires_grad = True

# 3. Descongela decoder — bloques 4 al 11 (índices 4..11) + LM head
for layer in model.text_decoder.bert.encoder.layer[4:]:
    for param in layer.parameters():
        param.requires_grad = True

for param in model.text_decoder.cls.parameters():
    param.requires_grad = True
```

**Por qué:** los bloques tempranos del encoder aprenden bordes y texturas —
útiles en cualquier dominio, no hace falta re-aprenderlos. Los tardíos codifican
semántica ("perro", "árbol") que necesita adaptarse al dominio médico.
Lo mismo en el decoder: los bloques tardíos son los que especializan el estilo
y vocabulario de generación.

**Verificación útil después de llamar a esta función:**
```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
# Esperado: ~40–50% del total
```

---

## Paso 2 — `build_optimizer(model)`

**Qué hacer:** AdamW con dos grupos de parámetros a LR distintos.

Los parámetros del encoder descongelado van con LR más bajo (5e-6) porque
el encoder ya tiene buenas representaciones visuales y no queremos destruirlas.
El decoder con LR más alto (1e-5) porque su cambio de dominio es mayor.

```python
encoder_params = [
    p for p in model.vision_model.encoder.layers[8:].parameters()
    if p.requires_grad
]
decoder_params = [
    p for p in model.text_decoder.parameters()
    if p.requires_grad
]

optimizer = AdamW([
    {"params": encoder_params, "lr": 5e-6},
    {"params": decoder_params, "lr": 1e-5},
], weight_decay=0.01)
```

**Nota:** `weight_decay=0.01` es el valor estándar para transformers con AdamW.
Regulariza los pesos y reduce sobreajuste con datasets chicos.

---

## Paso 3 — `build_scheduler(optimizer, num_training_steps)`

**Qué hacer:** scheduler lineal que sube el LR gradualmente al principio
(warmup) y luego baja linealmente hasta 0.

```python
from transformers import get_linear_schedule_with_warmup

warmup_steps = int(0.05 * num_training_steps)  # 5% del total

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)
```

`num_training_steps = len(train_dataloader) * num_epochs`

**Por qué el warmup:** al principio del entrenamiento los gradientes son
grandes e inestables. Arrancar con LR bajo y subirlo evita que los primeros
batches destruyan los pesos preentrenados.

---

## Paso 4 — `train_one_epoch(model, dataloader, optimizer, scheduler, device)`

**Qué hacer:** iterar sobre el dataloader, calcular loss, hacer backward, step.

Estructura del loop:
```
model.train()
for batch in dataloader:
    1. mover batch a device
    2. forward pass → model(**batch_sin_idx_sin_text)
    3. loss = outputs.loss
    4. loss.backward()
    5. clip_grad_norm_(model.parameters(), max_norm=1.0)   ← importante
    6. optimizer.step()
    7. scheduler.step()
    8. optimizer.zero_grad()
    9. acumular loss para el promedio de la época
```

**Claves:**
- El batch del DataLoader incluye `idx` y `text` — BLIP no los espera, hay que
  sacarlos antes del forward: `{k: v.to(device) for k, v in batch.items() if k not in ("idx", "text")}`
- `clip_grad_norm_` con `max_norm=1.0` evita explosión de gradientes — típico en transformers
- `zero_grad()` va **después** del step, no antes, para no olvidarlo

**Return:** el loss promedio de la época (float).

---

## Paso 5 — `eval_one_epoch(model, dataloader, device)`

Igual que `train_one_epoch` pero:
- `model.eval()`
- todo dentro de `torch.no_grad()`
- sin backward, sin step, sin scheduler
- solo acumula y devuelve el loss promedio

Este número es el criterio para elegir el mejor checkpoint.

---

## Paso 6 — `run_finetuning(...)` — función principal

**Signature sugerida:**
```python
def run_finetuning(
    model,
    processor,
    train_dataloader,
    val_dataloader,
    num_epochs: int,
    output_dir: Path,
    device: str,
    patience: int = 2,       ← early stopping
) -> None:
```

**Estructura:**
```
1. freeze_model(model)
2. build_optimizer(model) → optimizer
3. num_training_steps = len(train_dataloader) * num_epochs
4. build_scheduler(optimizer, num_training_steps) → scheduler
5. best_val_loss = inf
6. epochs_without_improvement = 0

por cada época (1..num_epochs):
    a. train_loss = train_one_epoch(...)
    b. val_loss = eval_one_epoch(...)
    c. loguear: "Época N | train_loss=X | val_loss=Y"
    d. guardar checkpoint en output_dir/epoch_N/ con save_model()
    e. si val_loss < best_val_loss:
           guardar también en output_dir/best/
           best_val_loss = val_loss
           epochs_without_improvement = 0
       sino:
           epochs_without_improvement += 1
    f. si epochs_without_improvement >= patience: break (early stopping)
```

Para guardar el checkpoint usar la función `save_model` de `blip_loader.py`:
```python
from src.models.blip_loader import save_model
save_model(model, processor, output_dir / f"epoch_{epoch}")
```

---

## Paso 7 — bloque `__main__` (CLI)

Permite correr el fine-tuning desde consola o desde el notebook con `!python -m src.models.finetuner ...`.

Argumentos mínimos con `argparse`:
- `--train-indices` — path a `data/splits/train_indices.json`
- `--val-indices` — path a `data/splits/val_indices.json`
- `--output-dir` — path a `models/blip_finetuned/`
- `--epochs` — int, default 3
- `--batch-size` — int, default 4 (en Kaggle T4 podés subir a 8)
- `--device` — "cuda" / "cpu" / None (autodetectar)

Lo que hace el bloque:
1. Parsear args
2. Cargar modelo con `load_model_and_processor()`
3. Cargar dataset HF desde `data/hf_cache/`
4. Crear los dos DataLoaders con `create_dataloader()`
5. Llamar a `run_finetuning()`

---

## Checklist antes de correr en GPU

- [ ] `freeze_model` imprime ~40–50% de parámetros entrenables
- [ ] Un batch de entrenamiento no da error (probar con `batch_size=1` en CPU)
- [ ] El loss de la primera época baja (si sube, el LR es demasiado alto)
- [ ] Se crea `models/blip_finetuned/epoch_1/` con `config.json` + `model.safetensors`
- [ ] Se crea `models/blip_finetuned/best/` al final de la época con menor val loss

---

## Errores comunes

| Error | Causa probable | Fix |
|---|---|---|
| `RuntimeError: Expected all tensors to be on the same device` | batch no movido a device | agregar `.to(device)` en cada tensor del batch |
| `KeyError: 'idx'` dentro de BLIP | `idx` y `text` no filtrados antes del forward | excluirlos del dict antes de pasarlo al modelo |
| Loss = NaN desde el primer batch | LR demasiado alto o gradiente explota | bajar LR, verificar clip_grad_norm_ |
| CUDA OOM | batch_size demasiado grande | bajar a 2 y usar gradient accumulation |
| `AttributeError: text_decoder` | modelo sin `.eval()` antes de acceder | no es el problema; verificar que cargaste `BlipForConditionalGeneration` |
