"""Fine-tuning de BLIP sobre MIMIC-CXR.

Este módulo implementa el loop de entrenamiento de BLIP para adaptar
Salesforce/blip-image-captioning-base al dominio de radiografías de tórax.

Uso típico:

    python -m src.models.finetuner \
        --train-indices data/splits/train_indices.json \
        --val-indices data/splits/val_indices.json \
        --output-dir models/blip_finetuned \
        --epochs 3 \
        --batch-size 4 \
        --device cuda

Para debug local en CPU:

    python -m src.models.finetuner \
        --output-dir models/blip_finetuned_debug \
        --epochs 1 \
        --batch-size 1 \
        --device cpu \
        --max-train-batches 2 \
        --max-val-batches 1 \
        --skip-checkpoint-save
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers import get_linear_schedule_with_warmup

from src.data.dataloader import create_dataloader
from src.data.utils import load_mimic_dataset
from src.models.blip_loader import load_model_and_processor, save_model


logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Detecta el mejor device disponible."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Fija seeds para reproducibilidad básica."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_model(model: BlipForConditionalGeneration) -> dict[str, float]:
    """Congela/descongela capas de BLIP según la estrategia de fine-tuning.

    Estrategia:
    - Congelar todo el modelo.
    - Descongelar últimos 4 bloques del encoder visual ViT.
    - Descongelar bloques 4..11 del decoder de texto.
    - Descongelar LM head del decoder.

    Args:
        model: BLIP cargado como BlipForConditionalGeneration.

    Returns:
        Diccionario con conteo de parámetros totales y entrenables.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Encoder visual: últimos 4 bloques ViT.
    for layer in model.vision_model.encoder.layers[8:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Decoder de texto: bloques 4..11.
    for layer in model.text_decoder.bert.encoder.layer[4:]:
        for param in layer.parameters():
            param.requires_grad = True

    # LM head del decoder.
    for param in model.text_decoder.cls.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100.0 * trainable_params / total_params

    stats = {
        "total_params": float(total_params),
        "trainable_params": float(trainable_params),
        "trainable_pct": trainable_pct,
    }

    logger.info(
        "Parámetros entrenables: %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        trainable_pct,
    )

    return stats


def build_optimizer(
    model: BlipForConditionalGeneration,
    encoder_lr: float = 5e-6,
    decoder_lr: float = 1e-5,
    weight_decay: float = 0.01,
) -> AdamW:
    """Construye AdamW con LR diferencial para encoder y decoder.

    Args:
        model: BLIP con requires_grad ya configurado.
        encoder_lr: learning rate para bloques visuales descongelados.
        decoder_lr: learning rate para decoder de texto descongelado.
        weight_decay: regularización AdamW.

    Returns:
        Optimizador AdamW.
    """
    encoder_params = [
        p
        for p in model.vision_model.encoder.layers[8:].parameters()
        if p.requires_grad
    ]

    decoder_params = [
        p
        for p in model.text_decoder.parameters()
        if p.requires_grad
    ]

    if not encoder_params:
        raise ValueError("No hay parámetros entrenables en el encoder visual.")

    if not decoder_params:
        raise ValueError("No hay parámetros entrenables en el decoder de texto.")

    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )

    logger.info(
        "AdamW creado: encoder_lr=%g, decoder_lr=%g, weight_decay=%g",
        encoder_lr,
        decoder_lr,
        weight_decay,
    )

    return optimizer


def build_scheduler(
    optimizer: AdamW,
    num_training_steps: int,
    warmup_ratio: float = 0.05,
):
    """Construye scheduler lineal con warmup.

    Args:
        optimizer: AdamW.
        num_training_steps: cantidad total de optimizer steps.
        warmup_ratio: fracción inicial usada como warmup.

    Returns:
        Scheduler de HuggingFace.
    """
    if num_training_steps <= 0:
        raise ValueError("num_training_steps debe ser positivo.")

    num_warmup_steps = int(warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    logger.info(
        "Scheduler creado: training_steps=%d, warmup_steps=%d",
        num_training_steps,
        num_warmup_steps,
    )

    return scheduler


def prepare_batch(batch: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    """Filtra y mueve un batch al device.

    El DataLoader devuelve también `idx` y `text`, pero BLIP no acepta esas claves
    en el forward. Por eso se excluyen.

    Args:
        batch: batch devuelto por DataLoader.
        device: "cpu", "cuda" o "mps".

    Returns:
        Diccionario compatible con model(**batch).
    """
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key not in ("idx", "text")
    }


def train_one_epoch(
    model: BlipForConditionalGeneration,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: str,
    max_batches: int | None = None,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Entrena una época.

    Args:
        model: BLIP.
        dataloader: DataLoader de entrenamiento.
        optimizer: AdamW.
        scheduler: scheduler lineal.
        device: device de ejecución.
        max_batches: si no es None, corta la época tras esa cantidad de batches.
        grad_clip_norm: norma máxima para clipping de gradientes.
        grad_accum_steps: cantidad de micro-batches antes de hacer optimizer.step().
        scaler: GradScaler para AMP. None desactiva AMP.

    Returns:
        Loss promedio de la época.
    """
    model.train()

    total_loss = 0.0
    n_batches = 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    amp_ctx = torch.cuda.amp.autocast if scaler is not None else nullcontext

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader, start=1):
        if max_batches is not None and step > max_batches:
            break

        model_inputs = prepare_batch(batch, device)

        with amp_ctx():
            outputs = model(**model_inputs)
            loss = outputs.loss

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss no finita en train step {step}: {loss}")

        scaled_loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if step % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu())
        n_batches += 1

        logger.info(
            "train step %d/%s | loss=%.4f",
            step,
            max_batches if max_batches is not None else len(dataloader),
            float(loss.detach().cpu()),
        )

    # flush gradientes acumulados si los batches no son múltiplo de grad_accum_steps
    if n_batches % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if n_batches == 0:
        raise RuntimeError("train_one_epoch no procesó ningún batch.")

    return total_loss / n_batches


def eval_one_epoch(
    model: BlipForConditionalGeneration,
    dataloader: DataLoader,
    device: str,
    max_batches: int | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    """Evalúa una época sin gradientes.

    Args:
        model: BLIP.
        dataloader: DataLoader de validación.
        device: device de ejecución.
        max_batches: si no es None, corta la evaluación tras esa cantidad de batches.
        use_amp: si True, usa autocast para el forward pass.

    Returns:
        Tupla (val_loss, token_accuracy). token_accuracy mide el porcentaje de tokens
        donde argmax(logits) == label, excluyendo padding (-100).
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    amp_ctx = torch.cuda.amp.autocast if (use_amp and device == "cuda") else nullcontext

    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            if max_batches is not None and step > max_batches:
                break

            model_inputs = prepare_batch(batch, device)

            with amp_ctx():
                outputs = model(**model_inputs)

            loss = outputs.loss

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss no finita en val step {step}: {loss}")

            # token accuracy
            labels = model_inputs["labels"]
            preds = outputs.logits.argmax(dim=-1)
            mask = labels != -100
            total_correct += int((preds[mask] == labels[mask]).sum())
            total_tokens += int(mask.sum())

            total_loss += float(loss.detach().cpu())
            n_batches += 1

            logger.info(
                "val step %d/%s | loss=%.4f",
                step,
                max_batches if max_batches is not None else len(dataloader),
                float(loss.detach().cpu()),
            )

    if n_batches == 0:
        raise RuntimeError("eval_one_epoch no procesó ningún batch.")

    token_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return total_loss / n_batches, token_acc


def _effective_num_steps(
    dataloader: DataLoader,
    num_epochs: int,
    max_train_batches: int | None,
    grad_accum_steps: int = 1,
) -> int:
    """Calcula cantidad efectiva de optimizer steps."""
    batches_per_epoch = len(dataloader)

    if max_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, max_train_batches)

    optimizer_steps_per_epoch = max(1, batches_per_epoch // grad_accum_steps)
    return max(1, optimizer_steps_per_epoch * num_epochs)


def _save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    """Guarda historial de entrenamiento como JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Historial guardado en %s", history_path)


def run_finetuning(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    output_dir: Path,
    device: str,
    patience: int = 2,
    encoder_lr: float = 5e-6,
    decoder_lr: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    skip_checkpoint_save: bool = False,
) -> list[dict[str, float]]:
    """Orquesta el fine-tuning completo.

    Args:
        model: BLIP.
        processor: processor asociado.
        train_dataloader: DataLoader de entrenamiento.
        val_dataloader: DataLoader de validación.
        num_epochs: cantidad máxima de épocas.
        output_dir: carpeta donde guardar checkpoints.
        device: device de ejecución.
        patience: early stopping si val_loss no mejora.
        encoder_lr: LR para encoder visual.
        decoder_lr: LR para decoder.
        weight_decay: weight decay AdamW.
        warmup_ratio: proporción de warmup.
        grad_accum_steps: pasos de acumulación de gradientes (batch efectivo = batch_size * grad_accum_steps).
        use_amp: si True y device==cuda, usa Automatic Mixed Precision (~1.5–2x más rápido).
        max_train_batches: límite de batches train para debug.
        max_val_batches: límite de batches val para debug.
        skip_checkpoint_save: si True, no guarda checkpoints pesados.

    Returns:
        Historial con métricas por época (loss, perplexidad, token accuracy).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)

    # AMP solo disponible en CUDA
    amp_active = use_amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if amp_active else None
    logger.info("AMP: %s", "activado" if amp_active else "desactivado")

    freeze_stats = freeze_model(model)

    optimizer = build_optimizer(
        model=model,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        weight_decay=weight_decay,
    )

    num_training_steps = _effective_num_steps(
        dataloader=train_dataloader,
        num_epochs=num_epochs,
        max_train_batches=max_train_batches,
        grad_accum_steps=grad_accum_steps,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=warmup_ratio,
    )

    best_val_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    logger.info("Inicio fine-tuning en device=%s", device)

    for epoch in range(1, num_epochs + 1):
        logger.info("=" * 80)
        logger.info("Época %d/%d", epoch, num_epochs)
        logger.info("=" * 80)

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_batches=max_train_batches,
            grad_accum_steps=grad_accum_steps,
            scaler=scaler,
        )

        val_loss, val_token_acc = eval_one_epoch(
            model=model,
            dataloader=val_dataloader,
            device=device,
            max_batches=max_val_batches,
            use_amp=amp_active,
        )

        val_perplexity = math.exp(min(val_loss, 20.0))  # clamped para evitar overflow

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss": float(min(best_val_loss, val_loss)),
            "val_perplexity": float(val_perplexity),
            "val_token_acc": float(val_token_acc),
            "trainable_params": freeze_stats["trainable_params"],
            "total_params": freeze_stats["total_params"],
            "trainable_pct": freeze_stats["trainable_pct"],
            "grad_accum_steps": grad_accum_steps,
        }

        history.append(row)

        logger.info(
            "Época %d | train_loss=%.4f | val_loss=%.4f | perplexity=%.1f | token_acc=%.3f",
            epoch,
            train_loss,
            val_loss,
            val_perplexity,
            val_token_acc,
        )

        if not skip_checkpoint_save:
            epoch_dir = output_dir / f"epoch_{epoch}"
            save_model(model, processor, epoch_dir)
            logger.info("Checkpoint de época guardado en %s", epoch_dir)

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            if not skip_checkpoint_save:
                best_dir = output_dir / "best"
                save_model(model, processor, best_dir)
                logger.info("Nuevo best checkpoint guardado en %s", best_dir)
        else:
            epochs_without_improvement += 1
            logger.info(
                "Sin mejora. epochs_without_improvement=%d/%d",
                epochs_without_improvement,
                patience,
            )

        _save_history(history, output_dir)

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping activado tras %d épocas sin mejora.",
                patience,
            )
            break

    logger.info("Fine-tuning finalizado. best_val_loss=%.4f", best_val_loss)

    return history


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de CLI."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning de BLIP sobre MIMIC-CXR."
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/blip_base"),
        help="Checkpoint base de BLIP. Default: models/blip_base",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hf_cache"),
        help="Cache local del dataset HuggingFace. Default: data/hf_cache",
    )
    parser.add_argument(
        "--train-indices",
        type=Path,
        default=Path("data/splits/train_indices.json"),
        help="JSON con índices de entrenamiento.",
    )
    parser.add_argument(
        "--val-indices",
        type=Path,
        default=Path("data/splits/val_indices.json"),
        help="JSON con índices de validación.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/blip_finetuned"),
        help="Carpeta donde guardar checkpoints.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="impression",
        choices=["impression", "findings"],
        help="Columna de texto usada como target. Default: impression.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Longitud máxima de tokenización. Default: 128.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Cantidad de épocas. Default: 3.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size. En GPU T4 probar 4 u 8. En CPU usar 1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Workers del DataLoader. Default: 0.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device. Si se omite, se detecta automáticamente.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience. Default: 2.",
    )
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=5e-6,
        help="Learning rate para encoder visual. Default: 5e-6.",
    )
    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=1e-5,
        help="Learning rate para decoder. Default: 1e-5.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay de AdamW. Default: 0.01.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Proporción de warmup. Default: 0.05.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Pasos de acumulación de gradientes. Batch efectivo = batch_size * grad_accum_steps. Default: 1.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Desactiva Automatic Mixed Precision. Por defecto AMP está activado en CUDA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed. Default: 42.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Límite de batches train para debug.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Límite de batches val para debug.",
    )
    parser.add_argument(
        "--skip-checkpoint-save",
        action="store_true",
        help="No guarda checkpoints pesados. Útil para debug local en CPU.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point CLI."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    set_seed(args.seed)

    device = args.device or detect_device()

    logger.info("Device seleccionado: %s", device)
    logger.info("Cargando modelo desde %s", args.model_dir)

    model, processor = load_model_and_processor(
        model_dir=args.model_dir,
        device=device,
    )

    logger.info("Cargando dataset desde cache_dir=%s", args.cache_dir)

    ds = load_mimic_dataset(cache_dir=str(args.cache_dir))
    hf_split = ds["train"]

    logger.info("Creando train DataLoader desde %s", args.train_indices)
    train_loader = create_dataloader(
        hf_split=hf_split,
        indices_path=args.train_indices,
        processor=processor,
        text_col=args.text_col,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_length=args.max_length,
    )

    logger.info("Creando val DataLoader desde %s", args.val_indices)
    val_loader = create_dataloader(
        hf_split=hf_split,
        indices_path=args.val_indices,
        processor=processor,
        text_col=args.text_col,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_length=args.max_length,
    )

    run_finetuning(
        model=model,
        processor=processor,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        device=device,
        patience=args.patience,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=not args.no_amp,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        skip_checkpoint_save=args.skip_checkpoint_save,
    )


if __name__ == "__main__":
    main()
