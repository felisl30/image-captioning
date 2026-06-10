"""Carga y verifica BlipForConditionalGeneration desde HuggingFace o disco."""

import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)

HF_MODEL_ID = "Salesforce/blip-image-captioning-base"


def load_model_and_processor(
    model_dir: Path | None = None,
    device: str | None = None,
) -> tuple[BlipForConditionalGeneration, BlipProcessor]:
    """Carga el modelo BLIP y su processor desde disco o desde HuggingFace.

    Si `model_dir` existe en disco (es decir, contiene un checkpoint guardado
    con `save_pretrained`), carga desde ahí. Si no se proporciona o la ruta no
    existe, descarga desde HuggingFace y, si se pasa `model_dir`, guarda el
    modelo allí para usos futuros.

    Args:
        model_dir: Path a la carpeta local del modelo. Puede ser
            `models/blip_base/` para el modelo preentrenado o
            `models/blip_finetuned/best/` para el fine-tuneado.
            Si es None, siempre descarga desde HuggingFace sin cachear a disco.
        device: "cuda", "cpu" o "mps". Si es None, se detecta automáticamente.

    Returns:
        Tupla (model, processor) listos para usar.
    """
    device = device or _detect_device()

    if model_dir is not None and Path(model_dir).exists():
        source = str(model_dir)
        logger.info("Cargando BLIP desde disco: %s", source)
    else:
        source = HF_MODEL_ID
        logger.info("Descargando BLIP desde HuggingFace: %s", source)

    processor = BlipProcessor.from_pretrained(source)
    model = BlipForConditionalGeneration.from_pretrained(source)
    model.to(device)
    model.eval()

    if model_dir is not None and not Path(model_dir).exists():
        logger.info("Guardando modelo en disco: %s", model_dir)
        _save_to_disk(model, processor, Path(model_dir))

    return model, processor


def save_model(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    output_dir: Path,
) -> None:
    """Guarda modelo y processor en `output_dir` con el formato HuggingFace.

    Args:
        model: Modelo a guardar (base o fine-tuneado).
        processor: Processor asociado.
        output_dir: Carpeta destino. Se crea si no existe.
    """
    _save_to_disk(model, processor, Path(output_dir))
    logger.info("Modelo guardado en: %s", output_dir)


def sanity_check(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    device: str | None = None,
) -> str:
    """Genera un caption sobre una imagen sintética para verificar que el modelo funciona.

    Usa una imagen RGB negra de 224×224. El caption resultante no tiene
    significado semántico, pero si el pipeline completo (processor → generate
    → decode) funciona sin errores, el modelo está listo.

    Args:
        model: Modelo cargado.
        processor: Processor asociado al modelo.
        device: Dispositivo donde está el modelo. Si es None, se detecta.

    Returns:
        El caption generado como string.
    """
    device = device or _detect_device()
    dummy_image = Image.new("RGB", (224, 224), color=(0, 0, 0))

    inputs = processor(images=dummy_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    logger.info("Sanity check caption: '%s'", caption)
    return caption


# ── helpers privados ──────────────────────────────────────────────────────────

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _save_to_disk(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    path: Path,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Carga BLIP y corre un sanity check.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/blip_base"),
        help="Carpeta local del modelo (default: models/blip_base)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Genera un caption sobre imagen dummy y sale",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model, processor = load_model_and_processor(
        model_dir=args.model_dir,
        device=args.device,
    )

    if args.sanity_check:
        caption = sanity_check(model, processor, device=args.device)
        print(f"Caption generado: '{caption}'")
