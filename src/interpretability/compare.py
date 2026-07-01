"""Orquestación de métodos de interpretabilidad para BLIP.

Este módulo corre post-softmax cross-attention, QK logits y Grad-CAM sobre
la misma imagen y la misma secuencia de tokens generada (`generated_ids`).

La función principal está pensada para que los notebooks sean delgados:
el notebook genera/elige una caption, obtiene sus token ids, y este módulo
calcula todos los mapas alineados.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from src.interpretability.cross_attention import eval_and_extract_cross_att
from src.interpretability.cross_att_logits import extract_cross_att_logits
from src.interpretability.gradcam import compute_gradcam
from src.interpretability.token_filter import filter_relevant_tokens


logger = logging.getLogger(__name__)


def _ensure_single_image_rgb(image: Image.Image) -> Image.Image:
    """Devuelve una imagen PIL RGB."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image debe ser PIL.Image.Image, recibido: {type(image)}")
    return image.convert("RGB")


def _ensure_generated_ids(generated_ids: torch.Tensor, device: str) -> torch.Tensor:
    """Valida y mueve generated_ids al device."""
    if not isinstance(generated_ids, torch.Tensor):
        raise TypeError(
            f"generated_ids debe ser torch.Tensor, recibido: {type(generated_ids)}"
        )

    if generated_ids.dim() != 2:
        raise ValueError(
            f"generated_ids debe tener shape (batch, seq_len), recibido: {tuple(generated_ids.shape)}"
        )

    if generated_ids.shape[0] != 1:
        raise ValueError(
            f"extract_all_methods trabaja con una imagen por vez. "
            f"Se recibió batch={generated_ids.shape[0]}"
        )

    if generated_ids.shape[1] < 2:
        raise ValueError(
            "generated_ids debe contener al menos BOS + un token visible."
        )

    return generated_ids.to(device)


def _filter_result_maps(result: dict[str, Any]) -> dict[str, Any]:
    """Aplica blacklist de tokens a result['maps'] preservando metadatos."""
    filtered = dict(result)
    filtered["maps"] = filter_relevant_tokens(result.get("maps", []))
    return filtered


def _assert_same_caption(results: dict[str, dict[str, Any]]) -> None:
    """Verifica que todos los métodos reconstruyan la misma caption."""
    captions = {
        method: str(payload.get("caption", "")).strip()
        for method, payload in results.items()
    }

    unique = set(captions.values())
    if len(unique) > 1:
        logger.warning("Las captions reconstruidas no son idénticas: %s", captions)


def extract_all_methods(
    model,
    processor,
    image: Image.Image,
    generated_ids: torch.Tensor,
    device: str = "cuda",
    layer_idx: int = 9,
    head_reduction: str = "max",
    include_gradcam: bool = True,
    filter_tokens: bool = True,
) -> dict[str, Any]:
    """Corre todos los métodos de interpretabilidad sobre una caption fija.

    Args:
        model: BlipForConditionalGeneration cargado.
        processor: BlipProcessor correspondiente.
        image: Imagen PIL. Se convierte a RGB internamente.
        generated_ids: Tensor (1, seq_len) con la caption ya generada.
            Si se pasa este argumento, los métodos NO deben regenerar captions.
        device: "cuda", "cpu" o "mps".
        layer_idx: Capa del decoder usada para cross-attention/QK.
        head_reduction: "max" o "mean" para combinar heads.
        include_gradcam: Si False, saltea Grad-CAM. Útil para pruebas rápidas.
        filter_tokens: Si True, descarta stopwords, puntuación y tokens no útiles.

    Returns:
        Diccionario con esta estructura:

        {
            "post_softmax": {
                "caption": str,
                "maps": [(word, np.ndarray 24x24), ...],
                "generated_ids": [...]
            },
            "qk_logits": {
                "caption": str,
                "maps": [(word, np.ndarray 24x24), ...],
                "generated_ids": [...]
            },
            "gradcam": {
                "caption": str,
                "maps": [(word, np.ndarray 24x24), ...],
                "generated_ids": [...]
            },
            "metadata": {
                "device": str,
                "layer_idx": int,
                "head_reduction": str,
                "include_gradcam": bool,
                "filter_tokens": bool
            }
        }
    """
    model.eval()
    model.to(device)

    image_rgb = _ensure_single_image_rgb(image)
    generated_ids = _ensure_generated_ids(generated_ids, device=device)

    inputs = processor(images=image_rgb, return_tensors="pt").to(device)

    logger.info("Extrayendo post-softmax cross-attention...")
    post_softmax = eval_and_extract_cross_att(
        model=model,
        processor=processor,
        inputs=inputs,
        num_batch=1,
        layer_idx=layer_idx,
        head_reduction=head_reduction,
        generated_ids=generated_ids,
    )[0]

    logger.info("Extrayendo QK logits...")
    qk_logits = extract_cross_att_logits(
        model=model,
        processor=processor,
        inputs=inputs,
        num_batch=1,
        layer_idx=layer_idx,
        head_reduction=head_reduction,
        generated_ids=generated_ids,
    )[0]

    results: dict[str, Any] = {
        "post_softmax": post_softmax,
        "qk_logits": qk_logits,
    }

    if include_gradcam:
        logger.info("Extrayendo Grad-CAM...")
        gradcam = compute_gradcam(
            model=model,
            processor=processor,
            images=[image_rgb],
            device=device,
            generated_ids_list=[generated_ids],
        )[0]
        results["gradcam"] = gradcam

    if filter_tokens:
        for method in list(results.keys()):
            results[method] = _filter_result_maps(results[method])

    _assert_same_caption(
        {
            method: payload
            for method, payload in results.items()
            if method != "metadata"
        }
    )

    results["metadata"] = {
        "device": device,
        "layer_idx": layer_idx,
        "head_reduction": head_reduction,
        "include_gradcam": include_gradcam,
        "filter_tokens": filter_tokens,
    }

    return results
