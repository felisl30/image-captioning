"""Grad-CAM for BLIP ViT encoder with optional externally generated ids.

Grad-CAM explains a generated token by computing the gradient of that token's
logit with respect to activations in the visual encoder.

If generated_ids_list is provided, this module does NOT call model.generate().
It explains those exact token sequences. This is required by the comparative
explainability notebook, where QK logits, post-softmax attention and Grad-CAM
must all explain the same caption.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.interpretability.cross_attention import merge_subword_attentions

logger = logging.getLogger(__name__)


def blip_vit_reshape_transform(
    tensor: torch.Tensor,
    height: int = 24,
    width: int = 24,
) -> torch.Tensor:
    """Reshape BLIP ViT token sequence to (B, C, H, W).

    BLIP uses 384x384 images with 16x16 patches:
        384 / 16 = 24 patches per side
        24 x 24 = 576 patch tokens
        + 1 CLS token = 577 tokens

    Args:
        tensor: Activation tensor with shape (B, 577, C) or sometimes
            (577, C) depending on the transformers/pytorch-grad-cam path.
        height: Patch-grid height.
        width: Patch-grid width.

    Returns:
        Tensor with shape (B, C, 24, 24).
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    if tensor.shape[1] == height * width + 1:
        tensor = tensor[:, 1:, :]  # remove CLS
    elif tensor.shape[1] != height * width:
        raise RuntimeError(
            f"Unexpected ViT token count: {tensor.shape[1]}. "
            f"Expected {height * width + 1} or {height * width}."
        )

    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class TokenTarget:
    """Grad-CAM target: selected token logit at final decoder position."""

    def __init__(self, token_id: int):
        self.token_id = int(token_id)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        # pytorch-grad-cam may iterate over batch and pass a 2D tensor:
        # (seq_len, vocab_size). Keep both paths.
        if model_output.dim() == 2:
            return model_output[-1, self.token_id]

        return model_output[:, -1, self.token_id]


class BLIPGradCAMWrapper(torch.nn.Module):
    """Wrapper exposing BLIP logits as a plain tensor for pytorch-grad-cam."""

    def __init__(
        self,
        model: BlipForConditionalGeneration,
        input_ids: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_ids = input_ids

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=self.input_ids,
            return_dict=True,
        )
        return outputs.logits


def _stop_token_ids(processor: BlipProcessor) -> set[int]:
    """Return token ids that should stop visible-token extraction."""
    tok = processor.tokenizer

    ids = {
        getattr(tok, "eos_token_id", None),
        getattr(tok, "sep_token_id", None),
        getattr(tok, "pad_token_id", None),
    }

    return {int(x) for x in ids if x is not None}


def _visible_token_ids(
    generated_ids: torch.Tensor,
    processor: BlipProcessor,
) -> torch.Tensor:
    """Remove BOS and stop at EOS/SEP/PAD."""
    if generated_ids.dim() != 2:
        raise ValueError("generated_ids must have shape (batch, seq_len).")

    if generated_ids.shape[0] != 1:
        raise ValueError("_visible_token_ids expects one image at a time.")

    seq = generated_ids[0]
    stop_ids = _stop_token_ids(processor)

    ids: list[int] = []
    for token_id in seq[1:].detach().cpu().tolist():
        token_id = int(token_id)
        if token_id in stop_ids:
            break
        ids.append(token_id)

    if not ids:
        return torch.empty(0, dtype=torch.long, device=generated_ids.device)

    return torch.tensor(ids, dtype=torch.long, device=generated_ids.device)


def _gradcam_single(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    pixel_values: torch.Tensor,
    generated_ids: torch.Tensor,
    device: str,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Compute Grad-CAM maps for one image and one generated token sequence.

    Args:
        model: BLIP model.
        processor: Matching processor.
        pixel_values: Tensor with shape (1, 3, 384, 384).
        generated_ids: Tensor with shape (1, seq_len).
        device: "cpu" or "cuda".
        max_tokens: Optional cap for smoke tests.

    Returns:
        Dict with caption and word-level maps.
    """
    model.eval()

    generated_ids = generated_ids.to(device)
    pixel_values = pixel_values.to(device)

    token_ids = _visible_token_ids(generated_ids, processor)

    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]

    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids.detach().cpu().tolist())

    target_layers = [model.vision_model.encoder.layers[-1].layer_norm1]

    subword_maps: list[np.ndarray] = []

    for t in range(len(token_ids)):
        # Teacher forcing prefix. For token t, feed BOS + generated tokens up to t.
        input_ids_t = generated_ids[:, : t + 2].to(device)

        wrapper = BLIPGradCAMWrapper(model=model, input_ids=input_ids_t)
        target = TokenTarget(token_id=int(token_ids[t].item()))

        grayscale_cam = None

        try:
            with GradCAM(
                model=wrapper,
                target_layers=target_layers,
                reshape_transform=blip_vit_reshape_transform,
            ) as cam:
                grayscale_cam = cam(input_tensor=pixel_values, targets=[target])
        except Exception as exc:
            token = tokens[t] if t < len(tokens) else f"#{t}"
            raise RuntimeError(f"Grad-CAM failed at token t={t} ({token!r}): {exc}") from exc

        if grayscale_cam is None:
            token = tokens[t] if t < len(tokens) else f"#{t}"
            raise RuntimeError(f"Grad-CAM returned None at token t={t} ({token!r}).")

        heatmap = torch.tensor(grayscale_cam[0]).float().unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(
            heatmap,
            size=(24, 24),
            mode="bilinear",
            align_corners=False,
        ).squeeze().detach().cpu().numpy()

        subword_maps.append(heatmap)

    result = merge_subword_attentions(tokens, subword_maps)
    result["generated_ids"] = [int(x) for x in generated_ids[0].detach().cpu().tolist()]
    return result


def compute_gradcam(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    images: list[Image.Image],
    device: str = "cpu",
    generated_ids_list: list[torch.Tensor] | None = None,
    max_new_tokens: int = 40,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Compute Grad-CAM maps per generated word for a batch of images.

    Args:
        model: BLIP model.
        processor: Matching processor.
        images: List of PIL images.
        device: "cpu" or "cuda".
        generated_ids_list: Optional list of generated id tensors, one per
            image. If provided, those exact sequences are explained. If None,
            this function generates greedily internally for backward compatibility.
        max_new_tokens: Used only when generated_ids_list is None.
        max_tokens: Optional cap on visible tokens. Useful for CPU smoke tests.

    Returns:
        List of dicts:
            {
                "caption": str,
                "maps": [(word, np.ndarray shape 24x24), ...],
                "generated_ids": list[int],
            }
    """
    model.eval()
    model.to(device)

    images = [img.convert("RGB") for img in images]

    if generated_ids_list is not None and len(generated_ids_list) != len(images):
        raise ValueError("generated_ids_list must have the same length as images.")

    captions_att: list[dict[str, Any]] = []

    for i, image in enumerate(images):
        single_inputs = processor(images=image, return_tensors="pt").to(device)
        pixel_values = single_inputs["pixel_values"]

        if generated_ids_list is None:
            with torch.no_grad():
                generated_ids = model.generate(
                    **single_inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                )
        else:
            generated_ids = generated_ids_list[i].to(device)

        result = _gradcam_single(
            model=model,
            processor=processor,
            pixel_values=pixel_values,
            generated_ids=generated_ids,
            device=device,
            max_tokens=max_tokens,
        )

        captions_att.append(result)

    return captions_att
