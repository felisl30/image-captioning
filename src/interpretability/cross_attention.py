"""Cross-attention post-softmax maps from BLIP decoder.

This module extracts word-level maps from the decoder cross-attention
probabilities. It supports two modes:

1. Legacy/internal generation:
   If generated_ids is None, the function calls model.generate(...).

2. Aligned external caption:
   If generated_ids is provided, the function runs a teacher-forced forward pass
   and extracts maps for that exact sequence. This is required for the
   comparative explainability notebook, where all methods must explain the same
   sampled caption.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from transformers import BlipForConditionalGeneration

logger = logging.getLogger(__name__)


def _move_to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a dict to device."""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _as_3d(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensors to (batch, seq_len, hidden)."""
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    return tensor


def _stop_token_ids(processor) -> set[int]:
    """Return token ids that should terminate visible token extraction."""
    tok = processor.tokenizer

    ids = {
        getattr(tok, "eos_token_id", None),
        getattr(tok, "sep_token_id", None),
        getattr(tok, "pad_token_id", None),
    }

    return {int(x) for x in ids if x is not None}


def _target_token_ids_from_sequence(
    sequence: torch.Tensor,
    q_steps: int,
    processor,
) -> list[int]:
    """Extract visible target token ids aligned to decoder positions."""
    seq = sequence.detach().cpu().tolist()

    target_ids = seq[1 : 1 + q_steps]
    stop_ids = _stop_token_ids(processor)

    cleaned: list[int] = []
    for token_id in target_ids:
        token_id = int(token_id)
        if token_id in stop_ids:
            break
        cleaned.append(token_id)

    return cleaned


def merge_subword_attentions(tokens: list[str], attention_maps: list[np.ndarray]) -> dict:
    """Merge WordPiece subwords and return caption + ordered maps.

    Args:
        tokens: Token strings from the tokenizer.
        attention_maps: One 24x24 heatmap per token.

    Returns:
        Dict with:
            caption: merged caption string.
            maps: ordered list of (word, heatmap) pairs.
    """
    merged_tokens: list[str] = []
    merged_maps: list[list[np.ndarray]] = []

    for token, attn_map in zip(tokens, attention_maps):
        if token.startswith("##"):
            if merged_tokens:
                merged_tokens[-1] += token[2:]
                merged_maps[-1].append(attn_map)
            else:
                merged_tokens.append(token[2:])
                merged_maps.append([attn_map])
        else:
            merged_tokens.append(token)
            merged_maps.append([attn_map])

    maps_list = [
        (token, np.mean(maps, axis=0))
        for token, maps in zip(merged_tokens, merged_maps)
    ]

    return {
        "caption": " ".join(merged_tokens),
        "maps": maps_list,
    }


def eval_and_extract_cross_att(
    model: BlipForConditionalGeneration,
    processor,
    inputs: dict,
    num_batch: int,
    layer_idx: int = 8,
    head_reduction: str = "max",
    subtract_uniform: bool = False,
    norm_correct: bool = True,
    logit_space: bool = False,
    generated_ids: torch.Tensor | None = None,
    max_new_tokens: int = 40,
) -> list[dict]:
    """Extract post-softmax cross-attention maps from BLIP decoder.

    Args:
        model: BLIP model.
        processor: Matching processor.
        inputs: Processor outputs containing pixel_values.
        num_batch: Batch size to process.
        layer_idx: Decoder layer to hook.
        head_reduction: "max" or "mean" over attention heads.
        subtract_uniform: If True, subtracts 1/576 baseline and clips at 0.
        norm_correct: If True, divides attention weights by K-vector norm.
        logit_space: If True, applies log to post-softmax weights before
            aggregation. This recovers score differences up to a constant.
        generated_ids: Optional external token ids to explain. If provided, no
            generation happens internally.
        max_new_tokens: Used only when generated_ids is None.

    Returns:
        List of dicts:
            {
                "caption": str,
                "maps": [(word, np.ndarray shape 24x24), ...],
                "generated_ids": list[int],
            }
    """
    if head_reduction not in {"max", "mean"}:
        raise ValueError("head_reduction must be 'max' or 'mean'.")

    model.eval()
    device = next(model.parameters()).device
    inputs = _move_to_device(inputs, device)

    n_heads = model.text_decoder.config.num_attention_heads
    head_dim = model.text_decoder.config.hidden_size // n_heads

    captured_attn: list[torch.Tensor] = []
    captured_k: list[torch.Tensor | None] = [None]

    target = model.text_decoder.bert.encoder.layer[layer_idx].crossattention.self

    def hook_attn(module, module_input, output):
        # output is usually (context_layer, attention_probs) when
        # output_attentions=True.
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn = output[1].detach().clone()
            if attn.shape[-1] in {576, 577}:
                captured_attn.append(attn)

    def hook_k(module, module_input, output):
        out = _as_3d(output.detach().clone())
        if captured_k[0] is None and out.shape[1] in {576, 577}:
            captured_k[0] = out

    attn_hook = target.register_forward_hook(hook_attn)

    k_hook = None
    if norm_correct:
        k_hook = target.key.register_forward_hook(hook_k)

    try:
        with torch.no_grad():
            if generated_ids is None:
                out = model.generate(
                    **inputs,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                )
                sequences = out.sequences.to(device)
            else:
                sequences = generated_ids.to(device)

                if sequences.dim() != 2:
                    raise ValueError("generated_ids must have shape (batch, seq_len).")

                if sequences.shape[1] < 2:
                    raise ValueError("generated_ids must contain at least BOS + one token.")

                decoder_input_ids = sequences[:, :-1]

                model(
                    pixel_values=inputs["pixel_values"],
                    input_ids=decoder_input_ids,
                    output_attentions=True,
                    return_dict=True,
                )
    finally:
        attn_hook.remove()
        if k_hook is not None:
            k_hook.remove()

    if not captured_attn:
        logger.warning("No cross-attention probabilities captured in layer %s.", layer_idx)
        return []

    # Attention handling:
    # - generation mode usually captures one tensor per generated token:
    #   [(B, heads, 1, 577), ...]
    # - teacher-forcing mode usually captures one tensor:
    #   [(B, heads, T, 577)]
    if generated_ids is None:
        attn_steps = captured_attn
    else:
        attn_all = captured_attn[0]
        attn_steps = [attn_all[:, :, i : i + 1, :] for i in range(attn_all.shape[2])]

    # Optional K-norm correction.
    k_norms = None
    if norm_correct and captured_k[0] is not None:
        K = captured_k[0]  # (B, 577, hidden) or (B, 576, hidden)

        if K.shape[1] == 577:
            K = K[:, 1:, :]  # remove CLS
        elif K.shape[1] != 576:
            raise RuntimeError(f"Unexpected K sequence length: {K.shape[1]}")

        K_heads = K.view(K.shape[0], 576, n_heads, head_dim)
        k_norms = K_heads.norm(dim=-1)  # (B, 576, heads)

    results: list[dict] = []
    actual_batch = min(num_batch, sequences.shape[0], attn_steps[0].shape[0])

    for batch in range(actual_batch):
        token_id_list = _target_token_ids_from_sequence(
            sequence=sequences[batch],
            q_steps=len(attn_steps),
            processor=processor,
        )

        tokens = processor.tokenizer.convert_ids_to_tokens(token_id_list)

        subword_maps: list[np.ndarray] = []
        n_tokens = min(len(tokens), len(attn_steps))

        for i in range(n_tokens):
            attn = attn_steps[i]  # (B, heads, 1, 576/577)

            if attn.shape[-1] == 577:
                attn_patches = attn[..., 1:]
            elif attn.shape[-1] == 576:
                attn_patches = attn
            else:
                raise RuntimeError(f"Unexpected attention key length: {attn.shape[-1]}")

            heads = attn_patches[batch, :, -1, :]  # (heads, 576)

            if logit_space:
                heads = torch.log(heads + 1e-10)

            if norm_correct and k_norms is not None:
                kn = k_norms[batch].T.to(heads.device)  # (heads, 576)

                if logit_space:
                    heads = heads - torch.log(kn + 1e-8)
                else:
                    heads = heads / (kn + 1e-8)

            if logit_space:
                heads = heads - heads.min(dim=1, keepdim=True).values

            if head_reduction == "max":
                attn_vec = heads.max(dim=0).values
            else:
                attn_vec = heads.mean(dim=0)

            if subtract_uniform and not logit_space:
                attn_vec = torch.relu(attn_vec - 1.0 / attn_vec.shape[0])

            subword_maps.append(attn_vec.detach().cpu().numpy().reshape(24, 24))

        result = merge_subword_attentions(tokens, subword_maps)
        result["generated_ids"] = [int(x) for x in sequences[batch].detach().cpu().tolist()]
        results.append(result)

    return results


def eval_and_extract_qk_logits(
    model: BlipForConditionalGeneration,
    processor,
    inputs: dict,
    num_batch: int,
    layer_idx: int = 8,
    head_reduction: str = "max",
    generated_ids: torch.Tensor | None = None,
    max_new_tokens: int = 40,
) -> list[dict]:
    """Compatibility wrapper for QK logits extraction.

    The production implementation lives in cross_att_logits.py. This wrapper
    preserves older imports that used eval_and_extract_qk_logits from this file.
    """
    from src.interpretability.cross_att_logits import extract_cross_att_logits

    return extract_cross_att_logits(
        model=model,
        processor=processor,
        inputs=inputs,
        num_batch=num_batch,
        layer_idx=layer_idx,
        head_reduction=head_reduction,
        generated_ids=generated_ids,
        max_new_tokens=max_new_tokens,
    )
