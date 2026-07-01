"""Cross-attention maps using pre-softmax Q·K^T / sqrt(d) logits.

This module extracts word-level spatial maps from BLIP's text decoder by
capturing the projected query vectors Q and visual key vectors K in a decoder
cross-attention layer.

Important:
    If `generated_ids` is provided, the function does NOT call generate().
    Instead, it runs a teacher-forced forward pass over those token ids and
    explains that exact sequence. This is required for the comparative notebook,
    where all methods must explain the same sampled caption.
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
    """Extract generated target token ids aligned to Q positions.

    During autoregressive generation, the Q vector at decoder position i is the
    context used to predict generated token i. For a sequence
    [BOS, t0, t1, ..., EOS], the target tokens are sequence[1:].
    """
    seq = sequence.detach().cpu().tolist()

    # Remove the first token position, normally BOS.
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


def extract_cross_att_logits(
    model: BlipForConditionalGeneration,
    processor,
    inputs: dict,
    num_batch: int,
    layer_idx: int = 9,
    head_reduction: str = "max",
    generated_ids: torch.Tensor | None = None,
    max_new_tokens: int = 40,
) -> list[dict]:
    """Extract Q·K^T/sqrt(d) maps from BLIP decoder cross-attention.

    Args:
        model: BLIP model loaded and usually in eval mode.
        processor: Matching BLIP processor.
        inputs: Processor outputs containing `pixel_values`.
        num_batch: Number of images in the batch to process.
        layer_idx: Decoder layer index used for hooks. Layer 9 usually gives
            more spatially differentiated QK maps in this project.
        head_reduction: "max" for sharp maps or "mean" for smoother maps.
        generated_ids: Optional generated token ids with shape (batch, seq_len).
            If provided, this exact sequence is explained and no new caption is
            generated internally.
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

    captured_q: list[torch.Tensor] = []
    k_proj: list[torch.Tensor | None] = [None]

    target = model.text_decoder.bert.encoder.layer[layer_idx].crossattention.self

    def hook_q(module, module_input, output):
        # generate mode: (batch, 1, hidden)
        # teacher-forced forward: (batch, T, hidden)
        captured_q.append(_as_3d(output.detach().clone()))

    def hook_k(module, module_input, output):
        # K over encoder output. Usually (batch, 577, hidden).
        out = _as_3d(output.detach().clone())
        if k_proj[0] is None and out.shape[1] in {576, 577}:
            k_proj[0] = out

    q_hook = target.query.register_forward_hook(hook_q)
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

                # Teacher-forced pass. The decoder inputs are [BOS, t0, ..., tN-1],
                # and each Q position is aligned to target tokens [t0, ..., tN].
                decoder_input_ids = sequences[:, :-1]

                model(
                    pixel_values=inputs["pixel_values"],
                    input_ids=decoder_input_ids,
                    output_attentions=True,
                    return_dict=True,
                )
    finally:
        q_hook.remove()
        k_hook.remove()

    if k_proj[0] is None:
        logger.warning("No K projection captured in cross-attention layer %s.", layer_idx)
        return []

    if not captured_q:
        logger.warning("No Q projections captured in cross-attention layer %s.", layer_idx)
        return []

    # Q handling:
    # - generated internally: captured_q is a list of one-step tensors.
    # - generated_ids provided: captured_q usually has one tensor with all T positions.
    if generated_ids is None:
        q_steps = captured_q
    else:
        q_all = captured_q[0]  # (batch, T, hidden)
        q_steps = [q_all[:, i : i + 1, :] for i in range(q_all.shape[1])]

    # K: (batch, 577, hidden) or rarely (batch, 576, hidden)
    K = k_proj[0]
    K_heads = K.view(K.shape[0], K.shape[1], n_heads, head_dim).permute(0, 2, 1, 3)

    if K_heads.shape[2] == 577:
        K_patches = K_heads[:, :, 1:, :]  # remove CLS -> (batch, heads, 576, head_dim)
    elif K_heads.shape[2] == 576:
        K_patches = K_heads
    else:
        raise RuntimeError(f"Unexpected K sequence length: {K_heads.shape[2]}")

    if K_patches.shape[2] != 576:
        raise RuntimeError(f"Expected 576 visual patches, got {K_patches.shape[2]}.")

    results: list[dict] = []
    actual_batch = min(num_batch, sequences.shape[0], K_patches.shape[0])

    for batch in range(actual_batch):
        token_id_list = _target_token_ids_from_sequence(
            sequence=sequences[batch],
            q_steps=len(q_steps),
            processor=processor,
        )

        tokens = processor.tokenizer.convert_ids_to_tokens(token_id_list)

        subword_maps: list[np.ndarray] = []
        n_tokens = min(len(tokens), len(q_steps))

        for i in range(n_tokens):
            q_vec = q_steps[i][batch, -1, :]       # (hidden_size,)
            Q_h = q_vec.view(n_heads, head_dim)   # (heads, head_dim)
            k = K_patches[batch]                  # (heads, 576, head_dim)

            logits = torch.einsum("hd,hpd->hp", Q_h, k) / (head_dim ** 0.5)

            if head_reduction == "max":
                attn_vec = logits.max(dim=0).values
            else:
                attn_vec = logits.mean(dim=0)

            # Shift to non-negative for visualization. Final normalization is
            # still handled by heatmap/plotting utilities.
            attn_vec = attn_vec - attn_vec.min()

            subword_maps.append(attn_vec.detach().cpu().numpy().reshape(24, 24))

        result = merge_subword_attentions(tokens, subword_maps)
        result["generated_ids"] = [int(x) for x in sequences[batch].detach().cpu().tolist()]
        results.append(result)

    return results
