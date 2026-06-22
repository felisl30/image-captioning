"""Mapas de cross-attention por palabra usando logits pre-softmax (Q·K^T / √d).

Por qué este enfoque en vez de los pesos post-softmax
------------------------------------------------------
El decoder de BLIP computa cross-attention sobre 576 patch tokens del encoder
visual. El softmax convierte los scores en pesos que suman 1; con 576 opciones
todos los pesos terminan ≈ 1/576 ≈ 0.0017, aplastando las diferencias entre
palabras hasta hacerlas visualmente indistinguibles.

Este módulo captura Q y K justo después de sus proyecciones lineales y calcula
Q·K^T / √d sin aplicar softmax. El resultado muestra la afinidad semántica
bruta entre cada token de texto y cada patch visual, donde las diferencias por
palabra sí son visibles.

Limitación a reportar: estos mapas son proxies de "intención" espacial del
decoder, no la distribución de atención que el modelo ejecuta internamente.
El modelo usa los pesos post-softmax·V para generar el output; los logits Q·K
son una señal de interpretabilidad, no el cómputo exacto.
"""
import torch
import numpy as np
from transformers import BlipForConditionalGeneration

from src.interpretability.cross_attention import merge_subword_attentions


def extract_cross_att_logits(
    model: BlipForConditionalGeneration,
    processor,
    inputs: dict,
    num_batch: int,
    layer_idx: int = 9,
    head_reduction: str = "max",
) -> list[dict]:
    """Extrae mapas de afinidad Q·K^T/√d del decoder de BLIP por token generado.

    Captura los vectores Q (query del token de texto) y K (keys de los patches
    visuales) después de sus proyecciones lineales, y calcula el producto punto
    escalado sin pasar por softmax.

    Args:
        model: BlipForConditionalGeneration cargado y en eval().
        processor: BlipProcessor correspondiente.
        inputs: dict con pixel_values (ya en el device correcto).
        num_batch: número de imágenes en el batch.
        layer_idx: capa del decoder donde registrar los hooks. La capa 9
            mostró mayor variabilidad entre palabras en los experimentos de
            diagnóstico. Default: 9.
        head_reduction: cómo agregar las n_heads.
            "max"  → pico más activado entre heads (más nítido).
            "mean" → promedio sobre heads (más suave).
            Default: "max".

    Returns:
        Lista de dicts {"caption": str, "maps": [(palabra, array(24,24)), ...]},
        uno por elemento del batch. Misma estructura que eval_and_extract_cross_att.
    """
    n_heads  = model.text_decoder.config.num_attention_heads
    head_dim = model.text_decoder.config.hidden_size // n_heads

    captured_q = []      # Q proyectado por paso de generación
    k_proj     = [None]  # K proyectado, capturado una sola vez (KV-cache)

    target = model.text_decoder.bert.encoder.layer[layer_idx].crossattention.self

    def hook_q(module, input, output):
        # output: (batch, T, hidden_size) — T=1 con KV-cache activo
        captured_q.append(output.detach().clone())

    def hook_k(module, input, output):
        # output: (batch, 577, hidden_size) — solo dispara en paso 0 con KV-cache
        if k_proj[0] is None and output.shape[1] == 577:
            k_proj[0] = output.detach().clone()

    q_hook = target.query.register_forward_hook(hook_q)
    k_hook = target.key.register_forward_hook(hook_k)

    try:
        out = model.generate(
            **inputs,
            output_attentions=True,
            return_dict_in_generate=True,
            max_new_tokens=40,
            num_beams=1,
        )
    finally:
        q_hook.remove()
        k_hook.remove()

    if k_proj[0] is None:
        return []

    # K: (batch, 577, hidden_size) → (batch, n_heads, 576, head_dim) — descarta CLS
    K = k_proj[0]
    K_heads   = K.view(K.shape[0], 577, n_heads, head_dim).permute(0, 2, 1, 3)
    K_patches = K_heads[:, :, 1:, :]   # (batch, n_heads, 576, head_dim)

    results = []
    for batch in range(num_batch):
        token_ids = out.sequences[batch][1:-1]   # sin BOS ni EOS
        tokens    = processor.tokenizer.convert_ids_to_tokens(token_ids)
        n_tokens  = len(tokens)

        subword_maps = []
        for i in range(min(n_tokens, len(captured_q))):
            q_vec  = captured_q[i][batch, -1, :]          # (hidden_size,)
            Q_h    = q_vec.view(n_heads, head_dim)         # (n_heads, head_dim)
            k      = K_patches[batch]                      # (n_heads, 576, head_dim)

            # logits[h, p] = Q_h[h] · K_patches[h, p] / √d
            logits = torch.einsum("hd,hpd->hp", Q_h, k) / (head_dim ** 0.5)
            # logits: (n_heads, 576)

            if head_reduction == "max":
                attn_vec = logits.max(dim=0).values        # (576,)
            else:
                attn_vec = logits.mean(dim=0)              # (576,)

            attn_vec = attn_vec - attn_vec.min()           # shift ≥ 0 para visualización
            subword_maps.append(attn_vec.cpu().numpy().reshape(24, 24))

        results.append(merge_subword_attentions(tokens, subword_maps))

    return results


def merge_subword_attentions(tokens: list, attention_maps: list) -> dict:
    """Agrupa subwords con ## y devuelve caption + lista ordenada de (palabra, mapa).

    Preserva orden de aparición y palabras repetidas.

    Returns:
        {
            "caption": "no acute no pneumonia",
            "maps": [("no", array(24,24)), ("acute", array(24,24)), ...]
        }
    """
    merged_tokens = []
    merged_maps   = []

    for token, attn_map in zip(tokens, attention_maps):
        if token.startswith("##"):
            merged_tokens[-1] += token[2:]
            merged_maps[-1].append(attn_map)
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