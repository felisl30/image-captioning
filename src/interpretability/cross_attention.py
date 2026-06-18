"""Extracción de mapas de cross-attention del decoder de BLIP por palabra generada.

# ── Introducción ────────────────────────────────────────────────────────────
#
# BLIP genera cada token del caption atendiendo a los patch tokens del encoder
# visual mediante cross-attention. Para cada token generado, el decoder tiene
# un tensor de pesos de atención de shape:
#
#     (batch, n_heads, 1, n_patches)
#
# donde n_patches = 576  (24×24, porque BLIP usa imágenes de 384×384 y patches
# de 16×16 → 384/16 = 24 patches por lado).
#
# ── Cómo obtener los tensores (enfoque actual: forward hook) ─────────────────
#
# En transformers ≤ 4.x, `generate(output_attentions=True)` devolvía
# `out.cross_attentions` con la estructura:
#
#     out.cross_attentions[paso][capa]  → (batch, n_heads, 1, 576)
#
# A partir de transformers 5.x el output unificado es `GenerateDecoderOnlyOutput`
# y `cross_attentions` ya no está expuesto en el dict de salida. La solución
# es registrar un `register_forward_hook` directamente sobre el módulo de
# cross-attention de la capa indicada del decoder:
#
#     model.text_decoder.bert.encoder.layer[layer_idx].crossattention.self
#
# El hook captura el output del módulo en cada paso de generación. El shape
# capturado en transformers 5.x es:
#
#     (batch, n_heads, 1, 577)   ← 1 CLS + 576 patches
#
# Se descarta el CLS (posición 0) y se trabaja con los 576 patches restantes.
# El hook dispara exactamente N veces (una por token generado), por lo que
# `captured[i]` ↔ `token_ids[i]` sin necesidad de índice adicional.
#
# ── Problema de anchor patches y corrección de norma ───────────────────────
#
# Diagnóstico (2026-06-17): el ratio max/min de las normas L2 de los 576 patch
# features del ViT es ~30×. La cross-attention calcula softmax(Q·K^T/√d); si
# un patch tiene norma K 30× mayor que otro, domina la atención sin importar
# la dirección de Q. Esto produce mapas visualmente idénticos para todas las
# palabras del caption.
#
# Fix (norm_correct=True): capturar los vectores K del módulo de proyección
# de claves, calcular sus normas por head, y dividir cada peso de atención
# por la norma del K correspondiente antes de agregar las heads. Esto anula
# el sesgo de norma y deja solo la componente direccional (semántica).
#
#     corrected[h, patch] = attn[h, patch] / ||K_h[patch]||
#
# ── Cómo construir el mapa por palabra ─────────────────────────────────────
#
# Para el token i (con norm_correct=True):
#   1. attn = captured[i]                      → (batch, n_heads, 1, 577)
#   2. attn_patches = attn[..., 1:]            → (batch, n_heads, 1, 576) descarta CLS
#   3. heads = attn_patches[batch, :, -1, :]   → (n_heads, 576)
#   4. heads /= k_norms.T                      → corrige sesgo de norma
#   5. attn_vec = heads.max(dim=0).values      → (576,) pico más activado
#   6. attn_vec.reshape(24, 24)                → grilla 2D
"""
import torch
from transformers import BlipForConditionalGeneration
import numpy as np


def eval_and_extract_cross_att(
    model: BlipForConditionalGeneration,
    processor,
    inputs,
    num_batch,
    layer_idx: int = 8,
    head_reduction: str = "max",
    subtract_uniform: bool = False,
    norm_correct: bool = True,
    logit_space: bool = False,
):
    """Extrae mapas de cross-attention del decoder de BLIP por token generado.

    Args:
        model: BlipForConditionalGeneration cargado.
        processor: BlipProcessor correspondiente.
        inputs: dict con pixel_values (output de BlipProcessor, ya en device).
        num_batch: número de imágenes en el batch.
        layer_idx: índice de capa del decoder sobre la que registrar el hook.
            Las capas 6–9 suelen dar atención espacialmente más coherente que
            la capa 11 (última), que ya está especializada en la proyección al
            vocabulario. Default: 8.
        head_reduction: cómo agregar las n_heads tras la corrección de norma.
            "max"  → preserva el pico más activado entre heads (más nítido).
            "mean" → promedio sobre todas las heads (más suave, puede diluir).
            Default: "max".
        subtract_uniform: si True, resta la baseline uniforme (1/N_patches) y
            conserva solo las desviaciones positivas con relu. Solo es útil
            cuando norm_correct=False y logit_space=False. Default: False.
        norm_correct: si True, divide cada peso de atención por la norma L2
            del vector K correspondiente (por head) antes de agregar las heads.
            En logit_space, la división se convierte en sustracción de log(norma).
            Default: True.
        logit_space: si True, aplica log() a los pesos post-softmax antes de
            procesar. log(softmax(s)) = s - log(Z), que recupera los scores
            pre-softmax salvo una constante por paso. Preserva diferencias
            relativas que el softmax aplana sobre 576 opciones. Default: False.

    Returns:
        Lista de dicts {"caption": str, "maps": [(palabra, array(24,24)), ...]},
        uno por elemento del batch.
    """
    n_heads  = model.text_decoder.config.num_attention_heads
    head_dim = model.text_decoder.config.hidden_size // n_heads

    captured = []
    k_norms  = None   # (576, n_heads) — se captura una sola vez (KV-cache fija K)

    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn = output[1]
            if attn.shape[-1] == 577:
                captured.append(attn.detach().clone())

    def hook_k_fn(module, input, output):
        # output: (batch, 577, hidden_size) — proyección lineal de K sobre encoder output
        # Con KV-cache, BLIP solo calcula K en el paso 0; en pasos siguientes usa el cache.
        # Solo capturamos la primera llamada (k_norms is None).
        nonlocal k_norms
        if k_norms is None and output.shape[1] == 577:
            k = output[0, 1:]                          # (576, hidden_size) — descarta CLS
            k_heads = k.view(576, n_heads, head_dim)   # (576, n_heads, head_dim)
            k_norms = k_heads.norm(dim=-1).detach()    # (576, n_heads)

    target = model.text_decoder.bert.encoder.layer[layer_idx].crossattention.self
    hook   = target.register_forward_hook(hook_fn)

    k_hook = None
    if norm_correct:
        k_target = model.text_decoder.bert.encoder.layer[layer_idx].crossattention.self.key
        k_hook   = k_target.register_forward_hook(hook_k_fn)

    try:
        out = model.generate(
            **inputs,
            output_attentions=True,
            return_dict_in_generate=True,
            max_new_tokens=40,
            num_beams=1,
        )
    finally:
        hook.remove()
        if k_hook is not None:
            k_hook.remove()

    captions_att = []
    for batch in range(num_batch):
        token_ids = out.sequences[batch][1:-1]  # sin BOS ni EOS
        tokens    = processor.tokenizer.convert_ids_to_tokens(token_ids)
        n_tokens  = len(tokens)

        subword_maps = []
        for i in range(min(n_tokens, len(captured))):
            attn         = captured[i]              # (batch, n_heads, T, 577)
            attn_patches = attn[..., 1:]            # descarta CLS → (batch, n_heads, T, 576)
            heads        = attn_patches[batch, :, -1, :]  # (n_heads, 576)

            if logit_space:
                # log(softmax(s_i)) = s_i - log(Z)  →  recupera logits pre-softmax
                # salvo la constante log(Z), que se cancela al normalizar a [0,1].
                # Amplifica diferencias relativas que el softmax comprime sobre 576 opciones.
                heads = torch.log(heads + 1e-10)    # (n_heads, 576), valores negativos

            if norm_correct and k_norms is not None:
                kn = k_norms.T.to(heads.device)     # (n_heads, 576)
                if logit_space:
                    # en espacio log: división → sustracción de log(norma)
                    heads = heads - torch.log(kn + 1e-8)
                else:
                    heads = heads / (kn + 1e-8)

            if logit_space:
                # desplazar por head para que cada una arranque en 0 antes de agregar
                heads = heads - heads.min(dim=1, keepdim=True).values

            if head_reduction == "max":
                attn_vec = heads.max(dim=0).values  # (576,)
            else:
                attn_vec = heads.mean(dim=0)        # (576,)

            if subtract_uniform and not logit_space:
                n_patches = attn_vec.shape[0]       # 576
                attn_vec  = torch.relu(attn_vec - 1.0 / n_patches)

            subword_maps.append(attn_vec.numpy().reshape(24, 24))

        result = merge_subword_attentions(tokens, subword_maps)
        captions_att.append(result)

    return captions_att


def eval_and_extract_qk_logits(
    model: BlipForConditionalGeneration,
    processor,
    inputs,
    num_batch,
    layer_idx: int = 8,
    head_reduction: str = "max",
):
    """Extrae logits pre-softmax Q·K^T/√d del decoder por token generado.

    Evita por completo el aplastamiento del softmax sobre 576 opciones.
    El mapa resultante es la similitud directa (en espacio proyectado) entre
    el query del token y cada patch del encoder visual.

    Returns:
        Lista de dicts {"caption": str, "maps": [(palabra, array(24,24)), ...]},
        uno por elemento del batch.
    """
    n_heads  = model.text_decoder.config.num_attention_heads
    head_dim = model.text_decoder.config.hidden_size // n_heads

    captured_q = []      # Q proyectado por paso de generación
    k_proj     = [None]  # K proyectado, capturado una sola vez (KV-cache)

    target = model.text_decoder.bert.encoder.layer[layer_idx].crossattention.self

    def hook_q_fn(module, input, output):
        # output: (batch, T, hidden_size) — T=1 con KV-cache
        captured_q.append(output.detach().clone())

    def hook_k_fn(module, input, output):
        if k_proj[0] is None and output.shape[1] == 577:
            k_proj[0] = output.detach().clone()  # (batch, 577, hidden_size)

    q_hook = target.query.register_forward_hook(hook_q_fn)
    k_hook = target.key.register_forward_hook(hook_k_fn)

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

    # K: (batch, 577, hidden_size) → (batch, n_heads, 576, head_dim) sin CLS
    K = k_proj[0]
    K_heads = K.view(K.shape[0], 577, n_heads, head_dim).permute(0, 2, 1, 3)
    K_patches = K_heads[:, :, 1:, :]  # (batch, n_heads, 576, head_dim)

    captions_att = []
    for batch in range(num_batch):
        token_ids = out.sequences[batch][1:-1]
        tokens    = processor.tokenizer.convert_ids_to_tokens(token_ids)
        n_tokens  = len(tokens)

        subword_maps = []
        for i in range(min(n_tokens, len(captured_q))):
            q     = captured_q[i][batch, -1, :]       # (hidden_size,)
            Q_h   = q.view(n_heads, head_dim)          # (n_heads, head_dim)
            k     = K_patches[batch]                   # (n_heads, 576, head_dim)

            # logits[h, p] = Q_h[h] · K_patches[h, p] / sqrt(d)
            logits   = torch.einsum("hd,hpd->hp", Q_h, k) / (head_dim ** 0.5)
            # logits: (n_heads, 576)

            if head_reduction == "max":
                attn_vec = logits.max(dim=0).values
            else:
                attn_vec = logits.mean(dim=0)

            attn_vec = attn_vec - attn_vec.min()       # shift ≥ 0 para visualización
            subword_maps.append(attn_vec.cpu().numpy().reshape(24, 24))

        captions_att.append(merge_subword_attentions(tokens, subword_maps))

    return captions_att


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


# ── Implementación original (transformers ≤ 4.x) ────────────────────────────
#
# def eval_and_extract_cross_att_legacy(model, processor, inputs, num_batch):
#     out = model.generate(
#         **inputs,
#         output_attentions=True,
#         return_dict_in_generate=True,
#     )
#     # out.cross_attentions[paso][capa] → (batch, n_heads, 1, 576)
#     captions_att = []
#     for batch in range(num_batch):
#         token_ids = out.sequences[batch][1:-1]
#         tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
#         subword_maps = []
#         for i in range(len(tokens)):
#             attention_maps = out.cross_attentions[i][-1]
#             attn_heads = attention_maps[batch]
#             attn = attn_heads.mean(0)[0].detach().numpy().reshape(24, 24)
#             subword_maps.append(attn)
#         captions_att.append(merge_subword_attentions(tokens, subword_maps))
#     return captions_att
