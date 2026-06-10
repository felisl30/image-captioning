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
# ── Cómo obtener los tensores ───────────────────────────────────────────────
#
# HuggingFace expone estos tensores pasando output_attentions=True a generate():
#
#     out = model.generate(
#         **inputs,
#         output_attentions=True,
#         return_dict_in_generate=True,
#     )
#
# La salida out.cross_attentions es una tupla anidada con estructura:
#
#     out.cross_attentions[paso][capa]
#     → tensor de shape (batch, n_heads, 1, 576)
#
# Donde:
#   - paso   : índice del token generado (0 = primer token, 1 = segundo, ...)
#   - capa   : índice de la capa del decoder (0 a 11 para BLIP base)
#   - batch  : tamaño del batch (1 para inferencia individual)
#   - n_heads: número de cabezas de atención (12 en ViT-Base)
#   - 1      : la query es un único token (el token que se está generando)
#   - 576    : los 576 patch tokens del encoder visual
#
# ── Cómo construir el mapa por palabra ─────────────────────────────────────
#
# Para obtener el heatmap correspondiente a la palabra en el paso t:
#
#   1. Elegir la capa a usar (estándar: la última, capa 11).
#      Justificación: la última capa del decoder tiene la representación más
#      elaborada antes de proyectar al vocabulario.
#
#   2. Extraer el tensor: attn = out.cross_attentions[t][11]
#      Shape resultante: (1, 12, 1, 576)
#
#   3. Promediar sobre las n_heads (dim=1):
#      attn_avg = attn.mean(dim=1)   → shape: (1, 1, 576)
#
#   4. Squeeze a 1D: attn_flat = attn_avg.squeeze()  → shape: (576,)
#
#   5. Reshape a grilla 2D: attn_2d = attn_flat.reshape(24, 24)
#
#   6. Upscale bilineal a 384×384 para superponer sobre la imagen original:
#      heatmap = F.interpolate(
#          attn_2d.unsqueeze(0).unsqueeze(0),
#          size=(384, 384),
#          mode="bilinear",
#          align_corners=False,
#      ).squeeze().numpy()
#
# ── Qué devolver ────────────────────────────────────────────────────────────
#
# La función principal `extract_cross_attention` debe devolver un diccionario:
#
#     {
#         "caption": "port-a-cath tip over mid svc",
#         "tokens": ["port", "-", "a", "-", "cath", ...],
#         "heatmaps": {
#             0: np.ndarray (384, 384),   ← heatmap para token 0
#             1: np.ndarray (384, 384),   ← heatmap para token 1
#             ...
#         }
#     }
#
# Los tokens se obtienen decodificando out.sequences token a token con:
#     processor.tokenizer.decode([token_id])
#
# ── Decisiones de implementación a tomar ───────────────────────────────────
#
# 1. ¿Última capa solamente o promedio de todas las capas?
#    Recomendación: última capa primero. Si los heatmaps salen muy difusos,
#    explorar Attention Rollout en rollout.py.
#
# 2. ¿Promedio sobre heads o seleccionar la head más "nítida"?
#    Recomendación: promedio. Seleccionar la head más informativa es válido
#    pero introduce sesgo en la elección.
#
# 3. ¿Normalizar el heatmap a [0,1] aquí o en visualization/?
#    Recomendación: devolver los valores crudos y normalizar en heatmap.py,
#    para no perder información entre módulos.
#
# ── Dependencias necesarias ─────────────────────────────────────────────────
#
#     import torch
#     import torch.nn.functional as F
#     import numpy as np
#     from transformers import BlipForConditionalGeneration, BlipProcessor
#
# ── Firma sugerida para la función principal ─────────────────────────────────
#
#     def extract_cross_attention(
#         model: BlipForConditionalGeneration,
#         processor: BlipProcessor,
#         image: PIL.Image,
#         layer: int = -1,
#         device: str = "cpu",
#     ) -> dict:
#         ...
#
# TODO: implementar esta función.
"""
