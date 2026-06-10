"""Grad-CAM para el encoder ViT de BLIP usando pytorch-grad-cam.

# ── Introducción ────────────────────────────────────────────────────────────
#
# Grad-CAM computa el gradiente del logit de un token objetivo respecto a las
# activaciones de una capa target del encoder visual. Las zonas de la imagen
# donde ese gradiente es alto son las que más influyen en predecir ese token.
#
# ── Qué capa usar como target ───────────────────────────────────────────────
#
# La capa target estándar para ViT es la LayerNorm de la última capa del
# encoder. En BLIP, el encoder visual es model.vision_model y sus bloques
# están en model.vision_model.encoder.layers.
#
# Target layer:
#     model.vision_model.encoder.layers[-1].layer_norm1
#
# Esta es la representación más semánticamente rica antes de que los tokens
# pasen al decoder. Las capas anteriores capturan features más genéricos.
#
# ── El problema del reshape en ViT ──────────────────────────────────────────
#
# ViT no produce feature maps 2D como una CNN. Su salida es una secuencia de
# tokens de shape (batch, n_tokens, embed_dim).
#
# Para BLIP:
#   - n_tokens = 577  (1 token CLS + 576 patch tokens)
#   - embed_dim = 768
#
# pytorch-grad-cam incluye vit_reshape_transform para convertir esta secuencia
# en un pseudo-feature-map 2D compatible con Grad-CAM:
#
#     from pytorch_grad_cam.utils.image import reshape_transform
#     # o, dependiendo de la versión:
#     from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
#
# Este transform:
#   1. Descarta el token CLS (posición 0).
#   2. Toma los 576 patch tokens restantes.
#   3. Los reorganiza como (batch, embed_dim, 24, 24).
#
# IMPORTANTE: el transform estándar asume 197 tokens (para 224×224). BLIP usa
# 384×384 → 577 tokens. Hay que pasar el parámetro correcto:
#
#     def blip_vit_reshape_transform(tensor, height=24, width=24):
#         result = tensor[:, 1:, :]          # descarta CLS
#         result = result.reshape(
#             result.size(0), height, width, result.size(2)
#         )
#         result = result.transpose(2, 3).transpose(1, 2)  # → (B, C, H, W)
#         return result
#
# ── El target de Grad-CAM: un token específico del caption ──────────────────
#
# Grad-CAM fue diseñado para clasificación, donde el "target" es un logit de
# clase. Para generación de texto, el target es el logit del token que se
# quiere explicar.
#
# El problema: model.generate() hace decodificación autoregresiva y no expone
# directamente los logits intermedios para Grad-CAM.
#
# Solución estándar: hacer un forward pass del modelo en modo "scoring", no
# en modo "generate". Se pasan los tokens ya generados como input_ids al
# modelo (modo teacher forcing) y se extrae el logit del token de interés en
# la posición correcta.
#
# Flujo concreto:
#   1. Obtener el caption completo: ids = model.generate(**inputs)
#   2. Para el token en posición t (token_id = ids[0][t]):
#      a. Hacer forward pass con input_ids = ids[0][:t+1]
#      b. Definir un target que devuelva logits[:, t, token_id]
#      c. Pasar ese target a cam()
#
# Clase target custom:
#
#     class TokenTarget:
#         def __init__(self, token_id):
#             self.token_id = token_id
#         def __call__(self, model_output):
#             # model_output es el tensor de logits del decoder
#             # shape: (batch, seq_len, vocab_size)
#             return model_output[:, -1, self.token_id]
#
# ── Estructura del objeto GradCAM ───────────────────────────────────────────
#
#     from pytorch_grad_cam import GradCAM
#
#     cam = GradCAM(
#         model=wrapper,           # un wrapper que expone solo el encoder ViT
#         target_layers=[model.vision_model.encoder.layers[-1].layer_norm1],
#         reshape_transform=blip_vit_reshape_transform,
#     )
#
# El "wrapper" es necesario porque GradCAM espera un modelo que devuelva un
# tensor sobre el cual calcular el gradiente. Si se pasa el modelo BLIP
# completo, el output es un dict o un dataclass, no un tensor plano.
# El wrapper debe encapsular el forward pass y devolver el logit del token
# objetivo como escalar (o tensor 1D).
#
# ── Qué devolver ────────────────────────────────────────────────────────────
#
# La función principal `compute_gradcam` debe devolver el mismo formato que
# extract_cross_attention para poder compararlos directamente:
#
#     {
#         "caption": "port-a-cath tip over mid svc",
#         "tokens": ["port", "-", "a", "-", "cath", ...],
#         "heatmaps": {
#             0: np.ndarray (384, 384),
#             1: np.ndarray (384, 384),
#             ...
#         }
#     }
#
# pytorch-grad-cam ya devuelve el heatmap upscaleado al tamaño de la imagen
# de entrada (384×384 en este caso), así que no hay que hacer el upscale manual.
#
# ── Decisiones de implementación a tomar ───────────────────────────────────
#
# 1. ¿Grad-CAM o GradCAM++ o EigenCAM?
#    Recomendación: GradCAM estándar primero. GradCAM++ puede dar mapas más
#    nítidos pero la diferencia suele ser pequeña. EigenCAM no usa gradientes
#    (más estable numéricamente pero menos interpretable semánticamente).
#
# 2. ¿Cómo manejar tokens de puntuación y stopwords?
#    Se pueden calcular para todos los tokens y dejar el filtrado a la
#    visualización. O filtrar antes de computar para ahorrar tiempo.
#    Recomendación: calcular para todos y filtrar en visualization/.
#
# 3. ¿Normalización del heatmap?
#    pytorch-grad-cam ya normaliza a [0,1] por defecto. Dejar ese comportamiento.
#
# ── Dependencias necesarias ─────────────────────────────────────────────────
#
#     import torch
#     import numpy as np
#     from pytorch_grad_cam import GradCAM
#     from transformers import BlipForConditionalGeneration, BlipProcessor
#
# ── Firma sugerida para la función principal ─────────────────────────────────
#
#     def compute_gradcam(
#         model: BlipForConditionalGeneration,
#         processor: BlipProcessor,
#         image: PIL.Image,
#         device: str = "cpu",
#     ) -> dict:
#         ...
#
# TODO: implementar esta función.
"""
