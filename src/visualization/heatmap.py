"""Superposición de mapas de atención (24×24) sobre imágenes originales."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torch
import torch.nn.functional as F


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Superpone un mapa de atención 24×24 sobre una imagen PIL.

    Args:
        image: imagen original (RGB o grayscale — se convierte a RGB internamente).
        heatmap: array (24, 24) con valores de atención sin normalizar.
        alpha: peso del heatmap en el blend. 0=solo imagen, 1=solo heatmap.
        colormap: nombre de colormap de matplotlib (jet, viridis, hot, etc.).

    Returns:
        PIL Image RGB con el heatmap superpuesto.
    """
    image = image.convert("RGB")
    img_array = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3) en [0,1]

    # normalizar heatmap a [0,1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    # upscale de 24×24 a tamaño de la imagen con interpolación bilineal
    h_tensor = torch.tensor(heatmap_norm).unsqueeze(0).unsqueeze(0).float()  # (1,1,24,24)
    h_up = F.interpolate(
        h_tensor,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()  # (H, W)

    # aplicar colormap → (H, W, 4) RGBA en [0,1]
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(h_up)[:, :, :3]  # descartar alpha del colormap → (H, W, 3)

    # blend
    blended = (1 - alpha) * img_array + alpha * heatmap_rgba
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype(np.uint8))


def _extract_result(results, idx: int) -> dict:
    """Acepta un dict suelto o una lista de dicts y devuelve el elemento en idx."""
    if isinstance(results, list):
        return results[idx]
    return results


def plot_word_heatmaps(
    image: Image.Image,
    results,
    idx: int = 0,
    title: str = "",
    alpha: float = 0.5,
    colormap: str = "jet",
    n_cols: int = 4,
    dpi: int = 200,
) -> plt.Figure:
    """Genera una figura con un subplot por palabra, cada uno con el heatmap superpuesto.

    Args:
        image: imagen original.
        results: dict con claves "caption" y "maps", o lista de esos dicts
            (como devuelve eval_and_extract_cross_att o compute_gradcam).
        idx: índice dentro de la lista. Ignorado si results es un dict suelto.
        title: título general de la figura. Si vacío, se usa el caption.
        alpha: transparencia del heatmap.
        colormap: colormap de matplotlib.
        n_cols: número de columnas en la grilla.
        dpi: resolución de la figura.

    Returns:
        Figura de matplotlib (no se muestra ni guarda — el caller decide).
    """
    result    = _extract_result(results, idx)
    maps_list = result["maps"]           # [(palabra, array), ...]
    caption   = result["caption"]

    n = len(maps_list) + 1               # +1 para la imagen original
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), dpi=dpi)
    axes = np.array(axes).flatten()

    # primer subplot: imagen original
    axes[0].imshow(image.convert("RGB"))
    axes[0].set_title("original", fontsize=8)
    axes[0].axis("off")

    for i, (word, heatmap) in enumerate(maps_list):
        overlay = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(word, fontsize=8)
        axes[i + 1].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title or caption, fontsize=10, y=1.01)
    plt.tight_layout()
    return fig


def save_heatmap_grid(
    image: Image.Image,
    results,
    output_path: Path,
    idx: int = 0,
    title: str = "",
    alpha: float = 0.5,
    colormap: str = "jet",
    n_cols: int = 4,
    dpi: int = 200,
) -> None:
    """Genera y guarda la grilla de heatmaps por palabra en disco.

    Args:
        image: imagen original.
        results: dict o lista de dicts como devuelve eval_and_extract_cross_att.
        output_path: ruta de destino (se crea el directorio si no existe).
        idx: índice dentro de la lista. Ignorado si results es un dict suelto.
        title: título de la figura. Si vacío, se usa el caption.
        alpha: transparencia del heatmap.
        colormap: colormap de matplotlib.
        n_cols: columnas en la grilla.
        dpi: resolución de salida (usar 200+ para poster).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_word_heatmaps(
        image=image,
        results=results,
        idx=idx,
        title=title,
        alpha=alpha,
        colormap=colormap,
        n_cols=n_cols,
        dpi=dpi,
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_comparison_heatmaps(
    image: Image.Image,
    cross_att_results,
    gradcam_results,
    idx: int = 0,
    alpha: float = 0.5,
    colormap: str = "jet",
    dpi: int = 200,
) -> plt.Figure:
    """Compara cross-attention y Grad-CAM lado a lado, una fila por palabra.

    Cada fila muestra: imagen original | cross-attention | Grad-CAM.
    El caption de la figura viene del resultado de cross-attention.

    Args:
        image: imagen original.
        cross_att_results: dict o lista de dicts de eval_and_extract_cross_att.
        gradcam_results: dict o lista de dicts de compute_gradcam.
        idx: índice dentro de las listas. Ignorado si se pasan dicts sueltos.
        alpha: transparencia del heatmap en el blend.
        colormap: colormap de matplotlib.
        dpi: resolución de la figura.

    Returns:
        Figura de matplotlib con 3 columnas (original, cross-att, grad-cam) × n_palabras filas.
    """
    ca_result  = _extract_result(cross_att_results, idx)
    gc_result  = _extract_result(gradcam_results, idx)

    ca_maps = ca_result["maps"]   # [(palabra, array), ...]
    gc_maps = gc_result["maps"]

    # alinear por posición — ambas listas deben tener el mismo largo
    n_words = min(len(ca_maps), len(gc_maps))

    fig, axes = plt.subplots(
        n_words, 3,
        figsize=(9, n_words * 3),
        dpi=dpi,
    )
    # garantizar shape (n_words, 3) aunque n_words == 1
    axes = np.array(axes).reshape(n_words, 3)

    col_titles = ["original", "cross-attention", "grad-cam"]
    for col, label in enumerate(col_titles):
        axes[0, col].set_title(label, fontsize=9, fontweight="bold")

    image_rgb = image.convert("RGB")

    for row in range(n_words):
        word_ca, heatmap_ca = ca_maps[row]
        word_gc, heatmap_gc = gc_maps[row]

        axes[row, 0].imshow(image_rgb)
        axes[row, 0].set_ylabel(word_ca, fontsize=8, rotation=0, labelpad=40, va="center")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay_heatmap(image, heatmap_ca, alpha=alpha, colormap=colormap))
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay_heatmap(image, heatmap_gc, alpha=alpha, colormap=colormap))
        axes[row, 2].axis("off")

    fig.suptitle(ca_result["caption"], fontsize=10, y=1.01)
    plt.tight_layout()
    return fig