"""Genera una grilla visual a partir de compare_heatmaps.npz.

No corre el modelo. Solo carga:
- imagen original
- archivo NPZ con heatmaps 24x24
y guarda una figura comparativa token x método.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import colormaps


METHODS = [
    ("post_softmax", "Post-softmax"),
    ("qk_logits", "QK logits"),
    ("gradcam", "Grad-CAM"),
]


def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = np.asarray(hm, dtype=np.float32)
    hm = np.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)

    vmin = float(hm.min())
    vmax = float(hm.max())

    if vmax - vmin < 1e-8:
        return np.zeros_like(hm, dtype=np.float32)

    return (hm - vmin) / (vmax - vmin)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone heatmap sobre imagen RGB."""
    image_rgb = image.convert("RGB")
    img_arr = np.asarray(image_rgb).astype(np.float32) / 255.0

    hm = normalize_heatmap(heatmap)
    hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(
        image_rgb.size,
        resample=Image.BILINEAR,
    )
    hm_resized = np.asarray(hm_img).astype(np.float32) / 255.0

    cmap = colormaps["jet"]
    hm_color = cmap(hm_resized)[..., :3]

    overlay = (1.0 - alpha) * img_arr + alpha * hm_color
    overlay = np.clip(overlay, 0.0, 1.0)

    return overlay


def load_method(npz, method: str):
    words_key = f"{method}_words"
    heatmaps_key = f"{method}_heatmaps"

    if words_key not in npz.files or heatmaps_key not in npz.files:
        return None, None

    words = [str(x) for x in npz[words_key].tolist()]
    heatmaps = np.asarray(npz[heatmaps_key], dtype=np.float32)

    return words, heatmaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot de comparación de heatmaps para una sola imagen."
    )

    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--npz-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.45)

    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"No existe image-path: {args.image_path}")

    if not args.npz_path.exists():
        raise FileNotFoundError(f"No existe npz-path: {args.npz_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image_path).convert("RGB")
    npz = np.load(args.npz_path, allow_pickle=True)

    loaded = {}
    for method, title in METHODS:
        words, heatmaps = load_method(npz, method)
        if words is not None:
            loaded[method] = {
                "title": title,
                "words": words,
                "heatmaps": heatmaps,
            }

    if not loaded:
        raise RuntimeError("No se encontraron métodos en el NPZ.")

    # Tomamos como referencia el primer método disponible.
    ref_method = next(iter(loaded.keys()))
    ref_words = loaded[ref_method]["words"]
    n_tokens = min(len(ref_words), args.max_tokens)

    methods_available = list(loaded.keys())
    n_cols = 1 + len(methods_available)
    n_rows = n_tokens

    fig_w = 4.0 * n_cols
    fig_h = 3.2 * max(1, n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for row in range(n_tokens):
        word = ref_words[row]

        # Columna original.
        ax = axes[row][0]
        ax.imshow(image)
        ax.set_title(f"Original\n{word}", fontsize=10)
        ax.axis("off")

        # Columnas de métodos.
        for col, method in enumerate(methods_available, start=1):
            item = loaded[method]
            title = item["title"]
            words = item["words"]
            heatmaps = item["heatmaps"]

            ax = axes[row][col]

            if row < len(words):
                hm = heatmaps[row]
                overlay = overlay_heatmap(image, hm, alpha=args.alpha)
                shown_word = words[row]
                ax.imshow(overlay)
                ax.set_title(f"{title}\n{shown_word}", fontsize=10)
            else:
                ax.imshow(image)
                ax.set_title(f"{title}\nSIN TOKEN", fontsize=10)

            ax.axis("off")

    fig.tight_layout()

    out_path = args.output_dir / "single_image_heatmap_grid.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura guardada en: {out_path}")


if __name__ == "__main__":
    main()
