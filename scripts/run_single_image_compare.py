"""Corre el pipeline comparativo sobre una sola imagen local.

No depende del dataset de Hugging Face ni de splits.
Sirve para probar el pipeline de explicabilidad con una imagen puntual.

Ejemplo:

    python scripts/run_single_image_compare.py \
        --image-path data/test_single/image.png \
        --model-dir models/blip_base \
        --output-dir outputs/single_image_compare/base_test \
        --device cpu \
        --skip-gradcam
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models.blip_loader import load_model_and_processor
from src.models.generation import generate_caption_best_of_n, token_ids_to_tensor
from src.interpretability.compare import extract_all_methods
from src.metrics.spatial_metrics import (
    compute_spatial_metrics,
    save_metrics_json,
    save_rows_csv,
)


def _to_jsonable(obj: Any) -> Any:
    """Convierte objetos numpy/torch a tipos serializables."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    return obj


def _save_compare_npz(compare_result: dict[str, Any], output_path: Path) -> None:
    """Guarda heatmaps principales en NPZ para análisis posterior."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}

    for method in ("post_softmax", "qk_logits", "gradcam"):
        if method not in compare_result:
            continue

        maps = compare_result[method].get("maps", [])
        words = [word for word, _ in maps]
        heatmaps = np.asarray([hm for _, hm in maps], dtype=np.float32)

        payload[f"{method}_words"] = np.asarray(words, dtype=object)
        payload[f"{method}_heatmaps"] = heatmaps
        payload[f"{method}_caption"] = np.asarray(
            [compare_result[method].get("caption", "")],
            dtype=object,
        )

    np.savez_compressed(output_path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de comparación de interpretabilidad para una imagen local."
    )

    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Ruta a una imagen local. Puede ser PNG/JPG.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/blip_base"),
        help="Ruta al modelo BLIP. Default: models/blip_base.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/single_image_compare"),
        help="Carpeta donde guardar outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device de ejecución. Default: cpu.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=9,
        help="Capa del decoder para post-softmax/QK. Default: 9.",
    )
    parser.add_argument(
        "--head-reduction",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Reducción de heads. Default: max.",
    )
    parser.add_argument(
        "--skip-gradcam",
        action="store_true",
        help="Saltea Grad-CAM para prueba rápida.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Temperatura para generación sampling. Default: 1.2.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling. Default: 0.95.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Máximo de tokens nuevos. Default: 40.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Seeds para best-of-N. Default: 42 43 44.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {args.image_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Imagen: {args.image_path}")
    print(f"Modelo: {args.model_dir}")
    print(f"Output: {args.output_dir}")

    image = Image.open(args.image_path).convert("RGB")

    print("Cargando modelo...")
    model, processor = load_model_and_processor(
        model_dir=args.model_dir,
        device=args.device,
    )

    print("Generando caption best-of-N...")
    gen = generate_caption_best_of_n(
        model=model,
        processor=processor,
        image=image,
        device=args.device,
        seeds=args.seeds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    caption = gen["caption"]
    token_ids = gen["token_ids"]
    generated_ids = token_ids_to_tensor(token_ids, device=args.device)

    print("Caption elegida:")
    print(caption)
    print(f"n_token_ids: {len(token_ids)}")

    caption_path = args.output_dir / "caption.json"
    with open(caption_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(gen), f, indent=2, ensure_ascii=False)

    print(f"Caption guardada en: {caption_path}")

    print("Corriendo métodos de interpretabilidad...")
    compare_result = extract_all_methods(
        model=model,
        processor=processor,
        image=image,
        generated_ids=generated_ids,
        device=args.device,
        layer_idx=args.layer_idx,
        head_reduction=args.head_reduction,
        include_gradcam=not args.skip_gradcam,
        filter_tokens=True,
    )

    compare_json_path = args.output_dir / "compare_result.json"
    with open(compare_json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(compare_result), f, indent=2, ensure_ascii=False)

    compare_npz_path = args.output_dir / "compare_heatmaps.npz"
    _save_compare_npz(compare_result, compare_npz_path)

    print(f"Compare JSON guardado en: {compare_json_path}")
    print(f"Heatmaps NPZ guardado en: {compare_npz_path}")

    print("Calculando métricas espaciales...")
    metrics = compute_spatial_metrics(compare_result)

    metrics_json_path = args.output_dir / "spatial_metrics.json"
    rows_csv_path = args.output_dir / "spatial_metrics_rows.csv"

    save_metrics_json(metrics, metrics_json_path)
    save_rows_csv(metrics["rows"], rows_csv_path)

    print(f"Métricas JSON guardadas en: {metrics_json_path}")
    print(f"Métricas CSV guardadas en: {rows_csv_path}")

    print("Resumen:")
    for row in metrics["summary"]:
        print(row)

    print("OK single-image compare finalizado.")


if __name__ == "__main__":
    main()
