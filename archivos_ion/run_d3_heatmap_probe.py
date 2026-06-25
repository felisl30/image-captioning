#!/usr/bin/env python3
"""D3 — Heatmap probe base vs fine-tuned.

Compara BLIP base contra BLIP fine-tuneado best/ en pocas radiografías fijas.

Genera:
- captions base y fine-tuned;
- grillas de QK logits / cross-attention por palabra;
- grillas de Grad-CAM por palabra;
- comparación cross-attention vs Grad-CAM;
- CSV/JSON resumen.

No entrena. No modifica checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


FT_CANDIDATES = [
    Path("models/blip_finetuned_5k/best"),
    Path("models/blip_finetuned/best"),
]

DEBUG_FT_CANDIDATES = [
    Path("models/blip_finetuned_debug_save/best"),
    Path("models/blip_finetuned_debug/best"),
    Path("models/blip_finetuned_notebook_debug/best"),
]


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()

    while current != current.parent:
        if (current / "src").exists() and (current / "requirements.txt").exists():
            return current
        current = current.parent

    raise RuntimeError("No pude encontrar la raíz del proyecto.")


def has_weights(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def is_usable_checkpoint(path: Path) -> bool:
    return path.exists() and (path / "config.json").exists() and has_weights(path)


def checkpoint_report(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "config": (path / "config.json").exists(),
        "weights": has_weights(path),
        "processor_config": (path / "processor_config.json").exists(),
        "preprocessor_config": (path / "preprocessor_config.json").exists(),
        "tokenizer_json": (path / "tokenizer.json").exists(),
        "generation_config": (path / "generation_config.json").exists(),
        "usable": is_usable_checkpoint(path),
    }


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def resolve_ft_model_dir(root: Path, explicit: Path | None, allow_debug: bool) -> Path | None:
    if explicit is not None:
        return resolve_path(root, explicit)

    candidates = FT_CANDIDATES.copy()
    if allow_debug:
        candidates += DEBUG_FT_CANDIDATES

    for candidate in candidates:
        path = root / candidate
        if is_usable_checkpoint(path):
            return path

    return None


def load_indices(path: Path, max_images: int) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo de índices: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"El archivo no contiene una lista: {path}")

    indices = [int(x) for x in data]

    if max_images > 0:
        indices = indices[:max_images]

    if not indices:
        raise ValueError("La lista de índices quedó vacía.")

    return indices


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def print_dependency_report(skip_gradcam: bool) -> None:
    print("=" * 80)
    print("DEPENDENCIAS")
    print("=" * 80)

    deps = [
        ("torch", "torch"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("pytorch_grad_cam", "grad-cam"),
    ]

    for import_name, _pip_name in deps:
        ok = dependency_available(import_name)
        extra = " (omitido por --skip-gradcam)" if import_name == "pytorch_grad_cam" and skip_gradcam else ""
        print(f"{import_name:<20} {'OK' if ok else 'FALTA'}{extra}")

    print()


def print_model_report(base_model_dir: Path, ft_model_dir: Path | None) -> None:
    print("=" * 80)
    print("MODELOS")
    print("=" * 80)

    print("Base model:")
    print(json.dumps(checkpoint_report(base_model_dir), indent=2, ensure_ascii=False))
    print()

    print("Fine-tuned model:")
    if ft_model_dir is None:
        print("No encontrado.")
    else:
        print(json.dumps(checkpoint_report(ft_model_dir), indent=2, ensure_ascii=False))
    print()


def make_inputs(processor, image, device: str):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    return inputs.to(device)


def run_cross_att_logits(model, processor, image, device: str, layer_idx: int, head_reduction: str):
    from src.interpretability.cross_att_logits import extract_cross_att_logits

    inputs = make_inputs(processor, image, device=device)

    results = extract_cross_att_logits(
        model=model,
        processor=processor,
        inputs=inputs,
        num_batch=1,
        layer_idx=layer_idx,
        head_reduction=head_reduction,
    )

    if not results:
        raise RuntimeError("extract_cross_att_logits devolvió lista vacía.")

    return results[0]


def run_gradcam(model, processor, image, device: str):
    from src.interpretability.gradcam import compute_gradcam

    results = compute_gradcam(
        model=model,
        processor=processor,
        images=[image],
        device=device,
    )

    if not results:
        raise RuntimeError("compute_gradcam devolvió lista vacía.")

    return results[0]


def save_heatmap_grid(image, result: dict, path: Path, title: str, alpha: float, colormap: str, dpi: int) -> None:
    from src.visualization.heatmap import save_heatmap_grid as _save_heatmap_grid

    _save_heatmap_grid(
        image=image,
        results=result,
        output_path=path,
        title=title,
        alpha=alpha,
        colormap=colormap,
        n_cols=4,
        dpi=dpi,
    )


def save_comparison(image, cross_result: dict, gradcam_result: dict, path: Path, alpha: float, colormap: str, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.visualization.heatmap import plot_comparison_heatmaps

    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_comparison_heatmaps(
        image=image,
        cross_att_results=cross_result,
        gradcam_results=gradcam_result,
        alpha=alpha,
        colormap=colormap,
        dpi=dpi,
    )
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def save_original(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def run_for_model(
    model_tag: str,
    model_path: Path,
    hf_split,
    indices: list[int],
    output_dir: Path,
    device: str,
    skip_gradcam: bool,
    layer_idx: int,
    head_reduction: str,
    alpha: float,
    colormap: str,
    dpi: int,
) -> list[dict[str, Any]]:
    from src.models.blip_loader import load_model_and_processor

    print("=" * 80)
    print(f"CARGANDO MODELO: {model_tag}")
    print("=" * 80)
    print(model_path)

    model, processor = load_model_and_processor(model_dir=model_path, device=device)
    model.eval()

    rows: list[dict[str, Any]] = []

    for pos, idx in enumerate(indices, start=1):
        print("=" * 80)
        print(f"{model_tag} | imagen {pos}/{len(indices)} | idx={idx}")
        print("=" * 80)

        item = hf_split[int(idx)]
        image = item["image"].convert("RGB")
        reference = item.get("impression", "")

        case_dir = output_dir / f"idx_{idx}" / model_tag
        case_dir.mkdir(parents=True, exist_ok=True)

        original_path = case_dir / "original.png"
        cross_path = case_dir / "cross_att_logits_grid.png"
        gradcam_path = case_dir / "gradcam_grid.png"
        comparison_path = case_dir / "cross_vs_gradcam.png"

        save_original(image, original_path)

        row: dict[str, Any] = {
            "model_tag": model_tag,
            "model_path": str(model_path),
            "idx": int(idx),
            "reference": reference,
            "original_path": str(original_path),
            "cross_caption": "",
            "gradcam_caption": "",
            "n_cross_maps": 0,
            "n_gradcam_maps": 0,
            "cross_grid_path": "",
            "gradcam_grid_path": "",
            "comparison_path": "",
            "cross_status": "not_run",
            "gradcam_status": "not_run",
            "error": "",
        }

        cross_result = None
        gradcam_result = None

        try:
            cross_result = run_cross_att_logits(
                model=model,
                processor=processor,
                image=image,
                device=device,
                layer_idx=layer_idx,
                head_reduction=head_reduction,
            )

            row["cross_caption"] = cross_result.get("caption", "")
            row["n_cross_maps"] = len(cross_result.get("maps", []))
            row["cross_status"] = "ok"
            row["cross_grid_path"] = str(cross_path)

            save_heatmap_grid(
                image=image,
                result=cross_result,
                path=cross_path,
                title=f"{model_tag} | idx={idx} | cross-att logits",
                alpha=alpha,
                colormap=colormap,
                dpi=dpi,
            )

            print("Cross-att caption:", row["cross_caption"])

        except Exception as e:
            row["cross_status"] = "error"
            row["error"] += f"[cross] {type(e).__name__}: {e} "
            print("ERROR cross-att:", repr(e))

        if skip_gradcam:
            row["gradcam_status"] = "skipped"
        else:
            try:
                gradcam_result = run_gradcam(
                    model=model,
                    processor=processor,
                    image=image,
                    device=device,
                )

                row["gradcam_caption"] = gradcam_result.get("caption", "")
                row["n_gradcam_maps"] = len(gradcam_result.get("maps", []))
                row["gradcam_status"] = "ok"
                row["gradcam_grid_path"] = str(gradcam_path)

                save_heatmap_grid(
                    image=image,
                    result=gradcam_result,
                    path=gradcam_path,
                    title=f"{model_tag} | idx={idx} | grad-cam",
                    alpha=alpha,
                    colormap=colormap,
                    dpi=dpi,
                )

                print("Grad-CAM caption:", row["gradcam_caption"])

            except Exception as e:
                row["gradcam_status"] = "error"
                row["error"] += f"[gradcam] {type(e).__name__}: {e} "
                print("ERROR gradcam:", repr(e))

        if cross_result is not None and gradcam_result is not None:
            try:
                save_comparison(
                    image=image,
                    cross_result=cross_result,
                    gradcam_result=gradcam_result,
                    path=comparison_path,
                    alpha=alpha,
                    colormap=colormap,
                    dpi=dpi,
                )
                row["comparison_path"] = str(comparison_path)
            except Exception as e:
                row["error"] += f"[comparison] {type(e).__name__}: {e} "
                print("ERROR comparison:", repr(e))

        rows.append(row)

    del model
    del processor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="D3 — Heatmap probe base vs fine-tuned.")

    parser.add_argument("--base-model-dir", type=Path, default=Path("models/blip_base"))
    parser.add_argument("--ft-model-dir", type=Path, default=None)
    parser.add_argument("--indices", type=Path, default=Path("data/selected_indices.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mode_collapse_debug/d3_heatmaps"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--skip-gradcam", action="store_true")
    parser.add_argument("--allow-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--layer-idx", type=int, default=9)
    parser.add_argument("--head-reduction", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--colormap", type=str, default="jet")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=4)

    args = parser.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    root = find_project_root()
    sys.path.insert(0, str(root))

    device = detect_device(args.device)

    base_model_dir = resolve_path(root, args.base_model_dir)
    ft_model_dir = resolve_ft_model_dir(
        root=root,
        explicit=args.ft_model_dir,
        allow_debug=args.allow_debug,
    )

    indices_path = resolve_path(root, args.indices)
    output_dir = resolve_path(root, args.output_dir)
    cache_dir = resolve_path(root, args.cache_dir)

    print("PROJECT_ROOT:", root)
    print("Device:", device)
    print("Torch threads:", torch.get_num_threads())
    print()

    print_dependency_report(skip_gradcam=args.skip_gradcam)
    print_model_report(base_model_dir=base_model_dir, ft_model_dir=ft_model_dir)

    print("=" * 80)
    print("PATHS")
    print("=" * 80)
    print("indices:", indices_path, "OK" if indices_path.exists() else "FALTA")
    print("cache_dir:", cache_dir, "OK" if cache_dir.exists() else "FALTA")
    print("output_dir:", output_dir)
    print()

    gradcam_available = dependency_available("pytorch_grad_cam")

    if not args.skip_gradcam and not gradcam_available:
        print("=" * 80)
        print("ADVERTENCIA: falta pytorch_grad_cam.")
        print("=" * 80)
        print("Para D3 completo instalá:")
        print("  pip install grad-cam")
        print()
        print("O corré solo cross-attention con:")
        print("  --skip-gradcam")
        print()

    if args.dry_run:
        print("Dry-run activado. No cargo modelos ni genero heatmaps.")
        return 0

    if not is_usable_checkpoint(base_model_dir):
        print("ERROR: models/blip_base no parece usable.")
        return 2

    if ft_model_dir is None or not is_usable_checkpoint(ft_model_dir):
        print("ERROR: no encontré checkpoint fine-tuned best usable.")
        print("Esperado, por ejemplo:")
        print("  models/blip_finetuned_5k/best")
        print("  models/blip_finetuned/best")
        print()
        print("Si querés probar con debug:")
        print("  --allow-debug")
        return 2

    if not args.skip_gradcam and not gradcam_available:
        print("ERROR: falta grad-cam para correr D3 completo.")
        print("Instalá: pip install grad-cam")
        print("O usá: --skip-gradcam")
        return 2

    from src.data.utils import load_mimic_dataset

    indices = load_indices(indices_path, max_images=args.max_images)

    print("=" * 80)
    print("ÍNDICES")
    print("=" * 80)
    print("Cantidad:", len(indices))
    print("Primeros:", indices[:10])
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CARGANDO DATASET")
    print("=" * 80)
    print("cache_dir:", cache_dir)
    ds = load_mimic_dataset(cache_dir=str(cache_dir))
    hf_split = ds["train"]
    print(hf_split)
    print()

    all_rows: list[dict[str, Any]] = []

    for model_tag, model_path in [("base", base_model_dir), ("finetuned", ft_model_dir)]:
        rows = run_for_model(
            model_tag=model_tag,
            model_path=model_path,
            hf_split=hf_split,
            indices=indices,
            output_dir=output_dir,
            device=device,
            skip_gradcam=args.skip_gradcam,
            layer_idx=args.layer_idx,
            head_reduction=args.head_reduction,
            alpha=args.alpha,
            colormap=args.colormap,
            dpi=args.dpi,
        )
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)

    summary_csv = output_dir / "d3_heatmap_summary.csv"
    summary_json = output_dir / "d3_heatmap_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 80)
    print("RESUMEN D3")
    print("=" * 80)

    cols = [
        "model_tag",
        "idx",
        "cross_status",
        "gradcam_status",
        "n_cross_maps",
        "n_gradcam_maps",
        "cross_caption",
        "gradcam_caption",
        "error",
    ]

    print(summary_df[cols].to_string(index=False))
    print()

    print("=" * 80)
    print("ARCHIVOS GUARDADOS")
    print("=" * 80)
    print(summary_csv)
    print(summary_json)
    print(output_dir)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
