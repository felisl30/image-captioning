from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.utils import load_mimic_dataset
from src.interpretability.gradcam import compute_gradcam
from src.models.blip_loader import load_model_and_processor
from src.models.generation import token_ids_to_tensor


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    caption_paths = sorted(
        (ROOT / "outputs" / "notebook_comparativo" / "captions").glob("smoke_base_idx_*.json")
    )

    if not caption_paths:
        raise FileNotFoundError("No encontré smoke_base_idx_*.json. Corré primero smoke_generation_visual_base.py.")

    caption_path = caption_paths[-1]
    obj = json.loads(caption_path.read_text(encoding="utf-8"))

    idx = int(obj["idx"])
    generated_ids = token_ids_to_tensor(obj["token_ids"], device=device)

    print(f"Usando JSON: {caption_path}")
    print(f"idx: {idx}")
    print(f"caption original: {obj['caption']}")
    print(f"generated_ids shape: {tuple(generated_ids.shape)}")

    print("\nCargando dataset...")
    ds = load_mimic_dataset(cache_dir=str(ROOT / "data" / "hf_cache"))
    image = ds["train"][idx]["image"].convert("RGB")

    print("Cargando BLIP base...")
    model, processor = load_model_and_processor(
        model_dir=ROOT / "models" / "blip_base",
        device=device,
    )

    print("Computando Grad-CAM con generated_ids externos...")
    print("Nota: en CPU puede tardar. Este smoke test limita a max_tokens=3.")

    results = compute_gradcam(
        model=model,
        processor=processor,
        images=[image],
        device=device,
        generated_ids_list=[generated_ids],
        max_tokens=3,
    )

    if not results:
        raise RuntimeError("compute_gradcam devolvió lista vacía.")

    result = results[0]
    maps = result["maps"]

    print("\nCaption reconstruida parcial:")
    print(result["caption"])

    print("\nCantidad de mapas:", len(maps))
    print("Mapas:")
    for word, heatmap in maps:
        print(
            f"- {word:<20} shape={heatmap.shape} "
            f"min={float(heatmap.min()):.8f} max={float(heatmap.max()):.8f} "
            f"std={float(heatmap.std()):.8f}"
        )

    assert len(maps) > 0
    assert len(maps) <= 3
    assert all(heatmap.shape == (24, 24) for _, heatmap in maps)

    out_dir = ROOT / "outputs" / "notebook_comparativo" / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"smoke_gradcam_base_idx_{idx}.npz"

    np.savez_compressed(
        out_path,
        generated_ids=generated_ids.detach().cpu().numpy(),
        words=np.array([word for word, _ in maps], dtype=object),
        heatmaps=np.stack([heatmap for _, heatmap in maps], axis=0),
    )

    print(f"\nGuardado NPZ: {out_path}")
    print("OK Grad-CAM acepta generated_ids.")


if __name__ == "__main__":
    main()
