from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# Permite ejecutar este archivo como:
# python scripts/smoke_generation_visual_base.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.utils import load_mimic_dataset
from src.models.blip_loader import load_model_and_processor
from src.models.generation import generate_caption_best_of_n, token_ids_to_tensor


OUT_DIR = ROOT / "outputs" / "notebook_comparativo" / "captions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    visual_indices_path = ROOT / "data" / "visual_test_indices.json"
    visual_indices = json.loads(visual_indices_path.read_text(encoding="utf-8"))

    idx = int(visual_indices[0])
    print(f"Usando idx visual: {idx}")

    print("Cargando dataset...")
    ds = load_mimic_dataset(cache_dir=str(ROOT / "data" / "hf_cache"))
    row = ds["train"][idx]

    image = row["image"].convert("RGB")
    reference = row.get("impression") or ""

    print("Cargando BLIP base...")
    model, processor = load_model_and_processor(
        model_dir=ROOT / "models" / "blip_base",
        device=device,
    )

    print("Generando best-of-3...")
    result = generate_caption_best_of_n(
        model=model,
        processor=processor,
        image=image,
        seeds=(42, 43, 44),
        temperature=1.2,
        top_p=0.95,
        max_new_tokens=40,
        device=device,
    )

    tensor_ids = token_ids_to_tensor(result["token_ids"], device=device)

    print("\nReferencia impression:")
    print(reference)

    print("\nCaption elegida:")
    print(result["caption"])

    print("\nSeed elegida:", result["chosen_seed"])
    print("Score:", result["score"])
    print("Tensor shape:", tuple(tensor_ids.shape))

    print("\nCandidatas:")
    for c in result["candidates"]:
        print("-" * 80)
        print(
            f"seed={c['seed']} | "
            f"score={c['score']} | "
            f"medical={c['medical_richness']} | "
            f"rep={c['rep_penalty']:.3f} | "
            f"len={c['len_penalty']:.3f} | generic={c.get('generic_penalty', 0.0):.3f}"
        )
        print(c["caption"])
        print("word_tokens:", c["word_tokens"])

    out = {
        "idx": idx,
        "reference": reference,
        "model_tag": "base",
        **result,
    }

    out_path = OUT_DIR / f"smoke_base_idx_{idx}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
