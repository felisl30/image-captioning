#!/usr/bin/env python3
"""D1 — Diagnóstico de mode collapse por checkpoint.

Compara captions generadas por checkpoints intermedios:
epoch_1, epoch_2, epoch_3 y best.

No entrena. No modifica checkpoints. Solo hace inferencia y guarda CSV/JSON.

Uso recomendado cuando estén los checkpoints reales:

    python scripts/run_d1_checkpoint_collapse.py \
        --checkpoint-root models/blip_finetuned_5k \
        --indices data/selected_indices.json \
        --max-images 30 \
        --device cpu

Para revisar si está todo sin correr inferencia:

    python scripts/run_d1_checkpoint_collapse.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch


CHECKPOINT_NAMES = ["epoch_1", "epoch_2", "epoch_3", "best"]

REAL_CANDIDATE_ROOTS = [
    Path("models/blip_finetuned_5k"),
    Path("models/blip_finetuned"),
]

DEBUG_CANDIDATE_ROOTS = [
    Path("models/blip_finetuned_debug_save"),
    Path("models/blip_finetuned_debug"),
    Path("models/blip_finetuned_notebook_debug"),
]


def find_project_root(start: Path | None = None) -> Path:
    """Encuentra la raíz del repo subiendo desde el cwd."""
    current = (start or Path.cwd()).resolve()

    while current != current.parent:
        if (current / "src").exists() and (current / "requirements.txt").exists():
            return current
        current = current.parent

    raise RuntimeError(
        "No pude encontrar la raíz del proyecto. "
        "Ejecutá el script desde dentro del repo."
    )


def has_weights(path: Path) -> bool:
    """Detecta si una carpeta parece checkpoint HF con pesos."""
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def has_tokenizer_or_processor(path: Path) -> bool:
    """Detecta si una carpeta tiene processor/tokenizer suficiente."""
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "processor_config.json",
        "preprocessor_config.json",
    ]
    return any((path / name).exists() for name in candidates)


def is_usable_checkpoint(path: Path) -> bool:
    """Checkpoint mínimo compatible con from_pretrained."""
    return path.exists() and (path / "config.json").exists() and has_weights(path)


def checkpoint_report(path: Path) -> dict[str, Any]:
    """Devuelve diagnóstico de una carpeta de checkpoint."""
    return {
        "path": str(path),
        "exists": path.exists(),
        "config": (path / "config.json").exists(),
        "weights": has_weights(path),
        "tokenizer_or_processor": has_tokenizer_or_processor(path),
        "generation_config": (path / "generation_config.json").exists(),
        "usable": is_usable_checkpoint(path),
    }


def discover_checkpoints(root: Path) -> dict[str, Path]:
    """Busca epoch_1/epoch_2/epoch_3/best dentro de una raíz."""
    found: dict[str, Path] = {}

    for name in CHECKPOINT_NAMES:
        path = root / name
        if is_usable_checkpoint(path):
            found[name] = path

    return found


def choose_checkpoint_root(
    root: Path,
    explicit_root: Path | None,
    allow_debug: bool,
) -> tuple[Path | None, dict[str, Path]]:
    """Elige la raíz de checkpoints a usar."""
    if explicit_root is not None:
        ckpt_root = (root / explicit_root).resolve() if not explicit_root.is_absolute() else explicit_root
        return ckpt_root, discover_checkpoints(ckpt_root)

    candidate_roots = REAL_CANDIDATE_ROOTS.copy()

    if allow_debug:
        candidate_roots += DEBUG_CANDIDATE_ROOTS

    for candidate in candidate_roots:
        ckpt_root = root / candidate
        found = discover_checkpoints(ckpt_root)
        if found:
            return ckpt_root, found

    return None, {}


def detect_device(requested: str) -> str:
    """Normaliza device."""
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_indices(path: Path, max_images: int) -> list[int]:
    """Carga lista de índices."""
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo de índices: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"El archivo de índices no contiene una lista: {path}")

    indices = [int(x) for x in data]

    if max_images > 0:
        indices = indices[:max_images]

    if not indices:
        raise ValueError("La lista de índices quedó vacía.")

    return indices


def normalize_caption(text: str) -> str:
    """Normalización simple para contar captions repetidas."""
    return " ".join((text or "").strip().lower().split())


def generate_caption(
    model,
    processor,
    image,
    device: str,
    max_new_tokens: int,
) -> str:
    """Genera caption greedy para una imagen."""
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def summarize_collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas de collapse por checkpoint."""
    rows = []

    for checkpoint, group in df.groupby("checkpoint", sort=False):
        captions = group["caption_norm"].tolist()
        counts = Counter(captions)

        top_caption, top_count = counts.most_common(1)[0]

        rows.append(
            {
                "checkpoint": checkpoint,
                "n_total": len(captions),
                "n_unique": len(counts),
                "unique_ratio": len(counts) / len(captions),
                "top_caption_norm": top_caption,
                "top_count": top_count,
                "top_pct": top_count / len(captions),
            }
        )

    return pd.DataFrame(rows)


def make_examples(df: pd.DataFrame, examples_per_checkpoint: int) -> pd.DataFrame:
    """Arma ejemplos cualitativos por checkpoint."""
    rows = []

    for checkpoint, group in df.groupby("checkpoint", sort=False):
        group = group.copy()

        # Algunos primeros casos fijos.
        for _, row in group.head(examples_per_checkpoint).iterrows():
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "kind": "first_examples",
                    "idx": row["idx"],
                    "reference": row["reference"],
                    "caption": row["caption"],
                }
            )

        # Algunos casos de la caption dominante.
        top_caption = Counter(group["caption_norm"]).most_common(1)[0][0]
        top_group = group[group["caption_norm"] == top_caption].head(examples_per_checkpoint)

        for _, row in top_group.iterrows():
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "kind": "top_caption_examples",
                    "idx": row["idx"],
                    "reference": row["reference"],
                    "caption": row["caption"],
                }
            )

    return pd.DataFrame(rows)


def print_checkpoint_diagnostics(ckpt_root: Path | None, checkpoints: dict[str, Path]) -> None:
    """Imprime estado de checkpoints."""
    print("=" * 80)
    print("CHECKPOINTS")
    print("=" * 80)

    if ckpt_root is None:
        print("No se encontró ninguna raíz de checkpoints usable.")
        return

    print(f"Raíz seleccionada: {ckpt_root}")
    print()

    for name in CHECKPOINT_NAMES:
        path = ckpt_root / name
        report = checkpoint_report(path)

        print(f"{name}: {path}")
        print(f"  exists                 : {report['exists']}")
        print(f"  config.json            : {report['config']}")
        print(f"  weights                : {report['weights']}")
        print(f"  tokenizer/processor    : {report['tokenizer_or_processor']}")
        print(f"  generation_config.json : {report['generation_config']}")
        print(f"  usable                 : {report['usable']}")
        print()

    print("Checkpoints usables detectados:", ", ".join(checkpoints.keys()) or "ninguno")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D1 — Diagnóstico de mode collapse por checkpoint."
    )

    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Raíz que contiene epoch_1/epoch_2/epoch_3/best. "
            "Ej: models/blip_finetuned_5k"
        ),
    )
    parser.add_argument(
        "--indices",
        type=Path,
        default=Path("data/selected_indices.json"),
        help="JSON con índices fijos para comparar checkpoints. Default: data/selected_indices.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mode_collapse_debug"),
        help="Carpeta donde guardar resultados.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hf_cache"),
        help="Cache local del dataset HuggingFace.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device para inferencia. Default: auto.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=30,
        help="Cantidad máxima de imágenes a usar. Default: 30.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Máximo de tokens generados por caption. Default: 40.",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="También evalúa models/blip_base como baseline.",
    )
    parser.add_argument(
        "--allow-debug",
        action="store_true",
        help="Permite usar checkpoints debug si no hay checkpoints reales.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo diagnostica paths/checkpoints; no carga modelos ni genera captions.",
    )
    parser.add_argument(
        "--examples-per-checkpoint",
        type=int,
        default=5,
        help="Cantidad de ejemplos cualitativos por checkpoint.",
    )

    args = parser.parse_args()

    root = find_project_root()
    print(f"PROJECT_ROOT: {root}")

    sys.path.insert(0, str(root))

    from src.data.utils import load_mimic_dataset
    from src.models.blip_loader import load_model_and_processor

    device = detect_device(args.device)
    print(f"Device seleccionado: {device}")

    indices_path = args.indices
    if not indices_path.is_absolute():
        indices_path = root / indices_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir

    explicit_root = args.checkpoint_root
    ckpt_root, checkpoints = choose_checkpoint_root(
        root=root,
        explicit_root=explicit_root,
        allow_debug=args.allow_debug,
    )

    print_checkpoint_diagnostics(ckpt_root, checkpoints)

    if args.include_base:
        base_path = root / "models" / "blip_base"
        if is_usable_checkpoint(base_path):
            checkpoints = {"base": base_path, **checkpoints}
        else:
            print("Advertencia: --include-base activado, pero models/blip_base no parece usable.")

    if not checkpoints:
        print("=" * 80)
        print("ERROR: no hay checkpoints usables para correr D1.")
        print("=" * 80)
        print()
        print("Esperado:")
        print("  models/blip_finetuned_5k/epoch_1")
        print("  models/blip_finetuned_5k/epoch_2")
        print("  models/blip_finetuned_5k/epoch_3")
        print("  models/blip_finetuned_5k/best")
        print()
        print("O alternativamente:")
        print("  models/blip_finetuned/epoch_1")
        print("  models/blip_finetuned/epoch_2")
        print("  models/blip_finetuned/epoch_3")
        print("  models/blip_finetuned/best")
        print()
        print("Cuando tengas esos checkpoints, volvé a correr este script.")
        print()
        return 2

    # Para D1 real, necesitamos al menos dos checkpoints.
    if len(checkpoints) < 2:
        print("=" * 80)
        print("ERROR: hay menos de dos checkpoints usables.")
        print("=" * 80)
        print("Para comparar collapse por época hacen falta al menos epoch_1 y best.")
        print("Detectados:", checkpoints)
        return 2

    indices = load_indices(indices_path, max_images=args.max_images)

    print("=" * 80)
    print("ÍNDICES")
    print("=" * 80)
    print(f"Archivo: {indices_path}")
    print(f"Cantidad usada: {len(indices)}")
    print(f"Primeros 10: {indices[:10]}")
    print()

    print("=" * 80)
    print("OUTPUTS")
    print("=" * 80)
    print(f"Output dir: {output_dir}")
    print()

    if args.dry_run:
        print("Dry-run activado. No cargo modelos ni genero captions.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CARGA DATASET")
    print("=" * 80)
    print(f"Cache dir: {cache_dir}")

    ds = load_mimic_dataset(cache_dir=str(cache_dir))
    hf_split = ds["train"]

    print(hf_split)
    print()

    rows: list[dict[str, Any]] = []

    for checkpoint_name, checkpoint_path in checkpoints.items():
        print("=" * 80)
        print(f"GENERANDO CAPTIONS — {checkpoint_name}")
        print("=" * 80)
        print(f"Path: {checkpoint_path}")

        model, processor = load_model_and_processor(
            model_dir=checkpoint_path,
            device=device,
        )
        model.eval()

        for n, idx in enumerate(indices, start=1):
            item = hf_split[int(idx)]
            image = item["image"]
            reference = item.get("impression", "")

            caption = generate_caption(
                model=model,
                processor=processor,
                image=image,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            print(f"[{checkpoint_name}] {n:02d}/{len(indices)} idx={idx} caption={caption!r}")

            rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": str(checkpoint_path),
                    "idx": int(idx),
                    "reference": reference,
                    "caption": caption,
                    "caption_norm": normalize_caption(caption),
                }
            )

        del model
        del processor
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    captions_df = pd.DataFrame(rows)
    summary_df = summarize_collapse(captions_df)
    examples_df = make_examples(
        captions_df,
        examples_per_checkpoint=args.examples_per_checkpoint,
    )

    captions_path = output_dir / "checkpoint_captions.csv"
    summary_csv_path = output_dir / "checkpoint_collapse_summary.csv"
    summary_json_path = output_dir / "checkpoint_collapse_summary.json"
    examples_path = output_dir / "checkpoint_examples.csv"

    captions_df.to_csv(captions_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    summary_json_path.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("RESUMEN D1")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()

    print("=" * 80)
    print("ARCHIVOS GUARDADOS")
    print("=" * 80)
    print(captions_path)
    print(summary_csv_path)
    print(summary_json_path)
    print(examples_path)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
