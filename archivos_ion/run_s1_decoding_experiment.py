#!/usr/bin/env python3
"""S1 — Decoding experiment para reducir mode collapse sin reentrenar.

Objetivo:
    Probar estrategias de generación alternativas sobre el mismo checkpoint
    fine-tuneado best/ para ver si el collapse viene de greedy decoding.

Estrategias:
    - greedy
    - temperature + nucleus sampling
    - diverse beam search
    - contrastive decoding

No entrena. No modifica checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REAL_MODEL_CANDIDATES = [
    Path("models/blip_finetuned_5k/best"),
    Path("models/blip_finetuned/best"),
]

DEBUG_MODEL_CANDIDATES = [
    Path("models/blip_finetuned_debug_save/best"),
    Path("models/blip_finetuned_debug/best"),
    Path("models/blip_finetuned_notebook_debug/best"),
]

CLINICAL_KEYWORDS = {
    "normal": ["no acute", "no evidence", "unremarkable", "no significant", "clear lungs"],
    "atelectasis": ["atelectasis", "atelectatic", "volume loss"],
    "effusion": ["effusion", "pleural effusion"],
    "edema": ["edema", "pulmonary edema", "vascular congestion", "interstitial edema"],
    "consolidation": ["consolidation", "pneumonia", "opacity", "opacities", "infiltrate"],
    "pneumothorax": ["pneumothorax"],
    "devices": ["tube", "catheter", "line", "picc", "endotracheal", "enteric", "port", "pacemaker", "lead"],
    "cardiac": ["cardiomegaly", "cardiac", "heart size", "enlarged heart"],
}


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


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def resolve_model_dir(root: Path, explicit: Path | None, allow_debug: bool) -> Path | None:
    if explicit is not None:
        return resolve_path(root, explicit)
    candidates = REAL_MODEL_CANDIDATES.copy()
    if allow_debug:
        candidates += DEBUG_MODEL_CANDIDATES
    for candidate in candidates:
        path = root / candidate
        if is_usable_checkpoint(path):
            return path
    return None


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_indices(path: Path, max_images: int) -> list[int]:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_caption(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def caption_len_words(text: str) -> int:
    return len(normalize_caption(text).split())


def classify_caption(text: str) -> list[str]:
    norm = normalize_caption(text)
    found = []
    for category, keywords in CLINICAL_KEYWORDS.items():
        if any(keyword in norm for keyword in keywords):
            found.append(category)
    return found or ["other"]


def build_strategy_configs(samples_per_image: int) -> list[dict[str, Any]]:
    return [
        {
            "strategy": "greedy",
            "description": "Greedy decoding baseline.",
            "num_return_sequences": 1,
            "generate_kwargs": {"num_beams": 1, "do_sample": False},
        },
        {
            "strategy": "sample_t1.2_p0.95",
            "description": "Temperature sampling suave + nucleus.",
            "num_return_sequences": samples_per_image,
            "generate_kwargs": {"do_sample": True, "temperature": 1.2, "top_p": 0.95, "num_beams": 1},
        },
        {
            "strategy": "sample_t1.3_p0.90",
            "description": "Temperature sampling recomendado + nucleus.",
            "num_return_sequences": samples_per_image,
            "generate_kwargs": {"do_sample": True, "temperature": 1.3, "top_p": 0.90, "num_beams": 1},
        },
        {
            "strategy": "sample_t1.5_p0.85",
            "description": "Sampling más agresivo; más diversidad, más riesgo.",
            "num_return_sequences": samples_per_image,
            "generate_kwargs": {"do_sample": True, "temperature": 1.5, "top_p": 0.85, "num_beams": 1},
        },
        {
            "strategy": "diverse_beam",
            "description": "Diverse beam search.",
            "num_return_sequences": min(4, max(1, samples_per_image)),
            "generate_kwargs": {"num_beams": 8, "num_beam_groups": 4, "diversity_penalty": 0.5, "do_sample": False},
        },
        {
            "strategy": "contrastive",
            "description": "Contrastive decoding.",
            "num_return_sequences": 1,
            "generate_kwargs": {"penalty_alpha": 0.6, "top_k": 4, "do_sample": False},
        },
    ]


def generate_captions_for_strategy(model, processor, image, device: str, strategy_config: dict[str, Any], max_new_tokens: int) -> list[str]:
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generate_kwargs = dict(strategy_config["generate_kwargs"])
    generate_kwargs["max_new_tokens"] = max_new_tokens
    generate_kwargs["num_return_sequences"] = int(strategy_config["num_return_sequences"])
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    return [processor.decode(seq, skip_special_tokens=True).strip() for seq in output_ids]


def summarize_strategy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in df.groupby("strategy", sort=False):
        captions = group["caption_norm"].tolist()
        counts = Counter(captions)
        top_caption, top_count = counts.most_common(1)[0]

        categories = Counter()
        for cats in group["categories"]:
            for cat in str(cats).split("|"):
                categories[cat] += 1

        rows.append({
            "strategy": strategy,
            "description": group["description"].iloc[0],
            "n_images": int(group["idx"].nunique()),
            "n_captions": int(len(group)),
            "n_unique": int(len(counts)),
            "unique_ratio": float(len(counts) / len(group)),
            "top_caption_norm": top_caption,
            "top_count": int(top_count),
            "top_pct": float(top_count / len(group)),
            "mean_len_words": float(group["len_words"].mean()),
            "median_len_words": float(group["len_words"].median()),
            "n_categories": int(len(categories)),
            "top_category": categories.most_common(1)[0][0] if categories else "",
            "top_category_count": categories.most_common(1)[0][1] if categories else 0,
            "pct_other": float((group["categories"].str.contains("other")).mean()),
            "pct_normal": float((group["categories"].str.contains("normal")).mean()),
            "pct_specific_clinical": float((~group["categories"].str.contains("other") & ~group["categories"].str.contains("normal")).mean()),
        })
    return pd.DataFrame(rows)


def summarize_image_strategy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, idx), group in df.groupby(["strategy", "idx"], sort=False):
        captions = group["caption_norm"].tolist()
        counts = Counter(captions)
        top_caption, top_count = counts.most_common(1)[0]
        rows.append({
            "strategy": strategy,
            "idx": int(idx),
            "reference": group["reference"].iloc[0],
            "n_captions": int(len(group)),
            "n_unique": int(len(counts)),
            "unique_ratio": float(len(counts) / len(group)),
            "top_caption_norm": top_caption,
            "top_count": int(top_count),
            "top_pct": float(top_count / len(group)),
            "captions_joined": " || ".join(group["caption"].tolist()),
        })
    return pd.DataFrame(rows)


def make_examples(df: pd.DataFrame, examples_per_strategy: int) -> pd.DataFrame:
    rows = []
    for strategy, group in df.groupby("strategy", sort=False):
        group = group.copy()
        for _, row in group.head(examples_per_strategy).iterrows():
            rows.append({
                "strategy": strategy,
                "kind": "first_examples",
                "idx": int(row["idx"]),
                "sample_id": int(row["sample_id"]),
                "reference": row["reference"],
                "caption": row["caption"],
                "categories": row["categories"],
            })
        top_caption = Counter(group["caption_norm"]).most_common(1)[0][0]
        top_group = group[group["caption_norm"] == top_caption].head(examples_per_strategy)
        for _, row in top_group.iterrows():
            rows.append({
                "strategy": strategy,
                "kind": "top_caption_examples",
                "idx": int(row["idx"]),
                "sample_id": int(row["sample_id"]),
                "reference": row["reference"],
                "caption": row["caption"],
                "categories": row["categories"],
            })
    return pd.DataFrame(rows)


def save_strategy_plot(summary_df: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    summary_df.plot(x="strategy", y=["unique_ratio", "top_pct"], kind="bar", ax=ax)
    ax.set_title("S1 — Diversidad vs repetición por estrategia de decoding")
    ax.set_ylabel("Proporción")
    ax.set_xlabel("Estrategia")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = output_dir / "s1_strategy_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="S1 — Decoding strategies para reducir mode collapse sin reentrenar.")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--indices", type=Path, default=Path("data/selected_indices.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/decoding_sampling/s1_selected_30"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--samples-per-image", type=int, default=4)
    parser.add_argument("--examples-per-strategy", type=int, default=5)
    parser.add_argument("--allow-debug", action="store_true")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    set_seed(args.seed)

    root = find_project_root()
    sys.path.insert(0, str(root))

    from src.data.utils import load_mimic_dataset
    from src.models.blip_loader import load_model_and_processor

    device = detect_device(args.device)
    model_dir = resolve_model_dir(root=root, explicit=args.model_dir, allow_debug=args.allow_debug)
    indices_path = resolve_path(root, args.indices)
    output_dir = resolve_path(root, args.output_dir)
    cache_dir = resolve_path(root, args.cache_dir)

    print("=" * 80)
    print("S1 — DECODING EXPERIMENT")
    print("=" * 80)
    print("PROJECT_ROOT:", root)
    print("Device:", device)
    print("Torch threads:", torch.get_num_threads())
    print("Seed:", args.seed)
    print()

    print("=" * 80)
    print("MODELO")
    print("=" * 80)
    if model_dir is None:
        print("No encontré checkpoint fine-tuned usable.")
        print("Esperado:")
        print("  models/blip_finetuned_5k/best")
        print("  models/blip_finetuned/best")
        print("Usá --allow-debug para probar con checkpoints debug.")
    else:
        print(json.dumps(checkpoint_report(model_dir), indent=2, ensure_ascii=False))
    print()

    print("=" * 80)
    print("PATHS")
    print("=" * 80)
    print("indices:", indices_path, "OK" if indices_path.exists() else "FALTA")
    print("cache_dir:", cache_dir, "OK" if cache_dir.exists() else "FALTA")
    print("output_dir:", output_dir)
    print()

    strategies = build_strategy_configs(samples_per_image=args.samples_per_image)
    print("=" * 80)
    print("ESTRATEGIAS")
    print("=" * 80)
    for config in strategies:
        print("-", config["strategy"], "|", config["description"])
        print("  num_return_sequences:", config["num_return_sequences"])
        print("  generate_kwargs:", config["generate_kwargs"])
    print()

    if args.dry_run:
        print("Dry-run activado. No cargo modelo ni genero captions.")
        return 0

    if model_dir is None or not is_usable_checkpoint(model_dir):
        print("ERROR: no hay checkpoint fine-tuned best usable para S1.")
        print("Cuando tengas el checkpoint real, usá por ejemplo:")
        print("  --model-dir models/blip_finetuned_5k/best")
        return 2

    indices = load_indices(indices_path, max_images=args.max_images)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CARGANDO DATASET")
    print("=" * 80)
    print("cache_dir:", cache_dir)
    ds = load_mimic_dataset(cache_dir=str(cache_dir))
    hf_split = ds["train"]
    print(hf_split)
    print()

    model_jobs = [("finetuned", model_dir)]
    if args.include_base:
        base_dir = root / "models" / "blip_base"
        if is_usable_checkpoint(base_dir):
            model_jobs.insert(0, ("base", base_dir))
        else:
            print("Advertencia: --include-base activado, pero models/blip_base no parece usable.")

    all_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for model_tag, model_path in model_jobs:
        print("=" * 80)
        print(f"CARGANDO MODELO: {model_tag}")
        print("=" * 80)
        print(model_path)
        model, processor = load_model_and_processor(model_dir=model_path, device=device)
        model.eval()

        for pos, idx in enumerate(indices, start=1):
            item = hf_split[int(idx)]
            image = item["image"]
            reference = item.get("impression", "")
            print("=" * 80)
            print(f"{model_tag} | imagen {pos}/{len(indices)} | idx={idx}")
            print("=" * 80)

            for config in strategies:
                strategy = config["strategy"]
                print(f"  estrategia: {strategy}")
                try:
                    captions = generate_captions_for_strategy(
                        model=model,
                        processor=processor,
                        image=image,
                        device=device,
                        strategy_config=config,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for sample_id, caption in enumerate(captions):
                        categories = classify_caption(caption)
                        all_rows.append({
                            "model_tag": model_tag,
                            "model_path": str(model_path),
                            "strategy": strategy,
                            "description": config["description"],
                            "generate_kwargs_json": json.dumps(config["generate_kwargs"], ensure_ascii=False),
                            "idx": int(idx),
                            "sample_id": int(sample_id),
                            "reference": reference,
                            "caption": caption,
                            "caption_norm": normalize_caption(caption),
                            "len_words": caption_len_words(caption),
                            "categories": "|".join(categories),
                        })
                        print(f"    [{sample_id}] {caption!r}")
                except Exception as e:
                    error_rows.append({
                        "model_tag": model_tag,
                        "model_path": str(model_path),
                        "strategy": strategy,
                        "idx": int(idx),
                        "error_type": type(e).__name__,
                        "error": str(e),
                    })
                    print(f"    ERROR {type(e).__name__}: {e}")

        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_rows:
        print("ERROR: no se generó ninguna caption.")
        return 2

    captions_df = pd.DataFrame(all_rows)
    strategy_summary_df = summarize_strategy(captions_df)
    image_strategy_df = summarize_image_strategy(captions_df)
    examples_df = make_examples(captions_df, examples_per_strategy=args.examples_per_strategy)
    errors_df = pd.DataFrame(error_rows)

    captions_path = output_dir / "s1_all_captions.csv"
    summary_path = output_dir / "s1_decoding_summary.csv"
    image_summary_path = output_dir / "s1_image_strategy_summary.csv"
    examples_path = output_dir / "s1_examples.csv"
    errors_path = output_dir / "s1_errors.csv"
    summary_json_path = output_dir / "s1_decoding_summary.json"

    captions_df.to_csv(captions_path, index=False)
    strategy_summary_df.to_csv(summary_path, index=False)
    image_strategy_df.to_csv(image_summary_path, index=False)
    examples_df.to_csv(examples_path, index=False)
    errors_df.to_csv(errors_path, index=False)
    summary_json_path.write_text(
        json.dumps(strategy_summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        plot_path = save_strategy_plot(strategy_summary_df, output_dir=output_dir, dpi=args.dpi)
    except Exception as e:
        plot_path = None
        print("Advertencia: no se pudo guardar plot:", repr(e))

    print()
    print("=" * 80)
    print("RESUMEN S1")
    print("=" * 80)
    cols = [
        "strategy",
        "n_images",
        "n_captions",
        "n_unique",
        "unique_ratio",
        "top_pct",
        "mean_len_words",
        "pct_normal",
        "pct_specific_clinical",
        "top_caption_norm",
    ]
    print(strategy_summary_df[cols].to_string(index=False))
    print()

    print("=" * 80)
    print("ARCHIVOS GUARDADOS")
    print("=" * 80)
    print(captions_path)
    print(summary_path)
    print(summary_json_path)
    print(image_summary_path)
    print(examples_path)
    print(errors_path)
    if plot_path is not None:
        print(plot_path)
    print()

    print("=" * 80)
    print("INTERPRETACIÓN RÁPIDA")
    print("=" * 80)
    print("- Si sampling/diverse_beam suben unique_ratio y bajan top_pct, S1 ayuda.")
    print("- Si todas las estrategias mantienen top_pct alto, el collapse está internalizado.")
    print("- Si aumenta diversidad pero baja coherencia clínica, conviene reportarlo como trade-off.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
