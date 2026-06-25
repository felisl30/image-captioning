#!/usr/bin/env python3
"""D2 — Probe de distribución top-1/top-2 durante generación.

Objetivo:
    Medir si el mode collapse viene de una distribución extremadamente confiada
    o si greedy decoding está ocultando alternativas cercanas.

Qué hace:
    - Carga checkpoints epoch_1/epoch_2/epoch_3/best.
    - Genera captions greedy sobre las mismas imágenes fijas.
    - Usa generate(output_scores=True, return_dict_in_generate=True).
    - Calcula p_top1, p_top2, gap, ratio, entropía y top-k mass por token.
    - Guarda CSVs agregados por token, imagen y checkpoint.

No entrena. No modifica checkpoints.

Uso real cuando estén los checkpoints:

    python scripts/run_d2_token_probe.py \
        --checkpoint-root models/blip_finetuned_5k \
        --indices data/selected_indices.json \
        --max-images 30 \
        --device cpu

Dry-run:

    python scripts/run_d2_token_probe.py --dry-run

Smoke test con checkpoints debug actuales:

    python scripts/run_d2_token_probe.py \
        --allow-debug \
        --indices data/selected_indices.json \
        --max-images 1 \
        --device cpu
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
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
        "Ejecutá este script desde dentro del repo."
    )


def has_weights(path: Path) -> bool:
    """Detecta si una carpeta parece checkpoint HuggingFace con pesos."""
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def has_tokenizer_or_processor(path: Path) -> bool:
    """Detecta tokenizer/processor."""
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
    """Diagnóstico de una carpeta de checkpoint."""
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


def token_to_str(tokenizer, token_id: int | None) -> str:
    """Convierte token id a string."""
    if token_id is None:
        return ""

    token = tokenizer.convert_ids_to_tokens(int(token_id))

    if isinstance(token, list):
        return token[0] if token else ""

    return str(token)


def decode_token(tokenizer, token_id: int | None) -> str:
    """Decodifica un token individual a texto legible."""
    if token_id is None:
        return ""

    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return token_to_str(tokenizer, token_id)


def tensor_entropy(probs: torch.Tensor) -> float:
    """Entropía de una distribución de probabilidad."""
    eps = 1e-12
    return float(-(probs * torch.log(probs + eps)).sum().item())


def generate_with_token_probe(
    model,
    processor,
    image,
    device: str,
    max_new_tokens: int,
    top_k: int,
    include_special_tokens: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Genera caption greedy y devuelve métricas top-k por paso.

    Returns:
        caption: texto decodificado completo.
        step_rows: lista de métricas por token generado.
    """
    tokenizer = processor.tokenizer
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    caption = processor.decode(out.sequences[0], skip_special_tokens=True).strip()

    if not hasattr(out, "scores") or out.scores is None:
        raise RuntimeError(
            "model.generate no devolvió scores. "
            "Verificar compatibilidad de transformers con output_scores=True."
        )

    scores = list(out.scores)
    sequence = out.sequences[0].detach().cpu()

    # En encoder-decoder, sequences suele ser:
    # [decoder_start/BOS, token_0, token_1, ..., EOS]
    generated_ids = sequence[1 : 1 + len(scores)]

    special_ids = set(getattr(tokenizer, "all_special_ids", []))

    rows: list[dict[str, Any]] = []

    for step, score_batch in enumerate(scores):
        logits = score_batch[0].detach().float().cpu()
        probs = torch.softmax(logits, dim=-1)

        k = min(top_k, probs.numel())
        top_probs, top_ids = torch.topk(probs, k=k)

        generated_token_id = int(generated_ids[step].item()) if step < len(generated_ids) else None

        if (
            generated_token_id is not None
            and not include_special_tokens
            and generated_token_id in special_ids
        ):
            continue

        p_generated = (
            float(probs[generated_token_id].item())
            if generated_token_id is not None
            else math.nan
        )

        top1_id = int(top_ids[0].item())
        top1_prob = float(top_probs[0].item())

        if k >= 2:
            top2_id = int(top_ids[1].item())
            top2_prob = float(top_probs[1].item())
        else:
            top2_id = None
            top2_prob = math.nan

        gap = top1_prob - top2_prob if top2_id is not None else math.nan
        ratio = top1_prob / max(top2_prob, 1e-12) if top2_id is not None else math.nan
        logit_top1 = float(logits[top1_id].item())
        logit_top2 = float(logits[top2_id].item()) if top2_id is not None else math.nan
        logit_gap = logit_top1 - logit_top2 if top2_id is not None else math.nan

        top_ids_list = [int(x.item()) for x in top_ids]
        top_probs_list = [float(x.item()) for x in top_probs]
        top_tokens_list = [token_to_str(tokenizer, x) for x in top_ids_list]

        rows.append(
            {
                "step": step,
                "generated_token_id": generated_token_id,
                "generated_token": token_to_str(tokenizer, generated_token_id),
                "generated_text_piece": decode_token(tokenizer, generated_token_id),
                "p_generated": p_generated,
                "generated_is_top1": generated_token_id == top1_id,
                "top1_id": top1_id,
                "top1_token": token_to_str(tokenizer, top1_id),
                "top1_text_piece": decode_token(tokenizer, top1_id),
                "p_top1": top1_prob,
                "top2_id": top2_id,
                "top2_token": token_to_str(tokenizer, top2_id),
                "top2_text_piece": decode_token(tokenizer, top2_id),
                "p_top2": top2_prob,
                "gap_top1_top2": gap,
                "ratio_top1_top2": ratio,
                "logit_top1": logit_top1,
                "logit_top2": logit_top2,
                "logit_gap_top1_top2": logit_gap,
                "entropy": tensor_entropy(probs),
                "topk_mass": float(top_probs.sum().item()),
                "topk_ids_json": json.dumps(top_ids_list),
                "topk_tokens_json": json.dumps(top_tokens_list, ensure_ascii=False),
                "topk_probs_json": json.dumps(top_probs_list),
            }
        )

    return caption, rows


def summarize_by_image(steps_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por checkpoint e imagen."""
    rows: list[dict[str, Any]] = []

    group_cols = ["checkpoint", "idx"]

    for (checkpoint, idx), group in steps_df.groupby(group_cols, sort=False):
        rows.append(
            {
                "checkpoint": checkpoint,
                "idx": idx,
                "checkpoint_path": group["checkpoint_path"].iloc[0],
                "caption": group["caption"].iloc[0],
                "reference": group["reference"].iloc[0],
                "n_steps": int(len(group)),
                "mean_p_top1": float(group["p_top1"].mean()),
                "median_p_top1": float(group["p_top1"].median()),
                "mean_p_top2": float(group["p_top2"].mean()),
                "median_p_top2": float(group["p_top2"].median()),
                "mean_gap": float(group["gap_top1_top2"].mean()),
                "median_gap": float(group["gap_top1_top2"].median()),
                "mean_ratio": float(group["ratio_top1_top2"].mean()),
                "median_ratio": float(group["ratio_top1_top2"].median()),
                "mean_entropy": float(group["entropy"].mean()),
                "median_entropy": float(group["entropy"].median()),
                "mean_topk_mass": float(group["topk_mass"].mean()),
                "pct_steps_top1_gt_090": float((group["p_top1"] > 0.90).mean()),
                "pct_steps_top1_gt_095": float((group["p_top1"] > 0.95).mean()),
                "pct_steps_top1_gt_099": float((group["p_top1"] > 0.99).mean()),
                "pct_steps_gap_lt_010": float((group["gap_top1_top2"] < 0.10).mean()),
                "pct_steps_gap_lt_005": float((group["gap_top1_top2"] < 0.05).mean()),
                "pct_generated_is_top1": float(group["generated_is_top1"].mean()),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_checkpoint(image_df: pd.DataFrame, steps_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por checkpoint."""
    rows: list[dict[str, Any]] = []

    for checkpoint, group in steps_df.groupby("checkpoint", sort=False):
        image_group = image_df[image_df["checkpoint"] == checkpoint]

        rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_path": group["checkpoint_path"].iloc[0],
                "n_images": int(image_group["idx"].nunique()),
                "n_steps_total": int(len(group)),
                "mean_p_top1": float(group["p_top1"].mean()),
                "median_p_top1": float(group["p_top1"].median()),
                "mean_p_top2": float(group["p_top2"].mean()),
                "median_p_top2": float(group["p_top2"].median()),
                "mean_gap": float(group["gap_top1_top2"].mean()),
                "median_gap": float(group["gap_top1_top2"].median()),
                "mean_ratio": float(group["ratio_top1_top2"].mean()),
                "median_ratio": float(group["ratio_top1_top2"].median()),
                "mean_entropy": float(group["entropy"].mean()),
                "median_entropy": float(group["entropy"].median()),
                "mean_topk_mass": float(group["topk_mass"].mean()),
                "pct_steps_top1_gt_090": float((group["p_top1"] > 0.90).mean()),
                "pct_steps_top1_gt_095": float((group["p_top1"] > 0.95).mean()),
                "pct_steps_top1_gt_099": float((group["p_top1"] > 0.99).mean()),
                "pct_steps_gap_lt_010": float((group["gap_top1_top2"] < 0.10).mean()),
                "pct_steps_gap_lt_005": float((group["gap_top1_top2"] < 0.05).mean()),
                "pct_generated_is_top1": float(group["generated_is_top1"].mean()),
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
        description="D2 — Probe de distribución top-1/top-2 durante generación."
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
        help="JSON con índices fijos. Default: data/selected_indices.json",
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
        help="Cantidad máxima de imágenes. Default: 30.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Máximo de tokens generados. Default: 40.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Cantidad de alternativas top-k a guardar por paso. Default: 5.",
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
        "--include-special-tokens",
        action="store_true",
        help="Incluye tokens especiales como EOS en las métricas.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo diagnostica paths/checkpoints; no carga modelos ni genera captions.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Cantidad máxima de threads CPU para torch. Default: 4.",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Cantidad de ejemplos high-confidence / low-margin a guardar. Default: 10.",
    )

    args = parser.parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    root = find_project_root()
    print(f"PROJECT_ROOT: {root}")

    sys.path.insert(0, str(root))

    from src.data.utils import load_mimic_dataset
    from src.models.blip_loader import load_model_and_processor

    device = detect_device(args.device)
    print(f"Device seleccionado: {device}")
    print(f"Torch threads: {torch.get_num_threads()}")

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
        print("ERROR: no hay checkpoints usables para correr D2.")
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

    if len(checkpoints) < 2:
        print("=" * 80)
        print("ADVERTENCIA: hay menos de dos checkpoints usables.")
        print("=" * 80)
        print("D2 puede correr con uno solo, pero para comparar evolución conviene tener epoch_1/2/3/best.")
        print()

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

    all_step_rows: list[dict[str, Any]] = []

    for checkpoint_name, checkpoint_path in checkpoints.items():
        print("=" * 80)
        print(f"D2 TOKEN PROBE — {checkpoint_name}")
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

            caption, step_rows = generate_with_token_probe(
                model=model,
                processor=processor,
                image=image,
                device=device,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                include_special_tokens=args.include_special_tokens,
            )

            print(f"[{checkpoint_name}] {n:02d}/{len(indices)} idx={idx} caption={caption!r}")

            for row in step_rows:
                row.update(
                    {
                        "checkpoint": checkpoint_name,
                        "checkpoint_path": str(checkpoint_path),
                        "idx": int(idx),
                        "caption": caption,
                        "reference": reference,
                    }
                )
                all_step_rows.append(row)

        del model
        del processor
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_step_rows:
        raise RuntimeError("No se generó ninguna fila de D2. Revisar tokens especiales o generación.")

    steps_df = pd.DataFrame(all_step_rows)

    # Orden de columnas principal.
    preferred_cols = [
        "checkpoint",
        "checkpoint_path",
        "idx",
        "caption",
        "reference",
        "step",
        "generated_token_id",
        "generated_token",
        "generated_text_piece",
        "p_generated",
        "generated_is_top1",
        "top1_id",
        "top1_token",
        "top1_text_piece",
        "p_top1",
        "top2_id",
        "top2_token",
        "top2_text_piece",
        "p_top2",
        "gap_top1_top2",
        "ratio_top1_top2",
        "logit_top1",
        "logit_top2",
        "logit_gap_top1_top2",
        "entropy",
        "topk_mass",
        "topk_ids_json",
        "topk_tokens_json",
        "topk_probs_json",
    ]

    remaining_cols = [c for c in steps_df.columns if c not in preferred_cols]
    steps_df = steps_df[preferred_cols + remaining_cols]

    image_df = summarize_by_image(steps_df)
    checkpoint_df = summarize_by_checkpoint(image_df=image_df, steps_df=steps_df)

    high_conf_df = image_df.sort_values(
        ["mean_p_top1", "mean_gap"],
        ascending=[False, False],
    ).head(args.n_examples)

    low_margin_df = image_df.sort_values(
        ["mean_gap", "mean_p_top1"],
        ascending=[True, True],
    ).head(args.n_examples)

    steps_path = output_dir / "d2_token_probe_steps.csv"
    image_path = output_dir / "d2_image_summary.csv"
    checkpoint_path = output_dir / "d2_checkpoint_summary.csv"
    high_conf_path = output_dir / "d2_high_confidence_examples.csv"
    low_margin_path = output_dir / "d2_low_margin_examples.csv"
    checkpoint_json_path = output_dir / "d2_checkpoint_summary.json"

    steps_df.to_csv(steps_path, index=False)
    image_df.to_csv(image_path, index=False)
    checkpoint_df.to_csv(checkpoint_path, index=False)
    high_conf_df.to_csv(high_conf_path, index=False)
    low_margin_df.to_csv(low_margin_path, index=False)

    checkpoint_json_path.write_text(
        json.dumps(checkpoint_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("RESUMEN D2 POR CHECKPOINT")
    print("=" * 80)

    cols_to_print = [
        "checkpoint",
        "n_images",
        "n_steps_total",
        "mean_p_top1",
        "mean_p_top2",
        "mean_gap",
        "mean_entropy",
        "pct_steps_top1_gt_090",
        "pct_steps_gap_lt_010",
    ]
    print(checkpoint_df[cols_to_print].to_string(index=False))
    print()

    print("=" * 80)
    print("ARCHIVOS GUARDADOS")
    print("=" * 80)
    print(steps_path)
    print(image_path)
    print(checkpoint_path)
    print(checkpoint_json_path)
    print(high_conf_path)
    print(low_margin_path)
    print()

    print("=" * 80)
    print("INTERPRETACIÓN RÁPIDA")
    print("=" * 80)
    print("- mean_p_top1 alto, por ejemplo > 0.90: el modelo está muy confiado.")
    print("- mean_gap alto: el top-1 está lejos del top-2; greedy no es el único problema.")
    print("- mean_gap bajo, por ejemplo < 0.10: hay alternativas cercanas; sampling puede ayudar.")
    print("- pct_steps_gap_lt_010 alto: muchas decisiones token a token son ambiguas.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
