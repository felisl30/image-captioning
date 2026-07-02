"""Calcula métricas de calidad de captions desde summary.csv.

Lee outputs/notebook_comparativo/summary.csv (columnas `reference`, `caption`,
`model_tag`, `idx`), calcula métricas por caption y agregados por modelo, y guarda:

    outputs/notebook_comparativo/metrics/caption_metrics_per_item.csv
    outputs/notebook_comparativo/metrics/caption_metrics_summary.csv

Uso:
    python scripts/run_caption_metrics.py
    python scripts/run_caption_metrics.py --summary <ruta> --coco
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists() and (p / "models").exists():
            return p
    raise RuntimeError("No se encontró la raíz del repo.")


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT))

from src.metrics.caption_metrics import (  # noqa: E402
    compute_caption_metrics,
    compute_coco_metrics,
    unique_ratio,
)


NUMERIC_METRICS = [
    "bleu1", "bleu2", "bleu3", "bleu4",
    "rougeL_f", "rougeL_p", "rougeL_r",
    "med_recall", "med_precision", "med_jaccard", "med_f1",
    "hyp_len", "ref_len",
]

CATEGORIES = [
    "correct_negative", "hallucination", "miss",
    "good_overlap", "partial_overlap", "uncategorized",
]


def load_rows(summary_path: Path) -> list[dict[str, str]]:
    with open(summary_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_per_item(rows: list[dict[str, str]]) -> list[dict]:
    out = []
    for r in rows:
        metrics = compute_caption_metrics(r.get("reference", ""), r.get("caption", ""))
        out.append({
            "idx": r.get("idx", ""),
            "model_tag": r.get("model_tag", ""),
            "reference": r.get("reference", ""),
            "caption": r.get("caption", ""),
            **metrics,
        })
    return out


def _nanmean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def summarize_by_model(
    per_item: list[dict],
    rows: list[dict[str, str]],
    coco: bool,
) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for item in per_item:
        by_model[item["model_tag"]].append(item)

    summary = []
    for model_tag, items in by_model.items():
        entry: dict = {"model_tag": model_tag, "n_items": len(items)}

        for m in NUMERIC_METRICS:
            entry[f"{m}_mean"] = _nanmean([it[m] for it in items])

        # Distribución de categorías clínicas
        cat_counts = {c: 0 for c in CATEGORIES}
        for it in items:
            cat_counts[it["clinical_category"]] = cat_counts.get(it["clinical_category"], 0) + 1
        for c in CATEGORIES:
            entry[f"pct_{c}"] = round(100 * cat_counts[c] / len(items), 1) if items else 0.0

        # Diversidad
        entry["unique_ratio"] = round(unique_ratio([it["caption"] for it in items]), 3)

        # CIDEr opcional
        if coco:
            refs = [it["reference"] for it in items]
            hyps = [it["caption"] for it in items]
            coco_scores = compute_coco_metrics(refs, hyps)
            if coco_scores:
                entry.update({k: round(v, 4) for k, v in coco_scores.items()})

        summary.append(entry)

    # Orden estable: base, ft5k, ft10k, resto
    order = {"base": 0, "ft5k": 1, "ft10k": 2}
    summary.sort(key=lambda e: order.get(e["model_tag"], 99))
    return summary


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def print_summary(summary: list[dict]) -> None:
    cols = ["model_tag", "n_items", "bleu4_mean", "rougeL_f_mean",
            "med_recall_mean", "med_f1_mean", "unique_ratio",
            "pct_good_overlap", "pct_hallucination", "pct_miss"]
    header = " | ".join(f"{c:>16}" for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for e in summary:
        vals = []
        for c in cols:
            v = e.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:>16.3f}")
            else:
                vals.append(f"{str(v):>16}")
        print(" | ".join(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Métricas de calidad de captions.")
    parser.add_argument("--summary", type=Path, default=None,
                        help="Ruta a summary.csv. Por defecto usa el del notebook comparativo.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Directorio de salida. Por defecto metrics/ del notebook comparativo.")
    parser.add_argument("--coco", action="store_true",
                        help="Calcula CIDEr con pycocoevalcap (Python puro, confiable).")
    args = parser.parse_args()

    summary_path = args.summary or ROOT / "outputs/notebook_comparativo/summary.csv"
    out_dir = args.out_dir or ROOT / "outputs/notebook_comparativo/metrics"

    if not summary_path.exists():
        print(f"No existe: {summary_path}")
        raise SystemExit(1)

    print(f"Leyendo: {summary_path}")
    rows = load_rows(summary_path)
    print(f"Filas: {len(rows)}")

    per_item = compute_per_item(rows)
    summary = summarize_by_model(per_item, rows, coco=args.coco)

    per_item_path = out_dir / "caption_metrics_per_item.csv"
    summary_path_out = out_dir / "caption_metrics_summary.csv"

    save_csv(per_item, per_item_path)
    save_csv(summary, summary_path_out)

    print(f"\nGuardado: {per_item_path.relative_to(ROOT)}")
    print(f"Guardado: {summary_path_out.relative_to(ROOT)}")

    print_summary(summary)


if __name__ == "__main__":
    main()
