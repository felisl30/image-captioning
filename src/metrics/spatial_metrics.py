"""Métricas espaciales para comparar mapas de interpretabilidad.

Este módulo compara heatmaps 24x24 producidos por distintos métodos:
- post-softmax cross-attention
- QK logits
- Grad-CAM

Trabaja sobre la salida de `src.interpretability.compare.extract_all_methods`.
No depende del dataset ni del modelo.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MapEntry = tuple[str, np.ndarray]


def _as_array_1d(heatmap: Any) -> np.ndarray:
    """Convierte un heatmap a vector 1D float64."""
    arr = np.asarray(heatmap, dtype=np.float64)

    if arr.size == 0:
        raise ValueError("Heatmap vacío.")

    arr = arr.reshape(-1)

    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


def pearson_corr(a: Any, b: Any) -> float:
    """Correlación de Pearson entre dos mapas aplanados.

    Devuelve NaN si alguno de los vectores es constante.
    """
    x = _as_array_1d(a)
    y = _as_array_1d(b)

    if x.shape != y.shape:
        raise ValueError(f"Shapes incompatibles: {x.shape} vs {y.shape}")

    x = x - x.mean()
    y = y - y.mean()

    denom = np.linalg.norm(x) * np.linalg.norm(y)

    if denom == 0:
        return float("nan")

    return float(np.dot(x, y) / denom)


def cosine_similarity(a: Any, b: Any) -> float:
    """Similitud coseno entre dos mapas."""
    x = _as_array_1d(a)
    y = _as_array_1d(b)

    if x.shape != y.shape:
        raise ValueError(f"Shapes incompatibles: {x.shape} vs {y.shape}")

    denom = np.linalg.norm(x) * np.linalg.norm(y)

    if denom == 0:
        return float("nan")

    return float(np.dot(x, y) / denom)


def mse(a: Any, b: Any) -> float:
    """Mean squared error entre dos mapas."""
    x = _as_array_1d(a)
    y = _as_array_1d(b)

    if x.shape != y.shape:
        raise ValueError(f"Shapes incompatibles: {x.shape} vs {y.shape}")

    return float(np.mean((x - y) ** 2))


def topk_iou(a: Any, b: Any, top_fraction: float = 0.10) -> float:
    """IoU entre los patches más activados de cada mapa.

    Por defecto compara el top 10% de los patches.
    Para mapas 24x24, eso son ~58 patches.
    """
    if not (0 < top_fraction <= 1):
        raise ValueError("top_fraction debe estar en (0, 1].")

    x = _as_array_1d(a)
    y = _as_array_1d(b)

    if x.shape != y.shape:
        raise ValueError(f"Shapes incompatibles: {x.shape} vs {y.shape}")

    n = x.size
    k = max(1, int(round(top_fraction * n)))

    idx_x = set(np.argpartition(x, -k)[-k:].tolist())
    idx_y = set(np.argpartition(y, -k)[-k:].tolist())

    union = idx_x | idx_y

    if not union:
        return float("nan")

    return float(len(idx_x & idx_y) / len(union))


def _normalize_word(word: str) -> str:
    """Normalización mínima de token/palabra para matching."""
    return str(word).strip().lower()


def _maps_to_occurrence_dict(maps: Iterable[MapEntry]) -> dict[tuple[str, int], dict[str, Any]]:
    """Convierte lista de mapas a dict por palabra + ocurrencia.

    Esto permite alinear correctamente palabras repetidas como 'no'.
    """
    counts: dict[str, int] = {}
    out: dict[tuple[str, int], dict[str, Any]] = {}

    for position, (word, heatmap) in enumerate(maps):
        norm = _normalize_word(word)
        occ = counts.get(norm, 0)
        counts[norm] = occ + 1

        out[(norm, occ)] = {
            "word": word,
            "position": position,
            "heatmap": heatmap,
        }

    return out


def align_maps(
    maps_a: Iterable[MapEntry],
    maps_b: Iterable[MapEntry],
) -> list[dict[str, Any]]:
    """Alinea mapas de dos métodos.

    Primero intenta alineación por posición si las palabras coinciden.
    Si no coinciden exactamente, usa palabra normalizada + número de ocurrencia.
    """
    list_a = list(maps_a)
    list_b = list(maps_b)

    if len(list_a) == len(list_b):
        words_a = [_normalize_word(w) for w, _ in list_a]
        words_b = [_normalize_word(w) for w, _ in list_b]

        if words_a == words_b:
            return [
                {
                    "word": list_a[i][0],
                    "position_a": i,
                    "position_b": i,
                    "heatmap_a": list_a[i][1],
                    "heatmap_b": list_b[i][1],
                    "alignment": "position",
                }
                for i in range(len(list_a))
            ]

    dict_a = _maps_to_occurrence_dict(list_a)
    dict_b = _maps_to_occurrence_dict(list_b)

    shared_keys = sorted(
        set(dict_a.keys()) & set(dict_b.keys()),
        key=lambda x: (dict_a[x]["position"], x[0], x[1]),
    )

    return [
        {
            "word": dict_a[key]["word"],
            "position_a": dict_a[key]["position"],
            "position_b": dict_b[key]["position"],
            "heatmap_a": dict_a[key]["heatmap"],
            "heatmap_b": dict_b[key]["heatmap"],
            "alignment": "word_occurrence",
        }
        for key in shared_keys
    ]


def compute_pair_metrics(
    method_a: str,
    payload_a: dict[str, Any],
    method_b: str,
    payload_b: dict[str, Any],
    top_fraction: float = 0.10,
) -> list[dict[str, Any]]:
    """Calcula métricas token a token entre dos métodos."""
    maps_a = payload_a.get("maps", [])
    maps_b = payload_b.get("maps", [])

    aligned = align_maps(maps_a, maps_b)

    rows: list[dict[str, Any]] = []

    for item in aligned:
        h_a = item["heatmap_a"]
        h_b = item["heatmap_b"]

        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "word": str(item["word"]),
                "position_a": int(item["position_a"]),
                "position_b": int(item["position_b"]),
                "alignment": item["alignment"],
                "pearson": pearson_corr(h_a, h_b),
                "cosine": cosine_similarity(h_a, h_b),
                "mse": mse(h_a, h_b),
                "top10_iou": topk_iou(h_a, h_b, top_fraction=top_fraction),
            }
        )

    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrega métricas por par de métodos."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        key = (row["method_a"], row["method_b"])
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []

    for (method_a, method_b), group in groups.items():
        item: dict[str, Any] = {
            "method_a": method_a,
            "method_b": method_b,
            "n_tokens": len(group),
        }

        for metric in ("pearson", "cosine", "mse", "top10_iou"):
            vals = np.asarray(
                [
                    row[metric]
                    for row in group
                    if row.get(metric) is not None and math.isfinite(float(row[metric]))
                ],
                dtype=np.float64,
            )

            if vals.size == 0:
                item[f"{metric}_mean"] = float("nan")
                item[f"{metric}_std"] = float("nan")
            else:
                item[f"{metric}_mean"] = float(vals.mean())
                item[f"{metric}_std"] = float(vals.std(ddof=0))

        summary.append(item)

    return summary


def compute_spatial_metrics(
    compare_result: dict[str, Any],
    method_pairs: list[tuple[str, str]] | None = None,
    top_fraction: float = 0.10,
) -> dict[str, Any]:
    """Calcula métricas espaciales para la salida de extract_all_methods.

    Args:
        compare_result: salida de `extract_all_methods`.
        method_pairs: pares de métodos a comparar. Si es None, usa todos los
            pares disponibles entre post_softmax, qk_logits y gradcam.
        top_fraction: fracción usada para IoU de zonas top activadas.

    Returns:
        {
            "rows": métricas por palabra,
            "summary": agregados por par de métodos
        }
    """
    available = [
        method
        for method in ("post_softmax", "qk_logits", "gradcam")
        if method in compare_result
    ]

    if method_pairs is None:
        method_pairs = []
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                method_pairs.append((available[i], available[j]))

    rows: list[dict[str, Any]] = []

    for method_a, method_b in method_pairs:
        if method_a not in compare_result:
            raise KeyError(f"No existe method_a={method_a} en compare_result.")
        if method_b not in compare_result:
            raise KeyError(f"No existe method_b={method_b} en compare_result.")

        rows.extend(
            compute_pair_metrics(
                method_a=method_a,
                payload_a=compare_result[method_a],
                method_b=method_b,
                payload_b=compare_result[method_b],
                top_fraction=top_fraction,
            )
        )

    return {
        "rows": rows,
        "summary": summarize_rows(rows),
    }


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    """Guarda métricas como JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=default)


def save_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Guarda rows de métricas como CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
