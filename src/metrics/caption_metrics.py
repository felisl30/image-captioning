"""Métricas de calidad de captions para BLIP/MIMIC-CXR.

Compara la caption generada contra la impresión clínica de referencia. A diferencia
de `spatial_metrics.py` (que compara heatmaps entre métodos de interpretabilidad),
este módulo evalúa **si el modelo genera buenas captions**.

Diseño en dos capas:

1. Métricas sin dependencias (stdlib + numpy). Siempre funcionan:
   - BLEU-1..4 con brevity penalty
   - ROUGE-L (basada en LCS)
   - Overlap de keywords médicos (recall/precision/F1/Jaccard)
   - Categorización clínica (hallucination / miss / correct-negative / overlap)
   - Estadísticas de longitud y diversidad (unique ratio)

2. Métrica opcional (requiere `pycocoevalcap`):
   - CIDEr (pondera n-gramas por TF-IDF, estándar en captioning)

Idea futura (ver METRICAS_CAPTIONS.md): BERTScore para similitud semántica.
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter
from typing import Any, Iterable

import numpy as np

try:
    from src.interpretability.token_filter import MEDICAL, normalize_token
except Exception:  # pragma: no cover - fallback si se corre fuera del paquete
    MEDICAL = set()

    def normalize_token(token: str) -> str:
        return str(token).strip().lower().strip(string.punctuation)


logger = logging.getLogger(__name__)


# Subconjunto de MEDICAL que corresponde a HALLAZGOS/patologías (no anatomía ni
# modificadores). Se usa para detectar si una caption "reporta un hallazgo".
FINDINGS: set[str] = {
    "effusion", "edema", "atelectasis", "consolidation", "opacity",
    "opacities", "opacification", "cardiomegaly", "pneumothorax",
    "pneumonia", "congestion", "infiltrate", "infiltrates", "nodule",
    "nodules", "mass", "fracture", "emphysema", "hyperinflation",
    "enlargement", "collapse", "granuloma", "granulomas",
    "scar", "scarring", "airspace",
}

# Frases que indican estudio normal / sin hallazgos agudos.
NORMAL_PATTERNS = [
    "no acute", "no evidence", "without acute", "unremarkable",
    "no significant", "no cardiopulmonary", "no acute cardiopulmonary",
    "clear lung", "clear lungs", "normal",
]


# --------------------------------------------------------------------------- #
# Tokenización
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """Tokeniza en palabras minúsculas, sin puntuación."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


# --------------------------------------------------------------------------- #
# BLEU (implementación pura, estilo corpus BLEU con smoothing simple)
# --------------------------------------------------------------------------- #
def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> dict[str, float]:
    """BLEU-1..max_n de una hipótesis contra una referencia.

    Usa clipping de n-gramas y brevity penalty. Smoothing +1 en numerador y
    denominador de cada precisión para evitar ceros (BLEU es frágil en textos
    cortos como estos).

    Returns:
        {"bleu1": ..., "bleu2": ..., "bleu3": ..., "bleu4": ...}
        donde bleuN es el BLEU acumulado hasta n-gramas de tamaño N.
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    out: dict[str, float] = {}

    if not hyp_tokens or not ref_tokens:
        for n in range(1, max_n + 1):
            out[f"bleu{n}"] = 0.0
        return out

    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = float(np.exp(1 - ref_len / hyp_len)) if hyp_len > 0 else 0.0

    log_precisions: list[float] = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngram_counts(hyp_tokens, n)
        ref_ngrams = _ngram_counts(ref_tokens, n)

        if not hyp_ngrams:
            log_precisions.append(np.log(1e-9))
        else:
            clipped = sum(
                min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items()
            )
            total = sum(hyp_ngrams.values())
            # smoothing +1
            precision = (clipped + 1.0) / (total + 1.0)
            log_precisions.append(float(np.log(precision)))

        # BLEU acumulado hasta n
        avg_log = float(np.mean(log_precisions))
        out[f"bleu{n}"] = float(bp * np.exp(avg_log))

    return out


# --------------------------------------------------------------------------- #
# ROUGE-L (LCS)
# --------------------------------------------------------------------------- #
def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longitud de la subsecuencia común más larga."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b):
            if x == y:
                curr.append(prev[j] + 1)
            else:
                curr.append(max(prev[j + 1], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(reference: str, hypothesis: str, beta: float = 1.2) -> dict[str, float]:
    """ROUGE-L: F-measure basada en la subsecuencia común más larga.

    Returns:
        {"rougeL_p": precision, "rougeL_r": recall, "rougeL_f": f}
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return {"rougeL_p": 0.0, "rougeL_r": 0.0, "rougeL_f": 0.0}

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    prec = lcs / len(hyp_tokens)
    rec = lcs / len(ref_tokens)

    if prec + rec == 0:
        f = 0.0
    else:
        b2 = beta ** 2
        f = ((1 + b2) * prec * rec) / (rec + b2 * prec)

    return {"rougeL_p": float(prec), "rougeL_r": float(rec), "rougeL_f": float(f)}


# --------------------------------------------------------------------------- #
# Overlap de keywords médicos
# --------------------------------------------------------------------------- #
def medical_keywords(text: str) -> set[str]:
    """Extrae el conjunto de keywords médicos presentes en el texto."""
    tokens = {normalize_token(t) for t in tokenize(text)}
    return {t for t in tokens if t in MEDICAL}


def medical_overlap(reference: str, hypothesis: str) -> dict[str, float]:
    """Overlap de vocabulario médico entre referencia e hipótesis.

    recall    = cuánto del vocabulario médico de la referencia capturó la caption.
    precision = cuánto del vocabulario médico de la caption está en la referencia.
    jaccard   = intersección / unión.
    """
    ref_kw = medical_keywords(reference)
    hyp_kw = medical_keywords(hypothesis)

    inter = ref_kw & hyp_kw
    union = ref_kw | hyp_kw

    recall = len(inter) / len(ref_kw) if ref_kw else float("nan")
    precision = len(inter) / len(hyp_kw) if hyp_kw else float("nan")
    jaccard = len(inter) / len(union) if union else float("nan")

    if recall and precision and not (np.isnan(recall) or np.isnan(precision)) and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    return {
        "med_recall": float(recall) if not np.isnan(recall) else float("nan"),
        "med_precision": float(precision) if not np.isnan(precision) else float("nan"),
        "med_jaccard": float(jaccard) if not np.isnan(jaccard) else float("nan"),
        "med_f1": float(f1),
        "n_ref_keywords": len(ref_kw),
        "n_hyp_keywords": len(hyp_kw),
    }


# --------------------------------------------------------------------------- #
# Categorización clínica
# --------------------------------------------------------------------------- #
def _is_normal(text: str) -> bool:
    """Heurística: el estudio se reporta como normal / sin hallazgos agudos."""
    low = str(text).lower()
    has_finding = bool(medical_keywords(text) & FINDINGS)
    has_normal_phrase = any(p in low for p in NORMAL_PATTERNS)
    # Normal si dice frase normal explícita y no menciona un hallazgo positivo.
    return has_normal_phrase and not has_finding


def clinical_category(reference: str, hypothesis: str) -> str:
    """Clasifica el par (referencia, caption) en una categoría clínica.

    Categorías (siguiendo analisis/02_captions_10k.md):
        correct_negative : ref normal  → cap normal
        hallucination    : ref normal  → cap inventa hallazgo
        miss             : ref hallazgo → cap dice normal
        good_overlap     : recall médico >= 0.5
        partial_overlap  : recall médico > 0 pero < 0.5
        uncategorized    : vocab médico presente pero sin overlap con la ref
    """
    ref_normal = _is_normal(reference)
    hyp_normal = _is_normal(hypothesis)
    hyp_has_finding = bool(medical_keywords(hypothesis) & FINDINGS)

    if ref_normal and (hyp_normal or not hyp_has_finding):
        return "correct_negative"
    if ref_normal and hyp_has_finding:
        return "hallucination"

    # ref tiene hallazgos
    if hyp_normal or not hyp_has_finding:
        return "miss"

    recall = medical_overlap(reference, hypothesis)["med_recall"]
    if np.isnan(recall):
        return "uncategorized"
    if recall >= 0.5:
        return "good_overlap"
    if recall > 0:
        return "partial_overlap"
    return "uncategorized"


# --------------------------------------------------------------------------- #
# Métrica por caption
# --------------------------------------------------------------------------- #
def compute_caption_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    """Calcula todas las métricas sin dependencias para un par (ref, hyp)."""
    row: dict[str, Any] = {}
    row.update(sentence_bleu(reference, hypothesis))
    row.update(rouge_l(reference, hypothesis))
    row.update(medical_overlap(reference, hypothesis))
    row["clinical_category"] = clinical_category(reference, hypothesis)
    row["hyp_len"] = len(tokenize(hypothesis))
    row["ref_len"] = len(tokenize(reference))
    return row


# --------------------------------------------------------------------------- #
# Diversidad (a nivel de conjunto de captions)
# --------------------------------------------------------------------------- #
def unique_ratio(captions: Iterable[str]) -> float:
    """Proporción de captions únicas (detecta mode collapse)."""
    caps = [str(c).strip().lower() for c in captions]
    if not caps:
        return float("nan")
    return len(set(caps)) / len(caps)


# --------------------------------------------------------------------------- #
# Métrica COCO opcional (pycocoevalcap)
# --------------------------------------------------------------------------- #
def compute_coco_metrics(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float] | None:
    """Calcula CIDEr con pycocoevalcap si está instalado.

    Devuelve None si la librería no está disponible. CIDEr es Python puro dentro
    de pycocoevalcap, así que corre confiablemente sin dependencias externas
    (no necesita Java).

    Args:
        references: lista de referencias (una por caption).
        hypotheses: lista de captions generadas (mismo orden y largo).
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except Exception:
        logger.warning("pycocoevalcap no disponible; se omite CIDEr.")
        return None

    gts = {i: [references[i]] for i in range(len(references))}
    res = {i: [hypotheses[i]] for i in range(len(hypotheses))}

    out: dict[str, float] = {}

    try:
        cider_score, _ = Cider().compute_score(gts, res)
        out["cider"] = float(cider_score)
    except Exception as exc:  # pragma: no cover
        logger.warning("CIDEr falló: %s", exc)

    return out or None
