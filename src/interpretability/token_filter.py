"""Token filtering utilities for BLIP interpretability heatmaps.

This module removes low-information tokens from word-level heatmap displays.
The medical vocabulary is used only for highlighting/scoring, not for strict
filtering. This keeps the analysis open to unexpected but meaningful tokens.
"""

from __future__ import annotations

import re
import string
from collections.abc import Iterable
from typing import Any

import numpy as np


STOPWORDS: set[str] = {
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those",

    # Prepositions / conjunctions
    "of", "in", "on", "at", "to", "with", "and", "or", "for",
    "from", "by", "as", "into", "than", "then", "but", "over",

    # Pronouns / expletives
    "it", "its", "there", "which", "who",

    # Auxiliaries / copulas
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",

    # Very common medical filler terms whose heatmaps are usually not spatially meaningful
    "no", "not", "without", "change", "changed", "unchanged",
    "evidence", "process", "acute", "significant", "interval", "stable",
    "again", "seen", "noted", "demonstrated", "visualized",
}

PUNCT: set[str] = set(string.punctuation) | {"``", "''", "“", "”", "’", "–", "—"}
SPECIAL: set[str] = {
    "[CLS]", "[SEP]", "[PAD]", "[BOS]", "[EOS]",
    "<s>", "</s>", "<pad>",
}

# Used for highlighting/scoring, not as a whitelist filter.
MEDICAL: set[str] = {
    # Findings / diagnoses
    "effusion", "edema", "atelectasis", "consolidation", "opacity",
    "opacities", "opacification", "cardiomegaly", "pneumothorax",
    "pneumonia", "congestion", "infiltrate", "infiltrates", "nodule",
    "nodules", "mass", "fracture", "emphysema", "hyperinflation",
    "enlargement", "silhouette", "collapse", "granuloma", "granulomas",
    "scar", "scarring", "airspace", "interstitial", "vascular",
    "pleural", "pulmonary", "cardiac", "mediastinal", "hilar",
    "bibasilar", "basilar", "apical",

    # Devices / lines
    "tube", "catheter", "line", "picc", "port", "device", "devices",
    "pacemaker", "lead", "leads", "endotracheal", "enteric", "ng",

    # Anatomy / spatial modifiers
    "left", "right", "upper", "lower", "bilateral", "unilateral",
    "mid", "base", "bases", "apex", "apices", "lung", "lungs",
    "chest", "heart", "cardiomediastinal", "diaphragm",

    # Severity / size modifiers
    "mild", "moderate", "severe", "small", "large", "trace",
    "minimal", "prominent", "patchy", "focal", "diffuse", "increased",
    "decreased", "new", "worsening", "improved",
}


def normalize_token(token: str) -> str:
    """Normalize a tokenizer token for filtering and comparison.

    Args:
        token: Raw token produced by the tokenizer or merged token text.

    Returns:
        Lowercased token stripped of whitespace and surrounding punctuation.
    """
    token = str(token).strip()
    token = token.replace("Ġ", "").replace("▁", "")
    token = token.lower()
    token = token.strip()
    token = token.strip(string.punctuation)
    return token


def is_special_token(token: str) -> bool:
    """Return True if token is a known special token."""
    raw = str(token).strip()
    return raw in SPECIAL or raw.upper() in SPECIAL


def is_punctuation(token: str) -> bool:
    """Return True if token is only punctuation or symbols."""
    raw = str(token).strip()
    if not raw:
        return True

    return all((ch in PUNCT or ch.isspace()) for ch in raw)


def is_subword_remainder(token: str) -> bool:
    """Return True for unmerged WordPiece remainders like '##ing'.

    In the normal pipeline, subwords should already be merged by
    merge_subword_attentions. This check is defensive.
    """
    return str(token).strip().startswith("##")


def is_blacklisted(token: str) -> bool:
    """Return True if a token should be hidden from heatmap display.

    Args:
        token: Token or merged word.

    Returns:
        True for stopwords, punctuation, special tokens, empty strings,
        numeric-only tokens, and unmerged WordPiece remainders.
    """
    raw = str(token).strip()

    if not raw:
        return True

    if is_special_token(raw):
        return True

    if is_subword_remainder(raw):
        return True

    if is_punctuation(raw):
        return True

    tok = normalize_token(raw)

    if not tok:
        return True

    if tok in STOPWORDS:
        return True

    if tok in SPECIAL:
        return True

    if tok.isnumeric():
        return True

    # Remove isolated non-alphanumeric fragments.
    if re.fullmatch(r"[^a-zA-Z0-9]+", tok):
        return True

    return False


def is_medical(token: str) -> bool:
    """Return True if token belongs to the medical highlight vocabulary.

    This is intentionally not used as a whitelist. Non-medical tokens can still
    be relevant and should remain visible if they are not blacklisted.
    """
    tok = normalize_token(token)
    return tok in MEDICAL


def filter_relevant_tokens(
    maps: Iterable[tuple[str, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    """Filter low-information tokens from a list of word-level heatmaps.

    Args:
        maps: Iterable of ``(token, heatmap)`` pairs.

    Returns:
        Filtered list preserving original order.
    """
    filtered: list[tuple[str, np.ndarray]] = []

    for token, heatmap in maps:
        if not is_blacklisted(token):
            filtered.append((token, heatmap))

    return filtered


def annotate_tokens(tokens: Iterable[str]) -> list[dict[str, Any]]:
    """Return token metadata useful for debugging or visualization.

    Args:
        tokens: Tokens or merged words.

    Returns:
        List of dictionaries with normalized token, blacklist flag and
        medical-vocabulary flag.
    """
    rows: list[dict[str, Any]] = []

    for token in tokens:
        rows.append(
            {
                "token": token,
                "normalized": normalize_token(token),
                "is_blacklisted": is_blacklisted(token),
                "is_medical": is_medical(token),
            }
        )

    return rows
