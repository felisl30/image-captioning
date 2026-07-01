"""Caption generation helpers for BLIP comparative explainability.

The main entry point is generate_caption_best_of_n, which generates several
sampled captions with fixed seeds and selects the best candidate using a simple,
deterministic score.

This is intended for the comparative explainability notebook: each model
generates its own caption with T=1.2/top-p sampling, and the interpretability
methods later explain that same generated sequence.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.interpretability.token_filter import MEDICAL, is_blacklisted, normalize_token

logger = logging.getLogger(__name__)


# Tokens that often appear in BLIP-base hallucinations on radiographs.
# They are not necessarily wrong in natural-image captioning, but they are bad
# signals for choosing a medical caption.
GENERIC_NON_MEDICAL_TOKENS: set[str] = {
    "photo", "image", "images", "picture", "pictures", "stock", "hd",
    "logo", "background", "camera", "lens", "person", "people", "man",
    "woman", "boy", "girl", "object", "objects", "view", "visible",
    "show", "shows", "shown", "look", "looks", "black", "white",
    "dark", "free", "download", "vector", "illustration", "abdomen",
    "abdominal", "body", "blade", "surgery", "test", "result",
    "function", "thick", "case", "described", "available", "cause",
}

# Anatomical terms are useful context, but weak evidence by themselves.
# A caption saying only "chest/lung" should not beat one mentioning a finding.
WEAK_ANATOMY_TOKENS: set[str] = {
    "chest", "lung", "lungs", "heart", "cardiomediastinal",
    "mediastinal", "hilar", "diaphragm", "abdomen", "body",
}

# Modifiers are useful when attached to findings, but weak alone.
MODIFIER_ONLY_TOKENS: set[str] = {
    "left", "right", "upper", "lower", "bilateral", "unilateral",
    "mid", "base", "bases", "apex", "apices",
    "mild", "moderate", "severe", "small", "large", "trace",
    "minimal", "prominent", "patchy", "focal", "diffuse",
    "increased", "decreased", "new", "worsening", "improved",
}


def set_generation_seed(seed: int) -> None:
    """Set random seeds used by sampling-based generation."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _word_tokens_from_caption(caption: str) -> list[str]:
    """Tokenize decoded text into rough word tokens for scoring."""
    return [
        normalize_token(tok)
        for tok in re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", caption.lower())
        if normalize_token(tok)
    ]


def _medical_richness(tokens: list[str], medical_vocab: set[str]) -> float:
    """Weighted medical richness score.

    Strong finding/device terms count fully. Anatomy and modifiers count only
    weakly, and only when the caption already contains at least one strong
    medical term. This avoids selecting captions that only say "chest/lung".
    """
    medical_vocab = set(medical_vocab)

    strong_terms = medical_vocab - WEAK_ANATOMY_TOKENS - MODIFIER_ONLY_TOKENS

    strong_hits = {tok for tok in tokens if tok in strong_terms}
    has_strong = bool(strong_hits)

    if not has_strong:
        return 0.0

    modifier_hits = {tok for tok in tokens if tok in MODIFIER_ONLY_TOKENS}
    weak_anatomy_hits = {tok for tok in tokens if tok in WEAK_ANATOMY_TOKENS}

    return (
        1.0 * len(strong_hits)
        + 0.25 * len(modifier_hits)
        + 0.10 * len(weak_anatomy_hits)
    )


def _repetition_penalty(tokens: list[str]) -> float:
    """Return a repetition penalty based on repeated content words and bigrams."""
    content = [tok for tok in tokens if not is_blacklisted(tok)]

    if len(content) < 3:
        return 0.0

    unigram_repeated = (len(content) - len(set(content))) / max(1, len(content))

    bigrams = list(zip(content[:-1], content[1:]))
    if not bigrams:
        return float(unigram_repeated)

    bigram_repeated = (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams))

    return float(max(unigram_repeated, bigram_repeated))


def _length_penalty(tokens: list[str]) -> float:
    """Penalize captions that are too short or suspiciously long."""
    content_tokens = [tok for tok in tokens if not is_blacklisted(tok)]
    n_content = len(content_tokens)
    n_total = len(tokens)

    penalty = 0.0

    if n_content == 0:
        penalty += 3.0
    elif n_content < 3:
        penalty += 1.5

    # Long sampled captions from BLIP-base often become incoherent.
    if n_total > 22:
        penalty += min(2.0, (n_total - 22) / 6.0)

    if n_content > 18:
        penalty += min(1.0, (n_content - 18) / 8.0)

    return float(penalty)


def _generic_penalty(tokens: list[str], medical_vocab: set[str]) -> float:
    """Penalize non-medical natural-image vocabulary and anatomy-only captions."""
    medical_vocab = set(medical_vocab)
    strong_terms = medical_vocab - WEAK_ANATOMY_TOKENS - MODIFIER_ONLY_TOKENS

    generic_hits = [tok for tok in tokens if tok in GENERIC_NON_MEDICAL_TOKENS]
    has_strong_medical = any(tok in strong_terms for tok in tokens)
    has_weak_anatomy = any(tok in WEAK_ANATOMY_TOKENS for tok in tokens)

    penalty = 0.35 * len(generic_hits)

    # "chest/lung" alone is not enough to call a caption medically rich.
    if has_weak_anatomy and not has_strong_medical:
        penalty += 1.0

    return float(min(penalty, 4.0))


def _score_caption(
    caption: str,
    medical_vocab: set[str],
    repetition_weight: float = 2.0,
    length_weight: float = 1.0,
    generic_weight: float = 1.0,
) -> dict[str, float | list[str]]:
    """Score a caption for best-of-N selection.

    The score favors specific medical findings and penalizes repetition,
    abnormal length, and generic natural-image hallucinations.
    """
    tokens = _word_tokens_from_caption(caption)

    medical_richness = _medical_richness(tokens, medical_vocab)
    repetition_penalty = _repetition_penalty(tokens)
    length_penalty = _length_penalty(tokens)
    generic_penalty = _generic_penalty(tokens, medical_vocab)

    score = (
        float(medical_richness)
        - repetition_weight * repetition_penalty
        - length_weight * length_penalty
        - generic_weight * generic_penalty
    )

    return {
        "score": float(score),
        "medical_richness": float(medical_richness),
        "rep_penalty": float(repetition_penalty),
        "len_penalty": float(length_penalty),
        "generic_penalty": float(generic_penalty),
        "word_tokens": tokens,
    }


def _to_device(inputs: Any, device: str) -> Any:
    """Move a HuggingFace BatchEncoding/dict to device."""
    if hasattr(inputs, "to"):
        return inputs.to(device)

    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def generate_caption_once(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
    seed: int,
    temperature: float = 1.2,
    top_p: float = 0.95,
    max_new_tokens: int = 40,
    device: str = "cuda",
) -> dict[str, Any]:
    """Generate one sampled caption with a fixed seed."""
    set_generation_seed(seed)

    model.eval()
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = _to_device(inputs, device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            return_dict_in_generate=False,
        )

    sequence = output_ids[0].detach().cpu()
    token_ids = sequence.tolist()

    caption = processor.decode(sequence, skip_special_tokens=True).strip()
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "caption": caption,
        "token_ids": token_ids,
        "tokens": tokens,
        "seed": int(seed),
    }


def generate_caption_best_of_n(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
    seeds: tuple[int, ...] = (42, 43, 44),
    temperature: float = 1.2,
    top_p: float = 0.95,
    max_new_tokens: int = 40,
    medical_vocab: set[str] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Generate N sampled captions and pick the best deterministic candidate.

    Args:
        model: BLIP model.
        processor: Matching BLIP processor.
        image: PIL image.
        seeds: Sampling seeds. One caption is generated per seed.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        max_new_tokens: Maximum generated caption length.
        medical_vocab: Vocabulary used for scoring. If None, uses token_filter.MEDICAL.
        device: "cuda" or "cpu".

    Returns:
        Dictionary with the selected caption, full token id sequence, decoded
        tokens, selected seed, and all candidate captions with scores.
    """
    if not seeds:
        raise ValueError("seeds must contain at least one seed.")

    medical_vocab = medical_vocab or MEDICAL

    candidates: list[dict[str, Any]] = []

    for seed in seeds:
        generated = generate_caption_once(
            model=model,
            processor=processor,
            image=image,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        score_info = _score_caption(
            caption=generated["caption"],
            medical_vocab=medical_vocab,
        )

        candidate = {
            **generated,
            **score_info,
        }
        candidates.append(candidate)

    # Max score wins. Deterministic tie-break: lower seed wins.
    best = sorted(candidates, key=lambda x: (-float(x["score"]), int(x["seed"])))[0]

    return {
        "caption": best["caption"],
        "token_ids": best["token_ids"],
        "tokens": best["tokens"],
        "chosen_seed": best["seed"],
        "score": best["score"],
        "medical_richness": best["medical_richness"],
        "rep_penalty": best["rep_penalty"],
        "len_penalty": best["len_penalty"],
        "generic_penalty": best.get("generic_penalty", 0.0),
        "candidates": [
            {
                "caption": c["caption"],
                "score": c["score"],
                "seed": c["seed"],
                "medical_richness": c["medical_richness"],
                "rep_penalty": c["rep_penalty"],
                "len_penalty": c["len_penalty"],
                "generic_penalty": c.get("generic_penalty", 0.0),
                "word_tokens": c["word_tokens"],
            }
            for c in candidates
        ],
    }


def token_ids_to_tensor(token_ids: list[int], device: str = "cuda") -> torch.Tensor:
    """Convert cached token id list back to a batch tensor for extractors."""
    return torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
