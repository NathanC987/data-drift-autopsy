"""Concept-level root cause analysis for visual drift.

Embedding-space drift attribution answers *which dimensions moved* but not *what
that means*. This module projects a reference->production drift direction in a
vision-language (CLIP) space onto a fixed basis of natural-language concepts, so
the cause reads as e.g. "the background class shifted toward night / low-light /
wide-shot photography and away from signage".

Method:
  1. Embed reference and production images with a CLIP image encoder.
  2. Embed a curated concept vocabulary with the CLIP text encoder
     (prompt-ensembled).
  3. For each image compute a temperature-scaled softmax over the concepts
     (standard CLIP zero-shot), then average over the population.
  4. Rank concepts by the shift in mean concept mass, ref -> prod.
The difference-vector alignment ``cos(mu_prod - mu_ref, concept)`` is reported as
a secondary cross-check; it is dominated by CLIP's global offset and is not used
for ranking. The probe runs in CLIP's own embedding space; it explains how the
image population moved, independently of the monitored model's embeddings.

The concept vocabulary is hand-curated for photo-stream drift (photographic
style/era, scene, lighting, weather, composition, people) plus the CLEAR-10
class names. It is English-only and reflects the authors' choices; see the
paper's Limitations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CONCEPT_GROUPS: Dict[str, List[str]] = {
    "photographic_style": [
        "a film photograph",
        "a digital photograph",
        "a black and white photo",
        "a sepia photo",
        "a grainy photo",
        "an old photograph",
        "a modern photograph",
        "a faded photo",
        "a low resolution photo",
        "a high resolution photo",
    ],
    "scene": [
        "an indoor scene",
        "an outdoor scene",
        "a street scene",
        "a natural landscape",
        "inside a building",
        "a stadium",
        "a sports field",
        "a domestic interior",
    ],
    "lighting": [
        "bright daylight",
        "low light",
        "a night photo",
        "a backlit photo",
        "an overexposed photo",
        "an underexposed photo",
        "flash photography",
    ],
    "weather": [
        "sunny weather",
        "overcast weather",
        "snowy weather",
        "foggy weather",
        "rainy weather",
    ],
    "composition": [
        "a close-up shot",
        "a wide shot",
        "an aerial view",
        "a blurry photo",
        "motion blur",
        "a cluttered background",
        "a plain background",
        "a centered subject",
    ],
    "people_and_marks": [
        "a crowd of people",
        "a single person",
        "no people",
        "text in the image",
        "a sign",
        "a logo",
        "a watermark",
    ],
}

# Photographic-drift vocabulary: the basis used to name a drift direction.
CONCEPT_VOCABULARY: List[str] = [phrase for group in CONCEPT_GROUPS.values() for phrase in group]

# CLEAR-10 object concepts, kept separate. A shift here is a change in the class
# mix (prior drift), not a photographic-style change, so these are reported on
# their own and never mixed into the drift-direction ranking.
CLASS_VOCABULARY: List[str] = [
    "a laptop",
    "a camera",
    "a sweater",
    "a soccer game",
    "an ice hockey game",
    "a bus",
    "a dress",
    "a car race",
    "a person in cosplay",
    "a baseball game",
]

# Concepts that map onto a CLEAR-10 ``AUTO_TAG_SCORES`` key, for weak-label
# validation. Several concepts intentionally share a tag.
CONCEPT_TO_AUTOTAG: Dict[str, str] = {
    "a black and white photo": "blackandwhite",
    "a sepia photo": "sepia",
    "a night photo": "night",
    "low light": "night",
    "an outdoor scene": "outdoor",
    "an indoor scene": "indoor",
    "a street scene": "city",
    "inside a building": "indoor",
    "a natural landscape": "landscape",
    "a stadium": "stadium",
    "a crowd of people": "people",
    "a single person": "portrait",
    "no people": "nobody",
    "text in the image": "text",
    "a sign": "sign",
    "sunny weather": "sunny",
    "snowy weather": "snow",
    "rainy weather": "rain",
    "foggy weather": "fog",
    "a blurry photo": "blur",
    "motion blur": "blur",
    "an aerial view": "aerial",
}


def build_concept_matrix(
    extractor: Any, vocabulary: Optional[Sequence[str]] = None
) -> tuple[List[str], np.ndarray]:
    """Encode the concept vocabulary with a CLIP text encoder.

    ``extractor`` must expose ``encode_text`` (see
    ``drift_autopsy.extractors.clip.CLIPEmbeddingExtractor``).
    """
    if not hasattr(extractor, "encode_text"):
        raise TypeError("extractor must provide encode_text(); use the 'clip' extractor")
    names = list(vocabulary) if vocabulary is not None else list(CONCEPT_VOCABULARY)
    matrix = np.asarray(extractor.encode_text(names), dtype=float)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return names, matrix


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def concept_distribution(
    image_embeddings: np.ndarray, concept_matrix: np.ndarray, temperature: float = 0.01
) -> np.ndarray:
    """Per-image softmax over concepts (CLIP zero-shot), shape (n_images, n_concepts)."""
    logits = _l2(np.asarray(image_embeddings, dtype=float)) @ concept_matrix.T / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def rank_concepts(
    ref_embeddings: np.ndarray,
    prod_embeddings: np.ndarray,
    concept_matrix: np.ndarray,
    concept_names: Sequence[str],
    temperature: float = 0.01,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rank concepts by the reference -> production shift in mean concept mass."""
    ref_embeddings = np.asarray(ref_embeddings, dtype=float)
    prod_embeddings = np.asarray(prod_embeddings, dtype=float)

    ref_mass = concept_distribution(ref_embeddings, concept_matrix, temperature).mean(axis=0)
    prod_mass = concept_distribution(prod_embeddings, concept_matrix, temperature).mean(axis=0)
    delta = prod_mass - ref_mass

    drift_direction = _l2(prod_embeddings.mean(axis=0) - ref_embeddings.mean(axis=0))
    alignment = concept_matrix @ drift_direction

    rows = [
        {
            "concept": str(concept_names[i]),
            "delta": float(delta[i]),
            "ref_mass": float(ref_mass[i]),
            "prod_mass": float(prod_mass[i]),
            "alignment": float(alignment[i]),
        }
        for i in range(len(concept_names))
    ]
    rows.sort(key=lambda r: r["delta"], reverse=True)
    if top_k is not None:
        head = rows[:top_k]
        tail = rows[-top_k:]
        return head + [r for r in tail if r not in head]
    return rows


def probe_window(
    ref_frame: pd.DataFrame,
    prod_frame: pd.DataFrame,
    ref_embeddings: np.ndarray,
    prod_embeddings: np.ndarray,
    concept_matrix: np.ndarray,
    concept_names: Sequence[str],
    class_column: str = "class_name",
    drifted_classes: Optional[Sequence[str]] = None,
    temperature: float = 0.01,
    top_k: int = 10,
    min_class_samples: int = 40,
) -> Dict[str, Any]:
    """Concept ranking for a window overall and per drifted class.

    ``ref_embeddings`` / ``prod_embeddings`` must be row-aligned with
    ``ref_frame`` / ``prod_frame``.
    """
    result: Dict[str, Any] = {
        "overall": rank_concepts(
            ref_embeddings, prod_embeddings, concept_matrix, concept_names, temperature, top_k
        ),
        "per_class": {},
    }

    if class_column not in ref_frame.columns or class_column not in prod_frame.columns:
        return result

    classes = (
        list(drifted_classes)
        if drifted_classes is not None
        else sorted(set(prod_frame[class_column].dropna().unique()))
    )
    ref_labels = ref_frame[class_column].to_numpy()
    prod_labels = prod_frame[class_column].to_numpy()

    for cls in classes:
        ref_mask = ref_labels == cls
        prod_mask = prod_labels == cls
        if ref_mask.sum() < min_class_samples or prod_mask.sum() < min_class_samples:
            continue
        result["per_class"][str(cls)] = {
            "n_ref": int(ref_mask.sum()),
            "n_prod": int(prod_mask.sum()),
            "concepts": rank_concepts(
                ref_embeddings[ref_mask],
                prod_embeddings[prod_mask],
                concept_matrix,
                concept_names,
                temperature,
                top_k,
            ),
        }
    return result


def concept_exemplars(
    ref_frame: pd.DataFrame,
    prod_frame: pd.DataFrame,
    ref_embeddings: np.ndarray,
    prod_embeddings: np.ndarray,
    concept_matrix: np.ndarray,
    concept_names: Sequence[str],
    concept: str,
    k: int = 5,
    temperature: float = 0.01,
    path_column: str = "image_path",
) -> Dict[str, Any]:
    """The k images with the highest probability for ``concept``, in each population.

    The same rule is applied to the reference and to production, so the figure
    shows how the most concept-like images differ, not a contrived contrast.
    Also returns the share of images whose top concept is ``concept``.
    """
    ci = list(concept_names).index(concept)
    out: Dict[str, Any] = {"concept": concept}
    for name, frame, emb in (("reference", ref_frame, ref_embeddings), ("production", prod_frame, prod_embeddings)):
        probs = concept_distribution(emb, concept_matrix, temperature)
        order = np.argsort(probs[:, ci])[::-1][:k]
        out[name] = {
            "images": frame.iloc[order][path_column].astype(str).tolist(),
            "probabilities": probs[order, ci].tolist(),
            "share_top_concept": float(np.mean(probs.argmax(axis=1) == ci)),
        }
    return out


def concept_shift_intervals(
    ref_embeddings: np.ndarray,
    prod_embeddings: np.ndarray,
    concept_matrix: np.ndarray,
    concept_names: Sequence[str],
    temperature: float = 0.01,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Bootstrap interval for each concept's mass shift (Equation of the paper).

    Images are resampled within the reference and within production. The
    interval level is Bonferroni-adjusted over the concepts, so a concept whose
    interval excludes zero has shifted at joint level ``alpha``.
    """
    ref_p = concept_distribution(ref_embeddings, concept_matrix, temperature)
    prod_p = concept_distribution(prod_embeddings, concept_matrix, temperature)
    delta = prod_p.mean(axis=0) - ref_p.mean(axis=0)
    rng = np.random.default_rng(random_state)
    boot = np.empty((n_bootstrap, len(concept_names)))
    for k in range(n_bootstrap):
        boot[k] = (prod_p[rng.integers(0, len(prod_p), len(prod_p))].mean(axis=0)
                   - ref_p[rng.integers(0, len(ref_p), len(ref_p))].mean(axis=0))
    tail = 100.0 * alpha / (2 * len(concept_names))
    lo, hi = np.percentile(boot, [tail, 100.0 - tail], axis=0)
    return {
        str(name): {"delta": float(delta[i]), "ci_low": float(lo[i]), "ci_high": float(hi[i]),
                    "significant": bool(lo[i] > 0 or hi[i] < 0)}
        for i, name in enumerate(concept_names)
    }


def validate_against_metadata(
    ranking: Sequence[Dict[str, Any]],
    autotag_shift: Dict[str, float],
    concept_to_tag: Optional[Dict[str, str]] = None,
    n_permutations: int = 10000,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Spearman correlation between concept mass shift and auto-tag shift, one pair per tag.

    Several phrases map to the same tag (for example "a night photo" and "low
    light" both map to "night"). Their deltas are averaged so each tag counts
    once and the pairs are independent units. Significance is a permutation test
    of the pairing (two-sided), which is exact enough for the small number of
    tags involved.
    """
    from scipy.stats import spearmanr

    mapping = concept_to_tag if concept_to_tag is not None else CONCEPT_TO_AUTOTAG
    by_concept = {row["concept"]: row["delta"] for row in ranking}
    per_tag: Dict[str, List[float]] = {}
    for concept, tag in mapping.items():
        if concept in by_concept and tag in autotag_shift:
            per_tag.setdefault(tag, []).append(float(by_concept[concept]))
    matched = [
        {"tag": tag, "concepts": [c for c, t in mapping.items() if t == tag and c in by_concept],
         "concept_delta": float(np.mean(vals)), "tag_shift": float(autotag_shift[tag])}
        for tag, vals in sorted(per_tag.items())
    ]
    if len(matched) < 4:
        return {"spearman_rho": None, "p_value": None, "n_tags": len(matched), "matched": matched}

    x = np.array([m["concept_delta"] for m in matched])
    y = np.array([m["tag_shift"] for m in matched])
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(random_state)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n_permutations)])
    p = float((1 + np.sum(np.abs(null) >= abs(rho))) / (1 + n_permutations))
    return {"spearman_rho": rho, "p_value": p, "n_tags": len(matched), "matched": matched,
            "n_permutations": n_permutations}
