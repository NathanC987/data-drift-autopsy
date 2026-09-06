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
for ranking.

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


def select_dim_exemplars(
    ref_frame: pd.DataFrame,
    prod_frame: pd.DataFrame,
    drift_scores: Dict[str, float],
    top_dims: int = 3,
    k: int = 4,
    path_column: str = "image_path",
) -> List[Dict[str, Any]]:
    """For each top-drifted embedding dimension, the k most extreme ref/prod images.

    ``drift_scores`` maps ``feature_<d>`` -> per-dimension KS drift statistic
    (as produced by ``clear10_report``); higher magnitude => stronger shift.
    """
    ordered = [
        name
        for name, _ in sorted(drift_scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
        if str(name) in ref_frame.columns and str(name) in prod_frame.columns
    ][:top_dims]

    out: List[Dict[str, Any]] = []
    for dim in ordered:
        ref_center = float(ref_frame[dim].mean())
        prod_extreme = prod_frame.reindex(
            (prod_frame[dim] - ref_center).abs().sort_values(ascending=False).index
        )
        ref_extreme = ref_frame.reindex(
            (ref_frame[dim] - ref_center).abs().sort_values(ascending=False).index
        )
        out.append(
            {
                "dimension": str(dim),
                "drift_score": float(drift_scores.get(dim, 0.0)),
                "ref_center": ref_center,
                "prod_mean": float(prod_frame[dim].mean()),
                "reference_images": ref_extreme[path_column].head(k).astype(str).tolist(),
                "production_images": prod_extreme[path_column].head(k).astype(str).tolist(),
            }
        )
    return out


def validate_against_metadata(
    ranking: Sequence[Dict[str, Any]],
    autotag_shift: Dict[str, float],
    concept_to_tag: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Spearman correlation between concept ``delta`` and the matching auto-tag shift.

    ``autotag_shift`` maps an ``AUTO_TAG_SCORES`` key -> (prod mean - ref mean).
    """
    from scipy.stats import spearmanr

    mapping = concept_to_tag if concept_to_tag is not None else CONCEPT_TO_AUTOTAG
    by_concept = {row["concept"]: row["delta"] for row in ranking}

    matched: List[Dict[str, Any]] = []
    for concept, tag in mapping.items():
        if concept in by_concept and tag in autotag_shift:
            matched.append(
                {
                    "concept": concept,
                    "tag": tag,
                    "concept_delta": float(by_concept[concept]),
                    "tag_shift": float(autotag_shift[tag]),
                }
            )

    if len(matched) < 3:
        return {"spearman_rho": None, "p_value": None, "n_matched": len(matched), "matched": matched}

    rho, p = spearmanr(
        [m["concept_delta"] for m in matched], [m["tag_shift"] for m in matched]
    )
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_matched": len(matched),
        "matched": matched,
    }
