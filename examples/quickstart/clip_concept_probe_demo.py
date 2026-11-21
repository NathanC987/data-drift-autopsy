"""Concept-level visual root cause analysis on CLEAR-10.

For every bucket 2 to 10, the CLIP concept probe (drift_autopsy.rca.concept_probe)
ranks a fixed vocabulary of photographic phrases by their concept mass shift
against bucket 1, the same fixed reference the drift detectors use. Each shift
gets a Bonferroni-adjusted bootstrap interval, and the ranking is validated
against the shift in the dataset's automatic image tags (AUTO_TAG_SCORES) with
a permutation test.

Data-driven choices (no manual picking):
* focus class per bucket = the predicted class with the most drifting
  embedding dimensions in outputs/clear10_drift_results.json;
* figure bucket = the bucket with the largest Frechet distance to the reference.

Output: outputs/clip_concepts/concept_rca_results.json (figures are rendered by
examples/paper/build_artifacts.py). Needs `pip install -e '.[image,concept]'`,
the embeddings from clear10_tabularize_demo.py and the drift study output.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import drift_autopsy.extractors  # noqa: F401  (registers the 'clip' extractor)
from drift_autopsy.experiments.clear10 import (
    ANALYSIS_BUCKETS,
    REFERENCE_BUCKET,
    class_names,
    embeddings,
    load_bucket,
    make_clear_model,
    split_reference,
)
from drift_autopsy.rca import concept_probe as cp
from drift_autopsy.registry import ExtractorRegistry
from drift_autopsy.utils import setup_logging

SEED = 42
ALPHA = 0.05
META_DIR = Path("data/clear10/labeled_metadata")
OUT_DIR = Path("outputs/clip_concepts")
DRIFT_RESULTS = Path("outputs/clear10_drift_results.json")


def clip_embeddings(bucket: int, frame: pd.DataFrame, extractor) -> np.ndarray:
    """CLIP image embeddings for a bucket, cached to .npy and row-aligned with ``frame``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"clip_emb_bucket_{bucket}.npy"
    ids_cache = OUT_DIR / f"clip_ids_bucket_{bucket}.json"
    ids = frame["sample_id"].astype(str).tolist()
    if cache.exists() and ids_cache.exists() and json.loads(ids_cache.read_text()) == ids:
        return np.load(cache)
    emb = extractor.extract(frame["image_path"].astype(str).tolist())
    np.save(cache, emb)
    ids_cache.write_text(json.dumps(ids))
    return emb


def image_metadata(bucket: int) -> Dict[str, dict]:
    """sample_id -> {tags: AUTO_TAG_SCORES, year: capture year} for one bucket."""
    out = {}
    for path in (META_DIR / str(bucket)).glob("*.json"):
        for sid, entry in json.loads(path.read_text()).items():
            dt = str(entry.get("DATE_TAKEN") or "")
            out[str(sid)] = {"tags": entry.get("AUTO_TAG_SCORES") or {},
                             "year": int(dt[:4]) if dt[:4].isdigit() else None}
    return out


def tag_means(ids: List[str], meta: Dict[str, dict]) -> Dict[str, float]:
    """Mean tag score over images; an image without a tag scores 0 for it."""
    acc: Dict[str, float] = defaultdict(float)
    for sid in ids:
        for tag, score in meta[sid]["tags"].items():
            acc[tag] += float(score)
    n = len(ids)
    return {tag: v / n for tag, v in acc.items()} if n else {}


def tag_shift(ref_ids, ref_meta, prod_ids, prod_meta) -> Dict[str, float]:
    ref, prod = tag_means(ref_ids, ref_meta), tag_means(prod_ids, prod_meta)
    return {t: prod.get(t, 0.0) - ref.get(t, 0.0) for t in set(ref) | set(prod)}


def median_year(ids: List[str], meta: Dict[str, dict]) -> float:
    years = [meta[s]["year"] for s in ids if meta[s]["year"] is not None]
    return float(np.median(years)) if years else float("nan")


def focus_class(drift: dict, bucket: int) -> str:
    slices = drift["windows"][str(bucket)]["predicted_class_slices"]
    return max(slices, key=lambda c: slices[c]["n_drifted_dimensions"])


def main() -> None:
    setup_logging(level="WARNING")
    drift = json.loads(DRIFT_RESULTS.read_text())
    figure_bucket = max(ANALYSIS_BUCKETS, key=lambda b: drift["windows"][str(b)]["detection"]["Frechet distance"]["score"])

    ref_frame = load_bucket(REFERENCE_BUCKET)
    names = class_names(ref_frame)
    split = split_reference(ref_frame, seed=SEED)
    with threadpool_limits(limits=1):
        model = make_clear_model(random_state=SEED).fit(embeddings(split.train), split.train["y_true"])

    extractor = ExtractorRegistry.create("clip", batch_size=256)
    concept_names, concept_matrix = cp.build_concept_matrix(extractor)
    ref_emb = clip_embeddings(REFERENCE_BUCKET, ref_frame, extractor)
    ref_meta = image_metadata(REFERENCE_BUCKET)
    ref_ids = ref_frame["sample_id"].astype(str).tolist()
    ref_pred = np.array([names[int(c)] for c in model.predict(embeddings(ref_frame))])

    results: Dict[str, dict] = {}
    for bucket in ANALYSIS_BUCKETS:
        prod_frame = load_bucket(bucket)
        prod_emb = clip_embeddings(bucket, prod_frame, extractor)
        prod_meta = image_metadata(bucket)
        prod_ids = prod_frame["sample_id"].astype(str).tolist()
        prod_pred = np.array([names[int(c)] for c in model.predict(embeddings(prod_frame))])

        overall = cp.rank_concepts(ref_emb, prod_emb, concept_matrix, concept_names)
        intervals = cp.concept_shift_intervals(ref_emb, prod_emb, concept_matrix, concept_names,
                                               alpha=ALPHA, random_state=SEED)
        validation = cp.validate_against_metadata(overall, tag_shift(ref_ids, ref_meta, prod_ids, prod_meta),
                                                  random_state=SEED)

        cls = focus_class(drift, bucket)
        rm, pm = ref_pred == cls, prod_pred == cls
        cls_rank = cp.rank_concepts(ref_emb[rm], prod_emb[pm], concept_matrix, concept_names)
        cls_validation = cp.validate_against_metadata(
            cls_rank, tag_shift([i for i, k in zip(ref_ids, rm) if k], ref_meta,
                                [i for i, k in zip(prod_ids, pm) if k], prod_meta), random_state=SEED)
        cls_intervals = cp.concept_shift_intervals(ref_emb[rm], prod_emb[pm], concept_matrix, concept_names,
                                                   alpha=ALPHA, random_state=SEED)

        top_sig = next((r["concept"] for r in overall if intervals[r["concept"]]["significant"]
                        and intervals[r["concept"]]["delta"] > 0), overall[0]["concept"])
        entry = {
            "overall": overall,
            "intervals": intervals,
            "n_significant_concepts": int(sum(v["significant"] for v in intervals.values())),
            "metadata_validation": validation,
            "focus_class": {
                "class": cls,
                "selection_rule": "predicted class with the most drifting embedding dimensions",
                "n_reference": int(rm.sum()), "n_production": int(pm.sum()),
                "concepts": cls_rank,
                "intervals": cls_intervals,
                "metadata_validation": cls_validation,
            },
            "capture_year": {
                "reference_median": median_year(ref_ids, ref_meta),
                "production_median": median_year(prod_ids, prod_meta),
                "focus_reference_median": median_year([i for i, k in zip(ref_ids, rm) if k], ref_meta),
                "focus_production_median": median_year([i for i, k in zip(prod_ids, pm) if k], prod_meta),
            },
            "top_significant_gaining_concept": top_sig,
        }
        if bucket == figure_bucket:
            entry["exemplars"] = cp.concept_exemplars(ref_frame[rm].reset_index(drop=True),
                                                      prod_frame[pm].reset_index(drop=True),
                                                      ref_emb[rm], prod_emb[pm], concept_matrix, concept_names,
                                                      cls_rank[0]["concept"], k=5)
        results[str(bucket)] = entry
        v = validation
        print(f"bucket {bucket}: top {[(r['concept'], round(r['delta'], 4)) for r in overall[:4]]}; "
              f"{entry['n_significant_concepts']} significant; tags rho={v['spearman_rho']} p={v['p_value']} "
              f"(n={v['n_tags']}); focus {cls}: rho={cls_validation['spearman_rho']} p={cls_validation['p_value']}")

    payload = {
        "config": {"seed": SEED, "alpha": ALPHA, "reference_bucket": REFERENCE_BUCKET,
                   "clip": extractor.get_extraction_metadata(), "vocabulary": concept_names,
                   "figure_bucket": figure_bucket,
                   "figure_bucket_rule": "bucket with the largest Frechet distance to the reference"},
        "buckets": results,
    }
    (OUT_DIR / "concept_rca_results.json").write_text(json.dumps(payload, indent=1, allow_nan=False))
    print(f"wrote {OUT_DIR / 'concept_rca_results.json'}")


if __name__ == "__main__":
    main()
