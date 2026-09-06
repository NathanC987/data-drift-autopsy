"""Concept-level visual root cause analysis on CLEAR-10.

Projects the reference->production embedding drift for selected CLEAR-10 buckets
onto a natural-language concept basis (CLIP), producing human-readable causes,
and validates the ranking against the dataset's AUTO_TAG_SCORES metadata.

Run:  python examples/quickstart/clip_concept_probe_demo.py
Needs: pip install -e '.[image,concept]' and the cached tabularised buckets in
       outputs/clear10_tabularized_demo/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import drift_autopsy.extractors  # noqa: F401  (registers the 'clip' extractor)
from drift_autopsy.rca import concept_probe as cp
from drift_autopsy.registry import ExtractorRegistry
from drift_autopsy.utils import setup_logging

TAB_DIR = Path("outputs/clear10_tabularized_demo")
META_DIR = Path("data/clear10/labeled_metadata")
OUT_DIR = Path("outputs/clip_concepts")
FIG_DIR = Path("paper/figures")

REFERENCE_BUCKET = 1
ANALYSIS_BUCKETS = [7, 8, 10]
FOCUS_CLASS = "BACKGROUND"  # the class with the strongest photographic drift
FEATURE_COLS = [f"feature_{i}" for i in range(512)]


def load_bucket(bucket: int) -> pd.DataFrame:
    return pd.read_parquet(TAB_DIR / f"bucket_{bucket}.parquet")


def clip_embeddings(bucket: int, frame: pd.DataFrame, extractor) -> np.ndarray:
    """CLIP image embeddings for a bucket, cached to .npy (row-aligned with frame)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / f"clip_emb_bucket_{bucket}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(frame):
            return emb
    emb = extractor.extract(frame["image_path"].astype(str).tolist())
    np.save(cache, emb)
    return emb


def autotag_shift(ref_bucket: int, prod_bucket: int, class_name: str | None = None) -> Dict[str, float]:
    """Mean AUTO_TAG_SCORES per tag, prod minus ref, over shared tags."""

    def tag_means(bucket: int) -> Dict[str, float]:
        classes = [class_name] if class_name else [p.stem for p in (META_DIR / str(bucket)).glob("*.json")]
        acc: Dict[str, List[float]] = defaultdict(list)
        for cls in classes:
            path = META_DIR / str(bucket) / f"{cls}.json"
            if not path.exists():
                continue
            for entry in json.loads(path.read_text()).values():
                for tag, score in (entry.get("AUTO_TAG_SCORES") or {}).items():
                    acc[tag].append(float(score))
        # presence = mean score across all images (missing tag counts as 0)
        n = sum(len(json.loads((META_DIR / str(bucket) / f"{c}.json").read_text()))
                for c in classes if (META_DIR / str(bucket) / f"{c}.json").exists())
        return {tag: sum(v) / n for tag, v in acc.items()} if n else {}

    ref, prod = tag_means(ref_bucket), tag_means(prod_bucket)
    return {tag: prod.get(tag, 0.0) - ref.get(tag, 0.0) for tag in set(ref) | set(prod)}


def drifted_classes(bucket_key: str, drift_json: dict, top: int = 4) -> List[str]:
    slices = drift_json["bucket_results"][bucket_key]["localization"].get("class_slice_summary", [])
    ranked = sorted(
        (s for s in slices if s.get("drift_detected")),
        key=lambda s: s.get("n_drifted_features", 0),
        reverse=True,
    )
    classes = [s["test_slice"] for s in ranked[:top]]
    if FOCUS_CLASS not in classes:
        classes.append(FOCUS_CLASS)
    return classes


def median_year(bucket: int, cls: str) -> float:
    """Median DATE_TAKEN year for a class in a bucket (photographic-era check)."""
    path = META_DIR / str(bucket) / f"{cls}.json"
    years = []
    for entry in json.loads(path.read_text()).values():
        dt = str(entry.get("DATE_TAKEN") or "")
        if len(dt) >= 4 and dt[:4].isdigit():
            years.append(int(dt[:4]))
    return float(np.median(years)) if years else float("nan")


def render_ranking_figure(results: dict, bucket: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = results[bucket]["overall"]
    per_class = results[bucket]["per_class"]
    focus_cls = FOCUS_CLASS if FOCUS_CLASS in per_class else next(iter(per_class), None)

    ncols = 2 if focus_cls else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2), squeeze=False)

    def bar(ax, rows, title):
        rows = [r for r in rows if abs(r["delta"]) > 1e-3]
        rows = sorted(rows, key=lambda r: r["delta"])[-12:]
        y = np.arange(len(rows))
        vals = [r["delta"] for r in rows]
        ax.barh(y, vals, color=["#c0392b" if v < 0 else "#27ae60" for v in vals])
        ax.set_yticks(y)
        ax.set_yticklabels([r["concept"] for r in rows], fontsize=8)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_xlabel("concept mass shift  (production - reference)")
        ax.set_title(title, fontsize=10)

    bar(axes[0][0], overall, f"CLEAR-10 bucket {bucket}: overall")
    if focus_cls:
        bar(axes[0][1], per_class[focus_cls]["concepts"], f"class: {focus_cls}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_exemplar_figure(
    ref_frame: pd.DataFrame,
    prod_frame: pd.DataFrame,
    ref_emb: np.ndarray,
    prod_emb: np.ndarray,
    concept_matrix: np.ndarray,
    concept_names: list,
    top_concept: str,
    out_path: Path,
    cls: str,
    k: int = 5,
) -> None:
    """Reference vs production images for the focus class, production ranked by
    how strongly they express the top gained concept (so the shift is visible)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ci = concept_names.index(top_concept)
    r_mask = (ref_frame["class_name"] == cls).to_numpy()
    p_mask = (prod_frame["class_name"] == cls).to_numpy()
    r_frame, p_frame = ref_frame[r_mask].reset_index(drop=True), prod_frame[p_mask].reset_index(drop=True)
    r_score = ref_emb[r_mask] @ concept_matrix[ci]
    p_score = prod_emb[p_mask] @ concept_matrix[ci]

    r_pick = r_frame.iloc[np.argsort(r_score)[: k]]          # weakest expression in reference
    p_pick = p_frame.iloc[np.argsort(p_score)[::-1][: k]]    # strongest in production

    fig, axes = plt.subplots(2, k, figsize=(2.1 * k, 4.6))
    for j in range(k):
        for row, pick in ((0, r_pick), (1, p_pick)):
            ax = axes[row][j]
            ax.axis("off")
            if j < len(pick):
                ax.imshow(Image.open(pick.iloc[j]["image_path"]).convert("RGB"))
    axes[0][0].set_title("reference (bucket 1)", loc="left", fontsize=9)
    axes[1][0].set_title(f"production, ranked by “{top_concept}”", loc="left", fontsize=9)
    fig.suptitle(f"CLEAR-10 {cls}: visual drift surfaced by concept-level RCA", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    setup_logging(level="INFO")
    drift_json = json.loads(Path("outputs/clear10_drift_results.json").read_text())

    extractor = ExtractorRegistry.create("clip", batch_size=256)
    concept_names, concept_matrix = cp.build_concept_matrix(extractor)
    print(f"concept basis: {len(concept_names)} phrases, CLIP {extractor.name}")

    ref_frame = load_bucket(REFERENCE_BUCKET)
    ref_emb = clip_embeddings(REFERENCE_BUCKET, ref_frame, extractor)

    results: Dict[str, dict] = {}
    for bucket in ANALYSIS_BUCKETS:
        key = str(bucket)
        prod_frame = load_bucket(bucket)
        prod_emb = clip_embeddings(bucket, prod_frame, extractor)
        classes = drifted_classes(key, drift_json)

        window = cp.probe_window(
            ref_frame, prod_frame, ref_emb, prod_emb, concept_matrix, concept_names,
            drifted_classes=classes, top_k=10,
        )

        drift_scores = drift_json["bucket_results"][key]["localization"]["drift_scores"]
        window["dim_exemplars"] = cp.select_dim_exemplars(
            ref_frame, prod_frame, drift_scores, top_dims=3, k=4
        )

        window["metadata_validation"] = cp.validate_against_metadata(
            window["overall"], autotag_shift(REFERENCE_BUCKET, bucket)
        )
        if FOCUS_CLASS in window["per_class"]:
            window["metadata_validation_focus"] = {
                "class": FOCUS_CLASS,
                **cp.validate_against_metadata(
                    window["per_class"][FOCUS_CLASS]["concepts"],
                    autotag_shift(REFERENCE_BUCKET, bucket, FOCUS_CLASS),
                ),
            }
        window["photographic_era"] = {
            "reference_median_year": median_year(REFERENCE_BUCKET, FOCUS_CLASS),
            "production_median_year": median_year(bucket, FOCUS_CLASS),
        }

        results[key] = window

        top = ", ".join(f"{r['concept']} ({r['delta']:+.3f})" for r in window["overall"][:5])
        mv = window["metadata_validation"]
        mvf = window.get("metadata_validation_focus", {})
        print(f"\nbucket {bucket}: overall gaining -> {top}")
        print(f"  overall metadata Spearman rho = {mv['spearman_rho']}  (n_matched={mv['n_matched']})")
        if FOCUS_CLASS in window["per_class"]:
            fc = window["per_class"][FOCUS_CLASS]["concepts"][:5]
            print(f"  {FOCUS_CLASS} gaining -> " + ", ".join(f"{r['concept']}({r['delta']:+.3f})" for r in fc))
            print(f"  {FOCUS_CLASS} metadata Spearman rho = {mvf.get('spearman_rho')}  (n={mvf.get('n_matched')})")
        pe = window["photographic_era"]
        print(f"  {FOCUS_CLASS} median photo year: {pe['reference_median_year']:.0f} -> {pe['production_median_year']:.0f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "concept_rca_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR / 'concept_rca_results.json'}")

    render_ranking_figure(results, "10", FIG_DIR / "clip_concept_ranking.png")
    b10_frame = load_bucket(10)
    b10_emb = clip_embeddings(10, b10_frame, extractor)
    top_concept = results["10"]["per_class"][FOCUS_CLASS]["concepts"][0]["concept"]
    render_exemplar_figure(
        ref_frame, b10_frame, ref_emb, b10_emb, concept_matrix, concept_names,
        top_concept, FIG_DIR / "concept_exemplars.png", FOCUS_CLASS,
    )
    print(f"wrote figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
