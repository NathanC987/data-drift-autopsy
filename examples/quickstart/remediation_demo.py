"""Localisation-driven remediation and remediation triage.

Runs the remediation strategies over three settings and shows that the
pipeline's own signals predict, in advance, whether retraining will help:

  * CLEAR-10 image stream   -> retrain-on-recent recovers most of the gap cheaply
  * real ACS shifts         -> nothing recovers the gap (largely irreducible)
  * one synthetic ACS shift -> feature-drop + importance weighting recover it

Run:  OMP_NUM_THREADS=1 python examples/quickstart/remediation_demo.py --shift both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from drift_autopsy.data import FolktablesLoader
from drift_autopsy.remediation import (
    RemediationContext,
    default_model_factory,
    inject_covariate_shift,
    parse_drifted_features,
    remediation_triage,
    results_to_frame,
    results_to_latex,
    run_remediation_suite,
)
from drift_autopsy.utils import setup_logging

FEATS = FolktablesLoader.ACS_INCOME_FEATURES
OUT = Path("outputs")
FIG_DIR = Path("paper/figures")
TAB_DIR = Path("outputs/clear10_tabularized_demo")
EMB_COLS = [f"feature_{i}" for i in range(512)]
SYNTHETIC_FEATURES = ["AGEP", "WKHP"]

ACS_STRATEGIES = [
    "full_retrain", "feature_drop_retrain", "importance_weighted_retrain",
    "head_refit", "calibration_only",
]
ACS_TEMPORAL_STRATEGIES = ACS_STRATEGIES + ["retrain_on_recent[previous]"]
CLEAR_STRATEGIES = [
    "full_retrain", "importance_weighted_retrain",
    "retrain_on_recent[previous]", "retrain_on_recent[all_prior]", "head_refit",
]


def _acs(year: int, state: str = "CA") -> tuple[np.ndarray, np.ndarray]:
    df = FolktablesLoader.load_acs_income_cached(year, state)
    return df[FEATS].to_numpy(float), df["target"].to_numpy(int)


def _cbpe_gap(base_model, ref_X, ref_y, prod_X) -> float:
    """Signed label-free accuracy-drop estimate (reference minus production)."""
    edges = np.linspace(0, 1, 11)
    rc = base_model.predict_proba(ref_X).max(1)
    correct = (base_model.predict(ref_X) == ref_y).astype(float)
    rb = np.clip(np.digitize(rc, edges[1:-1]), 0, 9)
    bin_acc = np.array([correct[rb == b].mean() if (rb == b).any() else correct.mean() for b in range(10)])
    pc = base_model.predict_proba(prod_X).max(1)
    pb = np.clip(np.digitize(pc, edges[1:-1]), 0, 9)
    ref_acc = base_model.score(ref_X, ref_y)
    return float(ref_acc - bin_acc[pb].mean())


# --------------------------------------------------------------------------- #
# context builders

def ctx_acs_temporal(drift_json: dict) -> RemediationContext:
    X14, y14 = _acs(2014)
    Xtr, Xh, ytr, yh = train_test_split(X14, y14, test_size=0.3, random_state=42, stratify=y14)
    base = default_model_factory().fit(Xtr, ytr)
    p17_X, p17_y = _acs(2017)
    h18_X, h18_y = _acs(2018)
    return RemediationContext(
        reference_X=Xtr, reference_y=ytr,
        production_X=p17_X, production_y=p17_y,
        holdout_X=h18_X, holdout_y=h18_y,
        feature_names=FEATS,
        drifted_features=parse_drifted_features(drift_json, "temporal"),
        shift_name="ACS temporal (CA 2014->2018)",
        base_model=base,
        recent_windows=[_acs(2015), _acs(2016)],
        reference_accuracy=float(base.score(Xh, yh)),
    )


def ctx_acs_geographic(drift_json: dict, target: str = "WA") -> RemediationContext:
    X14, y14 = _acs(2014)
    Xtr, Xh, ytr, yh = train_test_split(X14, y14, test_size=0.3, random_state=42, stratify=y14)
    base = default_model_factory().fit(Xtr, ytr)
    Xt, yt = _acs(2014, target)
    Xu, Xe, yu, ye = train_test_split(Xt, yt, test_size=0.5, random_state=42, stratify=yt)
    return RemediationContext(
        reference_X=Xtr, reference_y=ytr,
        production_X=Xu, production_y=yu,
        holdout_X=Xe, holdout_y=ye,
        feature_names=FEATS,
        drifted_features=parse_drifted_features(drift_json, "geographic", target) or ["POBP", "RAC1P"],
        shift_name=f"ACS geographic (CA->{target}, 2014)",
        base_model=base,
        recent_windows=[],
        reference_accuracy=float(base.score(Xh, yh)),
    )


def ctx_acs_synthetic() -> RemediationContext:
    X14, y14 = _acs(2014)
    Xtr, Xrest, ytr, yrest = train_test_split(X14, y14, test_size=0.4, random_state=42, stratify=y14)
    Xu, Xe, yu, ye = train_test_split(Xrest, yrest, test_size=0.5, random_state=42, stratify=yrest)
    base = default_model_factory().fit(Xtr, ytr)
    ref_acc = float(base.score(Xe, ye))  # accuracy on the un-shifted holdout == ceiling
    Xu_s = inject_covariate_shift(Xu, FEATS, SYNTHETIC_FEATURES)
    Xe_s = inject_covariate_shift(Xe, FEATS, SYNTHETIC_FEATURES)
    return RemediationContext(
        reference_X=Xtr, reference_y=ytr,
        production_X=Xu_s, production_y=yu,
        holdout_X=Xe_s, holdout_y=ye,
        feature_names=FEATS,
        drifted_features=list(SYNTHETIC_FEATURES),
        shift_name=f"ACS synthetic covariate shift ({'+'.join(SYNTHETIC_FEATURES)})",
        base_model=base,
        recent_windows=[],
        reference_accuracy=ref_acc,
    )


def _clear_bucket(b: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(TAB_DIR / f"bucket_{b}.parquet")
    return df[EMB_COLS].to_numpy(float), df["y_true"].to_numpy(int)


def ctx_clear10(bucket: int = 10) -> RemediationContext:
    factory = lambda: LogisticRegression(max_iter=1000, random_state=42)  # matches image pipeline
    X1, y1 = _clear_bucket(1)
    Xtr, Xh, ytr, yh = train_test_split(X1, y1, train_size=0.7, random_state=42, stratify=y1)
    base = factory().fit(Xtr, ytr)
    Xb, yb = _clear_bucket(bucket)
    Xu, Xe, yu, ye = train_test_split(Xb, yb, test_size=0.5, random_state=42, stratify=yb)
    return RemediationContext(
        reference_X=Xtr, reference_y=ytr,
        production_X=Xu, production_y=yu,
        holdout_X=Xe, holdout_y=ye,
        feature_names=EMB_COLS, drifted_features=[],
        shift_name=f"CLEAR-10 bucket {bucket}",
        base_model=base,
        recent_windows=[_clear_bucket(b) for b in range(2, bucket)],
        model_factory=factory,
        reference_accuracy=float(base.score(Xh, yh)),
    )


# --------------------------------------------------------------------------- #

def run_setting(ctx: RemediationContext, strategies: List[str]) -> dict:
    gap_est = _cbpe_gap(ctx.base_model, ctx.reference_X, ctx.reference_y, ctx.production_X)
    triage = remediation_triage(ctx, estimated_gap=gap_est)
    results = run_remediation_suite(ctx, strategies)

    print(f"\n=== {ctx.shift_name} ===")
    print(f"  reference acc {ctx.reference_accuracy:.4f}  ->  before remediation {ctx.baseline_before:.4f}"
          f"  (gap {ctx.reference_accuracy - ctx.baseline_before:+.4f})")
    print(f"  triage: will_retraining_help={triage['will_retraining_help']}  --  {triage['rationale']}")
    for r in results:
        print(f"    {r.strategy:30} after={r.accuracy_after:.4f}  recovered={r.accuracy_recovered:+.4f}"
              f"  gap_frac={r.fraction_of_gap_recovered:+.2f}  {r.wall_clock_seconds:.3f}s"
              f"  labels={r.n_production_labels_required}")
    return {
        "shift_name": ctx.shift_name,
        "reference_accuracy": ctx.reference_accuracy,
        "baseline_before": ctx.baseline_before,
        "triage": triage,
        "results": [r.to_dict() for r in results],
    }


STRATEGY_MARKER = {
    "full_retrain": ("full retrain", "o"),
    "feature_drop_retrain": ("feature-drop", "v"),
    "importance_weighted_retrain": ("importance weighting", "s"),
    "retrain_on_recent[previous]": ("retrain on previous window", "D"),
    "retrain_on_recent[all_prior]": ("retrain on all prior", "P"),
    "head_refit": ("head refit (labels)", "^"),
    "calibration_only": ("recalibrate (labels)", "X"),
}
SETTING_COLOR = {
    "bucket 10": "#1f77b4",
    "bucket 6": "#17becf",
    "synthetic covariate shift": "#2ca02c",
    "geographic": "#d62728",
    "temporal": "#9467bd",
}
PARETO_STRATEGIES = {
    "full_retrain", "importance_weighted_retrain",
    "retrain_on_recent[previous]", "retrain_on_recent[all_prior]", "head_refit",
}


def _setting_key(name: str) -> str:
    n = name.lower()
    for k in SETTING_COLOR:
        if k in n:
            return k
    return name


def render_pareto(payloads: List[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    seen_strats, seen_settings = set(), set()
    for p in payloads:
        colour = SETTING_COLOR.get(_setting_key(p["shift_name"]), "0.4")
        label_setting = (
            p["shift_name"].replace("CLEAR-10 ", "CLEAR-10 ").split(" (")[0]
        )
        for r in p["results"]:
            if r["strategy"] not in PARETO_STRATEGIES:
                continue
            flops, frac = r["fit_flops_proxy"], r["fraction_of_gap_recovered"]
            if not np.isfinite(flops) or flops <= 0 or not np.isfinite(frac):
                continue
            frac = min(max(frac, -0.12), 1.12)  # clip for readability
            _, marker = STRATEGY_MARKER.get(r["strategy"], (r["strategy"], "o"))
            ax.scatter(flops, frac, marker=marker, s=85, color=colour, edgecolor="0.2", linewidth=0.5,
                       label=label_setting if label_setting not in seen_settings else None)
            seen_settings.add(label_setting)
            seen_strats.add(r["strategy"])

    ax.set_xscale("log")
    ax.set_ylim(-0.15, 1.18)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axhline(1.0, color="0.8", lw=0.8, ls=":")
    ax.set_xlabel("fit cost (FLOP proxy, log scale)")
    ax.set_ylabel("fraction of accuracy gap recovered")
    ax.set_title("Remediation cost vs recovery (feature-drop and label-only omitted)")

    from matplotlib.lines import Line2D

    strat_handles = [
        Line2D([0], [0], marker=STRATEGY_MARKER[s][1], color="0.3", ls="", label=STRATEGY_MARKER[s][0])
        for s in STRATEGY_MARKER if s in seen_strats
    ]
    leg1 = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.72), fontsize=7, title="setting")
    ax.add_artist(leg1)
    ax.legend(handles=strat_handles, loc="center left", bbox_to_anchor=(1.01, 0.25),
              fontsize=7, title="strategy")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_summary_table(payloads: List[dict], out_path: Path) -> None:
    """One row per setting: gap, best label-free recovery, full retrain, triage verdict."""
    cheap = {"feature_drop_retrain", "importance_weighted_retrain", "retrain_on_recent[previous]"}
    rows = []
    for p in payloads:
        gap = p["reference_accuracy"] - p["baseline_before"]
        cands = [r for r in p["results"]
                 if r["strategy"] in cheap and r["n_production_labels_required"] == 0]
        best = max(cands, key=lambda r: r["fraction_of_gap_recovered"], default=None)
        full = next((r for r in p["results"] if r["strategy"] == "full_retrain"), None)
        rows.append({
            "setting": p["shift_name"].split(" (")[0].replace("_", " "),
            "gap (pp)": f"{100 * gap:.1f}",
            "cheapest fix": best["strategy"].replace("_retrain", "").replace("[previous]", "").replace("_", " ") if best else "--",
            "recovered": f"{best['fraction_of_gap_recovered']:.0%}" if best else "--",
            "cost (s)": f"{best['wall_clock_seconds']:.2f}" if best else "--",
            "full retrain": f"{full['fraction_of_gap_recovered']:.0%}" if full else "--",
            "triage": "escalate" if p["triage"]["will_retraining_help"] else "do not retrain",
        })
    df = pd.DataFrame(rows)
    body = " \\\\\n".join(" & ".join(str(v).replace("%", "\\%").replace("[", "(").replace("]", ")")
                                     for v in r) for r in df.itertuples(index=False))
    head = " & ".join(c.replace("_", " ") for c in df.columns)
    out_path.write_text(
        "\\begin{table}[!htbp]\n\\centering\n"
        "\\caption{Remediation triage: the cheap label-free probe's recovery predicts the verdict, "
        "which matches whether any strategy recovers the gap.}\n\\label{tab:triage}\n"
        "\\renewcommand{\\arraystretch}{1.2}\\setlength{\\tabcolsep}{3pt}\\scriptsize\n"
        f"\\begin{{tabular}}{{@{{}}l r l r r r l@{{}}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
        f"{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", choices=["temporal", "geographic", "synthetic", "clear10", "both", "all"],
                    default="all")
    ap.add_argument("--geo-target", default="WA")
    ap.add_argument("--drop-features", default="", help="comma-separated override for feature_drop_retrain")
    ap.add_argument("--rebuild-only", action="store_true",
                    help="regenerate tables/figure from outputs/remediation_results.json")
    args = ap.parse_args()
    setup_logging(level="WARNING")

    if args.rebuild_only:
        payloads = json.loads((OUT / "remediation_results.json").read_text())["settings"]
        frame = pd.DataFrame([r for p in payloads for r in p["results"]])
        (OUT / "remediation_table_full.tex").write_text(
            results_to_latex(frame, caption="Localisation-driven remediation, all strategies.",
                             label="tab:remediation-full"))
        render_summary_table(payloads, OUT / "remediation_table.tex")
        render_pareto(payloads, FIG_DIR / "remediation_pareto.png")
        print("rebuilt tables + figure from existing JSON")
        return

    drift_json = json.loads(Path("outputs/folktables_drift_results.json").read_text())
    want = {"temporal", "geographic", "synthetic", "clear10"} if args.shift in ("all", "both") else {args.shift}

    payloads: List[dict] = []
    if "temporal" in want:
        payloads.append(run_setting(ctx_acs_temporal(drift_json), ACS_TEMPORAL_STRATEGIES))
    if "geographic" in want:
        payloads.append(run_setting(ctx_acs_geographic(drift_json, args.geo_target), ACS_STRATEGIES))
    if "synthetic" in want:
        payloads.append(run_setting(ctx_acs_synthetic(), ACS_STRATEGIES))
    if "clear10" in want:
        for b in (6, 10):
            payloads.append(run_setting(ctx_clear10(b), CLEAR_STRATEGIES))

    OUT.mkdir(exist_ok=True)
    frame = pd.DataFrame([r for p in payloads for r in p["results"]])

    rich = {
        "settings": payloads,
        # back-compat block for the dashboard's existing reader
        "remediation_type": "localisation_driven_suite",
        "original_training_years": [2014],
        "retrained_training_years": [2014, 2015, 2016],
        "test_years": [2017, 2018],
        "comparison": _backcompat_comparison(payloads),
    }
    (OUT / "remediation_results.json").write_text(json.dumps(rich, indent=2, default=float))
    (OUT / "remediation_table_full.tex").write_text(
        results_to_latex(frame, caption="Localisation-driven remediation, all strategies.",
                         label="tab:remediation-full")
    )
    render_summary_table(payloads, OUT / "remediation_table.tex")
    render_pareto(payloads, FIG_DIR / "remediation_pareto.png")
    print(f"\nwrote outputs/remediation_results.json, outputs/remediation_table.tex, "
          f"outputs/remediation_table_full.tex, {FIG_DIR}/remediation_pareto.png")


def _backcompat_comparison(payloads: List[dict]) -> Dict[str, dict]:
    for p in payloads:
        if "temporal" in p["shift_name"]:
            full = next((r for r in p["results"] if r["strategy"] == "full_retrain"), None)
            if full:
                return {
                    "2018": {
                        "original_accuracy": round(full["accuracy_before"], 4),
                        "retrained_accuracy": round(full["accuracy_after"], 4),
                        "improvement": round(full["accuracy_recovered"], 4),
                        "improved": full["accuracy_recovered"] > 0,
                    }
                }
    return {}


if __name__ == "__main__":
    main()
