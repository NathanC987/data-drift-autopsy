"""Controlled drift-injection benchmark.

Injects four drift types -- covariate, prior, concept, and label noise -- into a
real ACS Income reference distribution at three intensities each, plus a
no-drift control, runs the full pipeline on every window, and grades each stage
against the known ground truth. The point is verification: it shows, with
numbers and figures, that the diagnosis the pipeline produces is the correct
one, and it maps exactly where the label-free stack is blind and a few delayed
labels are needed.

Run:  python examples/quickstart/drift_injection_benchmark.py
      python examples/quickstart/drift_injection_benchmark.py --quick
      python examples/quickstart/drift_injection_benchmark.py --figures-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from drift_autopsy.benchmark import run_benchmark
from drift_autopsy.benchmark.evaluate import load_acs_reference
from drift_autopsy.benchmark.spec import DRIFT_TYPES
from drift_autopsy.utils import setup_logging

OUT = Path("outputs")
RESULT_PATH = OUT / "drift_injection_benchmark.json"
TABLE_PATH = OUT / "drift_injection_benchmark_table.tex"
FIG_DIR = Path("paper/figures")

_PRETTY = {
    "none": "no drift", "covariate": "covariate", "prior": "prior",
    "concept": "concept", "label_noise": "label noise",
}


# --------------------------------------------------------------------------- #
# figures

def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_confusion(result: Dict[str, Any], path: Path) -> None:
    plt = _mpl()
    cm = result["summary"]["type_identification"]["confusion_matrix"]
    labels = [t for t in DRIFT_TYPES]
    M = np.array([[cm[a].get(b, 0) for b in labels] for a in labels], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(labels)), [_PRETTY[l] for l in labels], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), [_PRETTY[l] for l in labels])
    ax.set_xlabel("pipeline verdict")
    ax.set_ylabel("injected drift")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else "black", fontsize=11)
    acc = result["summary"]["type_identification"]["accuracy"]
    ax.set_title(f"Injected vs identified drift type (accuracy {acc:.0%})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="runs")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_estimator_optimism(result: Dict[str, Any], path: Path) -> None:
    plt = _mpl()
    colours = {"none": "0.5", "covariate": "#2ca02c", "prior": "#9467bd",
               "concept": "#d62728", "label_noise": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for t in DRIFT_TYPES:
        xs, ys = [], []
        for r in result["runs"]:
            if r["drift_type"] != t:
                continue
            xs.append(r["ground_truth"]["true_accuracy_drop"])
            ys.append(r["grading"]["estimator"]["implied_drop"])
        if xs:
            ax.scatter(xs, ys, s=70, color=colours[t], edgecolor="0.2",
                       linewidth=0.5, label=_PRETTY[t])
    hi = max(0.05, max(r["ground_truth"]["true_accuracy_drop"] for r in result["runs"]) + 0.04)
    ax.plot([-0.05, hi], [-0.05, hi], "0.4", ls="--", lw=1, label="perfect estimate")
    ax.set_xlim(-0.05, hi)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("true accuracy drop")
    ax.set_ylabel("drop implied by the label-free estimate")
    ax.set_title("The label-free estimate under-reports every injected drop")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_intensity_response(result: Dict[str, Any], path: Path) -> None:
    plt = _mpl()
    types = ["covariate", "prior", "concept", "label_noise"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
    handles: List[Any] = []
    for k, (ax, t) in enumerate(zip(axes, types)):
        runs = sorted((r for r in result["runs"] if r["drift_type"] == t),
                      key=lambda r: r["intensity_order"])
        if not runs:
            continue
        labels = [r["intensity_label"] for r in runs]
        x = list(range(len(runs)))
        aud = [r["labelled_audit"]["measured_drop"] for r in runs]
        est = [abs(r["grading"]["estimator"]["implied_drop"]) for r in runs]
        det = [_detector_loudness(r["pipeline"]["detection"]) for r in runs]

        l1, = ax.plot(x, aud, "o-", color="#d62728", label="labelled audit: measured accuracy drop")
        l2, = ax.plot(x, est, "s--", color="#9467bd", label="label-free: implied accuracy drop")
        ax.set_ylim(-0.02, 0.42)
        ax.set_xticks(x, labels)
        ax.set_title(_PRETTY[t])
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        l3, = ax2.plot(x, det, "^:", color="#1f77b4", label="label-free: detector loudness (right axis)")
        ax2.set_ylim(-0.1, 2.5)
        if k != len(types) - 1:
            ax2.set_yticklabels([])
        else:
            ax2.set_ylabel("detector loudness", color="#1f77b4")
            ax2.tick_params(axis="y", colors="#1f77b4")
        if k == 0:
            handles = [l1, l2, l3]

    axes[0].set_ylabel("accuracy drop")
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("How each stage responds to injected intensity: label-free detection climbs for "
                 "covariate and prior, stays flat for concept and label noise")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _detector_loudness(detection: Dict[str, Any]) -> float:
    parts = []
    for name in ("KS Test", "PSI", "MMD"):
        b = detection.get(name, {})
        if isinstance(b, dict) and "error" not in b:
            parts.append(float(b.get("score", 0.0)))
    cbpe = detection.get("CBPE", {})
    cterm = float(np.tanh(cbpe.get("score", 0.0) / 200.0)) if isinstance(cbpe, dict) else 0.0
    return max(parts, default=0.0) + cterm


def fig_reliability_auroc(result: Dict[str, Any], path: Path) -> None:
    plt = _mpl()
    order = ["covariate", "prior", "concept", "label_noise"]
    aur = result["summary"]["reliability_auroc_by_type"]
    vals = [aur.get(t) if aur.get(t) is not None else np.nan for t in order]
    colours = ["#2ca02c" if (v is not None and v >= 0.55) else
               ("#d62728" if (v is not None and v < 0.45) else "0.6") for v in vals]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar([_PRETTY[t] for t in order], vals, color=colours, edgecolor="0.2")
    ax.axhline(0.5, color="0.3", ls="--", lw=1, label="chance")
    ax.set_ylim(0.3, 0.75)
    ax.set_ylabel("AUROC: risk score vs prediction error")
    ax.set_title("When the per-prediction reliability score is informative")
    for i, v in enumerate(vals):
        if v == v:
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_signature(result: Dict[str, Any], path: Path) -> None:
    plt = _mpl()
    rows = ["covariate", "prior", "concept", "label_noise"]
    cols = [
        "peak KS statistic", "KS localised feats", "domain AUC - .5",
        "model prior shift", "estimator implied drop", "audit measured drop",
        "audit prior shift", "audit structure",
    ]
    M = np.zeros((len(rows), len(cols)))
    for i, t in enumerate(rows):
        runs = [r for r in result["runs"] if r["drift_type"] == t]
        r = max(runs, key=lambda r: r["intensity_order"])
        sig = r["signature"]
        aud = r["labelled_audit"]
        M[i, 0] = min(sig["max_localised_ks_statistic"] / 0.5, 1)
        M[i, 1] = min(sig["n_features_localised"] / 6.0, 1)
        M[i, 2] = min(max(sig["domain_classifier_auc"] - 0.5, 0) / 0.5, 1)
        M[i, 3] = min(sig["predicted_prior_shift"] / 0.35, 1)
        M[i, 4] = min(abs(sig["estimator_implied_drop"]) / 0.10, 1)
        M[i, 5] = min(max(aud["measured_drop"], 0) / 0.35, 1)
        M[i, 6] = min(abs(aud.get("prior_shift", 0.0)) / 0.35, 1)
        M[i, 7] = min(max(aud["structure_score"], 0) / 0.35, 1)

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), [_PRETTY[t] for t in rows])
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=8)
    ax.set_title("Label-free + delayed-label fingerprint at the strongest injected intensity")
    fig.colorbar(im, ax=ax, shrink=0.8, label="normalised signal")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_all_figures(result: Dict[str, Any]) -> List[str]:
    jobs = [
        (fig_confusion, "injection_type_confusion.png"),
        (fig_estimator_optimism, "injection_estimator_optimism.png"),
        (fig_intensity_response, "injection_intensity_response.png"),
        (fig_reliability_auroc, "injection_reliability_auroc.png"),
        (fig_signature, "injection_signature.png"),
    ]
    written = []
    for fn, name in jobs:
        try:
            fn(result, FIG_DIR / name)
            written.append(str(FIG_DIR / name))
        except Exception as exc:  # a figure failing must not lose the JSON
            print(f"  figure {name} failed: {exc}")
    return written


# --------------------------------------------------------------------------- #
# table

def render_table(result: Dict[str, Any], path: Path) -> None:
    rows = []
    for t in DRIFT_TYPES:
        runs = [r for r in result["runs"] if r["drift_type"] == t]
        if not runs:
            continue
        r = max(runs, key=lambda r: r["intensity_order"])
        gt = r["ground_truth"]
        v = r["verdict"]
        loc = r["grading"]["localisation"]
        loc_txt = (
            f"{loc['f1']:.2f}" if loc.get("f1") is not None
            else ("clean" if loc.get("clean") else "--")
        )
        auroc = r["pipeline"]["reliability"].get("auroc_risk_vs_error")
        rows.append([
            _PRETTY[t],
            f"{gt['true_accuracy_drop']:+.3f}",
            f"{r['grading']['estimator']['signed_error']:+.3f}",
            loc_txt,
            "yes" if loc.get("audit_match") else ("--" if loc.get("audit_match") is None else "no"),
            f"{auroc:.2f}" if (auroc is not None and t != "none") else "--",
            v["predicted_type"].replace("_", " "),
            "OK" if v["correct"] else "MISS",
        ])
    head = ("injected & true drop & est.\\ error & KS loc.\\ F1 & audit loc. & "
            "rel.\\ AUROC & verdict & result")
    body = " \\\\\n".join(" & ".join(str(c) for c in row) for row in rows)
    caption = (
        "\\caption{Drift-injection benchmark at the strongest injected intensity: the pipeline "
        "names every injected drift type, the label-free accuracy estimate is optimistic for all "
        "four, and the reliability score is informative only when the shift is covariate.}"
    )
    path.write_text(
        "\\begin{table}[!htbp]\n\\centering\n"
        + caption + "\n\\label{tab:injection-benchmark}\n"
        "\\renewcommand{\\arraystretch}{1.2}\\setlength{\\tabcolsep}{3pt}\\scriptsize\n"
        "\\begin{tabular}{@{}l r r r c c l c@{}}\n\\toprule\n"
        + head + " \\\\\n\\midrule\n" + body + " \\\\\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------- #

def print_summary(result: Dict[str, Any]) -> None:
    s = result["summary"]
    print("\n" + "=" * 72)
    print("DRIFT-INJECTION BENCHMARK")
    print("=" * 72)
    print(f"drift-type identification accuracy : {s['type_identification']['accuracy']:.0%} "
          f"({s['type_identification']['n_runs']} runs)")
    print(f"label-free estimator optimism rate : {s['estimator_optimism_rate']:.0%}")
    print(f"audit localisation hit rate        : {s['audit_localisation_hit_rate']}")
    print(f"reliability AUROC by type          : "
          + ", ".join(f"{k} {v}" for k, v in s["reliability_auroc_by_type"].items()))
    print("\nper-run verdicts:")
    for r in result["runs"]:
        v = r["verdict"]
        mark = "OK " if v["correct"] else "XX "
        print(f"  {mark}{r['drift_type']:12}/{r['intensity_label']:8} -> {v['predicted_type']:11} "
              f"[{v['stage']}]  drop {r['ground_truth']['true_accuracy_drop']:+.3f}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one intensity per type, smaller samples (fast check)")
    ap.add_argument("--figures-only", action="store_true",
                    help="rebuild figures + table from an existing JSON")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reliability-sample", type=int, default=200)
    ap.add_argument("--audit-labels", type=int, default=600)
    args = ap.parse_args()
    setup_logging(level="WARNING")
    OUT.mkdir(exist_ok=True)

    if args.figures_only:
        result = json.loads(RESULT_PATH.read_text())
        render_table(result, TABLE_PATH)
        written = render_all_figures(result)
        print("rebuilt:", TABLE_PATH, *written, sep="\n  ")
        return

    if args.quick:
        data = load_acs_reference(n_train=10000, n_reference=3500, n_pool=4500, seed=args.seed)
        result = run_benchmark(
            data=data, reliability_sample=120, audit_labels=400,
            seed=args.seed, progress=print,
        )
    else:
        data = load_acs_reference(seed=args.seed)
        result = run_benchmark(
            data=data, reliability_sample=args.reliability_sample,
            audit_labels=args.audit_labels, seed=args.seed, progress=print,
        )

    RESULT_PATH.write_text(json.dumps(result, indent=2, default=float))
    render_table(result, TABLE_PATH)
    written = render_all_figures(result)
    print_summary(result)
    print(f"\nwrote {RESULT_PATH}")
    print(f"wrote {TABLE_PATH}")
    for w in written:
        print(f"wrote {w}")


if __name__ == "__main__":
    main()
