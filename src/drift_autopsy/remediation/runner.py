"""Drive a set of remediation strategies over a context and tabulate the result."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import pandas as pd

from drift_autopsy.remediation.base import RemediationContext, RemediationResult
from drift_autopsy.remediation import strategies as S

DEFAULT_STRATEGIES: Dict[str, Callable[[RemediationContext], RemediationResult]] = {
    "full_retrain": S.full_retrain,
    "feature_drop_retrain": S.feature_drop_retrain,
    "importance_weighted_retrain": S.importance_weighted_retrain,
    "retrain_on_recent[previous]": lambda c: S.retrain_on_recent(c, window="previous"),
    "retrain_on_recent[all_prior]": lambda c: S.retrain_on_recent(c, window="all_prior"),
    "head_refit": S.head_refit,
    "calibration_only": S.calibration_only,
}


def run_remediation_suite(
    ctx: RemediationContext,
    strategies: Sequence[str] | None = None,
) -> List[RemediationResult]:
    names = list(strategies) if strategies is not None else list(DEFAULT_STRATEGIES)
    out: List[RemediationResult] = []
    for name in names:
        fn = DEFAULT_STRATEGIES.get(name)
        if fn is None:
            continue
        try:
            out.append(fn(ctx))
        except Exception as exc:  # keep going; a missing recent window etc. is fine
            print(f"  [{ctx.shift_name}] strategy '{name}' skipped: {exc}")
    return out


def results_to_frame(results: Sequence[RemediationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])


def results_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    cols = [
        "shift_name", "strategy", "accuracy_before", "accuracy_after",
        "fraction_of_gap_recovered", "wall_clock_seconds", "train_samples",
        "n_production_labels_required",
    ]
    view = df[[c for c in cols if c in df.columns]].copy()
    body = " \\\\\n".join(
        " & ".join(_fmt(v) for v in row) for row in view.itertuples(index=False)
    )
    head = " & ".join(c.replace("_", " ") for c in view.columns)
    return (
        "\\begin{table}[!htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\renewcommand{\\arraystretch}{1.15}\\setlength{\\tabcolsep}{3pt}\\footnotesize\n"
        f"\\begin{{tabular}}{{@{{}}l l {'r ' * (len(view.columns) - 2)}@{{}}}}\n\\toprule\n"
        f"{head} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
    )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.0f}"
    return (
        str(v)
        .replace("_", "\\_")
        .replace("->", "$\\rightarrow$")
        .replace("[", "(")
        .replace("]", ")")
    )


def parse_drifted_features(
    drift_json: Dict[str, Any],
    mode: str,
    target_label: str | None = None,
) -> List[str]:
    """Pull the localised drifted-feature list out of a drift-results JSON.

    ``mode`` is ``"temporal"`` (last year) or ``"geographic"`` (a state slice,
    identified by ``target_label`` e.g. "NY").
    """
    if mode == "temporal":
        years = sorted(k for k in drift_json if k.isdigit())
        loc = drift_json[years[-1]]["pipelines"]["KS Test"]["localization"]
        return list(loc.get("drifted_features", []))

    if mode == "geographic":
        geo = drift_json["geographic_analysis"]
        labels = {v: k for k, v in geo.get("slice_value_labels", {}).items()}
        code = labels.get(target_label)
        slices = geo["pipelines"]["KS Test"]["metadata"]["slice_analysis"]["slices"]
        for payload in slices.values():
            if str(payload.get("test_slice_value")) == str(code):
                return list(payload["result"]["localization"].get("drifted_features", []))
    return []
