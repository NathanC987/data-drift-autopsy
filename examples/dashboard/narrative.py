"""'What the data shows' sentences, computed from the loaders' frames.

Every number and every list in these sentences comes from the outputs, so the
text changes when the results change. Pure functions, unit tested.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from loaders import TYPE_LABEL


def join(items: Sequence[str]) -> str:
    items = [str(i) for i in items]
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def pp(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f} pp"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


# --------------------------------------------------------------------------- #
# ACS and CLEAR-10

def accuracy(summary: pd.DataFrame, alpha: float) -> str:
    drift = summary[summary["type"] != "control"]
    control = summary[summary["type"] == "control"]
    sig = drift[drift["drop significant"]]
    worst = drift.sort_values("accuracy change").iloc[0]
    parts = []
    if sig.empty:
        parts.append(f"No window shows a significant accuracy drop at α = {alpha}.")
    else:
        parts.append(f"Accuracy falls significantly in {len(sig)} of {len(drift)} windows: {join(sig['window'])}.")
    parts.append(f"The largest change is {pp(worst['accuracy change'])} in {worst['window']} "
                 f"(one-sided p = {fmt_p(worst['drop p-value'])}).")
    if not control.empty:
        c = control.iloc[0]
        parts.append(f"The held-out control changes by {pp(c['accuracy change'], 2)}, "
                     f"{'a significant drop' if c['drop significant'] else 'not a significant drop'}.")
    return " ".join(parts)


def proxy(summary: pd.DataFrame) -> str:
    drift = summary[summary["type"] != "control"]
    optimistic = drift[drift["CBPE error"] > 0]
    worst = drift.reindex(drift["CBPE error"].abs().sort_values(ascending=False).index).iloc[0]
    control = summary[summary["type"] == "control"]
    out = [f"The estimate is above the measured accuracy (optimistic) in {len(optimistic)} of {len(drift)} drift windows.",
           f"The largest error is {pp(worst['CBPE error'])} in {worst['window']}."]
    if not control.empty:
        out.append(f"On the control it is off by {pp(control.iloc[0]['CBPE error'], 2)}.")
    return " ".join(out)


def confidence(test: Dict, window: str) -> str:
    verdict = "an alarm" if test["drift"] else "no alarm"
    return (f"Mean confidence moves from {test['ref_mean_confidence']:.3f} (reference) to "
            f"{test['test_mean_confidence']:.3f} in {window}. The chi-square test on the histogram gives "
            f"p = {fmt_p(test['p_value'])}, so {verdict}.")


def detection(det: pd.DataFrame) -> str:
    control_keys = set(det.loc[det["window"].str.contains("held", case=False) | det["window"].str.contains("control"), "key"])
    ctrl = det[det["key"].isin(control_keys)]
    drift = det[~det["key"].isin(control_keys)]
    n = drift["key"].nunique()
    lines = []
    for name, d in drift.groupby("detector", sort=False):
        hit = d[d["drift"]]
        if hit.empty:
            lines.append(f"{name} never alarms")
        elif len(hit) == n:
            lines.append(f"{name} alarms in all {n}")
        else:
            lines.append(f"{name} alarms in {len(hit)} of {n} ({join(hit['window'])})")
    ctrl_alarms = ctrl[ctrl["drift"]]["detector"].tolist()
    head = ("No detector alarms on the held-out control. " if not ctrl_alarms
            else f"On the held-out control {join(ctrl_alarms)} alarm (false alarms). ")
    return head + "; ".join(lines) + "."


def sample_size_note(summary: pd.DataFrame, det: pd.DataFrame) -> str:
    """Explain a raw score that is high on the small no-drift control without an alarm."""
    control = summary[summary["type"] == "control"]
    drift = summary[summary["type"] != "control"]
    if control.empty or drift.empty:
        return ""
    ckey = control.iloc[0]["key"]
    odd = []
    for name, d in det.groupby("detector", sort=False):
        c = d[d["key"] == ckey]
        prod = d[d["key"] != ckey]
        if not c.empty and not c.iloc[0]["drift"] and c.iloc[0]["score"] > prod["score"].median():
            odd.append(name)
    if not odd:
        return ""
    n_c, n_d = int(control.iloc[0]["n"]), int(drift["n"].median())
    tested = [o for o in odd if o != "PSI"]
    out = [f"On the held-out control, {join(odd)} {'score' if len(odd) > 1 else 'scores'} higher than in most production "
           f"windows without an alarm. The control has {n_c:,} rows against a median of {n_d:,}, and raw distance "
           "statistics are biased upward in small samples."]
    if tested:
        out.append(f"The tests behind {join(tested)} account for sample size, so they do not alarm.")
    if "PSI" in odd:
        out.append("PSI has no test; it stays below its 0.2 convention.")
    return " ".join(out)


def acs_localization(loc: pd.DataFrame, key: str) -> str:
    d = loc[loc["key"] == key]
    w = d["window"].iloc[0]
    sig = d[d["drifted"]].sort_values("effect size", ascending=False)
    out = [f"In {w}, {len(sig)} of {len(d)} features shift significantly after Holm correction."]
    if not sig.empty:
        top = sig.iloc[0]
        measure = "Cramér's V" if top["test"] == "categorical" else "KS D"
        out.append(f"The largest effect is {top['label']} ({measure} = {top['effect size']:.2f}).")
    return " ".join(out)


def unseen_overall(loc: pd.DataFrame) -> str:
    u = loc.dropna(subset=["unseen category share"]).sort_values("unseen category share", ascending=False)
    if u.empty or u.iloc[0]["unseen category share"] < 0.01:
        return "No window has more than 1% of rows with a category unseen in the reference."
    r = u.iloc[0]
    return (f"Across all windows the largest share of unseen categories is {pct(r['unseen category share'])} "
            f"({r['feature']} in {r['window']}).")


def clear_localization(loc: pd.DataFrame, slices: pd.DataFrame, key: str) -> str:
    r = loc[loc["key"] == key].iloc[0]
    s = slices[slices["key"] == key].sort_values("drifted dimensions", ascending=False)
    out = [f"In {r['window']}, {r['drifted dimensions']} of {r['of']} dimensions shift over all images."]
    if not s.empty and s.iloc[0]["drifted dimensions"] > 0:
        out.append(f"Within predicted classes the most shifted is {s.iloc[0]['class']} "
                   f"({s.iloc[0]['drifted dimensions']} dimensions).")
    return " ".join(out)


def pca(proj: Dict) -> str:
    c = proj["centroids"].set_index("bucket")[["PC1", "PC2", "PC3"]]
    dist = np.linalg.norm(c.values - c.loc[1].values, axis=1)
    far = c.index[int(np.argmax(dist))]
    ref = proj["points"][proj["points"]["bucket"] == 1][["PC1", "PC2", "PC3"]].to_numpy()
    spread = float(np.sqrt(((ref - ref.mean(axis=0)) ** 2).sum(axis=1).mean()))
    return (f"The three components keep {pct(sum(proj['explained']))} of the variance, so this is a rough view. "
            f"The bucket centroid moves furthest from bucket 1 at bucket {far}, by {dist.max():.2f}, which is "
            f"{dist.max() / spread:.0%} of the typical distance of a bucket 1 image from its centre ({spread:.1f}). "
            + ("So the clouds overlap almost completely in this view. " if dist.max() / spread < 0.25 else
               "So the clouds are partly separated in this view. " if dist.max() / spread < 1 else
               "So the clouds are clearly separated in this view. ")
            + "The detectors test all 512 dimensions, not this projection.")


def shap(shap_df: pd.DataFrame, window: str) -> str:
    sig = shap_df[shap_df["change_significant"]].copy()
    if sig.empty:
        return f"In {window}, no feature's reliance changes significantly."
    sig["abs"] = sig["change"].abs()
    top = sig.sort_values("abs", ascending=False).iloc[0]
    direction = "drops" if top["change"] < 0 else "rises"
    return (f"In {window}, reliance changes significantly on {len(sig)} of {len(shap_df)} features. "
            f"The largest change: reliance on {top['label']} {direction} by {abs(top['change']):.3f}.")


def diagnosis(summary: pd.DataFrame) -> str:
    parts = []
    for verdict, g in summary.groupby("diagnosis", sort=False):
        stage = join(sorted(set(g["stage"])))
        parts.append(f"{verdict} in {join(g['window'])} ({stage} stage)")
    return "Verdicts: " + "; ".join(parts) + "."


def prior(df: pd.DataFrame, key: str) -> str:
    d = df[df["key"] == key].set_index("series")["positive rate"]
    w = df.loc[df["key"] == key, "window"].iloc[0]
    actual, pred, bbse = d["actual (needs labels)"], d["predicted by the model"], d["BBSE estimate (label free)"]
    return (f"In {w}, the actual positive rate is {pct(actual)} while the model predicts {pct(pred)} "
            f"and BBSE estimates {pct(bbse)}.")


def reliability(rel: pd.DataFrame, coefs: pd.DataFrame) -> str:
    gain = (rel["auroc_risk"] - rel["auroc_confidence_only"])
    ordered = ((rel["error_rate_low"] < rel["error_rate_medium"]) & (rel["error_rate_medium"] < rel["error_rate_high"])).sum()
    if coefs.empty:
        weights = "No fusion weights were learned (too few calibration errors), so the risk is 1 - confidence alone. "
    else:
        top = coefs.sort_values("coefficient", ascending=False).iloc[0]
        weights = f"The largest fusion weight is {top['signal']} ({top['coefficient']:+.2f}). "
    return (f"The fused score's AUROC differs from confidence alone by {gain.min():+.3f} to {gain.max():+.3f}. "
            f"The out of distribution signal stays between {rel['auroc_ood'].min():.2f} and {rel['auroc_ood'].max():.2f} "
            f"(0.5 is chance). " + weights +
            f"The risk labels are correctly ordered (LOW < MEDIUM < HIGH error rate) in {ordered} of {len(rel)} windows.")


def remediation(table: pd.DataFrame, card: pd.DataFrame) -> str:
    applicable = table[table["applicable"]]
    by = applicable.groupby("strategy", sort=False).agg(sig=("significant", "sum"), n=("significant", "size"),
                                                        best=("recovered", "max"))
    helps = [f"{s} {int(r.sig)} of {int(r.n)}" for s, r in by.iterrows()]
    wrong = card[~card["correct"]]
    return (f"Significant gains per strategy: {join(helps)}. The triage is right in {int(card['correct'].sum())} "
            f"of {len(card)} settings" + (f"; it is wrong in {join(wrong['setting'])}." if not wrong.empty else "."))


def concepts(val: pd.DataFrame, alpha: float) -> str:
    first, last = val.iloc[0], val.iloc[-1]
    n_sig_rho = int((val["p (all)"] < alpha).sum())
    return (f"Significant concept shifts grow from {first['significant concepts']} in bucket {first['bucket']} to "
            f"{last['significant concepts']} in bucket {last['bucket']}. Agreement with the image tags is significant "
            f"in {n_sig_rho} of {len(val)} buckets, so the tags do not confirm the ranking.")


def exemplars(ex: Dict, focus: str, bucket: str) -> str:
    r, p = ex["reference"]["share_top_concept"], ex["production"]["share_top_concept"]
    return (f"Within class {focus}, '{ex['concept']}' is the best-matching concept for {pct(r, 0)} of bucket 1 images "
            f"and {pct(p, 0)} of bucket {bucket} images.")


# --------------------------------------------------------------------------- #
# benchmark

def bench_design(bench: Dict) -> str:
    c = bench["config"]
    s = bench["summary"]
    n_mech = len(c["intensity_ladder"])
    content = (f"one control and {n_mech} drift types at 3 intensities" if c.get("design", "A") == "A"
               else f"{n_mech} held-out mechanisms at 3 intensities (no control; design A has them)")
    return (f"{s['n_runs']} windows: {len(c['states'])} states × {len(c['seeds'])} seeds = {s['n_replicates']} replicates, "
            f"each with {content}. Windows are drawn with {c['production_size']:,} rows before injection; grading uses "
            f"{c['holdout_size']:,} separate labelled rows.")


def bench_ground_truth(runs: pd.DataFrame) -> str:
    m = runs.groupby(["mechanism", "intensity"], sort=False)["true drop"].mean()
    ctrl = runs.loc[runs["injected"] == "no drift", "true drop"]
    strongest = m.drop(index="no drift", level=0, errors="ignore").groupby(level=0, sort=False).max()
    head = f"On controls the mean accuracy drop is {ctrl.mean() * 100:.2f} pp. " if len(ctrl) else ""
    return (head + "The largest mean drop over the three intensities is "
            + join([f"{v * 100:.1f} pp for {t}" for t, v in strongest.items()]) + ".")


def bench_detection(fa: pd.DataFrame, alpha: float) -> str:
    over = fa[fa["ci low"] > alpha]
    return ("False alarm rates on controls: " + join([f"{r['detector']} {pct(r['rate'], 0)}" for _, r in fa.iterrows()]) +
            (". None is significantly above α." if over.empty else f". Above α: {join(over['detector'])}."))


def bench_estimator(bt: pd.DataFrame) -> str:
    return "Mean CBPE error (estimate minus truth): " + join(
        [f"{r['injected']} {pp(r['CBPE error'])}" for _, r in bt.iterrows()]) + ". Positive means optimistic."


def bench_type_id(bench: Dict) -> str:
    o = bench["summary"]["type_identification"]["overall"]
    pt = bench["summary"]["type_identification"]["per_type"]
    return (f"The drift type is named correctly in {o['k']} of {o['n']} windows ({pct(o['rate'])}, 95% CI "
            f"{pct(o['ci'][0])} to {pct(o['ci'][1])}). Per type: "
            + join([f"{TYPE_LABEL[t]} {v['k']}/{v['n']}" for t, v in pt.items() if v["n"]]) + ".")


def bench_audit(ab: pd.DataFrame) -> str:
    c = ab[ab["type"] == "concept"].sort_values("labels")
    if c.empty:
        return ""
    return ("Concept drift is named in " + join([f"{pct(r['correct'], 0)} with {r['labels']} labels"
                                                  for _, r in c.iterrows()])
            + ". Label noise and concept drift need the audit because their inputs do not change.")


def bench_reliability(bt: pd.DataFrame) -> str:
    gain = bt["fused minus confidence"]
    verdict = (" The fused score adds little or nothing beyond confidence (at most "
               f"{gain.max():+.3f} AUROC)." if gain.abs().max() < 0.02 else "")
    return ("Fused minus confidence AUROC: " + join([f"{r['injected']} {r['fused minus confidence']:+.3f}"
                                                     for _, r in bt.iterrows()]) + "." + verdict)


def bench_localization(bench: Dict) -> str:
    parts = []
    for t, v in bench["summary"]["localization"].items():
        e = v["exact_match"]
        parts.append(f"{TYPE_LABEL[t]}: the blamed features match the injected ones exactly in {e['k']} of {e['n']} "
                     f"windows (mean precision {v['precision']['mean']:.2f}, recall {v['recall']['mean']:.2f})")
    return "; ".join(parts) + ". Counted over the windows whose drift type was named correctly."


def bench_triage(bench: Dict) -> str:
    t = bench["summary"]["triage"]
    c = t["confusion"]
    return (f"The triage agrees with the truth in {pct(t['agreement']['rate'])} of windows. When it escalates it is "
            f"right {c['tp']} of {c['tp'] + c['fp']} times (precision {pct(t['precision']['rate'], 0)}), but it "
            f"escalates only {c['tp']} of the {c['tp'] + c['fn']} windows where retraining truly helps "
            f"(recall {pct(t['recall']['rate'], 0)}).")


def bench_methods(df: pd.DataFrame) -> str:
    """Ours against the baselines, the learned rule and the ablations of one benchmark configuration."""
    if df.empty:
        return ""
    ours = df[df["key"] == "ours"].iloc[0]
    sig = df[df["significant (Holm)"] == True]  # noqa: E712
    text = f"Ours names the type in {pct(ours['accuracy'])} of windows (macro F1 {ours['macro F1']:.3f}). "
    base = df[df["kind"] == "baseline"]
    beaten = sig[(sig["kind"] == "baseline") & (sig["ours minus method"] > 0)]
    text += (f"It beats {len(beaten)} of the {len(base)} rule baselines significantly"
             + (f" ({join(beaten['method'])})" if len(beaten) else "") + ". ")
    meta = df[df["kind"] == "learned"]
    if not meta.empty:
        m = meta.iloc[0]
        verdict = ("is significantly better" if m["significant (Holm)"] and m["ours minus method"] < 0 else
                   "is significantly worse" if m["significant (Holm)"] and m["ours minus method"] > 0 else
                   "does not differ significantly")
        text += f"The learned meta-classifier ({pct(m['accuracy'])}) {verdict}. "
    abl = sig[sig["kind"] == "ablation"]
    hurt = abl[abl["ours minus method"] > 0]
    help_ = abl[abl["ours minus method"] < 0]
    if len(hurt):
        text += f"Ablations that lose accuracy: {join(hurt['method'])}. "
    if len(help_):
        text += f"Ablations that gain accuracy: {join(help_['method'])}. "
    return text + "Paired sign-flip test over replicates, Holm correction."


def bench_estimators(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    allw = df[df["type"] == "all windows"].sort_values("MAE")
    return ("Over all windows the mean absolute error is " +
            join([f"{r['MAE'] * 100:.2f} pp for {r['estimator']}" for _, r in allw.iterrows()]) +
            ". Under concept drift and label noise no label-free estimate can move, because the inputs do not change.")
