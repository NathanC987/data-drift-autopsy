"""Drift-type diagnosis from hypothesis tests.

Every decision below is a hypothesis test at one significance level ``alpha``
with Holm correction across features. No other tuned constants are involved.

Label-free stage (production labels never used)
-----------------------------------------------
1. Input shift: per-feature two-sample tests between the reference and the
   production inputs (KS for numeric features, chi-square homogeneity for
   nominal ones), plus one multivariate kernel test (MMD permutation test on
   standardised inputs), which keeps power when the inputs are many weakly
   shifted dimensions such as image embeddings.
2. Prediction shift: two-sample chi-square test on the predicted labels.
3. Black-box shift estimation (BBSE; Lipton, Wang and Smola, 2018). With C the
   joint matrix C[i, j] = P(yhat = i, y = j) on the labelled reference and mu
   the production predicted-label distribution, the class weights solve
   C w = mu (clipped at 0). w[j] = q(y = j) / p(y = j) estimates how the class
   prior moved.
4. Label-shift consistency. Each reference row is reweighted by w[y]. Under
   pure prior shift, P(X | Y) is unchanged and the reweighted reference has
   the production input distribution. The input tests are repeated against
   the weighted reference (weighted KS and weighted chi-square with Kish's
   effective sample size). Features that still differ are the residual shift.

   Steps 1 and 2 are tested hierarchically at family-wise level alpha. At the
   global level, Holm runs over three block tests: the univariate block
   (Bonferroni p-value m * min p over its m feature tests), the prediction test
   and the multivariate test. Only if the univariate block rejects are the
   individual features tested, by Holm within the block at the level the block
   was rejected at (a gatekeeping procedure, so family-wise control holds over
   all decisions). Pooling the feature tests into one block is what lets a
   permutation test, whose smallest attainable p-value is 1/(B + 1), compete
   with hundreds of feature tests.

   * no test in the family rejects           -> no label-free evidence of drift
   * residual shift on some features         -> covariate shift on those features
   * no residual and a prediction shift      -> prior shift
   * no residual and no prediction shift     -> covariate shift (seen only
     jointly). A prior shift always moves the predicted-label distribution of
     an informative classifier (the identifiability condition of BBSE), so a
     shift without one cannot be a prior shift.

Delayed-label audit (a small labelled production sample)
--------------------------------------------------------
5. Accuracy drop: one-sided two-proportion test, audit vs reference accuracy.
6. Error-structure change. A logistic model of "prediction correct" is fitted
   on the reference rows plus the audit rows:
       null:        conf_bin + f_bin + audit + audit x conf_bin
       alternative: null + audit x f_bin
   where conf_bin are confidence quintiles and f_bin are bins of feature f.
   The null lets the audit have its own confidence-to-accuracy curve, which
   absorbs any change that acts through confidence alone (label noise flips
   P(correct | x) to r + (1 - 2r) P(correct | x), a function of confidence).
   The likelihood-ratio test asks whether the change in errors also depends
   on f. A significant feature means P(Y | X) moved in a region of f
   (concept drift); none means the corruption is unstructured (label noise).

Ablations (``null`` argument; the default "confidence" is the method):
"none" drops audit x conf_bin from the null, so any change in errors is
attributed to features; "class_confidence" replaces conf_bin by the joint
cells of predicted class and confidence bin, which also absorbs label noise
that depends on the true class.

Bins. Numeric features: reference quintiles. Nominal features: categories
seen at least ``ceil(5 / min(err, 1 - err))`` times in the audit (so each
kept cell has an expected count of at least five of both outcomes, the usual
validity rule), the rest pooled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.detectors import MMD
from drift_autopsy.estimation import top_class_confidence
from drift_autopsy.localizers import UnivariateLocalizer
from drift_autopsy.stats import (
    chi2_homogeneity,
    holm_reject,
    two_proportion_test,
    weighted_ks,
)

N_QUANTILE_BINS = 5


# --------------------------------------------------------------------------- #
# BBSE

def bbse_weights(
    reference_pred: np.ndarray,
    reference_y: np.ndarray,
    production_pred: np.ndarray,
    classes: Sequence[Any],
) -> Dict[str, Any]:
    """Black-box shift estimation of production class weights q(y)/p(y)."""
    classes = list(classes)
    k = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    C = np.zeros((k, k))
    for yhat, y in zip(reference_pred, reference_y):
        C[index[yhat], index[y]] += 1
    C /= C.sum()
    mu = np.array([np.mean(np.asarray(production_pred) == c) for c in classes])
    try:
        w = np.linalg.solve(C, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(C, mu, rcond=None)[0]
    w = np.clip(w, 0.0, None)
    p_ref = np.array([np.mean(np.asarray(reference_y) == c) for c in classes])
    q_est = w * p_ref
    q_est = q_est / q_est.sum() if q_est.sum() > 0 else p_ref
    return {
        "classes": classes,
        "weights": w.tolist(),
        "reference_prior": p_ref.tolist(),
        "estimated_production_prior": q_est.tolist(),
        "condition_number": float(np.linalg.cond(C)),
    }


# --------------------------------------------------------------------------- #
# label-free stage

def _weighted_feature_tests(
    reference: pd.DataFrame, weights: np.ndarray, production: pd.DataFrame,
    numeric: Sequence[str], categorical: Sequence[str], alpha: float,
) -> List[Dict[str, Any]]:
    rows = []
    for f in numeric:
        r = weighted_ks(reference[f].to_numpy(float), weights, production[f].to_numpy(float))
        rows.append({"feature": f, "test": "weighted_ks", "statistic": r["statistic"], "p_value": r["p_value"]})
    for f in categorical:
        r = chi2_homogeneity(reference[f].to_numpy(), production[f].to_numpy(), ref_weights=weights)
        rows.append({"feature": f, "test": "weighted_chi2", "statistic": r["cramers_v"], "p_value": r["p_value"]})
    rejected = holm_reject([r["p_value"] for r in rows], alpha)
    for r, rej in zip(rows, rejected):
        r["drift_detected"] = bool(rej)
    return rows


def label_free_diagnosis(
    reference: Dataset,
    production: Dataset,
    alpha: float = 0.05,
    multivariate: bool = True,
    mmd_permutations: int = 200,
    mmd_max_samples: int = 3000,
    random_state: int = 0,
    precomputed_mmd: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Steps 1-4 of the module docstring. ``reference`` needs target and predictions;
    ``production`` needs predictions only. ``precomputed_mmd`` ({"statistic",
    "p_value"}) reuses an MMD test already run with the same settings and seed
    on the same pair of samples, instead of running it again."""
    if reference.target is None or reference.predictions is None or production.predictions is None:
        raise ValueError("reference needs target and predictions; production needs predictions")

    # Hierarchical (gatekeeping) testing with family-wise level alpha:
    #   global level  - Holm over three block tests: the univariate block
    #                   (Bonferroni p = m * min p over the feature tests), the
    #                   prediction test and the multivariate MMD test;
    #   feature level - only if the univariate block rejects, Holm within the
    #                   block at the level that block was rejected at.
    loc = UnivariateLocalizer(threshold=alpha, correction=None).localize(reference, production)
    input_tests = [
        {"feature": fd.feature_name, "statistic": fd.score, "p_value": fd.p_value,
         "test": (fd.distribution_shift or {}).get("test")}
        for fd in loc.feature_drifts
    ]
    pred_test = chi2_homogeneity(np.asarray(reference.predictions), np.asarray(production.predictions))
    mmd_test = None
    if multivariate and precomputed_mmd is not None:
        mmd_test = {"statistic": float(precomputed_mmd["statistic"]), "p_value": float(precomputed_mmd["p_value"])}
    elif multivariate:
        res = MMD(threshold=alpha, n_permutations=mmd_permutations, max_samples=mmd_max_samples,
                  standardize=True, random_state=random_state).fit_detect(reference, production)
        mmd_test = {"statistic": float(res.score), "p_value": float(res.p_value)}

    feature_p = np.array([t["p_value"] for t in input_tests], dtype=float)
    block_p = [min(1.0, len(feature_p) * float(feature_p.min())) if feature_p.size else 1.0, pred_test["p_value"]]
    if mmd_test is not None:
        block_p.append(mmd_test["p_value"])
    block_reject = holm_reject(block_p, alpha)
    ranks = np.empty(len(block_p), dtype=int)
    ranks[np.argsort(block_p, kind="stable")] = np.arange(len(block_p))
    univariate_level = alpha / (len(block_p) - ranks[0])
    feature_reject = (holm_reject(feature_p, univariate_level) if block_reject[0]
                      else np.zeros(len(feature_p), dtype=bool))
    for t, rej in zip(input_tests, feature_reject):
        t["drift_detected"] = bool(rej)
    input_shift = [t["feature"] for t in input_tests if t["drift_detected"]]
    prediction_shift = bool(block_reject[1])
    multivariate_shift = bool(block_reject[2]) if mmd_test is not None else False
    rejected = block_reject
    block_tests = {"univariate_bonferroni_p": block_p[0], "univariate_level": float(univariate_level),
                   "prediction_p": block_p[1], "multivariate_p": block_p[2] if mmd_test is not None else None,
                   "rejected": [bool(r) for r in block_reject]}
    if mmd_test is not None:
        mmd_test["drift_detected"] = multivariate_shift

    classes = sorted(set(np.asarray(reference.target).tolist()) | set(np.asarray(reference.predictions).tolist()))
    bbse = bbse_weights(np.asarray(reference.predictions), np.asarray(reference.target),
                        np.asarray(production.predictions), classes)
    w_by_class = dict(zip(classes, bbse["weights"]))
    row_weights = np.array([w_by_class[y] for y in np.asarray(reference.target)], dtype=float)

    numeric = reference.numeric_features
    categorical = reference.categorical_features or []
    residual_tests = _weighted_feature_tests(
        reference.to_pandas(), row_weights, production.to_pandas(), numeric, categorical, alpha,
    )
    residual_shift = [t["feature"] for t in residual_tests if t["drift_detected"]]

    if not bool(np.any(rejected)):
        verdict = "none"
    elif residual_shift:
        verdict = "covariate"
    elif prediction_shift:
        verdict = "prior"
    else:
        verdict = "covariate"

    return {
        "verdict": verdict,
        "input_shift_features": input_shift,
        "input_tests": input_tests,
        "prediction_shift": prediction_shift,
        "prediction_shift_test": pred_test,
        "multivariate_shift": multivariate_shift,
        "multivariate_test": mmd_test,
        "block_tests": block_tests,
        "bbse": bbse,
        "residual_shift_features": residual_shift,
        "residual_tests": residual_tests,
    }


# --------------------------------------------------------------------------- #
# audit stage

def _bins(values: np.ndarray, reference_values: np.ndarray, categorical: bool, min_count: int) -> np.ndarray:
    if not categorical:
        edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, N_QUANTILE_BINS + 1))[1:-1])
        return np.digitize(values, edges)
    counts = pd.Series(values).value_counts()
    keep = set(counts[counts >= min_count].index)
    codes = {c: i for i, c in enumerate(sorted(keep, key=str))}
    other = len(codes)
    return np.array([codes.get(v, other) for v in values])


def _design(blocks: List[np.ndarray]) -> np.ndarray:
    return np.hstack([b for b in blocks if b.size]) if blocks else np.empty((0, 0))


def _one_hot(codes: np.ndarray) -> np.ndarray:
    levels = np.unique(codes)
    return (codes[:, None] == levels[None, 1:]).astype(float)   # drop first level


def _loglik(X: np.ndarray, y: np.ndarray) -> float:
    model = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def audit_diagnosis(
    model: Any,
    reference: Dataset,
    audit_X: pd.DataFrame,
    audit_y: np.ndarray,
    alpha: float = 0.05,
    null: str = "confidence",
) -> Dict[str, Any]:
    """Steps 5-6 of the module docstring (``null`` selects an ablation)."""
    if null not in ("confidence", "none", "class_confidence"):
        raise ValueError(f"unknown null {null!r}")
    feats = reference.feature_names
    categorical = set(reference.categorical_features or [])
    ref_X = reference.to_pandas()[feats]
    audit_X = audit_X[feats].reset_index(drop=True)
    ref_y = np.asarray(reference.target)
    audit_y = np.asarray(audit_y)

    ref_proba = model.predict_proba(ref_X.to_numpy(float))
    aud_proba = model.predict_proba(audit_X.to_numpy(float))
    ref_correct = (model.classes_[ref_proba.argmax(1)] == ref_y).astype(int)
    aud_correct = (model.classes_[aud_proba.argmax(1)] == audit_y).astype(int)

    drop = two_proportion_test(int(ref_correct.sum()), len(ref_correct), int(aud_correct.sum()), len(aud_correct),
                               alternative="greater")
    pos = model.classes_[-1]
    prior = two_proportion_test(int((audit_y == pos).sum()), len(audit_y), int((ref_y == pos).sum()), len(ref_y))

    conf = np.concatenate([top_class_confidence(ref_proba), top_class_confidence(aud_proba)])
    ref_conf = top_class_confidence(ref_proba)
    conf_edges = np.unique(np.quantile(ref_conf, np.linspace(0, 1, N_QUANTILE_BINS + 1))[1:-1])
    conf_codes = np.digitize(conf, conf_edges)
    if null == "class_confidence":
        pred = np.concatenate([ref_proba.argmax(1), aud_proba.argmax(1)])
        conf_codes = pred * (len(conf_edges) + 1) + conf_codes
    conf_bin = _one_hot(conf_codes)
    is_audit = np.r_[np.zeros(len(ref_correct)), np.ones(len(aud_correct))][:, None]
    correct = np.concatenate([ref_correct, aud_correct])
    err = 1 - aud_correct.mean()
    min_count = int(np.ceil(5.0 / max(min(err, 1 - err), 1e-9)))

    null_kind = null
    rows = []
    for f in feats:
        values = np.concatenate([ref_X[f].to_numpy(float), audit_X[f].to_numpy(float)])
        codes = _bins(values, ref_X[f].to_numpy(float), f in categorical, min_count)
        if len(np.unique(codes[len(ref_correct):])) < 2:
            rows.append({"feature": f, "lr_statistic": 0.0, "dof": 0, "p_value": 1.0})
            continue
        f_bin = _one_hot(codes)
        null = _design([conf_bin, f_bin, is_audit] + ([] if null_kind == "none" else [is_audit * conf_bin]))
        alt = _design([null, is_audit * f_bin])
        lr = max(0.0, 2.0 * (_loglik(alt, correct) - _loglik(null, correct)))
        dof = alt.shape[1] - null.shape[1]
        rows.append({"feature": f, "lr_statistic": float(lr), "dof": int(dof),
                     "p_value": float(stats.chi2.sf(lr, dof)) if dof > 0 else 1.0})
    rejected = holm_reject([r["p_value"] for r in rows], alpha)
    for r, rej in zip(rows, rejected):
        r["significant"] = bool(rej)
    structured = sorted([r for r in rows if r["significant"]], key=lambda r: r["p_value"])

    return {
        "n_labels": int(len(audit_y)),
        "audit_accuracy": float(aud_correct.mean()),
        "reference_accuracy": float(ref_correct.mean()),
        "accuracy_drop_test": drop,
        "accuracy_drop_significant": bool(drop["p_value"] < alpha),
        "audit_positive_rate": float((audit_y == pos).mean()),
        "prior_test": prior,
        "error_structure_tests": rows,
        "structured_features": [r["feature"] for r in structured],
    }


# --------------------------------------------------------------------------- #
# combined verdict

@dataclass
class Diagnosis:
    drift_type: str
    stage: str
    rationale: str
    localized_features: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"drift_type": self.drift_type, "stage": self.stage,
                "rationale": self.rationale, "localized_features": self.localized_features}


def combine(label_free: Dict[str, Any], audit: Optional[Dict[str, Any]] = None) -> Diagnosis:
    """Name the drift type from the two stages (see module docstring)."""
    v = label_free["verdict"]
    if v == "covariate":
        # Localise to features that moved AND whose move a prior change does not
        # explain; reweighting by the BBSE weights can itself perturb features
        # that never moved, so the residual set alone over-reports.
        residual = label_free["residual_shift_features"]
        feats = [f for f in label_free["input_shift_features"] if f in residual] or residual
        if feats:
            return Diagnosis("covariate", "label-free",
                             f"inputs shifted and the shift on {', '.join(feats[:5])}"
                             f"{' and others' if len(feats) > 5 else ''} is not explained by a class-prior change",
                             feats)
        return Diagnosis("covariate", "label-free",
                         "the inputs shifted jointly (multivariate test) with no change in the predicted labels, "
                         "which rules out a prior shift", [])
    if v == "prior":
        q = label_free["bbse"]["estimated_production_prior"]
        p = label_free["bbse"]["reference_prior"]
        return Diagnosis("prior", "label-free",
                         f"the input shift is explained by a class-prior change "
                         f"(estimated positive rate {p[-1]:.3f} -> {q[-1]:.3f})", [])
    if audit is None:
        return Diagnosis("none", "label-free",
                         "no input or prediction shift; changes in P(Y|X) cannot be seen without labels", [])
    if not audit["accuracy_drop_significant"]:
        return Diagnosis("none", "audit",
                         f"no label-free shift and no significant accuracy drop in {audit['n_labels']} audit labels", [])
    if audit["structured_features"]:
        f = audit["structured_features"][0]
        return Diagnosis("concept", "audit",
                         f"inputs unchanged, accuracy fell, and the change in errors depends on {f}", [f])
    return Diagnosis("label_noise", "audit",
                     "inputs unchanged, accuracy fell, and the change in errors is explained by confidence alone", [])
