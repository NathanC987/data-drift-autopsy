"""Published diagnosis methods run on the same windows as ours (BNRC revision).

Each function names a drift type from one window, in the vocabulary of
:mod:`drift_autopsy.diagnosis_baselines` (none, covariate, prior, concept,
label_noise), and returns its raw test results next to the verdict.

* :func:`roschewitz_identification` - Roschewitz et al. (MICCAI 2025): shift
  detection with "Duo" (BBSD, a Kolmogorov-Smirnov test per class
  probability, and an MMD permutation test, Bonferroni over the C + 1
  p-values), then shift identification: the test prevalence is estimated
  by class probability matching (CPMCN, Wen et al.), the reference is
  resampled to that prevalence and MMD is run again. No remaining shift:
  prevalence shift. Remaining shift: covariate shift, or mixed shift when
  BBSD rejected before the resampling and not after it. It uses no labels,
  so it cannot name concept drift or label noise. Tabular adaptation: the
  MMD runs on the standardized inputs (the image embeddings for CLEAR-10)
  instead of an image encoder's principal components; a mixed shift is
  scored as covariate shift, since its inputs moved beyond the prevalence.
* :func:`detectshift_diagnosis` - DetectShift (Maia Polo et al., 2023), with
  the authors' package: KL divergences estimated by probabilistic classifiers
  (CatBoost, nominal codes as numbers) and tested by permutation (total, label and covariate shift),
  local permutation (P(X|Y)) and conditional randomization (P(Y|X)). It needs
  labels from both domains: the reference calibration sample and the same
  audit labels our audit stage receives, each split in half to fit the
  classifiers and to test. Holm over the label, covariate and two
  conditional tests; a label shift without a P(X|Y) shift is prior shift
  (P(Y|X) moves with it), else a P(Y|X) shift is concept drift (the
  framework has no label noise type), else any input shift is covariate
  shift.
* :func:`confident_learning_audit` - confident learning (Northcutt et al.,
  2021) with cleanlab: the share of audit labels flagged as label issues
  against the share in the reference calibration sample (one-sided
  two-proportion test). Used in place of our audit stage: when the
  label-free stage finds nothing, a significant rise is read as label noise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.pipeline import DriftPipeline
from drift_autopsy.detectors import MMD
from drift_autopsy.stats import holm_reject, two_proportion_test


# --------------------------------------------------------------------------- #
# Roschewitz et al. (2025)

def _class_columns(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    return proba[:, None] if proba.ndim == 1 else proba


def _bbsd_p_values(ref_proba: np.ndarray, prod_proba: np.ndarray) -> np.ndarray:
    """One KS p-value per class probability (C values, as in the paper)."""
    ref, prod = _class_columns(ref_proba), _class_columns(prod_proba)
    return np.array([stats.ks_2samp(ref[:, k], prod[:, k]).pvalue for k in range(ref.shape[1])])


def cpmcn_prevalence(ref_prior: np.ndarray, prod_proba: np.ndarray) -> np.ndarray:
    """Test prevalence by class probability matching (Roschewitz et al., Eqs. 3 and 4):
    w minimizes sum_i (P_ref(i) - mean_x p(i|x) / sum_j w_j p(j|x))^2 and
    P_test(i) = w_i P_ref(i), normalized."""
    p = _class_columns(prod_proba)
    ref_prior = np.asarray(ref_prior, dtype=float)

    def residual(w):
        return ref_prior - np.mean(p / np.maximum(p @ w, 1e-12)[:, None], axis=0)

    sol = optimize.least_squares(residual, np.ones(len(ref_prior)), bounds=(1e-6, np.inf))
    q = sol.x * ref_prior
    return q / q.sum()


def _resample_to_prevalence(target: np.ndarray, classes: Sequence[Any], prevalence: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of the largest class-stratified reference subsample (without
    replacement) whose class shares match ``prevalence``."""
    counts = np.array([np.sum(target == c) for c in classes], dtype=float)
    keep_frac = np.where(prevalence > 0, counts / np.maximum(prevalence, 1e-12), np.inf)
    n = float(np.min(keep_frac))
    mask = np.zeros(len(target), dtype=bool)
    for c, share in zip(classes, prevalence):
        idx = np.flatnonzero(target == c)
        k = min(len(idx), int(np.floor(share * n)))
        if k:
            mask[rng.choice(idx, k, replace=False)] = True
    return mask


def roschewitz_identification(reference: Dataset, production: Dataset, alpha: float = 0.05,
                              n_permutations: int = 1000, max_samples: int = 3000, seed: int = 0,
                              mmd_p: Optional[float] = None) -> Dict[str, Any]:
    """Shift detection and identification of Roschewitz et al. (2025); see module docstring.
    ``mmd_p`` reuses an MMD test already run on the same pair with the same settings."""
    classes = list(np.unique(np.asarray(reference.target)))
    ref_y = np.asarray(reference.target)
    ref_proba, prod_proba = reference.prediction_probabilities, production.prediction_probabilities

    def mmd(ref: Dataset) -> float:
        res = MMD(threshold=alpha, n_permutations=n_permutations, max_samples=max_samples,
                  standardize=True, random_state=seed).fit_detect(ref, production)
        return float(res.p_value)

    bbsd = _bbsd_p_values(ref_proba, prod_proba)
    mmd_before = mmd(reference) if mmd_p is None else float(mmd_p)
    detection_p = min(1.0, (len(bbsd) + 1) * float(min(bbsd.min(), mmd_before)))
    out: Dict[str, Any] = {"bbsd_p": bbsd.tolist(), "mmd_p": mmd_before, "duo_p": detection_p}
    if detection_p >= alpha:
        return {**out, "shift": "none", "verdict": "none"}

    ref_prior = np.array([np.mean(ref_y == c) for c in classes])
    prevalence = cpmcn_prevalence(ref_prior, prod_proba)
    mask = _resample_to_prevalence(ref_y, classes, prevalence, np.random.default_rng(seed))
    adjusted = DriftPipeline._slice_dataset(reference, mask)
    mmd_after = mmd(adjusted)
    out.update({"estimated_prevalence": prevalence.tolist(), "reference_prevalence": ref_prior.tolist(),
                "n_adjusted_reference": int(mask.sum()), "mmd_p_adjusted": mmd_after})
    if mmd_after >= alpha:
        return {**out, "shift": "prevalence", "verdict": "prior"}
    bbsd_after = _bbsd_p_values(adjusted.prediction_probabilities, prod_proba)
    before = min(1.0, len(bbsd) * float(bbsd.min())) < alpha
    after = min(1.0, len(bbsd_after) * float(bbsd_after.min())) < alpha
    out["bbsd_p_adjusted"] = bbsd_after.tolist()
    shift = "mixed" if before and not after else "covariate"
    return {**out, "shift": shift, "verdict": "covariate"}


# --------------------------------------------------------------------------- #
# DetectShift (Maia Polo et al., 2023)

def _kl_multinomial(ys, yt):
    """The package's label-shift KL with 0 log 0 = 0 (a class absent from the
    target sample would otherwise give nan)."""
    ys, yt = np.asarray(ys).squeeze(), np.asarray(yt).squeeze()
    kl = 0.0
    for y in np.unique(ys):
        pt, ps = np.mean(yt == y), np.mean(ys == y)
        if pt > 0:
            kl += pt * np.log(pt / ps)
    return kl


class _RandomShim:
    """np.random for the package: ``normal(0, s, 1)`` gives a scalar, since the
    package takes float() of it, which NumPy 2 refuses for a 1-element array."""

    def __getattr__(self, name):
        return getattr(np.random, name)

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        out = np.random.normal(loc, scale, size)
        return out[0] if size == 1 else out


class _NumpyShim:
    random = _RandomShim()

    def __getattr__(self, name):
        return getattr(np, name)


def _one_hot(y: np.ndarray, classes: Sequence[Any]) -> pd.DataFrame:
    """The package's label encoding (one-hot, first class dropped) over a fixed class list."""
    codes = np.searchsorted(np.asarray(classes), np.asarray(y))
    out = pd.DataFrame({f"y{j}": (codes == j + 1).astype("int32") for j in range(len(classes) - 1)})
    return out


def detectshift_diagnosis(ref_X: pd.DataFrame, ref_y: np.ndarray, tgt_X: pd.DataFrame, tgt_y: np.ndarray,
                          categorical: Sequence[str] = (), alpha: float = 0.05, n_permutations: int = 1000,
                          seed: int = 0) -> Dict[str, Any]:
    """DetectShift on a labeled reference sample and the labeled audit sample; see module docstring."""
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    import detectshift.tests as dst
    from detectshift.cdist import cde_class
    from detectshift.tools import KL

    dst.KL_multinomial = _kl_multinomial
    dst.np = _NumpyShim()

    class _KL(KL):
        def fit(self, Zs, Zt, random_state=0):       # the package's fit with one thread per worker
            self.nt, self.ns = Zt.shape[0], Zs.shape[0]
            Xw = pd.concat([Zt, Zs], axis=0)
            yw = np.hstack((np.ones(self.nt), np.zeros(self.ns)))
            Xa, Xv, ya, yv = train_test_split(Xw, yw, test_size=self.validation_split, random_state=random_state)
            self.model = CatBoostClassifier(loss_function="Logloss", cat_features=self.cat_features,
                                            thread_count=1, random_seed=random_state)
            self.model.fit(Xa, ya, verbose=False, eval_set=(Xv, yv), early_stopping_rounds=100)

    class _CDE(cde_class):
        def fit(self, X, y, random_seed=None):
            from detectshift.tools import get_classes
            self.n_columns = y.shape[1]
            y = get_classes(y)
            Xa, Xv, ya, yv = train_test_split(X, y, test_size=self.validation_split, stratify=y,
                                              random_state=random_seed)
            self.model = CatBoostClassifier(loss_function="MultiClass", cat_features=self.cat_features,
                                            thread_count=1, random_seed=random_seed)
            self.model.fit(Xa, ya, verbose=False, eval_set=(Xv, yv), early_stopping_rounds=100)

        def sample(self, X):                         # one-hot over every class, sampled or not
            ps = self.model.predict_proba(X)
            draw = np.array([np.argmax(np.random.multinomial(1, p / p.sum())) for p in ps])
            codes = np.asarray(self.model.classes_).astype(int)[draw]
            return pd.DataFrame({f"y{j}": (codes == j + 1).astype("int32") for j in range(self.n_columns)})

    classes = sorted(set(np.asarray(ref_y).tolist()) | set(np.asarray(tgt_y).tolist()))
    cols = list(ref_X.columns)
    Xs = ref_X.reset_index(drop=True).copy()
    Xt = tgt_X.reset_index(drop=True).copy()
    for c in categorical:
        Xs[c] = Xs[c].astype(int)
        Xt[c] = Xt[c].astype(int)
    Xs.columns = Xt.columns = [f"x{j}" for j in range(len(cols))]
    ys, yt = _one_hot(ref_y, classes), _one_hot(tgt_y, classes)

    def halves(X, y):
        return train_test_split(X, y, test_size=0.5, random_state=seed)

    Xs_tr, Xs_te, ys_tr, ys_te = halves(Xs, ys)
    Xt_tr, Xt_te, yt_tr, yt_te = halves(Xt, yt)
    Xs_tr, Xs_te, ys_tr, ys_te, Xt_tr, Xt_te, yt_tr, yt_te = (
        d.reset_index(drop=True) for d in (Xs_tr, Xs_te, ys_tr, ys_te, Xt_tr, Xt_te, yt_tr, yt_te))
    # nominal codes enter as numbers: the package's local permutation rebuilds the
    # target frame from a float array, which CatBoost refuses for declared categoricals
    cat = None
    tot = _KL(boost=True, cat_features=cat)
    tot.fit(pd.concat([Xs_tr, ys_tr], axis=1), pd.concat([Xt_tr, yt_tr], axis=1), random_state=seed)
    cov = _KL(boost=True, cat_features=cat)
    cov.fit(Xs_tr, Xt_tr, random_state=seed)
    cd = _CDE(boost=True, cat_features=cat)
    cd.fit(Xs_tr, ys_tr, random_seed=seed)

    np.random.seed(seed)
    res = dst.ShiftDiagnostics(Xs_te, ys_te, Xt_te, yt_te, totshift_model=tot, covshift_model=cov,
                               labshift_model=None, cd_model=cd, task="class", B=n_permutations, verbose=False)
    p = {k: float(res[k]["pval"]) for k in ("tot", "lab", "cov", "conc1", "conc2")}
    names = ("lab", "cov", "conc1", "conc2")
    rejected = dict(zip(names, holm_reject([p[k] for k in names], alpha)))
    if rejected["lab"] and not rejected["conc1"]:
        verdict = "prior"            # label shift with P(X|Y) intact; P(Y|X) moves with it
    elif rejected["conc2"]:
        verdict = "concept"
    elif rejected["cov"] or rejected["conc1"]:
        verdict = "covariate"
    elif rejected["lab"]:
        verdict = "prior"
    else:
        verdict = "none"
    return {"p_values": p, "rejected": {k: bool(v) for k, v in rejected.items()},
            "kl": {k: float(res[k]["kl"]) for k in res}, "verdict": verdict,
            "n_reference": int(len(Xs)), "n_target": int(len(Xt))}


# --------------------------------------------------------------------------- #
# confident learning (Northcutt et al., 2021)

def _label_issue_share(labels: np.ndarray, proba: np.ndarray, classes: Sequence[Any]) -> Dict[str, Any]:
    from cleanlab.filter import find_label_issues

    codes = np.searchsorted(np.asarray(classes), np.asarray(labels))
    proba = _class_columns(proba)
    present = np.unique(codes)
    if len(present) < 2:                   # cleanlab needs two observed classes
        return {"n": int(len(codes)), "issues": 0}
    issues = find_label_issues(labels=codes, pred_probs=proba)
    return {"n": int(len(codes)), "issues": int(np.sum(issues))}


def confident_learning_audit(model: Any, reference: Dataset, audit_X: pd.DataFrame, audit_y: np.ndarray,
                             alpha: float = 0.05) -> Dict[str, Any]:
    """Share of audit labels that confident learning flags, against the reference share."""
    classes = list(model.classes_)
    feats = reference.feature_names
    ref = _label_issue_share(np.asarray(reference.target), reference.prediction_probabilities, classes)
    aud = _label_issue_share(np.asarray(audit_y), model.predict_proba(audit_X[feats].to_numpy(float)), classes)
    test = two_proportion_test(aud["issues"], aud["n"], ref["issues"], ref["n"], alternative="greater")
    return {"reference": ref, "audit": aud, "p_value": float(test["p_value"]),
            "significant": bool(test["p_value"] < alpha)}


def label_free_then_cl(label_free_verdict: str, cl: Dict[str, Any]) -> str:
    """Our label-free stage followed by confident learning in place of our audit."""
    if label_free_verdict != "none":
        return label_free_verdict
    return "label_noise" if cl["significant"] else "none"
