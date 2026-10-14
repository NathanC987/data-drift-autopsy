"""Maximum Mean Discrepancy (MMD) two-sample test for multivariate drift."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import severity_from_p_value

logger = logging.getLogger(__name__)


@DetectorRegistry.register("mmd")
class MMD(BaseDriftDetector):
    """
    Kernel two-sample test (Gretton et al., 2012).

    The statistic is the unbiased MMD^2 between the reference and test samples;
    ``score`` reports sqrt(max(MMD^2, 0)). The decision is a permutation test:
    the pooled sample is randomly split ``n_permutations`` times, and
    p = (1 + #{MMD^2_perm >= MMD^2_obs}) / (1 + n_permutations).
    Drift is declared when p < ``threshold`` (a significance level).

    RBF bandwidth: gamma = 1 / (2 * median squared distance), the median
    heuristic, computed once on a seeded subsample of the pooled data and held
    fixed across permutations, as the test requires.

    Args:
        threshold: Significance level of the permutation test (default: 0.05)
        kernel: "rbf" or "linear"
        gamma: Fixed RBF parameter (default: None, median heuristic)
        n_permutations: Permutations for the p-value (default: 200; smallest
            attainable p is 1 / (n_permutations + 1))
        max_samples: Per-sample cap; larger samples are randomly subsampled
            (default: 3000)
        standardize: Z-score numeric features with the reference mean/std before
            the kernel, so features on large raw scales do not dominate the
            distance (default: False, which keeps raw embedding coordinates)
        random_state: Seed for subsampling, the bandwidth subsample and the
            permutations (default: 42)
    """

    def __init__(
        self,
        threshold: float = 0.05,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        n_permutations: int = 200,
        max_samples: int = 3000,
        standardize: bool = False,
        random_state: Optional[int] = 42,
    ):
        super().__init__(name="mmd")
        if n_permutations < 1:
            raise ValueError("n_permutations must be >= 1")
        if kernel not in ("rbf", "linear"):
            raise ValueError(f"Unknown kernel: {kernel}")
        self.threshold = threshold
        self.kernel = kernel
        self.gamma = gamma
        self.n_permutations = n_permutations
        self.max_samples = max_samples
        self.standardize = standardize
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # feature encoding

    def _encode(self, ref_df, test_df):
        """
        Build the feature matrices the kernel sees.

        Numeric features: missing values imputed with the reference mean, then
        z-scored with the reference mean and spread when ``standardize`` is on.
        Nominal features (``Dataset.categorical_features``): one-hot over the
        reference categories plus one "unseen" column, scaled by 1/sqrt(2) so
        two rows that differ on one nominal feature are exactly 1 apart, the
        same as one reference standard deviation on a numeric feature.
        """
        categorical = set(self._reference_data.categorical_features or [])
        numeric_cols = [
            c for c in ref_df.select_dtypes(include=[np.number]).columns if c not in categorical
        ]
        categorical_cols = [c for c in self._reference_data.feature_names if c in categorical]

        blocks_x, blocks_y = [], []
        if numeric_cols:
            Xn = ref_df[numeric_cols].to_numpy(dtype=float)
            Yn = test_df[numeric_cols].to_numpy(dtype=float)
            mean = np.nanmean(Xn, axis=0)
            Xn = np.where(np.isnan(Xn), mean, Xn)
            Yn = np.where(np.isnan(Yn), mean, Yn)
            if self.standardize:
                std = Xn.std(axis=0)
                std[std == 0] = 1.0
                Xn = (Xn - mean) / std
                Yn = (Yn - mean) / std
            blocks_x.append(Xn)
            blocks_y.append(Yn)
        for col in categorical_cols:
            cats = {c: i for i, c in enumerate(pd.unique(ref_df[col].dropna()))}
            unseen = len(cats)

            def one_hot(values, cats=cats, unseen=unseen):
                idx = np.array([cats.get(v, unseen) for v in values], dtype=int)
                out = np.zeros((len(values), unseen + 1))
                out[np.arange(len(values)), idx] = 1.0 / np.sqrt(2.0)
                return out

            blocks_x.append(one_hot(ref_df[col].to_numpy()))
            blocks_y.append(one_hot(test_df[col].to_numpy()))
        if not blocks_x:
            return np.empty((len(ref_df), 0)), np.empty((len(test_df), 0)), [], []
        return np.hstack(blocks_x), np.hstack(blocks_y), numeric_cols, categorical_cols

    # ------------------------------------------------------------------ #
    # kernel test

    def _pooled_kernel(self, Z: np.ndarray, rng: np.random.RandomState) -> tuple:
        if self.kernel == "linear":
            return Z @ Z.T, None
        gamma = self.gamma
        if gamma is None:
            idx = rng.choice(len(Z), size=min(500, len(Z)), replace=False)
            S = Z[idx]
            sq = np.sum(S ** 2, axis=1)
            d2 = sq[:, None] + sq[None, :] - 2 * S @ S.T
            med = float(np.median(d2[np.triu_indices(len(S), k=1)]))
            gamma = 1.0 / (2.0 * med) if med > 0 else 1.0
        sq = np.sum(Z ** 2, axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
        return np.exp(-gamma * d2), float(gamma)

    @staticmethod
    def _mmd2_unbiased(K: np.ndarray, diag: np.ndarray, in_x: np.ndarray) -> float:
        a = in_x.astype(float)
        b = 1.0 - a
        n, m = a.sum(), b.sum()
        Ka = K @ a
        Kb = K @ b
        s_xx = a @ Ka - diag @ a
        s_yy = b @ Kb - diag @ b
        s_xy = b @ Ka
        return float(s_xx / (n * (n - 1)) + s_yy / (m * (m - 1)) - 2.0 * s_xy / (n * m))

    def fit(self, reference_data: Dataset) -> None:
        super().fit(reference_data)
        logger.info(f"MMD fitted on {reference_data.n_samples} samples")

    def detect(self, test_data: Dataset) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        logger.info(f"Running MMD on {test_data.n_samples} test samples")

        ref_df = self._reference_data.to_pandas()
        test_df = test_data.to_pandas()
        X, Y, numeric_cols, categorical_cols = self._encode(ref_df, test_df)
        if X.shape[1] == 0 or len(X) < 2 or len(Y) < 2:
            logger.warning("MMD needs at least two rows and one usable feature per sample")
            return DetectionResult(
                detector_name=self.name, drift_detected=False, severity=DriftSeverity.NONE,
                score=0.0, threshold=self.threshold,
            )

        rng = np.random.RandomState(self.random_state)
        n_ref_full, n_test_full = len(X), len(Y)
        if len(X) > self.max_samples:
            X = X[rng.choice(len(X), self.max_samples, replace=False)]
        if len(Y) > self.max_samples:
            Y = Y[rng.choice(len(Y), self.max_samples, replace=False)]

        Z = np.vstack([X, Y])
        K, gamma = self._pooled_kernel(Z, rng)
        diag = np.diag(K).copy()
        in_x = np.zeros(len(Z), dtype=bool)
        in_x[: len(X)] = True

        observed = self._mmd2_unbiased(K, diag, in_x)
        exceed = 0
        for _ in range(self.n_permutations):
            exceed += self._mmd2_unbiased(K, diag, rng.permutation(in_x)) >= observed
        p_value = (1 + exceed) / (1 + self.n_permutations)

        drift_detected = p_value < self.threshold
        severity = severity_from_p_value(p_value, self.threshold)
        mmd_value = float(np.sqrt(max(observed, 0.0)))
        logger.info(f"MMD result: drift={drift_detected}, mmd={mmd_value:.4f}, p={p_value:.4f}")

        return DetectionResult(
            detector_name=self.name,
            drift_detected=bool(drift_detected),
            severity=severity,
            score=mmd_value,
            threshold=self.threshold,
            p_value=float(p_value),
            statistic=float(observed),
            metadata={
                "mmd2_unbiased": float(observed),
                "kernel": self.kernel,
                "gamma": gamma,
                "n_permutations": self.n_permutations,
                "min_attainable_p_value": 1.0 / (1 + self.n_permutations),
                "max_samples": self.max_samples,
                "standardized": self.standardize,
                "subsampled": len(X) < n_ref_full or len(Y) < n_test_full,
                "actual_ref_samples": int(len(X)),
                "actual_test_samples": int(len(Y)),
                "n_numeric_features": len(numeric_cols),
                "n_categorical_features": len(categorical_cols),
                "encoded_dimension": int(X.shape[1]),
            },
        )
