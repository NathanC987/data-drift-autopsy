"""PCA reconstruction-error detector for embedding drift."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from drift_autopsy import gpu
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import permutation_test, severity_from_p_value

logger = logging.getLogger(__name__)


def reconstruction_increase(X: np.ndarray, Y: np.ndarray, explained_variance_ratio: float,
                            random_state: int) -> float:
    """Relative rise in mean squared PCA reconstruction error of Y over held-out X.

    X is split at random into halves. PCA (keeping ``explained_variance_ratio``
    of the variance) is fitted on the standardised first half; the reference
    error is measured on the second half, so both errors are out of sample and
    the statistic is near zero when Y comes from the reference distribution.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    fit, hold = X[idx[: len(X) // 2]], X[idx[len(X) // 2:]]
    mean = fit.mean(axis=0)
    std = fit.std(axis=0)
    std[std == 0] = 1.0
    pca = PCA(n_components=explained_variance_ratio, svd_solver="full", random_state=random_state)
    pca.fit((fit - mean) / std)

    def err(Z):
        Zn = (Z - mean) / std
        return float(np.mean((Zn - pca.inverse_transform(pca.transform(Zn))) ** 2))

    ref = err(hold)
    return (err(Y) - ref) / max(ref, 1e-12)


def reconstruction_increase_batch(Xb, Yb, explained_variance_ratio: float, random_state: int):
    """GPU form of :func:`reconstruction_increase` for stacked samples (torch, float64).

    Same split of X (the generator is re-seeded with ``random_state`` for every
    call, as in the CPU form) and the same standardisation. The principal axes
    come from a batched eigendecomposition of the centred scatter matrix, which
    spans the same subspace as the SVD scikit-learn uses, and the number of
    components is chosen exactly as scikit-learn does for a float
    ``n_components`` (the smallest k whose cumulative explained variance ratio
    exceeds it)."""
    torch = gpu.torch_module()
    n = Xb.shape[1]
    idx = np.random.default_rng(random_state).permutation(n)
    fit = Xb[:, torch.as_tensor(idx[: n // 2], device=Xb.device)]
    hold = Xb[:, torch.as_tensor(idx[n // 2:], device=Xb.device)]
    mean = fit.mean(dim=1, keepdim=True)
    std = fit.std(dim=1, unbiased=False, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    F = (fit - mean) / std
    pca_mean = F.mean(dim=1, keepdim=True)
    Fc = F - pca_mean
    w, V = torch.linalg.eigh(Fc.transpose(1, 2) @ Fc)          # ascending eigenvalues
    w, V = w.flip(-1).clamp(min=0), V.flip(-1)
    cumulative = torch.cumsum(w / w.sum(dim=1, keepdim=True), dim=1).cpu().numpy()
    k = [int(np.searchsorted(c, explained_variance_ratio, side="right") + 1) for c in cumulative]
    keep = torch.zeros_like(w)
    for i, ki in enumerate(k):
        keep[i, :ki] = 1.0
    P = V @ torch.diag_embed(keep) @ V.transpose(1, 2)          # projector onto the kept axes

    def err(Z):
        Zc = (Z - mean) / std - pca_mean
        return ((Zc - Zc @ P) ** 2).mean(dim=(1, 2))

    ref = err(hold)
    return (err(Yb) - ref) / torch.clamp(ref, min=1e-12)


@DetectorRegistry.register("pca_reconstruction")
class PCAReconstructionError(BaseDriftDetector):
    """Relative rise in PCA reconstruction error, with a permutation test.

    Args:
        threshold: Significance level of the permutation test (default: 0.05)
        explained_variance_ratio: Variance kept by the PCA (default: 0.95)
        n_permutations: Permutations for the p-value (default: 200)
        max_samples: Per-sample cap; larger samples are randomly subsampled (default: 3000)
        random_state: Seed for subsampling, PCA and permutations (default: 42)
    """

    def __init__(self, threshold: float = 0.05, explained_variance_ratio: float = 0.95,
                 n_permutations: int = 200, max_samples: int = 3000, random_state: int = 42):
        super().__init__(name="pca_reconstruction")
        self.threshold = threshold
        self.explained_variance_ratio = explained_variance_ratio
        self.n_permutations = n_permutations
        self.max_samples = max_samples
        self.random_state = random_state

    def _matrix(self, dataset: Dataset, rng: np.random.Generator) -> np.ndarray:
        X = dataset.to_pandas()[dataset.numeric_features].to_numpy(dtype=float)
        if len(X) > self.max_samples:
            X = X[rng.choice(len(X), self.max_samples, replace=False)]
        return X

    def detect(self, test_data: Dataset) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        rng = np.random.default_rng(self.random_state)
        X = self._matrix(self._reference_data, rng)
        Y = self._matrix(test_data, rng)
        test = None
        if gpu.gpu_available():
            try:
                with gpu.gpu_slot():
                    test = gpu.permutation_test_gpu(
                        lambda a, b: reconstruction_increase_batch(a, b, self.explained_variance_ratio,
                                                                   self.random_state),
                        X, Y, self.n_permutations, random_state=self.random_state)
            except RuntimeError as error:
                if not gpu.is_oom(error):
                    raise
                gpu.release()
        if test is None:
            test = permutation_test(
                lambda a, b: reconstruction_increase(a, b, self.explained_variance_ratio, self.random_state),
                X, Y, self.n_permutations, random_state=self.random_state,
            )
            test["device"] = "cpu"
        drift = test["p_value"] < self.threshold
        return DetectionResult(
            detector_name=self.name,
            drift_detected=bool(drift),
            severity=severity_from_p_value(test["p_value"], self.threshold),
            score=test["statistic"],
            threshold=self.threshold,
            p_value=test["p_value"],
            statistic=test["statistic"],
            metadata={"n_permutations": self.n_permutations, "n_reference": len(X), "n_test": len(Y),
                      "explained_variance_ratio": self.explained_variance_ratio, "device": test["device"]},
        )
