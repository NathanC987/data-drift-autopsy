"""Optional GPU execution for the permutation tests (PyTorch, CUDA).

The permutation tests (MMD, Frechet distance, PCA reconstruction) repeat the
same dense linear algebra a few hundred times per window, which a GPU does
much faster than the processor. Everything runs in float64 and the
permutations are drawn on the CPU from the same seeded generator as the CPU
code, so a GPU run makes the same decisions as a CPU run; statistics agree to
floating-point rounding (about 1e-12 relative), which only matters for an
exact tie between a permuted and the observed statistic.

Control:
* ``DRIFT_AUTOPSY_DEVICE=cpu`` forces the CPU path; otherwise CUDA is used
  when PyTorch sees a GPU.
* ``DRIFT_AUTOPSY_GPU_SLOTS`` (default 4) caps how many processes use the GPU
  at once. Parallel benchmark workers wait for a free slot (file locks), so
  20 workers do not exhaust the GPU memory.
* A CUDA out-of-memory error falls back to the CPU path for that call.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

_TORCH = None
_CUDA: Optional[bool] = None


def torch_module():
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


def gpu_available() -> bool:
    """True when the GPU path should be used."""
    global _CUDA
    if os.environ.get("DRIFT_AUTOPSY_DEVICE", "").lower() == "cpu":
        return False
    if _CUDA is None:
        try:
            _CUDA = bool(torch_module().cuda.is_available())
        except Exception:
            _CUDA = False
    return _CUDA


@contextlib.contextmanager
def gpu_slot() -> Iterator[None]:
    """Hold one of DRIFT_AUTOPSY_GPU_SLOTS process-wide GPU slots (blocking)."""
    slots = int(os.environ.get("DRIFT_AUTOPSY_GPU_SLOTS", "4"))
    lock_dir = Path(tempfile.gettempdir())
    handles = [open(lock_dir / f"drift_autopsy_gpu_slot_{i}.lock", "w") for i in range(slots)]
    held = None
    try:
        while held is None:
            for h in handles:
                try:
                    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = h
                    break
                except BlockingIOError:
                    continue
            if held is None:
                time.sleep(0.05)
        yield
    finally:
        if held is not None:
            fcntl.flock(held, fcntl.LOCK_UN)
        for h in handles:
            h.close()


def is_oom(error: BaseException) -> bool:
    torch = torch_module()
    return isinstance(error, getattr(torch.cuda, "OutOfMemoryError", RuntimeError)) or "out of memory" in str(error)


def release() -> None:
    """Return cached GPU memory so other processes can use it."""
    try:
        torch_module().cuda.empty_cache()
    except Exception:
        pass


def permutation_indices(n_total: int, n_permutations: int, random_state: int):
    """The pooled-row permutations drawn by drift_autopsy.stats.permutation_test, in order."""
    import numpy as np

    rng = np.random.default_rng(random_state)
    return [rng.permutation(n_total) for _ in range(n_permutations)]


def permutation_test_gpu(batch_statistic, X, Y, n_permutations: int, random_state: int = 0, chunk: int = 25):
    """GPU twin of drift_autopsy.stats.permutation_test with the same permutations.

    ``batch_statistic(Xb, Yb)`` takes stacked (B, n, d) and (B, m, d) float64
    CUDA tensors and returns B statistics."""
    import numpy as np

    torch = torch_module()
    dev = torch.device("cuda")
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(X)
    Z = torch.as_tensor(np.vstack([X, Y]), dtype=torch.float64, device=dev)
    observed = float(batch_statistic(Z[None, :n], Z[None, n:])[0])
    perms = permutation_indices(len(Z), n_permutations, random_state)
    exceed = 0
    for start in range(0, n_permutations, chunk):
        idx = torch.as_tensor(np.stack(perms[start:start + chunk]), device=dev)
        Zb = Z[idx]
        stats = batch_statistic(Zb[:, :n], Zb[:, n:])
        exceed += int((stats >= observed).sum())
        del Zb, idx, stats
    del Z
    release()
    return {"statistic": observed, "p_value": (1 + exceed) / (1 + n_permutations),
            "n_permutations": int(n_permutations), "device": "cuda"}
