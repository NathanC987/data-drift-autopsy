"""Grad-CAM utilities for Chest X-ray CNN root-cause analysis."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    device: str | torch.device = "cpu",
    target_layer_index: int = 6,
) -> Tuple[np.ndarray, float]:
    """Compute Grad-CAM heatmap and predicted positive probability for a single image."""
    model.to(device)
    model.eval()

    x = image.unsqueeze(0) if image.ndim == 3 else image
    x = x.to(device)
    _, _, h, w = x.shape

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    target_layer = model.features[target_layer_index]

    def _fwd_hook(_module, _inp, out):
        activations.append(out.detach())

    def _bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(_fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        model.zero_grad()
        logit = model(x).squeeze()
        prob = float(torch.sigmoid(logit).item())
        target = logit if prob >= 0.5 else -logit
        target.backward()

        act = activations[0][0]  # (C, H', W')
        grad = gradients[0][0]   # (C, H', W')
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=0))
        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        max_val = float(np.max(cam_up))
        p99_val = float(np.percentile(cam_up, 99))
        scale = p99_val if p99_val > 1e-8 else max_val
        if scale > 1e-8:
            heatmap = np.clip(cam_up / scale, 0.0, 1.0).astype(np.float32)
        else:
            heatmap = np.zeros((h, w), dtype=np.float32)
    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    return heatmap, prob


def heatmap_to_rgb(heatmap: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] 2D heatmap to an RGB float32 array in [0, 1] using a jet-like ramp."""
    h = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def gradcam_overlay(image: torch.Tensor | np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a grayscale image and Grad-CAM heatmap into an RGB [0, 1] array."""
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)
    if img_np.ndim == 3 and img_np.shape[0] == 1:
        img_np = img_np[0]
    elif img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np[..., 0]
    img_np = np.clip(img_np.astype(np.float32), 0.0, 1.0)
    base_rgb = np.stack([img_np, img_np, img_np], axis=-1)
    heat_rgb = heatmap_to_rgb(heatmap)
    return np.clip((1.0 - alpha) * base_rgb + alpha * heat_rgb, 0.0, 1.0)


def weighted_focus_score(heatmap: np.ndarray, top_percent: float = 10.0) -> float:
    """Return the mean activation of the top `top_percent`% pixels in the heatmap."""
    flat = np.asarray(heatmap, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0
    k = max(1, int(np.ceil(flat.size * (top_percent / 100.0))))
    top_vals = np.partition(flat, -k)[-k:]
    return float(np.mean(top_vals))
