"""Class activation maps for a linear head on a frozen ResNet backbone.

The monitored CLEAR-10 model is a standardised logistic regression on the
global-average-pooled layer-4 features of a frozen ResNet. For an image with
layer-4 maps A_j(u, v), the pooled feature is e_j = mean_uv A_j(u, v) and the
class-c logit is sum_j (w_cj / s_j) e_j + const, with w the regression
coefficients and s the scaler's standard deviations. The logit is therefore
exactly linear in the pooled maps, so the class activation map

    CAM_c(u, v) = sum_j (w_cj / s_j) A_j(u, v)

(Zhou et al., 2016) averages to the logit up to a constant. Grad-CAM
(Selvaraju et al., 2017) reduces to the same map for this architecture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def _linear_head(model: Any):
    scaler = model.named_steps["scale"]
    clf = model.named_steps["classifier"]
    return clf.coef_ / scaler.scale_[None, :], clf.classes_


def class_activation_maps(
    image_paths: Sequence[str],
    model: Any,
    target_classes: Sequence[Any],
    stored_embeddings: np.ndarray,
    weights_name: str = "IMAGENET1K_V1",
    device: str | None = None,
    rtol: float = 1e-3,
) -> List[Dict[str, Any]]:
    """CAM for each image and its target class, upsampled to the input crop.

    ``stored_embeddings`` are the embeddings the model was trained and scored
    on; the pooled layer-4 maps must reproduce them to within ``rtol`` of the
    embedding's largest magnitude (float32 and cuDNN rounding differ by about
    1e-4), which checks that the maps come from the same preprocessing and
    backbone.
    """
    import torch
    from PIL import Image
    from torchvision import models

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = getattr(models.ResNet18_Weights, weights_name)
    backbone = models.resnet18(weights=weights).eval().to(device)
    transform = weights.transforms()
    body = torch.nn.Sequential(*list(backbone.children())[:-2])   # up to layer4
    head, classes = _linear_head(model)
    class_index = {c: i for i, c in enumerate(classes)}

    out = []
    with torch.no_grad():
        for path, target, stored in zip(image_paths, target_classes, stored_embeddings):
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            x = transform(rgb).unsqueeze(0).to(device)
            maps = body(x)[0].cpu().numpy()                      # (512, h, w)
            pooled = maps.mean(axis=(1, 2))
            if np.abs(pooled - stored).max() > rtol * np.abs(stored).max():
                raise ValueError(f"pooled maps do not reproduce the stored embedding for {path}")
            cam = np.tensordot(head[class_index[target]], maps, axes=1)   # (h, w)
            size = x.shape[-1]
            cam_t = torch.nn.functional.interpolate(torch.tensor(cam)[None, None], size=(size, size),
                                                    mode="bilinear", align_corners=False)[0, 0].numpy()
            crop = x[0].cpu().numpy().transpose(1, 2, 0)
            mean, std = np.array(transform.mean), np.array(transform.std)
            out.append({
                "image_path": str(path),
                "target_class": target,
                "cam": cam_t,
                "image": np.clip(crop * std + mean, 0, 1),
            })
    return out
