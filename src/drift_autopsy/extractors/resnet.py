"""ResNet embedding extractor with optional torch backend."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from drift_autopsy.core.extractor import BaseFeatureExtractor
from drift_autopsy.registry import ExtractorRegistry

try:
    import torch  # type: ignore[import-not-found]
    from PIL import Image
    from torchvision import models  # type: ignore[import-not-found]
    from torchvision import transforms as T  # type: ignore[import-not-found]

    _HAS_TORCH = True
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - handled via runtime guard
    _HAS_TORCH = False
    _TORCH_IMPORT_ERROR = exc


def _default_device() -> str:
    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@ExtractorRegistry.register("resnet")
class ResNetEmbeddingExtractor(BaseFeatureExtractor):
    """Extract image embeddings from a torchvision ResNet backbone."""

    _SUPPORTED_MODELS = {
        "resnet18": ("ResNet18_Weights", 512),
        "resnet34": ("ResNet34_Weights", 512),
        "resnet50": ("ResNet50_Weights", 2048),
    }

    def __init__(
        self,
        model_name: str = "resnet18",
        weights: Optional[str] = "IMAGENET1K_V1",
        device: Optional[str] = None,
        batch_size: int = 32,
        image_size: int = 224,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                "ResNetEmbeddingExtractor requires torch, torchvision, and Pillow. "
                "Install with: pip install 'drift-autopsy[image]'"
            ) from _TORCH_IMPORT_ERROR

        if model_name not in self._SUPPORTED_MODELS:
            supported = ", ".join(sorted(self._SUPPORTED_MODELS.keys()))
            raise ValueError(f"Unsupported model_name '{model_name}'. Supported: {supported}")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.model_name = model_name
        self.weights_name = weights
        self.device = device or _default_device()
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)

        weights_cls_name, embedding_dim = self._SUPPORTED_MODELS[model_name]
        self._weights = self._resolve_weights(model_name, weights_cls_name, weights)
        self._transform = self._build_transform(self._weights)
        self._backbone = self._build_backbone(model_name, self._weights).to(self.device).eval()

        super().__init__(name=f"resnet::{model_name}", embedding_dim=embedding_dim)

    def _resolve_weights(self, model_name: str, weights_cls_name: str, weights_name: Optional[str]):
        if not weights_name:
            return None
        weights_cls = getattr(models, weights_cls_name)
        try:
            return getattr(weights_cls, weights_name)
        except AttributeError as exc:
            raise ValueError(
                f"Unknown weights '{weights_name}' for {model_name}."
            ) from exc

    def _build_transform(self, weights):
        if weights is not None:
            return weights.transforms()

        return T.Compose(
            [
                T.Resize(self.image_size + 32),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _build_backbone(self, model_name: str, weights):
        model_ctor = getattr(models, model_name)
        model = model_ctor(weights=weights)
        model.fc = torch.nn.Identity()
        return model

    def _load_image_tensor(self, image_path: str):
        path = Path(image_path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return self._transform(rgb)

    def extract(self, image_paths: List[str]) -> np.ndarray:
        if not image_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        batches = []
        with torch.no_grad():
            for start in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[start : start + self.batch_size]
                image_tensors = [self._load_image_tensor(path) for path in batch_paths]
                batch_tensor = torch.stack(image_tensors).to(self.device)
                embeddings = self._backbone(batch_tensor)
                batches.append(embeddings.detach().cpu().numpy().astype(np.float32))

        return np.vstack(batches)

    def get_extraction_metadata(self) -> Dict[str, object]:
        metadata = super().get_extraction_metadata()
        metadata.update(
            {
                "model_name": self.model_name,
                "weights": self.weights_name,
                "device": self.device,
                "batch_size": self.batch_size,
                "image_size": self.image_size,
            }
        )

        if _HAS_TORCH:
            metadata["torch_version"] = torch.__version__
        return metadata
