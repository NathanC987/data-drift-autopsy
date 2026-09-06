"""CLIP embedding extractor with an image encoder and a text encoder.

Unlike the ResNet extractor this backbone is multimodal: ``extract`` returns
image embeddings and ``encode_text`` returns embeddings for natural-language
phrases in the same space. The text side is what the concept-level root cause
analyser (``drift_autopsy.rca.concept_probe``) uses to name a drift direction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from drift_autopsy.core.extractor import BaseFeatureExtractor
from drift_autopsy.registry import ExtractorRegistry

try:
    import torch  # type: ignore[import-not-found]
    import open_clip  # type: ignore[import-not-found]
    from PIL import Image

    _HAS_OPEN_CLIP = True
    _OPEN_CLIP_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - handled via runtime guard
    _HAS_OPEN_CLIP = False
    _OPEN_CLIP_IMPORT_ERROR = exc


# Prompt ensemble used when encoding a bare concept phrase. Averaging the
# embeddings of several templated forms is standard CLIP zero-shot practice and
# noticeably steadies the concept vectors.
DEFAULT_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of {}",
    "a picture of {}",
    "an image of {}",
    "a cropped photo of {}",
    "{}",
)


def _default_device() -> str:
    if _HAS_OPEN_CLIP and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@ExtractorRegistry.register("clip")
class CLIPEmbeddingExtractor(BaseFeatureExtractor):
    """Extract image and text embeddings from an OpenCLIP backbone."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        batch_size: int = 64,
        prompt_templates: Optional[Sequence[str]] = None,
    ):
        if not _HAS_OPEN_CLIP:
            raise ImportError(
                "CLIPEmbeddingExtractor requires open_clip_torch, torch and Pillow. "
                "Install with: pip install 'drift-autopsy[concept]'"
            ) from _OPEN_CLIP_IMPORT_ERROR

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        try:
            embed_dim = int(open_clip.get_model_config(model_name)["embed_dim"])
        except Exception as exc:  # pragma: no cover - depends on open_clip version
            raise ValueError(f"Unknown or unsupported CLIP model_name '{model_name}'") from exc

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or _default_device()
        self.batch_size = int(batch_size)
        self.prompt_templates = tuple(prompt_templates) if prompt_templates else DEFAULT_PROMPT_TEMPLATES

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)

        super().__init__(name=f"clip::{model_name}::{pretrained}", embedding_dim=embed_dim)

    def _load_image_tensor(self, image_path: str):
        with Image.open(Path(image_path)) as image:
            return self._preprocess(image.convert("RGB"))

    def extract(self, image_paths: List[str]) -> np.ndarray:
        """Return L2-normalised image embeddings, one row per path."""
        if not image_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        batches = []
        with torch.no_grad():
            for start in range(0, len(image_paths), self.batch_size):
                chunk = image_paths[start : start + self.batch_size]
                tensors = torch.stack([self._load_image_tensor(p) for p in chunk]).to(self.device)
                features = self._model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
                batches.append(features.detach().cpu().numpy().astype(np.float32))

        return np.vstack(batches)

    def encode_text(self, phrases: Sequence[str]) -> np.ndarray:
        """Return one L2-normalised embedding per phrase, prompt-ensembled."""
        phrases = list(phrases)
        if not phrases:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors = []
        with torch.no_grad():
            for phrase in phrases:
                tokens = self._tokenizer([t.format(phrase) for t in self.prompt_templates]).to(self.device)
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                pooled = features.mean(dim=0)
                pooled = pooled / pooled.norm()
                vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

        return np.vstack(vectors)

    def get_extraction_metadata(self) -> Dict[str, object]:
        metadata = super().get_extraction_metadata()
        metadata.update(
            {
                "model_name": self.model_name,
                "pretrained": self.pretrained,
                "device": self.device,
                "batch_size": self.batch_size,
                "prompt_templates": list(self.prompt_templates),
            }
        )
        if _HAS_OPEN_CLIP:
            metadata["open_clip_version"] = open_clip.__version__
            metadata["torch_version"] = torch.__version__
        return metadata
