"""Embedding extractor implementations."""

from drift_autopsy.extractors.resnet import ResNetEmbeddingExtractor

# CLIP is an optional backbone (drift-autopsy[concept]). clip.py guards its own
# optional imports and raises only when the extractor is used without open_clip,
# so importing it here is always safe. Registration is a side effect of the import.
from drift_autopsy.extractors.clip import CLIPEmbeddingExtractor

__all__ = ["ResNetEmbeddingExtractor", "CLIPEmbeddingExtractor"]
