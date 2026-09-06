"""Embedding extractor implementations."""

from drift_autopsy.extractors.resnet import ResNetEmbeddingExtractor

__all__ = ["ResNetEmbeddingExtractor"]

# CLIP is an optional backbone (drift-autopsy[concept]); importing it must never
# break the package when open_clip is absent. Registration is a side effect of
# the import.
try:  # pragma: no cover - exercised indirectly via the concept extra
    from drift_autopsy.extractors.clip import CLIPEmbeddingExtractor

    __all__.append("CLIPEmbeddingExtractor")
except Exception:
    pass
