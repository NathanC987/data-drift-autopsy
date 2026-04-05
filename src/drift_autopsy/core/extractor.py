"""Base protocol and helpers for embedding extractors."""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class FeatureExtractor(ABC):
    """Protocol-like base class for embedding extractors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return extractor name/identifier."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return output embedding dimensionality."""

    @abstractmethod
    def extract(self, image_paths: List[str]) -> np.ndarray:
        """Extract embeddings for a list of image paths."""

    @abstractmethod
    def get_extraction_metadata(self) -> Dict[str, object]:
        """Return reproducibility metadata for this extractor setup."""


class BaseFeatureExtractor(FeatureExtractor):
    """Base extractor with shared metadata behavior."""

    def __init__(self, name: str, embedding_dim: int):
        self._name = name
        self._embedding_dim = int(embedding_dim)

    @property
    def name(self) -> str:
        return self._name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def get_extraction_metadata(self) -> Dict[str, object]:
        return {
            "extractor": self.name,
            "embedding_dim": self.embedding_dim,
        }
