"""Unit tests for ResNet embedding extractor dependency and metadata behavior."""

import uuid

import pytest

from drift_autopsy.registry import ExtractorRegistry


def test_resnet_extractor_is_registered():
    # Import triggers registration side effect.
    from drift_autopsy.extractors import ResNetEmbeddingExtractor  # noqa: F401

    assert "resnet" in ExtractorRegistry.list()


def test_resnet_extractor_dependency_guard_or_basic_metadata():
    from drift_autopsy.extractors.resnet import ResNetEmbeddingExtractor, _HAS_TORCH

    if not _HAS_TORCH:
        with pytest.raises(ImportError, match=r"drift-autopsy\[image\]"):
            ResNetEmbeddingExtractor()
        return

    extractor = ResNetEmbeddingExtractor(model_name="resnet18", weights=None, batch_size=2)
    metadata = extractor.get_extraction_metadata()

    assert extractor.embedding_dim == 512
    assert metadata["model_name"] == "resnet18"
    assert metadata["weights"] is None

    empty_embeddings = extractor.extract([])
    assert empty_embeddings.shape == (0, extractor.embedding_dim)


def test_custom_extractor_registration_with_unique_name():
    class InlineExtractor:
        def __init__(self, flag: bool = False):
            self.flag = flag

    name = f"inline_extractor_{uuid.uuid4().hex}"
    ExtractorRegistry.register(name)(InlineExtractor)
    created = ExtractorRegistry.create(name, flag=True)

    assert isinstance(created, InlineExtractor)
    assert created.flag is True
