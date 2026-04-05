"""Unit tests for extractor registry behavior."""

import uuid

import pytest

from drift_autopsy.registry import ExtractorRegistry


def test_extractor_registry_register_and_create():
    class DummyExtractor:
        def __init__(self, value: int = 0):
            self.value = value

    name = f"dummy_test_extractor_{uuid.uuid4().hex}"
    ExtractorRegistry.register(name)(DummyExtractor)
    assert name in ExtractorRegistry.list()

    extractor = ExtractorRegistry.create(name, value=7)
    assert isinstance(extractor, DummyExtractor)
    assert extractor.value == 7


def test_extractor_registry_unknown_name():
    with pytest.raises(ValueError, match="Unknown extractor"):
        ExtractorRegistry.create("does_not_exist")
