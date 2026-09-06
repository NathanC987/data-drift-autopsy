"""Unit tests for the CLIP embedding extractor and concept-probe helpers."""

import numpy as np
import pytest

from drift_autopsy.registry import ExtractorRegistry


def test_clip_extractor_is_registered():
    import drift_autopsy.extractors  # noqa: F401  (registration side effect)

    assert "clip" in ExtractorRegistry.list()


def test_clip_dependency_guard_or_basic_behaviour():
    from drift_autopsy.extractors.clip import CLIPEmbeddingExtractor, _HAS_OPEN_CLIP

    if not _HAS_OPEN_CLIP:
        with pytest.raises(ImportError, match=r"drift-autopsy\[concept\]"):
            CLIPEmbeddingExtractor()
        return

    ex = CLIPEmbeddingExtractor(batch_size=8)
    assert ex.embedding_dim == 512

    assert ex.extract([]).shape == (0, 512)

    text = ex.encode_text(["a photo of a dog", "a photo of a car"])
    assert text.shape == (2, 512)
    np.testing.assert_allclose(np.linalg.norm(text, axis=1), 1.0, atol=1e-4)

    meta = ex.get_extraction_metadata()
    assert meta["model_name"] == "ViT-B-32"
    assert "open_clip_version" in meta


@pytest.mark.skipif(
    not __import__("drift_autopsy.extractors.clip", fromlist=["_HAS_OPEN_CLIP"])._HAS_OPEN_CLIP,
    reason="open_clip not installed",
)
def test_clip_image_text_alignment(tmp_path):
    from PIL import Image

    from drift_autopsy.extractors.clip import CLIPEmbeddingExtractor

    # solid red and solid blue frames — CLIP should still separate the colour words
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    Image.new("RGB", (64, 64), (220, 20, 20)).save(red)
    Image.new("RGB", (64, 64), (20, 20, 220)).save(blue)

    ex = CLIPEmbeddingExtractor(batch_size=2)
    imgs = ex.extract([str(red), str(blue)])
    words = ex.encode_text(["a red image", "a blue image"])
    sim = imgs @ words.T
    assert sim[0, 0] > sim[0, 1]  # red image closer to "a red image"
    assert sim[1, 1] > sim[1, 0]  # blue image closer to "a blue image"


def test_concept_probe_ranking_shapes():
    from drift_autopsy.rca import concept_probe as cp

    rng = np.random.default_rng(0)
    concept_matrix = cp._l2(rng.normal(size=(6, 16)))
    names = [f"c{i}" for i in range(6)]
    ref = cp._l2(rng.normal(size=(50, 16)))
    prod = cp._l2(rng.normal(loc=0.3, size=(50, 16)))

    ranked = cp.rank_concepts(ref, prod, concept_matrix, names)
    assert len(ranked) == 6
    assert {"concept", "delta", "ref_mass", "prod_mass", "alignment"} <= ranked[0].keys()
    assert ranked[0]["delta"] >= ranked[-1]["delta"]
    assert abs(sum(r["delta"] for r in ranked)) < 1e-6  # mass shift sums to zero
