"""Extract frozen ResNet-18 embeddings for every CLEAR-10 bucket.

One parquet per bucket in outputs/clear10_tabularized_demo/: the 512
global-average-pool features (feature_0 ... feature_511) of an ImageNet
ResNet-18 (torchvision IMAGENET1K_V1 weights and transforms, eval mode) plus
the image metadata (sample_id, class_name, y_true, image_path, timestamp, ...).
No model is fitted here; drift_autopsy.experiments.clear10 trains the
monitored head on these embeddings.

Run once (GPU recommended): python examples/quickstart/clear10_tabularize_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from drift_autopsy.data import Clear10Loader
from drift_autopsy.experiments.clear10 import ANALYSIS_BUCKETS, EMBEDDING_DIR, REFERENCE_BUCKET
from drift_autopsy.extractors import ResNetEmbeddingExtractor
from drift_autopsy.utils import setup_logging

ROOT = Path("data/clear10")


def main() -> None:
    setup_logging(level="INFO")
    extractor = ResNetEmbeddingExtractor(model_name="resnet18", weights="IMAGENET1K_V1", batch_size=64)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    for bucket in [REFERENCE_BUCKET] + ANALYSIS_BUCKETS:
        meta = Clear10Loader.build_bucket_dataframe(root_path=ROOT, bucket=bucket, include_background=True)
        emb = extractor.extract(meta["image_path"].astype(str).tolist())
        features = pd.DataFrame(emb, columns=[f"feature_{i}" for i in range(emb.shape[1])])
        frame = pd.concat([features, meta.reset_index(drop=True)], axis=1)
        frame.to_parquet(EMBEDDING_DIR / f"bucket_{bucket}.parquet", index=False)
        print(f"bucket {bucket}: {len(frame)} images")
    (EMBEDDING_DIR / "extraction_metadata.json").write_text(json.dumps(extractor.get_extraction_metadata(), indent=1))


if __name__ == "__main__":
    main()
