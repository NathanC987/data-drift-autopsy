"""Monitored-model adapters for image-derived tabularization pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

try:
    import torch  # type: ignore[import-not-found]
    from PIL import Image
    from torchvision import models  # type: ignore[import-not-found]
    from torchvision import transforms as T  # type: ignore[import-not-found]

    _HAS_TORCH = True
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - handled by runtime guard
    _HAS_TORCH = False
    _TORCH_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


def _default_device() -> str:
    if _HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MonitoredModelAdapter(Protocol):
    """Protocol for monitored model adapters used during tabularization."""

    model_name: str
    model_params: Dict[str, Any]

    def fit_with_split(
        self,
        reference_df: pd.DataFrame,
        train_fraction: float = 0.7,
        random_state: int = 42,
    ) -> Dict[str, float]:
        ...

    def attach_predictions(self, df: pd.DataFrame, class_count: int) -> pd.DataFrame:
        ...

    def save(self, path: str) -> None:
        ...


class EmbeddingBaselineClassifier:
    """Train and apply a baseline classifier on embedding feature columns."""

    def __init__(self, model_name: str = "logistic_regression", model_params: Dict[str, Any] | None = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = self._build_model(model_name, self.model_params)
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _feature_columns(df: pd.DataFrame, prefix: str = "feature_") -> List[str]:
        cols = [col for col in df.columns if col.startswith(prefix)]
        if not cols:
            raise ValueError("No embedding columns found with prefix 'feature_'")
        return cols

    def _build_model(self, model_name: str, model_params: Dict[str, Any]):
        if model_name != "logistic_regression":
            raise ValueError(
                f"Unsupported baseline model '{model_name}'. Supported: logistic_regression"
            )

        params = {
            "max_iter": 1000,
            "random_state": 42,
        }
        params.update(model_params)
        return LogisticRegression(**params)

    def fit(self, reference_df: pd.DataFrame) -> None:
        """Fit baseline model on a full reference bucket dataframe."""
        if "y_true" not in reference_df.columns:
            raise ValueError("reference_df must include y_true")

        feature_cols = self._feature_columns(reference_df)
        x = reference_df[feature_cols].to_numpy(dtype=float)
        y = reference_df["y_true"].to_numpy(dtype=int)

        self.model.fit(x, y)
        self.classes_ = np.array(self.model.classes_, dtype=int)

    def fit_with_split(
        self,
        reference_df: pd.DataFrame,
        train_fraction: float = 0.7,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """Fit baseline model with train/eval split and return baseline metrics."""
        if "y_true" not in reference_df.columns:
            raise ValueError("reference_df must include y_true")

        feature_cols = self._feature_columns(reference_df)
        x = reference_df[feature_cols].to_numpy(dtype=float)
        y = reference_df["y_true"].to_numpy(dtype=int)

        can_stratify = len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1

        if len(reference_df) < 4 or not can_stratify:
            # Fallback for very small or degenerate class distributions.
            logger.warning(
                "Skipping train/eval split due to low samples or class support; fitting on full reference"
            )
            self.model.fit(x, y)
            self.classes_ = np.array(self.model.classes_, dtype=int)
            preds = self.model.predict(x)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y,
                preds,
                average="macro",
                zero_division=0,
            )
            return {
                "train_samples": float(len(reference_df)),
                "test_samples": float(len(reference_df)),
                "accuracy": float(accuracy_score(y, preds)),
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1),
            }

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=train_fraction,
            random_state=random_state,
            stratify=y,
        )

        self.model.fit(x_train, y_train)
        self.classes_ = np.array(self.model.classes_, dtype=int)

        preds = self.model.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="macro",
            zero_division=0,
        )

        return {
            "train_samples": float(len(x_train)),
            "test_samples": float(len(x_test)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and probabilities from embedding dataframe."""
        if self.classes_ is None:
            raise RuntimeError("Baseline model is not fitted")

        feature_cols = self._feature_columns(df)
        x = df[feature_cols].to_numpy(dtype=float)

        y_pred = self.model.predict(x).astype(int)
        proba = self.model.predict_proba(x).astype(float)
        return y_pred, proba

    def attach_predictions(self, df: pd.DataFrame, class_count: int) -> pd.DataFrame:
        """Return a new dataframe with y_pred and pred_proba_* columns."""
        y_pred, proba = self.predict(df)

        out = df.copy()
        out["y_pred"] = y_pred

        # Ensure stable full class-space probabilities for downstream contract.
        full_proba = np.zeros((len(out), class_count), dtype=float)
        for model_idx, class_id in enumerate(self.classes_):
            if 0 <= int(class_id) < class_count:
                full_proba[:, int(class_id)] = proba[:, model_idx]

        row_sums = full_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        full_proba = full_proba / row_sums

        for class_idx in range(class_count):
            out[f"pred_proba_{class_idx}"] = full_proba[:, class_idx]

        return out

    def save(self, path: str) -> None:
        """Persist baseline model and metadata to disk."""
        payload = {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "classes": self.classes_,
            "model": self.model,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str) -> "EmbeddingBaselineClassifier":
        """Load baseline model from disk."""
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        instance = cls(
            model_name=payload["model_name"],
            model_params=payload["model_params"],
        )
        instance.classes_ = payload["classes"]
        instance.model = payload["model"]
        return instance


class ResNetMonitoredClassifier:
    """ResNet-based monitored classifier operating directly on image paths."""

    _SUPPORTED_MODELS = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
    }

    def __init__(self, model_name: str = "resnet18", model_params: Dict[str, Any] | None = None):
        if not _HAS_TORCH:
            raise ImportError(
                "ResNetMonitoredClassifier requires torch, torchvision, and Pillow. "
                "Install with: pip install 'drift-autopsy[image]'"
            ) from _TORCH_IMPORT_ERROR

        self.model_name = model_name
        self.model_params = model_params or {}
        if model_name not in self._SUPPORTED_MODELS:
            supported = ", ".join(sorted(self._SUPPORTED_MODELS.keys()))
            raise ValueError(f"Unsupported monitored model '{model_name}'. Supported: {supported}")

        self.weights_name = self.model_params.get("weights", "IMAGENET1K_V1")
        self.device = self.model_params.get("device", _default_device())
        self.batch_size = int(self.model_params.get("batch_size", 32))
        self.image_size = int(self.model_params.get("image_size", 224))
        self.epochs = int(self.model_params.get("epochs", 1))
        self.learning_rate = float(self.model_params.get("learning_rate", 1e-3))
        self.freeze_backbone = bool(self.model_params.get("freeze_backbone", True))

        self.classes_: np.ndarray | None = None
        self._weights = self._resolve_weights(model_name, self.weights_name)
        self._transform = self._build_transform(self._weights)
        self.model: torch.nn.Module | None = None

    def _resolve_weights(self, model_name: str, weights_name: Optional[str]):
        if not weights_name:
            return None

        weights_cls_name = self._SUPPORTED_MODELS[model_name]
        weights_cls = getattr(models, weights_cls_name)
        try:
            return getattr(weights_cls, weights_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown weights '{weights_name}' for {model_name}") from exc

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

    def _build_model(self, class_count: int) -> torch.nn.Module:
        model_ctor = getattr(models, self.model_name)
        model = model_ctor(weights=self._weights)

        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, class_count)
        return model.to(self.device)

    def _load_image_tensor(self, image_path: str):
        with Image.open(image_path) as image:
            return self._transform(image.convert("RGB"))

    def _predict_logits(self, image_paths: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Monitored model is not fitted")

        logits_batches = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[start : start + self.batch_size]
                batch_tensor = torch.stack([self._load_image_tensor(path) for path in batch_paths]).to(
                    self.device
                )
                logits = self.model(batch_tensor)
                logits_batches.append(logits.detach().cpu().numpy())

        return np.vstack(logits_batches)

    def fit_with_split(
        self,
        reference_df: pd.DataFrame,
        train_fraction: float = 0.7,
        random_state: int = 42,
    ) -> Dict[str, float]:
        if "y_true" not in reference_df.columns:
            raise ValueError("reference_df must include y_true")
        if "image_path" not in reference_df.columns:
            raise ValueError("reference_df must include image_path")

        y_all = reference_df["y_true"].to_numpy(dtype=int)
        classes = np.array(sorted(np.unique(y_all).tolist()), dtype=int)
        class_to_idx = {int(class_id): idx for idx, class_id in enumerate(classes)}
        self.classes_ = classes

        split_df = reference_df.reset_index(drop=True)
        can_stratify = len(classes) > 1 and np.min(np.bincount(y_all)) > 1
        if len(split_df) < 4 or not can_stratify:
            train_df = split_df
            test_df = split_df
        else:
            train_df, test_df = train_test_split(
                split_df,
                train_size=train_fraction,
                random_state=random_state,
                stratify=split_df["y_true"],
            )

        self.model = self._build_model(class_count=len(classes))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )

        train_paths = train_df["image_path"].astype(str).tolist()
        train_targets = np.array([class_to_idx[int(y)] for y in train_df["y_true"].to_numpy(dtype=int)], dtype=int)

        self.model.train()
        for _ in range(self.epochs):
            for start in range(0, len(train_paths), self.batch_size):
                batch_paths = train_paths[start : start + self.batch_size]
                batch_targets = train_targets[start : start + self.batch_size]

                batch_tensor = torch.stack([self._load_image_tensor(path) for path in batch_paths]).to(self.device)
                target_tensor = torch.tensor(batch_targets, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = self.model(batch_tensor)
                loss = criterion(logits, target_tensor)
                loss.backward()
                optimizer.step()

        test_paths = test_df["image_path"].astype(str).tolist()
        test_true = test_df["y_true"].to_numpy(dtype=int)

        test_logits = self._predict_logits(test_paths)
        test_pred_idx = np.argmax(test_logits, axis=1)
        idx_to_class = {idx: int(class_id) for idx, class_id in enumerate(classes)}
        test_pred = np.array([idx_to_class[idx] for idx in test_pred_idx], dtype=int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            test_true,
            test_pred,
            average="macro",
            zero_division=0,
        )

        return {
            "train_samples": float(len(train_df)),
            "test_samples": float(len(test_df)),
            "accuracy": float(accuracy_score(test_true, test_pred)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

    def attach_predictions(self, df: pd.DataFrame, class_count: int) -> pd.DataFrame:
        if self.classes_ is None:
            raise RuntimeError("Monitored model is not fitted")
        if "image_path" not in df.columns:
            raise ValueError("df must include image_path")

        paths = df["image_path"].astype(str).tolist()
        logits = self._predict_logits(paths)

        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        pred_idx = np.argmax(probs, axis=1)
        y_pred = np.array([int(self.classes_[idx]) for idx in pred_idx], dtype=int)

        out = df.copy()
        out["y_pred"] = y_pred

        full_proba = np.zeros((len(out), class_count), dtype=float)
        for model_idx, class_id in enumerate(self.classes_):
            class_int = int(class_id)
            if 0 <= class_int < class_count:
                full_proba[:, class_int] = probs[:, model_idx]

        row_sums = full_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        full_proba = full_proba / row_sums

        for class_idx in range(class_count):
            out[f"pred_proba_{class_idx}"] = full_proba[:, class_idx]

        return out

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Monitored model is not fitted")

        payload = {
            "adapter": "resnet_classifier",
            "model_name": self.model_name,
            "model_params": self.model_params,
            "weights_name": self.weights_name,
            "classes": self.classes_,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "ResNetMonitoredClassifier":
        if not _HAS_TORCH:
            raise ImportError(
                "ResNetMonitoredClassifier requires torch, torchvision, and Pillow."
            ) from _TORCH_IMPORT_ERROR

        # This checkpoint is produced by this codebase and contains a Python dict
        # payload (not just tensors). With PyTorch >=2.6, torch.load defaults to
        # weights_only=True, which blocks this payload shape unless explicitly
        # disabled for trusted local artifacts.
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Backward compatibility for older torch versions that do not expose
            # the weights_only argument.
            payload = torch.load(path, map_location="cpu")
        instance = cls(
            model_name=payload["model_name"],
            model_params=payload["model_params"],
        )
        instance.classes_ = payload["classes"]
        instance.model = instance._build_model(class_count=len(instance.classes_))
        instance.model.load_state_dict(payload["state_dict"])
        instance.model.eval()
        return instance


def create_monitored_model(
    model_name: str = "logistic_regression",
    model_params: Dict[str, Any] | None = None,
) -> MonitoredModelAdapter:
    """Create a monitored-model adapter for prediction generation.

    Supported adapters:
    - logistic_regression: baseline classifier over embedding columns.
    - resnet_classifier: classifier over raw image paths.

    Model-role separation remains explicit:
    - System extractor generates embedding feature columns.
    - Monitored model generates y_pred and pred_proba_* columns.
    """
    params = model_params or {}
    if model_name == "logistic_regression":
        return EmbeddingBaselineClassifier(model_name=model_name, model_params=params)
    if model_name == "resnet_classifier":
        return ResNetMonitoredClassifier(
            model_name=params.get("model_name", "resnet18"),
            model_params=params,
        )

    supported = "logistic_regression, resnet_classifier"
    raise ValueError(f"Unsupported monitored model '{model_name}'. Supported: {supported}")
