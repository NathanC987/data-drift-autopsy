"""Model-agnostic reliability analyzer orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.reliability.calibration import CalibrationChecker
from drift_autopsy.reliability.confidence import ConfidenceExtractor
from drift_autopsy.reliability.explanation import ExplanationConsistencyChecker
from drift_autopsy.reliability.ood import OODDetector
from drift_autopsy.reliability.risk_engine import RiskScoringEngine, RiskWeights
from drift_autopsy.reliability.stability import StabilityChecker


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def _is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class ReliabilityAnalyzer:
    """
    Model-agnostic reliability analyzer for hallucination risk estimation.

    Works without ground-truth labels and supports both single input and batch inputs.
    """

    def __init__(
        self,
        model: Any,
        data_type: str,
        reference_data: Any,
        task_type: str = "auto",
        cbpe_reference_score: Optional[float] = None,
        confidence_extractor: Optional[ConfidenceExtractor] = None,
        ood_detector: Optional[OODDetector] = None,
        stability_checker: Optional[StabilityChecker] = None,
        calibration_checker: Optional[CalibrationChecker] = None,
        explanation_checker: Optional[ExplanationConsistencyChecker] = None,
        risk_engine: Optional[RiskScoringEngine] = None,
        embedding_extractor: Optional[Callable[[Any], np.ndarray]] = None,
        gradcam_extractor: Optional[Callable[[Any, Any], np.ndarray]] = None,
    ):
        self.model = model
        self.data_type = data_type
        self.reference_data = reference_data
        self.task_type = task_type
        self.cbpe_reference_score = cbpe_reference_score

        self.confidence_extractor = confidence_extractor or ConfidenceExtractor(
            model=model,
            task_type=task_type,
        )
        self.ood_detector = ood_detector or OODDetector(
            data_type=data_type,
            method="auto",
            embedding_extractor=embedding_extractor,
        )
        self.stability_checker = stability_checker or StabilityChecker(
            model=model,
            data_type=data_type,
        )
        self.calibration_checker = calibration_checker or CalibrationChecker()
        self.explanation_checker = explanation_checker or ExplanationConsistencyChecker(
            model=model,
            data_type=data_type,
            gradcam_extractor=gradcam_extractor,
        )
        self.risk_engine = risk_engine or RiskScoringEngine(weights=RiskWeights())

        self._reference_confidences: Optional[np.ndarray] = None
        self._fit_reference()

    @staticmethod
    def detect_data_type(sample: Any) -> str:
        """Detect data type from an input sample."""
        if isinstance(sample, str):
            path = Path(sample)
            if path.exists() and _is_image_file(path):
                return "image"
            if path.exists() and _is_audio_file(path):
                return "audio"
            return "text"

        if isinstance(sample, Path):
            if _is_image_file(sample):
                return "image"
            if _is_audio_file(sample):
                return "audio"

        if isinstance(sample, pd.DataFrame):
            return "tabular"

        if isinstance(sample, np.ndarray):
            if sample.ndim in {1, 2}:
                return "tabular"
            if sample.ndim in {3, 4}:
                return "image"

        if isinstance(sample, list) and sample:
            first = sample[0]
            if isinstance(first, str):
                return "text"
            if isinstance(first, (int, float, np.number, list, np.ndarray, dict)):
                return "tabular"

        return "unstructured"

    @classmethod
    def from_dataset_dir(
        cls,
        model: Any,
        dataset_dir: str | Path,
        reference_file: Optional[str] = None,
        data_type: str = "auto",
        task_type: str = "auto",
        cbpe_reference_score: Optional[float] = None,
    ) -> "ReliabilityAnalyzer":
        """
        Create analyzer from /data/<dataset_name>/ directory.

        Tries CSV/Parquet/JSON first, then image/audio/text folders.
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        chosen: Optional[Path] = None
        if reference_file:
            chosen = dataset_path / reference_file
            if not chosen.exists():
                raise FileNotFoundError(f"Reference file not found: {chosen}")
        else:
            for pattern in ("*.csv", "*.parquet", "*.json"):
                matches = sorted(dataset_path.glob(pattern))
                if matches:
                    chosen = matches[0]
                    break

        if chosen is not None:
            if chosen.suffix.lower() == ".csv":
                reference_data = pd.read_csv(chosen)
            elif chosen.suffix.lower() == ".parquet":
                reference_data = pd.read_parquet(chosen)
            elif chosen.suffix.lower() == ".json":
                with open(chosen, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    reference_data = pd.DataFrame(payload)
                elif isinstance(payload, dict):
                    reference_data = pd.DataFrame(payload)
                else:
                    reference_data = payload
            else:
                reference_data = pd.DataFrame()
        else:
            files = [p for p in dataset_path.rglob("*") if p.is_file()]
            if not files:
                raise ValueError(f"No usable files found under {dataset_path}")
            if all(_is_image_file(p) for p in files[: min(20, len(files))]):
                reference_data = [str(p) for p in files]
            elif all(_is_audio_file(p) for p in files[: min(20, len(files))]):
                reference_data = [str(p) for p in files]
            else:
                txt_files = [p for p in files if p.suffix.lower() in {".txt", ".md"}]
                if txt_files:
                    reference_data = [p.read_text(encoding="utf-8", errors="ignore") for p in txt_files]
                else:
                    reference_data = [str(p) for p in files]

        inferred_data_type = cls.detect_data_type(reference_data) if data_type == "auto" else data_type
        return cls(
            model=model,
            data_type=inferred_data_type,
            reference_data=reference_data,
            task_type=task_type,
            cbpe_reference_score=cbpe_reference_score,
        )

    def _fit_reference(self) -> None:
        self.ood_detector.fit(self.reference_data)
        ref_conf = self.confidence_extractor.extract_batch(self.reference_data)
        self._reference_confidences = np.asarray(ref_conf["scores"], dtype=float)

    @staticmethod
    def _as_dataset_tabular(data: Any, feature_names: Optional[List[str]] = None) -> Dataset:
        if isinstance(data, Dataset):
            return data
        if isinstance(data, pd.DataFrame):
            return Dataset.from_pandas(data)
        if isinstance(data, pd.Series):
            if feature_names and len(feature_names) == len(data):
                frame = pd.DataFrame([data.values], columns=feature_names)
            else:
                frame = pd.DataFrame([data.values], columns=list(data.index))
            return Dataset.from_pandas(frame)
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return Dataset.from_numpy(data)
        if isinstance(data, list):
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if feature_names and arr.ndim == 2 and arr.shape[1] == len(feature_names):
                return Dataset.from_pandas(pd.DataFrame(arr, columns=feature_names))
            return Dataset.from_numpy(arr)

        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if feature_names and arr.ndim == 2 and arr.shape[1] == len(feature_names):
            return Dataset.from_pandas(pd.DataFrame(arr, columns=feature_names))
        return Dataset.from_numpy(arr)

    def analyze(
        self,
        input_sample: Any,
        prediction_id: Optional[str] = None,
        cbpe_score: Optional[float] = None,
        baseline_input: Optional[Any] = None,
        precomputed_explanation_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze reliability risk for a single prediction/input."""
        confidence_output = self.confidence_extractor.extract(input_sample)
        confidence_score = _safe_float(confidence_output["confidence_score"])

        ood_score = _safe_float(self.ood_detector.compute_ood_score(input_sample))

        stability_output = self.stability_checker.compute_stability(input_sample)
        stability_score = _safe_float(stability_output["stability_score"])

        current_conf = self.confidence_extractor.extract_batch([input_sample])["scores"]
        conf_shift = self.calibration_checker.confidence_shift(
            self._reference_confidences if self._reference_confidences is not None else np.array([]),
            current_conf,
        )
        calibration_output = self.calibration_checker.evaluate(
            confidence_score=confidence_score,
            cbpe_score=cbpe_score if cbpe_score is not None else self.cbpe_reference_score,
            confidence_distribution_shift=conf_shift,
        )

        if precomputed_explanation_output is not None:
            explanation_output = precomputed_explanation_output
        elif self.data_type == "tabular":
            reference_ds = self._as_dataset_tabular(self.reference_data)
            current_payload = (
                input_sample
                if isinstance(input_sample, (pd.DataFrame, pd.Series))
                else [input_sample]
            )
            current_ds = self._as_dataset_tabular(
                current_payload,
                feature_names=reference_ds.feature_names,
            )
            explanation_output = self.explanation_checker.compute(
                reference_data=reference_ds,
                current_data=current_ds,
            )
        else:
            explanation_output = self.explanation_checker.compute(
                baseline_input=baseline_input if baseline_input is not None else self.reference_data,
                current_input=input_sample,
            )

        explanation_score = _safe_float(explanation_output.get("explanation_score", 0.5), default=0.5)

        risk_output = self.risk_engine.combine(
            confidence_score=confidence_score,
            ood_score=ood_score,
            stability_score=stability_score,
            calibration_flag=calibration_output["calibration_flag"],
            calibration_risk=calibration_output["calibration_risk"],
            explanation_score=explanation_score,
        )

        return {
            "prediction_id": prediction_id or str(uuid4()),
            "confidence": confidence_score,
            "ood": ood_score,
            "stability": stability_score,
            "calibration": calibration_output["calibration_flag"],
            "calibration_risk": _safe_float(calibration_output["calibration_risk"]),
            "explanation": explanation_score,
            "cbpe_score": _safe_float(
                cbpe_score if cbpe_score is not None else self.cbpe_reference_score,
                default=0.5,
            ),
            "risk_score": _safe_float(risk_output["risk_score"]),
            "risk_label": risk_output["risk_label"],
            "details": {
                "confidence": confidence_output,
                "stability": stability_output,
                "calibration": calibration_output,
                "explanation": explanation_output,
                "risk": risk_output,
            },
        }

    def analyze_batch(
        self,
        input_batch: Any,
        cbpe_score: Optional[float] = None,
        prediction_ids: Optional[List[str]] = None,
        shared_explanation_for_batch: bool = True,
    ) -> List[Dict[str, Any]]:
        """Analyze reliability risk for a batch of inputs."""
        shared_explanation_output: Optional[Dict[str, Any]] = None

        if shared_explanation_for_batch and self.data_type == "tabular":
            reference_ds = self._as_dataset_tabular(self.reference_data)
            batch_ds = self._as_dataset_tabular(
                input_batch,
                feature_names=reference_ds.feature_names,
            )
            shared_explanation_output = self.explanation_checker.compute(
                reference_data=reference_ds,
                current_data=batch_ds,
            )

        if isinstance(input_batch, pd.DataFrame):
            iterable = [row for _, row in input_batch.iterrows()]
        elif isinstance(input_batch, np.ndarray):
            iterable = [row for row in input_batch]
        elif isinstance(input_batch, list):
            iterable = input_batch
        else:
            iterable = [input_batch]

        outputs = []
        for idx, sample in enumerate(iterable):
            pid = prediction_ids[idx] if prediction_ids and idx < len(prediction_ids) else None
            outputs.append(
                self.analyze(
                    input_sample=sample,
                    prediction_id=pid,
                    cbpe_score=cbpe_score,
                    precomputed_explanation_output=shared_explanation_output,
                )
            )
        return outputs

    @staticmethod
    def save_results(results: List[Dict[str, Any]] | Dict[str, Any], output_path: str | Path) -> None:
        """Persist reliability outputs as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = results if isinstance(results, list) else [results]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
