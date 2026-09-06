"""
Folktables Temporal Drift Demo

Demonstrates drift detection on ACS Income data across years (2014-2018) for California.
Trains a model on 2014 data and monitors temporal drift as the years progress.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline

# Import drift autopsy components
from drift_autopsy import DriftPipeline, Dataset
from drift_autopsy.data import FolktablesLoader
from drift_autopsy.detectors import KSTest, PSI, MMD, CBPE
from drift_autopsy.localizers import UnivariateLocalizer
from drift_autopsy.rca import SHAPAnalyzer
from drift_autopsy.utils import setup_logging


DETECTOR_THRESHOLDS = {
    "KS Test": 0.05,
    "PSI": 0.2,
    "MMD": 0.1,
    "CBPE": 0.05,
}

# Keep localizer threshold explicit at run call-time for consistency.
LOCALIZATION_THRESHOLD = 0.05


def build_state_code_mapping(year: int, states):
    """Build mapping between ACS state codes and state abbreviations."""
    code_to_state = {}
    state_to_code = {}

    print("Resolving ACS state metadata codes...")
    for state in states:
        ds = FolktablesLoader.load_acs_income(
            year=year,
            states=[state],
            download=True,
        )
        state_code = str(ds.metadata["state"].mode().iloc[0])
        code_to_state[state_code] = state
        state_to_code[state] = state_code
        print(f"  {state} -> state code {state_code}")

    return code_to_state, state_to_code


def create_pipelines(lr_model):
    """Create configured drift pipelines."""
    return {
        "KS Test": DriftPipeline(
            detector=KSTest(threshold=0.05, correction="bonferroni"),  # p-value threshold (95% confidence)
            localizer="univariate",
            rca="shap",  # Use SHAP analyzer from registry
            model=lr_model,  # Pass model for RCA
            enable_localization=True,
            enable_rca=True,  # Enable RCA for KS Test
        ),
        "PSI": DriftPipeline(
            detector=PSI(threshold=0.2, n_bins=10),  # Industry standard: >0.2 indicates significant drift
            localizer="univariate",
            enable_localization=True,
            enable_rca=False,
        ),
        "MMD": DriftPipeline(
            detector=MMD(threshold=0.1, kernel="rbf", n_permutations=20, max_samples=3000),  # Reasonable default for normalized features
            localizer="univariate",
            enable_localization=True,
            enable_rca=False,
        ),
        "CBPE": DriftPipeline(
            detector=CBPE(threshold=0.05, n_bins=10),  # Allow 5% performance degradation
            localizer="univariate",
            enable_localization=True,
            enable_rca=False,
        ),
    }


def run_pipelines(
    pipelines,
    train_dataset,
    test_dataset,
    train_dataset_with_preds,
    test_dataset_with_preds,
    slice_config=None,
):
    """Run all pipelines and return dictionary results."""
    run_results = {}

    for pipeline_name, pipeline in pipelines.items():
        print(f"Running {pipeline_name}...")

        try:
            run_kwargs = {
                "detection_threshold": DETECTOR_THRESHOLDS[pipeline_name],
                "localization_threshold": LOCALIZATION_THRESHOLD,
            }
            if slice_config is not None:
                run_kwargs["slice_config"] = slice_config

            # Use dataset with predictions for CBPE, without for others
            if pipeline_name == "CBPE":
                result = pipeline.run(train_dataset_with_preds, test_dataset_with_preds, **run_kwargs)
            else:
                result = pipeline.run(train_dataset, test_dataset, **run_kwargs)

            print(f"  Drift Detected: {result.detection.drift_detected}")
            print(f"  Severity: {result.detection.severity.value}")
            print(f"  Score: {result.detection.score:.4f}")

            if result.localization:
                n_drifted = len(result.localization.drifted_features)
                print(f"  Drifted Features: {n_drifted}")
                if n_drifted > 0:
                    top_3 = result.localization.drifted_features[:3]
                    print(f"    Top 3: {', '.join(top_3)}")

            if result.rca:
                n_recommendations = len(result.rca.recommendations)
                print(f"  RCA Recommendations: {n_recommendations}")
                if n_recommendations > 0:
                    print(f"    Sample: {result.rca.recommendations[0]}")

            if result.metadata.get("slice_analysis", {}).get("enabled"):
                n_slices = result.metadata.get("slice_analysis", {}).get("slice_count", 0)
                print(f"  Slice Analysis: {n_slices} slices evaluated")

            print(f"  Execution Time: {result.execution_time_seconds:.2f}s")
            print()

            run_results[pipeline_name] = result.to_dict()

        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            run_results[pipeline_name] = {"error": str(e)}

    return run_results


def load_multi_state_income_dataset(year: int, states):
    """Load ACS Income per-state and concatenate for deterministic geographic coverage."""
    state_datasets = []
    for state in states:
        ds = FolktablesLoader.load_acs_income(
            year=year,
            states=[state],
            download=True,
        )
        state_datasets.append(ds)

    first = state_datasets[0]
    all_data = pd.concat([ds.data for ds in state_datasets], axis=0, ignore_index=True)
    all_target = pd.concat([ds.target for ds in state_datasets], axis=0, ignore_index=True)
    all_metadata = pd.concat([ds.metadata for ds in state_datasets], axis=0, ignore_index=True)

    return Dataset(
        data=all_data,
        feature_names=first.feature_names,
        target=all_target,
        target_name=first.target_name,
        metadata=all_metadata,
    )


def estimate_cbpe_accuracy(
    reference_confidence: np.ndarray,
    reference_correct: np.ndarray,
    test_confidence: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Label-free accuracy estimate for the test set (CBPE-style).

    Calibrates a per-bin accuracy from reference-set confidence and correctness,
    then averages those calibrated bin accuracies over the test set's confidence
    histogram. Returns a bounded estimate in [0, 1] that the reliability layer's
    calibration check can compare against prediction confidence. Passing the CBPE
    detector's chi-square p-value here instead (as an earlier version did) is a
    category error: the p-value is not a performance estimate and drives the
    overconfidence gap to an artificial floor or ceiling.
    """
    reference_confidence = np.asarray(reference_confidence, dtype=float)
    reference_correct = np.asarray(reference_correct, dtype=float)
    test_confidence = np.asarray(test_confidence, dtype=float)

    if reference_confidence.size == 0 or test_confidence.size == 0:
        return 0.5

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ref_bin = np.clip(np.digitize(reference_confidence, bin_edges[1:-1]), 0, n_bins - 1)
    test_bin = np.clip(np.digitize(test_confidence, bin_edges[1:-1]), 0, n_bins - 1)

    global_accuracy = float(reference_correct.mean())
    bin_accuracy = np.full(n_bins, global_accuracy, dtype=float)
    for bin_idx in range(n_bins):
        mask = ref_bin == bin_idx
        if np.any(mask):
            bin_accuracy[bin_idx] = float(reference_correct[mask].mean())

    return float(np.clip(np.mean(bin_accuracy[test_bin]), 0.0, 1.0))


def build_reliability_records(
    model,
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    estimated_accuracy: float,
    analysis_key: str,
    max_samples: int = 200,
):
    """Generate reliability outputs for a subset of samples.

    ``estimated_accuracy`` is a label-free performance estimate in [0, 1]; the
    reliability layer uses it as the calibration reference for the overconfidence
    gap (confidence minus estimated correctness).
    """
    if test_df.empty:
        return []

    sample_df = test_df.sample(n=min(max_samples, len(test_df)), random_state=42).reset_index(drop=True)

    analyzer = ReliabilityAnalyzer(
        model=model,
        data_type="tabular",
        reference_data=reference_df,
        task_type="classification",
        cbpe_reference_score=estimated_accuracy,
    )

    records = analyzer.analyze_batch(sample_df)
    for idx, record in enumerate(records):
        record["analysis_key"] = analysis_key
        record["detector"] = "model_agnostic"
        if record.get("prediction_id") is None:
            record["prediction_id"] = f"{analysis_key}_{idx}"

    return records


def main():
    # Setup logging
    setup_logging(level="INFO")

    # Seed global RNG so SHAP sampling and reliability sampling are reproducible.
    np.random.seed(42)

    print("=" * 80)
    print("Folktables Temporal + Geographic Drift Analysis Demo")
    print("Dataset: ACS Income (Temporal: CA 2014-2018, Geographic: cross-state)")
    print("=" * 80)
    print()
    
    # Configuration
    BASE_YEAR = 2014
    TEST_YEARS = [2015, 2016, 2017, 2018]
    STATE = "CA"
    GEO_STATES = ["CA", "TX", "NY", "FL", "WA"]
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Step 1: Load training data (2014)
    print(f"Loading training data: {STATE} {BASE_YEAR}")
    train_dataset = FolktablesLoader.load_acs_income(
        year=BASE_YEAR,
        states=[STATE],
        download=True
    )
    print(f"  Loaded: {train_dataset.n_samples} samples, {train_dataset.n_features} features")
    print()
    
    # Step 2: Train model
    print("Training LogisticRegression model...")
    X_train = train_dataset.to_numpy()
    y_train = train_dataset.target.values
    
    model = SKPipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    print(f"  Training accuracy: {train_score:.4f}")
    print()
    
    # Get predictions on training data for CBPE baseline
    train_proba = model.predict_proba(X_train)
    train_confidence = train_proba.max(axis=1)
    train_correct = (model.predict(X_train) == y_train).astype(float)
    train_dataset_with_preds = Dataset(
        data=train_dataset.data,
        feature_names=train_dataset.feature_names,
        target=train_dataset.target,
        predictions=model.predict(X_train),
        prediction_probabilities=train_proba,
        metadata=train_dataset.metadata,
    )

    # Build explicit state<->code mapping for readable outputs (e.g., CA->TX).
    code_to_state, state_to_code = build_state_code_mapping(BASE_YEAR, GEO_STATES)
    reference_state_code = state_to_code[STATE]
    print(f"  Reference state '{STATE}' maps to metadata state value: {reference_state_code}")
    
    # Step 3: Setup drift detection pipelines
    print("Setting up drift detection pipelines...")
    
    # Extract the actual LogisticRegression model from the sklearn pipeline
    # (sklearn Pipeline wraps the model, but SHAP needs the raw model)
    lr_model = model.named_steps['classifier']
    
    pipelines = create_pipelines(lr_model)
    
    print(f"  Initialized {len(pipelines)} pipelines")
    print()
    
    # Step 4: Run drift detection for each year
    all_results = {}
    
    for year in TEST_YEARS:
        print("=" * 80)
        print(f"Analyzing Year: {year}")
        print("=" * 80)
        
        # Load test data
        print(f"Loading test data: {STATE} {year}")
        test_dataset = FolktablesLoader.load_acs_income(
            year=year,
            states=[STATE],
            download=True
        )
        print(f"  Loaded: {test_dataset.n_samples} samples")
        
        # Get predictions
        X_test = test_dataset.to_numpy()
        y_test = test_dataset.target.values
        test_proba = model.predict_proba(X_test)
        test_score = model.score(X_test, y_test)
        
        print(f"  Model accuracy on {year}: {test_score:.4f} (Δ = {test_score - train_score:+.4f})")
        print()
        
        # Create dataset with predictions
        test_dataset_with_preds = Dataset(
            data=test_dataset.data,
            feature_names=test_dataset.feature_names,
            target=test_dataset.target,
            predictions=model.predict(X_test),
            prediction_probabilities=test_proba,
            metadata=test_dataset.metadata,
        )
        
        year_results = run_pipelines(
            pipelines=pipelines,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_dataset_with_preds=train_dataset_with_preds,
            test_dataset_with_preds=test_dataset_with_preds,
        )

        estimated_accuracy = estimate_cbpe_accuracy(
            reference_confidence=train_confidence,
            reference_correct=train_correct,
            test_confidence=test_proba.max(axis=1),
        )
        print(
            f"  CBPE estimated accuracy: {estimated_accuracy:.4f} "
            f"(actual {test_score:.4f}, gap {estimated_accuracy - test_score:+.4f})"
        )

        reliability_records = build_reliability_records(
            model=model,
            reference_df=train_dataset.data,
            test_df=test_dataset.data,
            estimated_accuracy=estimated_accuracy,
            analysis_key=str(year),
        )

        all_results[year] = {
            "analysis_type": "temporal",
            "actual_accuracy": float(test_score),
            "accuracy_drop": float(test_score - train_score),
            "cbpe_estimated_accuracy": estimated_accuracy,
            "pipelines": year_results,
        }

    # Step 5: Run geographic drift analysis (cross-state at BASE_YEAR)
    print("=" * 80)
    print("Geographic Analysis")
    print("=" * 80)
    print(
        f"Comparing reference slice '{STATE}' against test slices in {GEO_STATES} "
        f"for year {BASE_YEAR}"
    )

    geo_dataset = load_multi_state_income_dataset(
        year=BASE_YEAR,
        states=GEO_STATES,
    )
    X_geo = geo_dataset.to_numpy()
    geo_proba = model.predict_proba(X_geo)
    geo_dataset_with_preds = Dataset(
        data=geo_dataset.data,
        feature_names=geo_dataset.feature_names,
        target=geo_dataset.target,
        predictions=model.predict(X_geo),
        prediction_probabilities=geo_proba,
        metadata=geo_dataset.metadata,
    )

    geo_slice_config = {
        "enabled": True,
        "column": "state",
        "reference_slice_value": reference_state_code,
        "min_samples_per_slice": 100,
    }

    geographic_results = run_pipelines(
        pipelines=pipelines,
        train_dataset=geo_dataset,
        test_dataset=geo_dataset,
        train_dataset_with_preds=geo_dataset_with_preds,
        test_dataset_with_preds=geo_dataset_with_preds,
        slice_config=geo_slice_config,
    )

    geo_estimated_accuracy = estimate_cbpe_accuracy(
        reference_confidence=train_confidence,
        reference_correct=train_correct,
        test_confidence=geo_proba.max(axis=1),
    )
    geo_actual_accuracy = float(model.score(X_geo, geo_dataset.target.values))
    print(
        f"  CBPE estimated accuracy (pooled states): {geo_estimated_accuracy:.4f} "
        f"(actual {geo_actual_accuracy:.4f})"
    )

    geographic_reliability = build_reliability_records(
        model=model,
        reference_df=geo_dataset.data,
        test_df=geo_dataset.data,
        estimated_accuracy=geo_estimated_accuracy,
        analysis_key="geographic_analysis",
    )

    all_results["geographic_analysis"] = {
        "analysis_type": "geographic",
        "year": BASE_YEAR,
        "reference_state": STATE,
        "reference_slice_value": reference_state_code,
        "slice_value_labels": code_to_state,
        "states_analyzed": GEO_STATES,
        "slice_column": "state",
        "cbpe_estimated_accuracy": geo_estimated_accuracy,
        "pipelines": geographic_results,
    }
    
    # Step 6: Save results
    print("=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    output_file = OUTPUT_DIR / "folktables_drift_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    # Step 7: Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Year-over-Year Performance:")
    print(f"  {BASE_YEAR} (train): {train_score:.4f}")
    for year in TEST_YEARS:
        acc = all_results[year]["actual_accuracy"]
        drop = all_results[year]["accuracy_drop"]
        print(f"  {year}:         {acc:.4f} (Δ = {drop:+.4f})")
    print()
    
    print("Drift Detection Summary:")
    for pipeline_name in pipelines.keys():
        print(f"\n{pipeline_name}:")
        for year in TEST_YEARS:
            if "error" not in all_results[year]["pipelines"][pipeline_name]:
                result = all_results[year]["pipelines"][pipeline_name]
                detected = result["detection"]["drift_detected"]
                severity = result["detection"]["severity"]
                print(f"  {year}: {'DRIFT' if detected else 'NO DRIFT':8s} ({severity})")
    print()

    print("Geographic Drift Summary:")
    geo_labels = all_results.get("geographic_analysis", {}).get("slice_value_labels", {})
    for pipeline_name in pipelines.keys():
        print(f"\n{pipeline_name}:")
        geo_pipeline = all_results.get("geographic_analysis", {}).get("pipelines", {}).get(pipeline_name, {})
        slice_info = geo_pipeline.get("metadata", {}).get("slice_analysis", {})
        if slice_info.get("enabled"):
            print(f"  Slices evaluated: {slice_info.get('slice_count', 0)}")
            for slice_key, payload in slice_info.get("slices", {}).items():
                detection = payload.get("result", {}).get("detection", {})
                detected = detection.get("drift_detected", False)
                severity = detection.get("severity", "none")
                ref_code = str(payload.get("reference_slice_value"))
                test_code = str(payload.get("test_slice_value"))
                ref_label = geo_labels.get(ref_code, ref_code)
                test_label = geo_labels.get(test_code, test_code)
                label_key = f"{ref_label}->{test_label}"
                print(f"  {label_key}: {'DRIFT' if detected else 'NO DRIFT':8s} ({severity})")
        else:
            print("  No slice analysis output")

    print()
    
    print("Demo complete!")


if __name__ == "__main__":
    main()
