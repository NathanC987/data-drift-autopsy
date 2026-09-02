"""
Remediation Retrain Script — trains on expanded data to fix drift.

Original model: trained on 2014 only → drifts on 2017-2018.
Retrained model: trained on 2014+2015+2016 → adapts to new patterns.

Compares both models and saves a remediation_results.json showing improvement.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SKPipeline

from drift_autopsy import DriftPipeline, Dataset
from drift_autopsy.data import FolktablesLoader
from drift_autopsy.detectors import KSTest, PSI, MMD, CBPE
from drift_autopsy.utils import setup_logging


STATE = "CA"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_year(year: int) -> Dataset:
    """Load ACS Income data for a single year."""
    return FolktablesLoader.load_acs_income(year=year, states=[STATE], download=True)


def train_model(datasets: list[Dataset]) -> SKPipeline:
    """Train a LogisticRegression model on one or more datasets combined."""
    X_parts = [ds.to_numpy() for ds in datasets]
    y_parts = [ds.target.values for ds in datasets]
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    model = SKPipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X, y)
    return model


def evaluate(model: SKPipeline, dataset: Dataset) -> float:
    """Return accuracy of model on dataset."""
    X = dataset.to_numpy()
    y = dataset.target.values
    return float(model.score(X, y))


def main():
    setup_logging(level="INFO")

    print("=" * 70)
    print("REMEDIATION: Retrain with Expanded Data")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 1: Load all yearly datasets
    # ------------------------------------------------------------------
    years = [2014, 2015, 2016, 2017, 2018]
    datasets = {}
    for yr in years:
        print(f"Loading {STATE} {yr}...")
        datasets[yr] = load_year(yr)
        print(f"  {datasets[yr].n_samples} samples")

    print()

    # ------------------------------------------------------------------
    # Step 2: Train ORIGINAL model (2014 only)
    # ------------------------------------------------------------------
    print("--- Original Model (trained on 2014 only) ---")
    original_model = train_model([datasets[2014]])

    original_scores = {}
    for yr in years:
        acc = evaluate(original_model, datasets[yr])
        original_scores[yr] = acc
        tag = "(train)" if yr == 2014 else ""
        print(f"  {yr}: {acc:.4f} {tag}")

    print()

    # ------------------------------------------------------------------
    # Step 3: Train RETRAINED model (2014 + 2015 + 2016)
    # ------------------------------------------------------------------
    print("--- Retrained Model (trained on 2014 + 2015 + 2016) ---")
    retrained_model = train_model([datasets[2014], datasets[2015], datasets[2016]])

    retrained_scores = {}
    for yr in years:
        acc = evaluate(retrained_model, datasets[yr])
        retrained_scores[yr] = acc
        tag = "(train)" if yr <= 2016 else ""
        print(f"  {yr}: {acc:.4f} {tag}")

    print()

    # ------------------------------------------------------------------
    # Step 4: Compare on the unseen test years (2017-2018)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("COMPARISON on unseen test data (2017 & 2018)")
    print("=" * 70)

    comparison = {}
    for yr in [2017, 2018]:
        old = original_scores[yr]
        new = retrained_scores[yr]
        improvement = new - old
        comparison[yr] = {
            "original_accuracy": round(old, 4),
            "retrained_accuracy": round(new, 4),
            "improvement": round(improvement, 4),
            "improved": improvement > 0,
        }
        symbol = "+" if improvement > 0 else ""
        print(f"  {yr}: {old:.4f} -> {new:.4f}  ({symbol}{improvement:.4f})")

    print()

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    results = {
        "remediation_type": "full_retraining_expanded_data",
        "original_training_years": [2014],
        "retrained_training_years": [2014, 2015, 2016],
        "test_years": [2017, 2018],
        "original_scores": {str(k): round(v, 4) for k, v in original_scores.items()},
        "retrained_scores": {str(k): round(v, 4) for k, v in retrained_scores.items()},
        "comparison": {str(k): v for k, v in comparison.items()},
    }

    output_file = OUTPUT_DIR / "remediation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()
    print("Remediation complete!")


if __name__ == "__main__":
    main()
