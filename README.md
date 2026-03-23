# Data Drift Autopsy

A modular framework for data drift detection, diagnosis, and remediation in production ML systems.

## Features

- **Detection**: Multiple drift detection methods (statistical tests, distribution-based, model-based)
- **Localization**: Identify which features and data slices are affected
- **Root Cause Analysis**: Understand why drift occurred (SHAP-based explanations)
- **Modular Design**: Easy to extend with custom detectors and analyzers
- **Config-Driven**: Support for both Python API and YAML configuration

## Quick Start

```python
from drift_autopsy import DriftPipeline
from drift_autopsy.detectors import KSTest
from drift_autopsy.data import Dataset

# Create pipeline
pipeline = DriftPipeline(
    detector=KSTest(threshold=0.05),
    localizer="univariate",
    enable_rca=True
)

# Run drift analysis
reference = Dataset.from_pandas(reference_df)
test = Dataset.from_pandas(test_df)
result = pipeline.run(reference, test)

print(f"Drift detected: {result.drift_detected}")
```

## Dashboard

Interactive web dashboard for visualizing drift analysis results:

```bash
# Install dashboard dependencies
pip install -e ".[dashboard]"

# Run analysis to generate results
python examples/quickstart/folktables_demo.py

# Launch dashboard
streamlit run examples/dashboard/app.py
# or
./launch_dashboard.sh
```

The dashboard provides:
- 📈 Real-time drift metrics and KPIs
- 📊 Interactive time-series charts
- 🎯 Feature-level drift analysis
- 🔍 Detector comparison visualizations
- 📉 Model performance tracking

See [examples/dashboard/README.md](examples/dashboard/README.md) for details.

## Installation

```bash
pip install -e .
pip install -e ".[all]"  # Install with all optional dependencies
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
mypy src/drift_autopsy
```

## Architecture

- `core/`: Abstract interfaces and protocols
- `detectors/`: Drift detection implementations
- `localizers/`: Feature-level drift localization
- `rca/`: Root cause analysis methods
- `data/`: Data loading and preprocessing
- `config/`: Configuration management

## Documentation Blueprint

For developer-facing project blueprint documentation, see:

- `docs/README.md`
- `docs/01-end-goal/` (enterprise target state)
- `docs/02-current-implementation/` (current as-built state and rationale)

## License

MIT
