# 🚀 Project Setup Guide

This guide covers how to set up the Data Drift Autopsy project as a GitHub repository and deploy it on a new device.

## 📦 Initial Setup on Current Device

### 1. Initialize and Push to GitHub

```bash
# Navigate to project directory
cd /home/nathan/data-drift-autopsy

# Add all files (except those in .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: Data Drift Autopsy with RCA integration"

# Create a new repository on GitHub (via web interface):
# Go to https://github.com/new
# Repository name: data-drift-autopsy
# Make it public or private
# Do NOT initialize with README (we already have one)

# Link to GitHub repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/data-drift-autopsy.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 🔧 Setup on New Device

### 1. Clone Repository

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/data-drift-autopsy.git
cd data-drift-autopsy
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Or install from pyproject.toml
pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. Download Folktables Data

The Folktables dataset will be downloaded automatically when you run the demo for the first time. It uses the ACS (American Community Survey) data from the US Census Bureau.

### 5. Run the Demo

```bash
# Run the Folktables demo (generates drift analysis with RCA)
python examples/quickstart/folktables_demo.py

# This will:
# - Download the Folktables data (~5-10 seconds)
# - Run drift detection across 4 years (2015-2018)
# - Perform RCA analysis using SHAP for KS Test
# - Save results to: outputs/folktables_drift_results.json
```

### 6. Launch Dashboard

```bash
# Make the launch script executable (Linux/Mac only)
chmod +x launch_dashboard.sh

# Launch the dashboard
./launch_dashboard.sh

# Or manually:
streamlit run examples/dashboard/app.py --server.port 8501
```

The dashboard will open at: http://localhost:8501

## 📋 Dependencies

Key dependencies (automatically installed):
- **NumPy >= 2.0**: Numerical computing (with SHAP compatibility shims)
- **Pandas**: Data manipulation
- **Scikit-learn**: ML models and metrics
- **SHAP 0.50.0**: Root cause analysis
- **Streamlit**: Dashboard UI
- **Plotly**: Interactive visualizations
- **Folktables**: Census data for temporal drift demo

## 🔍 Project Structure

```
data-drift-autopsy/
├── src/drift_autopsy/          # Main package
│   ├── detectors/              # Drift detectors (KS, PSI, MMD, CBPE)
│   ├── rca/                    # Root cause analysis (SHAP)
│   ├── localization/           # Feature-level drift analysis
│   └── registry.py             # Detector registry
├── examples/
│   ├── quickstart/
│   │   └── folktables_demo.py  # Demo with temporal drift
│   └── dashboard/
│       ├── app.py              # Streamlit dashboard
│       ├── data_loader.py      # JSON parser for drift results
│       └── visualizations.py   # Plotly chart components
├── configs/                     # Configuration files
├── outputs/                     # Generated drift analysis results
├── tests/                       # Unit tests
└── pyproject.toml              # Package metadata & dependencies
```

## 🎯 Key Features

### Drift Detection Methods
- **KS Test**: Kolmogorov-Smirnov test for distribution comparison
- **PSI**: Population Stability Index
- **MMD**: Maximum Mean Discrepancy (kernel-based)
- **CBPE**: Confidence-Based Performance Estimation (proxy performance)

### Root Cause Analysis (RCA)
- **SHAP-based**: Explains feature importance changes between reference and test data
- **Enabled for**: KS Test pipeline (configurable for others)
- **Analysis**: Compares SHAP values on 100 reference + 100 test samples

### Dashboard Visualizations
1. **Performance Estimator (CBPE)**: Timeline and comparison charts
2. **Drift Detectors (KS/PSI/MMD)**: Timeline and comparison charts (separate scales)
3. **Model Performance**: Accuracy tracking over time
4. **Feature Analysis**: Feature-level drift heatmap and top drifted features
5. **RCA Insights**: Feature importance changes, timeline, heatmap, recommendations
6. **Raw Data**: Tabular view of all results

## 🐛 Troubleshooting

### NumPy 2.0 Compatibility
If you encounter SHAP errors related to NumPy 2.0:
- The project includes compatibility shims in `src/drift_autopsy/rca/shap_analyzer.py`
- These automatically patch `np.trapz` → `np.trapezoid` and `np.in1d` → `np.isin`

### Missing Data
If demo fails to generate RCA data:
- Check that the model is correctly extracted from the pipeline
- Verify all 10 features are passed to SHAP (not just drifted features)
- Check `outputs/folktables_drift_results.json` for error messages

### Dashboard Shows "No data available"
- Ensure you've run `python examples/quickstart/folktables_demo.py` first
- Check that `outputs/folktables_drift_results.json` exists
- Verify the detector name transformations match (e.g., "Ks Test" not "KS Test")

## 📝 Making Changes

### Running After Changes

```bash
# If you modified the core library (src/drift_autopsy/)
pip install -e .  # Reinstall in dev mode

# If you modified examples only (dashboard, demo)
# No reinstall needed, just re-run:
python examples/quickstart/folktables_demo.py
./launch_dashboard.sh
```

### Committing Changes

```bash
git add .
git commit -m "Description of changes"
git push origin main
```

## 🔗 Useful Commands

```bash
# Check installed version
pip show drift-autopsy

# Run tests (if available)
pytest tests/

# Check detector registry
python -c "from drift_autopsy.registry import DetectorRegistry; print(DetectorRegistry.list())"

# Generate fresh drift results
python examples/quickstart/folktables_demo.py

# Launch dashboard
./launch_dashboard.sh
```

## 📧 Support

For issues or questions:
1. Check `outputs/` for error logs
2. Review the SHAP analyzer code: `src/drift_autopsy/rca/shap_analyzer.py`
3. Check dashboard data loading: `examples/dashboard/data_loader.py`

## 📚 Blueprint Docs

For complete developer blueprint references (end-goal + current implementation), see:

- `docs/README.md`
- `docs/01-end-goal/`
- `docs/02-current-implementation/`

When code changes, update the current-implementation docs using:

- `docs/02-current-implementation/UPDATE_PROTOCOL.md`
