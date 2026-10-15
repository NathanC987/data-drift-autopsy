# Dashboard

Streamlit view of the study outputs in `outputs/`.

```bash
pip install -e ".[dashboard]"
streamlit run examples/dashboard/app.py     # or ./launch_dashboard.sh
```

Four tabs, each a step-by-step story:

- **ACS** (Income or Employment), **CLEAR-10** and the **gas sensor array**
  follow the architecture: setup and results; proxy performance estimate,
  drift detection and localization; root cause and drift type; per-prediction
  reliability; remediation triage. A selector switches the monitored model
  (logistic regression or gradient boosting).
- **Injection benchmark** (dataset, design A or B, model) follows the
  procedure: design, ground truth, detection, proxy estimate, localization,
  drift type, baselines and ablations, reliability, triage.

Each step states the question it answers and how the method works, shows the
chart and ends with a "What the data shows" line computed from the outputs.
The dashboard only reads outputs; generate them first with the scripts listed
in `examples/README.md` (the CAM images come from
`examples/paper/build_artifacts.py`). Data contracts: `docs/02-current-implementation/DASHBOARD_AND_DATA_CONTRACTS.md`.
