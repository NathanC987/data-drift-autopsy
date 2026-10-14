# Dashboard

Streamlit view of the study outputs in `outputs/`.

```bash
pip install -e ".[dashboard]"
streamlit run examples/dashboard/app.py     # or ./launch_dashboard.sh
```

Tabs: ACS Income, CLEAR-10 (both in pipeline order: setup, accuracy vs. the
label-free estimate, detection, localization, root cause, diagnosis,
reliability, remediation) and the injection benchmark. The dashboard only
reads outputs; generate them first with the scripts listed in
`examples/README.md`. Data contracts: `docs/02-current-implementation/DASHBOARD_AND_DATA_CONTRACTS.md`.
