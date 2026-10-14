# Examples

| Script | What it produces |
|---|---|
| `quickstart/simple_example.py` | minimal `DriftPipeline` run on synthetic data |
| `quickstart/folktables_demo.py` | ACS Income study (CA 2014 reference; CA 2015-2018 and TX/NY/FL/WA 2014 windows) -> `outputs/folktables_drift_results.json` |
| `quickstart/clear10_tabularize_demo.py` | frozen ResNet-18 embeddings for every CLEAR-10 bucket -> `outputs/clear10_tabularized_demo/` |
| `quickstart/clear10_drift_study.py` | CLEAR-10 study (bucket 1 reference; buckets 2-10) -> `outputs/clear10_drift_results.json` |
| `quickstart/clip_concept_probe_demo.py` | CLIP concept probe for buckets 2-10 -> `outputs/clip_concepts/concept_rca_results.json` |
| `quickstart/remediation_demo.py` | remediation strategies and label-free triage on every real window -> `outputs/remediation_results.json` |
| `quickstart/drift_injection_benchmark.py` | 650-window injection benchmark (5 states x 10 seeds) -> `outputs/drift_injection_benchmark.json` |
| `paper/build_artifacts.py` | paper numbers (`paper/generated/results.tex`), tables and figures from the outputs |
| `dashboard/app.py` | Streamlit dashboard over the outputs |

Run order: the ACS and CLEAR-10 studies first (the concept probe and the
remediation study read their outputs), then the benchmark, then
`paper/build_artifacts.py`. Every script is seeded and deterministic.
