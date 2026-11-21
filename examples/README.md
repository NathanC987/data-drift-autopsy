# Examples

Every study script takes `--model lr` (logistic regression, the primary model)
or `--model hgb` (histogram gradient boosting).

| Script | What it produces |
|---|---|
| `quickstart/simple_example.py` | minimal `DriftPipeline` run on synthetic data |
| `quickstart/folktables_demo.py --task {income,employment}` | ACS study (CA 2014 reference; CA 2015-2019, 2021-2023 and TX/NY/FL/WA 2014 windows) -> `outputs/folktables_drift_results.json` (Income, lr) or `outputs/folktables_{task}_drift_results_{model}.json` |
| `quickstart/gas_drift_study.py` | UCI gas sensor array drift study (batches 1-2 reference; batches 3-10) -> `outputs/gas_drift_results_{model}.json` |
| `quickstart/clear10_tabularize_demo.py` | frozen ResNet-18 embeddings for every CLEAR-10 bucket -> `outputs/clear10_tabularized_demo/` |
| `quickstart/clear10_drift_study.py` | CLEAR-10 study (bucket 1 reference; buckets 2-10) -> `outputs/clear10_drift_results.json` (lr) or `outputs/clear10_drift_results_hgb.json` |
| `quickstart/clip_concept_probe_demo.py` | CLIP concept probe for buckets 2-10 -> `outputs/clip_concepts/concept_rca_results.json` |
| `quickstart/remediation_demo.py` | remediation strategies and label-free triage on every real window -> `outputs/remediation_results.json` (lr) or `outputs/remediation_results_hgb.json` |
| `quickstart/drift_injection_benchmark.py --dataset --design --model` | injection benchmark (5 states x 10 seeds per configuration; design A matched mechanisms, design B held-out mechanisms) -> `outputs/benchmark/{dataset}_{design}_{model}.json` |
| `quickstart/benchmark_summary.py` | baselines, ablations, meta-classifier transfer and paired tests across the benchmark configurations -> `outputs/benchmark/summary.json` |
| `paper/build_artifacts.py` | paper numbers (`paper/generated/results.tex`), tables and figures from the outputs; `--flatten` writes the single-file submission |
| `dashboard/app.py` | Streamlit dashboard over the outputs |

Run order: the ACS, gas and CLEAR-10 studies first (the concept probe and the
remediation study read their outputs), then the benchmark and its summary,
then `paper/build_artifacts.py`. Every script is seeded and deterministic.
