import json

with open("outputs/folktables_drift_results.json") as f:
    data = json.load(f)

rca = data["2018"]["pipelines"]["KS Test"]["rca"]
changes = rca["distribution_changes"]

with open("_rca_output.txt", "w", encoding="utf-8") as out:
    out.write("Feature Importance Shifts (2014 vs 2018):\n")
    out.write("=" * 60 + "\n")
    items = sorted(changes.items(), key=lambda x: abs(x[1]["change"]), reverse=True)
    for k, v in items:
        ref = v["ref_importance"]
        test = v["test_importance"]
        change = v["change"]
        out.write(f"  {k:10s}  ref={ref:.4f}  test={test:.4f}  change={change:+.4f}\n")

    out.write("\nRecommendations from RCA:\n")
    for r in rca.get("recommendations", []):
        out.write(f"  - {r}\n")
    
    out.write("\nFeature-level drift (KS Test 2018):\n")
    detection = data["2018"]["pipelines"]["KS Test"]["detection"]
    out.write(f"  Drift detected: {detection['drift_detected']}\n")
    out.write(f"  Score: {detection['score']}\n")
    out.write(f"  Severity: {detection['severity']}\n")
    
    loc = data["2018"]["pipelines"]["KS Test"].get("localization", {})
    if loc and loc.get("feature_drifts"):
        out.write("\n  Individual feature drifts:\n")
        for fd in sorted(loc["feature_drifts"], key=lambda x: x["score"], reverse=True):
            out.write(f"    {fd['feature_name']:10s}  score={fd['score']:.4f}  drifted={fd['drift_detected']}  severity={fd.get('severity','?')}\n")

print("Output saved to _rca_output.txt")
