# Limitations And Gaps (Current)

Verified limitations of the implemented method. Each is measured or argued in
the paper; none is hidden.

## Method

- **Changes in P(Y|X) are invisible without labels.** When the inputs and the
  model's outputs do not change (concept drift, label noise), no label-free
  signal can see them; the delayed-label audit is required.
- **CBPE goes stale.** The confidence-to-accuracy map holds only while
  P(correct | confidence) is unchanged; it is optimistic under covariate shift
  outside the reference support and blind to P(Y|X) changes.
- **Per-prediction reliability adds little beyond confidence** on ACS and
  CLEAR-10: the fused score's AUROC against errors matches confidence alone.
  The fusion is learned only with at least ten calibration errors per signal
  (Peduzzi et al., 1996); on the gas sensor data (4 and 11 errors) the risk is
  1 - confidence alone.
- **CBPE needs calibration rows at every confidence level.** Bins with fewer
  than five calibration rows keep the model's own confidence. Even so, CBPE
  stays optimistic when drift breaks the confidence-to-accuracy relation (gas
  batch 8: estimate 0.618, true 0.330).
- **Importance-weighted triage is conservative under support shift.** When
  production leaves the reference support, importance-weighted validation cannot
  see gains that come from extrapolation. With a labelled recent window the
  triage is almost always right on the real settings; without one (benchmark,
  geographic windows) it misses most useful retrains.
- **Significance at large n.** With hundreds of thousands of rows tiny shifts
  are significant; effect sizes (KS statistic, Cramer's V, PSI) are reported
  next to every decision.
- **Weighted residual tests treat the BBSE weights as fixed.** Their sampling
  error is ignored, which makes the prior-vs-covariate decision slightly
  anti-conservative; the benchmark measures the resulting error rate.
- **Mixed drift.** Benchmark windows inject one type at a time.
- **Design B is the honest test.** Design A partly matches the tests (a
  feature-wise shift, a one-feature concept region, uniform flips); the
  held-out mechanisms of design B are where the method's limits show. The
  diagnosis was frozen before design B was run.
- **Explanations need the logistic model.** SHAP (exact explainer cost) and
  class activation maps (linear head) are produced for logistic regression only.

## Scope

- Four real datasets (ACS Income, ACS Employment, gas sensor array, CLEAR-10)
  and two model families (logistic regression, histogram gradient boosting);
  no deep model trained end to end.
- Batch execution only; no streaming runtime, scheduler integration,
  automated retraining or rollback.
- No enterprise controls (RBAC, audit-grade event model, SLO enforcement).
