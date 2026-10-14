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
- **Per-prediction reliability adds little beyond confidence** for the studied
  models: the fused score's AUROC against errors matches confidence alone;
  OOD scores sit near chance for in-support shifts.
- **Importance-weighted triage is conservative under support shift.** When
  production leaves the reference support, importance-weighted validation cannot
  see gains that come from extrapolation, so the triage misses some useful
  retrains (measured in the benchmark).
- **Significance at large n.** With hundreds of thousands of rows tiny shifts
  are significant; effect sizes (KS statistic, Cramer's V, PSI) are reported
  next to every decision.
- **Weighted residual tests treat the BBSE weights as fixed.** Their sampling
  error is ignored, which makes the prior-vs-covariate decision slightly
  anti-conservative; the benchmark measures the resulting error rate.
- **Mixed drift.** Benchmark windows inject one type at a time.

## Scope

- Two real datasets (ACS Income, CLEAR-10), linear decision rules (logistic
  regression on encoded tabular features and on frozen embeddings).
- Batch execution only; no streaming runtime, scheduler integration,
  automated retraining or rollback.
- No enterprise controls (RBAC, audit-grade event model, SLO enforcement).
