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

## Corrections (2026-10-15)

A code review after design B was first run found two errors in the diagnosis,
both corrected without regard to the results; every result was regenerated.

- The label-free step tested features inside a gatekeeping rule that let the
  prediction and MMD tests move up while features kept their level, so alpha
  was spent twice: a simulated family-wise error rate of about 6.5% at a
  nominal 5% (`examples/quickstart/fwer_check.py`). It is now one weighted Holm
  family (`diagnosis.label_free_decisions`).
- The audit kept a nominal category when it occurred often enough in the
  reference and audit rows together, not in the audit as documented, so the
  error-structure tests had hundreds of near-empty cells. Categories are now
  counted in the audit, interaction columns of unvisited bins are dropped, and
  the fits are exact maximum likelihood.

`outputs/benchmark/prefix_verdicts.json` keeps the verdicts from before the
correction so the change can be checked. The correction changed 115 of the
5,300 benchmark verdicts, all of them between concept drift and label noise
(the audit fix); the label-free fix alone changed none. Type identification
moved from 87.1-94.3% to 85.7-94.3% on design A and from 63.2-81.8% to
62.4-81.5% on design B. On the real windows every type verdict is unchanged;
a few localized feature sets changed by one or two features.

Each stage of the diagnosis is tested at alpha, so the chance that a window
without drift gets a drift verdict is at most 2 alpha. On the 200 design A
controls the diagnosis raised 13 false alarms (6.5%): 7 in the label-free
stage and 6 in the audit.
