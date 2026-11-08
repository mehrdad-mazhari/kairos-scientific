# Final pre-data scientific implementation review

**RELEASE APPROVED** for `confirmatory_execution_v1.final_review.1`.

Exact implementation manifest SHA256:
`165b7e3e446ab04d8f59e72f694509ceea2deb19571a86dd8e9f09fce6e0522b`

Scientific freeze SHA256:
`f4f7b02e388e427727b86f82c2d5da3910348c3c98b6d7c6a04173100e1c9f49`

No independent validation devices were generated, previewed or evaluated; the
campaign was not executed. This is the assistant's user-authorized scientific
implementation review, not an external third-party audit.

## Conformity and fixes

All4383 frozen artifacts, root physical/statistical/audit dependencies, versions,
parameters, distributions and waveform roles/hashes verify unchanged. All five
nominal channels reproduce. Thirty nominal/saved-joint-stress CPU finer-step and
state-propagation comparisons pass, with maximum metric error
2.16457074487e-10.
The full conformance matrix is implementation_conformance_matrix.csv.

Only the previously unaccepted executor/report adapter was corrected; the
scientific protocol and original development data were not modified. Prior
proposal manifests/snapshots are preserved in draft_archive/before_final_review.
Corrections add full cohort metadata/schema/source/library/distribution/payload
verification, transaction evidence, immutable-cohort identity checks, strict
six-state/scalar/infidelity consistency and explicit physicality checks. Pending
work is cancelled on failure and returned in-flight results are preserved.
Adverse engineering outcomes are never filtered as numerical failures. Completed
records are verified and reused, and duplicate workers/alternate-cohort outputs
are blocked. New generation itself requires the accepted implementation release.
Any source, interpretation, protocol or version mismatch blocks execution;
there is no automatic approval, repair, redraw, solver retuning or optimization.
Earlier proposal audit/matrix documents are historical; this review, its matrix
and implementation_release_decision.json are the authoritative current decision.

## Statistical correctness

Count2000, reserved device seed941206, bootstrap seed941207, the six prespecified
pairs, physical model and engineering thresholds are unchanged. The frozen
empirical_cvar and paired_cvar_intervals sources are reused unchanged. Worst5%
tail mass is100 at n2000, with fractional boundary weighting, not101 whole devices.
CVaR is on infidelity, not fidelity or smoothed training loss.

Primary inference uses50000 paired common-index resamples. Ordinary95% and
six-contrast Bonferroni99.1666667% percentile CIs use quantiles .05/12 and1-.05/12.
The100000-replicate sensitivity calculation reuses the identical seed stream,
extending the50000-prefix. Original CIs/directions remain published; disagreement
is separately reported as inconclusive under the frozen sensitivity rule.
Secondary descriptive95% intervals use the same paired index stream. Three
optimizer seeds and54 reused-development trials are not independent devices.

The original one-sided95% P05 certificate is retained exactly (k=84, bound
F_(k), n2000). The simultaneous four-primary certificate uses the same method
at98.75% confidence (k=79). Nominal F>=.999, nominal and maximum sampled
leakage<=1e-4, P05 LCB>=.995 and existing waveform/numerical constraints are
unchanged. Diagnostic E is outside all six primary contrasts and cannot obtain
primary engineering eligibility, even if new outcomes are favorable.

## Numerical interpretation adopted before data

Numerical flags NEVER alter a statistical CI or its raw direction. The existing
2e-6 value is numerical reliability, not a new alpha, practical-effect margin or
statistical hypothesis. A contrast is NUMERICALLY_UNRESOLVED if any of:

1. Absolute point delta-CVaR<=2e-6.
2. A directional adjusted CI's boundary nearest zero is<=2e-6 away.
3. Direction can reverse under the sum of both candidates' maximum observed
   fidelity discrepancies on the20 frozen numerical-repeat indices.

The last condition is finite sampled evidence, not a uniform CVaR error bound.
Otherwise the label is CHECKED_NUMERICALLY_RESOLVED, not uniformly converged.
Bootstrap sensitivity is a separate statistical assessment. Unresolved or
bootstrap-unstable contrasts cannot support superiority claims. An engineering
failure never rewrites or erases a statistical CVaR result; it independently
prevents an engineering-qualified recommendation.

## Deterministic overall reporting convention

This is explicitly a pre-data IMPLEMENTATION INTERPRETATION, not an original
scientific rule or additional statistical superiority test. Individual contrast
results remain authoritative. Statistical results, numerical reliability,
engineering certificates and publication readiness are separately exported.

A qualified favorable contrast has the original adjusted CI favoring the
candidate, stable100k direction, a resolved checked numerical assessment and
simultaneous engineering PASS for both pulses. Overall labels summarize only
this directed favorable-order convention:

- CONFIRMED: all six favorable contrasts qualify.
- PARTIALLY CONFIRMED: at least one but fewer than six qualify.
- NOT CONFIRMED: none qualify.

All inferiority/inconclusive findings remain reported. NOT CONFIRMED does not
mean the research is invalid or unpublishable. Manuscript preparation is assessed
independently of favorable results; no hardware demonstration or journal
acceptance is implied. The exact interpretation is hash-bound in the release.

## Tests and remaining limitations

38 relevant automated tests passed, covering reference paired bootstrap and
Bonferroni quantiles, tail mass, P05 order indices, role exclusion, statistical/
engineering/numerical separation, all overall labels, accepted-release/source
binding, corrupted datasets/checkpoints, changed versions, no redraw, singleton
cohort, duplicate locks, failure draining, adverse-result preservation, no
resubmission, and mocked completed-grid/numerical-repeat transitions.
Tests used mathematical/cache fixtures and saved development rows only.

Bootstrap coverage is approximate; finite repeat checks are not uniform numerical
bounds. Duffing/RWA, quasi-static uniform uncertainty and Lindblad assumptions
remain modeled assumptions. No hardware evidence is added. The full independent
campaign has not run; release approval is not a confirmatory scientific verdict.

## Command — NOT EXECUTED DURING REVIEW

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kairos-mpl .venv/bin/python -u confirmatory_execution_v1.py --run --output results/confirmatory_validation_v1
```

The same command resumes the one saved cohort and verified completed records.
Changed hashes, ambiguous interruptions or numerical failures stop execution.
