# Final preconfirmatory scientific checks

**GO** for the next scientific stage; independent validation has not been generated or executed.

The original 27- and 54-trial experiments, saved ensembles and frozen optimization/source manifests remain unchanged. All 54 corrective checkpoints and aggregations passed integrity checks. New work is confined to this versioned directory.

Conventional was selected by the literal minimum feasible CVaR95. Continuation was not excluded: its best pulse differs by only about5.37e-7 in development CVaR, below the existing2e-6 numerical criterion. Fixed risk's best unfiltered pulse violates maximum ensemble leakage; it remains a diagnostic comparator. All54 candidates pass the later development P05 guard, so that additional draft guard changed no selection.

|Representative|Nominal F|CVaR95|Maximum selection leakage|Engineering eligible|Primary allowed after checks|
|---|---:|---:|---:|---|---|
|conventional|0.9996645463|0.00288117131|3.69030689e-05|True|True|
|risk|0.9996780708|0.00319133757|2.93193333e-05|True|True|
|continuation|0.9996770238|0.00288170882|5.63625296e-05|True|True|
|calibrated_drag|0.9996591328|0.00321135996|8.496656e-05|True|True|
|risk_unfiltered_diagnostic|0.9996645915|0.00293349336|0.000254158171|False|False|

The Hilbert checks use N5,6,7,8, the same exact sampled waveforms, 200 saved development selection scenarios and 10 saved training stress entries plus nominal. No new device draws or optimization runs occur. Absolute and relative errors are reported; finite sampled agreement is not uniform convergence or validation of the physical/noise assumptions.

Observable definitions are unchanged: each device fidelity and leakage is the mean over six computational input states. Maximum ensemble leakage is the maximum of these per-device means across the saved200 devices, not the maximum over arbitrary input states. State-specific results remain in channel records. Fidelity is unconditional; leaked outputs are never renormalized.

Independent CPU metric/physicality verification passed: True. Maximum observed dt/4 versus dt/8 or propagator versus state-solve metric discrepancy: 2.77641354e-10. Both propagation paths use QuTiP and may share an integration backend. This checks channel-versus-state propagation and metric construction, not independence of software implementations. The original GPU4/8 checks remain preserved as supporting evidence.

The mathematical dissipator construction, trace-preserving generator, sqrt(n) annihilation ladder and leakage projectors agree across dimensions. All returned density matrices are checked for trace, Hermiticity, positivity and computational-plus-leakage population balance with unchanged thresholds.

The frozen confirmatory plan specifies exactly2000 future devices with reserved seed941206, all frozen candidates on identical devices, CPU strict production solves, paired CVaR bootstrap with common index resampling and six-contrast Bonferroni correction. Existing one-sided95% distribution-free P05 certificates are preserved; an additional simultaneous engineering certificate uses the same method. Optimizer-seed variation is never counted as independent device sampling.

Protocol/source/candidate hashes are locked in freeze_manifest.json. Candidate and protocol files are read-only to prevent accidental changes; hash checking remains authoritative. This freeze is preparation only and grants no authorization to draw validation devices. Q1 submission readiness still requires confirmation and a defensible novelty/model-scope argument; numerical checks alone do not establish it.

Files: candidate_eligibility_audit.csv contains all54 trials plus calibrated DRAG; candidate_representatives.csv contains the five audited representatives. hilbert_convergence.csv preserves signed absolute changes; hilbert_convergence_with_relative_errors.csv adds relative leakage differences. Frozen arrays, coefficients/native parameters, exact source trials, optimization seeds, training configurations and costs are in frozen_candidates.json.
