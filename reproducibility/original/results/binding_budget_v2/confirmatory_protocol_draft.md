# Independent confirmatory protocol draft — NOT EXECUTED

This draft does not authorize validation execution. Freeze and review this document,
candidate manifest, source hashes and solver configuration before generating any devices.
No independent validation devices have been drawn or inspected by this analysis.

## Population, estimand and frozen candidates

Generate exactly 2000 iid devices once, with previously unused seed **941206**,
using the unchanged `challenge_model.monte_carlo_scenarios` function. Freeze its
source hash from the corrective protocol and use the existing independent uniform
amplitude factor [0.97,1.03], detuning [-3,3] MHz, anharmonicity shift [-8,8] MHz,
and four independent lifetime scale factors [0.8,1]. These are modeling assumptions.
All methods receive exactly the same device rows. Do not generate these rows now.
The device is the sampling unit; method results on a device are paired.

`confirmatory_frozen_candidates.json` contains one development-selected feasible
waveform per method, each SHA256, exact samples, originating trial and frozen
optimizer options. Select minimum development CVaR95 among candidates meeting
all nominal, hardware, sampled leakage and development P05 criteria, breaking ties
lexicographically by trial name. Also freeze the strongest overall feasible candidate.
If any method has no eligible waveform, do not substitute a violating pulse or change
thresholds: revise the comparison family explicitly before locking this draft.
The final comparisons estimate differences between these specific fixed pulses,
not average performance of an optimizer or computational-efficiency superiority.
No algorithm, hyperparameter, waveform, stopping rule or candidate is selected
after seeing independent data. Archive original candidate arrays/options/hashes.

## Endpoints and analysis

Primary estimand for each frozen pair: Δ = CVaR95(reference infidelity) −
CVaR95(candidate infidelity). Positive favors the candidate. Use the existing exact
empirical upper-tail CVaR with fractional boundary weights (`empirical_cvar`), not
smoothed training CVaR. At n=2000 the tail contains approximately 100 devices.
The family consists of all three planned contrasts: fixed risk vs conventional,
continuation vs conventional, continuation vs fixed risk. Include every contrast
regardless of observed direction; do not drop unfavorable results.

Use 50000 paired nonparametric bootstrap replicates, resampling 2000 device indices
with replacement with bootstrap seed 941207. Use each index vector jointly for all
methods and contrasts; recompute empirical CVaR per method in each replicate.
The ordinary paired percentile bootstrap yields approximate device-sampling CIs;
it is not an exact distribution-free CVaR guarantee. Report point estimates,
unadjusted two-sided 95% CIs and simultaneous Bonferroni two-sided 98.333333% CIs
for the three primary contrasts (quantiles .05/(2*3), 1−.05/(2*3)). Declare tail
improvement only if the simultaneous lower bound is >0 and numerical sensitivity
checks cannot reverse its sign. No favorable practical-effect margin is invented.
If the interval includes zero, report inconclusive; if its upper bound is <0,
report inferiority. Report bootstrap Monte Carlo sensitivity with 100000 replicates
using the same fixed resample stream extended; disagreement means inconclusive.
Do not try multiple bootstrap methods and select a favorable one.

Secondary endpoints: paired mean fidelity, paired P05 fidelity, nominal fidelity,
mean leakage, maximum sampled leakage and leakage exceedance count. Bootstrap
mean/P05/mean leakage using the same paired device indices, with descriptive 95%
CIs explicitly labeled secondary, without confirmatory superiority declarations.
Sample maximum leakage and exceedances are observations, not global worst-case
bounds; do not use a bootstrap CI on the maximum as safety certification.
Optimizer-seed spread in development remains descriptive: do not resample or pool
54 trials as independent device ensembles, nor use a three-seed bootstrap to claim
population-level algorithm superiority. This frozen-waveform confirmatory analysis
resamples devices only, unlike a future algorithm-level crossed-seed experiment.

Paired resampling semantics follow the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

## Engineering and numerical criteria

Keep nominal fidelity ≥.999, nominal and maximum sampled leakage ≤1e-4, amplitude
≤80 MHz including 1.03 manufacturing headroom, zero endpoints and 15.6 ns duration.
Preserve the project's `statistical_certification.tail_bound_details`: sorted
fidelities, k = binom.ppf(.05,n,.05), one-sided 95% lower bound F_(k), requiring
bound ≥.995 with nontrivial support. Report the existing per-pulse certificate
unchanged. For a simultaneous engineering claim across three pulses additionally
compute the same binomial order-statistic method at confidence 1−.05/3; require
all corresponding bounds ≥.995. This strengthens multiplicity treatment without
weakening the existing method or threshold. A positive CVaR contrast does not
override an engineering failure. No convergence requirement is imposed on a
valid resource-limited optimizer; its achieved status stays reported.

Use trusted CPU QuTiP for every validation channel and all final certification.
Before drawing devices, recheck nominal and saved training stress cases with
production strict/tighter solver settings and four/eight interpolation substeps;
maintain fidelity/leakage tolerance 2e-6, gradient atol2e-7/rtol2e-6, trace and
Hermiticity errors ≤2e-6 and minimum density eigenvalue ≥−2e-6. A gate failure
stops validation; do not switch precision or relax thresholds. Record all tolerances
from source snapshot. Freeze an evenly spaced set of 20 future device indices
[0, 105, 210, 315, 420, 526, 631, 736, 841, 946, 1052, 1157, 1262, 1367, 1472, 1578, 1683, 1788, 1893, 1999] for repeat
strict/tighter CPU propagation; These exact indices are frozen in this draft.
If a discrepancy exceeds 2e-6, stop and investigate; any rerun policy must be
frozen before outcomes and reported. Do not selectively replace bad devices.
Hilbert truncation and solver tolerances must pass the project's existing convergence
checks: five-level numerical equivalence does not prove convergence in level count.
Prior evidence must cover selected waveforms; otherwise run development-only
5/6-level checks first, without inspecting validation draws. No new physical noise
model or uncertainty distribution is introduced by this draft.

## Reproducibility and reporting

Archive data-independent source/config hashes, waveform hashes, seed declarations,
exact statistics code and version, package/hardware environment and full analysis
plan before execution. Log all CPU solves, retries and failures. Preserve nominal,
public-grid, truncation and adversarial diagnostics required by the project, using
only frozen diagnostic rules and keeping them separate from statistical sampling.
Report negative and mixed results. Seek independent implementation/hardware
replication before journal claims extending beyond this modeled distribution.
