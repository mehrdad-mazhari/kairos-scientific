# Code map and scientific audit corrections

The active workflow is `robust_optimization.run`, used by the CLI and desktop
application. The original `core.py`, `analysis.py`, `calibration.py`, and
`pulse-shaping.py` are exploratory scripts, not the production gate comparison.

| Component | Responsibility |
|---|---|
| `challenge_model.py` | Duffing Hamiltonian, Lindblad noise, six-state unconditional fidelity, total leakage and device scenarios |
| `pulse_shapes.py` / `pulse_methods.py` | Reference construction, hardware limits, analytic recipes and exact reconstruction |
| `reference_calibration.py` | Nominal-only calibration of each reference at fixed shape and duration |
| `robust_optimization.py` | Constraints, training archive, finalists, frozen selection and held-out validation |
| `nelder_mead.py` / `search_comparisons.py` | Multistart Nelder–Mead and optional random search |
| `grape.py` | Dissipative Fourier-control refinement with analytic derivatives |
| `study_validation.py` | Objective redundancy, paired comparisons and Hilbert-space truncation checks |
| `challenge_report.py` / `stage_report.py` / `pulse_diagnostics.py` | Reports, plots and diagnostic sweeps |
| `optimization_inputs.py` / `optimization_ui.py` | Shared configuration validation and optional Tk desktop interface |

## Corrections

- **Calibration confounding:** each textbook reference has a separately named,
  nominally calibrated comparator. Gain, phase ramp and applicable DRAG beta
  are fitted using nominal infidelity only. Shape parameters and duration stay
  fixed. Calibration cost and optimizer termination status are saved. The best
  feasible point is retained even if the evaluation budget is exhausted.
- **Optimizer replacement:** bounded multistart Nelder–Mead is the primary
  search. The population-based optimizer and its dependency have been removed.
  The first restart starts from calibrated DRAG; later seeds perturb that point.
  The scalar loss, constraints and independent validation remain unchanged.
  Search histories, convergence status, actual channel solves and wall times
  are exported. An optional random-search benchmark is disabled by default.
  GRAPE reports its own evaluated training loss rather than copying its seed.
- **Hardware limits:** raw reference amplitudes are projected into the hardware
  disk including manufacturing headroom; the scale is saved in the recipe.
  Projection can reduce rotation area below pi, which is then visible in
  nominal results. Calibrated references obey the same amplitude constraint.
  Training and selection use the greater of configured headroom and the
  configured manufacturing amplitude range.
- **CORPSE:** the exported waveform now has equal-height plateaus and exact
  segment areas in ratio 7:5:1. One-sample edge ramps are simulated explicitly.
  An independent two-level integration of those actual samples verifies small
  detuning suppression. Its transmon leakage can still be substantial; the
  software reports that result. Longer composites remain comparison-only for
  a fixed-duration task, with decoherence over their actual durations.
- **Hilbert-space truncation:** the default production model has five levels.
  All population in n>=2 counts as leakage. GRAPE and population plots use the
  same dimension. Frozen-pulse checks compare 3–6 levels plus production and
  production+1 for nominal, worst-fidelity held-out, and coherent nominal
  propagation. Overall quality cannot pass a failed truncation check or a
  three-level production model.
- **Higher-level noise:** the supplied n=1,2 lifetimes are preserved. For n>2,
  relaxation rates scale as n/2 relative to level 2; dephasing projector rates
  scale as (n/2)^2. This is an explicit extrapolation, not measured device data.
- **Objective redundancy:** correlation and normalized singular values are
  exported for the feasible training archive. A nearly one-dimensional
  objective set is not presented as three independent physical trade-offs.
- **Uncertainty consistency:** training stress cases now use configured
  manufacturing ranges rather than hard-coded values. Selection and validation
  remain separate, identically shared device samples across methods.
- **References and names:** DRAG attribution includes Motzoi et al.
  [0901.0534](https://arxiv.org/abs/0901.0534) and the analytic-control treatment
  of Gambetta et al. [1011.1949](https://arxiv.org/abs/1011.1949).
  Poudel & Vavilov's [1008.2554](https://arxiv.org/abs/1008.2554) remains a
  secondary discussion, not the original source. The legacy JSON key
  `higher_order_drag` is labeled **Hermite DRAG**; it is not the higher-order
  adiabatic expansion. CORPSE cites
  [Cummins et al.](https://arxiv.org/abs/quant-ph/0208092).
- **Headless tests:** shared configuration construction resides in
  `optimization_inputs.py`, so workflow tests no longer import or stub Tkinter.

## What these changes establish

The regressions check numerical implementation and specific physical properties.
The end-to-end smoke configuration deliberately uses small search and sampling
budgets; its quality verdict is a diagnostic, not a publication result.
A full production run and repeated-seed study are still needed to quantify
performance. Paired held-out mean-difference intervals are exploratory, with no
multiple-comparison correction. Truncation checks cover selected stress points,
not a uniform convergence proof over the complete uncertainty domain.

The model remains a rotating-wave Duffing approximation with phenomenological
noise. It does not establish hardware performance or model spectator-qubit
crosstalk. Scientific novelty, a duration/decoherence trade-off study, and journal
suitability cannot be established by repairing this implementation.


## Statistical reliability implementation

The pipeline now reports the support status of the unchanged exact one-sided
95% binomial P05 bound. Configured engineering thresholds replace hard-coded PDF
thresholds in sample pass fractions. Nominal quality, statistical certification,
optimizer termination, numerical convergence and engineering acceptance are
separate statuses. Explicit older budgets remain supported; new defaults are
200 selection draws, 2000 independent validation draws and five 500-call restarts.

`pipeline_settings.py` resolves legacy/new settings and the shared quick profile.
`statistical_certification.py` owns exact bounds and engineering failure reasons.
`robustness_diagnostics.py` owns bounded post-validation uncertainty searches.
The desktop and CLI share run-setting parsing. Regression coverage includes
independent draws, frozen-before-validation hashes, shared candidate samples,
exact confidence coverage, insufficient samples, backwards-compatible settings,
reproducible bounded diagnostics and failure reports.

Worst-case diagnostics cannot alter the pulse. They expose observed sensitivity
and joint violations, not certified global bounds. Strict solves and multilevel
checks now also inspect diagnostic extrema. The Hamiltonian, five-level default,
Lindblad model, six-state unconditional fidelity and uncertainty probability laws
have not been changed. Smoke tests do not constitute full statistical validation.


Verification of this implementation: 68 tests passed in about 12.2 seconds.
The [small integration run](results/reliability_smoke/report.md) used 4 selection
devices, 8 independent validation devices, two 40-call restarts, reduced GRAPE
and enabled bounded diagnostics. Wall time was 192.6 seconds on one BLAS thread;
54 unique diagnostic channel solves took about 4.1 seconds. Nominal, numerical
and truncation checks passed. Statistical certification correctly failed because
8 samples cannot establish a nontrivial distribution-free P05 bound. The observed
fidelities remained nonzero. Frozen pulse, reconstructed recipe, draw hashes and
source hashes were verified; the new plots were visually inspected. No full-budget
statistical validation or native Tk layout test was performed.


## Risk-aware GRAPE verification

The previous 4/8 counts came from an explicit smoke configuration; defaults were
not incorrectly propagated. The new risk method uses a softplus CVaR envelope
with analytic dissipative gradients and an explicit approximation bound. NM and
both GRAPE methods now share a fixed training ensemble when risk mode is enabled.

77 tests passed in about 12.2 seconds. The new integration experiment
(results/risk_aware_verified) used 8 common selection devices, 64 independent
validation devices, one optimizer seed and one GRAPE iteration. It took 268.8
seconds. Conventional GRAPE took 2.42 seconds and risk-aware GRAPE 6.72 seconds
inside that run; model/gradient call counts are saved separately. The selected
risk-aware pulse passed the configured engineering checks and exact P05
certification (lower bound 0.995844923). This is a small-budget result under the
configured simulation assumptions, not a full-budget or experimental claim.

Risk-aware empirical P05 was 0.996802990 versus conventional GRAPE 0.996771456.
Its CVaR95 infidelity was worse: 0.003668594 versus 0.003554461. Mean fidelity
was also lower. Thus selection-set success did not establish held-out CVaR
superiority. Paired exploratory mean comparisons and multiplicity-adjusted
approximate intervals are in research_comparisons.json; no tail-difference
confidence interval or confirmatory superiority claim is made.

Modified implementation files: grape.py, robust_optimization.py,
pipeline_settings.py, statistical_certification.py, optimization_inputs.py,
optimization_ui.py, challenge_config.json, pulse_methods.py, challenge_report.py,
test_workflow.py and test_statistical_reliability.py. New files:
risk_aware_grape.py, research_figures.py, test_risk_aware_grape.py and
RISK_AWARE_GRAPE_METHOD.md. TECHNICAL_GUIDE.md documents the configuration changes.

Next research experiments: execute the predeclared full configuration and all
optimizer seeds; compare lambda=0 and nonzero lambda with matched training
ensembles/costs; assess smoothing and interpolation-substep errors against
production solves; develop prespecified tail-difference intervals. Hyperparameter
selection must use training/tuning data, never these held-out observations.
The full 2000-device, repeated-optimizer-seed experiment remains unexecuted
because of its substantially larger dissipative simulation cost.

## Publication benchmark extension

New files: research_benchmark.py, research_statistics.py,
research_benchmark_report.py, research_config.json, test_research_benchmark.py,
and RESEARCH_BENCHMARK_PROTOCOL.md. grape.py now supports explicit per-run
objective/scenario-pass/wall limits and seeded paired initial perturbations.
risk_aware_grape.py supports mean/worst/CVaR/leakage ablation weights. The
Hamiltonian and original engineering thresholds remain unchanged.

83 tests passed, including every ablation gradient, distinct/reproducible
initialization, budget termination, profile guards, bootstrap estimators and
checkpoint reuse. The quick campaign in results/research_benchmark_complete
finished in 404.6 seconds. It used 4 selection devices, 8 validation devices,
one optimizer seed, 8 NM calls and 3 Fourier objective calls / 1-second wall
caps in the two resource regimes. It generated all requested CSV/JSON outputs,
robustness maps, eight figures and a research report.

Both Fourier methods retained identical early-stop controls in that pilot.
Their paired CVaR differences were zero; no algorithmic superiority is supported.
None of the 30 frozen method/trial groups passed the complete engineering
assessment. Some candidates failed nominal qualification and all eight-device
certification results remain insufficient evidence. This is not a reversal or
reinterpretation of the earlier, different 64-device integration experiment.

A resume test reproduced identical quantitative results while production solves
and reoptimization were explicitly blocked. Frozen hashes and the validation
barrier were checked. Rendering was subsequently polished without changing
quantitative data; report_render_sources.json records the rendering version and
the original renderer is retained. Completed protocol/source guards remain
strict: replay with the pinned original source to reproduce that exact run.

The full profile was prepared in results/research_full_plan, not executed. It
requires the full lambda/mu grids, ten optimizer trials, 100 random training
devices, 2000 validation devices and 500-call robustness searches. Completing
this campaign, matched-total-tuning-cost replication, independent outer optimizer
trials, and larger-bootstrap tail-coverage checks remain necessary before a
publication claim. Per-run resource caps are matched, but aggregate risk tuning
costs are larger and explicitly reported; they are not an equal-total-cost study.

## Audit of identical quick GRAPE results

The training-only audit is in `results/optimizer_diagnostic_final/`.
All seven deterministic three-call replays returned waveforms bitwise identical
to their saved outputs. Raw-seed and Fourier-start evaluations consumed two of
three calls; the remaining trial did not improve the retained feasible point.
The common perturbed start therefore explains identical results without an
aggregation or zero-gradient defect. The historical risk selection chose
lambda=mu=0 after ties, which cannot establish useful risk hyperparameters.

Physical adjoint directional checks on 100 distinct scenarios agreed with
finite differences within 3e-11. A separate 100-loss algebra check verifies the
infidelity CVaR sign, lambda/mu derivatives, regularization and smoothing bound.
The two-harmonic fit preserved nominal fidelity (0.999498); the .05 coefficient
perturbation reduced it to 0.978872. Constrained least squares is available by
explicit opt-in; initialization defaults and engineering thresholds are unchanged.

With the same qualified five-harmonic start and fixed 12-device training set,
conventional and risk GRAPE reduced their own objectives by approximately 20%
and changed coefficients. Both stopped at 12 iterations, not convergence.
Their nominal fidelities (0.999490802 and 0.999445648) remained below calibrated
DRAG (0.999513778), while leakage decreased. Neither result supports risk
superiority. Production waveform reloads reproduce the reported metrics exactly;
nominal surrogate errors decrease with substep refinement and are below the
unchanged solver tolerance. Ensemble-wide production agreement and independent
statistical comparison remain unresolved.

The research profile's scenario-pass cap was not forwarded to GRAPE; this is now
corrected. Budget resolution also clears inherited caps before selecting the
intended regime. Full settings remain 200 selection / 2000 validation devices,
100 random training devices, ten optimizer seeds and 40 iterations / 500 calls
or 300 seconds. No full campaign or independent validation was run for this audit.

## Qualified initialization and continuation development

Implemented opt-in qualified five-harmonic multistart and a separate experimental
continuation method; legacy initialization and fixed-weight risk GRAPE remain.
All starts/stages share aggregate resource accounting. Nominal qualification
uses the unchanged production solver/thresholds, and full research defaults now
request 150 total iterations, 1000 calls, five restart allocations and five
harmonics. Quick research mode explicitly preserves its legacy initializer and
reduced caps. The separate development protocol is GRAPE_DEVELOPMENT_PROTOCOL.md.

The executed results are in results/grape_development_short. This explicit
diagnostic used 100 iid training devices plus ten stress cases, 200 common
selection devices, seeds 7500/7501, 24 objective calls and 24 maximum iterations,
with two substeps. Six trials finished in 2131.30 seconds; the full three-regime,
150-iteration/1000-call development experiment remains unexecuted. Neither the
equal-pass nor equal-wall comparisons were executed. No validation was drawn.

101 tests pass. Fresh physical directional checks on 100 distinct scenarios
agree with finite differences within 9.1e-10; objective algebra/weight signs and
smoothing are separately tested. Ten admitted multistart pulses qualify;
both legacy perturbed pulses fail nominal qualification. The new fit starts at
F=0.999518779. Mean final nominal fidelities are 0.999657709 conventional,
0.999676333 fixed risk and 0.999681725 continuation. Mean nominal leakage falls
from 4.90555e-5 to 2.49746e-5, 2.25502e-5 and 1.96030e-5 respectively.

Mean per-trial selection CVaR95 is 0.003225365 conventional, 0.003132222 fixed
risk and 0.003123094 continuation. Positive conventional-minus-risk differences
are 9.31430e-5 and 1.02271e-4. Both risk methods have lower mean/P05 selection
fidelity than conventional; no uniform improvement is established. They use
twice the surrogate passes per objective call and about twice the runtime
(~393 versus ~195 seconds per trial). All six trials exhaust their call budgets;
none meets the convergence criteria. These are selection observations, not an
independently validated or equal-wall/equal-pass superiority result.

Continuation returns the same final pulse for both seeds. All four stages change
their input waveforms, while differing final-stage random starts fail to beat
the common deterministic warm-start candidate. This is not stalled optimization
or extra independent replication. Initializers and training archives are saved.

Production waveform reloads reproduce nominal metrics exactly. Nominal and three
stress-case checks across two/four substeps have maximum error 8.17e-7, below
the unchanged 2e-6 tolerance; this is not uniform ensemble convergence. Completed
checkpoint reuse preserves quantitative CSV hashes with optimization and new
selection solves blocked. Numerical sources, reference bytes and subsequent
input-guard/rendering changes are archived separately for reproducibility.

The reused DRAG reference has F=0.999513778 but only five calibration calls and
did not converge. The diagnostic cannot establish an advantage over fully
calibrated DRAG. Stronger nominal calibration, full matched-resource comparisons,
seed variability and final protocol freeze are required before independent
validation or a publication claim. Physics, noise, manufacturing distributions
and certification source hashes remain unchanged.

A subsequent nominal-only reference preparation used a 500-call solver cap
(414 unique channel evaluations, 30.74 seconds). It reached strict nominal
F=0.999659133 and leakage=4.26981e-5; all fifteen admitted starts qualify. It
still reported budget exhaustion, not convergence. No selection or validation
data tuned it. results/grape_development_full_plan uses this stronger common
source for a future rerun; the six-trial pilot remains tied to its original
five-call reference. A first ancillary export attempt lost its calibration
record after a list/array argument error; its unrecorded work is bounded by
500 nominal channels and documented separately. The correction checkpoints
calibration before export. No optimizer trial was discarded or retuned.

## Optional JAX acceleration

The optional training backend reuses the CPU five-level Liouvillians and
observables, with JIT compilation, scenario vectorization and reverse
differentiation through midpoint matrix exponentials. Conventional GRAPE,
fixed-risk GRAPE and CVaR continuation share this backend. CVaR weights use
the complete ensemble rather than independent batch tails. Both precisions are
explicit and subject to the unchanged production metric tolerance.

This changes the execution of the training surrogate, not the scientific model
or the final certification rule. Production nominal qualification, selection
and final statistical certification stay on CPU QuTiP. Source hashes of
challenge_model.py, challenge_config.json and statistical_certification.py
match the preceding development snapshot. Backend accuracy and timing do not
establish optimizer convergence or a risk-aware advantage.

The dedicated backend benchmark uses seed 88100 and common prefixes of 2,000
distinct uncertainty scenarios. These are numerical test devices, not reserved
final validation devices. It records serial production CPU and analytic-gradient
references, a matched JAX-CPU control and actual CUDA measurements separately.
See JAX_ACCELERATION.md for commands and qualification requirements. No full
research campaign is part of this acceleration task.

The actual RTX 2080 SUPER benchmark qualified float64, with maximum fidelity
and leakage errors of 1.49e-7 and 1.73e-8 versus strict production QuTiP.
At 2,000 scenarios, warm GPU speedups over the existing serial CPU were 14.39×
for propagation, 3.20× for conventional gradients and 5.21× for risk gradients.
The separate matched JAX-CPU control extends only through 100 scenarios.
Float32 failed the unchanged tolerance and remains unqualified; successful
gradient checks alone did not establish accurate states or objectives.
Actual accuracy/initialization CPU fallback, all 2,000 final density operators,
and all three optimizer paths on real float64 CUDA were checked. All 115 unit
tests passed. See results/jax_rtx2080_qualified/gpu_acceleration_report.md and
the accompanying raw measurements. These are backend qualification results,
not optimizer superiority or independent statistical certification.
