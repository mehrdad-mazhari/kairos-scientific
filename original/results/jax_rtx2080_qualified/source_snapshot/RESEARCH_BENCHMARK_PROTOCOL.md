# Publication-oriented paired benchmark

## Entry points and budgets

`research_benchmark.py` extends, rather than replaces, the normal optimizer and
desktop application. `research_config.json` is a separate campaign configuration;
it references the unchanged physical device in challenge_config.json.

| Setting | Full research | Explicit quick profile |
|---|---:|---:|
| Selection devices | 200 | 4 |
| Independent validation devices | 2000 | 8 |
| NM calls per restart | 500 | 8 |
| NM seeds | 7,17,27,37,47 | 7 |
| Random training devices, plus 10 fixed stress cases | 100 | 2 |
| Paired GRAPE optimizer seeds | 6200 through 6209 | 6200 |
| Worst-case calls per objective, plus axis probes | 500 | 8 |
| Objective/gradient cap per Fourier optimization run | 1000 | 3 |
| Wall cap per optimization run | 1800 seconds | 1 second |
| Fourier total iterations / harmonics / interpolation substeps | 150 / 5 / 4 | 1 / 2 / 1 |
| Restart allocations | 5 | 1 |
| Initializer | qualified Fourier multistart | legacy radial + .05 perturbation |
| Bootstrap repetitions | 2000 | 100 |
| Amplitude-detuning map points per axis | 15 | 3 |

Full mode explicitly rejects undersized research budgets. Quick mode requires
`--quick` and is saved as quick. Neither mode silently changes to the other.
All effective settings and source hashes are printed/saved before optimization.
The extra `scenario_passes` field is retained as a configuration sanity check;
the second executed resource regime is wall-budgeted, not claimed to match
production and surrogate channel solves. Those operations have different costs.

```bash
.venv/bin/python research_benchmark.py --output results/research_full --prepare-only
.venv/bin/python research_benchmark.py --output results/research_full
.venv/bin/python research_benchmark.py --quick --output results/research_quick
```

A prepared protocol is not an executed experiment. The prior 8/64 artifact was
explicitly a reduced integration run; it must not be relabeled as full research.

## Predeclared hypotheses and estimand

Primary null: Risk-Aware GRAPE does not reduce CVaR95 infidelity relative to
conventional GRAPE. Alternative: the paired risk-minus-conventional difference
is negative. The primary estimand averages per-trial device CVaR differences
over matched optimizer trials, rather than pooling every device/optimizer result
into a single unstructured distribution. P05 improvement is secondary.
Significance level is 0.05 with a Bonferroni family adjustment for the two
primary resource-regime comparisons. Hypotheses are saved before validation.

`research_statistics.py` uses a paired crossed percentile bootstrap: resample
optimizer-trial indices together for both algorithms, and independently resample
common device indices together for both algorithms and every trial. It reports
mean, P05 and CVaR effect sizes and approximate intervals. Bootstrap tail coverage
is asymptotic, not distribution-free. Secondary intervals remain exploratory;
small single-trial/device pilots cannot support superiority. The implementation
requires at least 10 trials and 2000 devices even to flag bootstrap-supported
improvement. This flag is conditional on the chosen hyperparameters and bootstrap
assumptions, not a platform-wide algorithmic theorem. Independent outer optimizer
trials and a confirmatory replication remain advisable after tuning.
Absolute P05 certification retains the existing exact binomial order statistic.
Individual marginal-interval overlap is never used as a superiority test.

## Tuning and ablations

Lambda grid: 0,0.1,0.5,1,2,5, with other objective settings initially fixed.
Then mu=0,2,5 is studied at the selection-chosen lambda. Group selection first
minimizes the number of ineligible trials, then mean selection CVaR. No validation
metric participates. Per-trial weights, gradients and histories are checkpointed.
`cvar_sensitivity.csv` explicitly contains selection-only statistics for both
lambda and mu phases. Its curves are not held-out performance estimates.

A: mean infidelity, no objective leakage term.
B: smooth worst infidelity, no mean/CVaR/leakage term.
C: CVaR only.
D: mean plus CVaR.
E: mean plus CVaR plus fixed nonzero initial leakage weight mu=2.

Objective constraints remain active for every ablation, including those with
zero leakage weight. The worst-loss smoothing is log-sum-exp, with gap at most
tau*log(N); the existing CVaR softplus gap is at most tau*log(2)/(1-alpha).
Gradient verification covers all five cases. CVaR itself is not claimed novel.

## Fairness and initialization

All NM and Fourier runs share the same fixed training devices. All candidates
share the same selection devices. Conventional and risk methods use paired
seeded Fourier perturbations (initialization_scale=0.05), with the same initial
coefficient realization for a given seed. Merely changing a seed with one
unperturbed deterministic restart did not create independent trials; the campaign
now explicitly perturbs starts and tests reproducibility/distinctness.

Two regimes cap per-run objective/gradient evaluations or elapsed optimization
time. Wall limits are checked between expensive evaluations and can overshoot
by one evaluation. Algorithms may converge or exhaust iteration limits before
using their cap. Report actual usage, not assumed equality. Objective calls,
gradients, surrogate scenario passes, production solves, termination flags and
wall time are saved. Static nominal calibration is a shared one-off expense.
Risk tuning uses many more runs than a fixed conventional objective: aggregate
tuning/selection cost is exposed and is not an equal-total-campaign-budget claim.
A separate matched-total-tuning-cost replication is still needed for strong
algorithm-efficiency conclusions.

## Freeze barrier, validation and resumption

Training seed=6100, selection seed=6101 and validation seed=6102 are distinct.
No validation dataset is generated before all hyperparameters, ablations and
waveforms are frozen in frozen_manifest.json. Fresh validation devices are
shared by every frozen method. No validation feedback changes the winner.

Complete optimization/selection jobs are atomic JSON checkpoints. Individual
production validation solves are atomic waveform-hash/device-index checkpoints;
per-trial JSONL records support continuation. Qualification, worst-case searches
and maps are checkpointed. Resuming the same command checks protocol/source hashes
and frozen waveform identities. Changed code/configuration requires a new output
directory, protecting completed results from accidental protocol mixing.
`--max-new-runs N` stops after N new optimizer trials, with an explicit incomplete
status and no validation generation. Setup/calibration is a prerequisite and is
not counted as an optimizer trial. This is not a silent reduction of any cap.

Each frozen waveform receives nominal, exact P05 certification, numerical
convergence, truncation and bounded worst-case checks using the production
solver. Strict checks include nominal, lowest-fidelity/highest-leakage validation
witnesses and diagnostic extrema. Reference composites retain longer durations
and cannot qualify as fixed-duration winners. The five-level Duffing model,
Lindblad dynamics, unconditional six-state fidelity, amplitude/headroom and
leakage constraints are unchanged. Numerical diagnostic searches never establish
a certified global worst case.

## Outputs and interpretation

- research_benchmark_summary.md: question, methods, budgets, results, ablations,
  limitations and reproduction.
- research_benchmark_results.json: actual effective settings, per-trial metrics,
  qualification, comparisons, selected hyperparameters and limitations.
- cvar_sensitivity.csv: tuning-phase curves, explicitly selection only.
- paired_algorithm_comparison.csv: paired effects and bootstrap intervals.
- optimization_costs.csv: actual trial costs and termination statuses.
- robustness_maps/: machine-readable amplitude-detuning maps for the first
  predeclared optimizer trial, not a validation-picked best seed.
- publication_figures/: eight scientifically labeled figures. CDF pooling is
  display only; inference preserves crossed trial/device structure. Scatter
  plots are not labeled a Pareto frontier. Map colors show infidelity with a
  common zero-based scale; red contours flag leakage-threshold crossings.

No hardware performance, quantum advantage, global optimality or algorithm
superiority is inferred from a smoke run. Full device/optimizer-trial uncertainty
and wall/resource differences must be assessed before a publication claim.

## Executed integration check

The campaign at results/research_benchmark_complete ran in explicit quick mode:
4 selection / 8 validation devices, one optimizer seed, two random training
devices plus stress cases, eight NM calls and one Fourier iteration. Its wall
time was 404.6 seconds. No frozen group passed the full engineering assessment;
paired risk/conventional effects were zero because both retained identical
budget-limited starts. This is workflow/estimator verification, not evidence
that the methods are equivalent. Resume was verified without repeating solves.
The full protocol at results/research_full_plan is prepared but unexecuted.
