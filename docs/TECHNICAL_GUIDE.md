# Kairos technical guide

Kairos compares simulated transmon X gates using textbook references, nominal
calibration, bounded multistart Nelder–Mead, optional Fourier GRAPE, and
independent validation. The default optimizer is Nelder–Mead in both the desktop
interface and CLI. No population optimizer or external evolutionary dependency
is required.

## Install and run

Python 3.12 is the verified environment. Install `requirements.txt` into a local
virtual environment, then run:

```bash
.venv/bin/python optimization_ui.py
.venv/bin/python run_all.py --non-interactive
.venv/bin/python run_all.py --non-interactive --quick
.venv/bin/python run_all.py --non-interactive --no-nelder-mead --grape
.venv/bin/python run_all.py --non-interactive --nelder-mead --no-grape
```

The desktop requires Tkinter; numerical tests and command-line execution do not.
The launcher runs all discovered tests first. Quick mode reduces search,
calibration, GRAPE and Monte Carlo budgets; it is an execution check, not
publication evidence. Use a new or empty output directory for every run.

```bash
.venv/bin/python run_all.py --non-interactive --output results/my_run
.venv/bin/python run_all.py --show-results results/my_run
.venv/bin/python -m unittest discover
```

The standalone optimizer defaults to `results/nelder_mead`. `run_all.py` uses a
timestamped directory unless `--output` is supplied.

## Fixed physical inputs

Times are ns and input frequencies are MHz. Frequencies in Hamiltonians and
waveforms are angular frequencies in rad/ns, using `2*pi*1e-3` for conversion.
Device inputs and requested duration are never optimized. Physical CLI options:

```text
--detuning-mhz --alpha-mhz --t1-1-ns --t1-2-ns
--tphi-1-ns --tphi-2-ns --tg-ns
```

CLI numeric flags use MHz/ns. Interactive prompts and desktop fields also parse
explicit units such as `300000 Hz`, `30 us`, and `0.3 MHz`. Duration must be
10–60 ns on the 0.2 ns sample grid. Anharmonicity must be negative; lifetimes and
hardware amplitude must be positive.

## Primary optimization

The four controls, in `bounds` order, are:

1. Gaussian width sigma divided by gate duration.
2. DRAG beta.
3. Dimensionless amplitude scale.
4. Constant I/Q phase ramp in MHz.

Each restart applies SciPy's bounded Nelder–Mead to normalized [0,1] coordinates,
so MHz and dimensionless controls have comparable step sizes. The first restart
starts from nominally calibrated DRAG when available. Subsequent restarts use
seeded perturbations of that point; every seed also defines simplex step sizes.
All restarts share an exact-candidate cache.

```json
"nelder_mead": {
  "enabled": true,
  "max_evaluations": 500,
  "xatol": 0.00001,
  "fatol": 0.0000000001
},
"search_seeds": [7, 17, 27, 37, 47]
```

`max_evaluations` is a cap per restart. `xatol` applies in normalized coordinates;
`fatol` applies to the scalar loss. Budget exhaustion is recorded separately from
convergence. Convergence is local and does not prove a global optimum.

Hardware-feasible and nominally qualified candidates are scored by

```text
loss = w_mean * mean_training_infidelity
     + w_worst * worst_training_infidelity
     + w_leakage * worst_training_leakage
```

The weights are explicit in `selection_weights`. Training uses nominal,
single-axis boundary, coherence-degraded and joint stress cases, with uncertainty
ranges taken from `manufacturing_ranges`. Nominal quality and amplitude failures
receive penalties and remain ineligible for final acceptance. No test samples
are consumed by search.

The archive retains all evaluated candidates, not just the optimizer's terminal
point. Top training finalists, the primary method's best candidate, all reference
families and the GRAPE proposal enter common Monte Carlo selection. An optional
random benchmark is configured through `random_search.enabled`; it is disabled
by default and uses the same loss, controls and constraints.

Search costs record objective calls, unique candidates, channel solves, elapsed
time, convergence status and termination messages. Calls and channel solves can
differ because of caching and early constraint rejection. Calibration costs are
reported separately. GRAPE's matrix-exponential surrogate cost is not included
in production-channel counts.

## Pulse construction and calibration

The primitive envelope is

```text
g = exp(-(t-T/2)^2/(2*sigma^2))*sin(pi*t/T)^2
I+iQ = A*(g-i*beta*g_dot/alpha)*exp(i*2*pi*f_ramp*0.001*(t-T/2))
```

A normalizes the sampled envelope to amplitude*pi area. Both endpoint samples
are zero; interpolation is linear. The amplitude constraint applies to the
joint I/Q magnitude, including manufacturing headroom.

Each textbook reference has a separate nominally calibrated comparator.
Calibration holds shape and duration fixed while fitting gain, phase ramp and
applicable beta using nominal infidelity only. It retains its best feasible
point even when its evaluation cap is reached. Raw reference waveforms are
scaled down if required by hardware headroom; that projection is explicit and
can reduce their rotation area below pi.

- Gaussian has no derivative quadrature before phase rotation.
- DRAG uses reference beta=0.5 for the sqrt(2) transition coupling.
- Hermite DRAG applies the second-order Hermite envelope correction. Its legacy
  JSON identifier is `higher_order_drag`; it is not the higher-order adiabatic
  DRAG expansion.
- Wah-Wah applies a cosine sideband modulation and its analytic derivative.
  The one-transmon model cannot assess spectator-qubit crosstalk suppression.
- BB1 uses four DRAG-shaped rotation segments with phases
  `(pi,0),(pi,phi),(2*pi,3*phi),(pi,phi)`, `phi=acos(-1/4)`.
- CORPSE uses `(7*pi/3,0),(5*pi/3,pi),(pi/3,0)` at a common plateau rate.
  One-sample edge ramps are included in the exact rotation areas. The waveform
  approaches square-pulse two-level cancellation as edge duration vanishes;
  it does not guarantee transmon leakage suppression.

Composites have explicitly longer actual durations with decoherence throughout.
They cannot win a fixed-duration task or seed its GRAPE refinement.

Saved specifications reconstruct exact sampled waveforms:

```python
import json
from pulse_methods import waveform_from_spec
spec = json.load(open('results/my_run/selected_pulse_parameters.json'))
time, pulse_i, pulse_q = waveform_from_spec(spec)
```

Calibrated specifications contain the underlying reference recipe and fitted
controls. Fourier specifications contain every I/Q coefficient and amplitude
scale. Gaussian A/B coefficients are reference quantities when another family
is selected; `pulse.csv` is authoritative.

## Physical model

The default production model uses five Duffing levels:

```text
H = Delta*n + alpha*n*(n-1)/2 + I*(a+a†)/2 - i*Q*(a-a†)/2
a = sum_n sqrt(n)*|n-1><n|
```

Detuning has a positive sign. Scenario amplitude error multiplies the fixed
waveform; detuning and anharmonicity errors add to nominal device inputs.
Reference pulses are constructed once from nominal parameters, never retuned
for individual perturbed devices.

Relaxation operators use sqrt(1/T1_j) transitions, while pure-dephasing operators
use sqrt(2/Tphi_j) excited-level projectors. These follow the challenge's detailed
page-5 convention. For n>2, relaxation rates scale as n/2 relative to level 2;
dephasing projector rates scale as (n/2)^2. Higher-level noise is an explicit
extrapolation from the supplied n=1,2 lifetimes, not a measured spectrum.

Fidelity averages unconditional X-gate overlap over the six Pauli eigenstates.
Leaked output is never renormalized. Leakage sums all modeled populations n>=2.
The six-state mean equals the qubit Haar average for the linear channel. No
unreported free virtual-Z correction is applied.

## GRAPE, selection and validation

Optional GRAPE refines Fourier I/Q controls with analytic Fréchet/adjoint
derivatives and constrained SLSQP. Its dissipative ensemble objective includes
explicit leakage and smoothness terms. Acceptance uses the production QuTiP
solver, not only the optimization approximation. Its own training loss is
computed under the same production constraints as other candidates.

All completed candidates see common selection draws. Selection maximizes the
configured fidelity statistic subject to duration, nominal quality, sampled
leakage and hardware constraints. If none passes, the best fixed-duration
diagnostic candidate is saved with quality FAIL. The waveform is frozen before
independent held-out draws are evaluated; validation cannot select another pulse.

Monte Carlo uses independent uniform quasi-static amplitude, detuning and
anharmonicity errors, plus independently scaled coherence times. The challenge
specifies ranges, not probability laws; these distributions are modeling choices.
The public 45-case grid retains the challenge's offsets around supplied nominal
inputs and nominal coherence. All methods share the same devices.

The P05 lower bound is a distribution-free, one-sided 95% binomial order-statistic
bound. Small samples can support only zero. Mean confidence intervals use a
Student-t approximation; threshold-yield intervals use Wilson's method. Sampled
extremes do not certify the entire uncertainty domain. Paired mean-difference
intervals against calibrated DRAG are exploratory, without multiplicity correction.

Strict-solver checks compare nominal and worst-fidelity held-out results on the
exported waveform. Truncation checks evaluate 3–6 levels plus production and
production+1 for nominal, worst-fidelity held-out and coherent nominal evolution.
Quality cannot pass with a three-level production model or failed convergence.
This is not a uniform convergence proof across all devices.

## Outputs and code map

- `nelder_mead_history.json`: every primary-search evaluation and best loss.
- `search_seed_<seed>.json`: restart-specific trace.
- `nelder_mead_archive.json`, `optimization_archive.json`: evaluated candidates.
- `search_costs.json`, `reference_calibration.json`: measured costs and statuses.
- `pareto.json`, `objective_diagnostics.json`: archive diagnostics only; the
  optimizer minimizes one weighted scalar loss, not a multi-objective front.
- `mc_selection_ranking.json`: selected statistics and rejection reasons.
- `*_public.csv`, `*_monte_carlo.csv`: per-device evaluations for each method.
- `pulse.csv`, `selected_pulse_parameters.json`: exact final pulse and recipe.
- `summary.json`, `truncation_check.json`, `paired_comparisons.json`: validation,
  numerical convergence, source hashes, versions and paired comparisons.
- `report.md`, `stage_comparisons.md`: readable results and figures.

`robust_optimization.py` owns the workflow; `nelder_mead.py` owns bounded simplex
search; `reference_calibration.py` owns nominal calibration; `challenge_model.py`
owns physics; `grape.py` owns Fourier refinement. `optimization_inputs.py` shares
configuration logic across CLI, tests and the optional Tk desktop interface.
Original exploratory scripts remain separate from the production comparison.

## References and interpretation

DRAG: [Motzoi et al.](https://arxiv.org/abs/0901.0534) and
[Gambetta et al.'s analytic-control treatment](https://arxiv.org/abs/1011.1949).
[Poudel & Vavilov](https://arxiv.org/abs/1008.2554) is a secondary discussion in
an Ohmic environment; this code uses phenomenological Lindblad noise.
[Composite rotations](https://arxiv.org/abs/quant-ph/0208092),
[Wah-Wah](https://arxiv.org/abs/1405.0450),
[SciPy Nelder–Mead](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html).

Passing implementation tests does not establish scientific novelty, hardware
performance, a global optimum or journal suitability. Robustness/duration/noise
trade-offs require full-budget, repeated-seed studies with independently assessed
outcomes. See [the audit corrections](SCIENTIFIC_AUDIT.md).


## Statistical reliability settings

The pipeline order is fixed: physical configuration, reference generation,
nominal calibration, multistart Nelder–Mead, optional GRAPE, common-device
selection, freeze/export, independent validation, statistical certification,
frozen-pulse robustness diagnostics, numerical/truncation checks, engineering
assessment. The optional random benchmark remains disabled by default and,
when enabled, runs before GRAPE/selection. Diagnostics and held-out results
never retune or reselect a pulse.

| Setting | Full-run default | Meaning |
|---|---|---|
| `mc_selection_samples` | 200 | Common iid device draws for candidate ranking |
| `monte_carlo_samples` | 2000 | Independent held-out validation draws |
| `nelder_mead.max_evaluations` | 500 | Objective-call cap per restart |
| `search_seeds` | [7,17,27,37,47] | Reproducible optimization restart seeds |
| `mc_selection_criterion` | p05_fidelity | Empirical lower-tail selection metric; alternatives are mean_fidelity and worst_sample_fidelity |
| `training` | {} | Optional overrides for deterministic stress cases; does not change the test distribution |
| `training.manufacturing_ranges` | Inherit manufacturing_ranges | Training-only amplitude, detuning and alpha ranges |
| `training.coherence_min` | Inherit coherence_min | Training-only lower coherence scale |
| `worst_case.enabled` | true | Run post-validation frozen-pulse diagnostic searches |
| `worst_case.max_evaluations` | 80 | Objective-call cap for each of two bounded local searches; axis/boundary probes are additional |
| `worst_case.seeds` | [2041,2042] | Separate fidelity/leakage diagnostic simplex seeds |
| `worst_case.axis_points` | 5 | Points per single-axis/coherence diagnostic slice |
| `worst_case.xatol` | 0.001 | Diagnostic tolerance in normalized uncertainty coordinates |
| `worst_case.fatol` | 0.00000001 | Diagnostic objective tolerance |
| `worst_case.fidelity_min` | null | Optional additional observed-minimum fidelity constraint; null does not turn a quantile target into a pointwise requirement |

Existing explicit budgets, restart seeds, physical inputs and distributions are
preserved when older JSON configurations omit the new sections. Resolved settings
are exported with the run. Training is deterministic. Selection, validation,
optimization, GRAPE and enabled diagnostic random seeds must be separate;
previous configurations with collisions now fail validation instead of silently
sharing random streams. Uniform quasi-static uncertainty distributions are
unchanged: amplitude multiplies controls, detuning/alpha errors add to device
inputs, and each coherence lifetime receives an independent uniform scale in
[coherence_min,1].

P05 selection targets the weak tail rather than allowing a high mean to hide
poor devices. A sample minimum is more sensitive to sample count and outliers.
Every candidate receives the exact same selection draw list. Selection-set
intervals remain descriptive; the dataset was used to choose the pulse.

The desktop exposes selection/validation budgets, calls per restart, restart
seeds, selection metric, diagnostic budget and its enable switch. Loaded JSON
preserves advanced training and diagnostic settings. The CLI exposes the same
controls:

```bash
.venv/bin/python run_all.py --non-interactive \
  --selection-samples 200 --validation-samples 2000 \
  --nm-evaluations 500 --restart-seeds 7 17 27 37 47 \
  --selection-metric p05_fidelity --worst-case --worst-case-evaluations 80
```

Use `--no-worst-case` to disable optional searches. Quick mode uses the same
worker profile for CLI and desktop: 32 selection draws, 128 validation draws,
one restart with 60 calls, 40 calibration calls, reduced GRAPE and 16 calls per
diagnostic search with 3 points per axis. Quick mode overrides displayed full-run
budgets but never physical inputs or quality thresholds.

The one-sided 95% lower confidence bound for the true P05 is a binomial order
statistic, not a normal approximation. At least 59 iid samples are needed for
a nontrivial bound at these fixed confidence/quantile settings. Insufficient
samples return the physical lower bound zero and certification FAIL. This zero
is not an observed gate fidelity. Reports separately show empirical P05,
minimum observed fidelity, mean/CI, maximum observed leakage, validation count,
confidence level and the fraction meeting configured nominal-fidelity/leakage
sample thresholds. Unlike the old PDF-threshold yield, this fraction now uses
the configured engineering thresholds; it is descriptive, not an extra yield
acceptance requirement.

After independent certification, bounded searches minimize fidelity and maximize
leakage over the configured seven-dimensional uncertainty domain. Nominal,
single-axis, all-coherence and joint boundary probes precede the searches.
Results are sampled/local observations, not mathematically certified global
worst-case bounds. Failure to find a violation does not certify the domain.
Observed hard-leakage violations (or an explicitly configured pointwise fidelity
violation) fail engineering assessment; an observed fidelity below the P05 target
alone does not prove the true P05 fails.

The frozen waveform is hashed before validation draws are generated. Hashes
are checked after all diagnostics. Selection/validation draws and deterministic
training scenarios are saved with hashes for reproduction. Strict production
solves now check nominal, minimum-fidelity and maximum-leakage held-out devices,
plus diagnostic extrema. Truncation checks also include diagnostic witnesses.
Finite metric discrepancies or post-validation solver errors yield explicit
failed summaries rather than an optimistic quality result.

Additional outputs:

- `statistical_certification.json` and `.png`: exact tail bound, support status,
  count, confidence, certification reasons and observed-versus-bound plots.
- `worst_case_robustness.json` and `.png`: axis sensitivities, joint witnesses,
  local-search termination status and separately measured simulation/time costs.
- `mc_validation_scenarios.json`, `frozen_waveform.json`: validation draws and
  pre-validation waveform hashes.
- `summary.json` adds engineering checks/failure codes, optimizer convergence
  status, dataset provenance, measured costs and an explicitly exploratory
  GRAPE-versus-calibrated-DRAG paired mean comparison.

Failure codes distinguish no eligible candidate, nominal quality, hardware or
duration violations, statistical certification, public-grid leakage, observed
robustness violations, diagnostic solver failure, numerical convergence and
Hilbert-space truncation. A nominally qualified pulse can still fail statistical
certification. Optimizer budget exhaustion is recorded independently and alone
does not force engineering FAIL if the best frozen candidate meets every check.
A local convergence flag never proves global optimality.

Full-run defaults are not evidence that a full run was executed. A smoke run
only checks integration. Paired comparisons remain exploratory and do not claim
GRAPE superiority; adversarial searches remain local; the Duffing/RWA and
higher-level-noise extrapolation assumptions remain unchanged. Native Tk layout
requires a desktop environment; headless tests exercise the same configuration
logic without GUI stubs.

## Risk-aware GRAPE extension

See [RISK_AWARE_GRAPE_METHOD.md](RISK_AWARE_GRAPE_METHOD.md) for the objective,
analytic gradient, smoothing error bound, cost, statistical interpretation and
reproduction protocol. `risk_aware_grape` defaults in the shipped configuration:

| Field | Default | Meaning |
|---|---|---|
| enabled | true | Keep conventional GRAPE and add a separate risk-aware method |
| alpha | .95 | Infidelity CVaR tail level; distinct from certification confidence |
| risk_weight | 1 | Lambda multiplying CVaR |
| leakage_weight | 2 | Mu multiplying mean leakage |
| smoothing | 1e-6 | Explicit softplus temperature |
| training_samples | 32 | Uniform training devices, plus fixed stress cases |
| training_seed | 2050 | Independent fixed training draws |
| optimizer_seeds | [2030,2031] | Matched conventional/risk optimizer seed runs |

Risk mode uses the same fixed combined ensemble for NM and both GRAPE methods.
Its numerical parameters inherit conventional GRAPE unless explicitly overridden.
Legacy configurations omitting risk_aware_grape retain conventional-only behavior.
CLI `--risk-aware-grape` / `--no-risk-aware-grape` and the desktop checkbox share
the same enabled field. `--no-grape` disables conventional GRAPE only.

The new default selection metric is `cvar95_infidelity` (minimize). The alternative
`mean_infidelity` also minimizes; existing fidelity metrics maximize. Effective
budgets are printed before optimization and saved unchanged, with warnings for
fewer than 59 validation devices. Older 4/8 artifacts were explicit smoke inputs,
not evidence of broken 200/2000 defaults. Full-mode diagnostic calls now default
to 200 per search in the shipped JSON. The original exact binomial certification
procedure and fidelity/leakage targets are unchanged.

New artifacts: grape_benchmark_runs.json (all seed histories, gradient and
objective counts, surrogate passes and time), research_comparisons.json (paired
exploratory differences with multiplicity-adjusted approximate mean intervals),
and research_figure_1.png through research_figure_7.png. No inferential interval
is asserted for empirical P05/CVaR differences; no superiority claim is automatic.

## Resumable publication benchmark

The separate `research_benchmark.py` entry point uses `research_config.json`.
Full research mode enforces 200 selection / 2000 validation devices, 100 random
training devices, ten paired optimizer seeds, five 500-call NM restarts and
500-call worst-case searches. `--quick` explicitly selects the reduced profile.
The ordinary desktop optimizer is preserved; research campaigns use the new CLI.
See [RESEARCH_BENCHMARK_PROTOCOL.md](RESEARCH_BENCHMARK_PROTOCOL.md) for every
setting, hypotheses, lambda/mu grids, ablations, bootstrap assumptions, budget
semantics, freeze barrier and checkpoint/resume commands.

## Optimizer execution diagnostics

`python optimizer_diagnostics.py --output results/optimizer_diagnostic`
performs a training-only audit; it does not generate or read validation devices.
It replays historical three-call runs, checks exact waveform equality, compares
unperturbed/perturbed Fourier fits to qualified DRAG, verifies gradients on 100
distinct training scenarios and runs both optimizers with usable diagnostic caps.
It writes optimizer_diagnostic_report.md, optimizer_traces.csv,
optimizer_run_summary.csv, historical_replay_traces.csv, gradient_validation.json,
initialization_comparison.csv and effective_research_config.json.

GRAPE histories now include execution_audit: actual initial/final Fourier
objectives, coefficients, hashes, gradient norms, evaluation-level traces,
SLSQP callback counts and retained feasible objective improvements. Callback
counts are not asserted to equal distinct accepted steps; historical logs did
not record enough information to reconstruct that count. No-change returns
emit an automatic RuntimeWarning instead of suggesting successful optimization.

The default initializer is unchanged (`initialization_strategy=least_squares_radial`).
Explicit `constrained_least_squares` fits both quadratures subject to sampled
joint amplitude disks, with analytic least-squares and constraint gradients.
Fourier endpoints remain zero. This fit cannot guarantee gate fidelity; the
production solver must qualify it. A perturbation of dimensionless coefficients
is not the same thing as the same fractional perturbation of pulse amplitude.

Research GRAPE options are resolved by `effective_grape_options`: full and quick
iteration budgets come from their explicit profiles, inherited call/wall/pass
caps are cleared, and `scenario_passes` is enforced in both budget regimes.
`effective_research_config.json` exports the resulting optimizer options as well
as the profile settings. Recorded costs count the raw seed evaluation and any
final diagnostic reevaluation, in addition to Fourier search evaluations.

## Qualified starts and experimental continuation

`initialization_strategy=qualified_fourier_multistart` opts into five-harmonic
constrained fitting, production nominal qualification and small reproducible
perturbations that shrink until qualified. The unperturbed qualified fit is
always included. Calibrated DRAG/Wah-Wah references are supported; the current
development experiment uses DRAG. Legacy radial and constrained-fit strategies
remain available. New research histories retain only nominally qualified
amplitude-feasible improvements and record production nominal and surrogate
training metrics separately. Qualification is a candidate-retention rule; it
does not change the differentiable ensemble objective.

`continuation_risk_grape.py` is a separate experimental method. Lambda increases
through [0,.1,.5,1], with exact feasible coefficient transfer and preservation
of the preceding waveform on a failed stage. A single ledger covers all stages,
all restart attempts, nominal qualification and monitoring. No CVaR improvement
theorem is assumed. Local convergence requires solver success, amplitude
feasibility, small KKT residual, objective stability and parameter stability.

The development runner has no validation-generation path:

```bash
.venv/bin/python grape_development.py --diagnostic --output results/grape_development_short
.venv/bin/python grape_development.py --prepare-only --output results/grape_development_plan
# Potentially many CPU-hours; run only after reviewing the development evidence:
.venv/bin/python grape_development.py --output results/grape_development_full
```

The full development config explicitly specifies 150 total iterations, 1000
objective calls, five restart allocations, 100 random training devices plus
stress cases, 200 selection devices, three optimizer seeds and three cost
regimes. The short diagnostic explicitly changes only its listed caps, seed
count, resource regimes and interpolation substeps; it retains the same sample
counts, physics, thresholds and five-start allocation. All effective options
and source/data hashes are saved. Completed trial and per-device selection
checkpoints resume without reoptimization. An interrupted unfinished trial
currently restarts from its original seed; completed trials are preserved.
See GRAPE_DEVELOPMENT_PROTOCOL.md for cost and interpretation limitations.

The diagnostic reference came from five nominal calibration calls. A stronger
nominal-only checkpoint can be prepared separately, without selection/validation:

```bash
.venv/bin/python prepare_grape_reference.py --output results/grape_development_reference
.venv/bin/python grape_development.py \
  --config results/grape_development_reference/full_config.json \
  --prepare-only --output results/grape_development_full_plan
```

This preserves the historical diagnostic reference and prepares a new common
source for the next matched-start experiment. Calibration success/budget status
remains explicit; nominal qualification is not optimizer convergence. The saved
reference checkpoint bytes are part of the new protocol signature. Use
`verify_grape_development.py` for artifact checks; the separate
`finalize_grape_development.py` summarizes only the short equal-call diagnostic,
so full results are never averaged across different resource regimes.

## Optional JAX training backend

See [JAX_ACCELERATION.md](JAX_ACCELERATION.md) for installation, configuration,
qualification and benchmark commands. The optional backend streams five-level
Liouvillians from the existing CPU model through JIT-compiled, vectorized JAX
propagation and reverse differentiation. Conventional, fixed-risk and
continuation GRAPE retain their existing Fourier controls and constraints.
Global host CVaR weights preserve the full-ensemble objective across batches.
The default backend, production candidate qualification and final statistical
certification remain CPU QuTiP. CUDA initialization/execution failures fall back
to the CPU in ordinary runs; a full GPU campaign requires a matching successful
CUDA qualification report and a live numerical preflight.
