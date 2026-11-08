# GRAPE development protocol

This is a pre-validation development experiment. No independent validation draw
is generated, opened or passed to optimization. A reserved validation seed is
recorded only to reject seed collisions. No certification verdict is computed.
Engineering thresholds, five-level Duffing dynamics, Lindblad channels,
six-state unconditional X fidelity, duration, manufacturing ranges and physical
amplitude constraints retain their existing definitions.

## Methods and initialization

The compared methods are conventional mean-infidelity GRAPE with leakage and
smoothness, fixed-weight risk GRAPE (lambda=1, mu=2), experimental continuation
(lambda=[0,.1,.5,1], mu=2), and a common previously nominally calibrated DRAG.
The reference calibration checkpoint and source waveform hash are exported;
it is not recalibrated against training, selection or validation data.
The supplied development reference used only five calibration calls and was
not converged. Its qualification suffices to test initialization and paired
optimizer behavior; it cannot support superiority over fully calibrated DRAG.
Before a publication comparison, use a predeclared sufficiently optimized
nominal reference checkpoint and rerun all methods from that common source.

Qualified Fourier multistart fits five odd I/even Q sine harmonics subject to
sampled joint amplitude disks including manufacturing headroom. Linear sample
interpolation keeps the disk constraint between samples by convexity. Fourier
endpoints are exactly zero. Nominal production fidelity and leakage qualify the
source, unperturbed fit and every retained perturbed start. Small random unit
directions use norm .002 in dimensionless coefficient space; step size halves
until amplitude/endpoint/nominal constraints hold, up to twelve attempts. Rejected
attempts and costs are recorded. The unperturbed fit is always start zero;
unsuccessful later starts are omitted rather than relabeled as qualified.

The initialization study separately evaluates legacy five-harmonic .05 iid
perturbations, two-harmonic unperturbed radial fits, five-harmonic unperturbed
radial fits and the qualified bank. It does not silently replace legacy behavior.

## Convergence and continuation

All accepted records mean retained nominally qualified, amplitude-feasible
objective improvements, including line-search trials; they are not asserted to
equal SLSQP major steps. Production nominal metrics are recorded at each retained
proposal; mean fidelity, CVaR and maximum leakage on training devices use the
analytic dissipative surrogate at the declared interpolation substeps. These
are explicitly distinguished in plots. Candidate selection uses production
channels on the shared 200-device draw.

Convergence requires optimizer success at the returned point, joint-constraint
satisfaction, projected-gradient/KKT residual <=1e-4, relative objective change
<=1e-7 and normalized parameter change <=1e-5. For active amplitude constraints,
nonnegative KKT multipliers are fitted to the gradient. Raw gradient norm is also
reported. Budget exhaustion is distinct from convergence. This local numerical
check does not prove a global optimum.

Continuation allocates remaining resources across the remaining stages, carrying
unused work forward. Five restart allocations are shared across the four stages,
not five additional starts per stage. Later stages receive exact retained
Fourier coefficients. A failed stage records the exception and preserves its
input waveform. The final candidate minimizes the final scheduled training
objective over archived nominally qualified improvements from all stages.
Different stage objectives are not compared by their raw values. CVaR need not
improve as lambda increases; no theoretical or independent superiority follows.
Stage completion means that a qualified evaluated proposal was returned, even
when its optimizer stopped on a budget. It does not mean optimizer convergence.
The deterministic stage perturbation seed is optimizer_seed + 10000*stage_index.

## Resources and pairing

Full maxima: 150 total major iterations; five restarts; five harmonics; four
interpolation substeps; 100 iid random training scenarios plus ten deterministic
stress cases; 200 common selection scenarios; optimizer seeds [7500,7501,7502].
Training, selection and reserved validation seeds are 7400,7401,7402.

Three regimes use equal caps for each optimized method:

1. 1000 combined objective/gradient calls across all starts/stages.
2. 220500 total dissipative passes, including production nominal channels.
3. 1800 seconds, including construction, fitting, qualification and monitoring.

An objective/gradient call uses N surrogate passes for conventional GRAPE and
2N for the separate infidelity/leakage adjoints of the risk methods. Production
qualification uses one six-input Lindblad channel per unique waveform. These
passes are not FLOP-equivalent; wall time is a separate required comparison.
Actual calls, surrogate passes, production channels, major iterations and time
are reported. Indivisible solves can overshoot a wall deadline; checks occur
between calls. The common 150-iteration cap may stop an optimizer before its
regime-specific cap. No assumption of equal actual cost is made.

Historical reference calibration is a shared expense. The initialization study,
gradient checks, final production verification, common selection evaluations and
figure generation are development overhead, separate from the optimization
ledger and labeled as such. Selection caching by waveform hash avoids duplicate
production solves but never supplies selection metrics to optimization.

The explicit short diagnostic uses two seeds [7500,7501], 24 total iterations,
24 objective calls, two substeps, and the equal-objective regime. All sample
counts, physics and thresholds are unchanged. This does not execute the full
intermediate budgets or the equal-pass/equal-wall experiments. No silent resource
reduction is permitted.

## Outputs, resuming and interpretation

Every completed method/seed/regime is atomically checkpointed. Selection channels
are individually checkpointed; all repeated candidates use the same draw list.
Source/config/data signatures prevent reuse under changed protocols. Completed
trials resume without reoptimization. Per-stage histories are included in a
completed continuation checkpoint. Mid-trial interruption currently requires
restarting that unfinished trial from its original seed; discarded work must be
reported as additional overhead, and a rerun is not an exact interrupted-state
optimizer replay. This is a limitation of checkpoint granularity, not free work.

The CSV comparisons report conventional CVaR minus risk/continuation CVaR on
common selection devices; positive favors the risk method. Mean, P05, nominal
fidelity, leakage and runtime differences are also reported. No uncertainty
intervals or statistical-superiority verdicts are inferred from this development
selection set. Manufacturing worst-case leakage failures remain ineligible even
if CVaR improves. The final research protocol must be frozen before independent
2000-device validation, including hyperparameters, budgets and decision rules.
