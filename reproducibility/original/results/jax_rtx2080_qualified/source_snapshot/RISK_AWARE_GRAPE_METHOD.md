# Risk-aware dissipative Fourier GRAPE

The opt-in qualified multistart and separate experimental lambda-continuation
extensions are specified in [GRAPE_DEVELOPMENT_PROTOCOL.md](GRAPE_DEVELOPMENT_PROTOCOL.md).
The fixed-weight risk objective and legacy initializer remain available. The
development runner stops at candidate selection and has no validation path.

## Scope and previous certification failure

The previous `results/reliability_smoke/config.json` explicitly requested four
selection and eight validation devices with a small-smoke run label. Those were
actual executed counts, not broken default propagation. At eight devices the
exact distribution-free 95% P05 confidence procedure cannot produce a nontrivial
bound. Returning zero and FAIL was correct. Zero was not observed fidelity.

Normal defaults remain selection=200, validation=2000, NM calls=500 per restart,
seeds=[7,17,27,37,47]. CLI overrides are respected; explicit quick mode overrides
budgets with its documented reduced profile. Effective settings are printed
before optimization, saved in config.json and summary.json, and low validation
counts produce an INSUFFICIENT_EVIDENCE warning. No empirical quantile substitutes
for the confidence bound. Sufficient sample count does not guarantee PASS.

## Objective and empirical CVaR

For a fixed training ensemble of N scenarios, define f_i=1-F_i and l_i=L_i.
The objective is mean(f) + lambda*CVaR_alpha(f) + mu*mean(l) + R(controls).
F is unconditional six-state X-gate fidelity. L sums all noncomputational
populations and averages over the same six inputs. CVaR applies to infidelity
alone, not to a mixture of fidelity and leakage penalties.

CVaR_alpha(f) = min_eta eta + sum(max(f_i-eta,0))/((1-alpha)*N).

The exact evaluation sorts losses descending and assigns tail mass
m=(1-alpha)*N. Each sorted loss receives weight
clip(m-j,0,1)/m for j=0,...,N-1. This correctly handles fractional mass,
ensembles smaller than 20 at alpha=.95, ties and alpha approaching one.
A rounded worst-5% sample count is not used. Alpha must be strictly below one.

For optimization replace max(x,0) by tau*log(1+exp(x/tau)), evaluated by
logaddexp. The inner scalar minimizer solves
mean(sigmoid((f_i-eta)/tau)) = 1-alpha, using a centered/scaled bracketed root.
The envelope gradient is

  dCVaR/d f_i = sigmoid((f_i-eta)/tau)/((1-alpha)*N).

The eta derivative vanishes at the root. Its implicit derivative is therefore
not needed in the outer gradient. For alpha=0 the exact mean is used directly.
Default alpha=.95, lambda=1, mu=2, tau=1e-6. Lambda and mu may be zero.

The approximation obeys

  exact CVaR <= smooth CVaR <= exact CVaR + tau*log(2)/(1-alpha).

At default alpha/tau this conservative bound is about 1.3863e-5 in infidelity
units; the objective error bound multiplies this by lambda. The final coefficient
point is re-evaluated and exact/smooth training CVaR and the bound are saved in
history[-1].risk_diagnostics. Held-out/selection metrics always use exact
empirical CVaR, not the smoothed quantity. A smaller tau increases curvature;
full studies should predeclare tau and assess a smoothing/substep sensitivity
sweep without tuning against held-out results.

## Adjoint and physical model

The five-level Duffing Hamiltonian, unit conversions, ladder couplings and
Lindblad relaxation/dephasing are unchanged. Existing Fourier controls and
joint I/Q manufacturing-headroom constraints are reused. Endpoints and requested
duration are fixed. Production qualification retains nominal F>=.999,
leakage<=.0001 and all prior engineering constraints.

Each scenario uses the existing dissipative Liouvillian midpoint-substep
propagator with analytic Frechet derivatives and a reverse adjoint sweep.
Two passes separate infidelity and leakage gradients: evaluate weights zero and
one, then subtract to obtain the leakage gradient. The risk gradient combines
mean fidelity gradients, CVaR-weighted fidelity gradients and mu times mean
leakage gradients. The existing finite-difference smoothness regularization
and Fourier chain rule remain unchanged. This is not a closed-system surrogate.
Independent production QuTiP propagation qualifies every proposal before
selection and validates the frozen pulse afterward. Coarse substeps are allowed
only as an explicit optimization approximation; they do not replace production
metrics or strict-solver/truncation checks.

## Training, selection and validation separation

When risk-aware GRAPE is enabled, a fixed iid uniform training ensemble is
combined with deterministic stress cases. Nelder–Mead, reference qualification,
conventional GRAPE and risk-aware GRAPE receive the same combined training
scenarios. Both GRAPE methods start from the same completed training-selected
seed waveform and use matched optimizer seeds, Fourier representation and
inherited conventional settings unless explicitly overridden. Each run keeps
its ensemble fixed. Conventional GRAPE remains directly available.

Default training_samples=32, training_seed=2050, optimizer_seeds=[2030,2031].
All seven uncertainties are sampled: multiplicative amplitude, additive detuning
and alpha, and four independent T1/Tphi scales. Their probability laws and ranges
are unchanged. Training seed collisions with selection/validation/optimizer or
diagnostic streams are rejected. Optimizer seeds are intentionally common
between the two GRAPE methods, but distinct from uncertainty data streams.
Training data, seed, resolved configuration and SHA256 provenance are exported.
With risk mode disabled, legacy conventional training remains available.

All completed eligible proposals, including each optimizer seed's candidate,
are compared on common selection devices. Default selection minimizes exact
CVaR95 infidelity, emphasizing average loss in the worst tail while retaining
hard nominal, sampled-leakage, duration, endpoint and amplitude constraints.
Mean fidelity, empirical P05, mean infidelity and sampled minimum fidelity remain
configurable alternatives. Statistics also include maximum observed leakage and
nominal fidelity. Selection-set confidence intervals are descriptive only.

Selection freezes and hashes one waveform before independent validation.
Validation is performed once with the predeclared seed. It cannot tune controls,
hyperparameters or select another pulse. The exact one-sided 95% binomial P05
lower bound remains the certification rule; CVaR alpha=.95 is a different
quantity from the certification confidence=.95. Nominal, hardware, leakage,
certification, solver, truncation, optimizer completion and diagnostic outcomes
remain separate. Optimizer budget exhaustion does not itself imply poor quality.

## Comparisons and figures

research_comparisons.json reports paired held-out mean, empirical P05, exact
CVaR95 and mean-leakage differences against calibrated DRAG. Mean confidence
intervals use a Student-t approximation with a Bonferroni adjustment for the
reported finite comparison family. They remain exploratory: tail differences
have no confidence interval and no confirmatory superiority claim is made.
The same data are paired device by device. A repeated-optimizer-seed validation
study, prespecified hypotheses and stronger inferential assumptions/design are
still needed before claiming an algorithmic advantage.

The seven research figures show workflow/data separation, each method's own
objective convergence, held-out CDFs, empirical lower tails/CVaR, fidelity/leakage
device scatter, amplitude/detuning slices, and total optimization time versus
representative-proposal CVaR. Different objectives are explicitly labeled as
not directly comparable. Scatter is not labeled a Pareto frontier. Representative
method proposals are selected using training loss before validation; the final
selection winner can be a different seed and is separately exported.

## Cost and limitations

For d physical levels, M interpolation substeps and N scenarios, dense
Liouvillian exponentiation scales approximately as O(N*M*d^6) per objective/
gradient call (M includes time intervals). Risk separation doubles the scenario
passes. Static Liouvillians and control generators are cached within each run,
six input states propagate together, and combined value/gradient evaluations
are cached by the existing optimizer wrapper. Objective calls, gradient calls,
surrogate scenario passes and wall time are recorded separately from production
channel solves. They are not equal computational budgets across methods.
No parallel nondeterministic execution or accuracy reduction is introduced.

Full-mode worst-case diagnostic budgets increase from 80 to 200 calls per
fidelity/leakage search, plus axis/boundary probes. They remain bounded local
searches with convergence/budget status, not certified global bounds. Pulse
hashes remain fixed. The Duffing/RWA and extrapolated higher-level-noise model
remain approximations, with no hardware or spectator-qubit validation.

## Reproduction

Full prescribed experiment (potentially substantial CPU time):

```bash
.venv/bin/python run_all.py --non-interactive --grape --risk-aware-grape \
  --selection-samples 200 --validation-samples 2000 \
  --nm-evaluations 500 --restart-seeds 7 17 27 37 47 \
  --selection-metric cvar95_infidelity --output results/risk_full
```

Explicit execution check:

```bash
.venv/bin/python run_all.py --non-interactive --quick --grape --risk-aware-grape \
  --output results/risk_quick
.venv/bin/python -m unittest discover
```

Use --no-risk-aware-grape for conventional-only comparisons and both --no-grape
and --no-risk-aware-grape to disable Fourier optimization. New settings inherit
conventional GRAPE numerical parameters; all risk-specific fields are under
risk_aware_grape. A smoke result cannot establish full-budget performance or
journal suitability. A full experiment must be labeled unexecuted until its
actual artifacts exist; no favorable-seed validation retries are permitted.

## Executed verification

77 tests passed, including CVaR/fractional-tail correctness, smoothing bounds,
finite-difference gradients, lambda/mu zero limits, data separation and existing
physics/workflow regressions. The integration experiment is documented in
[the generated report](results/risk_aware_verified/report.md) and
[verification metadata](results/risk_aware_verified/verification.json).

It used 8 selection and 64 independent validation devices, two random training
devices plus ten deterministic stress cases, one optimizer seed and one GRAPE
iteration. Wall time was 268.8 seconds. Conventional/risk optimization took
2.42/6.72 seconds, with 5/7 combined objective-gradient evaluations and 60/168
surrogate scenario passes respectively.

The selected risk pulse had nominal fidelity 0.999623820 and an exact P05 lower
bound 0.995844923, passing configured statistical/engineering checks. Held-out
empirical P05 was 0.996802990 versus conventional 0.996771456. Risk CVaR95
infidelity was worse (0.003668594 versus 0.003554461), as was its held-out mean.
No superiority claim follows. Full optimization budgets, 2000-device validation
and repeated-optimizer-seed performance distributions remain unexecuted. Native
Tk layout was not exercised; shared UI/CLI parsing was tested headlessly.
