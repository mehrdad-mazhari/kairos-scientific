# Full matched-budget development execution

Run from the project root with the existing virtual environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kairos-mpl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python full_matched_study.py --backend gpu
```

Default output: `results/full_matched_budget`. The same command resumes after
interruption; an exclusive process lock prevents duplicate campaigns. `--status`
reads the current status without launching any optimization. `--backend auto`
qualifies available FP64 CUDA and otherwise chooses the existing CPU backend;
`--backend cpu` uses CPU exclusively. Explicit `gpu` errors on unavailable or
unqualified hardware. A campaign's actual backend is frozen, so resuming on a
different backend requires a new versioned output directory. GPU models are
discovered, not hardcoded. FP32 is forbidden for this study.

The full profile is `full_matched_config.json`, derived from the stronger
reference's prepared full config. It declares 27 trials: conventional, fixed-risk
and continuation GRAPE; seeds 7500/7501/7502; equal-objective, equal-pass and
equal-wall regimes. All use 100 random training cases plus the existing ten
stress cases, 200 common selection cases, five harmonics, four substeps, five
aggregate restart allocations, and 150 aggregate iterations. Applicable caps are
1000 objective calls, 220500 dissipative passes, or 1800 optimizer seconds.
These are maximum allocations, not guarantees of using equal actual resources.
No quick override is active. The original Hamiltonian, noise, constraints,
manufacturing law and nominal thresholds are unchanged.

The stronger nominal DRAG checkpoint is the common source; it is reproduced
with strict CPU QuTiP and checked for headroom/endpoints. Its nominal fidelity is
approximately 0.9996591328 and leakage 4.26981471e-5. Calibration used 414 unique
channels and reported budget exhaustion. It is neither globally optimized nor
fully converged. No weak-pilot optimizer result is imported. The predeclared
development arms are the three GRAPE methods and DRAG. Nelder–Mead and other
reference methods remain available in the existing ordinary/research runners;
this profile declares no additional NM trial allocations.

Lambda=1, mu=2 and continuation [0,.1,.5,1] are fixed. No tuning or final
validation is performed. The reserved seed 7402 is recorded but never sampled.
Each returned waveform is frozen before CPU evaluation on the common selection
draw. No selection result enters an objective or candidate-retention rule.
Results are paired by seed and regime. Reusing 200 devices across three seeds
does not produce 600 independent device samples. Plots show observed seed
variation without invented inferential confidence intervals.

GPU qualification/warmup, historical nominal calibration, initialization and
gradient diagnostics are shared setup. Optimizer wall time includes Fourier
fitting, nominal qualification, objective/gradient/constraint evaluation,
restarts, continuation stages and progress checkpoint writes. Wall caps are
checked between indivisible evaluations; overshoot is reported. Post-trial CPU
selection and pointwise numerical verification are separately recorded analysis
costs outside the optimizer cap. Synchronized GPU-call wall time includes
transfers, dispatch and any compilation, excludes CPU qualification, and is not
CUDA-kernel-only time. Equal-call counts and equal scenario passes are not
claimed to be FLOP-equivalent.

Completed optimization is saved atomically before selection. Completed trials
are never rerun on a valid resume. Inputs, source and configuration signatures
are checked before resuming. An unfinished SLSQP trial restarts from the exact
source and seed; progress and known spent work are retained in interrupted
attempt archives. In-flight work at interruption can be unrecorded and must not
be reported as zero. Changing frozen code/config requires a new experiment
directory. The previous short pilot is separate evidence.

Each completed trial receives CPU production reproduction, four/eight-substep
checks at nominal and two existing joint stress cases, CPU/GPU gradient checks,
state trace/Hermiticity/positivity checks and amplitude/endpoint checks. The
tolerance stays 2e-6. These are pointwise checks, not uniform guarantees.
Convergence requires every original KKT/objective/parameter/constraint criterion.
Raw solver success or objective reduction is insufficient. Negative findings,
missing trials and incomplete convergence are explicit. NO-GO remains until all
required execution and numerical/convergence issues are resolved; it does not
mean that risk methods must win. A confirmatory command is not released while
the study remains incomplete or unresolved.

`study_report.md`, `study_summary.json`, the requested CSV/JSON files, source
snapshot, per-trial optimization checkpoints, progress files and figures are
updated during execution. No final independent validation dataset is generated.
