# Optional JAX acceleration on Linux

The default remains CPU QuTiP/SciPy. Installing JAX is optional: CPU runs do not
initialize JAX. Only the GRAPE training surrogate is accelerated. Production
nominal qualification, selected-waveform evaluations, Monte Carlo certification
and the statistical certification rule remain on CPU QuTiP. No Hamiltonian,
collapse operator, uncertainty law or quality threshold was changed.

## Installation

In the project virtual environment, on a Linux host with a working NVIDIA driver:

```bash
nvidia-smi
.venv/bin/python -m pip install -r requirements-jax-cuda12.txt
```

The pinned optional extra installs CUDA libraries through pip. System drivers
remain the user's system configuration. JAX documents Linux driver >=525 for
CUDA 12 and >=580 for CUDA 13. CUDA 12 supports SM >=5.2; the RTX 2080 family is
a Turing target. Check the installed wheel/driver against the official
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

CPU-only kernel development uses `requirements-jax-cpu.txt`; it does not install
or prove CUDA support. The tested JAX/jaxlib version is 0.12.0.1. Base project
requirements are unchanged.

## Configuration

Put this optional dictionary under `grape` in a normal configuration, or under
the top level of a research/development profile. Fixed risk GRAPE inherits it;
continuation passes it to every stage.

```json
"acceleration": {
  "backend": "jax_gpu",
  "precision": "float64",
  "batch_size": 32,
  "max_vram_gb": 8.0,
  "memory_fraction": 0.7,
  "fallback_to_cpu": true,
  "require_accuracy_gate": true,
  "qualification_report": "results/jax_gpu/gpu_qualification_float64.json"
}
```

`cpu` is the default. `jax_cpu` explicitly exercises the same JAX kernels on CPU
for development and hardware-matched controls. It is never reported as CUDA.
Precision selects float32/complex64 or float64/complex128 propagation and AD.
Host aggregation, Fourier constraints and SciPy SLSQP remain float64. X64 is
explicitly enabled; matrix products request highest precision rather than
TF32/FP16. Failed float32 accuracy checks are not repaired by weakening limits.

Ordinary initialization/execution/accuracy failures warn and fall back to the
original CPU objective and gradient. Histories record requested and actual
backends, precision, fallback reason, batch size, memory statistics and useful/
padded scenario counts. No implicit float64-to-float32 precision downgrade occurs.

Full GPU research fails closed unless a successful CUDA qualification file
matches the precision, physical device parameters, time grid, substeps, source,
packages and actual GPU model. It then repeats a live training-only preflight
before calibration/search. CPU-only evidence cannot authorize CUDA. Full CPU
research and explicit short GPU development runs retain their existing behavior.

## Numerical implementation

The existing CPU class constructs the five-level Liouvillians and six-state
input/target/leakage observables. JAX reuses these exact matrices. At each linear
waveform interpolation midpoint, it computes exp(dt*L) and propagates all six
input density vectors. `lax.scan` integrates time, `vmap` batches devices and
`jit` compiles kernels. Reverse AD through `jax.scipy.linalg.expm` supplies
gradients equivalent to the existing Fréchet/adjoint method for the same
discretization. Rematerialization trades recomputation for bounded tape memory.

CVaR and worst-loss weights are formed over the complete ensemble using the
existing host formulas. Weighted batch VJPs are summed globally. CVaR is applied
to infidelity alone; leakage and regularization retain their original weights.
Averaging per-batch CVaRs would be incorrect and is explicitly tested against.
Gradient, fidelity, leakage, trace, Hermiticity and positivity checks use the
trusted CPU and the unchanged numerical tolerances. Fourier constraints and
fixed-duration logic run through the existing optimizer code.

The preflight uses at most three fixed training cases, with additional
100-scenario verification and all-size forward comparisons in the qualification
benchmark. It is a numerical qualification, not a confidence bound or a uniform
proof over uncertainty space. Each optimizer model instance checks its first
evaluated waveform again. Production nominal checks still qualify every retained
candidate; the backend gate does not prove uniform discretization accuracy over
all later optimizer iterates.

## Memory and cost

Scenarios stream from host memory in bounded chunks. Small ensembles use smaller
batches, tail rows are padded with zero objective weights, and shared JIT kernels
avoid recompilation for every ensemble size. A conservative working-set estimate
and available allocator bytes set the initial batch. OOM retries halve it and
ultimately fail/fall back rather than change physics or precision.

Preallocation defaults off before JAX import; existing environment choices are
respected. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before launching a new process
to ensure it takes effect. The requested budget is at most 8 GiB, with headroom
for a display driver/other applications. This is not a hard allocator sandbox:
compiler caches, existing processes and fragmentation can still cause OOM.
Actual 8 GB operation must be tested on the target GPU. See
[JAX memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

Both ordinary and qualified optimizers charge useful model and qualification
passes, including CPU reference probes, to their histories/resource ledgers.
The backend also reports padded work, forward/adjoint passes and OOM retries.
These operation counts are not FLOP-equivalent; actual wall timing remains
necessary. Final statistical certification is never moved to JAX.

## Benchmark and qualification

On the RTX host, run the full declared benchmark:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python jax_benchmark.py --output results/jax_gpu
```

It covers 1, 10, 100, 500, 1000 and 2000 common uncertainty scenarios in both
precisions, including forward solves and both objective gradients. The default
also times the expensive CPU analytic gradients at every size and includes a
matched JAX-CPU control at 1, 10 and 100 scenarios. Set `jax_cpu_counts` to all
six sizes to extend that additional control; the measured CPU control cost was
high, so the default bounds this extra sweep. Production CPU and CUDA still
cover every requested size. Cold/first-call time, warm time, static setup, process
RSS, available GPU allocator peaks, errors and both speedup definitions are
exported. Single-repetition measurements have no confidence interval; increase
`repetitions` in a copied config for a timing study.

The explicit `--no-cpu-gradients` option skips the large SciPy timing sweep while
still verifying both gradients on 100 distinct scenarios. It is recorded in the
effective configuration; skipped times/speedups remain unmeasured. Completed
per-case checkpoints resume under the same source/config protocol. A source or
config change requires a fresh output directory.

Missing CUDA produces `GPU_UNAVAILABLE` rows with null GPU times, VRAM and
speedups. It cannot generate a successful GPU qualification file. `jax_cpu`
numbers measure software on CPU and must not be presented as RTX speedups.

The optimizer integration test defaults to JAX on CPU. To exercise conventional,
risk-aware and continuation optimization on actual CUDA with fallback disabled:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  KAIROS_TEST_OPTIMIZER_BACKEND=jax_gpu \
  KAIROS_TEST_OPTIMIZER_PRECISION=float64 \
  .venv/bin/python -m unittest \
  test_jax_backend.JAXNumericalTests.test_optimizer_and_continuation_use_jax_but_nominal_qualification_is_cpu
```

Use `float32` to check that precision separately. A rejected accuracy gate is a
test failure, never an implicit CPU pass. GPU device access may need a host
terminal when a desktop execution sandbox blocks driver access.

Turing has much less FP64 hardware throughput than FP32: NVIDIA lists two FP64
versus 64 FP32 cores per SM. This hardware ratio does not determine the speedup
of small complex matrix exponentials. FP64 may or may not beat CPU for this
workload; only a synchronized run on the actual RTX answers that question.
[NVIDIA Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html).

## Measured RTX 2080 Super result

The Linux RTX 2080 SUPER (8192 MiB, driver 580.178.04) benchmark is saved in
`results/jax_rtx2080_qualified/`. **Use float64 for qualified training.**
In this checked workspace, set `qualification_report` to
`results/jax_rtx2080_qualified/gpu_qualification_float64.json`. Requalify after
changing the numerical source, packages, physical device, grid or substeps.
All six requested scenario counts were measured with production CPU references,
both CUDA precisions and conventional/risk gradients. The additional matched
JAX-CPU control covers 1, 10 and 100 scenarios.

At 2,000 scenarios, synchronized warm timings were:

| Operation | Existing CPU (s) | CUDA float64 (s) | CPU / CUDA |
|---|---:|---:|---:|
| Production comparison: forward | 273.81 | 19.03 | 14.39× |
| Conventional analytic gradient comparison | 267.32 | 83.56 | 3.20× |
| Risk-aware analytic gradient comparison | 540.82 | 103.87 | 5.21× |

The maximum float64 fidelity error across the full benchmark ensemble was
1.49e-7, and leakage error was 1.73e-8, below the unchanged 2e-6 tolerance.
All-state trace/Hermiticity/positivity checks on 2,000 scenarios passed.
Analytic GPU/CPU gradients agree to approximately 1e-15; the 100-scenario
directional finite-difference errors were below 1.2e-10. The measured cumulative
JAX GPU allocator peak was about 129.25 MiB at batch 32. This excludes driver,
CUDA-context and other-application memory; it is not total board consumption.

Float32 propagation was faster (5.38 s at 2,000 scenarios), but **failed**
qualification: maximum production metric error was 2.03e-5, trace error 2.07e-5,
and Hermiticity error 6.44e-6. The precision comparison is consistent with
accumulated floating-point error. No normalization, physics adjustment or
tolerance change was used to conceal it. Float32 remains available for explicit
diagnostics; its accuracy gate rejected this waveform and actual CPU fallback
reproduced the original CPU objective/gradient exactly. It cannot authorize a
full float32 GPU campaign.

All 115 unit tests and a separate real-CUDA test exercising conventional,
risk-aware and four-stage continuation optimization passed. A successful live
float64 preflight and actual initialization-failure fallback were also tested.
The benchmark report, CSV, equivalence JSON, source/config provenance, test logs
and qualification files are in the result directory. Timings have one warm
repetition; they are descriptive rather than confidence intervals. The existing
serial CPU implementation is the reference, and kernel gains do not predict
whole-campaign speedup. No full research campaign or final validation was run.
