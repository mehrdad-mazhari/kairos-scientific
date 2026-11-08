# JAX acceleration benchmark

This is a backend development benchmark. It does not optimize pulses, generate final validation devices, or issue statistical certification. CPU QuTiP remains the production and certification path.

Target: Linux, NVIDIA RTX 2080 Super, 8 GB VRAM. Detected CUDA measurement availability: **True**.

**float32: NOT qualified on the measured CUDA workload.**

**float64: numerically qualified on the measured CUDA workload.**

At 2000 scenarios, forward: 19.033 s on CUDA; CPU reference / GPU = 14.39×.

At 2000 scenarios, risk_gradient: 103.869 s on CUDA; CPU reference / GPU = 5.21×.

| Scenarios | QuTiP strict CPU forward (s) | JAX CPU FP32 forward (s) | JAX CPU FP64 forward (s) | GPU FP32 (s) | GPU FP64 (s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.137295 | 0.023058 | 0.046831 | 0.089916 | 0.142246 |
| 10 | 1.359535 | 0.215659 | 0.486695 | 0.078797 | 0.184180 |
| 100 | 13.556608 | 8.041653 | 8.609125 | 0.331382 | 1.206956 |
| 500 | 67.827570 | unmeasured | unmeasured | 1.358094 | 4.790194 |
| 1000 | 137.095452 | unmeasured | unmeasured | 2.713361 | 9.696874 |
| 2000 | 273.809017 | unmeasured | unmeasured | 5.383766 | 19.032599 |

## Measured CUDA speed and memory

Speedups below 1 mean CUDA was slower. CPU production and matched JAX-CPU ratios answer different questions. An empty ratio indicates a missing reference or a failed component check. A gradient row checks the gradient component only: float32 remains unqualified even where its gradients agree. Research use requires the complete precision qualification, not a fast individual row.

| Scenarios | Precision | Operation | Warm CUDA (s) | First call (s) | CPU reference / CUDA | Matched JAX CPU / CUDA | GPU allocator peak (MiB) | Component check |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | float32 | forward | 0.089916 | 0.509838 | unmeasured | unmeasured | 100.053 | False |
| 1 | float32 | conventional_gradient | 0.255518 | 0.970020 | 0.530 | 0.400 | 128.029 | True |
| 1 | float32 | risk_gradient | 0.324815 | 0.329890 | 0.815 | 0.383 | 128.029 | True |
| 10 | float32 | forward | 0.078797 | 0.441572 | unmeasured | unmeasured | 128.029 | False |
| 10 | float32 | conventional_gradient | 0.288741 | 1.079068 | 4.572 | 4.826 | 128.200 | True |
| 10 | float32 | risk_gradient | 0.361751 | 0.373289 | 7.294 | 3.852 | 128.200 | True |
| 100 | float32 | forward | 0.331382 | 0.695303 | unmeasured | unmeasured | 128.200 | False |
| 100 | float32 | conventional_gradient | 1.242140 | 2.013704 | 10.736 | 19.635 | 128.620 | True |
| 100 | float32 | risk_gradient | 1.615271 | 1.586506 | 16.616 | 21.808 | 128.620 | True |
| 500 | float32 | forward | 1.358094 | 1.343513 | unmeasured | unmeasured | 128.620 | False |
| 500 | float32 | conventional_gradient | 5.146082 | 5.118601 | 13.014 | unmeasured | 128.620 | True |
| 500 | float32 | risk_gradient | 6.396926 | 6.425548 | unmeasured | unmeasured | 128.620 | False |
| 1000 | float32 | forward | 2.713361 | 2.742922 | unmeasured | unmeasured | 128.620 | False |
| 1000 | float32 | conventional_gradient | 10.213865 | 10.200435 | 13.012 | unmeasured | 128.620 | True |
| 1000 | float32 | risk_gradient | 12.893751 | 12.854722 | unmeasured | unmeasured | 128.620 | False |
| 2000 | float32 | forward | 5.383766 | 5.375647 | unmeasured | unmeasured | 128.620 | False |
| 2000 | float32 | conventional_gradient | 20.205283 | 20.159119 | 13.230 | unmeasured | 128.620 | True |
| 2000 | float32 | risk_gradient | 25.758975 | 25.412724 | unmeasured | unmeasured | 128.620 | False |
| 1 | float64 | forward | 0.142246 | 0.488935 | 0.965 | 0.329 | 128.620 | True |
| 1 | float64 | conventional_gradient | 0.502269 | 1.378730 | 0.270 | 0.438 | 128.620 | True |
| 1 | float64 | risk_gradient | 0.653040 | 0.655889 | 0.406 | 0.402 | 128.620 | True |
| 10 | float64 | forward | 0.184180 | 0.527892 | 7.382 | 2.642 | 128.620 | True |
| 10 | float64 | conventional_gradient | 0.690613 | 1.754746 | 1.911 | 3.443 | 128.620 | True |
| 10 | float64 | risk_gradient | 0.880621 | 0.882764 | 2.996 | 3.473 | 128.620 | True |
| 100 | float64 | forward | 1.206956 | 1.575960 | 11.232 | 7.133 | 128.620 | True |
| 100 | float64 | conventional_gradient | 5.357603 | 6.355813 | 2.489 | 7.047 | 129.246 | True |
| 100 | float64 | risk_gradient | 6.687685 | 6.614040 | 4.013 | 6.807 | 129.246 | True |
| 500 | float64 | forward | 4.790194 | 4.860959 | 14.160 | unmeasured | 129.246 | True |
| 500 | float64 | conventional_gradient | 21.310772 | 21.299134 | 3.143 | unmeasured | 129.246 | True |
| 500 | float64 | risk_gradient | 26.055113 | 26.012186 | 5.166 | unmeasured | 129.246 | True |
| 1000 | float64 | forward | 9.696874 | 9.651175 | 14.138 | unmeasured | 129.246 | True |
| 1000 | float64 | conventional_gradient | 42.270921 | 42.335758 | 3.144 | unmeasured | 129.246 | True |
| 1000 | float64 | risk_gradient | 51.955962 | 52.255860 | 5.170 | unmeasured | 129.246 | True |
| 2000 | float64 | forward | 19.032599 | 19.008815 | 14.386 | unmeasured | 129.246 | True |
| 2000 | float64 | conventional_gradient | 83.558182 | 83.704003 | 3.199 | unmeasured | 129.246 | True |
| 2000 | float64 | risk_gradient | 103.868576 | 102.836294 | 5.207 | unmeasured | 129.246 | True |

**float32 CUDA qualification passed: False.**

**float64 CUDA qualification passed: True.**

Measured benchmark memory: native CPU reference process RSS peak 214.86 MiB; JAX process RSS peak 2001.62 MiB; cumulative JAX GPU allocator peak 129.25 MiB. Allocator bytes exclude CUDA context/driver allocations and other applications, so they are not total board usage.

## Separate production-solver errors

| Backend | Precision | Scenarios | Maximum fidelity error | Maximum leakage error |
|---|---|---:|---:|---:|
| jax_cpu | float32 | 1 | 2.59e-07 | 4.95e-07 |
| jax_cpu | float32 | 10 | 1.23e-05 | 4.95e-07 |
| jax_cpu | float32 | 100 | 1.48e-05 | 7.08e-07 |
| jax_cpu | float64 | 1 | 3.37e-08 | 1.02e-08 |
| jax_cpu | float64 | 10 | 6.25e-08 | 1.32e-08 |
| jax_cpu | float64 | 100 | 1.25e-07 | 1.59e-08 |
| jax_gpu | float32 | 1 | 9.92e-06 | 4.06e-07 |
| jax_gpu | float32 | 10 | 1.82e-05 | 4.7e-07 |
| jax_gpu | float32 | 100 | 1.88e-05 | 7.86e-07 |
| jax_gpu | float32 | 500 | 1.88e-05 | 7.92e-07 |
| jax_gpu | float32 | 1000 | 1.91e-05 | 8.62e-07 |
| jax_gpu | float32 | 2000 | 2.03e-05 | 8.62e-07 |
| jax_gpu | float64 | 1 | 3.37e-08 | 1.02e-08 |
| jax_gpu | float64 | 10 | 6.25e-08 | 1.32e-08 |
| jax_gpu | float64 | 100 | 1.25e-07 | 1.59e-08 |
| jax_gpu | float64 | 500 | 1.34e-07 | 1.73e-08 |
| jax_gpu | float64 | 1000 | 1.42e-07 | 1.73e-08 |
| jax_gpu | float64 | 2000 | 1.49e-07 | 1.73e-08 |

Forward timings include transfers and synchronization; first-call compilation/startup is reported separately from warm timings. All methods use the same five-harmonic waveform and common scenario prefixes. QuTiP integrates the production piecewise-linear Lindblad model adaptively; JAX uses the declared midpoint-exponential substeps, so this comparison includes a numerical-method/software difference. A separate matched JAX-CPU/JAX-GPU speedup column isolates hardware when both are measured.

The CPU reference is the existing serial production/analytic implementation, with BLAS and OpenMP configured to one thread. It is not a benchmark against a separately optimized multicore CPU ensemble solver. Exact thread/XLA settings and installed core package versions are saved in environment.json.

Interpolation substeps: 4; maximum configured batch: 32; VRAM target: 8.0 GiB with fraction 0.7; repetitions: 1. Small ensembles use smaller batches; final chunks are masked/padded. JIT kernels are cached by time grid, dtype, device and observables. No quality threshold was loosened.

Additional matched JAX-CPU control sizes: [1, 10, 100]. Missing large matched-control ratios are unmeasured. All requested production CPU and CUDA sizes remain in the main sweep. checkpoint_provenance.json, when present, records reuse of completed cases and the unchanged numerical kernels.

Memory planning is conservative, not a measured allocation guarantee. Available allocator statistics reduce the initial batch and OOM retries halve it. XLA preallocation defaults off before import; preexisting allocator environment choices are respected. Process RSS includes Python, matrices and JIT caches. GPU allocator peaks, when available, are cumulative process peaks. Estimated bytes are not presented as measured VRAM.

| Execution | Precision | Objective | Scenarios | Production metric max error | CPU analytic gradient max error | FD directional error | Accuracy passed |
|---|---|---|---:|---:|---:|---:|---|
| jax_cpu | float32 | conventional | 100 | 1.48e-05 | 7.02e-08 | 8.66e-05 | False |
| jax_cpu | float32 | risk | 100 | 1.48e-05 | 1.01e-07 | 0.000333 | False |
| jax_cpu | float64 | conventional | 100 | 1.25e-07 | 1.05e-16 | 8.19e-12 | True |
| jax_cpu | float64 | risk | 100 | 1.25e-07 | 1.63e-16 | 1.38e-10 | True |
| jax_gpu | float32 | conventional | 100 | 1.88e-05 | 2.13e-07 | 0.000151 | False |
| jax_gpu | float32 | risk | 100 | 1.88e-05 | 2.09e-07 | 0.000527 | False |
| jax_gpu | float64 | conventional | 100 | 1.25e-07 | 1.77e-16 | 1.58e-12 | True |
| jax_gpu | float64 | risk | 100 | 1.25e-07 | 2.15e-16 | 1.17e-10 | True |

| Backend | Precision | Maximum trace error | Hermiticity error | Minimum density eigenvalue |
|---|---|---:|---:|---:|
| jax_cpu | float32 | 1.28e-05 | 1.07e-06 | 4.62e-10 |
| jax_cpu | float64 | 1.38e-14 | 4.59e-15 | 4.58e-10 |
| jax_gpu | float32 | 1.8e-05 | 5.84e-06 | 4.63e-10 |
| jax_gpu | float64 | 1.83e-14 | 6.65e-15 | 4.58e-10 |

The unchanged production metric tolerance is 2e-06. FP64 additionally requires directional finite-difference error <2e-6. FP32 finite differences are informational because cancellation can dominate; agreement with the CPU analytic gradient is the primary gradient check. Failed precision/case results remain failures.

Full-size CPU analytic gradient timings enabled: True. Both objective gradients are always verified on 100 distinct scenarios. If disabled, large CPU gradient runtime/speedup cells are unmeasured rather than extrapolated.

CVaR weights are calculated once from the full ensemble on the host, then the GPU differentiates globally weighted batch losses. Averaging batch CVaRs would change the objective and is not used. Float32 uses complex64, float64 uses complex128; the precision is explicit. Matrix products request highest precision, with no TF32/FP16 substitutions. The Hamiltonian, noise and six-state observables come from the existing CPU Liouvillian builder.

Conventional, risk-aware and continuation optimizers share the optional model factory. They retain production nominal qualification and unchanged Fourier constraints. CPU fallback records the actual backend and reason; a failed CUDA initialization is never labeled a GPU result. Full research requires a matching successful CUDA qualification report plus a live preflight. A CPU-only run cannot authorize a full GPU campaign.

The RTX 2080 Super is a Turing target. NVIDIA documents 64 FP32 versus two FP64 cores per Turing SM. This hardware ratio is not a runtime prediction: use the synchronized measurements and accuracy qualifications above. [NVIDIA Turing guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html).

Installation/allocator references: [JAX installation](https://docs.jax.dev/en/latest/installation.html), [JAX GPU memory](https://docs.jax.dev/en/latest/gpu_memory_allocation.html), [JAX matrix exponential](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.expm.html).

No full research campaign was executed. These kernel timings do not measure end-to-end optimizer or whole-campaign speedup; CPU calibration, nominal qualification, selection and certification still contribute runtime. CPU-only tests cannot establish CUDA correctness or throughput. Measured allocator peaks cover this workload, not all possible waveforms or concurrent GPU use. Forced OOM recovery is unit-tested with an injected allocation failure; this is separate from observing an actual hardware OOM. Single-repetition timings are descriptive and have no confidence interval.

## Additional live verification

| Precision | Scenarios | Trace error | Hermiticity error | Minimum eigenvalue | State checks passed |
|---|---:|---:|---:|---:|---|
| float32 | 2000 | 1.9e-05 | 2.93e-06 | 4.2e-10 | False |
| float64 | 2000 | 2.2e-14 | 7.72e-15 | 4.18e-10 | True |

verify_cuda_checks.py additionally exercised the successful float64 live preflight, rejected float32 accuracy with exact CPU objective/gradient fallback, and verified zero endpoints and amplitude headroom. See live_gpu_preflight.json, accuracy_fallback_smoke.json and constraint_checks.json.

A separate subprocess deliberately hid CUDA devices. Real JAX initialization then failed and the unchanged CPU objective/gradient were returned exactly; the physical host GPU was present. See initialization_fallback_smoke.json.

All 115 unit tests passed. A separate real-CUDA integration test passed for conventional GRAPE, Risk-Aware GRAPE and all four continuation stages, with CPU nominal qualification, leakage, zero endpoints and amplitude constraints checked. The isolated repeat exited cleanly: see unit_tests.log, cuda_optimizer_tests_isolated.log and cuda_optimizer_test_exit.json.
