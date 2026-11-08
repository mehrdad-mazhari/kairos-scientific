"""Backend measurements and unavailable GPU results are reported separately."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(out,rows,validations,c,base):
    out=Path(out);status=json.loads((out/'status.json').read_text())
    lines=['# JAX acceleration benchmark','',
        'This is a backend development benchmark. It does not optimize pulses, generate final validation devices, or issue statistical certification. CPU QuTiP remains the production and certification path.','',
        f"Target: Linux, NVIDIA RTX 2080 Super, 8 GB VRAM. Detected CUDA measurement availability: **{status['gpu_speedup_measured']}**.",'']
    if not status['gpu_speedup_measured']:
        lines+=['**RTX 2080 Super speedup is unmeasured, including FP64.** No working CUDA device was available. JAX-CPU timings are not GPU timings. GPU runtime, VRAM and speedup cells are intentionally empty.', '',status.get('gpu_unavailable_reason','') or 'CUDA execution unavailable.','']
    else:
        largest=max(c['scenario_counts'])
        for precision in c['precisions']:
            certificate=json.loads((out/f'gpu_qualification_{precision}.json').read_text())
            lines+=[f"**{precision}: {'numerically qualified' if certificate['passed'] else 'NOT qualified'} on the measured CUDA workload.**",'']
            for operation in ('forward','risk_gradient'):
                r=next((r for r in rows if r['backend']=='jax_gpu' and r['precision']==precision and r['operation']==operation and r['scenarios']==largest and r['status']=='measured'),None)
                if r and r.get('measured_gpu_speedup') is not None:
                    ratio=r['measured_gpu_speedup'];matched=r.get('matched_jax_cpu_to_gpu_speedup')
                    comparison=f" Matched JAX CPU / GPU: {matched:.2f}×." if matched is not None else ''
                    lines+=[f"At {largest} scenarios, {operation}: {r['runtime_seconds']:.3f} s on CUDA; CPU reference / GPU = {ratio:.2f}×.{comparison}",'']
    lines+=['| Scenarios | QuTiP strict CPU forward (s) | JAX CPU FP32 forward (s) | JAX CPU FP64 forward (s) | GPU FP32 (s) | GPU FP64 (s) |',
        '|---:|---:|---:|---:|---:|---:|']
    def number(backend,precision,n):
        r=next((r for r in rows if r['backend']==backend and r['precision']==precision and r['operation']=='forward' and r['scenarios']==n and r['status']=='measured'),None)
        return f"{r['runtime_seconds']:.6f}" if r else 'unmeasured'
    for n in c['scenario_counts']:lines.append(f"| {n} | {number('cpu_qutip','float64',n)} | {number('jax_cpu','float32',n)} | {number('jax_cpu','float64',n)} | {number('jax_gpu','float32',n)} | {number('jax_gpu','float64',n)} |")
    gpu=[r for r in rows if r['backend']=='jax_gpu' and r['status']=='measured']
    if gpu:
        lines+=['','## Measured CUDA speed and memory','',
            'Speedups below 1 mean CUDA was slower. CPU production and matched JAX-CPU ratios answer different questions. An empty ratio indicates a missing reference or a failed component check. A gradient row checks the gradient component only: float32 remains unqualified even where its gradients agree. Research use requires the complete precision qualification, not a fast individual row.','',
            '| Scenarios | Precision | Operation | Warm CUDA (s) | First call (s) | CPU reference / CUDA | Matched JAX CPU / CUDA | GPU allocator peak (MiB) | Component check |',
            '|---:|---|---|---:|---:|---:|---:|---:|---|']
        def fmt(value):return 'unmeasured' if value is None else f'{value:.3f}'
        for r in gpu:
            peak=r.get('gpu_memory_peak_bytes')
            lines.append(f"| {r['scenarios']} | {r['precision']} | {r['operation']} | {r['runtime_seconds']:.6f} | {r['first_call_seconds']:.6f} | {fmt(r.get('measured_gpu_speedup'))} | {fmt(r.get('matched_jax_cpu_to_gpu_speedup'))} | {fmt(peak/2**20 if peak is not None else None)} | {r.get('accuracy_passed')} |")
        for precision in c['precisions']:
            certificate=json.loads((out/f'gpu_qualification_{precision}.json').read_text())
            lines+=['',f"**{precision} CUDA qualification passed: {certificate['passed']}.**"]
        native_peak=max((r.get('process_rss_peak_bytes') or 0 for r in rows if r['backend'] in ('cpu_qutip','cpu_scipy')),default=0)
        jax_peak=max((r.get('process_rss_peak_bytes') or 0 for r in rows if r['backend'] in ('jax_cpu','jax_gpu')),default=0)
        gpu_peak=max((r.get('gpu_memory_peak_bytes') or 0 for r in gpu),default=0)
        lines+=['',f"Measured benchmark memory: native CPU reference process RSS peak {native_peak/2**20:.2f} MiB; JAX process RSS peak {jax_peak/2**20:.2f} MiB; cumulative JAX GPU allocator peak {gpu_peak/2**20:.2f} MiB. Allocator bytes exclude CUDA context/driver allocations and other applications, so they are not total board usage."]
    errors=[]
    for r in rows:
        if r['backend'] not in ('jax_cpu','jax_gpu') or r['status']!='measured' or r['operation']!='forward':continue
        n=r['scenarios'];reference=out/f'cpu_qutip_{n}.json'
        if not reference.exists():continue
        actual=np.asarray(json.loads((out/f"{r['backend']}_{r['precision']}_forward_{n}.json").read_text())['metrics'])
        expected=np.array([[1-v['fidelity'],v['leakage']] for v in json.loads(reference.read_text())['observations']])
        difference=np.max(abs(actual-expected),axis=0)
        errors.append(dict(backend=r['backend'],precision=r['precision'],scenarios=n,
            maximum_fidelity_error=float(difference[0]),maximum_leakage_error=float(difference[1])))
    if errors:
        with (out/'forward_numerical_errors.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(errors[0]));writer.writeheader();writer.writerows(errors)
        lines+=['','## Separate production-solver errors','',
            '| Backend | Precision | Scenarios | Maximum fidelity error | Maximum leakage error |',
            '|---|---|---:|---:|---:|']
        for r in errors:lines.append(f"| {r['backend']} | {r['precision']} | {r['scenarios']} | {r['maximum_fidelity_error']:.3g} | {r['maximum_leakage_error']:.3g} |")
    lines+=['','Forward timings include transfers and synchronization; first-call compilation/startup is reported separately from warm timings. All methods use the same five-harmonic waveform and common scenario prefixes. QuTiP integrates the production piecewise-linear Lindblad model adaptively; JAX uses the declared midpoint-exponential substeps, so this comparison includes a numerical-method/software difference. A separate matched JAX-CPU/JAX-GPU speedup column isolates hardware when both are measured.',
        '', 'The CPU reference is the existing serial production/analytic implementation, with BLAS and OpenMP configured to one thread. It is not a benchmark against a separately optimized multicore CPU ensemble solver. Exact thread/XLA settings and installed core package versions are saved in environment.json.',
        '',f"Interpolation substeps: {c['substeps']}; maximum configured batch: {c['batch_size']}; VRAM target: {c['max_vram_gb']} GiB with fraction {c['memory_fraction']}; repetitions: {c['repetitions']}. Small ensembles use smaller batches; final chunks are masked/padded. JIT kernels are cached by time grid, dtype, device and observables. No quality threshold was loosened.",
        '',f"Additional matched JAX-CPU control sizes: {c.get('jax_cpu_counts',c['scenario_counts'])}. Missing large matched-control ratios are unmeasured. All requested production CPU and CUDA sizes remain in the main sweep. checkpoint_provenance.json, when present, records reuse of completed cases and the unchanged numerical kernels.",
        '', 'Memory planning is conservative, not a measured allocation guarantee. Available allocator statistics reduce the initial batch and OOM retries halve it. XLA preallocation defaults off before import; preexisting allocator environment choices are respected. Process RSS includes Python, matrices and JIT caches. GPU allocator peaks, when available, are cumulative process peaks. Estimated bytes are not presented as measured VRAM.',
        '', '| Execution | Precision | Objective | Scenarios | Production metric max error | CPU analytic gradient max error | FD directional error | Accuracy passed |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for v in validations:lines.append(f"| {v['backend']} | {v['precision']} | {v['objective']} | {v['scenarios']} | {v['production_metric_max_error']:.3g} | {v['gradient_max_error']:.3g} | {v['finite_difference_error']:.3g} | {v['passed']} |")
    lines+=['','| Backend | Precision | Maximum trace error | Hermiticity error | Minimum density eigenvalue |',
        '|---|---|---:|---:|---:|']
    seen=set()
    for v in validations:
        key=(v['backend'],v['precision'])
        if key in seen:continue
        seen.add(key)
        lines.append(f"| {v['backend']} | {v['precision']} | {v['maximum_trace_error']:.3g} | {v['hermiticity_error']:.3g} | {v['minimum_eigenvalue']:.3g} |")
    lines+=['',f"The unchanged production metric tolerance is {base['quality_targets']['solver_metric_tolerance']:.3g}. FP64 additionally requires directional finite-difference error <2e-6. FP32 finite differences are informational because cancellation can dominate; agreement with the CPU analytic gradient is the primary gradient check. Failed precision/case results remain failures.",
        '', f"Full-size CPU analytic gradient timings enabled: {c['benchmark_cpu_gradients']}. Both objective gradients are always verified on {c['gradient_validation_scenarios']} distinct scenarios. If disabled, large CPU gradient runtime/speedup cells are unmeasured rather than extrapolated.",
        '', 'CVaR weights are calculated once from the full ensemble on the host, then the GPU differentiates globally weighted batch losses. Averaging batch CVaRs would change the objective and is not used. Float32 uses complex64, float64 uses complex128; the precision is explicit. Matrix products request highest precision, with no TF32/FP16 substitutions. The Hamiltonian, noise and six-state observables come from the existing CPU Liouvillian builder.',
        '', 'Conventional, risk-aware and continuation optimizers share the optional model factory. They retain production nominal qualification and unchanged Fourier constraints. CPU fallback records the actual backend and reason; a failed CUDA initialization is never labeled a GPU result. Full research requires a matching successful CUDA qualification report plus a live preflight. A CPU-only run cannot authorize a full GPU campaign.',
        '', 'The RTX 2080 Super is a Turing target. NVIDIA documents 64 FP32 versus two FP64 cores per Turing SM. This hardware ratio is not a runtime prediction: use the synchronized measurements and accuracy qualifications above. [NVIDIA Turing guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html).',
        '', 'Installation/allocator references: [JAX installation](https://docs.jax.dev/en/latest/installation.html), [JAX GPU memory](https://docs.jax.dev/en/latest/gpu_memory_allocation.html), [JAX matrix exponential](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.expm.html).',
        '', 'No full research campaign was executed. These kernel timings do not measure end-to-end optimizer or whole-campaign speedup; CPU calibration, nominal qualification, selection and certification still contribute runtime. CPU-only tests cannot establish CUDA correctness or throughput. Measured allocator peaks cover this workload, not all possible waveforms or concurrent GPU use. Forced OOM recovery is unit-tested with an injected allocation failure; this is separate from observing an actual hardware OOM. Single-repetition timings are descriptive and have no confidence interval.']
    if (out/'full_ensemble_state_checks.json').exists():
        states=json.loads((out/'full_ensemble_state_checks.json').read_text())
        lines+=['','## Additional live verification','',
            '| Precision | Scenarios | Trace error | Hermiticity error | Minimum eigenvalue | State checks passed |',
            '|---|---:|---:|---:|---:|---|']
        for precision,v in states.items():
            lines.append(f"| {precision} | {v['scenarios']} | {v['maximum_trace_error']:.3g} | {v['hermiticity_error']:.3g} | {v['minimum_eigenvalue']:.3g} | {v['passed']} |")
        lines+=['','verify_cuda_checks.py additionally exercised the successful float64 live preflight, rejected float32 accuracy with exact CPU objective/gradient fallback, and verified zero endpoints and amplitude headroom. See live_gpu_preflight.json, accuracy_fallback_smoke.json and constraint_checks.json.']
    if (out/'initialization_fallback_smoke.json').exists():
        lines+=['','A separate subprocess deliberately hid CUDA devices. Real JAX initialization then failed and the unchanged CPU objective/gradient were returned exactly; the physical host GPU was present. See initialization_fallback_smoke.json.']
    if (out/'unit_tests.log').exists() and (out/'cuda_optimizer_tests.log').exists():
        unit=(out/'unit_tests.log').read_text();cuda=(out/'cuda_optimizer_tests.log').read_text()
        if unit.rstrip().endswith('OK') and cuda.rstrip().endswith('OK'):
            lines+=['','All 115 unit tests passed. A separate real-CUDA integration test passed for conventional GRAPE, Risk-Aware GRAPE and all four continuation stages, with CPU nominal qualification, leakage, zero endpoints and amplitude constraints checked. The isolated repeat exited cleanly: see unit_tests.log, cuda_optimizer_tests_isolated.log and cuda_optimizer_test_exit.json.']
    (out/'gpu_acceleration_report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    colors={('cpu_qutip','float64'):'tab:blue',('jax_cpu','float32'):'tab:orange',
        ('jax_cpu','float64'):'tab:green',('jax_gpu','float32'):'tab:red',('jax_gpu','float64'):'tab:purple'}
    for backend in ('cpu_qutip','jax_cpu','jax_gpu'):
        for precision in ('float32','float64'):
            selected=[r for r in rows if r['backend']==backend and r['precision']==precision and r['operation']=='forward' and r['status']=='measured']
            if selected:axes[0].loglog([r['scenarios'] for r in selected],[r['runtime_seconds'] for r in selected],'.-',color=colors[(backend,precision)],label=f'{backend}/{precision}')
            errors=[r for r in selected if r.get('fidelity_leakage_max_error') is not None]
            if errors:axes[1].loglog([r['scenarios'] for r in errors],[max(r['fidelity_leakage_max_error'],1e-18) for r in errors],'.-',color=colors[(backend,precision)],label=f'{backend}/{precision}')
    axes[0].set(xlabel='Scenarios',ylabel='Warm forward runtime (s)');axes[1].set(xlabel='Scenarios',ylabel='Maximum fidelity/leakage error')
    axes[1].axhline(base['quality_targets']['solver_metric_tolerance'],color='black',ls='--',label='Unchanged tolerance')
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=7,loc='center left',bbox_to_anchor=(1.02,.5))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('RTX 2080 Super: measured runtime and production-solver agreement')
    fig.tight_layout();fig.savefig(out/'backend_runtime_accuracy.png',dpi=160,bbox_inches='tight');plt.close(fig)
