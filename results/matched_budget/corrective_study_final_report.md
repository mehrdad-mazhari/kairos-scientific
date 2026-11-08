# Corrective study development report

**54/54 complete. CONDITIONAL GO.** Complete verified budget-limited study; freeze confirmatory candidates and protocol after development-only Hilbert truncation checks and acknowledging approximate wall matching and timing limitations.

The original 27-trial experiment is unchanged. Frozen protocol and saved ensembles are checked independently. No independent validation devices were generated.

Budget levels 0/1 are respectively 24/48 joint objective-gradient calls, 5280/10560 total scenario-pass units, and 120/240 optimizer seconds. All runs use the same 110 training and 200 selection scenarios, five Fourier harmonics, five-level Lindblad model and unchanged physical/quality constraints.

## Direct answers

1–3. Fixed risk versus conventional, continuation versus conventional, and continuation versus fixed risk are tabulated below. Positive ΔCVaR favors the candidate. These are three paired optimizer seeds evaluated on the same 200 development devices, not independent replications of device sampling. Signs and seed ranges are descriptive; no inferential superiority is established.

|Regime|Level|Candidate / reference|Seeds|Mean ΔCVaR|Seed range|Seeds favoring candidate|Actual-cost comparable pairs|
|---|---:|---|---:|---:|---|---:|---:|
|equal_objectives|0|risk/conventional|3|1.6844698e-05|[-1.5743464e-05, 4.9626484e-05]|2|3|
|equal_objectives|0|continuation/conventional|3|4.9200646e-05|[3.5022725e-05, 6.8445922e-05]|3|3|
|equal_objectives|0|continuation/risk|3|3.2355947e-05|[1.8819438e-05, 5.0766188e-05]|3|3|
|equal_objectives|1|risk/conventional|3|0.00020720976|[6.3851883e-05, 0.00029499122]|3|3|
|equal_objectives|1|continuation/conventional|3|0.00036521551|[0.00034677623, 0.00038699702]|3|3|
|equal_objectives|1|continuation/risk|3|0.00015800575|[5.1785019e-05, 0.0002980214]|3|3|
|equal_passes|0|risk/conventional|3|5.9581415e-05|[1.5660669e-06, 0.00010060771]|3|3|
|equal_passes|0|continuation/conventional|3|9.3321554e-05|[5.3716447e-05, 0.00012081134]|3|3|
|equal_passes|0|continuation/risk|3|3.3740139e-05|[2.020363e-05, 5.215038e-05]|3|3|
|equal_passes|1|risk/conventional|3|-0.00031527857|[-0.00032952734, -0.00030754382]|0|3|
|equal_passes|1|continuation/conventional|3|-3.207497e-05|[-3.8946334e-05, -2.8103557e-05]|0|3|
|equal_passes|1|continuation/risk|3|0.0002832036|[0.0002783688, 0.00029058101]|3|3|
|equal_wall|0|risk/conventional|3|2.0814917e-06|[-6.448655e-06, 1.6430882e-05]|1|3|
|equal_wall|0|continuation/conventional|3|2.0994482e-05|[1.0497122e-05, 3.6365351e-05]|3|3|
|equal_wall|0|continuation/risk|3|1.8912991e-05|[1.4234874e-05, 2.256963e-05]|3|3|
|equal_wall|1|risk/conventional|3|6.4643896e-05|[3.7146066e-05, 8.6034512e-05]|3|3|
|equal_wall|1|continuation/conventional|3|0.00013911622|[0.00010982742, 0.00016154785]|3|3|
|equal_wall|1|continuation/risk|3|7.4472324e-05|[7.2681359e-05, 7.5513341e-05]|3|3|

Descriptive answers at comparable realized costs:
- risk versus conventional: 12/18 comparable pairs favor the candidate; mixed across available seeds/budgets. Both pulses meet development engineering eligibility in 16/18 pairs. This is descriptive, conditional on the named resource, and not a population superiority test.
- continuation versus conventional: 15/18 comparable pairs favor the candidate; mixed across available seeds/budgets. Both pulses meet development engineering eligibility in 18/18 pairs. This is descriptive, conditional on the named resource, and not a population superiority test.
- continuation versus risk: 18/18 comparable pairs favor the candidate; uniformly favorable in available pairs. Both pulses meet development engineering eligibility in 16/18 pairs. This is descriptive, conditional on the named resource, and not a population superiority test.

Do not collapse budget regimes. Equal calls can involve roughly twice the pass cost for risk methods. The 5% realized-cost rule is a descriptive rule added after the first three low-call outcomes; it does not create statistical significance. Pairs with material cost mismatch are reported without an efficiency interpretation.

4. Leakage trade-offs: see leakage_tradeoff_analysis.md and paired_method_comparisons.csv. Nominal qualification and ensemble leakage are different. All engineering thresholds stay unchanged.

5. Integrity checks: {'frozen_sources': True, 'original_experiment_unchanged': True, 'training_ensemble_unchanged': True, 'selection_ensemble_unchanged': True, 'all_optimization_checkpoints_valid': True}. Invalid completed trial checks: []. Resource utilization, overshoot and exact termination strings are in resource_utilization.csv. A 100000-iteration guard is checked for nonbinding status. Nonpreemptive wall deadlines allow an in-flight operation to finish; utilization is not silently forced to 100%.

6. Numerical verification passed for 54/54 completed waveforms. Scope: nominal plus two existing training stress cases, 4/8 substeps, CPU/GPU pointwise metrics and gradients, density physicality and sampled waveform constraints. Not a uniform numerical bound, not independent validation, not convergence with Hilbert-space size.

7. Strongest feasible development method: conventional. Selection uses the existing engineering criteria, then minimum development CVaR95. It selects a fixed waveform, not a universally superior optimization algorithm.

## Computational costs and limitations

Successful optimizer wall: 12560.869 s; synchronized GPU-backend wall: 12338.968 s; calls: 2016; passes: 353449. Selection: 532.981 s; numerical checks: 147.152 s. Shared qualification/warmup and interrupted-work lower bounds are in campaign_cost_accounting.json.

CPU nominal solve counts are recorded, but exact CPU nominal wall time was not separately instrumented. Wall minus synchronized GPU backend time is only an upper bound including orchestration and CPU surrogate checks. Exact restart timers are also unavailable; trace spans are lower bounds, and stage incremental ledgers are available. The frozen experiment is not modified to retrofit timing. These limits prevent kernel-level or exact CPU-time efficiency claims; total optimizer wall remains directly measured.

## Reproducibility and remaining work

Verify the checkpoint and input hash manifest, complete all 54 trials, inspect all eight figures and unfavorable seeds, review feasibility, run selected-waveform development-only truncation/convergence checks, then freeze the draft protocol and source snapshots. Independent validation remains prohibited here. Journal claims need the frozen CPU-confirmatory results, clear model/distribution limitations, and external/hardware replication where appropriate.

Resume command (only if no worker holds study.lock):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kairos-mpl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_cpu_multi_thread_eigen=false .venv/bin/python -u binding_budget_study.py
```

Checkpoint files: trial_*_optimization.json preserve completed optimization before CPU selection; trial_*.json preserve complete checked trials. Unfinished progress files are archived as interrupted attempts before restarting their trial. Existing completed trials are reused. Final analysis: `.venv/bin/python corrective_study_analysis.py --wait`.

No superiority or final readiness claim is issued from an incomplete experiment.
