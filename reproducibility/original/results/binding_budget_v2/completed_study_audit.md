# Completed 27-trial audit

All 27 trials are complete; none converged. All nine method/seed groups have identical stored waveform hashes across regimes. This is a stopping-rule confound, not evidence of equal computational efficiency.

The cause is `grape_development.optimizer_options`: every regime has `max_total_iterations=150`. The refiner divides it among five starts; continuation divides it across its stages and starts. All trials reached iteration limits before their advertised resource cap. The deterministic method/seed trajectories therefore repeat. Changing a nonbinding limit cannot change the result. Stored coefficients and hashes, rather than just aggregated metrics, confirm the equality.

|Method|Actual joint value-gradient calls|Actual total passes|Actual optimizer seconds|
|---|---:|---:|---:|
|Conventional|180–182|19833–20053|945.27–991.89|
|Fixed risk|208–222|45794–48876|1344.42–1455.96|
|Continuation|181–199|39932–43893|1186.95–1330.66|

The named caps were 1000 calls, 220500 passes, and 1800 seconds respectively. All were nonbinding. Other resource dimensions were intentionally uncapped. Wall differences include execution conditions and dispatch; they do not establish matched wall time.

All 27 call counts agree with summed per-stage evaluation traces. All total-pass ledgers equal surrogate passes plus production solves. Training propagation accounting is 110 devices times calls times one for conventional or two for risk/continuation, with additional accuracy-qualification work. Joint value-gradient calls count as one objective and one gradient evaluation, not two separate calls. Production nominal qualification and per-model accuracy-gate work are charged. Passes are ledger units, not FLOP-equivalent costs: CPU channel solves and different gradient propagations differ in actual work. Wall deadlines are checked between calls, allowing a running call to overrun; this was not the cause of the completed experiment's termination.

Two interrupted attempts add known lower bounds of **206 calls, 45488 passes, 1356.764 seconds**, absent from successful-trial sums. Uncheckpointed in-flight work may add more. Shared calibration, warmup, numerical verification, and selection are also outside optimizer counters. Successful-trial cost is not total campaign cost.

Per-trial fidelity, CVaR95, nominal and ensemble leakage, termination and cap accounting are in `completed_audit.csv` and `audit.json`. Nominal qualification does not establish ensemble leakage compliance. The completed study supports descriptive performance observations at 150 aggregate iterations, not matched-cost efficiency claims.

## Separate corrective experiment

The completed directory and original code are preserved and SHA256-manifested. Explicit new optimizer forks change resource scheduling and reservation only. They retain the same physics, Fourier representation, gradients, quality thresholds, saved training/selection devices and nominal retention rules.

The new protocol uses two budget levels: **24/48 joint calls; 5280/10560 total ledger passes; 120/240 optimizer seconds**. Three saved seeds and three methods produce 54 trials. These are exploratory development cost curves, not a converged research campaign. A 100000-iteration safety guard allows the named resource to bind. Stage and restart allocations share the intended resource, carrying unused work forward. A possible nominal solve is reserved before admitting a gradient call. Small residual pass budgets are unavoidable because calls are indivisible.

Common wall deadlines allow at most an in-flight call overrun, recorded explicitly. A cap is an allowance, not exact expenditure. Compare actual resource usage and utilization; early convergence or unused budget must not be called matched expenditure. No efficiency ranking is warranted where expenditures differ materially. Qualification and warmup are shared and separately timed; runtime accuracy checks and nominal qualification remain charged to trials. Pointwise CPU/GPU checks gate completed trials. Convergence criteria remain unchanged.

Read `performance_vs_cost.csv`, `status.json`, and `study_report.md` as the separate study completes. Three seeds on 200 reused development devices provide descriptive evidence only. No independent validation devices are generated. No efficiency superiority is established.

## Completed study performance (three seeds; repeated regimes omitted)

|Method|Nominal fidelity|Selection mean fidelity|CVaR95 infidelity|Nominal leakage|Mean selection leakage|Worst selection leakage|
|---|---:|---:|---:|---:|---:|---:|
|conventional|0.999661024|0.998672894|0.00288804175|2.27237974e-05|2.66263423e-05|3.6903099e-05|
|risk|0.999659307|0.998770886|0.00251543451|3.36807191e-05|6.42845278e-05|0.000307081072|
|continuation|0.999663684|0.998873028|0.00228154818|3.3200608e-05|6.74696807e-05|0.000270727576|
