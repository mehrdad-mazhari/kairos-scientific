# Resume binding-budget audit study

User authorized audit of the completed 27 trials and a separate correctly resource-limited experiment. The original completed directory must remain unchanged. Its full file manifest is completed_experiment_manifest.json. Audit findings are completed_study_audit.md, completed_audit.csv, audit.json.

Separate new worker: binding_budget_study.py. Launcher PID and command in launcher.json. GPU access requires host execution. Check host process identity and study.lock before launching a duplicate. The worker performs 54 trials: 2 resource levels x 3 resources x 3 saved seeds x 3 methods, using only existing 110 training and 200 selection devices. No independent validation. Existing physical constraints and scientific thresholds unchanged.

Read status.json, execution.log, performance_vs_cost.csv, study_report.md. The runner freezes all source hashes in protocol.json. Do not edit those sources while running. New optimizer files are explicit forks; original files are untouched. Numerical checks gate every completed trial. Report actual utilization and overshoot; no efficiency claim from a cap label alone. Early convergence and underutilization remain limitations.

Resume stopped worker with project venv and OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kairos-mpl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_cpu_multi_thread_eigen=false; execute .venv/bin/python -u binding_budget_study.py from repository root with host GPU permission. Completed optimization checkpoints are reused; interrupted progress is archived and the unfinished trial restarts. Sunk work must be included in total campaign cost. No independent final validation path exists in this runner.

Tests before launch: 15 passed (14 inherited Fourier/continuation/ledger tests plus one test that separately verifies binding call/pass/wall budgets and restart allocation with a high iteration guard). The toy wall test uses a fake clock with explicitly initialized budget time.

Do not claim this exploratory short-budget study establishes convergence, statistical superiority, or genuine efficiency unless actual costs are sufficiently matched. A wall deadline is nonpreemptive and can overrun by an in-flight call; actual overhead is recorded.

## Latest completion request

User explicitly requests completion and analysis of all 54 trials under the existing protocol. Do not redesign algorithms, change thresholds or generate validation. GPU worker remains PID11512 (check host identity before actions). Final analysis worker PID12940 waits on study.lock; final command: .venv/bin/python -u corrective_study_analysis.py --wait. Its launcher and source SHA are analysis_launcher.json; source snapshot in analysis_source_snapshot/. Do not modify the queued analysis source without replacing that worker.

As of this handoff: 6/54 independently verified, original experiment and frozen sources unchanged, all available numerical checks passed. 25 relevant automated tests passed. Snapshot reports, CSVs and eight PDF/PNG/SVG figure types exist and are explicitly incomplete. Finalizer regenerates them after worker release, reports missing trials if interrupted, and creates confirmatory_protocol_draft.md only when all54 complete and readiness is GO/CONDITIONAL GO. It never generates validation devices. The proposed paired bootstrap protocol uses future seed941206 and bootstrap941207 but neither validation draw is made.

Exact CPU nominal wall and per-restart full wall were never recorded in frozen optimization telemetry. Report solve counts, non-GPU residual upper bounds and trace spans with their scope, not fabricated exact times. Stage ledgers and total optimizer wall are available. The 5% realized-cost comparability rule is descriptive and was introduced after first3 low-call trials, not a preregistered hypothesis. Convergence is not required for a valid resource-limited development trial.
