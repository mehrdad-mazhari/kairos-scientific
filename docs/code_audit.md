# Code and evidence audit

| Area | Modules / evidence |
|---|---|
| Five-level Duffing/RWA, collapse operators, six-state CPU channels | `kairos.challenge_model`, `kairos.preconfirmatory_checks` |
| Fourier controls, Frechet gradients, conventional GRAPE | `kairos.grape`, `kairos.grape_research` |
| CVaR infidelity and risk gradients | `kairos.risk_aware_grape` |
| Continuation and shared budgets | `kairos.continuation_risk_grape`, `kairos.binding_budget_continuation`, `kairos.binding_budget_optimizer` |
| DRAG baseline and calibration | `kairos.reference_calibration`, `kairos.pulse_methods`, `kairos.pulse_shapes` |
| Optional JAX jit/vmap backend and numerical gating | `kairos.jax_backend`, `kairos.jax_benchmark` |
| Monte Carlo and distribution-free certification | `kairos.challenge_model`, `kairos.statistical_certification` |
| Frozen paired bootstrap and reporting | `kairos.confirmatory_statistics`, `kairos.confirmatory_analysis_v1` |
| Original 27 development trials | `reproducibility/original/results/full_matched_budget` |
| Corrected 54 trials and actual cost accounting | `reproducibility/original/results/binding_budget_v2` |
| Hilbert and CPU solver convergence | `reproducibility/original/results/preconfirmatory_checks_v1` |
| Independent cohort, checkpoints and reporting | `reproducibility/original/results/confirmatory_validation_v1` |

The package converts local imports to relative imports; numerical code is preserved. Historical drivers remain available for inspection but retain historical working-directory assumptions. Use the verified examples as the portable package entry points. Frozen executors remain archival: moving files does not authorize a new independent campaign. The retained GUI is a legacy optional Tk interface and is not needed for CPU simulation; Tk requires an OS package on some Linux distributions. Frozen source snapshots are deliberately retained, not redundant editable implementations.
