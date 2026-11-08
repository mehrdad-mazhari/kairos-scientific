# Full matched-budget development study

**NO-GO: Incomplete development trials and/or numerical or convergence limitations remain.**
Completed selection-evaluated trials: 27/27. Backend: jax_gpu; FP64. No independent validation was generated.

All methods use the stronger nominal DRAG source. Its calibration was budget limited, not globally optimized or fully converged. Comparisons use the same 110 training and 200 selection devices, five harmonics, four interpolation substeps and unchanged physical/quality constraints.

| Regime | Seed | Method | Termination | Calls | Passes | Optimizer wall (s) | Converged |
|---|---:|---|---|---:|---:|---:|---|
| equal_objectives | 7500 | conventional | ITERATION_LIMIT | 180 | 19833 | 962.03 | False |
| equal_objectives | 7500 | risk | ITERATION_LIMIT | 222 | 48876 | 1455.96 | False |
| equal_objectives | 7500 | continuation | ITERATION_LIMIT | 199 | 43892 | 1316.59 | False |
| equal_objectives | 7501 | conventional | ITERATION_LIMIT | 182 | 20053 | 960.70 | False |
| equal_objectives | 7501 | risk | ITERATION_LIMIT | 208 | 45794 | 1348.81 | False |
| equal_objectives | 7501 | continuation | ITERATION_LIMIT | 181 | 39932 | 1197.73 | False |
| equal_objectives | 7502 | conventional | ITERATION_LIMIT | 180 | 19833 | 947.47 | False |
| equal_objectives | 7502 | risk | ITERATION_LIMIT | 220 | 48436 | 1420.54 | False |
| equal_objectives | 7502 | continuation | ITERATION_LIMIT | 199 | 43893 | 1292.83 | False |
| equal_passes | 7500 | conventional | ITERATION_LIMIT | 180 | 19833 | 991.89 | False |
| equal_passes | 7500 | risk | ITERATION_LIMIT | 222 | 48876 | 1448.21 | False |
| equal_passes | 7500 | continuation | ITERATION_LIMIT | 199 | 43892 | 1311.00 | False |
| equal_passes | 7501 | conventional | ITERATION_LIMIT | 182 | 20053 | 971.47 | False |
| equal_passes | 7501 | risk | ITERATION_LIMIT | 208 | 45794 | 1359.01 | False |
| equal_passes | 7501 | continuation | ITERATION_LIMIT | 181 | 39932 | 1186.95 | False |
| equal_passes | 7502 | conventional | ITERATION_LIMIT | 180 | 19833 | 945.27 | False |
| equal_passes | 7502 | risk | ITERATION_LIMIT | 220 | 48436 | 1416.24 | False |
| equal_passes | 7502 | continuation | ITERATION_LIMIT | 199 | 43893 | 1290.38 | False |
| equal_wall | 7500 | conventional | ITERATION_LIMIT | 180 | 19833 | 965.74 | False |
| equal_wall | 7500 | risk | ITERATION_LIMIT | 222 | 48876 | 1449.04 | False |
| equal_wall | 7500 | continuation | ITERATION_LIMIT | 199 | 43892 | 1323.50 | False |
| equal_wall | 7501 | conventional | ITERATION_LIMIT | 182 | 20053 | 955.88 | False |
| equal_wall | 7501 | risk | ITERATION_LIMIT | 208 | 45794 | 1344.42 | False |
| equal_wall | 7501 | continuation | ITERATION_LIMIT | 181 | 39932 | 1210.17 | False |
| equal_wall | 7502 | conventional | ITERATION_LIMIT | 180 | 19833 | 951.55 | False |
| equal_wall | 7502 | risk | ITERATION_LIMIT | 220 | 48436 | 1449.34 | False |
| equal_wall | 7502 | continuation | ITERATION_LIMIT | 199 | 43893 | 1330.66 | False |

| Regime | Seed | Method | Nominal F | Mean F | P05 F | CVaR95 | Mean leakage | Maximum leakage | Minimum F | Peak MHz |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| equal_objectives | 7500 | conventional | 0.9996617149 | 0.9986682400 | 0.9975475768 | 0.0028909417 | 2.66373901e-05 | 3.51331875e-05 | 0.9964504357 | 77.66990 |
| equal_objectives | 7500 | risk | 0.9996597383 | 0.9987569304 | 0.9977771364 | 0.00252316181 | 6.67316787e-05 | 0.000195093521 | 0.9970331783 | 77.64936 |
| equal_objectives | 7500 | continuation | 0.9996576777 | 0.9988557722 | 0.9979573514 | 0.00230914484 | 8.09670653e-05 | 0.000270727576 | 0.9973585623 | 77.64507 |
| equal_objectives | 7501 | conventional | 0.9996568105 | 0.9986733226 | 0.9975579997 | 0.00289201316 | 2.63775741e-05 | 3.1719672e-05 | 0.9964582588 | 77.66990 |
| equal_objectives | 7501 | risk | 0.9996655096 | 0.9987076348 | 0.9976960576 | 0.00267960837 | 3.93578788e-05 | 8.66426356e-05 | 0.9967483094 | 77.66990 |
| equal_objectives | 7501 | continuation | 0.9996668919 | 0.9988807646 | 0.9979909630 | 0.00226655031 | 5.73578923e-05 | 0.000168713943 | 0.9973529249 | 77.66990 |
| equal_objectives | 7502 | conventional | 0.9996645463 | 0.9986771196 | 0.9976211907 | 0.00288117038 | 2.68640626e-05 | 3.6903099e-05 | 0.9963749360 | 77.66990 |
| equal_objectives | 7502 | risk | 0.9996526720 | 0.9988480943 | 0.9979330512 | 0.00234353335 | 8.67640258e-05 | 0.000307081072 | 0.9972943418 | 77.66990 |
| equal_objectives | 7502 | continuation | 0.9996664836 | 0.9988825476 | 0.9980025673 | 0.00226894941 | 6.40840845e-05 | 0.000199973348 | 0.9973186354 | 77.66990 |
| equal_passes | 7500 | conventional | 0.9996617149 | 0.9986682400 | 0.9975475768 | 0.0028909417 | 2.66373901e-05 | 3.51331875e-05 | 0.9964504357 | 77.66990 |
| equal_passes | 7500 | risk | 0.9996597383 | 0.9987569304 | 0.9977771364 | 0.00252316181 | 6.67316787e-05 | 0.000195093521 | 0.9970331783 | 77.64936 |
| equal_passes | 7500 | continuation | 0.9996576777 | 0.9988557722 | 0.9979573514 | 0.00230914484 | 8.09670653e-05 | 0.000270727576 | 0.9973585623 | 77.64507 |
| equal_passes | 7501 | conventional | 0.9996568105 | 0.9986733226 | 0.9975579997 | 0.00289201316 | 2.63775741e-05 | 3.1719672e-05 | 0.9964582588 | 77.66990 |
| equal_passes | 7501 | risk | 0.9996655096 | 0.9987076348 | 0.9976960576 | 0.00267960837 | 3.93578788e-05 | 8.66426356e-05 | 0.9967483094 | 77.66990 |
| equal_passes | 7501 | continuation | 0.9996668919 | 0.9988807646 | 0.9979909630 | 0.00226655031 | 5.73578923e-05 | 0.000168713943 | 0.9973529249 | 77.66990 |
| equal_passes | 7502 | conventional | 0.9996645463 | 0.9986771196 | 0.9976211907 | 0.00288117038 | 2.68640626e-05 | 3.6903099e-05 | 0.9963749360 | 77.66990 |
| equal_passes | 7502 | risk | 0.9996526720 | 0.9988480943 | 0.9979330512 | 0.00234353335 | 8.67640258e-05 | 0.000307081072 | 0.9972943418 | 77.66990 |
| equal_passes | 7502 | continuation | 0.9996664836 | 0.9988825476 | 0.9980025673 | 0.00226894941 | 6.40840845e-05 | 0.000199973348 | 0.9973186354 | 77.66990 |
| equal_wall | 7500 | conventional | 0.9996617149 | 0.9986682400 | 0.9975475768 | 0.0028909417 | 2.66373901e-05 | 3.51331875e-05 | 0.9964504357 | 77.66990 |
| equal_wall | 7500 | risk | 0.9996597383 | 0.9987569304 | 0.9977771364 | 0.00252316181 | 6.67316787e-05 | 0.000195093521 | 0.9970331783 | 77.64936 |
| equal_wall | 7500 | continuation | 0.9996576777 | 0.9988557722 | 0.9979573514 | 0.00230914484 | 8.09670653e-05 | 0.000270727576 | 0.9973585623 | 77.64507 |
| equal_wall | 7501 | conventional | 0.9996568105 | 0.9986733226 | 0.9975579997 | 0.00289201316 | 2.63775741e-05 | 3.1719672e-05 | 0.9964582588 | 77.66990 |
| equal_wall | 7501 | risk | 0.9996655096 | 0.9987076348 | 0.9976960576 | 0.00267960837 | 3.93578788e-05 | 8.66426356e-05 | 0.9967483094 | 77.66990 |
| equal_wall | 7501 | continuation | 0.9996668919 | 0.9988807646 | 0.9979909630 | 0.00226655031 | 5.73578923e-05 | 0.000168713943 | 0.9973529249 | 77.66990 |
| equal_wall | 7502 | conventional | 0.9996645463 | 0.9986771196 | 0.9976211907 | 0.00288117038 | 2.68640626e-05 | 3.6903099e-05 | 0.9963749360 | 77.66990 |
| equal_wall | 7502 | risk | 0.9996526720 | 0.9988480943 | 0.9979330512 | 0.00234353335 | 8.67640258e-05 | 0.000307081072 | 0.9972943418 | 77.66990 |
| equal_wall | 7502 | continuation | 0.9996664836 | 0.9988825476 | 0.9980025673 | 0.00226894941 | 6.40840845e-05 | 0.000199973348 | 0.9973186354 | 77.66990 |
| shared | None | calibrated_drag | 0.9996591328 | 0.9984786463 | 0.9971924559 | 0.00321135941 | 4.78631969e-05 | 8.49679431e-05 | 0.9959593693 | 73.95362 |

| Regime | Seed | Method | Conventional − method CVaR | Improvement (%) | Mean F change | P05 change |
|---|---:|---|---:|---:|---:|---:|
| equal_objectives | 7500 | risk | 0.000367779891 | 12.7218 | 8.86904581e-05 | 0.000229559642 |
| equal_objectives | 7500 | continuation | 0.000581796863 | 20.1248 | 0.000187532218 | 0.000409774662 |
| equal_objectives | 7501 | risk | 0.000212404788 | 7.3445 | 3.43122105e-05 | 0.000138057916 |
| equal_objectives | 7501 | continuation | 0.000625462851 | 21.6272 | 0.00020744206 | 0.00043296333 |
| equal_objectives | 7502 | risk | 0.000537637037 | 18.6604 | 0.000170974722 | 0.000311860498 |
| equal_objectives | 7502 | continuation | 0.000612220972 | 21.2490 | 0.000205428002 | 0.000381376553 |
| equal_passes | 7500 | risk | 0.000367779891 | 12.7218 | 8.86904581e-05 | 0.000229559642 |
| equal_passes | 7500 | continuation | 0.000581796863 | 20.1248 | 0.000187532218 | 0.000409774662 |
| equal_passes | 7501 | risk | 0.000212404788 | 7.3445 | 3.43122105e-05 | 0.000138057916 |
| equal_passes | 7501 | continuation | 0.000625462851 | 21.6272 | 0.00020744206 | 0.00043296333 |
| equal_passes | 7502 | risk | 0.000537637037 | 18.6604 | 0.000170974722 | 0.000311860498 |
| equal_passes | 7502 | continuation | 0.000612220972 | 21.2490 | 0.000205428002 | 0.000381376553 |
| equal_wall | 7500 | risk | 0.000367779891 | 12.7218 | 8.86904581e-05 | 0.000229559642 |
| equal_wall | 7500 | continuation | 0.000581796863 | 20.1248 | 0.000187532218 | 0.000409774662 |
| equal_wall | 7501 | risk | 0.000212404788 | 7.3445 | 3.43122105e-05 | 0.000138057916 |
| equal_wall | 7501 | continuation | 0.000625462851 | 21.6272 | 0.00020744206 | 0.00043296333 |
| equal_wall | 7502 | risk | 0.000537637037 | 18.6604 | 0.000170974722 | 0.000311860498 |
| equal_wall | 7502 | continuation | 0.000612220972 | 21.2490 | 0.000205428002 | 0.000381376553 |

Positive CVaR deltas favor the risk method. These are paired development observations on common devices, not independent statistical superiority. Three seeds are not 600 independent devices. No inferential interval or significance claim is manufactured.

Equal-call comparisons do not equalize compute. Equal-pass ledgers include surrogate passes and CPU nominal channels as separate quantities. Equal-wall caps include initialization, restarts, propagation, constraints and nominal qualification, checked between indivisible evaluations. Overshoot is explicit in computational_costs.csv. Fixed post-optimization CPU selection, post-trial numerical verification and rendering are measurement/analysis overhead outside the optimizer cap and separately recorded.

Known optimizer GPU-call wall seconds: 32636.786. synchronized optimizer GPU calls including transfer/dispatch/JIT; excludes CPU probes, shared setup and post-trial numerical verification; not CUDA-kernel-only time. Shared GPU qualification/warmup costs are recorded in gpu_qualification.json. Historical baseline calibration is in common_reference.json. Lambda/mu were not tuned: hyperparameter-search cost in this campaign is zero.

Converged: 0/27. Convergence requires all predeclared KKT, relative-objective, parameter-change and feasibility criteria, not objective decrease alone. Numerical failures: []. Checks are pointwise at nominal/two joint stress cases with four/eight substeps, not uniform accuracy guarantees.

## Direct answers

Does fixed-risk improve CVaR?
- equal_objectives: {'paired_seeds': 3, 'mean_delta_cvar': 0.00037260723869787506, 'mean_percent_improvement': 12.90890077703934, 'mean_fidelity_change': 9.799246350239432e-05, 'p05_change': 0.00022649268550082682}
- equal_passes: {'paired_seeds': 3, 'mean_delta_cvar': 0.00037260723869787506, 'mean_percent_improvement': 12.90890077703934, 'mean_fidelity_change': 9.799246350239432e-05, 'p05_change': 0.00022649268550082682}
- equal_wall: {'paired_seeds': 3, 'mean_delta_cvar': 0.00037260723869787506, 'mean_percent_improvement': 12.90890077703934, 'mean_fidelity_change': 9.799246350239432e-05, 'p05_change': 0.00022649268550082682}
Does continuation improve CVaR relative to conventional?
- equal_objectives: {'paired_seeds': 3, 'mean_delta_cvar': 0.0006064935620707125, 'mean_percent_improvement': 21.000368962450683, 'mean_fidelity_change': 0.00020013409336228083, 'p05_change': 0.00040803818185686236}
- equal_passes: {'paired_seeds': 3, 'mean_delta_cvar': 0.0006064935620707125, 'mean_percent_improvement': 21.000368962450683, 'mean_fidelity_change': 0.00020013409336228083, 'p05_change': 0.00040803818185686236}
- equal_wall: {'paired_seeds': 3, 'mean_delta_cvar': 0.0006064935620707125, 'mean_percent_improvement': 21.000368962450683, 'mean_fidelity_change': 0.00020013409336228083, 'p05_change': 0.00040803818185686236}

Additional continuation-versus-fixed-risk improvement must be assessed from matched seed/regime selection_metrics.csv; no winner is asserted from an incomplete study.

Equal-pass and equal-wall results must both support any efficiency claim; favorable equal-call results alone are insufficient.

Unfinished trials: none.

Resume after an interruption (an active-worker lock prevents duplication):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python full_matched_study.py --config full_matched_config.json --output results/full_matched_budget --backend gpu
```

Confirmatory study command: **not issued**. The confirmatory protocol is not frozen while required trials, convergence or numerical checks remain unresolved. No final independent validation has been generated.

## Prespecified final paired analysis

The following answers use completed strong-reference checkpoints only. A positive delta favors the second method. The range is observed variation across three optimizer seeds, not a confidence interval. Device-level pairs are in paired_selection_devices.csv; reused devices are not independent extra replications.

| Comparison | Regime | Paired seeds | Mean CVaR delta | Seed range | Interpretation |
|---|---|---:|---:|---|---|
| fixed_risk_vs_conventional | equal_objectives | 3 | 0.00037260723869787506 | [0.00021240478804817036, 0.0005376370369115533] | Lower empirical mean CVaR for risk |
| fixed_risk_vs_conventional | equal_passes | 3 | 0.00037260723869787506 | [0.00021240478804817036, 0.0005376370369115533] | Lower empirical mean CVaR for risk |
| fixed_risk_vs_conventional | equal_wall | 3 | 0.00037260723869787506 | [0.00021240478804817036, 0.0005376370369115533] | Lower empirical mean CVaR for risk |
| continuation_vs_conventional | equal_objectives | 3 | 0.0006064935620707125 | [0.0005817968626485428, 0.0006254628513215118] | Lower empirical mean CVaR for continuation |
| continuation_vs_conventional | equal_passes | 3 | 0.0006064935620707125 | [0.0005817968626485428, 0.0006254628513215118] | Lower empirical mean CVaR for continuation |
| continuation_vs_conventional | equal_wall | 3 | 0.0006064935620707125 | [0.0005817968626485428, 0.0006254628513215118] | Lower empirical mean CVaR for continuation |
| continuation_vs_fixed_risk | equal_objectives | 3 | 0.00023388632337283748 | [7.458393533052964e-05, 0.00041305806327334143] | Lower empirical mean CVaR for continuation |
| continuation_vs_fixed_risk | equal_passes | 3 | 0.00023388632337283748 | [7.458393533052964e-05, 0.00041305806327334143] | Lower empirical mean CVaR for continuation |
| continuation_vs_fixed_risk | equal_wall | 3 | 0.00023388632337283748 | [7.458393533052964e-05, 0.00041305806327334143] | Lower empirical mean CVaR for continuation |

Fixed-risk versus conventional answers question 1. Continuation versus fixed-risk answers question 2. The separate equal-pass/equal-wall rows and recorded cost ratios answer question 3; no statistical or population-level superiority follows.

Final development decision: **NO-GO**. Interrupted work requires separate cost audit before a matched-efficiency claim.

Completed trials: 27/27; optimizer GPU-call wall seconds: 32636.786.

No independent validation was generated. A confirmatory command remains withheld while the final protocol is unfrozen or development limitations remain. Existing research_benchmark.py does not implement this three-method/three-regime continuation confirmation; a command for that different study would be misleading.
