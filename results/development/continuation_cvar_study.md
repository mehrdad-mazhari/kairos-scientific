# Continuation CVaR study

Mode: **intermediate**. Status: **COMPLETE DEVELOPMENT EXPERIMENT**. This is development on common selection scenarios, not independent validation. No validation dataset was generated.


Schedule: [0, 0.1, 0.5, 1.0]. The optimized, qualified waveform is transferred exactly to the next stage. Failed stages preserve the previous feasible waveform. All stages share the method-wide resource ledger; unused stage allocations roll forward. Final retention minimizes the final scheduled training objective over archived feasible improvements. No monotonic CVaR theorem is asserted.

| Regime | Seed | Stage lambda | Status | Nominal F | Nominal leakage | Training CVaR | Converged | Stop |
|---|---:|---:|---|---:|---:|---:|---|---|
| equal_objectives | 7500 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_objectives | 7500 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_objectives | 7500 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_objectives | 7500 | 1.0 | completed | 0.9996576777416722 | 3.8078424201239494e-05 | 0.00255462304856302 | False | Iteration limit reached; Iteration limit reached |
| equal_objectives | 7501 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_objectives | 7501 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_objectives | 7501 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_objectives | 7501 | 1.0 | completed | 0.9996668918707189 | 3.0268645723077427e-05 | 0.002502579694858584 | False | Optimization terminated successfully |
| equal_objectives | 7502 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_objectives | 7502 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_objectives | 7502 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_objectives | 7502 | 1.0 | completed | 0.9996664836130432 | 3.125475396487105e-05 | 0.0025208689463595705 | False | Iteration limit reached; Iteration limit reached |
| equal_passes | 7500 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_passes | 7500 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_passes | 7500 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_passes | 7500 | 1.0 | completed | 0.9996576777416722 | 3.8078424201239494e-05 | 0.00255462304856302 | False | Iteration limit reached; Iteration limit reached |
| equal_passes | 7501 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_passes | 7501 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_passes | 7501 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_passes | 7501 | 1.0 | completed | 0.9996668918707189 | 3.0268645723077427e-05 | 0.002502579694858584 | False | Optimization terminated successfully |
| equal_passes | 7502 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_passes | 7502 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_passes | 7502 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_passes | 7502 | 1.0 | completed | 0.9996664836130432 | 3.125475396487105e-05 | 0.0025208689463595705 | False | Iteration limit reached; Iteration limit reached |
| equal_wall | 7500 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_wall | 7500 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_wall | 7500 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_wall | 7500 | 1.0 | completed | 0.9996576777416722 | 3.8078424201239494e-05 | 0.00255462304856302 | False | Iteration limit reached; Iteration limit reached |
| equal_wall | 7501 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_wall | 7501 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_wall | 7501 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_wall | 7501 | 1.0 | completed | 0.9996668918707189 | 3.0268645723077427e-05 | 0.002502579694858584 | False | Optimization terminated successfully |
| equal_wall | 7502 | 0.0 | completed | 0.9996618695090635 | 2.198738092578934e-05 | 0.0033972247436951115 | False | Iteration limit reached |
| equal_wall | 7502 | 0.1 | completed | 0.9996480548110912 | 2.3291016800266264e-05 | 0.0033585801665778417 | False | Iteration limit reached |
| equal_wall | 7502 | 0.5 | completed | 0.9996569868673033 | 3.858029026094549e-05 | 0.0025683881567310943 | False | Iteration limit reached |
| equal_wall | 7502 | 1.0 | completed | 0.9996664836130432 | 3.125475396487105e-05 | 0.0025208689463595705 | False | Iteration limit reached; Iteration limit reached |

Stages can reduce their own changing objectives without reducing final CVaR. Read paired_selection_comparisons.csv for common-device outcomes. Unexecuted budget regimes in this profile: none. Missing trial checkpoints: none. No 2000-device validation was generated. Five-level/RWA noise extrapolation, ensemble-wide surrogate accuracy, larger-budget convergence, matched-cost variability and independent generalization remain limitations. No hardware results are claimed.
