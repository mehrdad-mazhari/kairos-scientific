# Intermediate benchmark

Mode: **intermediate**. Status: **COMPLETE DEVELOPMENT EXPERIMENT**. This is development on common selection scenarios, not independent validation. No validation dataset was generated.


Requested full intermediate maxima: 150 aggregate iterations, 1000 objective calls in the equal-call regime, five restart allocations, five harmonics, 100 random training devices plus ten deterministic cases and 200 selection devices. Executed profile caps: 150 iterations, 1000 calls, 4 interpolation substeps, seeds [7500, 7501, 7502], regimes ['equal_objectives', 'equal_passes', 'equal_wall']. Budgets are maxima, not promises of executed work.


The reused reference nominal calibration used 414 calls and reported optimizer_success=False. Its pulse is nominally qualified, but a budget-limited calibration is not an optimized-to-convergence DRAG baseline. Improvements relative to this reference do not establish an algorithmic advantage over fully calibrated DRAG. All optimized methods share this exact reference; risk-versus-conventional differences remain conditional on this start.


| Regime | Method | Seed | Nominal F | Mean selection F | Selection P05 | Selection CVaR95 | Mean selection leakage | Selection feasible | Converged |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| equal_objectives | conventional | 7500 | 0.9996617149 | 0.9986682400 | 0.9975475768 | 0.0028909417 | 2.66373901e-05 | True | False |
| equal_objectives | risk | 7500 | 0.9996597383 | 0.9987569304 | 0.9977771364 | 0.00252316181 | 6.67316787e-05 | False | False |
| equal_objectives | continuation | 7500 | 0.9996576777 | 0.9988557722 | 0.9979573514 | 0.00230914484 | 8.09670653e-05 | False | False |
| equal_objectives | conventional | 7501 | 0.9996568105 | 0.9986733226 | 0.9975579997 | 0.00289201316 | 2.63775741e-05 | True | False |
| equal_objectives | risk | 7501 | 0.9996655096 | 0.9987076348 | 0.9976960576 | 0.00267960837 | 3.93578788e-05 | True | False |
| equal_objectives | continuation | 7501 | 0.9996668919 | 0.9988807646 | 0.9979909630 | 0.00226655031 | 5.73578923e-05 | False | False |
| equal_objectives | conventional | 7502 | 0.9996645463 | 0.9986771196 | 0.9976211907 | 0.00288117038 | 2.68640626e-05 | True | False |
| equal_objectives | risk | 7502 | 0.9996526720 | 0.9988480943 | 0.9979330512 | 0.00234353335 | 8.67640258e-05 | False | False |
| equal_objectives | continuation | 7502 | 0.9996664836 | 0.9988825476 | 0.9980025673 | 0.00226894941 | 6.40840845e-05 | False | False |
| equal_passes | conventional | 7500 | 0.9996617149 | 0.9986682400 | 0.9975475768 | 0.0028909417 | 2.66373901e-05 | True | False |
| equal_passes | risk | 7500 | 0.9996597383 | 0.9987569304 | 0.9977771364 | 0.00252316181 | 6.67316787e-05 | False | False |
| equal_passes | continuation | 7500 | 0.9996576777 | 0.9988557722 | 0.9979573514 | 0.00230914484 | 8.09670653e-05 | False | False |
| equal_passes | conventional | 7501 | 0.9996568105 | 0.9986733226 | 0.9975579997 | 0.00289201316 | 2.63775741e-05 | True | False |
| equal_passes | risk | 7501 | 0.9996655096 | 0.9987076348 | 0.9976960576 | 0.00267960837 | 3.93578788e-05 | True | False |
| equal_passes | continuation | 7501 | 0.9996668919 | 0.9988807646 | 0.9979909630 | 0.00226655031 | 5.73578923e-05 | False | False |
| equal_passes | conventional | 7502 | 0.9996645463 | 0.9986771196 | 0.9976211907 | 0.00288117038 | 2.68640626e-05 | True | False |
| equal_passes | risk | 7502 | 0.9996526720 | 0.9988480943 | 0.9979330512 | 0.00234353335 | 8.67640258e-05 | False | False |
| equal_passes | continuation | 7502 | 0.9996664836 | 0.9988825476 | 0.9980025673 | 0.00226894941 | 6.40840845e-05 | False | False |
| equal_wall | conventional | 7500 | 0.9996617149 | 0.9986682400 | 0.9975475768 | 0.0028909417 | 2.66373901e-05 | True | False |
| equal_wall | risk | 7500 | 0.9996597383 | 0.9987569304 | 0.9977771364 | 0.00252316181 | 6.67316787e-05 | False | False |
| equal_wall | continuation | 7500 | 0.9996576777 | 0.9988557722 | 0.9979573514 | 0.00230914484 | 8.09670653e-05 | False | False |
| equal_wall | conventional | 7501 | 0.9996568105 | 0.9986733226 | 0.9975579997 | 0.00289201316 | 2.63775741e-05 | True | False |
| equal_wall | risk | 7501 | 0.9996655096 | 0.9987076348 | 0.9976960576 | 0.00267960837 | 3.93578788e-05 | True | False |
| equal_wall | continuation | 7501 | 0.9996668919 | 0.9988807646 | 0.9979909630 | 0.00226655031 | 5.73578923e-05 | False | False |
| equal_wall | conventional | 7502 | 0.9996645463 | 0.9986771196 | 0.9976211907 | 0.00288117038 | 2.68640626e-05 | True | False |
| equal_wall | risk | 7502 | 0.9996526720 | 0.9988480943 | 0.9979330512 | 0.00234353335 | 8.67640258e-05 | False | False |
| equal_wall | continuation | 7502 | 0.9996664836 | 0.9988825476 | 0.9980025673 | 0.00226894941 | 6.40840845e-05 | False | False |
| shared | calibrated DRAG | — | 0.9996591328 | 0.9984786463 | 0.9971924559 | 0.00321135941 | 4.78631969e-05 | True | calibration reference |

| Regime | Seed | Risk method | Conventional CVaR minus method CVaR | Mean F difference | P05 difference | Nominal F difference | Mean leakage difference | Runtime difference (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| equal_objectives | 7500 | risk | 0.000367779891 | 8.86904581e-05 | 0.000229559642 | -1.97656224e-06 | 4.00942886e-05 | 493.934 |
| equal_objectives | 7500 | continuation | 0.000581796863 | 0.000187532218 | 0.000409774662 | -4.03713547e-06 | 5.43296752e-05 | 354.560 |
| equal_objectives | 7501 | risk | 0.000212404788 | 3.43122105e-05 | 0.000138057916 | 8.69917201e-06 | 1.29803047e-05 | 388.114 |
| equal_objectives | 7501 | continuation | 0.000625462851 | 0.00020744206 | 0.00043296333 | 1.00814149e-05 | 3.09803181e-05 | 237.028 |
| equal_objectives | 7502 | risk | 0.000537637037 | 0.000170974722 | 0.000311860498 | -1.18743074e-05 | 5.98999631e-05 | 473.067 |
| equal_objectives | 7502 | continuation | 0.000612220972 | 0.000205428002 | 0.000381376553 | 1.93727064e-06 | 3.72200218e-05 | 345.362 |
| equal_passes | 7500 | risk | 0.000367779891 | 8.86904581e-05 | 0.000229559642 | -1.97656224e-06 | 4.00942886e-05 | 456.320 |
| equal_passes | 7500 | continuation | 0.000581796863 | 0.000187532218 | 0.000409774662 | -4.03713547e-06 | 5.43296752e-05 | 319.105 |
| equal_passes | 7501 | risk | 0.000212404788 | 3.43122105e-05 | 0.000138057916 | 8.69917201e-06 | 1.29803047e-05 | 387.539 |
| equal_passes | 7501 | continuation | 0.000625462851 | 0.00020744206 | 0.00043296333 | 1.00814149e-05 | 3.09803181e-05 | 215.478 |
| equal_passes | 7502 | risk | 0.000537637037 | 0.000170974722 | 0.000311860498 | -1.18743074e-05 | 5.98999631e-05 | 470.966 |
| equal_passes | 7502 | continuation | 0.000612220972 | 0.000205428002 | 0.000381376553 | 1.93727064e-06 | 3.72200218e-05 | 345.107 |
| equal_wall | 7500 | risk | 0.000367779891 | 8.86904581e-05 | 0.000229559642 | -1.97656224e-06 | 4.00942886e-05 | 483.297 |
| equal_wall | 7500 | continuation | 0.000581796863 | 0.000187532218 | 0.000409774662 | -4.03713547e-06 | 5.43296752e-05 | 357.757 |
| equal_wall | 7501 | risk | 0.000212404788 | 3.43122105e-05 | 0.000138057916 | 8.69917201e-06 | 1.29803047e-05 | 388.542 |
| equal_wall | 7501 | continuation | 0.000625462851 | 0.00020744206 | 0.00043296333 | 1.00814149e-05 | 3.09803181e-05 | 254.286 |
| equal_wall | 7502 | risk | 0.000537637037 | 0.000170974722 | 0.000311860498 | -1.18743074e-05 | 5.98999631e-05 | 497.795 |
| equal_wall | 7502 | continuation | 0.000612220972 | 0.000205428002 | 0.000381376553 | 1.93727064e-06 | 3.72200218e-05 | 379.112 |

Positive delta CVaR favors the named risk method. Every difference uses common selection devices; these observations support no statistical-superiority claim. Hyperparameters and method choice must not be presented as independently validated after this development study.

The ledger includes all starts/stages and unsuccessful nominal qualification attempts. Equal-call budgets do not equalize dissipative work because risk gradients require two scenario passes. Equal-pass budgets count both surrogate passes and production nominal channels, which are not identical in FLOPs. Equal-wall budgets include fitting, qualification, gradients and monitoring, checked between indivisible solves. A single call can overshoot a wall deadline; actual costs are reported. Reference calibration and development-only gradient/initialization/selection studies are separate shared or diagnostic costs.

Convergence requires optimizer success, constraint satisfaction, small local KKT gradient, small relative objective change and small coefficient change. Objective reduction alone is insufficient. Metrics at retained feasible improvements use production nominal fidelity/leakage and surrogate training quantities; selection metrics use production propagation. A retained proposal can be a line-search trial rather than an SLSQP major iterate.


Unexecuted budget regimes in this profile: none. Missing trial checkpoints: none. No 2000-device validation was generated. Five-level/RWA noise extrapolation, ensemble-wide surrogate accuracy, larger-budget convergence, matched-cost variability and independent generalization remain limitations. No hardware results are claimed.

Continuation selection delta CVaR range: [0.000581796863, 0.000625462851]. These are observations at the executed caps, not a population-level winner.

Decision: do not launch confirmatory validation based on this diagnostic. Finish the predeclared matched-budget development comparisons and numerical checks before freezing a final research protocol.
