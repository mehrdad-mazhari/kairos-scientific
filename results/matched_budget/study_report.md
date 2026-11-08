# Binding computational-budget study

54/54 trials complete. Development selection only; no independent validation.

Caps are upper bounds, not proof of equal expenditure. Compare actual resource consumption and utilization in performance_vs_cost.csv. Wall deadlines are checked between synchronized calls: one in-flight call may overrun. No efficiency superiority is established. Early convergence or unused cap must be reported rather than called a matched comparison.

Two predeclared levels: 24/48 joint objective-gradient calls; 5280/10560 total dissipative passes; 120/240 optimizer seconds. Three saved seeds, three methods, five restart allocations, unchanged continuation schedule and physical constraints. The 100000-iteration ceiling is a safety guard, not an intended stopping rule. Per-stage and per-restart resource shares carry unused work forward. All qualification and nominal retention inside optimization are charged; shared warmup, post-trial numerical checks, and CPU selection are separately timed. Pass units are ledger units, not equivalent FLOPs. Risk uses two forward/gradient passes per device; conventional one. Interrupted trials are archived and their sunk work is additional campaign cost.

|Regime / level / method|n|Mean CVaR95|Mean fidelity|Mean leakage|Mean actual cost|
|---|---:|---:|---:|---:|---:|
|equal_objectives/0/conventional|3|0.003225352|0.998505917|2.3556825e-05|24|
|equal_objectives/0/risk|3|0.0032085073|0.998499176|2.3546945e-05|24|
|equal_objectives/0/continuation|3|0.0031761514|0.998509121|2.2693524e-05|24|
|equal_objectives/1/conventional|3|0.0032577724|0.998514603|2.195161e-05|48|
|equal_objectives/1/risk|3|0.0030505627|0.998563926|6.1459887e-05|48|
|equal_objectives/1/continuation|3|0.0028925569|0.998633055|2.9975448e-05|48|
|equal_passes/0/conventional|3|0.0032680887|0.998516969|2.1878467e-05|5196.7|
|equal_passes/0/risk|3|0.0032085073|0.998499176|2.3546945e-05|5090|
|equal_passes/0/continuation|3|0.0031747672|0.998514889|2.2016816e-05|5152|
|equal_passes/1/conventional|3|0.0028880417|0.998672894|2.6626342e-05|10483|
|equal_passes/1/risk|3|0.0032033203|0.998504147|2.4946182e-05|10371|
|equal_passes/1/continuation|3|0.0029201167|0.998616071|3.1344647e-05|10445|
|equal_wall/0/conventional|3|0.0032257656|0.998502561|2.3331295e-05|122.56|
|equal_wall/0/risk|3|0.0032236841|0.998494231|2.366043e-05|121.92|
|equal_wall/0/continuation|3|0.0032047712|0.998506352|2.3627062e-05|121.9|
|equal_wall/1/conventional|3|0.0032577724|0.998514603|2.195161e-05|243.47|
|equal_wall/1/risk|3|0.0031931285|0.998505824|2.2312385e-05|241.86|
|equal_wall/1/continuation|3|0.0031186562|0.998533379|2.2293091e-05|243.39|
