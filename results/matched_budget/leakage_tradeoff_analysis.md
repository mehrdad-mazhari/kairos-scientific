# Development fidelity–leakage trade-offs

54/54 complete; one shared 200-device selection ensemble. Positive ΔCVaR favors the candidate; positive leakage change is worse. Nominal leakage is a retention qualification, not an ensemble guarantee. Threshold remains 1e-4 for nominal and maximum sampled leakage.

|Method|Completed|Nominal F range|Mean F range|CVaR range|Mean leakage range|Maximum leakage range|Eligible|
|---|---:|---|---|---|---|---|---:|
|conventional|18|[0.9996438, 0.9996734]|[0.9985003, 0.9986771]|[0.00288117, 0.003295579]|[2.148348e-05, 2.686406e-05]|[2.65692e-05, 3.925136e-05]|18|
|risk|18|[0.9996532, 0.9996803]|[0.9984906, 0.9986093]|[0.002933492, 0.003226918]|[2.219878e-05, 8.709369e-05]|[2.707701e-05, 0.000355285]|16|
|continuation|18|[0.9996706, 0.99968]|[0.998505, 0.9986346]|[0.002881707, 0.003205986]|[2.201682e-05, 3.134465e-05]|[2.629309e-05, 5.636196e-05]|18|

Unfavorable observations (all comparisons reported, without cost-matching inference):
- equal_objectives, level 0, seed 7500, risk/conventional: ΔCVaR -1.574346e-05; nominal F change 3.116781e-06; mean leakage change -1.009577e-07; max leakage change 5.617596e-06; wall ratio 1.21.
- equal_objectives, level 0, seed 7501, risk/conventional: ΔCVaR 4.962648e-05; nominal F change 1.364933e-05; mean leakage change -5.146052e-07; max leakage change 9.590434e-07; wall ratio 1.233.
- equal_objectives, level 0, seed 7502, risk/conventional: ΔCVaR 1.665107e-05; nominal F change 4.474499e-06; mean leakage change 5.859236e-07; max leakage change 3.572993e-06; wall ratio 1.235.
- equal_objectives, level 1, seed 7500, risk/conventional: ΔCVaR 0.0002949912; nominal F change 6.661671e-06; mean leakage change 5.250756e-05; max leakage change 0.0002258442; wall ratio 1.228.
- equal_objectives, level 1, seed 7500, continuation/conventional: ΔCVaR 0.0003467762; nominal F change 1.909388e-05; mean leakage change 7.239418e-06; max leakage change 2.804839e-05; wall ratio 1.267.
- equal_objectives, level 1, seed 7501, continuation/conventional: ΔCVaR 0.0003618733; nominal F change 2.025726e-05; mean leakage change 8.010887e-06; max leakage change 2.622615e-05; wall ratio 1.262.
- equal_objectives, level 1, seed 7501, continuation/risk: ΔCVaR 0.0002980214; nominal F change -9.675105e-06; mean leakage change 7.515012e-06; max leakage change 2.626642e-05; wall ratio 1.026.
- equal_objectives, level 1, seed 7502, risk/conventional: ΔCVaR 0.0002627862; nominal F change 5.030438e-06; mean leakage change 6.552139e-05; max leakage change 0.0003279236; wall ratio 1.235.
- equal_objectives, level 1, seed 7502, continuation/conventional: ΔCVaR 0.000386997; nominal F change 2.616622e-05; mean leakage change 8.821209e-06; max leakage change 2.73469e-05; wall ratio 1.256.
- equal_passes, level 0, seed 7500, risk/conventional: ΔCVaR 1.566067e-06; nominal F change 1.826882e-05; mean leakage change 7.112266e-07; max leakage change 1.07809e-05; wall ratio 0.6096.
- equal_passes, level 0, seed 7500, continuation/conventional: ΔCVaR 5.371645e-05; nominal F change 2.200766e-05; mean leakage change -5.628124e-07; max leakage change 3.095826e-06; wall ratio 0.6275.
- equal_passes, level 0, seed 7501, risk/conventional: ΔCVaR 0.0001006077; nominal F change 3.287226e-05; mean leakage change 2.541852e-06; max leakage change 1.261695e-05; wall ratio 0.6133.
- equal_passes, level 0, seed 7501, continuation/conventional: ΔCVaR 0.0001208113; nominal F change 3.609387e-05; mean leakage change 5.333389e-07; max leakage change 4.840191e-06; wall ratio 0.6391.
- equal_passes, level 0, seed 7502, risk/conventional: ΔCVaR 7.657047e-05; nominal F change 2.898187e-05; mean leakage change 1.752357e-06; max leakage change 9.988978e-06; wall ratio 0.6059.
- equal_passes, level 0, seed 7502, continuation/conventional: ΔCVaR 0.0001054369; nominal F change 3.176105e-05; mean leakage change 4.445222e-07; max leakage change 4.048056e-06; wall ratio 0.6303.
- equal_passes, level 1, seed 7500, risk/conventional: ΔCVaR -0.0003075438; nominal F change 1.568342e-05; mean leakage change -3.602556e-06; max leakage change -8.05618e-06; wall ratio 0.6072.
- equal_passes, level 1, seed 7500, continuation/conventional: ΔCVaR -2.917502e-05; nominal F change 1.735332e-05; mean leakage change 4.707257e-06; max leakage change 1.962631e-05; wall ratio 0.6263.
- equal_passes, level 1, seed 7500, continuation/risk: ΔCVaR 0.0002783688; nominal F change 1.669905e-06; mean leakage change 8.309813e-06; max leakage change 2.768249e-05; wall ratio 1.031.
- equal_passes, level 1, seed 7501, risk/conventional: ΔCVaR -0.0003087646; nominal F change 2.349483e-05; mean leakage change -4.178793e-06; max leakage change -3.812066e-06; wall ratio 0.6033.
- equal_passes, level 1, seed 7501, continuation/conventional: ΔCVaR -2.810356e-05; nominal F change 2.225774e-05; mean leakage change 4.967073e-06; max leakage change 2.303983e-05; wall ratio 0.6373.
- equal_passes, level 1, seed 7501, continuation/risk: ΔCVaR 0.000280661; nominal F change -1.237087e-06; mean leakage change 9.145867e-06; max leakage change 2.685189e-05; wall ratio 1.056.
- equal_passes, level 1, seed 7502, risk/conventional: ΔCVaR -0.0003295273; nominal F change 2.629822e-06; mean leakage change 2.740869e-06; max leakage change 2.496221e-05; wall ratio 0.6061.
- equal_passes, level 1, seed 7502, continuation/conventional: ΔCVaR -3.894633e-05; nominal F change 1.452186e-05; mean leakage change 4.480585e-06; max leakage change 1.78564e-05; wall ratio 0.6191.
- equal_wall, level 0, seed 7501, risk/conventional: ΔCVaR -3.737752e-06; nominal F change -9.458698e-07; mean leakage change 1.035515e-06; max leakage change 3.38864e-06; wall ratio 0.9669.
- equal_wall, level 0, seed 7501, continuation/conventional: ΔCVaR 1.049712e-05; nominal F change 2.020631e-06; mean leakage change 2.188794e-07; max leakage change 1.135899e-06; wall ratio 0.9806.
- equal_wall, level 0, seed 7502, risk/conventional: ΔCVaR -6.448655e-06; nominal F change 1.908657e-06; mean leakage change 2.45251e-07; max leakage change 2.919899e-06; wall ratio 1.011.
- equal_wall, level 0, seed 7502, continuation/conventional: ΔCVaR 1.612097e-05; nominal F change 2.723591e-06; mean leakage change 1.064989e-06; max leakage change 5.02578e-06; wall ratio 1.016.
- equal_wall, level 0, seed 7502, continuation/risk: ΔCVaR 2.256963e-05; nominal F change 8.149331e-07; mean leakage change 8.197382e-07; max leakage change 2.105881e-06; wall ratio 1.004.
- equal_wall, level 1, seed 7500, risk/conventional: ΔCVaR 3.714607e-05; nominal F change 2.014093e-05; mean leakage change -2.41153e-07; max leakage change 1.005833e-06; wall ratio 0.9897.
- equal_wall, level 1, seed 7501, risk/conventional: ΔCVaR 7.075111e-05; nominal F change 2.676051e-05; mean leakage change 5.36268e-07; max leakage change 4.443479e-08; wall ratio 1.002.
- equal_wall, level 1, seed 7502, risk/conventional: ΔCVaR 8.603451e-05; nominal F change 3.150844e-05; mean leakage change 7.872102e-07; max leakage change 3.364119e-06; wall ratio 0.9882.
- equal_wall, level 1, seed 7502, continuation/risk: ΔCVaR 7.551334e-05; nominal F change -4.546218e-07; mean leakage change -6.641334e-08; max leakage change -4.432369e-06; wall ratio 1.007.
