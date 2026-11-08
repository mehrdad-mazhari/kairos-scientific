# Independent confirmatory validation

**PARTIALLY CONFIRMED** under the operational interpretation reviewed before generation.

2000 independent simulated devices;10000 candidate-device production evaluations. Four primary eligible frozen pulses and one explicitly diagnostic pulse, all on identical device indices. No optimization or candidate changes.

|Candidate|Mean F|P05|CVaR95|Mean leakage|Max leakage|Engineering|
|---|---:|---:|---:|---:|---:|---|
|conventional|0.998653219|0.997398629|0.0030237417|2.6950612e-05|3.8533658e-05|PASS|
|risk|0.998498366|0.997058879|0.00338305312|2.234638e-05|3.1404057e-05|PASS|
|continuation|0.998626354|0.997388012|0.00303112568|2.9849566e-05|6.195734e-05|PASS|
|calibrated_drag|0.998480563|0.997004501|0.00340873105|4.6423332e-05|8.9568706e-05|PASS|
|risk_unfiltered_diagnostic|0.998589824|0.99730966|0.00305398711|7.2369407e-05|0.00025673855|DIAGNOSTIC_ONLY|

## All six prespecified contrasts

Positive ΔCVaR favors the candidate. Simultaneous Bonferroni99.1666667% intervals; ordinary95% intervals are separately exported.
- conventional vs calibrated_drag: Δ=0.000384989346, adjusted CI [0.0003227513409798, 0.00045000746087596016], candidate improvement; sensitivity agreement=True; resolution review=False; candidate engineering PASS=True.
- risk vs calibrated_drag: Δ=2.56779288e-05, adjusted CI [5.607020187499714e-06, 4.7162812934856225e-05], candidate improvement; sensitivity agreement=True; resolution review=False; candidate engineering PASS=True.
- continuation vs calibrated_drag: Δ=0.000377605373, adjusted CI [0.0003493476926575633, 0.00040715411355690353], candidate improvement; sensitivity agreement=True; resolution review=False; candidate engineering PASS=True.
- risk vs conventional: Δ=-0.000359311417, adjusted CI [-0.0004082295836575685, -0.00031232040435354727], candidate inferior; sensitivity agreement=True; resolution review=False; candidate engineering PASS=True.
- continuation vs conventional: Δ=-7.38397281e-06, adjusted CI [-4.893173177861862e-05, 3.363316641992915e-05], inconclusive; sensitivity agreement=True; resolution review=False; candidate engineering PASS=True.
- continuation vs risk: Δ=0.000351927445, adjusted CI [0.0003319240292942029, 0.0003716725267862103], candidate improvement; sensitivity agreement=True; resolution review=False; candidate engineering PASS=True.

## Scientific interpretation

Conventional versus DRAG, fixed risk versus conventional, continuation versus conventional and continuation versus fixed risk are answered by their respective prespecified contrast and certificate above. Point estimates alone do not establish superiority. Negative or inconclusive contrasts are retained.

The earlier approximately21% development improvement was a different budget/seed aggregate on reused development devices. This fixed-pulse confirmatory experiment cannot confirm that algorithm-level percentage. It estimates the selected pulses' differences; report their independent point percentages and simultaneous intervals without equating estimands.

The one-sided95% distribution-free P05 lower bound and additional simultaneous four-pulse certificate are exported separately from empirical P05 and CVaR. Sampled maximum leakage is not a global bound. The excluded diagnostic can never be promoted into primary engineering eligibility.

Statistical confidence intervals and their original50000-replicate directions remain unchanged. The100000-replicate sensitivity assessment is reported separately. Numerical repeats give finite sensitivity evidence, not a uniform CVaR error bound. Effects on the2e-6 resolution scale are labeled NUMERICALLY_UNRESOLVED; they cannot support a superiority claim even when the statistical CI excludes zero. Engineering certification and publication readiness are independent reported assessments.

Limitations: five-level Duffing/RWA approximation; finite development truncation checks; iid quasi-static uniform uncertainty assumptions; phenomenological Lindblad noise and extrapolated higher-level rates; finite2000-device tail sample; budget-limited optimization without convergence; resource-specific trade-offs and missing exact historical CPU qualification timings; no experimental hardware verification. Optimizer seeds are not independent validation samples.

A complete numerically interpretable simulation study permits manuscript preparation regardless of favorable or adverse outcomes. Q1 submission still needs a defensible novelty assessment, transparent reproducibility package, appropriately scoped claims and external/hardware evidence where those claims require it. No hardware demonstration or journal acceptance is implied.
