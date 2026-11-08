# Candidate selection audit

All 54 frozen trials, source hashes, optimization checkpoints, exact waveforms,
selection aggregation and saved ensembles were verified. All original artifacts
are covered by a separate preservation manifest. No optimization was rerun.

Continuation was **not excluded**. Conventional's best eligible CVaR95 was
0.00288117038204; continuation's was 0.00288170740761.
The difference is 5.37025570035e-07, smaller than the existing 2e-6 numerical metric
tolerance. The literal minimum selects conventional, but this gap does not justify
a numerically resolved or statistically significant superiority claim. Group means
under different cost regimes do not determine the minimum individual feasible pulse.

Fixed risk's unfiltered best-CVaR waveform is excluded solely by maximum observed
selection leakage >1e-4. It remains a frozen diagnostic comparator. The eligible
fixed-risk representative is the existing draft's saved waveform; no replacement
search is performed. Nominal leakage qualification is separate from maximum
selection leakage, and mean leakage is not the maximum-leakage acceptance rule.

The frozen optimization runner used nominal qualification, waveform constraints,
and maximum selection leakage. The later analysis draft additionally applied a
development P05 lower-bound guard >=.995. That guard belongs to the post-analysis
selection draft, rather than the optimizer's original eligibility rule. Every one
of the 54 trials passes it (minimum observed bound >.9967), so it changes no
exclusions, representatives or ranking. No implementation correction is required;
this provenance distinction is documented instead of calling the extra guard an
original optimizer rule. No new exclusion criterion is introduced here.

The full candidate table records each exact exclusion. The calibrated DRAG
baseline has native analytic calibration parameters; exact sampled I/Q is authoritative.
It must not be replaced by an approximate Fourier fit during these checks.
