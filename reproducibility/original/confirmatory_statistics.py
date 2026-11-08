"""Data-independent frozen-pulse paired inference. No device generation API."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from risk_aware_grape import empirical_cvar
from statistical_certification import tail_bound_details

PRIMARY=('calibrated_drag','conventional','risk','continuation')
PAIRS=tuple(itertools.combinations(PRIMARY,2))

def paired_cvar_intervals(fidelities,*,repetitions=50000,seed=941207):
    arrays={k:np.asarray(fidelities[k],dtype=float) for k in PRIMARY}
    n=len(arrays[PRIMARY[0]])
    if n<2 or repetitions<100 or any(a.shape!=(n,) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Finite paired vectors of equal length and >=100 repetitions required')
    rng=np.random.default_rng(seed);bootstrap=np.empty((repetitions,len(PAIRS)))
    points={k:empirical_cvar(1-a) for k,a in arrays.items()}
    for b in range(repetitions):
        indices=rng.integers(n,size=n)
        cv={k:empirical_cvar(1-a[indices]) for k,a in arrays.items()}
        bootstrap[b]=[cv[a]-cv[c] for a,c in PAIRS]
    result=[];q=.05/(2*len(PAIRS))
    for i,(a,c) in enumerate(PAIRS):
        ci=np.quantile(bootstrap[:,i],[q,1-q]);plain=np.quantile(bootstrap[:,i],[.025,.975])
        result.append(dict(reference=a,candidate=c,delta_cvar=points[a]-points[c],
            ci95=plain.tolist(),family_adjusted_ci=ci.tolist(),family_size=len(PAIRS),
            adjusted_confidence=1-.05/len(PAIRS),nominal_alpha=.05,
            direction='candidate improvement' if ci[0]>0 else 'candidate inferior' if ci[1]<0 else 'inconclusive',
            engineering_pass_not_implied=True,scope='Approximate paired device-bootstrap inference for fixed pulses, conditional on development selection'))
    return dict(n_devices=n,tail_mass=(1-.95)*n,bootstrap_repetitions=repetitions,bootstrap_seed=seed,comparisons=result)

def freeze_protocol(out,check_protocol,candidates,decision):
    out=Path(out);source=Path(__file__)
    parameters=dict(validation_seed=941206,validation_count=2000,bootstrap_seed=941207,
        bootstrap_repetitions=50000,sensitivity_repetitions=100000,alpha=.05,primary_pairs=[list(p) for p in PAIRS],
        physical_model=check_protocol['base_config'],candidate_hashes={k:c['waveform_sha256'] for k,c in candidates.items()},
        hilbert_dimension=5,source_hashes=check_protocol['source_hashes'],statistics_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        production_solver=dict(backend='CPU QuTiP propagator',atol=1e-12,rtol=1e-10,max_step_ns=.05,nsteps=100000,normalize_output=False),
        decision=decision,validation_generated=False,validation_execution_authorized=False)
    (out/'confirmatory_protocol.json').write_text(json.dumps(parameters,indent=2)+'\n')
    text=f'''# Frozen confirmatory protocol — version 1 — NOT EXECUTED

Preconfirmatory development decision: **{decision}**. Execution of independent
validation is not authorized by this task. No validation devices have been generated.
The attached request ended at “PHASE 7 — Freeze Protocol and Source / Create”; this
version supplies explicit candidate, source and protocol manifests for that phase.

## Candidate and source lock

`frozen_candidates.json` contains exact waveform arrays (authoritative piecewise
linear I/Q), SHA256, Fourier coefficients or native DRAG calibration parameters,
source trial, optimizer seed, options, training configuration, costs and every
engineering/numerical decision. Include DRAG, conventional, eligible fixed risk,
and continuation. The excluded unfiltered best fixed-risk pulse remains an
explicit diagnostic comparator. Do not replace a waveform by a fit or reoptimize it.
`freeze_manifest.json` detects every change to artifacts and copied sources.
Check all manifest hashes before any future validation run; abort on mismatch.
Run independent validation only after explicit authorization and review of any
NO-GO or CONDITIONAL GO conditions. A NO-GO dimension check does not authorize
silently replacing N5 by N8; a revised physical protocol would need a new version.

## Independent sample — preparation only

Once authorized, generate exactly 2000 iid scenarios once with fresh, reserved seed
941206 through the frozen `challenge_model.monte_carlo_scenarios`. No call to this
generator occurs in the present task. Use the same 2000 device indices for every
frozen waveform, including diagnostic comparators. Preserve amplitude factor
U[.97,1.03], detuning U[-3,3] MHz, anharmonicity shift U[-8,8] MHz and four independent
T1/Tphi scale factors U[.8,1]. Preserve all nominal parameters from
confirmatory_protocol.json: five levels, 15.6 ns, 0.2 ns sampled grid, 80 MHz limit
and the original Lindblad and six-state unconditional X observables. These
distributions and higher-level noise extrapolations are model assumptions.
The reserved validation seed differs from training7400, selection7401 and the
unused earlier reservation7402. No information from validation may change the
candidates, methods, uncertainty laws, hyperparameters, analysis or thresholds.

## CPU solver and numerical gates

Use trusted CPU QuTiP production propagation for all validation and certification,
with strict atol1e-12, rtol1e-10, max_step .05 ns, nsteps100000 and no state
renormalization. Record trace/Hermiticity errors, minimum eigenvalues and total
noncomputational population. Numerical tolerance remains 2e-6. Repeat nominal
and saved joint stress checks at maximum step .025 ns (dt/8), and with independent
six-state mesolve propagation. Existing GPU4/8 checks are supporting development
evidence; they are not substituted for CPU certification.
Frozen repeat-device indices are {np.linspace(0,1999,20,dtype=int).tolist()}.
For those indices, repeat strict/tighter CPU propagation and compare all fidelity
and leakage metrics. Any error >2e-6, trace/Hermiticity error >2e-6, minimum
eigenvalue <−2e-6 or constraint failure stops analysis with a numerical failure.
Report failed solves and costs. Do not omit devices, silently normalize states,
relax tolerances, or replace unfavorable data. Resolve numerical failures in a
documented new version before asserting an engineering or statistical PASS.

## Statistical endpoints and multiplicity

Primary endpoint: paired difference Δ=CVaR95(reference infidelity)−CVaR95(candidate
infidelity), positive favors candidate. Use the unchanged `empirical_cvar`: sort
losses descending and use fractional boundary weights. With 2000 devices, the
worst 5% has mass100, not 99 or 101 whole devices; floating-point boundary roundoff
does not change the definition. CVaR is on infidelity, not fidelity. Smooth training
CVaR is not used as the confirmatory endpoint.

The four primary representatives give SIX prespecified pairwise contrasts:
{[list(p) for p in PAIRS]}. This explicitly extends the earlier three-method
draft's family of three when adding calibrated DRAG. Preserve every contrast and
all adverse outcomes. An engineering-ineligible comparator is labeled diagnostic;
it still appears in the outcome table and cannot earn an engineering PASS.
Do not select a favorable family after observing validation outcomes.

Resample common device indices with replacement using bootstrap seed941207 and
50000 paired replicates. Recompute per-pulse empirical CVaR in each replicate.
Each replicate uses the same index vector for all pulses. Report ordinary 95%
percentile CIs and simultaneous Bonferroni 99.1666667% CIs for the six primary
contrasts (quantiles .05/12 and 1−.05/12). The bootstrap is an approximate
device-sampling interval, not exact distribution-free CVaR certification.
Declare directional improvement only if the simultaneous lower bound is >0;
inferiority if its upper bound is <0; otherwise inconclusive. Also require that
the sign is not reversed by numerical sensitivity of the fixed waveforms.
Repeat with the same resample stream extended to100000 replicates; disagreement
means inconclusive. No optional favorable bootstrap-method switching or stopping.
The numerical 2e-6 per-metric tolerance is not a proof of an exact CVaR error bound;
describe sensitivity and avoid resolved advantages on that scale without stronger
numerical evidence. Nominal ranking separated by ~5.37e-7 is not itself evidence
of superiority. No minimum favorable effect size is invented after outcomes.

Secondary endpoints: paired mean fidelity, P05 fidelity, nominal fidelity, mean
leakage, maximum observed leakage, exceedance counts and engineering PASS/FAIL.
Use the same resamples for descriptive 95% intervals on mean/P05/mean leakage;
do not treat those as additional confirmatory superiority tests. Report maximum
leakage as a sampled maximum, not a global bound; no bootstrap maximum CI claims
global safety. Three optimizer seeds and 54 development runs on the same devices
are descriptive optimizer variation, not independent device ensembles. Inference
here is conditional on frozen selected pulses, not a universal algorithm or
efficiency claim. Report optimizer costs separately; no validation-driven selection.

## Engineering acceptance, unchanged

Nominal fidelity >=.999, nominal leakage <=1e-4, maximum observed validation
leakage <=1e-4, zero endpoints (1e-10 numerical amplitude check), fixed15.6 ns,
and peak amplitude including1.03 uncertainty headroom <=80 MHz.
Preserve `statistical_certification.tail_bound_details` exactly: for sorted
fidelities, k=binom.ppf(.05,n,.05), one-sided95% lower bound F_(k), requiring
bound>=.995 and supported k>=1. Publish these existing per-pulse95% certificates.
Additionally, a simultaneous engineering claim across the four representatives
uses the SAME order-statistic construction with confidence1−.05/4, not a weaker
threshold. The diagnostic leaky fixed-risk waveform is not included in a primary
engineering success claim. Numerical adequacy and development Hilbert checks
must pass independently of nominal fidelity. A favorable CVaR result cannot
override any engineering failure; optimization convergence is not required for a
valid frozen budget-limited pulse.

## Reporting and remaining scientific scope

Archive raw CPU channels, exact input scenario rows, solver/package versions,
all seeds and hashes, failures, runtimes and every endpoint. Report negative and
mixed results. Five-to-eight-level agreement on the existing200 selection and
saved stress scenarios is finite-sample numerical evidence, not a uniform proof,
validation outcome, hardware test or proof of the Duffing/RWA/noise assumptions.
Q1 submission readiness still requires authorized confirmation, a defensible
novelty claim, a complete methods/reproducibility package and suitable external
or hardware replication for the claims made. This document does not promise
journal acceptance.
'''
    (out/'confirmatory_protocol_frozen.md').write_text(text)
