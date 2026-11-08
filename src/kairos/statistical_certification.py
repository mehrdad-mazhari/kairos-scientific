"""Distribution-free tail certification and explicit engineering verdicts."""
import numpy as np
from scipy.stats import binom

QUANTILE=.05
CONFIDENCE=.95


def tail_bound_details(fidelities, quantile=QUANTILE, confidence=CONFIDENCE):
    values=np.sort(np.asarray(fidelities,dtype=float))
    if (values.ndim!=1 or not len(values) or not np.all(np.isfinite(values))
            or np.any(values < -2e-6) or np.any(values > 1+2e-6)
            or not 0<quantile<1 or not 0<confidence<1):
        raise ValueError('invalid fidelity sample, quantile or confidence')
    k=int(binom.ppf(1-confidence,len(values),quantile))
    return dict(lower_bound=float(values[k-1]) if k>=1 else 0., order_statistic_index=k,
                supported=bool(k>=1), status='established' if k>=1 else 'insufficient_samples',
                n_samples=len(values), quantile=quantile, confidence_level=confidence,
                method='one-sided binomial order statistic; distribution-free',
                interpretation='A zero lower bound is not an observed zero fidelity.')


def certify(statistics, quality):
    supported=statistics.get('tail_bound_supported', statistics['n_samples']>=59)
    tail_pass=bool(supported and statistics['p05_fidelity_lcb95']>=quality['p05_validation_lcb_min'])
    leakage_pass=bool(statistics['worst_sample_leakage']<=quality['leakage_max'])
    reasons=[]
    if not supported:
        reasons.append('Insufficient independent validation samples for a nontrivial 95% P05 lower bound')
    elif not tail_pass:
        reasons.append('95% lower bound for true P05 fidelity is below the configured threshold')
    if not leakage_pass:
        reasons.append('Maximum observed independent-validation leakage exceeds the configured limit')
    return dict(status='INSUFFICIENT_EVIDENCE' if not supported else 'PASS' if tail_pass and leakage_pass else 'FAIL',passed=bool(tail_pass and leakage_pass), tail_passed=tail_pass,
                sampled_leakage_passed=leakage_pass, n_samples=statistics['n_samples'],
                confidence_level=CONFIDENCE, quantile=QUANTILE,
                tail_bound_supported=bool(supported),
                lower_bound=statistics['p05_fidelity_lcb95'],
                threshold=quality['p05_validation_lcb_min'], reasons=reasons,
                scope='True P05 lower bound for the configured iid uniform device distribution; maximum leakage is a sampled observation, not a global bound.')


def engineering_assessment(*, selection_feasible, nominal, hardware, duration_ns,
                           config, certification, public, numerical_passed, truncation,
                           worst_case, search_costs):
    q=config['quality_targets']
    checks={
        'selection':bool(selection_feasible),
        'nominal':bool(nominal['fidelity']>=q['nominal_fidelity_min'] and nominal['leakage']<=q['leakage_max']),
        'hardware':bool(hardware['peak_amplitude_mhz']*max(config['amplitude_headroom_scale'],1+config['manufacturing_ranges']['amplitude_fraction']) <= config['device']['amplitude_limit_mhz']+1e-10 and hardware.get('feasible',True)),
        'duration':bool(np.isclose(duration_ns,config['gate_duration_ns'],rtol=0,atol=1e-10)),
        'statistical_certification':bool(certification['passed']),
        'public_grid':bool(public['worst_leakage']<=q['leakage_max']),
        'numerical_convergence':bool(numerical_passed),
        'hilbert_space_truncation':bool(truncation['passed']),
        'worst_case_diagnostic':not worst_case.get('enabled',False) or (not worst_case.get('violations',[]) and 'error' not in worst_case)}
    labels={'selection':'OPTIMIZATION_NO_ELIGIBLE_CANDIDATE','nominal':'NOMINAL_QUALITY_FAILURE',
            'hardware':'HARDWARE_CONSTRAINT_FAILURE','duration':'DURATION_CONSTRAINT_FAILURE',
            'statistical_certification':'STATISTICAL_CERTIFICATION_FAILURE','public_grid':'PUBLIC_GRID_LEAKAGE_FAILURE',
            'numerical_convergence':'NUMERICAL_CONVERGENCE_FAILURE','hilbert_space_truncation':'HILBERT_SPACE_TRUNCATION_FAILURE',
            'worst_case_diagnostic':'OBSERVED_ROBUSTNESS_VIOLATION'}
    if 'error' in worst_case:
        labels['worst_case_diagnostic']='ROBUSTNESS_DIAGNOSTIC_FAILURE'
    failures=[dict(code=labels[k],check=k) for k,v in checks.items() if not v]
    nm=[r for r in search_costs if r['method']=='Nelder–Mead']
    optimizer_status=('not_run' if not nm else 'converged' if any(r['success'] for r in nm)
                      else 'budget_or_termination_limit_best_candidate_retained')
    return dict(passed=bool(all(checks.values())), checks=checks, failures=failures,
                optimizer_status=optimizer_status,
                nominal_quality_passed=bool(checks['nominal'] and checks['hardware'] and checks['duration']),
                converged_restarts=sum(r['success'] for r in nm),
                nominal_acceptance_passed=bool(checks['selection'] and checks['nominal'] and checks['hardware'] and checks['duration']),
                interpretation='Local optimizer convergence is not global optimality; budget exhaustion alone does not invalidate a qualified frozen pulse. Diagnostic searches can find violations but cannot certify a global worst case.')
