"""Scientific diagnostics: objective redundancy, paired comparisons and truncation."""
from dataclasses import replace
import numpy as np
from .challenge_model import evaluate_waveform, Scenario


def objective_diagnostics(records):
    values = np.array([r['objectives'] for r in records if r.get('training') is not None])
    if len(values) < 2:
        return dict(n_candidates=len(values), correlation=None, normalized_singular_values=[],
                    interpretation='Too few feasible training candidates to assess objective redundancy.')
    scale = values.std(axis=0)
    active = scale > 1e-15
    standardized = (values-values.mean(axis=0))/np.where(active, scale, 1.)
    correlation = standardized.T@standardized/len(values)
    singular = np.linalg.svd(standardized, compute_uv=False)
    return dict(n_candidates=len(values), correlation=correlation.tolist(),
                constant_objectives=np.where(~active)[0].tolist(),
                normalized_singular_values=(singular/singular[0]).tolist() if singular[0] else [],
                interpretation='Correlation and effective dimension are diagnostics; a Pareto front alone does not establish an optimizer advantage.')


def paired_comparisons(mc_rows, baseline='drag_baseline_calibrated'):
    if baseline not in mc_rows:
        return {}
    reference = np.array([r['fidelity'] for r in mc_rows[baseline]])
    results = {}
    for key, rows in mc_rows.items():
        difference = np.array([r['fidelity'] for r in rows])-reference
        # Same device draws preserve pairing. This CI concerns mean differences,
        # not the 5th percentile or a claim about unobserved device distributions.
        from scipy.stats import t
        half = float(t.ppf(.975, len(rows)-1)*difference.std(ddof=1)/np.sqrt(len(rows)))
        mean = float(difference.mean())
        results[key] = dict(baseline=baseline, mean_fidelity_difference=mean,
                            paired_mean_ci95=[mean-half, mean+half])
    return results


def truncation_check(pulse, device, scenario, options):
    levels = sorted(set([3, 4, 5, 6, device.levels, device.levels+1]))
    cases={'nominal':Scenario(),'perturbed':scenario}
    for i,values in enumerate(options.get('diagnostic_scenarios',[])):
        cases[f'diagnostic_{i}']=Scenario(**values)
    rows = []
    for size in levels:
        model = replace(device, levels=size)
        row={name:evaluate_waveform(*pulse,model,value,strict=True) for name,value in cases.items()}
        row['coherent']=evaluate_waveform(*pulse,model,strict=True,coherent=True)
        rows.append(dict(levels=size,**row))
    production=next(r for r in rows if r['levels']==device.levels)
    higher=next(r for r in rows if r['levels']==device.levels+1)
    differences={f'{case}_{metric}':abs(production[case][metric]-higher[case][metric])
                 for case in list(cases)+['coherent'] for metric in ('fidelity','leakage')}
    passed=device.levels>=4 and max(differences.values())<=options.get('metric_tolerance',2e-6)
    return dict(rows=rows,production_levels=device.levels,differences=differences,passed=bool(passed),
                scope='Frozen selected pulse: nominal, worst-fidelity held-out device, coherent nominal and optional diagnostic extrema. Not a uniform convergence proof over the uncertainty domain.',
                higher_level_noise='For n>2: relaxation rate (n/2)/T1_2 and projector dephasing rate (n/2)^2/Tphi_2; explicit Duffing-model extrapolation.')


def nondominated_fronts(objectives, violations=None):
    """Diagnostic Pareto layers; scalar Nelder–Mead does not optimize a front."""
    values = np.asarray(objectives, dtype=float)
    cv = np.zeros(len(values)) if violations is None else np.asarray(violations)
    fronts = []
    for violation in np.unique(cv):
        remaining = np.flatnonzero(cv == violation)
        while len(remaining):
            current = values[remaining]
            dominated = np.array([np.any(np.all(current <= row, axis=1)
                                         & np.any(current < row, axis=1)) for row in current])
            fronts.append(remaining[~dominated])
            remaining = remaining[dominated]
    return fronts
