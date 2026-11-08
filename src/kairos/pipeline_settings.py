"""Backward-compatible settings shared by CLI, desktop and numerical workflow."""
from copy import deepcopy
import numpy as np

WORST_DEFAULTS = dict(enabled=True, max_evaluations=80, seeds=[2041,2042],
                      axis_points=5, xatol=1e-3, fatol=1e-8, fidelity_min=None)


def resolve_config(config):
    resolved = deepcopy(config)
    resolved.setdefault('mc_selection_samples', 200)
    resolved.setdefault('monte_carlo_samples', 2000)
    resolved.setdefault('search_seeds', [7,17,27,37,47])
    from .risk_aware_grape import settings as risk_settings
    resolved['risk_aware_grape']=risk_settings(resolved)
    from .nelder_mead import settings
    resolved['nelder_mead'] = settings(resolved)
    training = resolved.get('training', {})
    resolved['training'] = dict(manufacturing_ranges=deepcopy(training.get('manufacturing_ranges', resolved['manufacturing_ranges'])),
                                coherence_min=training.get('coherence_min', resolved['coherence_min']))
    resolved['worst_case'] = {**deepcopy(WORST_DEFAULTS), **resolved.get('worst_case', {})}
    return resolved


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int,np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def validate_reliability_settings(config):
    c=resolve_config(config)
    for name in ('mc_selection_samples','monte_carlo_samples'):
        integer(c[name],name,2)
    for name in ('mc_selection_seed','mc_seed'):
        integer(c[name],name)
    w=c['worst_case']
    integer(w['max_evaluations'],'worst_case.max_evaluations',8)
    integer(w['axis_points'],'worst_case.axis_points',2)
    if not isinstance(w['enabled'], bool):
        raise ValueError('worst_case.enabled must be boolean')
    if len(w['seeds']) != 2:
        raise ValueError('worst_case.seeds requires two distinct seeds (fidelity and leakage)')
    for seed in w['seeds']:
        integer(seed,'worst_case seed')
    if not all(np.isfinite(w[k]) and w[k]>0 for k in ('xatol','fatol')):
        raise ValueError('worst-case tolerances must be positive and finite')
    # Training stress cases are deterministic; optimizer and diagnostic random
    # streams must still not accidentally share held-out random streams.
    protected=[c['mc_selection_seed'],c['mc_seed'],c.get('grape',{}).get('seed',2030)]
    if len(set(protected)) != len(protected) or set(protected)&set(c['search_seeds']):
        raise ValueError('selection, validation, GRAPE and optimization seeds must be separate')
    if w['enabled'] and (len(set(w['seeds'])) != 2 or set(w['seeds']) & (set(protected)|set(c['search_seeds']))):
        raise ValueError('worst-case seeds must be distinct from all optimization/selection/validation seeds')
    if w.get('fidelity_min') is not None and (not np.isfinite(w['fidelity_min']) or not 0<w['fidelity_min']<=1):
        raise ValueError('worst_case.fidelity_min must be null or a probability in (0,1]')
    risk=c['risk_aware_grape']
    if risk['enabled']:
        integer(risk['training_samples'],'risk training samples',2)
        integer(risk['training_seed'],'risk training seed')
        if not 0<=risk['alpha']<1 or not all(np.isfinite(risk[k]) and risk[k]>=0 for k in ('risk_weight','leakage_weight')) or not np.isfinite(risk['smoothing']) or risk['smoothing']<=0:
            raise ValueError('invalid risk alpha, weights or smoothing')
        if not risk['optimizer_seeds'] or len(set(risk['optimizer_seeds']))!=len(risk['optimizer_seeds']):
            raise ValueError('distinct risk optimizer seeds required')
        reserved=set(c['search_seeds'])|set(w['seeds'])|{c['mc_seed'],c['mc_selection_seed'],risk['training_seed']}
        integer(risk['training_seed'],'risk training seed')
        if risk['training_seed'] in set(protected)|set(c['search_seeds'])|set(w['seeds']):
            raise ValueError('risk training seed must be independent')
        for seed in risk['optimizer_seeds']:
            integer(seed,'risk optimizer seed')
            if seed in reserved:raise ValueError('optimizer and data seeds must be separate')
    t=c['training']
    r=t['manufacturing_ranges']
    if (set(r) != {'amplitude_fraction','detuning_mhz','alpha_shift_mhz'}
            or not all(np.isfinite(v) and v>=0 for v in r.values())
            or r['amplitude_fraction']>=1 or not 0<t['coherence_min']<=1):
        raise ValueError('invalid training uncertainty ranges or coherence_min')


def apply_quick_mode(config):
    """One reduced-budget profile used by the worker for CLI and desktop runs."""
    config['grape']={**config.get('grape',{}), 'iterations':3,'samples':2,'restarts':1,
                     'amplitude_grid_points':3,'detuning_grid_points':3}
    config['calibration']={**config.get('calibration',{}),'max_evaluations':40}
    config['nelder_mead']={**config.get('nelder_mead',{}),'max_evaluations':60}
    config['random_search']={**config.get('random_search',{}),'max_evaluations':60}
    config['worst_case']={**config.get('worst_case',{}),'max_evaluations':16,'axis_points':3}
    config.update(search_seeds=[7],monte_carlo_samples=128,mc_selection_samples=32,mc_selection_candidates=3)
    config['risk_aware_grape']={**config.get('risk_aware_grape',{}),'iterations':3,'restarts':1,'training_samples':4,'optimizer_seeds':[2030]}
    config['run_mode']='quick_smoke_not_full_validation'
    return config
