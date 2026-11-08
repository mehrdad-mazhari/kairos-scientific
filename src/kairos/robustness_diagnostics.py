"""Post-validation, fixed-waveform bounded searches; never a global certificate."""
from dataclasses import asdict
from time import perf_counter
from itertools import product
import numpy as np
from scipy.optimize import minimize
from .challenge_model import Scenario, evaluate_waveform

NAMES=('amplitude_scale','detuning_mhz','alpha_shift_mhz','t1_1_scale','t1_2_scale','tphi_1_scale','tphi_2_scale')


def diagnose(pulse, device, config, evaluate=None):
    options=config['worst_case']
    if not options['enabled']:
        return dict(enabled=False, violations=[], channel_simulations=0, elapsed_seconds=0.)
    evaluate=evaluate or evaluate_waveform
    started=perf_counter()
    frozen=np.array(pulse,dtype=float,copy=True)
    frozen.setflags(write=False)
    ranges=config['manufacturing_ranges'];c=config['coherence_min']
    low=np.array([1-ranges['amplitude_fraction'],-ranges['detuning_mhz'],-ranges['alpha_shift_mhz'],c,c,c,c])
    high=np.array([1+ranges['amplitude_fraction'],ranges['detuning_mhz'],ranges['alpha_shift_mhz'],1.,1.,1.,1.])
    active=high>low
    nominal=np.array([.5,.5,.5,1.,1.,1.,1.])
    cache={}
    def observe(u, tag):
        values=low+np.clip(u,0.,1.)*(high-low)
        key=tuple(values)
        if key not in cache:
            metric=evaluate(*frozen,device,Scenario(*values))
            cache[key]=dict(scenario=asdict(Scenario(*values)),fidelity=metric['fidelity'],leakage=metric['leakage'],tags=[])
        if tag not in cache[key]['tags']:
            cache[key]['tags'].append(tag)
        return cache[key]
    observe(nominal,'nominal')
    for j,name in enumerate(NAMES):
        for value in np.linspace(0.,1.,options['axis_points']):
            u=nominal.copy();u[j]=value
            observe(u,'axis_'+name)
    for value in np.linspace(0.,1.,options['axis_points']):
        u=nominal.copy();u[3:]=value
        observe(u,'axis_joint_coherence')
    # Joint boundaries probe combinations rather than single-axis sensitivities only.
    for corner in product([0.,1.],repeat=3):
        observe(np.r_[corner,np.zeros(4)],'joint_boundary')
    statuses=[]
    for seed,metric in zip(options['seeds'],('fidelity','leakage')):
        initial=min(cache.values(),key=lambda r:r['fidelity']) if metric=='fidelity' else max(cache.values(),key=lambda r:r['leakage'])
        physical=np.array([initial['scenario'][name] for name in NAMES])
        start=np.divide(physical-low,high-low,out=nominal.copy(),where=active)[active]
        def objective(v):
            u=nominal.copy();u[active]=v
            row=observe(u,'joint_search_'+metric)
            return row[metric] if metric=='fidelity' else -row[metric]
        if len(start):
            # Nondegenerate inward/outward simplex even at a domain boundary.
            rng=np.random.default_rng(seed)
            simplex=np.tile(start,(len(start)+1,1))
            for j,step in enumerate(rng.uniform(.05,.15,len(start))):
                simplex[j+1,j]+=step if start[j]<=.8 else -step
            result=minimize(objective,start,method='Nelder-Mead',bounds=[(0.,1.)]*len(start),
                options=dict(maxfev=options['max_evaluations'],xatol=options['xatol'],fatol=options['fatol'],initial_simplex=simplex))
            statuses.append(dict(metric=metric,seed=seed,success=bool(result.success),message=str(result.message),objective_calls=int(result.nfev)))
        else:
            objective(start)
            statuses.append(dict(metric=metric,seed=seed,success=True,message='All uncertainty ranges are fixed',objective_calls=1))
    rows=list(cache.values())
    minimum=min(rows,key=lambda r:r['fidelity']);maximum=max(rows,key=lambda r:r['leakage'])
    sensitivities={}
    for name in list(NAMES)+['joint_coherence']:
        axis=[r for r in rows if 'axis_'+name in r['tags']]
        sensitivities[name]=dict(min_observed_fidelity=min(r['fidelity'] for r in axis),
            max_observed_leakage=max(r['leakage'] for r in axis),
            fidelity_span=max(r['fidelity'] for r in axis)-min(r['fidelity'] for r in axis))
    violations=[]
    if maximum['leakage']>config['quality_targets']['leakage_max']:
        violations.append('Observed leakage exceeds the configured hard leakage limit')
    if options.get('fidelity_min') is not None and minimum['fidelity']<options['fidelity_min']:
        violations.append('Observed fidelity below the explicitly configured worst-case fidelity limit')
    return dict(enabled=True,minimum_observed=minimum,maximum_leakage_observed=maximum,
                sensitivities=sensitivities,observations=rows,search_status=statuses,
                channel_simulations=len(rows),elapsed_seconds=perf_counter()-started,violations=violations,
                mathematically_certified_global_bound=False,
                interpretation='Finite axis/boundary probes and bounded local searches on a frozen waveform. Failure to find a violation does not certify the domain. P05 is a quantile threshold, not a minimum-fidelity requirement.')
