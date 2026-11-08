"""Resumable predeclared research campaign; held-out data never tune hyperparameters."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .challenge_model import Device,Scenario,make_pulse,PulseParameters,monte_carlo_scenarios,hardware_metrics,evaluate_waveform
from .robust_optimization import (TrainingEvaluator,validate_config,write_json,write_csv,summarize_trials,
                                 select_robust_candidate,candidate_pulse)
from .pipeline_settings import resolve_config
from .pulse_methods import reference_pulses,analytic_spec,fourier_spec
from .reference_calibration import calibrate_reference
from .grape import refine as conventional_refine
from .risk_aware_grape import refine as risk_refine
from .nelder_mead import minimize as nm_minimize
from .statistical_certification import certify,engineering_assessment
from .robustness_diagnostics import diagnose
from .study_validation import truncation_check
from .research_statistics import paired_bootstrap


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_profile(path,quick=False):
    c=json.loads(Path(path).read_text())
    if quick:
        if 'quick' not in c:raise ValueError('explicit quick profile required')
        c={**c,**c['quick']};c['mode']='quick'
    else:c['mode']='full'
    c.pop('quick',None)
    if c['mode']=='full':
        if (c['selection_samples']<200 or c['validation_samples']<2000 or c['nm_calls']<500 or c['nm_seeds']!=[7,17,27,37,47]
            or c['training_samples']<100 or len(c['optimizer_seeds'])<10 or c['worst_calls']<500):
            raise ValueError('full research profile cannot inherit reduced smoke budgets')
    if len(set(c['optimizer_seeds']))!=len(c['optimizer_seeds']):raise ValueError('optimizer seeds must be distinct')
    all_seeds=[c[k] for k in ('training_seed','selection_seed','validation_seed','bootstrap_seed')]+c['optimizer_seeds']+c['nm_seeds']
    if len(set(all_seeds))!=len(all_seeds):raise ValueError('data and optimizer seeds must be distinct')
    if c['objective_calls']<3 or c['scenario_passes']<6*(c['training_samples']+10):raise ValueError('budget too small for initial feasible objective evaluations')
    return c


def effective_grape_options(base,profile,seed,regime,weights=None):
    """Resolve all research limits explicitly; do not inherit caps from another run."""
    options={k:v for k,v in base['grape'].items() if k not in
             ('max_objective_evaluations','max_wall_seconds','max_surrogate_passes')}
    options.update(seed=seed,alpha=.95,smoothing=profile['smoothing'],risk_weight=0.,
        leakage_weight=profile['mu'],mean_weight=1.,worst_weight=0.,
        initialization_scale=profile['initialization_scale'],iterations=profile['iterations'],
        harmonics=profile['harmonics'],substeps=profile['substeps'],restarts=1,
        max_surrogate_passes=profile['scenario_passes'])
    options.update(restarts=profile.get('restarts',1),
        initialization_strategy=profile.get('initialization_strategy',options.get('initialization_strategy','least_squares_radial')),
        quality_targets=base.get('quality_targets',dict(nominal_fidelity_min=.999,leakage_max=.0001)))
    if options['initialization_strategy']=='qualified_fourier_multistart':
        options['qualified_perturbation_scale']=profile.get('qualified_perturbation_scale',.002)
        options['max_total_iterations']=profile['iterations']
    if 'acceleration' in profile:options['acceleration']=profile['acceleration']
    if profile.get('mode')=='full' and options.get('acceleration',{}).get('backend')=='jax_gpu':
        options['require_verified_gpu']=True
    if weights:options.update(weights)
    if regime=='equal_evaluations':options['max_objective_evaluations']=profile['objective_calls']
    elif regime=='equal_wall':options['max_wall_seconds']=profile['wall_seconds']
    else:raise ValueError('unknown research budget regime')
    return options


def checkpoint(path,compute):
    if path.exists():return json.loads(path.read_text())
    result=compute();temp=path.with_suffix('.tmp');write_json(temp,result);temp.replace(path);return result


def trial_score(record):
    # Eligibility is checked with the original production hard constraints.
    selection=record['mc_selection']
    return (not selection['selection_feasible'], -selection['score'])


def run(profile,output,prepare_only=False,max_new_runs=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    base=resolve_config(json.loads(Path(profile['base_config']).read_text()))
    base.update(mc_selection_samples=profile['selection_samples'],monte_carlo_samples=profile['validation_samples'],
        mc_selection_seed=profile['selection_seed'],mc_seed=profile['validation_seed'],search_seeds=profile['nm_seeds'],mc_selection_criterion='cvar95_infidelity')
    base['risk_aware_grape'].update(training_samples=profile['training_samples'],training_seed=profile['training_seed'],optimizer_seeds=profile['optimizer_seeds'])
    base['nelder_mead']['max_evaluations']=profile['nm_calls'];base['worst_case']['max_evaluations']=profile['worst_calls']
    base['calibration']['max_evaluations']=profile['calibration_calls']
    base['grape'].update(iterations=profile['iterations'],harmonics=profile['harmonics'],substeps=profile['substeps'],restarts=1)
    validate_config(base)
    protocol=dict(profile=profile,effective_config=base,primary='Risk minus conventional mean per-trial CVaR95 infidelity; negative is better',
        secondary='Mean per-trial P05 fidelity difference',alpha=.05,family_size=len(profile['regimes']),
        selection='Choose lambda then mu by mean common-selection CVaR across paired optimizer seeds, with hard eligibility first',
        validation='Generate only after all hyperparameters/waves frozen; one predeclared seed; no winner reselection',
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')})
    signature=digest(protocol)
    if (out/'protocol.json').exists():
        if digest(json.loads((out/'protocol.json').read_text()))!=signature:raise ValueError('protocol/source/config changed; choose a new output directory')
    else:write_json(out/'protocol.json',protocol)
    print('Effective research settings:',json.dumps(profile),flush=True)
    if prepare_only:return dict(status='PREPARED_NOT_EXECUTED',protocol_sha256=signature)
    device=Device(**base['device']);evaluator=TrainingEvaluator(base)
    train=evaluator.scenarios
    preflight_options=effective_grape_options(base,profile,profile['optimizer_seeds'][0],profile['regimes'][0])
    if profile['mode']=='full' and preflight_options.get('acceleration',{}).get('backend')=='jax_gpu':
        from .jax_backend import gpu_preflight
        probe=make_pulse(PulseParameters(base['gate_duration_ns'],.3,.5,1.,.3),device)
        write_json(out/'gpu_preflight.json',gpu_preflight(probe,device,train,preflight_options))
    selection=monte_carlo_scenarios(profile['selection_samples'],profile['selection_seed'],base['coherence_min'],**base['manufacturing_ranges'])
    write_json(out/'training_devices.json',[asdict(s) for s in train]);write_json(out/'selection_devices.json',[asdict(s) for s in selection])
    def setup():
        refs=reference_pulses(base,device)
        raw=refs['drag_baseline'];cal=calibrate_reference(raw,'drag_baseline',base,device)
        return dict(references=refs,calibrated_drag=cal)
    setup_data=checkpoint(out/'references.json',setup);cal=setup_data['calibrated_drag'];initial=candidate_pulse(cal,device)
    start=np.array([cal['parameters'][k] for k in ('sigma_fraction','beta','amplitude','phase_ramp_mhz')])
    class WallBudget(Exception):pass
    new_runs=0;costs=[];sensitivity=[];frozen={};chosen_hyperparameters={}
    def produce(name,regime,seed,weights=None):
        nonlocal new_runs
        path=out/f'{regime}_{name}_{seed}.json'
        if not path.exists() and max_new_runs is not None and new_runs>=max_new_runs:raise StopIteration
        def compute():
            nonlocal new_runs
            print(f'Optimizing {regime} / {name} / seed {seed}',flush=True)
            new_runs+=1;started=perf_counter();before=evaluator.simulations
            if name=='nm':
                nm_evaluator=TrainingEvaluator(base)
                nm_started=perf_counter()
                def nm_objective(x):
                    if regime=='equal_wall' and nm_evaluator.cache and perf_counter()-nm_started>=profile['wall_seconds']:raise WallBudget
                    return nm_evaluator.scalar(x)
                try:
                    result=nm_minimize(nm_objective,np.asarray(base['bounds']),np.clip(start,np.array(base['bounds'])[:,0],np.array(base['bounds'])[:,1]),seed=seed,max_evaluations=profile['nm_calls'],randomize_start=seed!=profile['nm_seeds'][0])
                except WallBudget:
                    from types import SimpleNamespace
                    result=SimpleNamespace(history=[],nfev=nm_evaluator.objective_calls,success=False,message='wall budget reached between evaluations')
                record=min(nm_evaluator.cache.values(),key=lambda r:r['selection_loss']);wave=candidate_pulse(record,device)
                evaluator.simulations+=nm_evaluator.simulations
                history=result.history;cost=dict(objective_evaluations=result.nfev,gradient_evaluations=0,surrogate_scenario_passes=0,success=bool(result.success),message=str(result.message))
            elif name=='drag':
                wave=initial;history=[];cost=dict(objective_evaluations=0,gradient_evaluations=0,surrogate_scenario_passes=0,success=cal['calibration']['optimizer_success'])
            else:
                options=effective_grape_options(base,profile,seed,regime,weights)
                wave,history=(conventional_refine if name=='conventional' else risk_refine)(initial,device,train,options,max(base['amplitude_headroom_scale'],1+base['manufacturing_ranges']['amplitude_fraction']))
                cost={**history[-1],'success':any(r.get('optimizer_success',False) for r in history), 'message':next((r['message'] for r in reversed(history) if 'message' in r),'No terminal optimizer message')}
            parameters=record['parameters'] if name=='nm' else cal['parameters']
            record=evaluator.evaluate_samples(wave,parameters)
            record.update(method=name,waveform=np.asarray(wave).tolist(),pulse_spec=cal['pulse_spec'] if name=='drag' else fourier_spec(wave,history,cal['pulse_spec'],device) if name!='nm' else analytic_spec(PulseParameters(**record['parameters']),device,'Nelder–Mead'))
            folder=out/f'{regime}_{name}_{seed}_selection';folder.mkdir(exist_ok=True)
            selected=select_robust_candidate([record],{**base,'mc_selection_candidates':1},device,folder,diagnostic_fallback=True)
            return dict(candidate=selected,history=history,cost={**cost,'elapsed_total_seconds':perf_counter()-started,'production_training_solves':evaluator.simulations-before},weights=weights,seed=seed,name=name,regime=regime)
        result=checkpoint(path,compute)
        costs.append(dict(regime=regime,method=name,seed=seed,objective_evaluations=result['cost'].get('objective_evaluations',0),gradient_evaluations=result['cost'].get('gradient_evaluations',0),surrogate_scenario_passes=result['cost'].get('surrogate_scenario_passes',0),production_training_solves=result['cost']['production_training_solves'],elapsed_seconds=result['cost']['elapsed_total_seconds'],converged_or_completed=result['cost'].get('success',False),termination=result['cost'].get('message','one-off nominal calibration reused')))
        return result
    try:
        for regime in profile['regimes']:
            frozen[f'{regime}_drag']=[produce('drag',regime,profile['optimizer_seeds'][0])]
            frozen[f'{regime}_nm']=[produce('nm',regime,s) for s in profile['nm_seeds']]
            frozen[f'{regime}_conventional']=[produce('conventional',regime,s) for s in profile['optimizer_seeds']]
            groups=[]
            for lam in profile['lambdas']:
                trials=[produce('lambda_'+str(lam),regime,s,dict(risk_weight=lam)) for s in profile['optimizer_seeds']]
                groups.append((lam,trials))
                sensitivity.append(dict(regime=regime,lambda_value=lam,mu=profile['mu'],phase='selection_only',p05=np.mean([t['candidate']['mc_selection']['statistics']['p05_fidelity'] for t in trials]),cvar95=np.mean([t['candidate']['mc_selection']['statistics']['cvar95_infidelity'] for t in trials])))
            def group_score(group):return (sum(trial_score(t['candidate'])[0] for t in group[1]),np.mean([trial_score(t['candidate'])[1] for t in group[1]]))
            lam,trials=min(groups,key=group_score)
            mu_groups=[]
            for mu in profile['mus']:
                ts=[produce('mu_'+str(mu),regime,s,dict(risk_weight=lam,leakage_weight=mu)) for s in profile['optimizer_seeds']];mu_groups.append((mu,ts))
            for mu_value,mu_trials in mu_groups:
                sensitivity.append(dict(regime=regime,lambda_value=lam,mu=mu_value,phase='mu_selection_only',p05=np.mean([t['candidate']['mc_selection']['statistics']['p05_fidelity'] for t in mu_trials]),cvar95=np.mean([t['candidate']['mc_selection']['statistics']['cvar95_infidelity'] for t in mu_trials])))
            mu,trials=min(mu_groups,key=group_score)
            chosen_hyperparameters[regime]=dict(lambda_value=lam,mu=mu)
            frozen[f'{regime}_risk']=trials
            for label,weights in [('A_mean',dict(risk_weight=0.,leakage_weight=0.)),('B_worst',dict(mean_weight=0.,risk_weight=0.,worst_weight=1.,leakage_weight=0.)),('C_cvar',dict(mean_weight=0.,risk_weight=1.,leakage_weight=0.)),('D_mean_cvar',dict(risk_weight=1.,leakage_weight=0.)),('E_mean_cvar_leak',dict(risk_weight=1.,leakage_weight=profile['mu']))]:
                frozen[f'{regime}_{label}']=[produce(label,regime,s,weights) for s in profile['optimizer_seeds']]
            # Original references remain comparisons at their actual durations.
            for name,record in setup_data['references'].items():
                def qualify_reference(record=record):
                    wave=candidate_pulse(record,device)
                    metric=evaluator.evaluate_samples(wave,record['parameters'])
                    observations=[evaluate_waveform(*wave,device,s) for s in selection]
                    stats=summarize_trials(observations,base['quality_targets'])
                    feasible=metric['violation']==0 and np.isclose(wave[0,-1],base['gate_duration_ns'],rtol=0,atol=1e-10) and stats['worst_sample_leakage']<=base['quality_targets']['leakage_max']
                    return dict(record,**{k:v for k,v in metric.items() if k not in record},mc_selection=dict(selection_feasible=bool(feasible),score=-stats['cvar95_infidelity'],statistics=stats))
                qualified=checkpoint(out/f'{regime}_reference_{name}_selection.json',qualify_reference)
                frozen[f'{regime}_reference_{name}']=[dict(candidate=qualified,seed=-1,name=name,regime=regime,history=[])]
    except StopIteration:
        write_json(out/'progress.json',dict(status='TRAINING_INCOMPLETE_VALIDATION_NOT_GENERATED',new_runs=new_runs,remaining='Resume the identical protocol to finish the training/selection grid'))
        return dict(status='TRAINING_INCOMPLETE')
    write_csv(out/'optimization_costs.csv',costs);write_csv(out/'cvar_sensitivity.csv',sensitivity)
    freeze=dict(protocol_sha256=signature,chosen_hyperparameters=chosen_hyperparameters,groups={key:[dict(path_key=t['name'],seed=t['seed'],waveform_sha256=hashlib.sha256(np.asarray(t['candidate']['waveform']).tobytes()).hexdigest()) for t in trials] for key,trials in frozen.items()})
    checkpoint(out/'frozen_manifest.json',lambda:freeze)
    if json.loads((out/'frozen_manifest.json').read_text())!=freeze:raise ValueError('frozen candidates changed')
    # No validation draw is generated or read anywhere above this barrier.
    validation=monte_carlo_scenarios(profile['validation_samples'],profile['validation_seed'],base['coherence_min'],**base['manufacturing_ranges'])
    write_json(out/'validation_devices.json',[asdict(s) for s in validation])
    results={};values={};maps={};mapdir=out/'robustness_maps';mapdir.mkdir(exist_ok=True)
    for key,trials in frozen.items():
        values[key]=[];results[key]=[]
        for index,trial in enumerate(trials):
            wave=np.asarray(trial['candidate']['waveform']);wavehash=hashlib.sha256(wave.tobytes()).hexdigest();wave.setflags(write=False)
            log=out/f'{key}_{index}_validation.jsonl'
            checkpoint(out/f'{key}_{index}_identity.json',lambda:dict(waveform_sha256=wavehash,protocol_sha256=signature))
            if json.loads((out/f'{key}_{index}_identity.json').read_text())!=dict(waveform_sha256=wavehash,protocol_sha256=signature):raise ValueError('validation checkpoint belongs to another waveform')
            rows=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            if len(rows)>len(validation):raise ValueError('invalid checkpoint length')
            with log.open('a') as stream:
                for i in range(len(rows),len(validation)):
                    row=checkpoint(out/f'evaluation_{wavehash}_{i}.json',lambda i=i:evaluate_waveform(*wave,device,validation[i]))
                    stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
            values[key].append([r['fidelity'] for r in rows]);stats=summarize_trials(rows,base['quality_targets'])
            def checks():
                nominal=evaluate_waveform(*wave,device,strict=True)
                production_nominal=evaluate_waveform(*wave,device)
                worst=diagnose(wave,device,base)
                worst_index=int(np.argmin([r['fidelity'] for r in rows]));leak_index=int(np.argmax([r['leakage'] for r in rows]))
                differences=[abs(nominal[k]-production_nominal[k]) for k in ('fidelity','leakage')]
                for i in {worst_index,leak_index}:
                    strict=evaluate_waveform(*wave,device,validation[i],strict=True)
                    differences.extend(abs(strict[k]-rows[i][k]) for k in ('fidelity','leakage'))
                for witness in (worst['minimum_observed'],worst['maximum_leakage_observed']):
                    strict=evaluate_waveform(*wave,device,Scenario(**witness['scenario']),strict=True)
                    differences.extend(abs(strict[k]-witness[k]) for k in ('fidelity','leakage'))
                trunc=truncation_check(wave,device,validation[worst_index],{**base.get('truncation',{}),'diagnostic_scenarios':[worst[k]['scenario'] for k in ('minimum_observed','maximum_leakage_observed')]})
                cert=certify(stats,base['quality_targets']);hardware=hardware_metrics(*wave,device)
                selection_feasible=trial['candidate'].get('mc_selection',{}).get('selection_feasible',False)
                assessment=engineering_assessment(selection_feasible=selection_feasible,nominal=nominal,hardware=hardware,duration_ns=float(wave[0,-1]),config=base,certification=cert,public=dict(worst_leakage=worst['maximum_leakage_observed']['leakage']),numerical_passed=max(differences)<=base['quality_targets']['solver_metric_tolerance'],truncation=trunc,worst_case=worst,search_costs=[])
                return dict(nominal=nominal,statistics=stats,certification=cert,worst_case=worst,truncation=trunc,engineering=assessment)
            checked=checkpoint(out/f'{key}_{index}_checks.json',checks);results[key].append(checked)
            if ('conventional' in key or key.endswith('_risk')) and index==0:
                def mapping():
                    data=[]
                    for amplitude in np.linspace(1-base['manufacturing_ranges']['amplitude_fraction'],1+base['manufacturing_ranges']['amplitude_fraction'],profile['map_points']):
                        for delta in np.linspace(-base['manufacturing_ranges']['detuning_mhz'],base['manufacturing_ranges']['detuning_mhz'],profile['map_points']):
                            metric=evaluate_waveform(*wave,device,Scenario(amplitude_scale=float(amplitude),detuning_mhz=float(delta)))
                            data.append(dict(amplitude_scale=float(amplitude),detuning_error_mhz=float(delta),fidelity=metric['fidelity'],leakage=metric['leakage']))
                    return data
                maps[key]=checkpoint(mapdir/f'{key}.json',mapping)
            assert hashlib.sha256(wave.tobytes()).hexdigest()==wavehash
    comparisons={}
    for regime in profile['regimes']:
        comparisons[regime]=paired_bootstrap(values[f'{regime}_risk'],values[f'{regime}_conventional'],repetitions=profile['bootstrap_repetitions'],seed=profile['bootstrap_seed'],family_size=len(profile['regimes']))
    write_csv(out/'paired_algorithm_comparison.csv',[dict(regime=k,**v) for k,v in comparisons.items()])
    result=dict(chosen_hyperparameters=chosen_hyperparameters,shared_calibration_cost=cal['calibration'],status='COMPLETE',mode=profile['mode'],effective_settings=profile,protocol_sha256=signature,results=results,comparisons=comparisons,
                validation_not_used_for_selection=True,trial_device_variability='Crossed paired bootstrap; optimizer trials and common devices resampled separately',remaining='Full campaign unexecuted' if profile['mode']=='quick' else 'External/hardware validation; confirmatory bootstrap approximation assumptions')
    write_json(out/'research_benchmark_results.json',result)
    from .research_benchmark_report import report
    report(out,result,sensitivity,costs,values,maps)
    write_json(out/'progress.json',dict(status='COMPLETE',new_runs=new_runs))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='research_config.json');p.add_argument('--output',required=True);p.add_argument('--quick',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--max-new-runs',type=int)
    a=p.parse_args();print(run(load_profile(a.config,a.quick),a.output,a.prepare_only,a.max_new_runs)['status'])
if __name__=='__main__':main()
