"""Checkpointed development comparisons. This entry point has no validation path."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from challenge_model import (Device,Scenario,training_scenarios,monte_carlo_scenarios,
    evaluate_waveform,hardware_metrics,MHZ_TO_RAD_NS)
from grape import DEFAULTS,FourierControls,EnsembleGRAPE
from risk_aware_grape import RiskEnsembleGRAPE
from grape_research import qualified_fourier_multistart,ResourceBudget,refine,waveform_hash,nominal_qualified
from continuation_risk_grape import refine as continuation,validate_schedule
from robust_optimization import summarize_trials
from research_benchmark import checkpoint,digest
from optimizer_diagnostics import validate_objective_algebra


METHODS=('conventional','risk','continuation')
REGIMES=('equal_objectives','equal_passes','equal_wall')


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def load_config(path,diagnostic=False):
    c=json.loads(Path(path).read_text());short=c.pop('diagnostic',{})
    if diagnostic:c.update(short)
    c['mode']='short_diagnostic' if diagnostic else 'intermediate'
    if c['training_samples']<100 or c['selection_samples']<200 or c['harmonics']!=5 or c['restarts']!=5:
        raise ValueError('development requires >=100 training / >=200 selection, five harmonics and five restart allocations')
    seeds=[c[k] for k in ('training_seed','selection_seed','reserved_validation_seed')]+c['optimizer_seeds']
    if len(seeds)!=len(set(seeds)):raise ValueError('training/selection/reserved validation/optimizer seeds must be distinct')
    if len(c['optimizer_seeds'])<2:raise ValueError('paired development comparison needs multiple optimizer seeds')
    if any(r not in REGIMES for r in c['regimes']):raise ValueError('unknown budget regime')
    validate_schedule(c['continuation_schedule'])
    if c['objective_calls']<4 or c['iterations']<4:raise ValueError('budget cannot cover continuation stages')
    return c


def load_reference_checkpoint(path):
    raw=Path(path).read_bytes()
    return json.loads(raw)['calibrated_drag'],hashlib.sha256(raw).hexdigest()


def amplitude_headroom(base):
    return max(base['amplitude_headroom_scale'],1+base['manufacturing_ranges']['amplitude_fraction'])


def validate_reference_waveform(reference,base):
    wave=np.asarray(reference['waveform'],dtype=float)
    if wave.ndim!=2 or wave.shape[0]!=3 or wave.shape[1]<3 or not np.isfinite(wave).all():
        raise ValueError('reference must contain finite time/I/Q samples')
    if not np.isclose(wave[0,0],0.,rtol=0,atol=1e-10) or not np.isclose(wave[0,-1],base['gate_duration_ns'],rtol=0,atol=1e-10):
        raise ValueError('reference duration differs from the fixed configured gate')
    if not np.allclose(np.diff(wave[0]),base['device']['dt_ns'],rtol=0,atol=1e-10):
        raise ValueError('reference sampling differs from the configured grid')
    hw=hardware_metrics(*wave,Device(**base['device']))
    if hw['endpoint_amplitude_rad_ns']>1e-10 or hw['peak_amplitude_mhz']*amplitude_headroom(base)>base['device']['amplitude_limit_mhz']+1e-8:
        raise ValueError('reference violates endpoint or manufacturing-headroom constraints')
    return wave


def make_ensembles(config,base):
    kwargs=dict(coherence_min=base['coherence_min'],**base['manufacturing_ranges'])
    random=monte_carlo_scenarios(config['training_samples'],config['training_seed'],**kwargs)
    stress=training_scenarios(**kwargs)
    selection=monte_carlo_scenarios(config['selection_samples'],config['selection_seed'],**kwargs)
    return random+stress,selection


def optimizer_options(c,base,seed,regime,method):
    options={**DEFAULTS,**base['grape'],'advanced_execution':True,
        'initialization_strategy':'qualified_fourier_multistart','initialization_reference_family':'calibrated_drag',
        'initialization_scale':0.,'seed':seed,'iterations':c['iterations'],'max_total_iterations':c['iterations'],
        'restarts':c['restarts'],'harmonics':5,'substeps':c['substeps'],'tail_weight':0.,
        'alpha':.95,'risk_weight':0. if method=='conventional' else c['risk_weight'],
        'leakage_weight':c['leakage_weight'],'smoothing':c['smoothing'],
        'quality_targets':base['quality_targets'],'qualified_perturbation_scale':c['qualified_perturbation_scale'],
        'continuation_schedule':c['continuation_schedule'],
        **{k:c[k] for k in ('gradient_tolerance','relative_objective_tolerance','parameter_tolerance')}}
    for name in ('max_objective_evaluations','max_total_scenario_passes','max_surrogate_passes','max_wall_seconds'):
        options.pop(name,None)
    if 'acceleration' in c:options['acceleration']=c['acceleration']
    if c.get('mode')=='intermediate' and options.get('acceleration',{}).get('backend')=='jax_gpu':options['require_verified_gpu']=True
    # Other resource dimensions are uncapped; the shared aggregate iteration cap remains.
    if regime=='equal_objectives':options['max_objective_evaluations']=c['objective_calls']
    elif regime=='equal_passes':options['max_total_scenario_passes']=c['scenario_passes'];options['max_objective_evaluations']=10**9
    elif regime=='equal_wall':options['max_wall_seconds']=c['wall_seconds'];options['max_objective_evaluations']=10**9
    else:raise ValueError('unknown budget regime')
    return options


def initialization_study(out,c,base,pulse):
    device=Device(**base['device']);limit=device.amplitude_limit_mhz*MHZ_TO_RAD_NS/amplitude_headroom(base);rows=[]
    for seed in c['optimizer_seeds']:
        for label,harmonics,perturb in [('legacy',5,.05),('two_harmonic_unperturbed',2,0.),('five_harmonic_unperturbed',5,0.)]:
            basis=FourierControls(pulse[0],harmonics,device.amplitude_limit_mhz*MHZ_TO_RAD_NS)
            coeff=basis.fit(np.array(pulse)[1:].T)
            coeff*=min(1.,.999*limit/max(np.linalg.norm(basis.waveform(coeff),axis=1).max(),1e-30))
            if perturb:coeff+=np.random.default_rng(seed).normal(0.,perturb,len(coeff))
            coeff*=min(1.,.999*limit/max(np.linalg.norm(basis.waveform(coeff),axis=1).max(),1e-30))
            wave=np.array([pulse[0],*basis.waveform(coeff).T]);m=evaluate_waveform(*wave,device,strict=True)
            rows.append(dict(strategy=label,seed=seed,start=0,fidelity=m['fidelity'],leakage=m['leakage'],
                qualified=nominal_qualified(m,base['quality_targets']),waveform_sha256=waveform_hash(wave),
                coefficients=json.dumps(coeff.tolist()),peak_amplitude_mhz=hardware_metrics(*wave,device)['peak_amplitude_mhz'],
                production_channel_solves=1))
        options=optimizer_options(c,base,seed,'equal_objectives','conventional');budget=ResourceBudget()
        _,_,records,_=qualified_fourier_multistart(pulse,device,options,amplitude_headroom(base),budget)
        for record in records:
            rows.append(dict(strategy='qualified_fourier_multistart',seed=seed,start=record['restart'],
                fidelity=record['nominal_fidelity'],leakage=record['nominal_leakage'],qualified=record['accepted'],
                waveform_sha256=record['waveform_sha256'],coefficients=json.dumps(record['coefficients']),
                peak_amplitude_mhz=record['peak_amplitude_mhz'],production_channel_solves=budget.production_channel_solves,
                attempt=record['attempt'],perturbation_magnitude=record['perturbation_magnitude']))
    return rows


def gradient_verification(c,base,pulse,training):
    device=Device(**base['device']);basis=FourierControls(pulse[0],5,device.amplitude_limit_mhz*MHZ_TO_RAD_NS)
    coeff,_=basis.constrained_fit(np.array(pulse)[1:].T,device.amplitude_limit_mhz*MHZ_TO_RAD_NS/amplitude_headroom(base))
    direction=np.random.default_rng(7600).normal(size=10);direction/=np.linalg.norm(direction);eps=1e-5
    checks={'objective_algebra':validate_objective_algebra()}
    for name,cls in [('conventional',EnsembleGRAPE),('risk',RiskEnsembleGRAPE)]:
        model=cls(pulse[0],device,training[:100],optimizer_options(c,base,c['optimizer_seeds'][0],'equal_objectives',name))
        value,g=model.value_gradient(basis.waveform(coeff));analytic=float(basis.gradient(g)@direction)
        difference=(model.value_gradient(basis.waveform(coeff+eps*direction))[0]-model.value_gradient(basis.waveform(coeff-eps*direction))[0])/(2*eps)
        checks[name]=dict(scenarios=100,substeps=c['substeps'],value=value,analytic=analytic,finite_difference=difference,
            error=abs(analytic-difference),passed=abs(analytic-difference)<2e-6)
    if not all(row['passed'] for row in checks.values()):raise RuntimeError('gradient verification failed')
    return checks


def selection_measure(out,wave,device,selection,quality):
    key=waveform_hash(wave);directory=out/'selection_channels';directory.mkdir(exist_ok=True)
    rows=[];new_solves=0;started=perf_counter()
    for index,scenario in enumerate(selection):
        path=directory/f'{key}_{index}.json'
        if not path.exists():new_solves+=1
        rows.append(checkpoint(path,lambda scenario=scenario:evaluate_waveform(*wave,device,scenario)))
    return dict(statistics=summarize_trials(rows,quality),waveform_sha256=key,
        new_selection_channel_solves=new_solves,selection_elapsed_seconds=perf_counter()-started,
        phase='candidate_selection_development_only')


def render_outputs(out,c,base,pulse):
    """Can render an incomplete campaign; missing trials remain explicitly listed."""
    runs=[];missing=[];convergence=[];costs=[];stages=[];results=[]
    for regime in c['regimes']:
        for seed in c['optimizer_seeds']:
            for method in METHODS:
                path=out/f'{regime}_{method}_{seed}.json'
                if not path.exists():missing.append(f'{regime}/{method}/{seed}');continue
                run=json.loads(path.read_text());runs.append(run);last=run['history'][-1]
                stats=run['selection']['statistics'];nominal=run['nominal']
                results.append(dict(regime=regime,method=method,seed=seed,nominal_fidelity=nominal['fidelity'],
                    nominal_leakage=nominal['leakage'],**stats,converged=last['converged'],
                    objective_evaluations=last['objective_evaluations'],elapsed_seconds=last['elapsed_seconds'],
                    selection_feasible=run['selection_feasible']))
                costs.append(dict(regime=regime,method=method,seed=seed,
                    **{k:last[k] for k in ('objective_evaluations','gradient_evaluations','surrogate_scenario_passes','production_channel_solves','total_dissipative_passes','solver_iterations','elapsed_seconds')},
                    selection_channel_solves=run['selection']['new_selection_channel_solves'],
                    selection_elapsed_seconds=run['selection']['selection_elapsed_seconds'],
                    checkpoint_resume_overhead_seconds=run.get('resume_overhead_seconds',0.)))
                for step,record in enumerate(last['accepted_steps']):
                    convergence.append(dict(regime=regime,method=method,seed=seed,step=step,**record))
                for stage in last.get('stages',[]):
                    stages.append(dict(regime=regime,seed=seed,**{k:v for k,v in stage.items() if k not in ('history','incremental_cost')},**stage['incremental_cost']))
    for name,rows in [('optimizer_convergence.csv',convergence),('computational_costs.csv',costs),('continuation_results.csv',stages),('selection_results.csv',results)]:
        if rows:write_csv(out/name,[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
        else:(out/name).write_text('status\nUNEXECUTED\n')
    comparisons=[]
    for regime in c['regimes']:
        for seed in c['optimizer_seeds']:
            conventional=next((r for r in results if r['regime']==regime and r['seed']==seed and r['method']=='conventional'),None)
            if conventional is None:continue
            for method in ('risk','continuation'):
                other=next((r for r in results if r['regime']==regime and r['seed']==seed and r['method']==method),None)
                if other is None:continue
                comparisons.append(dict(regime=regime,seed=seed,method=method,
                    delta_cvar=conventional['cvar95_infidelity']-other['cvar95_infidelity'],
                    mean_fidelity_difference=other['mean_fidelity']-conventional['mean_fidelity'],
                    p05_fidelity_difference=other['p05_fidelity']-conventional['p05_fidelity'],
                    nominal_fidelity_difference=other['nominal_fidelity']-conventional['nominal_fidelity'],
                    mean_leakage_difference=other['mean_leakage']-conventional['mean_leakage'],
                    max_leakage_difference=other['worst_sample_leakage']-conventional['worst_sample_leakage'],
                    runtime_difference=other['elapsed_seconds']-conventional['elapsed_seconds']))
    if comparisons:write_csv(out/'paired_selection_comparisons.csv',comparisons)
    write_json(out/'progress.json',dict(mode=c['mode'],completed_runs=len(runs),missing_runs=missing,
        requested_intermediate_regimes=list(REGIMES),unexecuted_regimes=[r for r in REGIMES if r not in c['regimes']],
        intermediate_full_budget_executed=c['mode']=='intermediate' and not missing,validation_generated=False))
    from grape_development_report import render
    render(out,c,base,pulse,results,comparisons,convergence,stages,costs,missing)
    return runs


def run(c,output,prepare_only=False,max_new_runs=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);base=json.loads(Path(c['base_config']).read_text());device=Device(**base['device'])
    if device.levels!=5:raise ValueError('development protocol requires the unchanged five-level model')
    reference,reference_digest=load_reference_checkpoint(c['reference_checkpoint'])
    sources=['grape.py','grape_research.py','risk_aware_grape.py','continuation_risk_grape.py','grape_development.py',
        'grape_development_report.py','challenge_model.py','robust_optimization.py']
    sources+=c.get('additional_source_files',[])
    acceleration=c.get('acceleration',base['grape'].get('acceleration',{}))
    if acceleration.get('backend','cpu').startswith('jax_'):sources.append('jax_backend.py')
    protocol=dict(config=c,base_config=base,source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        validation_generated=False,phase='development_selection_only',reference_checkpoint_sha256=reference_digest)
    signature=digest(protocol)
    existing=checkpoint(out/'protocol.json',lambda:dict(signature=signature,**protocol))
    if existing['signature']!=signature:raise ValueError('configuration/source changed; use a fresh output directory')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in sources:
        target=snapshot/Path(name).name
        if not target.exists():target.write_bytes(Path(name).read_bytes())
    print('Effective development profile:',json.dumps(c),flush=True)
    training,selection=make_ensembles(c,base)
    for label,scenarios in [('training',training),('selection',selection)]:
        data=[asdict(s) for s in scenarios];stored=checkpoint(out/f'{label}_devices.json',lambda data=data:data)
        if stored!=data:raise ValueError('fixed ensemble changed')
    pulse=validate_reference_waveform(reference,base);nominal=evaluate_waveform(*pulse,device,strict=True)
    if not nominal_qualified(nominal,base['quality_targets']):raise ValueError('reference is unqualified')
    write_json(out/'shared_reference.json',dict(waveform=pulse.tolist(),waveform_sha256=waveform_hash(pulse),nominal=nominal,
        calibration_provenance=c['reference_checkpoint'],calibration=reference.get('calibration'),
        note='Previously calibrated nominal reference reused equally; historical calibration is shared, not charged to individual optimizers.'))
    write_json(out/'effective_config.json',dict(**protocol,signature=signature,
        resolved_options={regime:{method:optimizer_options(c,base,c['optimizer_seeds'][0],regime,method) for method in METHODS} for regime in c['regimes']}))
    if prepare_only:
        render_outputs(out,c,base,pulse);return
    preflight_options=optimizer_options(c,base,c['optimizer_seeds'][0],c['regimes'][0],'risk')
    if c['mode']=='intermediate' and preflight_options.get('acceleration',{}).get('backend')=='jax_gpu':
        from jax_backend import gpu_preflight
        write_json(out/'gpu_preflight.json',gpu_preflight(pulse,device,training,preflight_options))
    initializations=checkpoint(out/'initialization_checkpoint.json',lambda:initialization_study(out,c,base,pulse))
    write_csv(out/'initialization_results.csv',initializations)
    checkpoint(out/'gradient_verification.json',lambda:gradient_verification(c,base,pulse,training))
    checkpoint(out/'drag_selection.json',lambda:dict(nominal=nominal,**selection_measure(out,pulse,device,selection,base['quality_targets'])))
    if c.get('full_matched_outputs'):
        from full_matched_report import export
        export(out)
    new=0
    for regime in c['regimes']:
        for seed in c['optimizer_seeds']:
            for method in METHODS:
                path=out/f'{regime}_{method}_{seed}.json'
                if path.exists():continue
                if max_new_runs is not None and new>=max_new_runs:
                    render_outputs(out,c,base,pulse);return
                print(f'Optimizing {regime}/{method}/{seed}',flush=True);options=optimizer_options(c,base,seed,regime,method)
                optimization_path=out/f'{regime}_{method}_{seed}_optimization.json'
                options['progress_checkpoint']=str(out/f'{regime}_{method}_{seed}_progress.json')
                if optimization_path.exists():
                    saved=json.loads(optimization_path.read_text())
                    if saved['protocol_signature']!=signature:raise ValueError('optimizer checkpoint protocol changed')
                    wave=np.asarray(saved['waveform']);history=saved['history']
                    if waveform_hash(wave)!=saved['waveform_sha256']:raise ValueError('optimizer waveform checkpoint changed')
                else:
                    progress=Path(options['progress_checkpoint'])
                    if progress.exists():
                        archive=out/'interrupted_attempts';archive.mkdir(exist_ok=True)
                        attempt=archive/f'{progress.stem}_{len(list(archive.glob(progress.stem+"_*")))}.json'
                        attempt.write_bytes(progress.read_bytes())
                    wave,history=(continuation(pulse,device,training,options,amplitude_headroom(base)) if method=='continuation'
                        else refine(pulse,device,training,options,amplitude_headroom(base),model_class=EnsembleGRAPE if method=='conventional' else RiskEnsembleGRAPE))
                    write_json(optimization_path,dict(protocol_signature=signature,regime=regime,seed=seed,method=method,
                        options=options,waveform=np.asarray(wave).tolist(),waveform_sha256=waveform_hash(wave),history=history))
                if c.get('full_matched_outputs'):
                    from full_matched_report import export
                    export(out)
                # Freeze one training-retained candidate before any selection observations.
                frozen=waveform_hash(wave);nom=evaluate_waveform(*wave,device,strict=True)
                measured=selection_measure(out,wave,device,selection,base['quality_targets'])
                assert waveform_hash(wave)==frozen
                hardware=hardware_metrics(*wave,device)
                result=dict(regime=regime,seed=seed,method=method,options=options,waveform=np.asarray(wave).tolist(),
                    waveform_sha256=frozen,history=history,nominal=nom,hardware=hardware,selection=measured,
                    selection_feasible=bool(nominal_qualified(nom,base['quality_targets']) and
                        measured['statistics']['worst_sample_leakage']<=base['quality_targets']['leakage_max'] and
                        hardware['peak_amplitude_mhz']*amplitude_headroom(base)<=device.amplitude_limit_mhz+1e-8),
                    validation_generated=False)
                write_json(path,result);new+=1
                render_outputs(out,c,base,pulse)
                if c.get('full_matched_outputs'):
                    from full_matched_report import export
                    export(out)
    render_outputs(out,c,base,pulse)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='grape_development_config.json');parser.add_argument('--output',required=True)
    parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--max-new-runs',type=int)
    args=parser.parse_args();run(load_config(args.config,args.diagnostic),args.output,args.prepare_only,args.max_new_runs)
