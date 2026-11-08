"""Export saved and replayed optimizer telemetry without touching validation data."""
import csv,json
from pathlib import Path
import numpy as np
from grape import refine
from risk_aware_grape import refine as risk_refine
from challenge_model import Device,Scenario


def replay(out,source='results/research_benchmark_complete'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);source=Path(source)
    protocol=json.loads((source/'protocol.json').read_text());base=protocol['effective_config'];profile=protocol['profile']
    seed=np.asarray(json.loads((source/'references.json').read_text())['calibrated_drag']['waveform'])
    training=[Scenario(**r) for r in json.loads((source/'training_devices.json').read_text())]
    rows=[];results={}
    for name in ('conventional','mu_0','A_mean','B_worst','C_cvar','D_mean_cvar','E_mean_cvar_leak'):
        old=json.loads((source/f'equal_evaluations_{name}_6200.json').read_text())
        options={**base['grape'],'seed':6200,'alpha':.95,'smoothing':profile['smoothing'],'risk_weight':0.,'leakage_weight':profile['mu'],'mean_weight':1.,'worst_weight':0.,'initialization_scale':profile['initialization_scale'],'max_objective_evaluations':3}
        options.update(old['weights'] or {})
        wave,h=(refine if name=='conventional' else risk_refine)(seed,Device(**base['device']),training,options,1.03)
        a=h[-1]['execution_audit']
        a['matches_saved_waveform']=bool(np.array_equal(wave,np.asarray(old['candidate']['waveform'])))
        results[name]=a
        for entry in a['evaluation_trace']:rows.append(dict(method=name,**entry))
    (out/'historical_replays.json').write_text(json.dumps(results,indent=2)+'\n')
    with (out/'historical_replay_traces.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return results


def summaries(out):
    out=Path(out);runs=json.loads((out/'diagnostic_optimizer_runs.json').read_text());old=json.loads((out/'historical_execution_audit.json').read_text())
    rows=[]
    for r in old:
        rows.append(dict(phase='historical',method=r['method'],regime=r['regime'],initial_objective=r['initial_objective'],final_objective=r['final_objective'],initial_gradient_norm=r['gradient_norm'],final_gradient_norm=r['gradient_norm'] if r['coefficient_change']==0 else None,initial_coefficients=json.dumps(r['initial_coefficients']),final_coefficients=json.dumps(r['final_coefficients']),objective_evaluations=r['objective_evaluations'],gradient_evaluations=r['gradient_evaluations'],accepted_feasible_improvements=None,solver_callback_steps=None,termination=json.dumps(r['termination']),initial_hash=r['initial_waveform_sha256'],final_hash=r['final_waveform_sha256'],initial_fidelity=r['initial_fidelity'],final_fidelity=r['final_fidelity'],initial_leakage=r['initial_leakage'],final_leakage=r['final_leakage']))
    for name,r in runs.items():
        last=r['history'][-1];a=last['execution_audit'];first=a['evaluation_trace'][0]
        rows.append(dict(phase='controlled_diagnostic',method=name,regime='fixed training only',initial_objective=a['initial_objective'],final_objective=a['final_objective'],initial_gradient_norm=first['gradient_norm'],final_gradient_norm=a['final_gradient_norm'],initial_coefficients=json.dumps(a['initial_coefficients']),final_coefficients=json.dumps(a['final_coefficients']),objective_evaluations=last['objective_evaluations'],gradient_evaluations=last['gradient_evaluations'],accepted_feasible_improvements=a['retained_feasible_improvements'],solver_callback_steps=a['solver_callback_steps'],termination=json.dumps([t['message'] for t in r['history'] if 'message' in t]),initial_hash=a['initial_waveform_sha256'],final_hash=a['final_waveform_sha256'],initial_fidelity=r['production_initial']['fidelity'],final_fidelity=r['production_final']['fidelity'],initial_leakage=r['production_initial']['leakage'],final_leakage=r['production_final']['leakage']))
    replayed=json.loads((out/'historical_replays.json').read_text())
    historical=json.loads((out/'historical_execution_audit.json').read_text())
    for row in rows:
        if row['phase']=='controlled_diagnostic':
            last=runs[row['method']]['history'][-1]
        else:
            previous=next(r for r in historical if r['method']==row['method'] and r['regime']==row['regime'])
            last=previous
            if row['regime']=='equal_evaluations':
                replay_audit=replayed[row['method']]
                row['accepted_feasible_improvements']=replay_audit['retained_feasible_improvements']
                row['solver_callback_steps']=replay_audit['solver_callback_steps']
        row['optimizer_elapsed_seconds']=last.get('elapsed_seconds')
        row['surrogate_scenario_passes']=last.get('surrogate_scenario_passes')
        row['step_count_provenance']='instrumented replay' if row['phase']=='historical' and row['regime']=='equal_evaluations' else ('unavailable in original wall-limited trace' if row['phase']=='historical' else 'instrumented run')
    with (out/'optimizer_run_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_production(out):
    from grape import EnsembleGRAPE,DEFAULTS
    from challenge_model import evaluate_waveform
    out=Path(out);runs=json.loads((out/'diagnostic_optimizer_runs.json').read_text())
    c=json.loads((out/'effective_research_config.json').read_text());base=json.loads(Path('challenge_config.json').read_text());device=Device(**base['device'])
    comparisons={}
    for name,run in runs.items():
        wave=np.loadtxt(out/f'{name}_final_waveform.csv',delimiter=',',skiprows=1).T
        actual=evaluate_waveform(*wave,device,strict=True)
        detail=dict(exported_waveform_reproduction={k:abs(actual[k]-run['production_final'][k]) for k in ('fidelity','leakage')},substeps={})
        for substeps in (2,4):
            model=EnsembleGRAPE(wave[0],device,[Scenario()],{**DEFAULTS,'substeps':substeps,'leakage_weight':0.})
            f,_=model.scenario_value_gradient(wave[1:].T,model.models[0]);model.options['leakage_weight']=1.
            combined,_=model.scenario_value_gradient(wave[1:].T,model.models[0]);leak=combined-f
            detail['substeps'][substeps]=dict(fidelity=1-f,leakage=leak,fidelity_error=abs(1-f-actual['fidelity']),leakage_error=abs(leak-actual['leakage']))
        comparisons[name]=detail
    tolerance=c['engineering_thresholds']['solver_metric_tolerance']
    for detail in comparisons.values():
        detail['tolerance']=tolerance
        detail['passed']=all(error<=tolerance for error in detail['exported_waveform_reproduction'].values()) and all(max(row['fidelity_error'],row['leakage_error'])<=tolerance for row in detail['substeps'].values())
    (out/'production_reproduction.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    if not all(detail['passed'] for detail in comparisons.values()):
        raise RuntimeError('Production/surrogate reproduction exceeded the existing solver tolerance')
    return comparisons


def finalize(out):
    out=Path(out);summaries(out);verification=verify_production(out)
    runs=json.loads((out/'diagnostic_optimizer_runs.json').read_text())
    with (out/'optimizer_run_summary.csv').open() as f:metadata=list(csv.DictReader(f))
    with (out/'optimizer_traces.csv').open() as f:trace=[r for r in csv.DictReader(f) if r.get('phase')!='historical_summary']
    fresh={r['method']:r for r in metadata if r['phase']=='controlled_diagnostic'}
    trace=[{**fresh[r['method']],**r,'phase':'controlled_evaluation'} for r in trace]
    trace += [{**r,'phase':'historical_summary'} for r in metadata if r['phase']=='historical']
    fields=list(dict.fromkeys(k for r in trace for k in r))
    with (out/'optimizer_traces.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(trace)
    report=out/'optimizer_diagnostic_report.md'
    replays=json.loads((out/'historical_replays.json').read_text())
    matched=sum(bool(r['matches_saved_waveform']) for r in replays.values())
    addition=f'\n\n## Verified conclusions\n\nHistorical replays match {matched}/{len(replays)} saved waveforms exactly. Inspect historical_replay_traces.csv for trial feasibility and retention; a solver callback is not counted as a feasible objective improvement. Historical equal-evaluation step counts in optimizer_run_summary.csv come from deterministic instrumented replays; original wall-limited step counts are unavailable.\n\n'
    addition+='The selected historical risk row used lambda=0 and mu=0 after selection ties; this is not evidence that these hyperparameters are optimal. The third evaluation in each replay failed to improve the retained feasible point.\n\n'
    with (out/'initialization_comparison.csv').open() as f:initializations={r['stage']:r for r in csv.DictReader(f)}
    raw=float(initializations['2 harmonics: unperturbed least-squares']['fidelity'])
    perturbed=float(initializations['2 harmonics: seeded perturbation .05']['fidelity'])
    drag=float(initializations['qualified calibrated DRAG']['fidelity'])
    addition+=f'The unperturbed two-harmonic fit had F={raw:.9f}; the .05 perturbed start had F={perturbed:.9f}. The seeded perturbation causes the large loss of nominal fidelity. Compare the constrained and ordinary fits in initialization_comparison.csv; legacy initialization defaults remain unchanged.\n\n'
    for name,run in runs.items():
        last=run['history'][-1];a=last['execution_audit'];reduction=100*(a['initial_objective']-a['final_objective'])/abs(a['initial_objective'])
        addition+=f"{name}: objective reduction {reduction:.2f}%; coefficient change norm {a['coefficient_change_norm']:.6g}; {last['objective_evaluations']} objective/gradient evaluations; {last['surrogate_scenario_passes']} dissipative scenario passes; optimizer time {last['elapsed_seconds']:.2f}s. Nominal final F={run['production_final']['fidelity']:.9f}, compared with calibrated DRAG F={drag:.9f}. Termination: {', '.join(t['message'] for t in run['history'] if 'message' in t)}.\n\n"
    addition+='Both runs remain iteration-limited. Training-objective improvement is not nominal-fidelity or independent robustness superiority. Nominal fidelity falls slightly while leakage decreases; the objective optimizes an uncertain ensemble plus leakage and smoothness, so these metrics need not improve together. This limited experiment cannot separate insufficient convergence, objective tradeoffs and representation limits.\n\n'
    config=json.loads((out/'effective_research_config.json').read_text());full=config['full'];quick=config['quick']
    addition+=f"Full mode uses {full['selection_samples']} selection devices, {full['validation_samples']} validation devices, {full['training_samples']} random training devices plus deterministic cases, {len(full['optimizer_seeds'])} optimizer seeds, {full['iterations']} iterations, {full['objective_calls']} calls in the equal-evaluation regime and {full['wall_seconds']:g}s in the equal-wall regime. Quick mode remains separate ({quick['objective_calls']} calls/{quick['iterations']} iteration). These are caps, not guaranteed executed work: wall or scenario-pass limits can stop a run earlier. No quick-budget propagation was found in the current full profile. The configured scenario-pass cap was previously not passed to GRAPE; it is now enforced ({full['scenario_passes']} full / {quick['scenario_passes']} quick), and inherited call/wall caps are cleared before resolving each research regime. This does not change the historical three-call replay: its existing evaluation cap already enforced the same limit.\n\n"
    addition+='Production waveform-reload and surrogate-substep checks are in production_reproduction.json. No independent validation data were used or generated. The full campaign was not run.\n'
    report.write_text(report.read_text().split('\n\n## Verified conclusions')[0]+addition)
    return verification
