"""Reproducible descriptive exports for the frozen matched development study."""
import json,sys
from pathlib import Path
from time import perf_counter
import numpy as np
from grape_development import write_json,write_csv,METHODS,REGIMES,amplitude_headroom
from grape_research import waveform_hash


def termination(last):
    if last.get('converged'):return 'CONVERGED'
    text=' '.join([str(last.get('budget_exhaustion','')),str(last.get('message','')),
        *[str(s.get('termination','')) for s in last.get('stages',[])] ]).lower()
    for words,label in [(('wall','time limit'),'WALL_TIME_LIMIT'),(('scenario-pass','scenario pass'),'SCENARIO_PASS_LIMIT'),
        (('objective allocation','objective evaluation','evaluation limit'),'EVALUATION_LIMIT'),
        (('iteration',),'ITERATION_LIMIT'),(('nan','nonfinite','numerical','accuracy gate'),'NUMERICAL_FAILURE'),
        (('incompatible','constraint failure'),'CONSTRAINT_FAILURE')]:
        if any(w in text for w in words):return label
    return 'STOPPED_WITHOUT_CONVERGENCE'


def all_histories(last):
    return [s['history'][-1] for s in last.get('stages',[]) if s.get('history')] or [last]


def verify_trial(run,c,base):
    """Pointwise CPU/GPU and 4/8-substep checks, never new uncertainty draws."""
    from challenge_model import Device,Scenario,evaluate_waveform,hardware_metrics,training_scenarios
    from grape import DEFAULTS,EnsembleGRAPE
    from risk_aware_grape import RiskEnsembleGRAPE
    from jax_backend import JAXEnsemble
    started=perf_counter();device=Device(**base['device']);wave=np.asarray(run['waveform']);controls=wave[1:].T
    stress=training_scenarios(base['coherence_min'],**base['manufacturing_ranges'])
    scenarios=[Scenario(),*stress[-2:]]
    reference=[evaluate_waveform(*wave,device,s,strict=True) for s in scenarios]
    expected=np.array([[1-r['fidelity'],r['leakage']] for r in reference])
    detail={};gradient={};physical={};gpu=c['acceleration']['backend']=='jax_gpu'
    options={**DEFAULTS,**run['options'],'risk_weight':0. if run['method']=='conventional' else c['risk_weight']}
    cls=EnsembleGRAPE if run['method']=='conventional' else RiskEnsembleGRAPE
    for substeps in (4,8):
        o={**options,'substeps':substeps,'require_verified_gpu':False,'acceleration':{**c['acceleration'],
            'require_accuracy_gate':False,'fallback_to_cpu':False}}
        cpu=cls(wave[0],device,scenarios,dict(o))
        if gpu:
            model=JAXEnsemble(wave[0],device,scenarios,o,cls)
            metrics=model.metrics(controls);state=model.metrics(controls,states=True)
            rho=state.transpose(0,2,1).reshape(-1,5,5).transpose(0,2,1)
            physical[substeps]=dict(trace_error=float(np.max(abs(np.trace(rho,axis1=1,axis2=2)-1))),
                hermiticity_error=float(np.max(abs(rho-np.conj(rho.transpose(0,2,1))))),
                minimum_eigenvalue=float(np.linalg.eigvalsh(rho).min()))
            if substeps==4:
                a,ga=cpu.value_gradient(controls);b,gb=model.value_gradient(controls)
                gradient=dict(objective_error=abs(a-b),maximum_gradient_error=float(np.max(abs(ga-gb))),
                    passed=bool(abs(a-b)<=2e-6 and np.allclose(ga,gb,atol=2e-7,rtol=2e-6)))
        else:
            rows=[]
            for packet in cpu.models:
                cpu.scenario_value_gradient(controls,packet);m=cpu.last_scenario_metrics
                rows.append([1-m['fidelity'],m['leakage']])
            metrics=np.asarray(rows)
        detail[substeps]=dict(maximum_fidelity_error=float(np.max(abs(metrics[:,0]-expected[:,0]))),
            maximum_leakage_error=float(np.max(abs(metrics[:,1]-expected[:,1]))))
    hardware=hardware_metrics(*wave,device)
    nominal_error=max(abs(reference[0][k]-run['nominal'][k]) for k in ('fidelity','leakage'))
    constraints=bool(np.all(wave[1:,[0,-1]]==0) and hardware['peak_amplitude_mhz']*amplitude_headroom(base)<=device.amplitude_limit_mhz+1e-8)
    passed=nominal_error<=2e-6 and constraints and all(max(v.values())<=2e-6 for v in detail.values())
    passed=passed and (not gpu or (gradient['passed'] and all(v['trace_error']<=2e-6 and
        v['hermiticity_error']<=2e-6 and v['minimum_eigenvalue']>=-2e-6 for v in physical.values())))
    return dict(passed=bool(passed),waveform_sha256=waveform_hash(wave),nominal_reproduction_error=nominal_error,
        interpolation_checks=detail,gradient=gradient,physical_states=physical,constraints_passed=constraints,
        verification_seconds=perf_counter()-started,scope='nominal and two predeclared joint stress cases; pointwise, not a uniform bound')


def export(output):
    out=Path(output);protocol=json.loads((out/'protocol.json').read_text());c=protocol['config'];base=protocol['base_config']
    trials=[];evaluations=[];costs=[];selected=[];hashes={};missing=[];runs=[];convergence=[]
    numerical_path=out/'numerical_verification.json'
    numerical=json.loads(numerical_path.read_text()) if numerical_path.exists() else {}
    for regime in c['regimes']:
        for seed in c['optimizer_seeds']:
            for method in METHODS:
                stem=f'{regime}_{method}_{seed}';path=out/f'{stem}.json';opt=out/f'{stem}_optimization.json'
                if not path.exists():missing.append(stem)
                if not path.exists() and not opt.exists():continue
                run=json.loads((path if path.exists() else opt).read_text());last=run['history'][-1]
                if path.exists():
                    runs.append(run)
                    if stem not in numerical:
                        numerical[stem]=verify_trial(run,c,base);write_json(numerical_path,numerical)
                histories=all_histories(last);first=next((r for h in histories for r in h.get('accepted_steps',[])),None)
                initial=last.get('initial_objective')
                if method=='continuation' and first:
                    initial=1-first['mean_training_fidelity']+c['continuation_schedule'][-1]*first['smooth_cvar95_training_infidelity']+c['leakage_weight']*first['mean_training_leakage']+first['regularization']
                final=last['loss'];outcomes=[r for h in histories for r in h.get('restart_outcomes',[]) if r['returned_waveform_sha256']==run['waveform_sha256']]
                retained=outcomes[-1] if outcomes else {}
                coefficient_change=last.get('coefficient_change_norm')
                if coefficient_change is None and first and last.get('coefficients') is not None:
                    coefficient_change=float(np.linalg.norm(np.asarray(last['coefficients'])-first['coefficients']))
                trial=dict(regime=regime,method=method,seed=seed,status=termination(last),converged=last['converged'],
                    selection_completed=path.exists(),initial_objective=initial,final_objective=final,
                    relative_objective_reduction=(initial-final)/abs(initial) if initial and final is not None else None,
                    gradient_norm=retained.get('gradient_norm'),projected_gradient_norm=retained.get('projected_gradient_norm'),
                    coefficient_change=coefficient_change,accepted_feasible_improvements=len(last.get('accepted_steps',[])),
                    termination_reason=last['message'],actual_backend=c['acceleration']['backend'],**{k:last[k] for k in
                    ('objective_evaluations','gradient_evaluations','surrogate_scenario_passes','production_channel_solves',
                     'total_dissipative_passes','solver_iterations','elapsed_seconds','gpu_backend_wall_seconds')})
                trials.append(trial);hashes[stem]=run['waveform_sha256']
                cost={k:trial[k] for k in ('regime','method','seed','objective_evaluations','gradient_evaluations',
                    'surrogate_scenario_passes','production_channel_solves','total_dissipative_passes','elapsed_seconds','gpu_backend_wall_seconds')}
                cost['wall_cap_overshoot_seconds']=max(0.,last['elapsed_seconds']-c['wall_seconds']) if regime=='equal_wall' else 0.
                cost.update(selection_seconds=run.get('selection',{}).get('selection_elapsed_seconds'),
                    selection_channel_solves=run.get('selection',{}).get('new_selection_channel_solves'))
                costs.append(cost)
                for stage,h in enumerate(histories):
                    for record in h.get('evaluation_trace',[]):evaluations.append(dict(regime=regime,method=method,seed=seed,stage=stage,**record))
                convergence += [dict(regime=regime,method=method,seed=seed,**r) for r in last.get('accepted_steps',[])]
                if path.exists():selected.append(dict(regime=regime,method=method,seed=seed,nominal_fidelity=run['nominal']['fidelity'],
                    nominal_leakage=run['nominal']['leakage'],peak_amplitude_mhz=run['hardware']['peak_amplitude_mhz'],**run['selection']['statistics']))
    if (out/'drag_selection.json').exists():
        drag=json.loads((out/'drag_selection.json').read_text());ref=json.loads((out/'common_reference.json').read_text())
        selected.append(dict(regime='shared',method='calibrated_drag',seed=None,nominal_fidelity=drag['nominal']['fidelity'],
            nominal_leakage=drag['nominal']['leakage'],peak_amplitude_mhz=ref['hardware']['peak_amplitude_mhz'],**drag['statistics']))
    comparisons=[];regimes={}
    for regime in c['regimes']:
        comparisons_regime=[]
        for seed in c['optimizer_seeds']:
            common=dict(regime=regime,seed=seed)
            a=next((r for r in selected if all(r[k]==v for k,v in common.items()) and r['method']=='conventional'),None)
            if a is None:continue
            for method in ('risk','continuation'):
                b=next((r for r in selected if all(r[k]==v for k,v in common.items()) and r['method']==method),None)
                if b is None:continue
                d=a['cvar95_infidelity']-b['cvar95_infidelity']
                row=dict(**common,method=method,delta_cvar=d,percent_cvar_improvement=100*d/a['cvar95_infidelity'],
                    **{k+'_difference':b[k]-a[k] for k in ('nominal_fidelity','mean_fidelity','p05_fidelity','mean_leakage','worst_sample_leakage')})
                comparisons.append(row);comparisons_regime.append(row)
        regimes[regime]={}
        for method in ('risk','continuation'):
            rows=[r for r in comparisons_regime if r['method']==method]
            regimes[regime][method]=dict(paired_seeds=len(rows),mean_delta_cvar=float(np.mean([r['delta_cvar'] for r in rows])) if rows else None,
                mean_percent_improvement=float(np.mean([r['percent_cvar_improvement'] for r in rows])) if rows else None,
                mean_fidelity_change=float(np.mean([r['mean_fidelity_difference'] for r in rows])) if rows else None,
                p05_change=float(np.mean([r['p05_fidelity_difference'] for r in rows])) if rows else None)
    for name,rows in [('optimization_trials.csv',trials),('objective_evaluations.csv',evaluations),
        ('computational_costs.csv',costs),('selection_metrics.csv',selected),('paired_selection_comparisons.csv',comparisons),
        ('optimizer_convergence.csv',convergence)]:
        if rows:write_csv(out/name,[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows])
        else:(out/name).write_text('status\nUNEXECUTED\n')
    write_json(out/'waveform_hashes.json',hashes)
    if not numerical_path.exists():write_json(numerical_path,{})
    failures=[k for k,v in numerical.items() if not v['passed']]
    complete=len(runs)==27
    converged=sum(t['converged'] for t in trials if t['selection_completed'])
    summary=dict(completed_trials=len(runs),optimized_trials=len(trials),required_trials=27,missing_trials=missing,
        actual_backend=c['acceleration']['backend'],precision='float64 / complex128',converged_trials=converged,
        convergence_rate=converged/len(runs) if runs else None,total_optimizer_wall_seconds=sum(t['elapsed_seconds'] for t in trials),
        gpu_backend_wall_seconds=sum(t['gpu_backend_wall_seconds'] for t in trials),
        gpu_time_scope='synchronized optimizer GPU calls including transfer/dispatch/JIT; excludes CPU probes, shared setup and post-trial numerical verification; not CUDA-kernel-only time',
        numerical_failures=failures,regime_comparisons=regimes,hyperparameter_search_trials=0,
        hyperparameters_predeclared=True,validation_generated=False,statistical_superiority_established=False,
        decision='GO' if complete and not failures and converged==27 else 'NO-GO',
        decision_reason='All completed, numerical and convergence criteria passed' if complete and not failures and converged==27 else
            'Incomplete development trials and/or numerical or convergence limitations remain',
        resume_command=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python full_matched_study.py --config full_matched_config.json --output {out} --backend {c["backend_requested"]}',
        confirmatory_command=None,confirmatory_status='Not frozen or authorized by this development output; existing research_benchmark confirmation does not cover three-regime continuation comparison')
    write_json(out/'study_summary.json',summary)
    report(out,c,trials,selected,comparisons,summary)
    return summary


def report(out,c,trials,selected,comparisons,summary):
    lines=['# Full matched-budget development study','',f'**{summary["decision"]}: {summary["decision_reason"]}.**',
        f'Completed selection-evaluated trials: {summary["completed_trials"]}/27. Backend: {summary["actual_backend"]}; FP64. No independent validation was generated.',
        '', 'All methods use the stronger nominal DRAG source. Its calibration was budget limited, not globally optimized or fully converged. Comparisons use the same 110 training and 200 selection devices, five harmonics, four interpolation substeps and unchanged physical/quality constraints.',
        '', '| Regime | Seed | Method | Termination | Calls | Passes | Optimizer wall (s) | Converged |',
        '|---|---:|---|---|---:|---:|---:|---|']
    for t in trials:lines.append(f'| {t["regime"]} | {t["seed"]} | {t["method"]} | {t["status"]} | {t["objective_evaluations"]} | {t["total_dissipative_passes"]} | {t["elapsed_seconds"]:.2f} | {t["converged"]} |')
    lines+=['','| Regime | Seed | Method | Nominal F | Mean F | P05 F | CVaR95 | Mean leakage | Maximum leakage | Minimum F | Peak MHz |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected:lines.append(f'| {r["regime"]} | {r["seed"]} | {r["method"]} | {r["nominal_fidelity"]:.10f} | {r["mean_fidelity"]:.10f} | {r["p05_fidelity"]:.10f} | {r["cvar95_infidelity"]:.9g} | {r["mean_leakage"]:.9g} | {r["worst_sample_leakage"]:.9g} | {r["worst_sample_fidelity"]:.10f} | {r["peak_amplitude_mhz"]:.5f} |')
    lines+=['','| Regime | Seed | Method | Conventional − method CVaR | Improvement (%) | Mean F change | P05 change |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in comparisons:lines.append(f'| {r["regime"]} | {r["seed"]} | {r["method"]} | {r["delta_cvar"]:.9g} | {r["percent_cvar_improvement"]:.4f} | {r["mean_fidelity_difference"]:.9g} | {r["p05_fidelity_difference"]:.9g} |')
    lines+=['','Positive CVaR deltas favor the risk method. These are paired development observations on common devices, not independent statistical superiority. Three seeds are not 600 independent devices. No inferential interval or significance claim is manufactured.',
        '', 'Equal-call comparisons do not equalize compute. Equal-pass ledgers include surrogate passes and CPU nominal channels as separate quantities. Equal-wall caps include initialization, restarts, propagation, constraints and nominal qualification, checked between indivisible evaluations. Overshoot is explicit in computational_costs.csv. Fixed post-optimization CPU selection, post-trial numerical verification and rendering are measurement/analysis overhead outside the optimizer cap and separately recorded.',
        '',f'Known optimizer GPU-call wall seconds: {summary["gpu_backend_wall_seconds"]:.3f}. {summary["gpu_time_scope"]}. Shared GPU qualification/warmup costs are recorded in gpu_qualification.json. Historical baseline calibration is in common_reference.json. Lambda/mu were not tuned: hyperparameter-search cost in this campaign is zero.',
        '',f'Converged: {summary["converged_trials"]}/{summary["completed_trials"]}. Convergence requires all predeclared KKT, relative-objective, parameter-change and feasibility criteria, not objective decrease alone. Numerical failures: {summary["numerical_failures"]}. Checks are pointwise at nominal/two joint stress cases with four/eight substeps, not uniform accuracy guarantees.',
        '', '## Direct answers','']
    for question,method in [('Does fixed-risk improve CVaR?','risk'),('Does continuation improve CVaR relative to conventional?','continuation')]:
        lines.append(question)
        for regime,data in summary['regime_comparisons'].items():lines.append(f'- {regime}: {data[method]}')
    lines+=['','Additional continuation-versus-fixed-risk improvement must be assessed from matched seed/regime selection_metrics.csv; no winner is asserted from an incomplete study.',
        '', 'Equal-pass and equal-wall results must both support any efficiency claim; favorable equal-call results alone are insufficient.',
        '',f'Unfinished trials: {summary["missing_trials"] or "none"}.',
        '', 'Resume after an interruption (an active-worker lock prevents duplication):','', '```bash',summary['resume_command'],'```',
        '', 'Confirmatory study command: **not issued**. The confirmatory protocol is not frozen while required trials, convergence or numerical checks remain unresolved. No final independent validation has been generated.']
    (out/'study_report.md').write_text('\n'.join(lines)+'\n')
    figures(out,trials,selected,comparisons)


def figures(out,trials,selected,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(conventional='tab:blue',risk='tab:orange',continuation='tab:green')
    def save(fig,name):fig.tight_layout();fig.savefig(out/name,dpi=140);plt.close(fig)
    for index,(key,label) in enumerate([('objective_evaluations','Objective calls'),('total_dissipative_passes','Dissipative scenario passes'),('elapsed_seconds','Optimizer wall seconds')],1):
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,regime in zip(axes,REGIMES):
            for s in selected:
                if s['regime']!=regime:continue
                t=next(t for t in trials if all(t[k]==s[k] for k in ('regime','method','seed')))
                ax.scatter(t[key],s['cvar95_infidelity'],color=colors[s['method']],label=s['method'])
            ax.set(title=regime,xlabel=label,ylabel='Selection CVaR95 infidelity')
        save(fig,f'figure_{index}_cvar_vs_{key}.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,regime in zip(axes,REGIMES):
        for r in comparisons:
            if r['regime']==regime:ax.scatter(r['seed'],r['delta_cvar'],color=colors[r['method']],marker='o' if r['method']=='risk' else '^')
        ax.axhline(0,color='gray',ls='--');ax.set(title=regime,xlabel='Optimizer seed',ylabel='Paired CVaR improvement')
    save(fig,'figure_4_paired_cvar_improvements.png')
    for index,x,y in [(5,'mean_fidelity','p05_fidelity'),(6,'mean_leakage','mean_fidelity'),
        (9,'nominal_fidelity','cvar95_infidelity'),(10,'mean_leakage','cvar95_infidelity')]:
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,regime in zip(axes,REGIMES):
            for s in selected:
                if s['regime']==regime:ax.scatter(s[x],s[y],color=colors[s['method']])
            ax.set(title=regime,xlabel=x,ylabel=y)
        save(fig,f'figure_{index}_{x}_vs_{y}.png')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,regime in zip(axes,REGIMES):
        for t in trials:
            if t['regime']!=regime:continue
            run=json.loads((out/f'{regime}_{t["method"]}_{t["seed"]}_optimization.json').read_text())
            a=run['history'][-1]['accepted_steps']
            ax.plot([r['objective_evaluations'] for r in a],[r['cvar95_training_infidelity'] for r in a],color=colors[t['method']],alpha=.65,label=f'{t["method"]}/{t["seed"]}')
        ax.set(title=regime,xlabel='Objective calls',ylabel='Retained training CVaR95')
        if ax.lines:ax.legend(fontsize=5)
    save(fig,'figure_7_seed_convergence.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key in zip(axes,('objective_evaluations','total_dissipative_passes','elapsed_seconds')):
        for i,t in enumerate(trials):ax.bar(i,t[key],color=colors[t['method']])
        ax.set(xlabel='Trial in regime/seed/method order',ylabel=key)
    save(fig,'figure_8_computational_costs.png')
