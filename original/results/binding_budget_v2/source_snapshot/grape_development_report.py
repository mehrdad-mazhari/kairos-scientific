"""Descriptive development reports and scientific plots; no validation inference."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS={'conventional':'#246ca7','risk':'#d25f23','continuation':'#328456'}


def render(out,c,base,pulse,results,comparisons,convergence,stages,costs,missing):
    out=Path(out);status='COMPLETE FOR EXPLICIT SHORT DIAGNOSTIC' if c['mode']=='short_diagnostic' and not missing else ('INCOMPLETE' if missing else 'COMPLETE DEVELOPMENT EXPERIMENT')
    intro=f"Mode: **{c['mode']}**. Status: **{status}**. This is development on common selection scenarios, not independent validation. No validation dataset was generated.\n\n"
    budget=(f"Requested full intermediate maxima: 150 aggregate iterations, 1000 objective calls in the equal-call regime, five restart allocations, five harmonics, 100 random training devices plus ten deterministic cases and 200 selection devices. Executed profile caps: {c['iterations']} iterations, {c['objective_calls']} calls, {c['substeps']} interpolation substeps, seeds {c['optimizer_seeds']}, regimes {c['regimes']}. Budgets are maxima, not promises of executed work.\n\n")
    initpath=out/'initialization_results.csv'
    initializations=list(csv.DictReader(initpath.open())) if initpath.exists() else []
    initial=['# Initialization study','',intro,'Qualified starts preserve the amplitude disk, fixed duration and zero endpoints. The unperturbed constrained fit is always start zero; seeded perturbations shrink until nominal fidelity/leakage qualify. Legacy initialization remains available.','',
        '| Strategy | Evaluated attempts | Nominally qualified | Minimum F | Maximum F |','|---|---:|---:|---:|---:|']
    if initializations:
        for label in dict.fromkeys(row['strategy'] for row in initializations):
            rows=[r for r in initializations if r['strategy']==label];values=[float(r['fidelity']) for r in rows if r['fidelity']]
            initial.append(f"| {label} | {len(rows)} | {sum(r['qualified']=='True' for r in rows)} | {min(values):.10f} | {max(values):.10f} |")
    else:initial+=['Initialization experiment unexecuted.']
    initial+=['','Qualification frequency is a descriptive initialization result. It does not establish optimization reliability across unseen devices. Rejected attempts and coefficients/hashes are preserved in initialization_results.csv. Its production_channel_solves field for qualified starts is the total for that seed bank, repeated for provenance; it must not be summed across rows.']
    (out/'initialization_study.md').write_text('\n'.join(initial)+'\n')
    table=['| Regime | Method | Seed | Nominal F | Mean selection F | Selection P05 | Selection CVaR95 | Mean selection leakage | Selection feasible | Converged |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in results:table.append(f"| {r['regime']} | {r['method']} | {r['seed']} | {r['nominal_fidelity']:.10f} | {r['mean_fidelity']:.10f} | {r['p05_fidelity']:.10f} | {r['cvar95_infidelity']:.9g} | {r['mean_leakage']:.9g} | {r['selection_feasible']} | {r['converged']} |")
    dragpath=out/'drag_selection.json'
    if dragpath.exists():
        drag=json.loads(dragpath.read_text());s=drag['statistics']
        table.append(f"| shared | calibrated DRAG | — | {drag['nominal']['fidelity']:.10f} | {s['mean_fidelity']:.10f} | {s['p05_fidelity']:.10f} | {s['cvar95_infidelity']:.9g} | {s['mean_leakage']:.9g} | {s['worst_sample_leakage']<=base['quality_targets']['leakage_max']} | calibration reference |")
    reference=json.loads((out/'shared_reference.json').read_text())
    calibration=reference.get('calibration') or {}
    reference_limit=(f"The reused reference nominal calibration used {calibration.get('objective_calls','unknown')} calls and reported optimizer_success={calibration.get('optimizer_success','unknown')}. Its pulse is nominally qualified, but a budget-limited calibration is not an optimized-to-convergence DRAG baseline. Improvements relative to this reference do not establish an algorithmic advantage over fully calibrated DRAG. All optimized methods share this exact reference; risk-versus-conventional differences remain conditional on this start.\n\n")
    differences=['| Regime | Seed | Risk method | Conventional CVaR minus method CVaR | Mean F difference | P05 difference | Nominal F difference | Mean leakage difference | Runtime difference (s) |',
                 '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:differences.append(f"| {r['regime']} | {r['seed']} | {r['method']} | {r['delta_cvar']:.9g} | {r['mean_fidelity_difference']:.9g} | {r['p05_fidelity_difference']:.9g} | {r['nominal_fidelity_difference']:.9g} | {r['mean_leakage_difference']:.9g} | {r['runtime_difference']:.3f} |")
    absent=[r for r in ('equal_objectives','equal_passes','equal_wall') if r not in c['regimes']]
    discussion=('Positive delta CVaR favors the named risk method. Every difference uses common selection devices; these observations support no statistical-superiority claim. Hyperparameters and method choice must not be presented as independently validated after this development study.\n\n'
        'The ledger includes all starts/stages and unsuccessful nominal qualification attempts. Equal-call budgets do not equalize dissipative work because risk gradients require two scenario passes. Equal-pass budgets count both surrogate passes and production nominal channels, which are not identical in FLOPs. Equal-wall budgets include fitting, qualification, gradients and monitoring, checked between indivisible solves. A single call can overshoot a wall deadline; actual costs are reported. Reference calibration and development-only gradient/initialization/selection studies are separate shared or diagnostic costs.\n\n'
        'Convergence requires optimizer success, constraint satisfaction, small local KKT gradient, small relative objective change and small coefficient change. Objective reduction alone is insufficient. Metrics at retained feasible improvements use production nominal fidelity/leakage and surrogate training quantities; selection metrics use production propagation. A retained proposal can be a line-search trial rather than an SLSQP major iterate.\n\n')
    unresolved=f"Unexecuted budget regimes in this profile: {absent or 'none'}. Missing trial checkpoints: {missing or 'none'}. "
    if c['mode']=='short_diagnostic':unresolved+='The full 150-iteration/1000-call, four-substep, three-seed intermediate experiment remains unexecuted in all three budget regimes. '
    unresolved+='No 2000-device validation was generated. Five-level/RWA noise extrapolation, ensemble-wide surrogate accuracy, larger-budget convergence, matched-cost variability and independent generalization remain limitations. No hardware results are claimed.'
    lines=['# Intermediate benchmark','',intro,budget,reference_limit,*table,'',*differences,'',discussion,unresolved]
    if comparisons:
        values=[r['delta_cvar'] for r in comparisons if r['method']=='continuation']
        if values:lines+=['',f"Continuation selection delta CVaR range: [{min(values):.9g}, {max(values):.9g}]. These are observations at the executed caps, not a population-level winner."]
    lines+=['','Decision: do not launch confirmatory validation based on this diagnostic. Finish the predeclared matched-budget development comparisons and numerical checks before freezing a final research protocol.']
    (out/'intermediate_benchmark.md').write_text('\n'.join(lines)+'\n')
    continuation_lines=['# Continuation CVaR study','',intro,f"Schedule: {c['continuation_schedule']}. The optimized, qualified waveform is transferred exactly to the next stage. Failed stages preserve the previous feasible waveform. All stages share the method-wide resource ledger; unused stage allocations roll forward. Final retention minimizes the final scheduled training objective over archived feasible improvements. No monotonic CVaR theorem is asserted.",'',
        '| Regime | Seed | Stage lambda | Status | Nominal F | Nominal leakage | Training CVaR | Converged | Stop |',
        '|---|---:|---:|---|---:|---:|---:|---|---|']
    for r in stages:continuation_lines.append(f"| {r['regime']} | {r['seed']} | {r['lambda_value']} | {r['status']} | {r.get('nominal_fidelity','—')} | {r.get('nominal_leakage','—')} | {r.get('training_cvar','—')} | {r['converged']} | {r['termination']} |")
    continuation_lines+=['','Stages can reduce their own changing objectives without reducing final CVaR. Read paired_selection_comparisons.csv for common-device outcomes. '+unresolved]
    (out/'continuation_cvar_study.md').write_text('\n'.join(continuation_lines)+'\n')
    figures(out,initializations,results,convergence,stages,costs)


def figures(out,initializations,results,convergence,stages,costs):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.tight_layout();fig.savefig(out/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    if initializations:
        labels=list(dict.fromkeys(r['strategy'] for r in initializations))
        for i,label in enumerate(labels):
            rows=[r for r in initializations if r['strategy']==label and r['fidelity']]
            ax.scatter(np.full(len(rows),i),[float(r['fidelity']) for r in rows],alpha=.65)
        ax.set_xticks(range(len(labels)),[l.replace('_','\n') for l in labels]);ax.axhline(.999,color='gray',ls='--',label='Unchanged nominal threshold');ax.legend()
    else:ax.text(.5,.5,'Initialization study unexecuted',ha='center',transform=ax.transAxes)
    ax.set_ylabel('Initial nominal fidelity');save(fig,'figure_1_initialization.png')
    metrics=[('nominal_fidelity','Production nominal F'),('nominal_leakage','Production nominal leakage'),
        ('mean_training_fidelity','Surrogate mean training F'),('cvar95_training_infidelity','Surrogate training CVaR95'),
        ('maximum_training_leakage','Surrogate max training leakage'),('peak_amplitude_mhz','Peak amplitude (MHz)'),
        ('gradient_norm','Own stage gradient norm'),('objective','Own stage objective'),('elapsed_seconds','Elapsed seconds'),('total_dissipative_passes','Total dissipative passes')]
    fig,axes=plt.subplots(5,2,figsize=(12,15))
    groups=list(dict.fromkeys((r['regime'],r['method'],r['seed']) for r in convergence))
    curves={}
    for group in groups:
        regime,method,seed=group
        rows=[r for r in convergence if (r['regime'],r['method'],r['seed'])==group]
        trial=json.loads((out/f'{regime}_{method}_{seed}.json').read_text());last=trial['history'][-1]
        retained=next(r for r in reversed(rows) if r['waveform_sha256']==trial['waveform_sha256'])
        # This is a terminal reporting checkpoint, not an invented accepted step.
        terminal={**retained,**last['training_metrics'],**{k:last[k] for k in
            ('objective_evaluations','total_dissipative_passes','elapsed_seconds')},
            'objective':last['loss'],'nominal_fidelity':trial['nominal']['fidelity'],
            'nominal_leakage':trial['nominal']['leakage'],'event':'terminal retained waveform'}
        curves[group]=rows+[terminal]
    for ax,(metric,label) in zip(axes.flat,metrics):
        for regime,method,seed in groups:
            rows=curves[(regime,method,seed)]
            ax.plot([r['objective_evaluations'] for r in rows],[r[metric] for r in rows],'.-',color=COLORS[method],alpha=.7,label=f'{method}/{seed}/{regime}')
            ax.plot(rows[-1]['objective_evaluations'],rows[-1][metric],'s',ms=4,color=COLORS[method])
        ax.set_ylabel(label);ax.set_xlabel('Cumulative objective evaluations')
    if groups:axes.flat[0].legend(fontsize=6)
    fig.suptitle('Retained proposals and terminal checkpoints (squares); all attempted work counted',y=1.005)
    save(fig,'figure_2_convergence.png')
    for name,metric,label in [('figure_3_nominal_vs_cvar.png','nominal_fidelity','Nominal production fidelity'),
                              ('figure_4_leakage_vs_cvar.png','nominal_leakage','Nominal production leakage')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for regime,method,seed in groups:
            rows=[r for r in convergence if (r['regime'],r['method'],r['seed'])==(regime,method,seed)]
            ax.plot([r['cvar95_training_infidelity'] for r in rows],[r[metric] for r in rows],'.-',color=COLORS[method],alpha=.7,label=f'{method}/{seed}/{regime}')
        ax.set_xlabel('Surrogate training CVaR95 infidelity');ax.set_ylabel(label)
        if groups:ax.legend(fontsize=6)
        save(fig,name)
    fig,ax=plt.subplots(figsize=(7,4))
    markers={'conventional':'o','risk':'s','continuation':'^'}
    for method in dict.fromkeys(r['method'] for r in results):
        rows=[r for r in results if r['method']==method]
        ax.scatter([r['elapsed_seconds'] for r in rows],[r['cvar95_infidelity'] for r in rows],
            color=COLORS[method],marker=markers[method],alpha=.8,label=f'{method} ({len(rows)} trials)')
    if results:ax.legend(fontsize=8)
    ax.set_xlabel('Total optimizer time, all stages/starts (s)');ax.set_ylabel('Production selection CVaR95 infidelity');save(fig,'figure_5_cost_vs_robustness.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric,label in zip(axes,['training_cvar','nominal_fidelity','nominal_leakage'],['Training CVaR95','Nominal fidelity','Nominal leakage']):
        for index,(regime,seed) in enumerate(dict.fromkeys((r['regime'],r['seed']) for r in stages)):
            rows=[r for r in stages if r['seed']==seed and r['regime']==regime and metric in r]
            ax.plot([r['lambda_value'] for r in rows],[r[metric] for r in rows],marker='o',linestyle='-' if index==0 else '--',label=f'{regime}/{seed}')
        ax.set_xlabel('Stage lambda');ax.set_ylabel(label)
    if stages:axes[0].legend(title='Seed',fontsize=7)
    save(fig,'figure_6_continuation_stages.png')
