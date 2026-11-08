"""Read-only post-analysis of frozen corrective trials; never samples devices.

--wait holds no study lock until the worker releases it. --snapshot reports
incomplete progress explicitly. Analysis is separate from the frozen runner.
"""
import argparse,fcntl,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from .challenge_model import Device,hardware_metrics
from .grape_research import waveform_hash
from .grape_development import amplitude_headroom,write_csv,write_json
from .full_matched_report import all_histories
from .robust_optimization import summarize_trials

OUT=Path('results/binding_budget_v2')
METHODS=('conventional','risk','continuation')
REGIMES=('equal_objectives','equal_passes','equal_wall')
COMPARISONS=(('conventional','risk'),('conventional','continuation'),('risk','continuation'))
MATCH_TOLERANCE=.05

def expected_stems():
    return [f'trial_{r}_{l}_{s}_{m}' for r,l,s,m in itertools.product(REGIMES,(0,1),(7500,7501,7502),METHODS)]

def matching(a,b):
    """Descriptive 5% tolerance on realized resource; no statistical test."""
    ratio=max(a['actual_resource'],b['actual_resource'])/min(a['actual_resource'],b['actual_resource'])
    ok=ratio<=1+MATCH_TOLERANCE and min(a['utilization'],b['utilization'])>=.9
    ok=ok and max(a['overshoot_fraction'],b['overshoot_fraction'])<=MATCH_TOLERANCE
    return bool(ok),float(ratio)

def nondominated(points):
    p=np.asarray(points,dtype=float)
    return np.array([not any(np.all(q<=x) and np.any(q<x) for q in p) for x in p])

def load_and_verify(out):
    protocol=json.loads((out/'protocol.json').read_text());base=protocol['base_config'];device=Device(**base['device'])
    checks={};errors=[]
    checks['frozen_sources']=all(Path(f).is_file() and hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in protocol['source_hashes'].items())
    manifest=json.loads((out/'completed_experiment_manifest.json').read_text())
    checks['original_experiment_unchanged']=all(Path(f).is_file() and hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in manifest.items())
    for label,count in [('training',110),('selection',200)]:
        f=out/f'{label}_devices.json';old=Path('results/full_matched_budget')/f.name
        checks[label+'_ensemble_unchanged']=f.read_bytes()==old.read_bytes() and len(json.loads(f.read_text()))==count
    pending_checks={}
    for path in out.glob('trial_*_optimization.json'):
        d=json.loads(path.read_text());wave=np.asarray(d['waveform']);h=d['history'][-1]
        pending_checks[path.name]=bool(wave.shape[0]==3 and np.isfinite(wave).all() and waveform_hash(wave)==h['final_waveform_sha256'])
    checks['all_optimization_checkpoints_valid']=all(pending_checks.values())
    runs=[];rows=[];resources=[];details={}
    for stem in expected_stems():
        path=out/f'{stem}.json'
        if not path.exists():continue
        r=json.loads(path.read_text());h=r['history'][-1];wave=np.asarray(r['waveform']);o=r['options'];key=next(iter(r['caps']));cap=r['caps'][key]
        rowchecks={}
        rowchecks['waveform_hash']=waveform_hash(wave)==r['waveform_sha256']==h['final_waveform_sha256']
        saved=json.loads((out/f'{stem}_optimization.json').read_text())
        rowchecks['optimization_checkpoint']=np.array_equal(saved['waveform'],r['waveform']) and saved['history']==r['history']
        rowchecks['protocol_settings']=o['harmonics']==5 and o['substeps']==4 and o['restarts']==5 and o['seed']==r['seed'] and o['quality_targets']==base['quality_targets']
        rowchecks['budget_level']=cap=={'equal_objectives':[24,48],'equal_passes':[5280,10560],'equal_wall':[120,240]}[r['regime']][r['level']]
        rowchecks['pass_ledger']=h['total_dissipative_passes']==h['surrogate_scenario_passes']+h['production_channel_solves']
        histories=all_histories(h);traced=sum(len(x['evaluation_trace']) for x in histories)
        # A deadline may interrupt nominal retention after a counted gradient;
        # the ledger still charges it, but no completed trace record is emitted.
        rowchecks['trace_not_overcounted']=traced<=h['objective_evaluations']
        rowchecks['iteration_guard_nonbinding']=h['solver_iterations']<100000 and all('Iteration limit reached' not in x['message'] for x in histories)
        actual=h[key];rowchecks['hard_cap']=r['regime']=='equal_wall' or actual<=cap
        rowchecks['gpu_time_scope']=0<=h['gpu_backend_wall_seconds']<=h['elapsed_seconds']+1e-5
        hw=hardware_metrics(*wave,device)
        rowchecks['physical_constraints']=bool(wave.shape[0]==3 and np.isfinite(wave).all() and
            np.all(abs(wave[1:,[0,-1]])<=1e-10) and np.allclose(np.diff(wave[0]),.2,rtol=0,atol=1e-10) and
            abs(wave[0,-1]-15.6)<=1e-10 and hw['peak_amplitude_mhz']*amplitude_headroom(base)<=80+1e-8)
        rowchecks['nominal_qualified']=r['nominal']['fidelity']>=.999 and r['nominal']['leakage']<=1e-4
        rowchecks['numerical_accuracy']=bool(r['numerical']['passed'])
        channels=[json.loads((out/'selection_channels'/f'{r["waveform_sha256"]}_{i}.json').read_text()) for i in range(200)]
        statistics=summarize_trials(channels,base['quality_targets'])
        rowchecks['selection_aggregation']=all(abs(statistics[k]-r['selection']['statistics'][k])<1e-12 for k in ('mean_fidelity','p05_fidelity','cvar95_infidelity','mean_leakage','worst_sample_leakage'))
        terms=[x['message'] for x in histories];last_termination=terms[-1].split('; ')[-1];budget_stop='allocation exhausted' in last_termination or 'budget exhausted' in last_termination
        terminal='CONVERGED' if h['converged'] else 'RESOURCE_LIMIT' if budget_stop else 'SOLVER_STOPPED_NOT_CONVERGED'
        details[stem]=dict(checks=rowchecks,passed=all(rowchecks.values()),traced_evaluations=traced,
            counted_without_completed_trace=h['objective_evaluations']-traced,numerical=r['numerical'])
        if not all(rowchecks.values()):errors.append(stem)
        stats=r['selection']['statistics'];nom=r['nominal']
        row=dict(trial=stem,regime=r['regime'],level=r['level'],seed=r['seed'],method=r['method'],waveform_sha256=r['waveform_sha256'],
            nominal_fidelity=nom['fidelity'],nominal_leakage=nom['leakage'],
            **{k:stats[k] for k in ('mean_fidelity','p05_fidelity','cvar95_infidelity','mean_leakage','worst_sample_leakage')},
            peak_amplitude_mhz=hw['peak_amplitude_mhz'],amplitude_with_headroom_mhz=hw['peak_amplitude_mhz']*amplitude_headroom(base),
            objective_calls=h['objective_evaluations'],scenario_passes=h['total_dissipative_passes'],optimizer_seconds=h['elapsed_seconds'],
            gpu_backend_seconds=h['gpu_backend_wall_seconds'],actual_resource=actual,resource_cap=cap,utilization=actual/cap,
            overshoot_fraction=max(0.,actual/cap-1),converged=h['converged'],termination_class=terminal,
            termination=json.dumps(terms),numerical_pass=r['numerical']['passed'],checkpoint_verified=all(rowchecks.values()),
            engineering_selection_eligible=bool(rowchecks['nominal_qualified'] and rowchecks['physical_constraints'] and stats['worst_sample_leakage']<=1e-4 and stats['p05_fidelity_lcb95']>=.995))
        runtime_gates=sum(1 for x in histories if x.get('acceleration',{}).get('accuracy_gate',{}).get('passed'))
        nominal_solves=h['production_channel_solves']-3*runtime_gates
        row['cpu_nominal_solve_count']=nominal_solves
        res=dict(trial=stem,regime=r['regime'],level=r['level'],seed=r['seed'],method=r['method'],resource=key,cap=cap,
            actual=actual,utilization=actual/cap,overshoot=max(0.,actual-cap),termination_class=terminal,
            objective_gradient_calls=h['objective_evaluations'],surrogate_scenario_passes=h['surrogate_scenario_passes'],
            total_scenario_passes=h['total_dissipative_passes'],production_channel_solves=h['production_channel_solves'],
            nominal_production_solves=nominal_solves,runtime_gate_production_solves=3*runtime_gates,
            optimizer_seconds=h['elapsed_seconds'],external_optimizer_seconds=r['external_optimizer_seconds'],
            synchronized_gpu_backend_seconds=h['gpu_backend_wall_seconds'],
            non_gpu_optimizer_seconds_upper_bound=max(0.,h['elapsed_seconds']-h['gpu_backend_wall_seconds']),
            cpu_nominal_wall_seconds=None,cpu_timing_status='Not separately instrumented in frozen protocol; residual is an upper bound including orchestration and CPU surrogate checks',
            selection_seconds=r['selection']['selection_elapsed_seconds'],selection_new_solves=r['selection']['new_selection_channel_solves'],
            numerical_verification_seconds=r['numerical']['verification_seconds'],converged=h['converged'],numerical_pass=r['numerical']['passed'])
        rows.append(row);resources.append(res);runs.append(r)
    return protocol,checks,errors,runs,rows,resources,details

def paired_rows(rows):
    result=[]
    for regime,level,seed in itertools.product(REGIMES,(0,1),(7500,7501,7502)):
        group={r['method']:r for r in rows if (r['regime'],r['level'],r['seed'])==(regime,level,seed)}
        for first,second in COMPARISONS:
            if first not in group or second not in group:continue
            a,b=group[first],group[second];matched,ratio=matching(a,b)
            result.append(dict(regime=regime,level=level,seed=seed,reference=first,candidate=second,
                delta_cvar=a['cvar95_infidelity']-b['cvar95_infidelity'],percent_cvar_improvement=100*(a['cvar95_infidelity']-b['cvar95_infidelity'])/a['cvar95_infidelity'],
                mean_fidelity_change=b['mean_fidelity']-a['mean_fidelity'],p05_fidelity_change=b['p05_fidelity']-a['p05_fidelity'],
                nominal_fidelity_change=b['nominal_fidelity']-a['nominal_fidelity'],nominal_leakage_change=b['nominal_leakage']-a['nominal_leakage'],
                mean_leakage_change=b['mean_leakage']-a['mean_leakage'],maximum_leakage_change=b['worst_sample_leakage']-a['worst_sample_leakage'],
                realized_resource_ratio=ratio,comparable_actual_resource=matched,
                both_engineering_eligible=a['engineering_selection_eligible'] and b['engineering_selection_eligible'],
                objective_call_ratio=b['objective_calls']/a['objective_calls'],scenario_pass_ratio=b['scenario_passes']/a['scenario_passes'],
                wall_time_ratio=b['optimizer_seconds']/a['optimizer_seconds'],statistical_superiority_established=False))
    return result

def figures(out,rows,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'pdf.fonttype':42})
    folder=out/'publication_figures';folder.mkdir(exist_ok=True)
    colors=dict(conventional='#0072B2',risk='#D55E00',continuation='#009E73')
    markers={7500:'o',7501:'s',7502:'^'}
    note='Development only · 3 optimizer seeds · same 200 selection devices · bars: seed range, not a confidence interval'
    def save(fig,name):
        fig.text(.5,.015,note,ha='center',fontsize=8);fig.subplots_adjust(left=.12,right=.98,bottom=.27,top=.82,wspace=.30)
        for suffix in ('pdf','png','svg'):fig.savefig(folder/f'{name}.{suffix}')
        plt.close(fig)
    def panels(xfield,yfield,name,xlabel,ylabel,logx=False):
        fig,axes=plt.subplots(1,3,figsize=(13.5,4.4),sharey=True)
        fig.suptitle('Kairos development: '+ylabel)
        for ax,regime in zip(axes,REGIMES):
            ax.set_title(regime.replace('equal_','Budget: ').replace('_',' '),y=1.03);ax.set_xlabel(xlabel)
            for method in METHODS:
                for seed in (7500,7501,7502):
                    rr=sorted([r for r in rows if (r['regime'],r['method'],r['seed'])==(regime,method,seed)],key=lambda r:r['level'])
                    if rr:ax.plot([r[xfield] for r in rr],[r[yfield] for r in rr],marker=markers[seed],color=colors[method],alpha=.45,lw=.9,ms=4)
                for level in (0,1):
                    rr=[r for r in rows if (r['regime'],r['method'],r['level'])==(regime,method,level)]
                    if len(rr)==3:
                        xs=np.array([r[xfield] for r in rr]);ys=np.array([r[yfield] for r in rr]);xm=xs.mean();ym=ys.mean()
                        ax.errorbar(xm,ym,xerr=[[max(0.,xm-xs.min())],[max(0.,xs.max()-xm)]],yerr=[[max(0.,ym-ys.min())],[max(0.,ys.max()-ym)]],fmt='D',color=colors[method],ms=5,capsize=3)
            ax.grid(alpha=.2)
            if logx:ax.set_xscale('log')
            else:
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,4))
        axes[0].set_ylabel(ylabel)
        for method in METHODS:axes[-1].plot([],[],color=colors[method],label=method)
        axes[-1].legend(fontsize=8);save(fig,name)
    panels('objective_calls','cvar95_infidelity','01_cvar_vs_calls','Actual joint objective-gradient calls','CVaR95 infidelity')
    panels('scenario_passes','cvar95_infidelity','02_cvar_vs_passes','Actual total scenario-pass units','CVaR95 infidelity')
    panels('optimizer_seconds','cvar95_infidelity','03_cvar_vs_wall','Actual optimizer wall time (s)','CVaR95 infidelity')
    # Each column's x axis is the resource for that named regime.
    panels('actual_resource','p05_fidelity','04_p05_vs_cost','Actual regime resource (calls / passes / s)','P05 fidelity')
    panels('mean_leakage','cvar95_infidelity','05_cvar_vs_mean_leakage','Mean selection leakage','CVaR95 infidelity')
    panels('worst_sample_leakage','cvar95_infidelity','06_cvar_vs_max_leakage','Maximum sampled selection leakage','CVaR95 infidelity')
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.4),sharey=True);fig.suptitle('Kairos development: paired CVaR95 improvements')
    for ax,comp in zip(axes,COMPARISONS):
        names=dict(conventional='conventional',risk='fixed risk',continuation='continuation')
        ax.set_title(names[comp[1]]+' vs '+names[comp[0]],y=1.03,fontsize=10);ax.axhline(0,color='.4',lw=.8)
        for i,(regime,level) in enumerate(itertools.product(REGIMES,(0,1))):
            pp=[p for p in pairs if (p['regime'],p['level'],p['reference'],p['candidate'])==(regime,level,*comp)]
            for p in pp:
                ax.scatter(i+.09*(p['seed']-7501),p['delta_cvar'],marker=markers[p['seed']],color=colors[comp[1]],facecolors=colors[comp[1]] if p['comparable_actual_resource'] else 'none',s=35)
            ax.set_xticks(range(6),['calls L','calls H','passes L','passes H','wall L','wall H'],rotation=45)
        ax.grid(alpha=.2);ax.set_xlim(-.35,5.35);ax.set_xlabel('Hollow: costs differ >5% or low utilization',fontsize=9)
    axes[0].set_ylabel('ΔCVaR95 (positive favors candidate)');save(fig,'07_paired_improvements')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True);fig.suptitle('Kairos development: descriptive robustness–leakage Pareto set')
    for ax,field,label in zip(axes,('mean_leakage','worst_sample_leakage'),('Mean leakage','Maximum sampled leakage')):
        p=np.array([[r[field],r['cvar95_infidelity']] for r in rows]);nd=nondominated(p) if len(p) else []
        for method in METHODS:
            rr=[r for r in rows if r['method']==method];ax.scatter([r[field] for r in rr],[r['cvar95_infidelity'] for r in rr],color=colors[method],s=22,label=method,alpha=.65)
        if len(p):
            front=p[nd];front=front[np.argsort(front[:,0])];ax.plot(front[:,0],front[:,1],'k--',lw=1,label='Observed nondominated set')
        if field=='worst_sample_leakage':ax.axvline(1e-4,color='firebrick',ls=':',label='Leakage limit')
        ax.set_title(label,y=1.03);ax.set_xlabel(label);ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[0].set_ylabel('CVaR95 infidelity');axes[-1].legend(fontsize=8);save(fig,'08_pareto')
    (folder/'FIGURE_NOTES.md').write_text('All figures show development results on one shared 200-device ensemble. Lines connect budget levels within the same optimizer seed. Circles/squares/triangles identify seeds 7500/7501/7502. Diamonds and bars show means and min–max over three seeds, not device confidence intervals. Empty cells are unfinished, never zero. Pareto lines are descriptive across unequal costs and are not a matched-efficiency frontier. The leakage acceptance line applies only to maximum leakage. The 5% realized-cost comparability rule is a descriptive analysis rule introduced after three low-call results were observed, not a prespecified statistical hypothesis.\n')

def draft_protocol(out,protocol,candidates):
    """Freeze proposed data-free validation decisions; no RNG or device draw."""
    write_json(out/'confirmatory_frozen_candidates.json',candidates)
    extra=['statistical_certification.py','research_statistics.py','corrective_study_analysis.py']
    hashes={**protocol['source_hashes'],**{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in extra}}
    write_json(out/'confirmatory_protocol_manifest.json',dict(status='DRAFT_NOT_EXECUTED',validation_seed=941206,
        validation_devices=2000,bootstrap_seed=941207,bootstrap_repetitions=50000,source_hashes=hashes,
        base_config=protocol['base_config'],candidate_hashes={m:c['waveform_sha256'] for m,c in candidates.items()},
        validation_generated=False,validation_execution_authorized=False,
        numerical_repeat_indices=np.linspace(0,1999,20,dtype=int).tolist()))
    text='''# Independent confirmatory protocol draft — NOT EXECUTED

This draft does not authorize validation execution. Freeze and review this document,
candidate manifest, source hashes and solver configuration before generating any devices.
No independent validation devices have been drawn or inspected by this analysis.

## Population, estimand and frozen candidates

Generate exactly 2000 iid devices once, with previously unused seed **941206**,
using the unchanged `challenge_model.monte_carlo_scenarios` function. Freeze its
source hash from the corrective protocol and use the existing independent uniform
amplitude factor [0.97,1.03], detuning [-3,3] MHz, anharmonicity shift [-8,8] MHz,
and four independent lifetime scale factors [0.8,1]. These are modeling assumptions.
All methods receive exactly the same device rows. Do not generate these rows now.
The device is the sampling unit; method results on a device are paired.

`confirmatory_frozen_candidates.json` contains one development-selected feasible
waveform per method, each SHA256, exact samples, originating trial and frozen
optimizer options. Select minimum development CVaR95 among candidates meeting
all nominal, hardware, sampled leakage and development P05 criteria, breaking ties
lexicographically by trial name. Also freeze the strongest overall feasible candidate.
If any method has no eligible waveform, do not substitute a violating pulse or change
thresholds: revise the comparison family explicitly before locking this draft.
The final comparisons estimate differences between these specific fixed pulses,
not average performance of an optimizer or computational-efficiency superiority.
No algorithm, hyperparameter, waveform, stopping rule or candidate is selected
after seeing independent data. Archive original candidate arrays/options/hashes.

## Endpoints and analysis

Primary estimand for each frozen pair: Δ = CVaR95(reference infidelity) −
CVaR95(candidate infidelity). Positive favors the candidate. Use the existing exact
empirical upper-tail CVaR with fractional boundary weights (`empirical_cvar`), not
smoothed training CVaR. At n=2000 the tail contains approximately 100 devices.
The family consists of all three planned contrasts: fixed risk vs conventional,
continuation vs conventional, continuation vs fixed risk. Include every contrast
regardless of observed direction; do not drop unfavorable results.

Use 50000 paired nonparametric bootstrap replicates, resampling 2000 device indices
with replacement with bootstrap seed 941207. Use each index vector jointly for all
methods and contrasts; recompute empirical CVaR per method in each replicate.
The ordinary paired percentile bootstrap yields approximate device-sampling CIs;
it is not an exact distribution-free CVaR guarantee. Report point estimates,
unadjusted two-sided 95% CIs and simultaneous Bonferroni two-sided 98.333333% CIs
for the three primary contrasts (quantiles .05/(2*3), 1−.05/(2*3)). Declare tail
improvement only if the simultaneous lower bound is >0 and numerical sensitivity
checks cannot reverse its sign. No favorable practical-effect margin is invented.
If the interval includes zero, report inconclusive; if its upper bound is <0,
report inferiority. Report bootstrap Monte Carlo sensitivity with 100000 replicates
using the same fixed resample stream extended; disagreement means inconclusive.
Do not try multiple bootstrap methods and select a favorable one.

Secondary endpoints: paired mean fidelity, paired P05 fidelity, nominal fidelity,
mean leakage, maximum sampled leakage and leakage exceedance count. Bootstrap
mean/P05/mean leakage using the same paired device indices, with descriptive 95%
CIs explicitly labeled secondary, without confirmatory superiority declarations.
Sample maximum leakage and exceedances are observations, not global worst-case
bounds; do not use a bootstrap CI on the maximum as safety certification.
Optimizer-seed spread in development remains descriptive: do not resample or pool
54 trials as independent device ensembles, nor use a three-seed bootstrap to claim
population-level algorithm superiority. This frozen-waveform confirmatory analysis
resamples devices only, unlike a future algorithm-level crossed-seed experiment.

Paired resampling semantics follow the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

## Engineering and numerical criteria

Keep nominal fidelity ≥.999, nominal and maximum sampled leakage ≤1e-4, amplitude
≤80 MHz including 1.03 manufacturing headroom, zero endpoints and 15.6 ns duration.
Preserve the project's `statistical_certification.tail_bound_details`: sorted
fidelities, k = binom.ppf(.05,n,.05), one-sided 95% lower bound F_(k), requiring
bound ≥.995 with nontrivial support. Report the existing per-pulse certificate
unchanged. For a simultaneous engineering claim across three pulses additionally
compute the same binomial order-statistic method at confidence 1−.05/3; require
all corresponding bounds ≥.995. This strengthens multiplicity treatment without
weakening the existing method or threshold. A positive CVaR contrast does not
override an engineering failure. No convergence requirement is imposed on a
valid resource-limited optimizer; its achieved status stays reported.

Use trusted CPU QuTiP for every validation channel and all final certification.
Before drawing devices, recheck nominal and saved training stress cases with
production strict/tighter solver settings and four/eight interpolation substeps;
maintain fidelity/leakage tolerance 2e-6, gradient atol2e-7/rtol2e-6, trace and
Hermiticity errors ≤2e-6 and minimum density eigenvalue ≥−2e-6. A gate failure
stops validation; do not switch precision or relax thresholds. Record all tolerances
from source snapshot. Freeze an evenly spaced set of 20 future device indices
(0,105,...,1890,1999 if specified exactly in the locked implementation) for repeat
strict/tighter CPU propagation; specify the exact 20 indices before locking.
If a discrepancy exceeds 2e-6, stop and investigate; any rerun policy must be
frozen before outcomes and reported. Do not selectively replace bad devices.
Hilbert truncation and solver tolerances must pass the project's existing convergence
checks: five-level numerical equivalence does not prove convergence in level count.
Prior evidence must cover selected waveforms; otherwise run development-only
5/6-level checks first, without inspecting validation draws. No new physical noise
model or uncertainty distribution is introduced by this draft.

## Reproducibility and reporting

Archive data-independent source/config hashes, waveform hashes, seed declarations,
exact statistics code and version, package/hardware environment and full analysis
plan before execution. Log all CPU solves, retries and failures. Preserve nominal,
public-grid, truncation and adversarial diagnostics required by the project, using
only frozen diagnostic rules and keeping them separate from statistical sampling.
Report negative and mixed results. Seek independent implementation/hardware
replication before journal claims extending beyond this modeled distribution.
'''
    # Exact set is specified deterministically here without any scenario draw.
    indices=np.linspace(0,1999,20,dtype=int).tolist()
    text=text.replace('(0,105,...,1890,1999 if specified exactly in the locked implementation)',str(indices)).replace('specify the exact 20 indices before locking.','These exact indices are frozen in this draft.')
    (out/'confirmatory_protocol_draft.md').write_text(text)

def analyze(out=OUT,make_figures=True):
    protocol,checks,errors,runs,rows,resources,details=load_and_verify(out)
    missing=[s for s in expected_stems() if not (out/f'{s}.json').exists()];complete=not missing
    pairs=paired_rows(rows)
    if rows:write_csv(out/'matched_cost_results.csv',rows);write_csv(out/'resource_utilization.csv',resources)
    if pairs:write_csv(out/'paired_method_comparisons.csv',pairs)
    write_json(out/'numerical_verification.json',dict(completed=len(rows),required=54,all_completed_checks_pass=bool(rows) and not errors and all(checks.values()),
        global_checks=checks,invalid_trials=errors,trial_checks=details,
        inherited_finite_difference_validation=json.loads(Path('results/full_matched_budget/gradient_verification.json').read_text()),
        finite_difference_scope='Inherited 100 saved training-device directional checks; numerical/model sources unchanged. New final-waveform checks compare CPU/GPU analytic gradients on three cases, not a new 100-device finite-difference campaign.',
        scope='Every completed waveform: nominal and two saved deterministic training stress cases; CPU production vs GPU FP64 at 4/8 substeps, gradients, physicality. Not all 200 devices, not new validation, not a Hilbert-level convergence proof.'))
    stagecost=[];restartcost=[]
    for r in runs:
        stem=f"trial_{r['regime']}_{r['level']}_{r['seed']}_{r['method']}";h=r['history'][-1]
        for s in h.get('stages',[]):stagecost.append(dict(trial=stem,stage=s['stage'],lambda_value=s['lambda_value'],status=s['status'],converged=s['converged'],termination=s['termination'],**s['incremental_cost']))
        for stage,x in enumerate(all_histories(h)):
            for restart in sorted({t['restart'] for t in x['evaluation_trace']}):
                tt=[t for t in x['evaluation_trace'] if t['restart']==restart]
                restartcost.append(dict(trial=stem,stage=stage,restart=restart,traced_calls=len(tt),first_trace_elapsed=tt[0]['elapsed_seconds'],last_trace_elapsed=tt[-1]['elapsed_seconds'],
                    span_seconds=tt[-1]['elapsed_seconds']-tt[0]['elapsed_seconds'],exact_restart_cost_available=False,
                    scope='Trace span excludes first call and pre-trace qualification; not an exact restart timer or complete allocation cost'))
    if stagecost:write_csv(out/'continuation_stage_costs.csv',stagecost)
    if restartcost:write_csv(out/'restart_trace_spans.csv',restartcost)
    interrupted=[]
    for f in (out/'interrupted_attempts').glob('*.json'):
        d=json.loads(f.read_text());interrupted.append(dict(file=str(f),**{k:d.get(k) for k in ('objective_evaluations','total_dissipative_passes','elapsed_seconds')}))
    opt_pending=[f.stem for f in out.glob('trial_*_optimization.json') if not (out/(f.stem.removesuffix('_optimization')+'.json')).exists()]
    costs=dict(successful_optimizer_seconds=sum(r['optimizer_seconds'] for r in rows),successful_gpu_backend_seconds=sum(r['gpu_backend_seconds'] for r in rows),
        successful_calls=sum(r['objective_calls'] for r in rows),successful_passes=sum(r['scenario_passes'] for r in rows),
        selection_seconds=sum(r['selection_seconds'] for r in resources),numerical_verification_seconds=sum(r['numerical_verification_seconds'] for r in resources),
        shared_setup=json.loads((out/'qualification.json').read_text()) if (out/'qualification.json').exists() else None,
        interrupted_lower_bounds=interrupted,optimization_complete_selection_pending=opt_pending,
        cost_limitations=['Exact CPU nominal wall and full restart timers were not recorded; solve counts and non-GPU residual upper bounds are available.',
            'GPU backend seconds include synchronized dispatch/transfers/JIT; not CUDA kernel-only time.',
            'Successful-trial totals exclude in-progress trials and interrupted work; in-flight losses may exceed saved lower bounds.'])
    write_json(out/'campaign_cost_accounting.json',costs)
    candidates={}
    for method in METHODS:
        eligible=[r for r in rows if r['method']==method and r['engineering_selection_eligible'] and r['numerical_pass'] and r['checkpoint_verified']]
        if eligible:
            best=min(eligible,key=lambda r:(r['cvar95_infidelity'],r['trial']));run=next(r for r in runs if f"trial_{r['regime']}_{r['level']}_{r['seed']}_{r['method']}"==best['trial'])
            candidates[method]=dict(trial=best['trial'],waveform_sha256=best['waveform_sha256'],waveform=run['waveform'],options=run['options'],selection_cvar95=best['cvar95_infidelity'])
    strongest=min(candidates,key=lambda m:candidates[m]['selection_cvar95']) if candidates else None
    intact=all(checks.values()) and not errors
    numerical=all(r['numerical_pass'] for r in rows)
    decision='NO-GO'
    reason='Incomplete corrective development study; do not start independent validation.'
    if complete and intact and numerical:
        decision='CONDITIONAL GO'
        reason='Complete verified budget-limited study; freeze confirmatory candidates and protocol after development-only Hilbert truncation checks and acknowledging approximate wall matching and timing limitations.'
        if len(candidates)!=3:reason='Study complete, but one or more methods lack an engineering-eligible waveform; comparison family must be explicitly revised before freezing confirmation.'
    elif complete:reason='Material integrity, checkpoint or numerical errors remain.'
    readiness=dict(decision=decision,reason=reason,completed_trials=len(rows),required_trials=54,complete=complete,missing_trials=missing,
        integrity_checks=checks,invalid_trials=errors,numerical_checks_passed=numerical,converged_trials=sum(r['converged'] for r in rows),
        statistical_superiority_established=False,strongest_feasible_development_method=strongest if complete else None,
        provisional_feasible_leader=strongest,eligible_methods=list(candidates),validation_generated=False,validation_execution_authorized=False,
        actual_cost_comparable_pairs=sum(p['comparable_actual_resource'] for p in pairs),pairs_available=len(pairs),
        descriptive_cost_tolerance=MATCH_TOLERANCE,
        analysis_rule_timing='5% comparability tolerance introduced after first three low-call trials were observed; exploratory, not a preregistered hypothesis',
        resume_command="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/kairos-mpl XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_cpu_multi_thread_eigen=false .venv/bin/python -u binding_budget_study.py")
    lines=['# Corrective study development report','',f'**{len(rows)}/54 complete. {decision}.** {reason}', '',
        'The original 27-trial experiment is '+('unchanged' if checks['original_experiment_unchanged'] else 'NOT verified unchanged')+'. Frozen protocol and saved ensembles are checked independently. No independent validation devices were generated.', '',
        'Budget levels 0/1 are respectively 24/48 joint objective-gradient calls, 5280/10560 total scenario-pass units, and 120/240 optimizer seconds. All runs use the same 110 training and 200 selection scenarios, five Fourier harmonics, five-level Lindblad model and unchanged physical/quality constraints.', '',
        '## Direct answers','',
        '1–3. Fixed risk versus conventional, continuation versus conventional, and continuation versus fixed risk are tabulated below. Positive ΔCVaR favors the candidate. These are three paired optimizer seeds evaluated on the same 200 development devices, not independent replications of device sampling. Signs and seed ranges are descriptive; no inferential superiority is established.', '',
        '|Regime|Level|Candidate / reference|Seeds|Mean ΔCVaR|Seed range|Seeds favoring candidate|Actual-cost comparable pairs|', '|---|---:|---|---:|---:|---|---:|---:|']
    for regime,level,(first,second) in itertools.product(REGIMES,(0,1),COMPARISONS):
        pp=[p for p in pairs if (p['regime'],p['level'],p['reference'],p['candidate'])==(regime,level,first,second)]
        if pp:
            v=[p['delta_cvar'] for p in pp];lines.append(f'|{regime}|{level}|{second}/{first}|{len(v)}|{np.mean(v):.8g}|[{min(v):.8g}, {max(v):.8g}]|{sum(x>0 for x in v)}|{sum(p["comparable_actual_resource"] for p in pp)}|')
    lines+=['','Descriptive answers at comparable realized costs:']
    for first,second in COMPARISONS:
        pp=[p for p in pairs if (p['reference'],p['candidate'])==(first,second) and p['comparable_actual_resource']]
        pos=sum(p['delta_cvar']>0 for p in pp);eligible=sum(p['both_engineering_eligible'] for p in pp)
        signs='uniformly favorable in available pairs' if pp and pos==len(pp) else 'uniformly unfavorable in available pairs' if pp and pos==0 else 'mixed across available seeds/budgets' if pp else 'not yet assessable'
        lines.append(f'- {second} versus {first}: {pos}/{len(pp)} comparable pairs favor the candidate; {signs}. Both pulses meet development engineering eligibility in {eligible}/{len(pp)} pairs. This is descriptive, conditional on the named resource, and not a population superiority test.')
    lines+=['', 'Do not collapse budget regimes. Equal calls can involve roughly twice the pass cost for risk methods. The 5% realized-cost rule is a descriptive rule added after the first three low-call outcomes; it does not create statistical significance. Pairs with material cost mismatch are reported without an efficiency interpretation.', '',
        '4. Leakage trade-offs: see leakage_tradeoff_analysis.md and paired_method_comparisons.csv. Nominal qualification and ensemble leakage are different. All engineering thresholds stay unchanged.', '',
        f'5. Integrity checks: {checks}. Invalid completed trial checks: {errors}. Resource utilization, overshoot and exact termination strings are in resource_utilization.csv. A 100000-iteration guard is checked for nonbinding status. Nonpreemptive wall deadlines allow an in-flight operation to finish; utilization is not silently forced to 100%.', '',
        f'6. Numerical verification passed for {sum(r["numerical_pass"] for r in rows)}/{len(rows)} completed waveforms. Scope: nominal plus two existing training stress cases, 4/8 substeps, CPU/GPU pointwise metrics and gradients, density physicality and sampled waveform constraints. Not a uniform numerical bound, not independent validation, not convergence with Hilbert-space size.', '',
        f'7. Strongest feasible development method: {strongest if complete else "not selected until study completion"}. Selection uses the existing engineering criteria, then minimum development CVaR95. It selects a fixed waveform, not a universally superior optimization algorithm.', '',
        '## Computational costs and limitations','',
        f'Successful optimizer wall: {costs["successful_optimizer_seconds"]:.3f} s; synchronized GPU-backend wall: {costs["successful_gpu_backend_seconds"]:.3f} s; calls: {costs["successful_calls"]}; passes: {costs["successful_passes"]}. '
        f'Selection: {costs["selection_seconds"]:.3f} s; numerical checks: {costs["numerical_verification_seconds"]:.3f} s. Shared qualification/warmup and interrupted-work lower bounds are in campaign_cost_accounting.json.', '',
        'CPU nominal solve counts are recorded, but exact CPU nominal wall time was not separately instrumented. Wall minus synchronized GPU backend time is only an upper bound including orchestration and CPU surrogate checks. Exact restart timers are also unavailable; trace spans are lower bounds, and stage incremental ledgers are available. The frozen experiment is not modified to retrofit timing. These limits prevent kernel-level or exact CPU-time efficiency claims; total optimizer wall remains directly measured.', '',
        '## Reproducibility and remaining work','',
        'Verify the checkpoint and input hash manifest, complete all 54 trials, inspect all eight figures and unfavorable seeds, review feasibility, run selected-waveform development-only truncation/convergence checks, then freeze the draft protocol and source snapshots. Independent validation remains prohibited here. Journal claims need the frozen CPU-confirmatory results, clear model/distribution limitations, and external/hardware replication where appropriate.', '',
        'Resume command (only if no worker holds study.lock):', '', '```bash',readiness['resume_command'],'```', '',
        'Checkpoint files: trial_*_optimization.json preserve completed optimization before CPU selection; trial_*.json preserve complete checked trials. Unfinished progress files are archived as interrupted attempts before restarting their trial. Existing completed trials are reused. Final analysis: `.venv/bin/python corrective_study_analysis.py --wait`.', '',
        'No superiority or final readiness claim is issued from an incomplete experiment.']
    (out/'corrective_study_final_report.md').write_text('\n'.join(lines)+'\n')
    leak=['# Development fidelity–leakage trade-offs','',f'{len(rows)}/54 complete; one shared 200-device selection ensemble. Positive ΔCVaR favors the candidate; positive leakage change is worse. Nominal leakage is a retention qualification, not an ensemble guarantee. Threshold remains 1e-4 for nominal and maximum sampled leakage.','',
        '|Method|Completed|Nominal F range|Mean F range|CVaR range|Mean leakage range|Maximum leakage range|Eligible|','|---|---:|---|---|---|---|---|---:|']
    for method in METHODS:
        rr=[r for r in rows if r['method']==method]
        if rr:
            span=lambda k:f'[{min(r[k] for r in rr):.7g}, {max(r[k] for r in rr):.7g}]'
            leak.append(f'|{method}|{len(rr)}|{span("nominal_fidelity")}|{span("mean_fidelity")}|{span("cvar95_infidelity")}|{span("mean_leakage")}|{span("worst_sample_leakage")}|{sum(r["engineering_selection_eligible"] for r in rr)}|')
    leak+=['','Unfavorable observations (all comparisons reported, without cost-matching inference):']
    for p in pairs:
        if p['delta_cvar']<0 or p['maximum_leakage_change']>0 or p['nominal_fidelity_change']<0:
            leak.append(f'- {p["regime"]}, level {p["level"]}, seed {p["seed"]}, {p["candidate"]}/{p["reference"]}: ΔCVaR {p["delta_cvar"]:.7g}; nominal F change {p["nominal_fidelity_change"]:.7g}; mean leakage change {p["mean_leakage_change"]:.7g}; max leakage change {p["maximum_leakage_change"]:.7g}; wall ratio {p["wall_time_ratio"]:.4g}.')
    (out/'leakage_tradeoff_analysis.md').write_text('\n'.join(leak)+'\n')
    if make_figures and rows:figures(out,rows,pairs)
    if complete and decision in ('GO','CONDITIONAL GO'):draft_protocol(out,protocol,candidates)
    write_json(out/'development_readiness.json',readiness)
    write_json(out/'analysis_provenance.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_optimizer_sources_unchanged=checks['frozen_sources'],snapshot_incomplete=not complete,
        scope='Separate read-only post-analysis; no optimizer/config changes and no uncertainty draws',comparability_tolerance=MATCH_TOLERANCE))
    return readiness

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--snapshot',action='store_true');p.add_argument('--no-figures',action='store_true');args=p.parse_args()
    if not args.snapshot:
        lock=(OUT/'study.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX if args.wait else fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Study active; use --wait or --snapshot.')
    analysis_lock=(OUT/'analysis.lock').open('a');fcntl.flock(analysis_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        result=analyze(make_figures=not args.no_figures);print(json.dumps(result,indent=2),flush=True)
    except Exception as error:
        import traceback
        write_json(OUT/'analysis_error.json',dict(error=str(error),traceback=traceback.format_exc()))
        write_json(OUT/'development_readiness.json',dict(decision='NO-GO',complete=False,
            reason='Post-analysis failed; inspect analysis_error.json. No confirmation authorized.',
            validation_generated=False,validation_execution_authorized=False))
        raise

if __name__=='__main__':main()
