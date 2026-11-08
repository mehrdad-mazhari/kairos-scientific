"""Proposed analysis/report adapter; audited before any independent draws."""
import csv,json,itertools,time
from pathlib import Path
import numpy as np
from confirmatory_statistics import PRIMARY,PAIRS,paired_cvar_intervals
from risk_aware_grape import empirical_cvar
from statistical_certification import tail_bound_details
from grape_development import write_json,write_csv
from confirmatory_execution_v1 import result_path,verify_checkpoint,require,TOL,DIAGNOSTIC

def secondary_intervals(fidelity,leakage,repetitions=50000,seed=941207):
    """Replay the primary RNG prefix; device indices remain paired."""
    names=list(fidelity);n=len(fidelity[names[0]]);rng=np.random.default_rng(seed)
    sample={m:np.empty((repetitions,3)) for m in names};diff={p:np.empty((repetitions,3)) for p in PAIRS}
    for b in range(repetitions):
        idx=rng.integers(n,size=n);metrics={}
        for m in names:
            metrics[m]=np.array([fidelity[m][idx].mean(),np.quantile(fidelity[m][idx],.05),leakage[m][idx].mean()]);sample[m][b]=metrics[m]
        for a,c in PAIRS:diff[a,c][b]=metrics[c]-metrics[a]
    individual={m:np.quantile(v,[.025,.975],axis=0).T.tolist() for m,v in sample.items()}
    paired={a+'/'+c:np.quantile(v,[.025,.975],axis=0).T.tolist() for (a,c),v in diff.items()}
    return dict(individual=individual,paired=paired,metrics=['mean_fidelity','p05_fidelity','mean_leakage'],
        scope='Descriptive percentile 95% CIs; same index stream as primary50k, no secondary superiority tests')

def engineering_result(name,candidate,statistics,nominal,simultaneous,numerical_pass=True):
    hardware=candidate['hardware'];q=dict(nominal_fidelity=nominal['fidelity']>=.999,nominal_leakage=nominal['leakage']<=1e-4,
        p05_bound=statistics['lower_bound']>=.995,maximum_sampled_leakage=statistics['maximum_leakage']<=1e-4,
        amplitude=hardware['peak_amplitude_mhz']*1.03<=80+1e-8,endpoints=hardware['endpoint_amplitude_rad_ns']<=1e-10,
        duration=abs(hardware['duration_ns']-15.6)<=1e-10,numerical=bool(numerical_pass),
        development_hilbert=bool(candidate['hilbert_check']['five_level_pass']),frozen_primary_eligible=bool(candidate['allowed_primary_comparator'] and name in PRIMARY))
    passed=all(q.values());sim=bool(passed and simultaneous and simultaneous['lower_bound']>=.995)
    return dict(status='DIAGNOSTIC_ONLY' if name==DIAGNOSTIC else 'PASS' if passed else 'FAIL',checks=q,passed=passed,
        simultaneous_engineering_pass=sim,failures=[k for k,v in q.items() if not v],
        leakage_scope='Maximum over sampled devices of the six-input-state mean leakage; not a uniform/global bound')

def summarize_claim(row,sensitivity,engineering,observed_error):
    """Keep statistical direction distinct from finite numerical sensitivity."""
    ci=row['family_adjusted_ci'];observed=observed_error.get(row['reference'],0)+observed_error.get(row['candidate'],0)
    stable=row['direction']==sensitivity['direction']
    signstable=ci[0]>observed if row['direction']=='candidate improvement' else ci[1]<-observed if row['direction']=='candidate inferior' else True
    # The frozen text warns against resolved effects on the2e-6 scale but did
    # not supply a uniform CVaR error bound. Flag those for explicit review.
    point_small=abs(row['delta_cvar'])<=TOL
    near_boundary=row['direction']!='inconclusive' and min(abs(x) for x in ci)<=TOL
    unresolved=point_small or near_boundary or not signstable
    return dict(bootstrap_sensitivity_agrees=stable,observed_repeat_envelope=observed,
        sign_stable_under_observed_repeats=bool(signstable),
        statistical_direction=row['direction'],statistical_ci_excludes_zero=row['direction']!='inconclusive',
        bootstrap_sensitivity_status='STABLE' if stable else 'UNSTABLE',
        sensitivity_conclusion='DIRECTION_STABLE' if stable else 'INCONCLUSIVE_UNDER_FROZEN_BOOTSTRAP_SENSITIVITY_RULE',
        numerical_reliability='NUMERICALLY_UNRESOLVED' if unresolved else 'CHECKED_NUMERICALLY_RESOLVED',
        numerical_resolution_review_required=bool(unresolved),
        numerical_unresolved_reasons=[reason for reason,flag in [('point_difference_at_or_below_2e-6',point_small),('adjusted_interval_boundary_within_2e-6_of_zero',near_boundary),('observed_repeat_envelope_can_reverse_direction',not signstable)] if flag],
        superiority_claim_supported=bool(stable and not unresolved and row['direction']=='candidate improvement'),
        engineering_reference_pass=engineering[row['reference']]['passed'],
        engineering_candidate_pass=engineering[row['candidate']]['passed'],
        interpretation='Observed numerical repeats are finite sampled sensitivity evidence, not a uniform error bound. Resolution flags need review; no alternate inferential method is selected.')

def overall_reporting(contrasts,engineering):
    """Pre-data reporting convention only; individual contrast CIs are authoritative."""
    require(len(contrasts)==6,'All six contrasts required for an overall summary')
    require({(p['reference'],p['candidate']) for p in contrasts}==set(PAIRS),'Each prespecified contrast must appear exactly once')
    qualified=[p for p in contrasts if p['superiority_claim_supported'] and
        engineering[p['reference']]['simultaneous_engineering_pass'] and engineering[p['candidate']]['simultaneous_engineering_pass']]
    label='CONFIRMED' if len(qualified)==6 else 'PARTIALLY CONFIRMED' if qualified else 'NOT CONFIRMED'
    return dict(classification=label,qualified_favorable_contrasts=len(qualified),family_size=6,
        convention='Pre-data implementation interpretation: summary of the six directed favorable claims with stable bootstrap, resolved numerical assessment and simultaneous engineering passes for both pulses. Not an additional statistical hypothesis or original scientific rule.',
        individual_contrast_results_authoritative=True,
        publication_readiness=dict(manuscript_preparation='COMPLETE_SIMULATION_WITH_REPORTED_LIMITATIONS',q1_acceptance_claim=False,
            independent_of_favorable_outcomes=True),
        numerical_resolution_review_required=any(p['numerical_resolution_review_required'] for p in contrasts))

def analyze(out,cfg,candidates,cohort,audit):
    out=Path(out);names=list(candidates);fidelity={};leakage={};costs=[];integrity=[]
    for name,c in candidates.items():
        records=[json.loads(result_path(out,name,i).read_text()) for i in range(2000)]
        for i,r in enumerate(records):verify_checkpoint(r,name,i,c,cohort)
        fidelity[name]=np.array([r['metrics']['fidelity'] for r in records]);leakage[name]=np.array([r['metrics']['leakage'] for r in records])
        repeat_records=[json.loads(result_path(out,name,i,mode).read_text()) for i in np.linspace(0,1999,20,dtype=int).tolist() for mode in ('eight','mesolve_eight')]
        costs.append(dict(candidate=name,completed_device_evaluations=len(records),production_solver_seconds=sum(r['elapsed_seconds'] for r in records),numerical_repeat_solver_seconds=sum(r['elapsed_seconds'] for r in repeat_records),
            original_optimization_cost=json.dumps(c.get('complete_computational_cost',c['cost'])),scope='Saved successful production evaluations; historical optimization cost scopes differ'))
        integrity.append(dict(candidate=name,passed=all(r['metrics']['physicality_pass'] for r in records),
            maximum_trace_error=max(r['metrics']['max_trace_error'] for r in records),maximum_hermiticity_error=max(r['metrics']['max_hermiticity_error'] for r in records),
            minimum_eigenvalue=min(r['metrics']['min_output_eigenvalue'] for r in records),
            maximum_population_balance_error=max(r['metrics']['subspace_population_balance_error'] for r in records)))
    require(all(r['passed'] for r in integrity),'Per-device numerical gate failure')
    statistics_started=time.perf_counter()
    primary=paired_cvar_intervals(fidelity,repetitions=cfg['bootstrap_repetitions'],seed=cfg['bootstrap_seed'])
    sensitivity=paired_cvar_intervals(fidelity,repetitions=cfg['sensitivity_repetitions'],seed=cfg['bootstrap_seed'])
    secondary=secondary_intervals(fidelity,leakage,cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
    statistics_seconds=time.perf_counter()-statistics_started
    write_json(out/'primary_bootstrap.json',primary);write_json(out/'sensitivity_bootstrap.json',sensitivity);write_json(out/'secondary_intervals.json',secondary)
    rows=[];bounds=[];certificates={}
    for name,c in candidates.items():
        f,l=fidelity[name],leakage[name];bound=tail_bound_details(f);sim=tail_bound_details(f,confidence=1-.05/4) if name in PRIMARY else None
        nominal=audit['nominal_reproduction'][name]
        row=dict(candidate=name,role='PRIMARY' if name in PRIMARY else 'DIAGNOSTIC',n_devices=2000,nominal_fidelity=nominal['fidelity'],nominal_leakage=nominal['leakage'],
            mean_fidelity=float(f.mean()),p05_fidelity=float(np.quantile(f,.05)),cvar95_infidelity=empirical_cvar(1-f),mean_leakage=float(l.mean()),
            maximum_leakage=float(l.max()),minimum_fidelity=float(f.min()),leakage_exceedances=int(np.sum(l>1e-4)),lower_bound=bound['lower_bound'],
            mean_fidelity_ci95=secondary['individual'][name][0],p05_fidelity_ci95=secondary['individual'][name][1],mean_leakage_ci95=secondary['individual'][name][2])
        rows.append(row);certificates[name]=engineering_result(name,c,row,nominal,sim)
        bounds.append(dict(candidate=name,role=row['role'],empirical_p05=row['p05_fidelity'],p05_lcb95=bound['lower_bound'],order_index95=bound['order_statistic_index'],
            confidence95=.95,simultaneous_lcb=sim['lower_bound'] if sim else None,simultaneous_confidence=1-.05/4 if sim else None,threshold=.995,
            status=certificates[name]['status'],simultaneous_engineering_pass=certificates[name]['simultaneous_engineering_pass']))
    repeats=json.loads((out/'numerical_repeat_checks.json').read_text());observed={n:max(abs(r['fidelity']) for r in repeats if r['candidate']==n) for n in names}
    contrasts=[]
    for p,s in zip(primary['comparisons'],sensitivity['comparisons']):
        flags=summarize_claim(p,s,certificates,observed)
        contrasts.append({**p,**flags,'percent_point_improvement':100*p['delta_cvar']/next(r['cvar95_infidelity'] for r in rows if r['candidate']==p['reference'])})
    interval_rows=[dict(reference=p['reference'],candidate=p['candidate'],delta_cvar=p['delta_cvar'],ci95_low=p['ci95'][0],ci95_high=p['ci95'][1],
        bonferroni_low=p['family_adjusted_ci'][0],bonferroni_high=p['family_adjusted_ci'][1],confidence=1-.05/6,
        repetitions=50000,bootstrap_seed=941207,sensitivity_agrees=p['bootstrap_sensitivity_agrees']) for p in contrasts]
    write_csv(out/'candidate_validation_metrics.csv',rows)
    write_csv(out/'paired_cvar_comparisons.csv',[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in p.items()} for p in contrasts])
    write_csv(out/'bootstrap_confidence_intervals.csv',interval_rows);write_csv(out/'p05_certification.csv',bounds)
    write_json(out/'engineering_certification.json',certificates);write_csv(out/'computational_costs.csv',costs)
    write_json(out/'analysis_costs.json',dict(statistical_wall_seconds=statistics_seconds,preflight_solver_seconds=audit['preflight_solver_seconds'],
        validation_production_solver_seconds=sum(r['production_solver_seconds'] for r in costs),repeat_solver_seconds=sum(r['numerical_repeat_solver_seconds'] for r in costs),
        scope='Solver seconds summed over CPU workers, not wall time; active attempt wall and interruption lower bounds are in execution_attempts/. Shared historical optimization costs remain separate.'))
    write_json(out/'validation_integrity.json',dict(devices=2000,candidate_device_evaluations=10000,dataset_sha256=cohort['dataset_sha256'],
        one_shared_cohort=True,development_exact_overlaps=cohort['development_exact_overlaps'],waveforms_unchanged=True,scientific_manifest_sha256=audit['freeze_manifest_sha256']))
    write_json(out/'numerical_integrity.json',dict(all_pass=True,candidates=integrity,repeats=repeats,
        observed_fidelity_sensitivity=observed,scope='All2000 device physicality checks; only20 prespecified device indices receive independent/tighter repeats. Not a uniform numerical error bound.'))
    reporting=overall_reporting(contrasts,certificates);verdict=reporting['classification']
    text=['# Independent confirmatory validation','',f'**{verdict}** under the operational interpretation reviewed before generation.', '',
        '2000 independent simulated devices;10000 candidate-device production evaluations. Four primary eligible frozen pulses and one explicitly diagnostic pulse, all on identical device indices. No optimization or candidate changes.', '',
        '|Candidate|Mean F|P05|CVaR95|Mean leakage|Max leakage|Engineering|','|---|---:|---:|---:|---:|---:|---|']
    for r in rows:text.append(f"|{r['candidate']}|{r['mean_fidelity']:.9g}|{r['p05_fidelity']:.9g}|{r['cvar95_infidelity']:.9g}|{r['mean_leakage']:.8g}|{r['maximum_leakage']:.8g}|{certificates[r['candidate']]['status']}|")
    text+=['','## All six prespecified contrasts','','Positive ΔCVaR favors the candidate. Simultaneous Bonferroni99.1666667% intervals; ordinary95% intervals are separately exported.']
    for p in contrasts:text.append(f"- {p['candidate']} vs {p['reference']}: Δ={p['delta_cvar']:.9g}, adjusted CI {p['family_adjusted_ci']}, {p['direction']}; sensitivity agreement={p['bootstrap_sensitivity_agrees']}; resolution review={p['numerical_resolution_review_required']}; candidate engineering PASS={p['engineering_candidate_pass']}.")
    text+=['','## Scientific interpretation','',
        'Conventional versus DRAG, fixed risk versus conventional, continuation versus conventional and continuation versus fixed risk are answered by their respective prespecified contrast and certificate above. Point estimates alone do not establish superiority. Negative or inconclusive contrasts are retained.', '',
        'The earlier approximately21% development improvement was a different budget/seed aggregate on reused development devices. This fixed-pulse confirmatory experiment cannot confirm that algorithm-level percentage. It estimates the selected pulses\' differences; report their independent point percentages and simultaneous intervals without equating estimands.', '',
        'The one-sided95% distribution-free P05 lower bound and additional simultaneous four-pulse certificate are exported separately from empirical P05 and CVaR. Sampled maximum leakage is not a global bound. The excluded diagnostic can never be promoted into primary engineering eligibility.', '',
        'Statistical confidence intervals and their original50000-replicate directions remain unchanged. The100000-replicate sensitivity assessment is reported separately. Numerical repeats give finite sensitivity evidence, not a uniform CVaR error bound. Effects on the2e-6 resolution scale are labeled NUMERICALLY_UNRESOLVED; they cannot support a superiority claim even when the statistical CI excludes zero. Engineering certification and publication readiness are independent reported assessments.', '',
        'Limitations: five-level Duffing/RWA approximation; finite development truncation checks; iid quasi-static uniform uncertainty assumptions; phenomenological Lindblad noise and extrapolated higher-level rates; finite2000-device tail sample; budget-limited optimization without convergence; resource-specific trade-offs and missing exact historical CPU qualification timings; no experimental hardware verification. Optimizer seeds are not independent validation samples.', '',
        'A complete numerically interpretable simulation study permits manuscript preparation regardless of favorable or adverse outcomes. Q1 submission still needs a defensible novelty assessment, transparent reproducibility package, appropriately scoped claims and external/hardware evidence where those claims require it. No hardware demonstration or journal acceptance is implied.']
    report='\n'.join(text)+'\n';(out/'confirmatory_validation_report.md').write_text(report);(out/'scientific_conclusions.md').write_text(report)
    write_json(out/'confirmation_decision.json',reporting)
    figures(out,rows,contrasts,bounds,fidelity,leakage,candidates)

def figures(out,rows,contrasts,bounds,fidelity,leakage,candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(out)/'publication_figures';folder.mkdir(exist_ok=True);names=[r['candidate'] for r in rows]
    labels=[n.replace('_',' ') + (' (diagnostic)' if n==DIAGNOSTIC else '') for n in names]
    def save(fig,name):
        fig.suptitle('Independent simulated-device validation',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ('pdf','svg','png'):fig.savefig(folder/f'{name}.{ext}',dpi=300)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for n,l in zip(names,labels):ax.plot(np.sort(fidelity[n]),np.arange(1,2001)/2000,label=l)
    ax.set(xlabel='Unconditional X fidelity',ylabel='Empirical cumulative probability');ax.legend(fontsize=8);save(fig,'01_fidelity_distributions')
    fig,ax=plt.subplots(figsize=(9,5));ax.barh(labels,[r['cvar95_infidelity'] for r in rows]);ax.set_xlabel('CVaR95 infidelity');save(fig,'02_cvar')
    fig,ax=plt.subplots(figsize=(10,5))
    for i,p in enumerate(contrasts):
        lo,hi=p['family_adjusted_ci'];x=p['delta_cvar'];ax.hlines(i,lo,hi);ax.plot(x,i,'o');ax.plot([lo,hi],[i,i],'|')
    ax.axvline(0,color='.5');ax.set_yticks(range(6),[p['candidate']+' vs '+p['reference'] for p in contrasts]);ax.set_xlabel('Paired ΔCVaR95; six-contrast Bonferroni CI');save(fig,'03_paired_adjusted_intervals')
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(names));ax.scatter(x,[r['empirical_p05'] for r in bounds],label='Empirical P05');ax.scatter(x,[r['p05_lcb95'] for r in bounds],marker='_',label='One-sided95% LCB');ax.axhline(.995,color='r',ls='--');ax.set_xticks(x,labels,rotation=25,ha='right');ax.legend();save(fig,'04_p05_certificate')
    fig,ax=plt.subplots(figsize=(9,5))
    for n,l in zip(names,labels):ax.scatter(leakage[n],fidelity[n],s=4,alpha=.2,label=l)
    ax.axvline(1e-4,color='r',ls='--');ax.set(xlabel='Per-device six-state mean leakage',ylabel='Fidelity');ax.legend(fontsize=8);save(fig,'05_fidelity_leakage')
    fig,ax=plt.subplots(figsize=(9,5))
    for r in rows:
        n=r['candidate'];ax.plot([0,1],[candidates[n]['development_metrics']['cvar95_infidelity'],r['cvar95_infidelity']],'o-',label=n)
    ax.set_xticks([0,1],['Development selection','Independent validation']);ax.set_ylabel('CVaR95 infidelity');ax.legend(fontsize=8);save(fig,'06_development_validation')
    fig,ax=plt.subplots(figsize=(9,5))
    for r in rows:
        n=r['candidate'];c=candidates[n];cost=c.get('complete_computational_cost',c['cost']);seconds=cost.get('elapsed_seconds')
        if seconds is not None:ax.scatter(seconds,r['cvar95_infidelity'],label=n)
    ax.set(xlabel='Historical successful optimization/calibration seconds (different scopes)',ylabel='Validation CVaR95');ax.legend(fontsize=8);save(fig,'07_robustness_cost')
