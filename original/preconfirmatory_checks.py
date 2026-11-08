"""Versioned development-only checks of immutable Kairos waveforms.

No optimization and no scenario generator. Production QuTiP adaptive propagator
is reused unchanged; a second mesolve calculation checks selected channels.
"""
import concurrent.futures as cf
from dataclasses import replace,asdict
import fcntl,hashlib,itertools,json,os,time
from pathlib import Path
import numpy as np
from qutip import propagator,mesolve,liouvillian
from challenge_model import (Device,Scenario,hamiltonian,collapse_operators,
    model_operators,channel_metrics,evaluate_waveform,hardware_metrics)
from grape import FourierControls
from grape_research import waveform_hash
from grape_development import write_json,write_csv,amplitude_headroom
from robust_optimization import summarize_trials
from corrective_study_analysis import load_and_verify

OLD=Path('results/binding_budget_v2')
OUT=Path('results/preconfirmatory_checks_v1')
TOL=2e-6

def digest_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True)
    protocol,checks,errors,runs,rows,resources,details=load_and_verify(OLD)
    assert len(runs)==54 and all(checks.values()) and not errors
    baseline=OUT/'development_preservation_manifest.json'
    if not baseline.exists():write_json(baseline,{str(p):digest_file(p) for p in OLD.rglob('*') if p.is_file()})
    assert all(digest_file(p)==h for p,h in json.loads(baseline.read_text()).items()),'Development artifacts changed'
    base=protocol['base_config'];rowmap={r['trial']:r for r in rows};candidates={};audit=[]
    for r in runs:
        stem=f"trial_{r['regime']}_{r['level']}_{r['seed']}_{r['method']}";s=r['selection']['statistics'];nom=r['nominal'];hw=r['hardware']
        checks0=dict(nominal_fidelity=nom['fidelity']>=.999,nominal_leakage=nom['leakage']<=1e-4,
            maximum_ensemble_leakage=s['worst_sample_leakage']<=1e-4,
            amplitude=hw['peak_amplitude_mhz']*amplitude_headroom(base)<=80+1e-8,
            endpoints=hw['endpoint_amplitude_rad_ns']<=1e-10,numerical=r['numerical']['passed'])
        # The post-analysis draft also applied development P05 >=.995. Audit
        # this separately: the runner did not use it to retain waveforms.
        p05=s['p05_fidelity_lcb95']>=.995
        audit.append(dict(candidate=stem,method=r['method'],nominal_fidelity=nom['fidelity'],nominal_leakage=nom['leakage'],
            mean_ensemble_leakage=s['mean_leakage'],maximum_ensemble_leakage=s['worst_sample_leakage'],
            cvar95_infidelity=s['cvar95_infidelity'],p05_fidelity=s['p05_fidelity'],p05_lower_bound=s['p05_fidelity_lcb95'],
            peak_amplitude_mhz=hw['peak_amplitude_mhz'],amplitude_with_headroom_mhz=hw['peak_amplitude_mhz']*1.03,
            endpoint_amplitude=hw['endpoint_amplitude_rad_ns'],numerical_pass=r['numerical']['passed'],
            original_runner_eligible=r['selection_feasible'],audited_existing_criteria_eligible=all(checks0.values()),
            post_analysis_p05_guard_pass=p05,exclusion_reasons='; '.join(k for k,v in checks0.items() if not v),waveform_sha256=r['waveform_sha256']))
    write_csv(OUT/'candidate_eligibility_audit.csv',audit)
    saved=json.loads((OLD/'confirmatory_frozen_candidates.json').read_text())
    for method in ('conventional','risk','continuation'):
        record=saved[method];r=next(r for r in runs if f"trial_{r['regime']}_{r['level']}_{r['seed']}_{r['method']}"==record['trial'])
        eligible=[a for a in audit if a['method']==method and a['audited_existing_criteria_eligible'] and a['post_analysis_p05_guard_pass']]
        expected=min(eligible,key=lambda a:(a['cvar95_infidelity'],a['candidate']))
        assert expected['candidate']==record['trial'] and waveform_hash(r['waveform'])==record['waveform_sha256']
        coefficients=r['history'][-1]['coefficients'];basis=FourierControls(np.array(r['waveform'])[0],5,80*.002*np.pi)
        assert np.allclose(basis.waveform(np.array(coefficients)),np.array(r['waveform'])[1:].T,rtol=0,atol=1e-12)
        candidates[method]=dict(method=method,source_trial=record['trial'],optimization_seed=r['seed'],
            waveform=r['waveform'],waveform_sha256=r['waveform_sha256'],coefficients=coefficients,
            coefficient_representation='five Fourier coefficients per I/Q at scale 80 MHz',
            optimization_options=r['options'],training_config=protocol,cost=r['history'][-1]['incremental_cost'] if 'incremental_cost' in r['history'][-1] else {k:r['history'][-1][k] for k in ('objective_evaluations','total_dissipative_passes','elapsed_seconds')},
            engineering_eligible=True,role='primary_candidate')
    reference=json.loads(Path('results/grape_development_reference/references.json').read_text())['calibrated_drag']
    candidates['calibrated_drag']=dict(method='calibrated_drag',source_trial='results/grape_development_reference/references.json:calibrated_drag',
        optimization_seed=None,waveform=reference['waveform'],waveform_sha256=waveform_hash(reference['waveform']),
        coefficients=reference['parameters'],coefficient_representation='native calibrated DRAG parameters; not an exact Fourier waveform',
        pulse_spec=reference['pulse_spec'],training_config='Nominal-only calibration; no ensemble optimization',
        cost=reference['calibration'],engineering_eligible=None,role='baseline_pending_development_checks')
    leaky=min([r for r in runs if r['method']=='risk'],key=lambda r:r['selection']['statistics']['cvar95_infidelity'])
    candidates['risk_unfiltered_diagnostic']=dict(method='risk_unfiltered_diagnostic',source_trial=f"trial_{leaky['regime']}_{leaky['level']}_{leaky['seed']}_risk",
        optimization_seed=leaky['seed'],waveform=leaky['waveform'],waveform_sha256=leaky['waveform_sha256'],
        coefficients=leaky['history'][-1]['coefficients'],coefficient_representation='five Fourier coefficients per I/Q at scale 80 MHz',
        optimization_options=leaky['options'],training_config=protocol,
        cost={k:leaky['history'][-1][k] for k in ('objective_evaluations','total_dissipative_passes','elapsed_seconds')},
        engineering_eligible=False,role='diagnostic_comparator',exclusion='maximum ensemble leakage >1e-4')
    p=OUT/'input_candidates.json'
    if p.exists():assert json.loads(p.read_text())==candidates
    else:write_json(p,candidates)
    for label in ('training','selection'):
        p=OUT/f'{label}_devices.json';data=(OLD/p.name).read_bytes()
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
    sources=['preconfirmatory_checks.py','confirmatory_statistics.py','challenge_model.py','grape.py','risk_aware_grape.py','statistical_certification.py','robust_optimization.py']
    new=dict(levels=[5,6,7,8],base_config=base,tolerance=TOL,selection_devices=200,
        stress_cases='last 10 stored training scenarios, including nominal',strict_solver=dict(atol=1e-12,rtol=1e-10,max_step_ns=.05,nsteps=100000,normalize_output=False),
        source_hashes={s:digest_file(s) for s in sources},candidate_hashes={k:c['waveform_sha256'] for k,c in candidates.items()},
        independent_validation_generated=False)
    path=OUT/'check_protocol.json'
    if path.exists():assert json.loads(path.read_text())==new,'Check protocol/source changed; use new version'
    else:write_json(path,new)
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for s in sources:
        p=snapshot/s
        if not p.exists():p.write_bytes(Path(s).read_bytes())
    conv=saved['conventional'];cont=saved['continuation'];gap=cont['selection_cvar95']-conv['selection_cvar95']
    text=f'''# Candidate selection audit

All 54 frozen trials, source hashes, optimization checkpoints, exact waveforms,
selection aggregation and saved ensembles were verified. All original artifacts
are covered by a separate preservation manifest. No optimization was rerun.

Continuation was **not excluded**. Conventional's best eligible CVaR95 was
{conv['selection_cvar95']:.12g}; continuation's was {cont['selection_cvar95']:.12g}.
The difference is {gap:.12g}, smaller than the existing 2e-6 numerical metric
tolerance. The literal minimum selects conventional, but this gap does not justify
a numerically resolved or statistically significant superiority claim. Group means
under different cost regimes do not determine the minimum individual feasible pulse.

Fixed risk's unfiltered best-CVaR waveform is excluded solely by maximum observed
selection leakage >1e-4. It remains a frozen diagnostic comparator. The eligible
fixed-risk representative is the existing draft's saved waveform; no replacement
search is performed. Nominal leakage qualification is separate from maximum
selection leakage, and mean leakage is not the maximum-leakage acceptance rule.

The frozen optimization runner used nominal qualification, waveform constraints,
and maximum selection leakage. The later analysis draft additionally applied a
development P05 lower-bound guard >=.995. That guard belongs to the post-analysis
selection draft, rather than the optimizer's original eligibility rule. Every one
of the 54 trials passes it (minimum observed bound >.9967), so it changes no
exclusions, representatives or ranking. No implementation correction is required;
this provenance distinction is documented instead of calling the extra guard an
original optimizer rule. No new exclusion criterion is introduced here.

The full candidate table records each exact exclusion. The calibrated DRAG
baseline has native analytic calibration parameters; exact sampled I/Q is authoritative.
It must not be replaced by an approximate Fourier fit during these checks.
'''
    (OUT/'candidate_selection_audit.md').write_text(text)
    return new,candidates

def solve(task):
    candidate,n,scenario_dict,mode,wave,base=task
    scenario=Scenario(**scenario_dict);device=replace(Device(**base['device']),levels=n);wave=np.asarray(wave)
    h=hamiltonian(*wave,device,scenario);ops=collapse_operators(device,scenario)
    options=dict(atol=1e-12,rtol=1e-10,max_step=device.dt_ns/(8 if mode in ('eight','mesolve_eight') else 4),nsteps=100000,normalize_output=False)
    started=time.perf_counter()
    _,states,target,_,_,_,_,leak=model_operators(n)
    if mode.startswith('mesolve'):
        outputs=[mesolve(h,s.proj(),[0.,float(wave[0,-1])],c_ops=ops,options=options).states[-1] for s in states]
        f=[float(np.real((target*s).dag()*rho*(target*s))) for s,rho in zip(states,outputs)]
        l=[float(np.real((leak*rho).tr())) for rho in outputs]
        metric=dict(fidelity=float(np.mean(f)),leakage=float(np.mean(l)),state_fidelities=f,state_leakages=l)
    else:
        channel=propagator(h,float(wave[0,-1]),c_ops=ops,options=options)
        metric=channel_metrics(channel,n);outputs=[channel(s.proj()) for s in states]
    metric.update(max_trace_error=max(abs(rho.tr()-1) for rho in outputs),
        max_hermiticity_error=max(np.max(abs(rho.full()-rho.full().conj().T)) for rho in outputs),
        min_output_eigenvalue=min(np.linalg.eigvalsh((rho.full()+rho.full().conj().T)/2).min() for rho in outputs),
        subspace_population_balance_error=max(abs(float(np.real((rho.full()[0,0]+rho.full()[1,1])))+l-1) for rho,l in zip(outputs,metric['state_leakages'])))
    metric={k:float(v) if isinstance(v,(float,np.floating)) else v for k,v in metric.items()}
    metric['elapsed_seconds']=time.perf_counter()-started
    metric['physicality_pass']=bool(metric['max_trace_error']<=TOL and metric['max_hermiticity_error']<=TOL and metric['min_output_eigenvalue']>=-TOL and metric['subspace_population_balance_error']<=TOL)
    return metric

def task_path(candidate,n,scenario,mode):
    key=hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest()
    return OUT/'channels'/f'{candidate}_N{n}_{mode}_{key}.json'

def dissipator_checks(base):
    results={}
    for n in (5,6,7,8):
        device=replace(Device(**base['device']),levels=n);ops=collapse_operators(device,Scenario())
        _,_,_,a,number,_,_,leak=model_operators(n)
        manual=np.zeros((n*n,n*n),complex);identity=np.eye(n)
        for op in ops:
            c=op.full();cc=c.conj().T@c
            manual+=np.kron(c.conj(),c)-.5*np.kron(identity,cc)-.5*np.kron(cc.T,identity)
        qutip=liouvillian(0*number,ops).full();trace=np.eye(n).reshape(-1,order='F')
        results[n]=dict(operator_count=len(ops),manual_liouvillian_error=float(np.max(abs(manual-qutip))),
            trace_generator_error=float(np.max(abs(trace@qutip))),lowering_matrix_error=float(np.max(abs(a.full()-np.diag(np.sqrt(np.arange(1,n)),1)))),
            leakage_projector_error=float(np.max(abs(leak.full()-np.diag([0,0]+[1]*(n-2))))))
        results[n]['passed']=all(v<=TOL for k,v in results[n].items() if k!='operator_count')
    return results

def summarize(protocol,candidates,scenarios,verification):
    rows=[];perlevel={};allphys=True
    for name,c in candidates.items():
        for n in (5,6,7,8):
            nominal=json.loads(task_path(name,n,scenarios[0][2],'four').read_text())
            channels=[json.loads(task_path(name,n,s,'four').read_text()) for label,i,s in scenarios if label=='selection']
            st=summarize_trials(channels,protocol['base_config']['quality_targets'])
            row=dict(candidate=name,levels=n,nominal_fidelity=nominal['fidelity'],nominal_leakage=nominal['leakage'],
                mean_fidelity=st['mean_fidelity'],p05_fidelity=st['p05_fidelity'],cvar95_infidelity=st['cvar95_infidelity'],
                mean_leakage=st['mean_leakage'],maximum_leakage=st['worst_sample_leakage'],p05_lower_bound=st['p05_fidelity_lcb95'])
            perlevel[name,n]=row;rows.append(row);allphys=allphys and nominal['physicality_pass'] and all(x['physicality_pass'] for x in channels)
    metrics=('nominal_fidelity','nominal_leakage','mean_fidelity','p05_fidelity','cvar95_infidelity','mean_leakage','maximum_leakage')
    status={}
    for row in rows:
        name,n=row['candidate'],row['levels']
        for k in metrics:
            row['delta_previous_'+k]=None if n==5 else row[k]-perlevel[name,n-1][k]
            row['difference_from_N8_'+k]=row[k]-perlevel[name,8][k]
    for name in candidates:
        pointwise=[];adjacent=[]
        for label,i,s in scenarios:
            values={n:json.loads(task_path(name,n,s,'four').read_text()) for n in (5,6,7,8)}
            allphys=allphys and all(v['physicality_pass'] for v in values.values())
            pointwise.extend(abs(values[5][k]-values[8][k]) for k in ('fidelity','leakage'))
            adjacent.extend(abs(values[7][k]-values[8][k]) for k in ('fidelity','leakage'))
        aggregate=max(abs(perlevel[name,5][k]-perlevel[name,8][k]) for k in metrics)
        status[name]=dict(maximum_pointwise_N5_N8_difference=max(pointwise),maximum_pointwise_N7_N8_difference=max(adjacent),
            maximum_aggregate_N5_N8_difference=aggregate,five_level_pass=bool(max(pointwise)<=TOL and max(adjacent)<=TOL and aggregate<=TOL))
    write_csv(OUT/'hilbert_convergence.csv',rows)
    write_json(OUT/'hilbert_convergence_status.json',status)
    verification['all_physicality_pass']=allphys;write_json(OUT/'independent_solver_verification.json',verification)
    freeze={};eligible=[]
    for name,c in candidates.items():
        row=perlevel[name,5];hw=hardware_metrics(*np.array(c['waveform']),Device(**protocol['base_config']['device']))
        engineering=bool(row['nominal_fidelity']>=.999 and row['nominal_leakage']<=1e-4 and row['maximum_leakage']<=1e-4 and
            row['p05_lower_bound']>=.995 and hw['endpoint_amplitude_rad_ns']<=1e-10 and hw['peak_amplitude_mhz']*1.03<=80+1e-8)
        allow=engineering and status[name]['five_level_pass'] and verification['passed'] and allphys
        freeze[name]={**c,'engineering_eligible_after_checks':engineering,'hilbert_check':status[name],
            'independent_solver_pass':verification['passed'],'nominal_fidelity':row['nominal_fidelity'],'nominal_leakage':row['nominal_leakage'],
            'development_metrics':row,'hardware':hw,'allowed_primary_comparator':bool(allow and name!='risk_unfiltered_diagnostic'),
            'role_after_checks':'primary_candidate' if allow and name!='risk_unfiltered_diagnostic' else 'diagnostic_comparator'}
        if freeze[name]['allowed_primary_comparator']:eligible.append(name)
    write_json(OUT/'frozen_candidates.json',freeze)
    numerical=verification['passed'] and allphys
    ready=numerical and all(status[n]['five_level_pass'] for n in ('calibrated_drag','conventional','risk','continuation'))
    decision='GO' if ready and len(eligible)==4 else 'CONDITIONAL GO' if ready else 'NO-GO'
    write_json(OUT/'preconfirmatory_readiness.json',dict(decision=decision,numerical_pass=numerical,
        allowed_primary_candidates=eligible,hilbert_checks=status,independent_validation_generated=False,
        independent_validation_execution_authorized=False,
        scope='Finite existing development ensemble and saved stress cases; not uniform Hilbert convergence, hardware validation or journal acceptance.'))
    report=['# Hilbert-space convergence report','',f'Decision: **{decision}**. CPU QuTiP strict solver, unchanged waveforms and model, N=5,6,7,8.', '',
        '200 existing selection scenarios, nominal and 10 existing training stress entries were used. Duplicate nominal entries reuse one channel solve. No new uncertainty scenarios or independent validation devices were drawn.', '',
        'Higher-level relaxation uses n/2 times the n=2 rate and dephasing uses (n/2)^2 times its projector rate, exactly as the existing model. This extrapolation is a model assumption, not measured higher-level noise. The Duffing ladder and annihilation sqrt(n) couplings, six unconditional computational X-state fidelities and total population outside |0>,|1> are preserved.', '',
        '|Candidate|Max pointwise N5−N8 error|Max N7−N8 error|Five-level check|', '|---|---:|---:|---|']
    for name,s in status.items():report.append(f"|{name}|{s['maximum_pointwise_N5_N8_difference']:.8g}|{s['maximum_pointwise_N7_N8_difference']:.8g}|{s['five_level_pass']}|")
    report+=['', 'All aggregate metrics, signed adjacent differences and signed differences from N8 are in hilbert_convergence.csv. Pointwise tests include saved training stress cases as well as the common 200 selection devices. The existing absolute 2e-6 metric tolerance is used unchanged as a numerical adequacy check. It is not a derived truncation bound or a relative leakage-error guarantee. Report relative errors too when leakage is tiny, and do not infer uniform convergence from this finite sample.', '',
        'Independent solver verification reports adaptive CPU maximum steps dt/4 versus dt/8 with the same continuous linear waveform interpolant and unchanged atol/rtol. These are not new sampled waveforms. Existing GPU four/eight piecewise interpolation checks remain in the preserved development evidence; QuTiP max-step checks isolate CPU integration error. Selected channels are also checked with six independent QuTiP mesolve state propagations.', '',
        'An engineering-eligible pulse must separately satisfy nominal fidelity and leakage, sampled ensemble leakage, amplitude/endpoints and the existing post-analysis P05 guard. A high nominal fidelity alone is insufficient. Candidates failing engineering remain diagnostic comparators. If five-level truncation fails, this task does not switch the frozen production model to N8 or silently reoptimize.', '',
        'Validation protocol, exact waveform arrays, coefficients/native DRAG parameters, costs and source hashes are frozen in this versioned directory. Independent validation has not been executed. Numerical readiness does not establish the RWA/Duffing model, distributions or hardware accuracy, nor Q1 journal readiness.']
    (OUT/'hilbert_convergence_report.md').write_text('\n'.join(report)+'\n')
    plots(rows)
    from confirmatory_statistics import freeze_protocol
    freeze_protocol(OUT,protocol,freeze,decision)

def plots(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'hilbert_convergence_figures';folder.mkdir(exist_ok=True)
    for field in ('nominal_fidelity','nominal_leakage','cvar95_infidelity','maximum_leakage'):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5));fig.suptitle('Development-only Hilbert convergence: '+field.replace('_',' '))
        for name in sorted({r['candidate'] for r in rows}):
            rr=[r for r in rows if r['candidate']==name]
            axes[0].plot([r['levels'] for r in rr],[r[field] for r in rr],'o-',label=name)
            axes[1].plot([r['levels'] for r in rr],[abs(r['difference_from_N8_'+field]) for r in rr],'o-')
        axes[1].axhline(TOL,color='r',ls='--',label='Existing 2e-6 tolerance');axes[1].set_yscale('symlog',linthresh=1e-12)
        for ax in axes:ax.set_xlabel('Hilbert levels');ax.set_xticks([5,6,7,8]);ax.grid(alpha=.2)
        axes[0].set_ylabel(field.replace('_',' '));axes[1].set_ylabel('Absolute difference from N=8');axes[0].legend(fontsize=7)
        fig.tight_layout(rect=(0,0,1,.93))
        for suffix in ('png','pdf','svg'):fig.savefig(folder/f'{field}.{suffix}',dpi=300)
        plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);lock=(OUT/'checks.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol,candidates=prepare();(OUT/'channels').mkdir(exist_ok=True)
    selection=json.loads((OUT/'selection_devices.json').read_text());training=json.loads((OUT/'training_devices.json').read_text())
    scenarios=[('nominal',0,asdict(Scenario()))]+[('selection',i,s) for i,s in enumerate(selection)]+[('stress',i,s) for i,s in enumerate(training[-10:])]
    tasks={}
    for name,c in candidates.items():
        for n in (5,6,7,8):
            for label,i,s in scenarios:
                path=task_path(name,n,s,'four')
                if not path.exists():tasks[path]=(name,n,s,'four',c['waveform'],protocol['base_config'])
    total=len(tasks);done=0
    print('Production channel tasks pending',total,flush=True)
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(solve,t):p for p,t in tasks.items()}
        for future in cf.as_completed(pending):
            path=pending[future];write_json(path,future.result());done+=1
            if done%20==0 or done==total:
                write_json(OUT/'status.json',dict(status='RUNNING',completed_new_channels=done,pending_new_channels=total-done,independent_validation_generated=False))
                print('Channels',done,'/',total,flush=True)
    verification=dict(dissipators=dissipator_checks(protocol['base_config']),comparisons=[])
    for name,c in candidates.items():
        for n in (5,6,7,8):
            for label,i,s in [scenarios[0],scenarios[-2],scenarios[-1]]:
                ref=json.loads(task_path(name,n,s,'four').read_text())
                for mode in ('eight','mesolve_eight'):
                    path=task_path(name,n,s,mode)
                    if not path.exists():write_json(path,solve((name,n,s,mode,c['waveform'],protocol['base_config'])))
                    m=json.loads(path.read_text());err=max(abs(m[k]-ref[k]) for k in ('fidelity','leakage'))
                    verification['comparisons'].append(dict(candidate=name,levels=n,case=label,index=i,mode=mode,
                        fidelity_difference=m['fidelity']-ref['fidelity'],leakage_difference=m['leakage']-ref['leakage'],
                        metric_error=err,physicality_pass=m['physicality_pass'],passed=bool(err<=TOL and m['physicality_pass'])))
    verification['passed']=all(c['passed'] for c in verification['comparisons']) and all(c['passed'] for c in verification['dissipators'].values())
    summarize(protocol,candidates,scenarios,verification)
    manifest=json.loads((OUT/'development_preservation_manifest.json').read_text());assert all(digest_file(p)==h for p,h in manifest.items())
    write_json(OUT/'status.json',dict(status='COMPLETE',channels=len(list((OUT/'channels').glob('*.json'))),independent_validation_generated=False,development_unchanged=True))
    freeze={str(p.relative_to(OUT)):digest_file(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ('execution.log','launcher.json','checks.lock','freeze_manifest.json')}
    write_json(OUT/'freeze_manifest.json',freeze)
    print('COMPLETE; original development unchanged; no validation drawn',flush=True)

if __name__=='__main__':main()
