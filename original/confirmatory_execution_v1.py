"""Proposed execution adapter for the frozen preconfirmatory v1 protocol.

Default is audit-only. Execution requires a separately audited implementation
release bound to source hashes. No automatic acceptance or device-generation
override is provided. This adapter is NOT part of the original scientific freeze.
"""
import argparse,concurrent.futures as cf,fcntl,hashlib,importlib.metadata,json,os,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
from challenge_model import Device,Scenario,monte_carlo_scenarios,hardware_metrics
from grape_research import waveform_hash
from grape_development import write_json
from preconfirmatory_checks import solve
from confirmatory_statistics import PRIMARY,PAIRS

FREEZE=Path('results/preconfirmatory_checks_v1')
PACKAGE=Path('results/confirmatory_implementation_v1')
DEFAULT_OUTPUT=Path('results/confirmatory_validation_v1')
SOURCES=('confirmatory_execution_v1.py','confirmatory_analysis_v1.py')
INTERPRETATION=PACKAGE/'reporting_interpretation.json'
TOL=2e-6
DIAGNOSTIC='risk_unfiltered_diagnostic'

class FreezeError(RuntimeError):pass

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def seal_record(record):
    record['checkpoint_sha256']=hashlib.sha256(canonical(record)).hexdigest()
    return record
def require(condition,message):
    if not condition:raise FreezeError(message)

def verify_freeze(root=FREEZE,nominal=True):
    root=Path(root);manifest=json.loads((root/'freeze_manifest.json').read_text())
    bad=[f for f,h in manifest.items() if not (root/f).is_file() or sha(root/f)!=h]
    require(not bad,'Frozen artifacts changed: '+str(bad))
    cfg=json.loads((root/'confirmatory_protocol.json').read_text());candidates=json.loads((root/'frozen_candidates.json').read_text())
    require(cfg['validation_seed']==941206 and cfg['validation_count']==2000,'Reserved validation seed/count mismatch')
    require(cfg['bootstrap_seed']==941207 and cfg['bootstrap_repetitions']==50000 and cfg['sensitivity_repetitions']==100000,'Bootstrap configuration mismatch')
    require(cfg['decision']=='GO' and json.loads((root/'preconfirmatory_readiness.json').read_text())['decision']=='GO','Preconfirmatory scientific GO missing')
    require(cfg['primary_pairs']==[list(p) for p in PAIRS] and cfg['alpha']==.05,'Contrast family mismatch')
    require(cfg['hilbert_dimension']==5 and cfg['physical_model']['device']['levels']==5,'Physical dimension mismatch')
    require(cfg['physical_model']['gate_duration_ns']==15.6 and cfg['physical_model']['device']['amplitude_limit_mhz']==80.,'Hardware configuration mismatch')
    require(cfg['production_solver']==dict(backend='CPU QuTiP propagator',atol=1e-12,rtol=1e-10,max_step_ns=.05,nsteps=100000,normalize_output=False),'Solver configuration mismatch')
    base=cfg['physical_model']
    require(base['manufacturing_ranges']==dict(amplitude_fraction=.03,detuning_mhz=3.,alpha_shift_mhz=8.) and base['coherence_min']==.8,'Uncertainty distributions mismatch')
    require(base['quality_targets']==dict(nominal_fidelity_min=.999,leakage_max=.0001,p05_validation_lcb_min=.995,solver_metric_tolerance=2e-6),'Scientific thresholds mismatch')
    bad=[f for f,h in cfg['source_hashes'].items() if not Path(f).is_file() or sha(f)!=h]
    require(not bad,'Frozen execution dependencies changed: '+str(bad))
    require(sha('confirmatory_statistics.py')==cfg['statistics_source_sha256'],'Frozen statistics changed')
    for snapshot in (root/'audit_source_snapshot').iterdir():
        require(Path(snapshot.name).is_file() and sha(snapshot.name)==sha(snapshot),'Frozen audit dependency changed: '+snapshot.name)
    environment=json.loads((root/'environment.json').read_text())
    for name,version in environment['packages'].items():require(importlib.metadata.version(name)==version,'Package version changed: '+name)
    require(set(candidates)==set(PRIMARY)|{DIAGNOSTIC},'Candidate set changed')
    nominal_rows={};repeat_rows=[];preflight_solver_seconds=0.
    for name,c in candidates.items():
        wave=np.asarray(c['waveform'],dtype=float)
        require(waveform_hash(wave)==c['waveform_sha256']==cfg['candidate_hashes'][name],'Waveform hash mismatch: '+name)
        require(c['allowed_primary_comparator']==(name in PRIMARY),'Frozen candidate role mismatch: '+name)
        hw=hardware_metrics(*wave,Device(**base['device']))
        require(hw['endpoint_amplitude_rad_ns']<=1e-10 and hw['peak_amplitude_mhz']*1.03<=80+1e-8 and abs(hw['duration_ns']-15.6)<=1e-10,'Waveform constraint failure')
        if nominal:
            m=solve((name,5,asdict(Scenario()),'four',wave,base))
            require(m['physicality_pass'] and abs(m['fidelity']-c['nominal_fidelity'])<=TOL and abs(m['leakage']-c['nominal_leakage'])<=TOL,'Nominal reproduction failure: '+name)
            nominal_rows[name]=m
            preflight_solver_seconds+=m['elapsed_seconds']
            stress=json.loads((root/'training_devices.json').read_text())[-2:]
            for case,s in enumerate([asdict(Scenario()),*stress]):
                baseline=m if case==0 else solve((name,5,s,'four',wave,base))
                if case:preflight_solver_seconds+=baseline['elapsed_seconds']
                for mode in ('eight','mesolve_eight'):
                    alt=solve((name,5,s,mode,wave,base))
                    preflight_solver_seconds+=alt['elapsed_seconds']
                    error=max(abs(alt[k]-baseline[k]) for k in ('fidelity','leakage'))
                    require(baseline['physicality_pass'] and alt['physicality_pass'] and error<=TOL,'CPU preflight numerical failure: '+name)
                    repeat_rows.append(dict(candidate=name,saved_case=case,mode=mode,metric_error=error,passed=True))
    return cfg,candidates,dict(freeze_manifest_sha256=sha(root/'freeze_manifest.json'),verified_artifacts=len(manifest),
        verified_candidates=len(candidates),nominal_reproduction=nominal_rows,preflight_repeats=repeat_rows,preflight_solver_seconds=preflight_solver_seconds,source_integrity=True,package_versions=environment['packages'],
        source_environment_note='CPU production only; no optimizer or development budget settings are read for execution')

def verify_release(package=PACKAGE):
    """A review record is not manufactured by this program."""
    package=Path(package);manifest_path=package/'implementation_manifest.json';release=package/'implementation_review.json'
    require(manifest_path.exists() and release.exists(),'Missing audited implementation release; no validation generation is allowed')
    manifest=json.loads(manifest_path.read_text());r=json.loads(release.read_text())
    require(r.get('status')=='ACCEPTED' and r.get('implementation_manifest_sha256')==sha(manifest_path) and bool(r.get('reviewer')),'Implementation review does not accept the exact manifest')
    require(r.get('review_completed_before_validation') is True,'Implementation review must precede validation')
    require(set(manifest['sources'])==set(SOURCES),'Incomplete execution-source manifest')
    for path,h in manifest['sources'].items():require(sha(path)==h,'Reviewed implementation source changed: '+path)
    require(bool(manifest.get('reviewed_artifacts')),'Reviewed reporting interpretation is missing')
    for path,h in manifest['reviewed_artifacts'].items():require(sha(path)==h,'Reviewed artifact changed: '+path)
    require(manifest['frozen_scientific_manifest_sha256']==sha(FREEZE/'freeze_manifest.json'),'Implementation is bound to another scientific freeze')
    return sha(manifest_path)

def development_scenario_keys():
    values=[]
    for label in ('training','selection'):values+=json.loads((FREEZE/f'{label}_devices.json').read_text())
    return set(map(canonical,values))

def validate_cohort(d,cfg,release_hash):
    require(d.get('seed')==941206 and d.get('n_devices')==2000 and len(d.get('scenarios',[]))==2000,'Saved cohort count/seed mismatch')
    require(d.get('implementation_manifest_sha256')==release_hash and d.get('generator_source_sha256')==sha('challenge_model.py'),'Saved cohort implementation mismatch')
    require(d.get('numpy_version')==importlib.metadata.version('numpy') and d.get('generator')=='challenge_model.monte_carlo_scenarios / numpy.default_rng','Saved generator version mismatch')
    base=cfg['physical_model'];distribution=dict(**base['manufacturing_ranges'],coherence_min=base['coherence_min'])
    require(d.get('distribution')==distribution,'Saved distribution metadata mismatch')
    expected_fields=set(asdict(Scenario()))
    lower=np.array([.97,-3.,-8.,.8,.8,.8,.8]);upper=np.array([1.03,3.,8.,1.,1.,1.,1.])
    for s in d['scenarios']:
        require(isinstance(s,dict) and set(s)==expected_fields,'Invalid scenario schema')
        require(all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in s.values()),'Non-numeric scenario field')
        x=np.array([s[k] for k in asdict(Scenario())],dtype=float)
        require(np.isfinite(x).all() and np.all(x>=lower) and np.all(x<=upper),'Scenario outside frozen uncertainty ranges')
    require(hashlib.sha256(canonical(d['scenarios'])).hexdigest()==d.get('dataset_sha256'),'Saved dataset checksum failure')
    payload={k:v for k,v in d.items() if k!='cohort_payload_sha256'}
    require(hashlib.sha256(canonical(payload)).hexdigest()==d.get('cohort_payload_sha256'),'Saved cohort metadata checksum failure')
    overlap=development_scenario_keys()&set(map(canonical,d['scenarios']))
    require(not overlap and d.get('development_exact_overlaps')==0,'Saved cohort overlaps development data')
    return d

def load_or_generate_dataset(out,cfg,release_hash,generator=monte_carlo_scenarios):
    """One atomic cohort commit. Ambiguous interrupted generation fails closed."""
    out=Path(out);path=out/'validation_dataset.json';marker=out/'dataset_generation_started.json'
    if path.exists():
        require(marker.exists(),'Committed cohort lacks generation transaction evidence')
        expected=dict(seed=941206,count=2000,status='GENERATION_STARTED',implementation_manifest_sha256=release_hash)
        require(json.loads(marker.read_text())==expected,'Generation transaction belongs to another version')
        d=validate_cohort(json.loads(path.read_text()),cfg,release_hash);os.chmod(path,0o444);return d
    require(not marker.exists(),'Generation was interrupted before atomic cohort commit; inspect evidence, do not redraw')
    require(verify_release()==release_hash,'Generation requires the accepted implementation release')
    write_json(marker,dict(seed=941206,count=2000,status='GENERATION_STARTED',implementation_manifest_sha256=release_hash))
    base=cfg['physical_model'];scenarios=[asdict(s) for s in generator(2000,941206,coherence_min=base['coherence_min'],**base['manufacturing_ranges'])]
    overlap=development_scenario_keys()&set(map(canonical,scenarios))
    require(len(scenarios)==2000 and not overlap,'Cohort size or development-overlap failure; no redraw')
    d=dict(seed=941206,n_devices=2000,scenarios=scenarios,dataset_sha256=hashlib.sha256(canonical(scenarios)).hexdigest(),
        generator='challenge_model.monte_carlo_scenarios / numpy.default_rng',generator_source_sha256=sha('challenge_model.py'),
        numpy_version=importlib.metadata.version('numpy'),distribution=dict(**base['manufacturing_ranges'],coherence_min=base['coherence_min']),
        development_exact_overlaps=len(overlap),independence_scope='Fresh reserved seed and frozen iid sampler; exact nonoverlap is a consistency check, not a statistical proof of independence',
        implementation_manifest_sha256=release_hash)
    d['cohort_payload_sha256']=hashlib.sha256(canonical(d)).hexdigest()
    validate_cohort(d,cfg,release_hash)
    write_json(path,d);os.chmod(path,0o444);return d

def result_path(out,name,index,mode='four'):return Path(out)/'channels'/name/f'{index:04d}_{mode}.json'

def evaluate(task):
    name,index,scenario,candidate,cfg,dataset_sha,mode=task
    started=time.perf_counter()
    try:
        metrics=solve((name,5,scenario,mode,candidate['waveform'],cfg['physical_model']))
        validate_metrics(metrics)
        valid=metrics['physicality_pass'] and all(np.isfinite(x) and -TOL<=x<=1+TOL for x in metrics['state_fidelities']+metrics['state_leakages'])
        return seal_record(dict(status='COMPLETE' if valid else 'NUMERICAL_FAILURE',candidate=name,index=index,mode=mode,
            candidate_sha256=candidate['waveform_sha256'],dataset_sha256=dataset_sha,scenario_sha256=hashlib.sha256(canonical(scenario)).hexdigest(),
            metrics=metrics,elapsed_seconds=time.perf_counter()-started))
    except Exception as error:
        return seal_record(dict(status='SOLVER_FAILURE',candidate=name,index=index,mode=mode,candidate_sha256=candidate['waveform_sha256'],
            dataset_sha256=dataset_sha,scenario_sha256=hashlib.sha256(canonical(scenario)).hexdigest(),
            error=str(error),raw_metrics_repr=repr(locals().get('metrics')),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started))

def verify_checkpoint(record,name,index,candidate,dataset,mode='four'):
    require(record.get('checkpoint_sha256')==hashlib.sha256(canonical({k:v for k,v in record.items() if k!='checkpoint_sha256'})).hexdigest(),'Checkpoint content checksum failure')
    require(record['candidate']==name and record['index']==index and record['mode']==mode,'Checkpoint index/candidate mismatch')
    require(record['candidate_sha256']==candidate['waveform_sha256'] and record['dataset_sha256']==dataset['dataset_sha256'],'Checkpoint waveform/cohort mismatch')
    require(record['scenario_sha256']==hashlib.sha256(canonical(dataset['scenarios'][index])).hexdigest(),'Checkpoint scenario mismatch')
    require(record['status']=='COMPLETE','Preserved solver/numerical failure blocks confirmation')
    require(mode!='four' or 'infidelity' in record['metrics'],'Production infidelity missing')
    validate_metrics(record['metrics'])

def validate_metrics(m):
    require(len(m.get('state_fidelities',[]))==6 and len(m.get('state_leakages',[]))==6,'Six-state records missing')
    require(m.get('physicality_pass') is True and all(np.isfinite(x) and -TOL<=x<=1+TOL for x in m['state_fidelities']+m['state_leakages']),'Checkpoint numerical diagnostics failed')
    require(np.isfinite(m['fidelity']) and np.isfinite(m['leakage']) and abs(np.mean(m['state_fidelities'])-m['fidelity'])<=1e-12 and abs(np.mean(m['state_leakages'])-m['leakage'])<=1e-12,'Scalar/six-state metric inconsistency')
    if 'infidelity' in m:require(np.isfinite(m['infidelity']) and abs(m['infidelity']-(1-m['fidelity']))<=1e-12,'Infidelity inconsistent with fidelity')
    for key in ('max_trace_error','max_hermiticity_error','subspace_population_balance_error'):
        require(np.isfinite(m[key]) and 0<=m[key]<=TOL,'Invalid '+key)
    require(np.isfinite(m['min_output_eigenvalue']) and m['min_output_eigenvalue']>=-TOL,'Invalid density positivity')

def execute(out,cfg,candidates,audit):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return execute_locked(out,cfg,candidates,audit)

def execute_locked(out,cfg,candidates,audit):
    started=time.perf_counter()
    release=verify_release()
    frozen_cfg,frozen_candidates,current=verify_freeze(nominal=False)
    require(cfg==frozen_cfg and candidates==frozen_candidates and audit['freeze_manifest_sha256']==current['freeze_manifest_sha256'],'Execution input is not the verified scientific freeze')
    binding=dict(scientific_manifest_sha256=audit['freeze_manifest_sha256'],implementation_manifest_sha256=release)
    p=out/'execution_binding.json'
    if p.exists():require(json.loads(p.read_text())==binding,'Output contains another protocol version')
    else:write_json(p,binding)
    write_json(out/'freeze_verification.json',audit)
    write_json(out/'execution_environment.json',dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','qutip','matplotlib')},
        source_hashes={f:sha(f) for f in SOURCES},solver=cfg['production_solver'],workers=4))
    cohort=reserve_and_load_cohort(out,cfg,release)
    pending,complete=plan_evaluations(out,cfg,candidates,cohort)
    attempt=out/'execution_attempts'/f'{time.time_ns()}.json';attempt.parent.mkdir(exist_ok=True)
    def status(state):
        ledger=dict(status=state,completed_candidate_device_evaluations=complete,
            required_candidate_device_evaluations=10000,validation_devices=2000,unique_dataset_sha256=cohort['dataset_sha256'],
            attempt_elapsed_seconds=time.perf_counter()-started,scope='Active execution-attempt wall; interrupted attempts retain their last checkpoint lower bound')
        write_json(out/'completion_ledger.json',ledger);write_json(attempt,ledger)
    status('RUNNING')
    def update_completion(record,failed):
        nonlocal complete
        if record['status']=='COMPLETE':complete+=1
        if failed or complete%25==0:
            status('NUMERICAL_FAILURE' if failed else 'RUNNING');print('Completed',complete,'/10000',flush=True)
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        run_pending(pool,pending,out,update_completion)
    require(complete==10000,'Incomplete candidate-device evaluation grid')
    finish_validation(out,cfg,candidates,cohort,audit,status)

def reserve_and_load_cohort(out,cfg,release,package=PACKAGE):
    out=Path(out);package=Path(package)
    # One global reservation prevents --output from creating another cohort.
    with (package/'cohort_registry.lock').open('a') as registry_lock:
        fcntl.flock(registry_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=package/'reserved_cohort_registry.json';reservation=dict(output=str(out.resolve()),seed=941206,count=2000,implementation_manifest_sha256=release)
        existing=json.loads(path.read_text()) if path.exists() else None
        if existing:
            require(all(existing.get(k)==v for k,v in reservation.items()),'Reserved cohort already assigned to another output/version')
            require(not existing.get('cohort_payload_sha256') or (out/'validation_dataset.json').exists(),'Committed cohort missing; do not regenerate')
        else:write_json(path,reservation)
        cohort=load_or_generate_dataset(out,cfg,release)
        if existing and existing.get('cohort_payload_sha256'):require(existing['cohort_payload_sha256']==cohort['cohort_payload_sha256'],'Committed cohort changed')
        write_json(path,{**reservation,'cohort_payload_sha256':cohort['cohort_payload_sha256']})
    return cohort

def plan_evaluations(out,cfg,candidates,cohort):
    out=Path(out);pending=[];complete=0
    for name,c in candidates.items():
        (out/'channels'/name).mkdir(parents=True,exist_ok=True)
        for i,s in enumerate(cohort['scenarios']):
            path=result_path(out,name,i)
            if path.exists():verify_checkpoint(json.loads(path.read_text()),name,i,c,cohort);complete+=1
            else:pending.append((name,i,s,c,cfg,cohort['dataset_sha256'],'four'))
    return pending,complete

def run_pending(pool,pending,out,on_record):
    failed=False
    futures={pool.submit(evaluate,t):t for t in pending}
    for future in cf.as_completed(futures):
        if future.cancelled():continue
        task=futures[future];name,i,scenario,candidate,_,dataset_sha,mode=task
        try:record=future.result()
        except Exception as error:
            record=seal_record(dict(status='SOLVER_FAILURE',candidate=name,index=i,mode=mode,candidate_sha256=candidate['waveform_sha256'],
                dataset_sha256=dataset_sha,scenario_sha256=hashlib.sha256(canonical(scenario)).hexdigest(),
                error=str(error),traceback=traceback.format_exc(),elapsed_seconds=None,scope='Worker-level failure; elapsed unavailable'))
        write_json(result_path(out,name,i),record)
        if record['status']!='COMPLETE':
            failed=True
            for f in futures:f.cancel()
        on_record(record,failed)
    require(not failed,'Validation stopped on preserved numerical/solver failure; all returned in-flight results retained')

def finish_validation(out,cfg,candidates,cohort,audit,status):
    status('NUMERICAL_REPEATS');repeat_indices=np.linspace(0,1999,20,dtype=int).tolist();repeat=[]
    for name,c in candidates.items():
        for i in repeat_indices:
            baseline=json.loads(result_path(out,name,i).read_text())['metrics']
            for mode in ('eight','mesolve_eight'):
                path=result_path(out,name,i,mode)
                if path.exists():record=json.loads(path.read_text())
                else:record=evaluate((name,i,cohort['scenarios'][i],c,cfg,cohort['dataset_sha256'],mode));write_json(path,record)
                verify_checkpoint(record,name,i,c,cohort,mode)
                differences={k:record['metrics'][k]-baseline[k] for k in ('fidelity','leakage')}
                repeat.append(dict(candidate=name,index=i,mode=mode,**differences))
                if max(map(abs,differences.values()))>TOL:
                    write_json(out/'numerical_repeat_failure.json',repeat);status('NUMERICAL_FAILURE');raise FreezeError('Frozen numerical repeat check failed')
    write_json(out/'numerical_repeat_checks.json',repeat)
    status('STATISTICAL_ANALYSIS')
    from confirmatory_analysis_v1 import analyze
    analyze(out,cfg,candidates,cohort,audit)
    # Detect any change to the original scientific artifacts during execution.
    verify_freeze(nominal=False);verify_release();status('COMPLETE')

def audit_package():
    cfg,candidates,audit=verify_freeze();PACKAGE.mkdir(exist_ok=True)
    write_json(PACKAGE/'scientific_freeze_audit.json',audit)
    write_json(PACKAGE/'nominal_reproduction.json',audit['nominal_reproduction'])
    mapping=dict(confirmatory_protocol_md=str(FREEZE/'confirmatory_protocol_frozen.md'),
        confirmatory_configuration_json=str(FREEZE/'confirmatory_protocol.json'),statistical_analysis_plan_md='Statistical endpoints and multiplicity section of confirmatory_protocol_frozen.md',
        solver_verification_json=str(FREEZE/'independent_solver_verification.json'))
    write_json(PACKAGE/'frozen_artifact_mapping.json',mapping)
    require(INTERPRETATION.is_file(),'Reporting interpretation must be documented before implementation release')
    manifest=dict(status='REVIEWED_IMPLEMENTATION_V1',implementation_id='confirmatory_execution_v1.final_review.1',sources={p:sha(p) for p in SOURCES},reviewed_artifacts={str(INTERPRETATION):sha(INTERPRETATION)},frozen_scientific_manifest_sha256=audit['freeze_manifest_sha256'],
        no_validation_devices_generated=True,original_freeze_has_complete_executor=False)
    p=PACKAGE/'implementation_manifest.json'
    if p.exists():require(json.loads(p.read_text())==manifest,'Proposed implementation changed; version/review must be updated')
    else:write_json(p,manifest)
    snapshot=PACKAGE/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in SOURCES:
        p=snapshot/name
        if p.exists():require(sha(p)==sha(name),'Proposed source snapshot changed')
        else:p.write_bytes(Path(name).read_bytes())
    return cfg,candidates,audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--output',default=str(DEFAULT_OUTPUT));args=p.parse_args()
    try:
        cfg,candidates,audit=audit_package()
        if args.run:execute(Path(args.output),cfg,candidates,audit)
        else:
            if (PACKAGE/'implementation_review.json').exists():verify_release();print('Accepted implementation and scientific freeze verified. Audit-only: no validation devices generated.',flush=True)
            else:print('Audit complete. Implementation awaits acceptance. No validation devices generated.',flush=True)
    except Exception as error:
        dest=Path(args.output) if args.run else PACKAGE;dest.mkdir(parents=True,exist_ok=True)
        evidence=dict(status='BLOCKED_FAILURE',error=str(error),traceback=traceback.format_exc(),confirmation_interpretable=False)
        write_json(dest/'fatal_error.json',evidence)
        if args.run:
            ledger=dest/'completion_ledger.json';previous=json.loads(ledger.read_text()) if ledger.exists() else {}
            write_json(ledger,{**previous,**evidence})
            (dest/'scientific_conclusions.md').write_text('# Validation blocked\n\nAn execution/integrity failure prevents confirmation. Inspect fatal_error.json and preserved checkpoints. No favorable conclusion is authorized.\n')
            decision=dest/'confirmation_decision.json'
            if decision.exists():write_json(dest/'provisional_confirmation_decision.json',json.loads(decision.read_text()))
            write_json(decision,dict(classification=None,status='BLOCKED_FAILURE',confirmation_interpretable=False,error=str(error)))
        raise

if __name__=='__main__':main()
