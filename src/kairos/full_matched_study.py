"""Full pre-validation study. Reuses grape_development; never draws validation."""
import argparse,fcntl,hashlib,importlib.metadata,json,os,subprocess,sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
from .challenge_model import Device,evaluate_waveform,hardware_metrics
from .grape import DEFAULTS,FourierControls,EnsembleGRAPE
from .risk_aware_grape import RiskEnsembleGRAPE
from .grape_development import (load_config,load_reference_checkpoint,validate_reference_waveform,
    make_ensembles,amplitude_headroom,optimizer_options,write_json,run)
from .grape_research import waveform_hash,nominal_qualified
from .jax_backend import JAXEnsemble,gpu_preflight


def environment():
    try:
        smi=subprocess.run(['nvidia-smi','--query-gpu=name,compute_cap,driver_version,memory.total',
            '--format=csv,noheader'],text=True,capture_output=True,timeout=15)
        gpu=smi.stdout.strip() if smi.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):gpu=None
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()
        if d.metadata['Name'].lower().startswith(('jax','nvidia-','qutip','numpy','scipy'))}
    return dict(gpu_inventory=gpu,gpu_model=gpu.split(',')[0].strip() if gpu else None,
        driver_version=gpu.split(',')[2].strip() if gpu else None,
        jax_version=packages.get('jax'),packages=packages,python=sys.version,platform=sys.platform,
        thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','XLA_FLAGS')})


def qualify(pulse,device,training,options,out):
    """Use current matching benchmark proof, or qualify a new GPU on training only."""
    certificate=options['acceleration'].get('qualification_report')
    if certificate and Path(certificate).exists():
        try:return gpu_preflight(pulse,device,training,options),certificate
        except Exception as error:previous_error=str(error)
    else:previous_error='No existing matching benchmark certificate'
    checks={};gpu_kind=None
    for name,cls in [('conventional',EnsembleGRAPE),('risk',RiskEnsembleGRAPE)]:
        o={**options,'require_verified_gpu':False,'acceleration':{**options['acceleration'],
            'fallback_to_cpu':False,'require_accuracy_gate':False}}
        cpu=cls(pulse[0],device,training[:100],dict(o))
        model=JAXEnsemble(pulse[0],device,training[:100],o,cls)
        controls=np.asarray(pulse)[1:].T;gate=model.validate(controls)
        a,ga=cpu.value_gradient(controls);b,gb=model.value_gradient(controls)
        if abs(a-b)>2e-6 or not np.allclose(ga,gb,atol=2e-7,rtol=2e-6):
            raise RuntimeError('100-scenario CPU/GPU objective or gradient qualification failed')
        checks[name]=dict(gate=gate,gradient_max_error=float(np.max(abs(ga-gb))),scenarios=100)
        gpu_kind=model.jax_device.device_kind
    packages=environment()['packages']
    report=out/'local_gpu_certificate.json'
    write_json(report,dict(passed=True,actual_platform='gpu',precision='float64',substeps=options['substeps'],
        device=asdict(device),times_sha256=hashlib.sha256(np.asarray(pulse[0],dtype=float).tobytes()).hexdigest(),
        gpu_device_kind=gpu_kind,packages=packages,
        numerical_source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ('jax_backend.py','grape.py','risk_aware_grape.py','challenge_model.py')},
        scope='live training qualification; pointwise, not uniform convergence',previous_certificate_issue=previous_error))
    return checks,str(report)


def resolve_backend(mode,pulse,device,training,options,out):
    info=environment();info.update(requested_backend=mode,actual_backend='cpu',precision='float64 / complex128',
        cuda_backend=None,qualification_passed=False)
    if mode=='cpu':return dict(backend='cpu',precision='float64'),info,dict(status='CPU_REQUESTED')
    acceleration=dict(backend='jax_gpu',precision='float64',batch_size=32,max_vram_gb=8.,memory_fraction=.7,
        fallback_to_cpu=False,require_accuracy_gate=True,
        qualification_report=options.get('acceleration',{}).get('qualification_report'))
    started=perf_counter()
    try:
        checks,certificate=qualify(pulse,device,training,{**options,'acceleration':acceleration},out)
        acceleration['qualification_report']=certificate
        # Warm the actual training batch shape for both algorithms as shared setup.
        basis=FourierControls(pulse[0],5,device.amplitude_limit_mhz*.002*np.pi)
        coeff,_=basis.constrained_fit(np.asarray(pulse)[1:].T,device.amplitude_limit_mhz*.002*np.pi/1.03)
        for cls in (EnsembleGRAPE,RiskEnsembleGRAPE):
            model=JAXEnsemble(pulse[0],device,training,{**options,'acceleration':{**acceleration,
                'require_accuracy_gate':False}},cls)
            model.value_gradient(basis.waveform(coeff))
        info.update(actual_backend='jax_gpu',cuda_backend=model.jax_device.platform,
            gpu_model=str(model.jax_device.device_kind),compute_capability=str(getattr(model.jax_device,'compute_capability','see gpu_inventory')),
            qualification_passed=True)
        return acceleration,info,dict(status='PASS',checks=checks,shared_setup_seconds=perf_counter()-started,certificate=certificate)
    except Exception as error:
        if mode=='gpu':raise RuntimeError(f'Explicit GPU mode requires qualified FP64 CUDA: {error}') from error
        info['fallback_reason']=str(error)
        return dict(backend='cpu',precision='float64'),info,dict(status='CPU_FALLBACK',reason=str(error),shared_setup_seconds=perf_counter()-started)


def validate_full(c,base):
    expected=dict(training_samples=100,selection_samples=200,harmonics=5,restarts=5,iterations=150,
        objective_calls=1000,substeps=4,optimizer_seeds=[7500,7501,7502],
        regimes=['equal_objectives','equal_passes','equal_wall'],scenario_passes=220500,wall_seconds=1800,
        risk_weight=1.,leakage_weight=2.,continuation_schedule=[0,.1,.5,1.])
    for key,value in expected.items():
        if c[key]!=value:raise ValueError(f'Full predeclared {key} must be {value}')
    if c['mode']!='intermediate':raise ValueError('Quick/diagnostic overrides are forbidden')
    if base['quality_targets']['nominal_fidelity_min']!=.999 or base['quality_targets']['leakage_max']!=.0001:
        raise ValueError('Scientific thresholds changed')
    if base['device']['levels']!=5 or base['device']['alpha_mhz']!=-220 or base['device']['amplitude_limit_mhz']!=80 or base['gate_duration_ns']!=15.6:
        raise ValueError('Physical model differs from predeclared study')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='full_matched_config.json')
    p.add_argument('--output',default='results/full_matched_budget');p.add_argument('--backend',choices=['auto','gpu','cpu'],default='auto')
    p.add_argument('--status',action='store_true');p.add_argument('--report-only',action='store_true');p.add_argument('--max-new-runs',type=int)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.status:
        for name in ('execution_status.json','study_summary.json'):
            if (out/name).exists():print((out/name).read_text())
        return
    lock=(out/'study.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Study already running; use --status. No duplicate optimizer was launched.')
    if args.report_only:
        from .full_matched_report import export
        export(out);return
    c=load_config(args.config);base=json.loads(Path(c['base_config']).read_text());validate_full(c,base)
    c.update(backend_requested=args.backend,full_matched_outputs=True,
        additional_source_files=['full_matched_study.py','full_matched_report.py','full_matched_config.json','FULL_MATCHED_STUDY.md'])
    previous=json.loads((out/'protocol.json').read_text()) if (out/'protocol.json').exists() else None
    if previous:
        for name,digest in previous['source_hashes'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed; use a new versioned directory')
        if previous['base_config']!=base or {k:v for k,v in previous['config'].items() if k!='acceleration'}!={k:v for k,v in c.items() if k!='acceleration'}:
            raise ValueError('Frozen configuration changed; use a new versioned directory')
    device=Device(**base['device']);reference,reference_file_hash=load_reference_checkpoint(c['reference_checkpoint'])
    pulse=validate_reference_waveform(reference,base);nominal=evaluate_waveform(*pulse,device,strict=True)
    if not nominal_qualified(nominal,base['quality_targets']):raise ValueError('Stronger reference failed qualification')
    saved=reference['strict_nominal']
    if max(abs(nominal[k]-saved[k]) for k in ('fidelity','leakage'))>1e-12:raise ValueError('Stronger reference failed exact metric reproduction')
    training,selection=make_ensembles(c,base)
    if len(training)!=110 or len(selection)!=200:raise ValueError('Training or selection ensemble size changed')
    options=optimizer_options(c,base,c['optimizer_seeds'][0],c['regimes'][0],'risk')
    acceleration,hardware,qualification=resolve_backend(args.backend,pulse,device,training,options,out)
    c['acceleration']=acceleration
    if previous and previous['config']!=c:raise ValueError('Actual backend changed; cannot mix backends in an existing campaign')
    # Volatile hardware readings and setup times do not enter frozen signatures.
    setup=out/'setup_attempts';setup.mkdir(exist_ok=True)
    write_json(setup/f'{len(list(setup.glob("*.json"))):03d}.json',dict(hardware=hardware,qualification=qualification))
    if not (out/'hardware_environment.json').exists():write_json(out/'hardware_environment.json',hardware)
    if not (out/'gpu_qualification.json').exists():write_json(out/'gpu_qualification.json',qualification)
    write_json(out/'common_reference.json',dict(waveform=pulse.tolist(),waveform_sha256=waveform_hash(pulse),
        checkpoint_sha256=reference_file_hash,nominal=nominal,hardware=hardware_metrics(*pulse,device),
        calibration=reference['calibration'],calibration_fully_converged=False,source=c['reference_checkpoint']))
    write_json(out/'effective_configuration.json',dict(profile=c,base_config=base,random_training=100,deterministic_stress=10,
        independent_validation_generated=False,resolved_options={r:{m:optimizer_options(c,base,7500,r,m) for m in
            ('conventional','risk','continuation')} for r in c['regimes']}))
    print('FULL effective configuration:',json.dumps(json.loads((out/'effective_configuration.json').read_text())),flush=True)
    write_json(out/'execution_status.json',dict(status='RUNNING',pid=os.getpid(),actual_backend=hardware['actual_backend']))
    try:
        run(c,str(out),max_new_runs=args.max_new_runs)
        from .full_matched_report import export
        summary=export(out)
        write_json(out/'execution_status.json',dict(status='COMPLETE' if summary['completed_trials']==27 else 'INCOMPLETE',
            pid=os.getpid(),actual_backend=hardware['actual_backend']))
    except BaseException as error:
        write_json(out/'execution_status.json',dict(status='FAILED',pid=os.getpid(),error=repr(error),actual_backend=hardware['actual_backend']))
        raise


if __name__=='__main__':main()
