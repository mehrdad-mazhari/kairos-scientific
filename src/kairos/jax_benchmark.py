"""Numerical/backend benchmark, not a research campaign or final validation.

CUDA rows never fall back: missing hardware produces null measurements.
The optional JAX-CPU control separates software speedups from GPU speedups.
"""
import argparse,csv,hashlib,importlib.metadata,json,os,platform as host_platform,subprocess,sys,threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
from .challenge_model import Device,make_pulse,PulseParameters,monte_carlo_scenarios,evaluate_waveform,hardware_metrics,MHZ_TO_RAD_NS
from .grape import DEFAULTS,FourierControls,EnsembleGRAPE
from .risk_aware_grape import RiskEnsembleGRAPE
from .jax_backend import JAXEnsemble,BackendUnavailable


def write_json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def rss_bytes():
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    except OSError:pass
    return None


def timed(function):
    samples=[rss_bytes()];stop=threading.Event()
    def monitor():
        while not stop.wait(.02):samples.append(rss_bytes())
    thread=threading.Thread(target=monitor,daemon=True);thread.start();started=perf_counter()
    try:result=function()
    finally:
        seconds=perf_counter()-started;stop.set();thread.join();samples.append(rss_bytes())
    valid=[n for n in samples if n is not None]
    return result,dict(seconds=seconds,process_rss_peak_bytes=max(valid) if valid else None,
        process_rss_start_bytes=samples[0],rss_scope='sampled process RSS including interpreter/models/JIT caches; not VRAM')


def inventory():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    except (OSError,subprocess.TimeoutExpired) as error:return dict(returncode=None,error=str(error))


def checkpoint(out,name,function):
    path=out/name
    if path.exists():return json.loads(path.read_text())
    value=function();write_json(path,value);return value


def run(c,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);base=json.loads(Path(c['base_config']).read_text());device=Device(**base['device'])
    if device.levels!=5:raise ValueError('benchmark requires the five-level production model')
    counts=c['scenario_counts']
    if not counts or any(not isinstance(n,int) or n<1 for n in counts):raise ValueError('positive scenario counts required')
    if not isinstance(c['repetitions'],int) or c['repetitions']<1:raise ValueError('positive repetition count required')
    source_names=['jax_backend.py','jax_benchmark.py','grape.py','risk_aware_grape.py','challenge_model.py']
    protocol=dict(config=c,base_config=base,sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_names},
        purpose='backend_numerical_benchmark_not_statistical_validation',certification_backend='cpu_qutip')
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('benchmark source/config changed; choose new output directory')
    else:write_json(out/'protocol.json',protocol)
    package_names=['numpy','scipy','qutip','jax','jaxlib']
    package_names+=sorted(d.metadata['Name'] for d in importlib.metadata.distributions() if d.metadata['Name'].lower().startswith(('jax-cuda','nvidia-')))
    info=dict(nvidia_smi=inventory(),python=sys.version,host=host_platform.platform(),
        cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None),
        packages={name:importlib.metadata.version(name) for name in package_names},
        environment={key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','XLA_FLAGS','XLA_PYTHON_CLIENT_PREALLOCATE')})
    write_json(out/'environment.json',info);print('Effective benchmark configuration:',json.dumps(c),flush=True)
    scenarios=monte_carlo_scenarios(max(100,max(counts)),c['scenario_seed'],base['coherence_min'],**base['manufacturing_ranges'])
    write_json(out/'benchmark_scenarios.json',[asdict(s) for s in scenarios])
    seed=np.asarray(make_pulse(PulseParameters(base['gate_duration_ns'],.3,.5,1.,.3),device))
    basis=FourierControls(seed[0],5,device.amplitude_limit_mhz*MHZ_TO_RAD_NS)
    coeff,_=basis.constrained_fit(seed[1:].T,device.amplitude_limit_mhz*MHZ_TO_RAD_NS/max(base['amplitude_headroom_scale'],1+base['manufacturing_ranges']['amplitude_fraction']))
    controls=basis.waveform(coeff);wave=np.array([seed[0],*controls.T])
    write_json(out/'pulse.json',dict(waveform=wave.tolist(),coefficients=coeff.tolist(),hardware=hardware_metrics(*wave,device)))
    rows=[];cpu={};gpu_missing=None
    for n in counts:
        print('CPU QuTiP forward:',n,flush=True)
        def production(n=n):
            observations,cost=timed(lambda:[evaluate_waveform(*wave,device,s,strict=True) for s in scenarios[:n]])
            return dict(observations=observations,cost=cost)
        result=checkpoint(out,f'cpu_qutip_{n}.json',production);cpu[n]=result
        rows.append(dict(backend='cpu_qutip',platform='cpu',precision='float64',operation='forward',scenarios=n,status='measured',
            runtime_seconds=result['cost']['seconds'],measured_gpu_speedup=None,gpu_memory_peak_bytes=None,**{k:v for k,v in result['cost'].items() if k!='seconds'}))
        csv_write(out/'benchmark_results.csv',rows)
    options={**DEFAULTS,'substeps':c['substeps'],'alpha':.95,'risk_weight':1.,'smoothing':1e-6,'quality_targets':base['quality_targets']}
    gradient_reference={}
    # Always verify actual gradients on >=100 distinct benchmark scenarios.
    validation_n=c['gradient_validation_scenarios']
    if validation_n<100 or validation_n>len(scenarios):raise ValueError('gradient verification needs at least 100 available scenarios')
    for name,cls in [('conventional',EnsembleGRAPE),('risk',RiskEnsembleGRAPE)]:
        for n in sorted(set(([validation_n]+counts) if c['benchmark_cpu_gradients'] else [validation_n])):
            print('CPU analytic gradient:',name,n,flush=True)
            def analytic(n=n,cls=cls):
                started=perf_counter();model=cls(wave[0],device,scenarios[:n],dict(options));setup=perf_counter()-started
                (value,g),cost=timed(lambda:model.value_gradient(controls))
                return dict(value=value,gradient=g.tolist(),cost=cost,model_setup_seconds=setup)
            result=checkpoint(out,f'cpu_gradient_{name}_{n}.json',analytic);gradient_reference[(name,n)]=result
            if c['benchmark_cpu_gradients']:
                rows.append(dict(backend='cpu_scipy',platform='cpu',precision='float64',operation=name+'_gradient',scenarios=n,status='measured',
                    runtime_seconds=result['cost']['seconds'],model_setup_seconds=result['model_setup_seconds'],measured_gpu_speedup=None,
                    **{k:v for k,v in result['cost'].items() if k!='seconds'}))
    validations=[]
    platforms=['jax_cpu','jax_gpu'] if c['include_jax_cpu'] else ['jax_gpu']
    direction=np.random.default_rng(c['direction_seed']).normal(size=10);direction/=np.linalg.norm(direction)
    validation_reference=np.array([[1-r['fidelity'],r['leakage']] for r in cpu[validation_n]['observations']]) if validation_n in cpu else None
    if validation_reference is None:
        validation_reference=np.array([[1-r['fidelity'],r['leakage']] for r in [evaluate_waveform(*wave,device,s,strict=True) for s in scenarios[:validation_n]]])
    for platform in platforms:
        for precision in c['precisions']:
            accel=dict(backend=platform,precision=precision,batch_size=c['batch_size'],max_vram_gb=c['max_vram_gb'],
                memory_fraction=c['memory_fraction'],fallback_to_cpu=False,require_accuracy_gate=False)
            started=perf_counter()
            try:model=JAXEnsemble(wave[0],device,scenarios,{**options,'acceleration':accel},RiskEnsembleGRAPE)
            except BackendUnavailable as error:
                if platform=='jax_cpu':raise
                gpu_missing=str(error)
                for n in counts:
                    for operation in ('forward','conventional_gradient','risk_gradient'):
                        rows.append(dict(backend=platform,platform='unavailable',precision=precision,operation=operation,scenarios=n,
                            status='GPU_UNAVAILABLE',runtime_seconds=None,measured_gpu_speedup=None,gpu_memory_peak_bytes=None,reason=str(error)))
                csv_write(out/'benchmark_results.csv',rows);continue
            setup_seconds=perf_counter()-started
            control_counts=c.get('jax_cpu_counts',counts) if platform=='jax_cpu' else counts
            if not set(control_counts).issubset(counts):raise ValueError('JAX CPU control counts must be requested benchmark sizes')
            for n in control_counts:
                indices=np.arange(n)
                for operation in ('forward','conventional_gradient','risk_gradient'):
                    path=out/f'{platform}_{precision}_{operation}_{n}.json'
                    model.risk=operation!='conventional_gradient'
                    def execute():
                        if operation=='forward':return model.metrics(controls,indices)
                        return model._with_oom_retry(lambda:model._evaluate(controls,indices))
                    print(platform,precision,operation,n,flush=True)
                    if path.exists():data=json.loads(path.read_text())
                    else:
                        first,first_cost=timed(execute);times=[];peak=0
                        for _ in range(c['repetitions']):
                            result,cost=timed(execute);times.append(cost['seconds']);peak=max(peak,cost['process_rss_peak_bytes'] or 0)
                        data=dict(first_call_seconds=first_cost['seconds'],warm_seconds=float(np.median(times)),
                            repetitions=c['repetitions'],process_rss_peak_bytes=peak,model_setup_seconds=setup_seconds,
                            diagnostics=model.diagnostics(),value=None if operation=='forward' else result[0],
                            metrics=result.tolist() if operation=='forward' else None,
                            gradient=result[1].tolist() if operation!='forward' else None)
                        write_json(path,data)
                    error=None;gradient_error=None;accurate=None
                    if operation=='forward':
                        reference=np.array([[1-r['fidelity'],r['leakage']] for r in cpu[n]['observations']]);actual=np.asarray(data['metrics'])
                        error=float(np.max(abs(actual-reference)));accurate=error<=base['quality_targets']['solver_metric_tolerance']
                        baseline=cpu[n]['cost']['seconds']
                    else:
                        key=(operation.removesuffix('_gradient'),n);ref=gradient_reference.get(key);baseline=None
                        if ref:
                            gradient_error=float(np.max(abs(np.asarray(data['gradient'])-np.asarray(ref['gradient']))))
                            accurate=bool(np.allclose(data['gradient'],ref['gradient'],atol=2e-5 if precision=='float32' else 2e-7,rtol=2e-4 if precision=='float32' else 2e-6))
                            baseline=ref['cost']['seconds'] if c['benchmark_cpu_gradients'] else None
                    stats=data['diagnostics'].get('memory_stats',{})
                    row=dict(backend=platform,platform=model.jax_device.platform,precision=precision,operation=operation,scenarios=n,status='measured',
                        gpu_device_kind=str(model.jax_device.device_kind) if platform=='jax_gpu' else None,
                        runtime_seconds=data['warm_seconds'],first_call_seconds=data['first_call_seconds'],model_setup_seconds=data['model_setup_seconds'],
                        process_rss_peak_bytes=data['process_rss_peak_bytes'],gpu_memory_peak_bytes=stats.get('peak_bytes_in_use') if platform=='jax_gpu' else None,
                        gpu_peak_scope='JAX allocator cumulative process peak, not an isolated per-kernel allocation',
                        fidelity_leakage_max_error=error,gradient_max_error=gradient_error,accuracy_passed=accurate,
                        batch_size=data['diagnostics']['last_execution_batch_size'],estimated_bytes_per_scenario=data['diagnostics']['estimated_bytes_per_scenario'],
                        measured_gpu_speedup=baseline/data['warm_seconds'] if platform=='jax_gpu' and baseline is not None and accurate else None)
                    rows.append(row);csv_write(out/'benchmark_results.csv',rows)
            # Trace/positivity, finite differences and objectives on the same 100 scenarios.
            states=model.metrics(controls,np.arange(validation_n),states=True)
            rho=states.transpose(0,2,1).reshape(-1,5,5).transpose(0,2,1)
            trace_error=float(np.max(abs(np.trace(rho,axis1=1,axis2=2)-1)))
            hermiticity_error=float(np.max(abs(rho-np.conj(rho.transpose(0,2,1)))))
            minimum_eigenvalue=float(np.linalg.eigvalsh(rho).min())
            for kind in ('conventional','risk'):
                model.risk=kind=='risk';indices=np.arange(validation_n)
                value,g=model._with_oom_retry(lambda:model._evaluate(controls,indices));reference=gradient_reference[(kind,validation_n)]
                metric=model.metrics(controls,indices);metric_error=float(np.max(abs(metric-validation_reference)))
                eps=1e-3 if precision=='float32' else 1e-5
                plus=basis.waveform(coeff+eps*direction);minus=basis.waveform(coeff-eps*direction)
                fd=(model._with_oom_retry(lambda:model._evaluate(plus,indices))[0]-model._with_oom_retry(lambda:model._evaluate(minus,indices))[0])/(2*eps)
                analytic=float(basis.gradient(g)@direction)
                gradient_ok=np.allclose(g,reference['gradient'],atol=2e-5 if precision=='float32' else 2e-7,rtol=2e-4 if precision=='float32' else 2e-6)
                passed=metric_error<=base['quality_targets']['solver_metric_tolerance'] and gradient_ok and abs(value-reference['value'])<=base['quality_targets']['solver_metric_tolerance'] and trace_error<=2e-6 and hermiticity_error<=2e-6 and minimum_eigenvalue>=-2e-6
                if precision=='float64':passed=passed and abs(fd-analytic)<2e-6
                validations.append(dict(backend=platform,actual_platform=model.jax_device.platform,precision=precision,objective=kind,scenarios=validation_n,
                    passed=bool(passed),production_metric_max_error=metric_error,objective_error=abs(value-reference['value']),
                    gradient_max_error=float(np.max(abs(g-np.asarray(reference['gradient'])))),
                    analytic_directional_gradient=analytic,finite_difference=fd,finite_difference_error=abs(fd-analytic),
                    maximum_trace_error=trace_error,hermiticity_error=hermiticity_error,minimum_eigenvalue=minimum_eigenvalue,
                    finite_difference_role='primary' if precision=='float64' else 'informational: cancellation in float32; CPU analytic comparison is primary'))
            write_json(out/'numerical_equivalence.json',validations)
    # A hardware-matched speedup is distinct from a change of solver/software.
    for row in rows:
        if row['backend']=='jax_gpu' and row['status']=='measured':
            cpu_control=next((r for r in rows if r['backend']=='jax_cpu' and r['precision']==row['precision'] and r['operation']==row['operation'] and r['scenarios']==row['scenarios']),None)
            row['matched_jax_cpu_to_gpu_speedup']=cpu_control['runtime_seconds']/row['runtime_seconds'] if cpu_control and row['accuracy_passed'] else None
    csv_write(out/'benchmark_results.csv',rows)
    for precision in c['precisions']:
        forward=[r for r in rows if r['backend']=='jax_gpu' and r['precision']==precision and r['operation']=='forward']
        checks=[v for v in validations if v['backend']=='jax_gpu' and v['precision']==precision]
        passed=bool(len(checks)==2 and all(v['passed'] for v in checks) and {1,10,100,500,1000,2000}.issubset({r['scenarios'] for r in forward if r.get('accuracy_passed')}))
        write_json(out/f'gpu_qualification_{precision}.json',dict(passed=passed,actual_platform='gpu' if passed else 'unverified',
            precision=precision,substeps=c['substeps'],device=asdict(device),times_sha256=hashlib.sha256(np.asarray(wave[0],dtype=float).tobytes()).hexdigest(),
            gpu_device_kind=next((r.get('gpu_device_kind') for r in forward if r['status']=='measured'),None),
            numerical_source_hashes={name:protocol['sources'][name] for name in ('jax_backend.py','grape.py','risk_aware_grape.py','challenge_model.py')},
            packages=info['packages'],note='Numerical qualification only; not statistical certification. CPU-only results cannot authorize CUDA research.'))
    write_json(out/'status.json',dict(gpu_available=gpu_missing is None,gpu_unavailable_reason=gpu_missing,
        gpu_speedup_measured=any(r['backend']=='jax_gpu' and r['status']=='measured' for r in rows),
        all_requested_counts=counts,full_research_campaign_executed=False,final_validation_generated=False,certification_backend='cpu_qutip'))
    from .jax_benchmark_report import render
    render(out,rows,validations,c,base)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='jax_benchmark_config.json');p.add_argument('--output',required=True)
    p.add_argument('--no-cpu-gradients',action='store_true',help='Explicitly skip large SciPy gradient timings; still verify both gradients on 100 scenarios')
    args=p.parse_args();c=json.loads(Path(args.config).read_text())
    if args.no_cpu_gradients:c['benchmark_cpu_gradients']=False
    run(c,args.output)
