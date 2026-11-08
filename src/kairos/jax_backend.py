"""Optional batched dissipative GRAPE surrogate. Production remains QuTiP.

JAX imports are lazy. Static Liouvillians and six-state observables are built by
the existing CPU implementation, avoiding a second Hamiltonian/noise definition.
"""
import os
from time import perf_counter
import numpy as np
from scipy.special import logsumexp


class BackendUnavailable(RuntimeError):pass
class AccuracyError(RuntimeError):pass


DEFAULTS=dict(backend='cpu',precision='float64',batch_size=32,max_vram_gb=8.,
    memory_fraction=.7,fallback_to_cpu=True,require_accuracy_gate=True)
_KERNEL_CACHE={}


def settings(options):
    result={**DEFAULTS,**options.get('acceleration',{})}
    if result['backend'] not in ('cpu','jax_cpu','jax_gpu'):raise ValueError('unknown acceleration backend')
    if result['precision'] not in ('float32','float64'):raise ValueError('precision must be float32 or float64')
    if isinstance(result['batch_size'],bool) or not isinstance(result['batch_size'],int) or result['batch_size']<1:raise ValueError('batch_size must be a positive integer')
    if not 0<float(result['memory_fraction'])<1 or not 0<float(result['max_vram_gb'])<=8:raise ValueError('memory budget must fit within 8 GiB with a safety fraction')
    return result


def load_jax(platform):
    # Set before JAX initialization. Respect an explicit user allocator setting.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.linalg import expm
        jax.config.update('jax_enable_x64',True)
        devices=jax.devices(platform)
        if not devices:raise RuntimeError('no matching JAX devices')
        return jax,jnp,expm,devices[0]
    except Exception as error:
        raise BackendUnavailable(f'JAX {platform} initialization failed: {error}') from error


def plan_batch_size(requested,steps,precision,budget_bytes,free_bytes=None):
    """Conservative working-set estimate, not a measured VRAM guarantee.

    Rematerialized scan limits the tape; reserve matrices/state buffers and a
    large factor for Pade/AD/compiler temporaries. OOM still triggers halving.
    """
    item=16 if precision=='float64' else 8
    per_scenario=item*(3*625+steps*(625+150)*8+625*128)
    available=min(budget_bytes,free_bytes) if free_bytes is not None else budget_bytes
    return max(1,min(requested,int(available//max(per_scenario,1)))),per_scenario


def is_oom(error):
    text=str(error).lower()
    return any(word in text for word in ('out of memory','resource_exhausted','resource exhausted','allocation failed'))


class JAXEnsemble:
    def __init__(self,times,device,scenarios,options,cpu_class):
        self.options=dict(options);self.acceleration=settings(options)
        if device.levels!=5:raise ValueError('JAX backend currently requires five Duffing levels')
        self.jax,self.jnp,self.expm,self.jax_device=load_jax('gpu' if self.acceleration['backend']=='jax_gpu' else 'cpu')
        self.times=np.asarray(times,dtype=float);self.device=device;self.scenarios=list(scenarios)
        if not len(scenarios):raise ValueError('nonempty uncertainty ensemble required')
        if not np.all(np.diff(self.times)>0):raise ValueError('times must be strictly increasing')
        self.cpu_class=cpu_class;self.cpu=cpu_class(times,device,scenarios,dict(options))
        from .risk_aware_grape import RiskEnsembleGRAPE
        self.risk=issubclass(cpu_class,RiskEnsembleGRAPE)
        self.models=self.cpu.models
        self.packets=np.asarray(self.models)
        self.real=np.float64 if self.acceleration['precision']=='float64' else np.float32
        self.complex=np.complex128 if self.real is np.float64 else np.complex64
        self.surrogate_pass_multiplier=2 if self.risk or options.get('tail_weight',0.) else 1
        self.batch_size=self.acceleration['batch_size'];self.qualified=False;self.disabled=False
        self.info=dict(requested_backend=self.acceleration['backend'],actual_backend=self.acceleration['backend'],
            platform=self.jax_device.platform,device=str(self.jax_device.device_kind),precision=self.acceleration['precision'],
            gpu_qualified=False,oom_reductions=0,forward_scenario_passes=0,adjoint_scenario_passes=0,
            padded_forward_scenario_passes=0,batch_size=self.batch_size)
        self.backend_info=self.info
        self._build_kernels()

    def _memory_stats(self):
        try:return self.jax_device.memory_stats() or {}
        except Exception:return {}

    def _build_kernels(self):
        jax,jnp=self.jax,self.jnp
        substeps=self.options['substeps']
        if not isinstance(substeps,int) or substeps<1:raise ValueError('substeps must be positive integer')
        count=(len(self.times)-1)*substeps
        limit=int(self.acceleration['max_vram_gb']*1024**3*self.acceleration['memory_fraction'])
        stats=self._memory_stats();free=None
        if stats.get('bytes_limit') is not None:free=max(1,int(stats['bytes_limit']-stats.get('bytes_in_use',0)))
        self.batch_size,per=plan_batch_size(self.batch_size,count,self.acceleration['precision'],limit,free)
        self.info.update(batch_size=self.batch_size,estimated_bytes_per_scenario=per,memory_budget_bytes=limit)
        key=(self.times.tobytes(),substeps,self.acceleration['precision'],str(self.jax_device),
             self.cpu.initial.tobytes(),self.cpu.target.tobytes(),self.cpu.leak.tobytes())
        if key in _KERNEL_CACHE:
            self._metrics,self._states,self._weighted=_KERNEL_CACHE[key];self.info['kernel_cache_hit']=True;return
        self.info['kernel_cache_hit']=False
        expm=self.expm
        # Explicit dtypes; float32 never silently becomes the research float64 path.
        with jax.default_device(self.jax_device):
            initial=jnp.asarray(self.cpu.initial,dtype=self.complex)
            target=jnp.asarray(self.cpu.target,dtype=self.complex)
            leak=jnp.asarray(self.cpu.leak,dtype=self.complex)
            indices=jnp.asarray(np.repeat(np.arange(len(self.times)-1),substeps))
            weights=jnp.asarray(np.tile((np.arange(substeps)+.5)/substeps,len(self.times)-1),dtype=self.real)
            dt=jnp.asarray(np.repeat(np.diff(self.times)/substeps,substeps),dtype=self.real)
        def state(controls,packet):
            def step(rho,data):
                index,w,interval=data
                u=(1-w)*controls[index]+w*controls[index+1]
                generator=interval*(packet[0]+u[0]*packet[1]+u[1]*packet[2])
                with jax.default_matmul_precision('highest'):
                    return expm(generator)@rho,None
            return jax.lax.scan(jax.checkpoint(step),initial,(indices,weights,dt))[0]
        def single(controls,packet):
            rho=state(controls,packet)
            return jnp.stack((1-jnp.real(jnp.vdot(target,rho))/6,jnp.real(jnp.vdot(leak,rho))/6))
        vector=jax.vmap(single,in_axes=(None,0))
        def weighted(controls,packets,weight):
            metrics=vector(controls,packets)
            return jnp.sum(metrics*weight),metrics
        self._metrics=jax.jit(vector)
        self._states=jax.jit(jax.vmap(state,in_axes=(None,0)))
        self._weighted=jax.jit(jax.value_and_grad(weighted,has_aux=True))
        _KERNEL_CACHE[key]=(self._metrics,self._states,self._weighted)

    def _chunks(self,controls,packets,weight=None,states=False):
        """Stream host scenarios; padding avoids per-tail-shape JIT compilation."""
        result=[];gradient=np.zeros_like(controls,dtype=float)
        batch=min(self.batch_size,len(packets));self.info['last_execution_batch_size']=batch
        for start in range(0,len(packets),batch):
            part=packets[start:start+batch];valid=len(part)
            padded=np.concatenate([part,np.repeat(part[-1:],batch-valid,axis=0)])
            with self.jax.default_device(self.jax_device):
                u=self.jnp.asarray(controls,dtype=self.real);p=self.jnp.asarray(padded,dtype=self.complex)
                if weight is None:
                    output=(self._states if states else self._metrics)(u,p)
                    values=np.asarray(output)[:valid]  # synchronizes accelerator work
                else:
                    w=np.zeros((batch,2),dtype=self.real);w[:valid]=weight[start:start+valid]
                    (_,aux),g=self._weighted(u,p,self.jnp.asarray(w))
                    values=np.asarray(aux)[:valid];gradient+=np.asarray(g,dtype=float)
            result.append(values)
            self.info['forward_scenario_passes']+=valid
            self.info['padded_forward_scenario_passes']+=batch
            if weight is not None:self.info['adjoint_scenario_passes']+=valid
        return np.concatenate(result),gradient

    def _with_oom_retry(self,operation):
        while True:
            try:return operation()
            except Exception as error:
                if not is_oom(error) or self.batch_size==1 or self.info.get('last_execution_batch_size')==1:raise
                self.batch_size=max(1,self.batch_size//2);self.info['oom_reductions']+=1
                self._build_kernels()

    def metrics(self,controls,indices=None,states=False):
        packets=self.packets if indices is None else self.packets[np.asarray(indices)]
        return self._with_oom_retry(lambda:self._chunks(np.asarray(controls),packets,states=states)[0])

    def _evaluate(self,controls,indices=None):
        from .risk_aware_grape import smooth_cvar,empirical_cvar
        controls=np.asarray(controls,dtype=float)
        packets=self.packets if indices is None else self.packets[np.asarray(indices)]
        n=len(packets);mu=self.options['leakage_weight'];alpha=self.options.get('alpha',.95);tau=self.options.get('smoothing',1e-6)
        if self.risk or self.options.get('tail_weight',0.):
            metrics,_=self._chunks(controls,packets)
            loss,leak=metrics.T
            if self.risk:
                cv,risk_weights,_=smooth_cvar(loss,alpha,tau)
                worst=np.exp((loss-loss.max())/tau-logsumexp((loss-loss.max())/tau))
                wf=self.options.get('mean_weight',1.)/n+self.options['risk_weight']*risk_weights+self.options.get('worst_weight',0.)*worst
                weight=np.column_stack([wf,np.full(n,mu/n)])
                value=self.options.get('mean_weight',1.)*loss.mean()+self.options['risk_weight']*cv+mu*leak.mean()+self.options.get('worst_weight',0.)*(loss.max()+tau*logsumexp((loss-loss.max())/tau))
            else:
                combined=loss+mu*leak;count=max(1,int(np.ceil(self.options['tail_fraction']*n)))
                wf=np.full(n,(1-self.options['tail_weight'])/n);wf[np.argsort(combined)[-count:]]+=self.options['tail_weight']/count
                weight=np.column_stack([wf,mu*wf]);value=float(wf@combined)
            _,gradient=self._chunks(controls,packets,weight)
        else:
            weight=np.column_stack([np.full(n,1/n),np.full(n,mu/n)])
            metrics,gradient=self._chunks(controls,packets,weight)
            loss,leak=metrics.T;value=float(loss.mean()+mu*leak.mean())
        differences=np.diff(controls,axis=0);regularization=self.options['smoothness_weight']/len(differences)
        value+=regularization*np.sum(differences**2)
        gradient[:-1]-=2*regularization*differences;gradient[1:]+=2*regularization*differences
        self.last_ensemble_metrics=dict(mean_training_fidelity=float(1-loss.mean()),
            cvar95_training_infidelity=empirical_cvar(loss),smooth_cvar95_training_infidelity=smooth_cvar(loss,alpha,tau)[0],
            mean_training_leakage=float(leak.mean()),maximum_training_leakage=float(leak.max()))
        if self.risk:
            self.last_risk=dict(mean_infidelity=float(loss.mean()),exact_cvar=empirical_cvar(loss,alpha),
                smoothed_cvar=smooth_cvar(loss,alpha,tau)[0],smoothing_error_bound=tau*np.log(2)/(1-alpha))
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise AccuracyError('nonfinite JAX objective/gradient')
        return float(value),gradient

    def validate(self,controls):
        """Training-only device checks, never certification or validation draws."""
        from .challenge_model import evaluate_waveform
        started=perf_counter();before=self.info['forward_scenario_passes'];cpu_passes=0;production_solves=0
        indices=np.unique([0,len(self.scenarios)//2,len(self.scenarios)-1])
        measured=self.metrics(controls,indices)
        states=self.metrics(controls,indices,states=True)
        rho=states.transpose(0,2,1).reshape(-1,5,5).transpose(0,2,1)
        trace_error=float(np.max(abs(np.trace(rho,axis1=1,axis2=2)-1)))
        hermiticity_error=float(np.max(abs(rho-np.conj(rho.transpose(0,2,1)))))
        minimum_eigenvalue=float(np.linalg.eigvalsh(rho).min())
        production=[]
        for i in indices:
            production_solves+=1
            production.append(evaluate_waveform(self.times,controls[:,0],controls[:,1],self.device,self.scenarios[i],strict=True))
        self.info['qualification_production_solves']=self.info.get('qualification_production_solves',0)+production_solves
        error=max(max(abs(1-row[0]-reference['fidelity']),abs(row[1]-reference['leakage'])) for row,reference in zip(measured,production))
        cpu=self.cpu_class(self.times,self.device,[self.scenarios[i] for i in indices],dict(self.options))
        cpu_passes=len(indices)*(2 if self.risk else 1)
        self.info['qualification_cpu_passes']=self.info.get('qualification_cpu_passes',0)+cpu_passes
        cv,cg=cpu.value_gradient(controls);jv,jg=self._with_oom_retry(lambda:self._evaluate(controls,indices))
        atol=2e-5 if self.real is np.float32 else 2e-7;rtol=2e-4 if self.real is np.float32 else 2e-6
        tolerance=self.options.get('quality_targets',{}).get('solver_metric_tolerance',2e-6)
        passed=error<=tolerance and abs(cv-jv)<=tolerance and np.allclose(cg,jg,rtol=rtol,atol=atol) and trace_error<=2e-6 and hermiticity_error<=2e-6 and minimum_eigenvalue>=-2e-6
        self.info['accuracy_gate']=dict(passed=bool(passed),scenarios=indices.tolist(),production_metric_error=float(error),
            solver_metric_tolerance=tolerance,objective_error=abs(cv-jv),gradient_max_error=float(np.max(abs(cg-jg))),
            gradient_atol=atol,gradient_rtol=rtol,production_channel_solves=len(indices),
            cpu_surrogate_passes=cpu_passes,jax_forward_passes=self.info['forward_scenario_passes']-before,
            maximum_trace_error=trace_error,hermiticity_error=hermiticity_error,minimum_eigenvalue=minimum_eigenvalue,
            elapsed_seconds=perf_counter()-started)
        self.info['gpu_qualified']=bool(passed and self.jax_device.platform=='gpu')
        if not passed:raise AccuracyError(f'JAX accuracy gate failed: {self.info["accuracy_gate"]}')
        self.qualified=True
        return self.info['accuracy_gate']

    def value_gradient(self,controls):
        before=(self.info['forward_scenario_passes'],self.info.get('qualification_cpu_passes',0),self.info.get('cpu_fallback_passes',0),self.info.get('qualification_production_solves',0))
        try:
            if self.disabled:return self._cpu_fallback(controls)
            if not self.qualified and self.acceleration['require_accuracy_gate']:self.validate(controls)
            return self._with_oom_retry(lambda:self._evaluate(controls))
        except Exception as error:
            if self.disabled or not self.acceleration['fallback_to_cpu'] or self.options.get('require_verified_gpu',False):raise
            import warnings
            warnings.warn(f'JAX execution failed; returning trusted CPU GRAPE results: {error}',RuntimeWarning)
            self.disabled=True;self.info.update(actual_backend='cpu',effective_precision='float64',fallback_reason=str(error))
            return self._cpu_fallback(controls)
        finally:
            self.last_call_work=dict(surrogate_passes=self.info['forward_scenario_passes']-before[0]+
                self.info.get('qualification_cpu_passes',0)-before[1]+self.info.get('cpu_fallback_passes',0)-before[2],
                production_solves=self.info.get('qualification_production_solves',0)-before[3])

    def pending_qualification_work(self):
        if self.disabled or self.qualified or not self.acceleration['require_accuracy_gate']:
            return dict(surrogate_passes=0,production_solves=0)
        n=min(3,len(self.scenarios));mult=self.surrogate_pass_multiplier
        return dict(surrogate_passes=n*(2+mult+(2 if self.risk else 1)),production_solves=n)

    def _cpu_fallback(self,controls):
        self.info['cpu_fallback_passes']=self.info.get('cpu_fallback_passes',0)+len(self.scenarios)*(2 if self.risk else 1)
        value,g=self.cpu.value_gradient(controls);self.last_ensemble_metrics=self.cpu.last_ensemble_metrics
        if hasattr(self.cpu,'last_risk'):self.last_risk=self.cpu.last_risk
        return value,g

    def diagnostics(self):
        info=dict(self.info);info['memory_stats']={k:int(v) for k,v in self._memory_stats().items() if isinstance(v,(int,np.integer))}
        return info


def make_model(times,device,scenarios,options,cpu_class):
    config=settings(options)
    if config['backend']=='cpu':return cpu_class(times,device,scenarios,options)
    if options.get('require_verified_gpu') and not config['require_accuracy_gate']:
        raise AccuracyError('Full research requires the GPU accuracy gate')
    try:return JAXEnsemble(times,device,scenarios,options,cpu_class)
    except (BackendUnavailable,RuntimeError) as error:
        if not config['fallback_to_cpu'] or options.get('require_verified_gpu',False):raise
        import warnings
        warnings.warn(f'{error}; falling back to CPU QuTiP/SciPy GRAPE',RuntimeWarning)
        model=cpu_class(times,device,scenarios,options)
        model.backend_info=dict(requested_backend=config['backend'],actual_backend='cpu',effective_precision='float64',fallback_reason=str(error),gpu_qualified=False)
        return model


def gpu_preflight(pulse,device,scenarios,options):
    """Fail closed before expensive full work when CUDA is unavailable/unqualified."""
    from .grape import EnsembleGRAPE
    from .risk_aware_grape import RiskEnsembleGRAPE
    result={}
    options={**options,'require_verified_gpu':True}
    if not settings(options)['require_accuracy_gate']:raise AccuracyError('Full GPU research requires accuracy tests')
    from pathlib import Path
    from dataclasses import asdict
    import hashlib,json
    path=options.get('acceleration',{}).get('qualification_report')
    if not path or not Path(path).is_file():raise AccuracyError('Full GPU research requires a successful CUDA qualification report from jax_benchmark.py')
    certificate=json.loads(Path(path).read_text())
    if not certificate.get('passed') or certificate.get('actual_platform')!='gpu':raise AccuracyError('Qualification report does not establish successful CUDA accuracy tests')
    if certificate['precision']!=settings(options)['precision'] or certificate['substeps']!=options['substeps'] or certificate['device']!=asdict(device):raise AccuracyError('GPU qualification precision/model/substeps differ from this campaign')
    if certificate['times_sha256']!=hashlib.sha256(np.asarray(pulse[0],dtype=float).tobytes()).hexdigest():raise AccuracyError('GPU qualification gate duration/grid differs')
    for name,value in certificate['numerical_source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=value:raise AccuracyError('GPU numerical source changed since qualification')
    import importlib.metadata
    for name,version in certificate['packages'].items():
        if importlib.metadata.version(name)!=version:raise AccuracyError('Numerical dependency versions differ from GPU qualification')
    for name,cls in [('conventional',EnsembleGRAPE),('risk',RiskEnsembleGRAPE)]:
        model=make_model(pulse[0],device,scenarios,{**options,'risk_weight':1.},cls)
        if not isinstance(model,JAXEnsemble) or model.jax_device.platform!='gpu':raise BackendUnavailable('Verified CUDA execution required before a full GPU campaign')
        if str(model.jax_device.device_kind)!=certificate['gpu_device_kind']:raise AccuracyError('GPU hardware differs from the qualified device')
        model.validate(np.asarray(pulse)[1:].T);result[name]=model.diagnostics()
    return result
