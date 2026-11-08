"""Opt-in qualified Fourier optimization, shared resource limits and telemetry.

Production nominal qualification is a retention rule, not a differentiable
constraint inserted into the training objective. No selection data enter here.
"""
from dataclasses import dataclass,field
import hashlib
from time import perf_counter
import warnings
import numpy as np
from scipy.optimize import minimize,nnls
from challenge_model import evaluate_waveform,hardware_metrics,MHZ_TO_RAD_NS
from grape import FourierControls,EnsembleGRAPE


class BudgetExhausted(RuntimeError):
    pass


def waveform_hash(wave):
    return hashlib.sha256(np.asarray(wave,dtype=np.float64).tobytes()).hexdigest()


@dataclass
class ResourceBudget:
    """One ledger across all starts/stages; failed attempts are never refunded.

    A scenario pass is one six-input dissipative forward/adjoint propagation.
    Production channel propagations count separately and also toward total
    dissipative passes. These units are not FLOP-equivalent; wall comparison
    provides a separate empirical comparison. Wall caps are checked between calls.
    """
    objective_limit:int=1000
    pass_limit:float=float('inf')
    wall_limit:float=float('inf')
    iteration_limit:int=150
    started:float=field(default_factory=perf_counter)
    objective_evaluations:int=0
    surrogate_scenario_passes:int=0
    production_channel_solves:int=0
    iterations:int=0
    nominal_cache:dict=field(default_factory=dict)

    @property
    def elapsed(self):return perf_counter()-self.started

    @property
    def passes(self):return self.surrogate_scenario_passes+self.production_channel_solves

    def check(self,objectives=0,passes=0):
        if self.elapsed>=self.wall_limit:raise BudgetExhausted('wall-clock budget exhausted between calls')
        if self.objective_evaluations+objectives>self.objective_limit:raise BudgetExhausted('objective evaluation budget exhausted')
        if self.passes+passes>self.pass_limit:raise BudgetExhausted('total dissipative scenario-pass budget exhausted')

    def nominal(self,wave,device):
        key=waveform_hash(wave)
        if key not in self.nominal_cache:
            self.check(passes=1);self.production_channel_solves+=1
            self.nominal_cache[key]=evaluate_waveform(*wave,device,strict=True)
        return self.nominal_cache[key]

    def snapshot(self):
        return dict(objective_evaluations=self.objective_evaluations,
            gradient_evaluations=self.objective_evaluations,
            surrogate_scenario_passes=self.surrogate_scenario_passes,
            production_channel_solves=self.production_channel_solves,
            total_dissipative_passes=self.passes,solver_iterations=self.iterations,
            elapsed_seconds=self.elapsed)


def nominal_qualified(metrics,quality):
    return bool(metrics['fidelity']>=quality['nominal_fidelity_min'] and
                metrics['leakage']<=quality['leakage_max'])


def qualified_fourier_multistart(pulse,device,options,amplitude_scale,budget=None):
    """Five-harmonic fit plus seeded, adaptively shrunk qualified perturbations."""
    if options['harmonics']!=5:raise ValueError('qualified_fourier_multistart requires five harmonics')
    family=options.get('initialization_reference_family','calibrated_drag')
    if family not in ('calibrated_drag','calibrated_wah_wah'):
        raise ValueError('qualified Fourier initialization requires a calibrated DRAG or Wah-Wah reference')
    quality=options.get('quality_targets',dict(nominal_fidelity_min=.999,leakage_max=.0001))
    budget=budget if budget is not None else ResourceBudget()
    times=np.asarray(pulse[0]);scale=device.amplitude_limit_mhz*MHZ_TO_RAD_NS
    basis=FourierControls(times,5,scale);limit=scale/amplitude_scale
    source=budget.nominal(pulse,device)
    if not nominal_qualified(source,quality):raise ValueError('source pulse is not nominally qualified')
    warm=options.get('warm_start_coefficients')
    if warm is None:coeff,fit=basis.constrained_fit(np.asarray(pulse)[1:].T,limit)
    else:
        coeff=np.asarray(warm,dtype=float)
        if coeff.shape!=(10,) or not np.allclose(basis.waveform(coeff),np.asarray(pulse)[1:].T,rtol=0,atol=1e-12):
            raise ValueError('warm coefficients must reproduce the transferred waveform')
        fit=dict(success=True,message='Exact transferred Fourier coefficients')
    def waveform(c):return np.array([times,*basis.waveform(c).T])
    records=[];starts=[]
    def qualify(c,restart,attempt,magnitude):
        wave=waveform(c);hardware=hardware_metrics(*wave,device)
        amplitude_ok=bool(np.min(basis.constraints(c,limit))>=-1e-10 and hardware['endpoint_amplitude_rad_ns']<=1e-10)
        metric=budget.nominal(wave,device) if amplitude_ok else None
        accepted=amplitude_ok and nominal_qualified(metric,quality)
        records.append(dict(restart=restart,attempt=attempt,perturbation_magnitude=magnitude,
            accepted=bool(accepted),amplitude_feasible=amplitude_ok,waveform_sha256=waveform_hash(wave),
            coefficients=c.tolist(),nominal_fidelity=None if metric is None else metric['fidelity'],
            nominal_leakage=None if metric is None else metric['leakage'],
            peak_amplitude_mhz=hardware['peak_amplitude_mhz'],endpoint_amplitude=hardware['endpoint_amplitude_rad_ns']))
        return accepted
    if not qualify(coeff,0,0,0.):raise ValueError('unperturbed constrained Fourier fit fails nominal qualification')
    starts.append(coeff.copy())
    rng=np.random.default_rng(options['seed'])
    magnitude=options.get('qualified_perturbation_scale',.002)
    if not np.isfinite(magnitude) or magnitude<=0:raise ValueError('qualified perturbation scale must be positive')
    for restart in range(1,options.get('restarts',5)):
        direction=rng.normal(size=10);direction/=np.linalg.norm(direction)
        for attempt in range(options.get('qualification_attempts',12)):
            step=magnitude*options.get('qualification_shrink',.5)**attempt
            candidate=coeff+step*direction
            if qualify(candidate,restart,attempt,step):
                starts.append(candidate);break
    return basis,starts,records,fit


def convergence_assessment(status,gradient,coefficients,basis,limit,last_changes,options):
    """Local KKT check with nonnegative active amplitude multipliers."""
    constraint=basis.constraints(coefficients,limit)
    active=constraint<options.get('active_constraint_tolerance',1e-6)
    projected=np.asarray(gradient)
    if np.any(active) and np.all(np.isfinite(projected)):
        jac=basis.constraint_gradient(coefficients,limit)[active]
        multipliers,_=nnls(jac.T,projected)
        projected=projected-jac.T@multipliers
    relative,parameter=last_changes
    checks=dict(optimizer_success=bool(status),constraint_satisfied=bool(constraint.min()>=-1e-8),
        projected_gradient_small=bool(np.linalg.norm(projected)<=options.get('gradient_tolerance',1e-4)),
        relative_objective_stable=bool(relative<=options.get('relative_objective_tolerance',1e-7)),
        parameter_stable=bool(parameter<=options.get('parameter_tolerance',1e-5)))
    finite=lambda x:float(x) if np.isfinite(x) else None
    return dict(converged=all(checks.values()),criteria=checks,gradient_norm=finite(np.linalg.norm(gradient)),
        projected_gradient_norm=finite(np.linalg.norm(projected)),relative_objective_change=finite(relative),
        relative_parameter_change=finite(parameter))


def refine(pulse,device,scenarios,options,amplitude_scale,*,model_class=EnsembleGRAPE,budget=None,quota=None):
    """Qualified retention with aggregate (not per-restart) budgets and exact starts."""
    options=dict(options);quality=options.get('quality_targets',dict(nominal_fidelity_min=.999,leakage_max=.0001))
    budget=budget if budget is not None else ResourceBudget(
        objective_limit=options.get('max_objective_evaluations',1000),
        pass_limit=options.get('max_total_scenario_passes',options.get('max_surrogate_passes',float('inf'))),
        wall_limit=options.get('max_wall_seconds',float('inf')),
        iteration_limit=options.get('max_total_iterations',options.get('iterations',150)))
    before=budget.snapshot();quota=quota or {}
    end_calls=min(budget.objective_limit,budget.objective_evaluations+quota.get('objectives',budget.objective_limit))
    end_passes=min(budget.pass_limit,budget.passes+quota.get('passes',budget.pass_limit))
    end_iterations=min(budget.iteration_limit,budget.iterations+quota.get('iterations',budget.iteration_limit))
    end_wall=min(budget.wall_limit,budget.elapsed+quota.get('wall',budget.wall_limit))
    basis,starts,initializations,fit=qualified_fourier_multistart(pulse,device,options,amplitude_scale,budget)
    times=np.asarray(pulse[0]);limit=device.amplitude_limit_mhz*MHZ_TO_RAD_NS/amplitude_scale
    from jax_backend import make_model
    model=make_model(times,device,scenarios,options,model_class)
    multiplier=getattr(model,'surrogate_pass_multiplier',1 if model_class is EnsembleGRAPE else 2)
    best_coefficients=starts[0].copy();best_value=float('inf');best_gradient=None;best_metrics=None
    trace=[];accepted=[];history=[];terminations=[];first_value=None;exhaustion=None
    initial_hash=waveform_hash([times,*basis.waveform(starts[0]).T]);restart_outcomes=[]
    def local_check(calls=0,passes=0):
        budget.check(calls,passes)
        if budget.objective_evaluations+calls>end_calls:raise BudgetExhausted('stage objective allocation exhausted')
        if budget.passes+passes>end_passes:raise BudgetExhausted('stage scenario-pass allocation exhausted')
        if budget.elapsed>=end_wall:raise BudgetExhausted('stage wall allocation exhausted')
    def wave(c):return np.array([times,*basis.waveform(c).T])
    for restart,initial in enumerate(starts):
        cache={};callback_points=[];callback_losses=[]
        # Reserve a fair share for later starts, while carrying unused work forward.
        remaining=len(starts)-restart
        call_end=budget.objective_evaluations+max(1,int(np.ceil((end_calls-budget.objective_evaluations)/remaining)))
        iteration_end=budget.iterations+max(1,int(np.ceil((end_iterations-budget.iterations)/remaining)))
        def objective(c):
            nonlocal best_value,best_coefficients,best_gradient,best_metrics,first_value
            key=c.tobytes()
            if key in cache:return cache[key]
            pending=model.pending_qualification_work() if hasattr(model,'pending_qualification_work') else dict(surrogate_passes=0,production_solves=0)
            local_check(1,len(scenarios)*multiplier+pending['surrogate_passes']+pending['production_solves'])
            if budget.objective_evaluations>=call_end:raise BudgetExhausted('restart objective allocation exhausted')
            budget.objective_evaluations+=1;budget.surrogate_scenario_passes+=len(scenarios)*multiplier
            try:value,g=model.value_gradient(basis.waveform(c))
            finally:
                work=getattr(model,'last_call_work',dict(surrogate_passes=len(scenarios)*multiplier,production_solves=0))
                budget.surrogate_scenario_passes+=work['surrogate_passes']-len(scenarios)*multiplier
                budget.production_channel_solves+=work['production_solves']
            gradient=basis.gradient(g)
            metrics=dict(model.last_ensemble_metrics)
            differences=np.diff(basis.waveform(c),axis=0)
            metrics['regularization']=float(options['smoothness_weight']*np.sum(differences**2)/len(differences))
            if first_value is None:first_value=float(value)
            feasible=bool(basis.constraints(c,limit).min()>=-1e-10)
            retained=False
            if feasible and value<best_value:
                # Nominal checks use the production solver; failed checks consume cost.
                local_check(passes=0 if waveform_hash(wave(c)) in budget.nominal_cache else 1)
                nominal=budget.nominal(wave(c),device)
                if nominal_qualified(nominal,quality):
                    retained=True;best_value=float(value);best_coefficients=c.copy();best_gradient=gradient.copy();best_metrics=metrics
                    hardware=hardware_metrics(*wave(c),device)
                    accepted.append(dict(event='initial' if len(accepted)==0 else 'retained_feasible_improvement',
                        restart=restart,objective=float(value),gradient_norm=float(np.linalg.norm(gradient)),
                        nominal_fidelity=nominal['fidelity'],nominal_leakage=nominal['leakage'],
                        peak_amplitude_mhz=hardware['peak_amplitude_mhz'],waveform_sha256=waveform_hash(wave(c)),
                        coefficients=c.tolist(),**metrics,**budget.snapshot()))
            trace.append(dict(restart=restart,objective=float(value),gradient_norm=float(np.linalg.norm(gradient)),
                amplitude_feasible=feasible,retained=retained,coefficients=c.tolist(),**budget.snapshot()))
            cache.clear();cache[key]=(value,gradient)
            return cache[key]
        def callback(c):
            if budget.iterations>=end_iterations:raise BudgetExhausted('aggregate iteration budget exhausted')
            budget.iterations+=1
            value,_=objective(c);callback_points.append(c.copy());callback_losses.append(value)
        result=None;reason=None
        try:
            initial_value,_=objective(initial)
            callback_points.append(initial.copy());callback_losses.append(initial_value)
            result=minimize(objective,initial,method='SLSQP',jac=True,
                constraints={'type':'ineq','fun':lambda c:basis.constraints(c,limit),'jac':lambda c:basis.constraint_gradient(c,limit)},
                callback=callback,options={'maxiter':max(1,min(options['iterations'],iteration_end-budget.iterations)),'ftol':options.get('ftol',1e-9)})
            reason=str(result.message)
        except BudgetExhausted as error:
            reason=str(error)
            if not reason.startswith('restart '):exhaustion=reason
        relative=parameter=float('inf')
        if len(callback_points)>=2:
            relative=abs(callback_losses[-1]-callback_losses[-2])/max(abs(callback_losses[-2]),1e-12)
            parameter=np.linalg.norm(callback_points[-1]-callback_points[-2])/(1+np.linalg.norm(callback_points[-2]))
        status=convergence_assessment(result is not None and result.success,
            best_gradient if best_gradient is not None else np.full(len(initial),np.inf),best_coefficients,basis,limit,(relative,parameter),options)
        # An earlier retained point cannot inherit stationarity of another terminal point.
        if result is not None and not np.allclose(result.x,best_coefficients,rtol=0,atol=1e-10):
            status['converged']=False;status['criteria']['returned_point_is_solver_terminal']=False
        restart_outcomes.append(dict(restart=restart,termination=reason,returned_waveform_sha256=waveform_hash(wave(best_coefficients)),**status));terminations.append(reason)
        if exhaustion:break
        if budget.iterations>=end_iterations:
            exhaustion='aggregate iteration budget exhausted';break
    if best_metrics is None:
        warnings.warn('No objective-evaluated qualified improvement; returning the qualified unperturbed fit',RuntimeWarning)
    delta=float(np.linalg.norm(best_coefficients-starts[0]))
    if delta<=1e-8*(1+np.linalg.norm(starts[0])):
        warnings.warn('GRAPE returned its initial Fourier waveform without meaningful coefficient changes',RuntimeWarning)
    terminal_converged=any(r['converged'] and r['returned_waveform_sha256']==waveform_hash(wave(best_coefficients)) for r in restart_outcomes) and exhaustion is None
    history.append(dict(kind='qualified research Fourier result',loss=None if not np.isfinite(best_value) else best_value,
        a_coefficients=best_coefficients[:5].tolist(),b_coefficients=best_coefficients[5:].tolist(),
        scale_rad_ns=device.amplitude_limit_mhz*MHZ_TO_RAD_NS,
        optimizer_success=terminal_converged,converged=terminal_converged,message='; '.join(terminations),
        budget_exhaustion=exhaustion,restart_outcomes=restart_outcomes,
        initializations=initializations,initialization_fit=fit,accepted_steps=accepted,evaluation_trace=trace,
        training_metrics=best_metrics,initial_objective=first_value,initial_waveform_sha256=initial_hash,
        final_waveform_sha256=waveform_hash(wave(best_coefficients)),coefficient_change_norm=delta,
        coefficients=best_coefficients.tolist(),**budget.snapshot(),
        incremental_cost={k:budget.snapshot()[k]-before[k] for k in before},
        acceleration=model.diagnostics() if hasattr(model,'diagnostics') else getattr(model,'backend_info',dict(actual_backend='cpu'))))
    return tuple(wave(best_coefficients)),history
