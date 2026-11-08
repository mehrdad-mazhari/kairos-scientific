"""Constrained Fourier GRAPE with a dissipative ensemble objective.

Frechet derivatives of Liouvillian exponentials give analytic control gradients.
Linear sample interpolation is integrated with midpoint substeps; acceptance
uses the production QuTiP solver, not this optimization approximation.
"""
import hashlib
import warnings
from time import perf_counter
import numpy as np
from scipy.linalg import expm_frechet
from scipy.optimize import minimize
from qutip import liouvillian
from .challenge_model import (DRIVE_X, DRIVE_Y, KETS, TEST_STATES, TARGET_X,
                             Scenario, collapse_operators, hamiltonian, MHZ_TO_RAD_NS, model_operators)

DEFAULTS = dict(enabled=True, iterations=40, samples=8, seed=2030,
                substeps=4, tail_fraction=.1, leakage_weight=2.,
                smoothness_weight=.001,
                harmonics=5, restarts=2, restart_scale=.08, tail_weight=0.,
                amplitude_grid_points=3, detuning_grid_points=5, ftol=1e-9)


def settings(config):
    resolved = {**DEFAULTS, **config.get('grape', {})}
    resolved.pop('initial_step', None)
    resolved.pop('backtracks', None)
    if resolved.get('initialization_strategy')=='qualified_fourier_multistart':
        resolved['quality_targets']=config.get('quality_targets',dict(nominal_fidelity_min=.999,leakage_max=.0001))
    return resolved


def _vec(matrix):
    return matrix.full().reshape(-1, order='F')


class EnsembleGRAPE:
    def __init__(self, times, device, scenarios, options):
        self.times, self.options = np.asarray(times), options
        kets, states, target, _, _, drive_x, drive_y, leakage = model_operators(device.levels)
        self.initial = np.column_stack([_vec(s.proj()) for s in states])
        self.target = np.column_stack([_vec((target*s).proj()) for s in states])
        self.leak = np.column_stack([_vec(leakage)]*6)
        self.models = []
        for scenario in scenarios:
            static = hamiltonian(times, np.zeros(len(times)), np.zeros(len(times)), device, scenario)(0)
            self.models.append((liouvillian(static, collapse_operators(device, scenario)).full(),
                liouvillian(drive_x*scenario.amplitude_scale, []).full(),
                liouvillian(drive_y*scenario.amplitude_scale, []).full()))

    def scenario_value_gradient(self, controls, model):
        base, di, dq = model
        state = self.initial.copy()
        history = []
        substeps = self.options['substeps']
        for j, interval in enumerate(np.diff(self.times)):
            dt = interval/substeps
            for sub in range(substeps):
                w = (sub+.5)/substeps
                u = (1-w)*controls[j]+w*controls[j+1]
                generator = dt*(base+u[0]*di+u[1]*dq)
                propagator, gi = expm_frechet(generator, dt*di)
                gq = expm_frechet(generator, dt*dq, compute_expm=False)
                history.append((j, w, state, propagator, gi, gq))
                state = propagator@state
        costate = (-self.target+self.options['leakage_weight']*self.leak)/6
        value = 1+np.vdot(costate, state).real
        self.last_scenario_metrics = dict(fidelity=float(np.vdot(self.target,state).real/6),
                                         leakage=float(np.vdot(self.leak,state).real/6))
        gradient = np.zeros_like(controls)
        for j, w, forward, propagator, gi, gq in reversed(history):
            derivative = np.array([np.vdot(costate, gi@forward).real,
                                   np.vdot(costate, gq@forward).real])
            gradient[j] += (1-w)*derivative
            gradient[j+1] += w*derivative
            costate = propagator.conj().T@costate
        return float(value), gradient

    def value_gradient(self, controls):
        evaluated=[];metrics=[]
        for model in self.models:
            evaluated.append(self.scenario_value_gradient(controls,model))
            if hasattr(self,'last_scenario_metrics'):metrics.append(self.last_scenario_metrics.copy())
        if metrics:
            from .risk_aware_grape import empirical_cvar,smooth_cvar
            losses=1-np.array([m['fidelity'] for m in metrics]);leaks=np.array([m['leakage'] for m in metrics])
            self.last_ensemble_metrics=dict(mean_training_fidelity=float(1-losses.mean()),
                cvar95_training_infidelity=empirical_cvar(losses),
                smooth_cvar95_training_infidelity=smooth_cvar(losses,.95,self.options.get('smoothing',1e-6))[0],
                mean_training_leakage=float(leaks.mean()),maximum_training_leakage=float(leaks.max()))
        values = np.array([v for v, _ in evaluated])
        gradients = np.array([g for _, g in evaluated])
        count = max(1, int(np.ceil(self.options['tail_fraction']*len(values))))
        tail = np.argsort(values)[-count:]
        tail_weight = self.options.get('tail_weight', 0.)
        value = tail_weight*values[tail].mean()+(1-tail_weight)*values.mean()
        gradient = tail_weight*gradients[tail].mean(axis=0)+(1-tail_weight)*gradients.mean(axis=0)
        differences = np.diff(controls, axis=0)
        weight = self.options['smoothness_weight']/len(differences)
        value += weight*np.sum(differences**2)
        gradient[:-1] -= 2*weight*differences
        gradient[1:] += 2*weight*differences
        return float(value), gradient


class FourierControls:
    """Wright et al. Eq. (2): odd sine harmonics for I, even for Q.

    dimensionless coefficients multiply the nominal hardware scale (rad/ns).
    Sample values use linear interpolation in the challenge solver.
    """
    def __init__(self, times, harmonics, scale):
        phase = np.pi*(np.asarray(times)-times[0])/(times[-1]-times[0])
        n = np.arange(1, harmonics+1)
        self.i = scale*np.sin(phase[:, None]*(2*n-1))
        self.q = scale*np.sin(phase[:, None]*(2*n))
        self.i[[0, -1]] = 0.
        self.q[[0, -1]] = 0.
        self.harmonics = harmonics

    def waveform(self, coefficients):
        return np.column_stack([self.i@coefficients[:self.harmonics],
                                self.q@coefficients[self.harmonics:]])

    def gradient(self, sample_gradient):
        return np.concatenate([self.i.T@sample_gradient[:, 0], self.q.T@sample_gradient[:, 1]])

    def fit(self, controls):
        return np.concatenate([np.linalg.lstsq(self.i, controls[:, 0], rcond=None)[0],
                               np.linalg.lstsq(self.q, controls[:, 1], rcond=None)[0]])

    def constrained_fit(self, controls, limit):
        """Explicit opt-in convex sampled least squares with joint amplitude disks."""
        start=self.fit(controls)
        peak=np.linalg.norm(self.waveform(start),axis=1).max()
        start*=min(1.,.999*limit/max(peak,1e-30))
        def loss(c):
            difference=self.waveform(c)-controls
            return float(np.sum(difference**2)/len(controls)),self.gradient(2*difference/len(controls))
        result=minimize(loss,start,method='SLSQP',jac=True,
            constraints={'type':'ineq','fun':lambda c:self.constraints(c,limit),'jac':lambda c:self.constraint_gradient(c,limit)},
            options={'maxiter':200,'ftol':1e-13})
        if self.constraints(result.x,limit).min()>=-1e-10 and loss(result.x)[0]<=loss(start)[0]:
            return result.x,dict(success=bool(result.success),message=str(result.message),residual=loss(result.x)[0])
        return start,dict(success=False,message='Fit candidate rejected; feasible initial fit retained',residual=loss(start)[0])

    def constraints(self, coefficients, limit):
        u = self.waveform(coefficients)
        return 1-np.sum(u*u, axis=1)/limit**2

    def constraint_gradient(self, coefficients, limit):
        u = self.waveform(coefficients)
        return np.column_stack([-2*u[:, [0]]*self.i, -2*u[:, [1]]*self.q])/limit**2


def refine(pulse, device, scenarios, options, amplitude_scale, *, model_class=EnsembleGRAPE):
    """Constrained Fourier-GRAPE with analytic chain-rule gradients and restarts.

    Eq. (2), ensemble averaging Eq. (4) and Appendix B of Wright et al.;
    smooth sinusoidal initialization/restarts from Rowland & Jones Sec. 1(d).
    We retain dissipative six-state X fidelity rather than copying their closed
    system X_pi/2 trace objective, and enforce the challenge's joint I/Q bound.
    """
    if options.get('initialization_strategy')=='qualified_fourier_multistart' or options.get('advanced_execution',False):
        from .grape_research import refine as research_refine
        return research_refine(pulse,device,scenarios,options,amplitude_scale,model_class=model_class)
    times, i, q = pulse
    controls = np.column_stack([i, q])
    started=perf_counter()
    from .jax_backend import make_model
    model = make_model(times, device, scenarios, options,model_class)
    original_evaluate=model.value_gradient
    evaluations=0
    pass_multiplier=getattr(model,'surrogate_pass_multiplier',2 if model_class is not EnsembleGRAPE else 1)
    actual_passes=0;backend_production_solves=0
    class BudgetExhausted(Exception):pass
    budget_reason=None
    def counted(controls):
        nonlocal evaluations,actual_passes,backend_production_solves
        if evaluations>=options.get('max_objective_evaluations',float('inf')):
            raise BudgetExhausted('objective evaluation budget exhausted')
        pending=model.pending_qualification_work() if hasattr(model,'pending_qualification_work') else dict(surrogate_passes=0)
        if actual_passes+len(scenarios)*pass_multiplier+pending['surrogate_passes']>options.get('max_surrogate_passes',float('inf')):
            raise BudgetExhausted('dissipative surrogate scenario-pass budget exhausted')
        if evaluations and perf_counter()-started>=options.get('max_wall_seconds',float('inf')):
            raise BudgetExhausted('wall budget reached (checked between evaluations)')
        evaluations+=1
        try:return original_evaluate(controls)
        finally:
            work=getattr(model,'last_call_work',dict(surrogate_passes=len(scenarios)*pass_multiplier,production_solves=0))
            actual_passes+=work['surrogate_passes'];backend_production_solves+=work['production_solves']
    model.value_gradient=counted
    scale = device.amplitude_limit_mhz*MHZ_TO_RAD_NS
    limit = scale/amplitude_scale
    basis = FourierControls(times, options['harmonics'], scale)
    start = basis.fit(controls)
    def feasible_start(c):
        peak = np.linalg.norm(basis.waveform(c), axis=1).max()
        return c*min(1., .999*limit/max(peak, 1e-30))
    fit_info={'strategy':'least_squares_radial','success':True}
    if options.get('initialization_strategy','least_squares_radial')=='constrained_least_squares':
        start,fit_info=basis.constrained_fit(controls,limit)
        fit_info['strategy']='constrained_least_squares'
    elif options.get('initialization_strategy','least_squares_radial')!='least_squares_radial':
        raise ValueError('unknown Fourier initialization strategy')
    start = feasible_start(start)
    fitted_coefficients=start.copy()
    if options.get('initialization_scale',0.):
        start=feasible_start(start+np.random.default_rng(options['seed']).normal(0.,options['initialization_scale'],len(start)))
    initial_coefficients=start.copy()
    initial_waveform=basis.waveform(start)
    evaluation_trace=[]
    callback_steps=0
    improvements=0
    best_gradient=None
    first_objective=None
    original_loss = model.value_gradient(controls)[0]
    history = [{'iteration': 0, 'loss': original_loss, 'kind': 'pre-GRAPE seed'}]
    rng = np.random.default_rng(options['seed'])
    best_value, best_coefficients = float('inf'), start.copy()
    best_risk_diagnostics=None
    for restart in range(options['restarts']):
        initial = start if restart == 0 else feasible_start(start+rng.normal(0, options['restart_scale'], len(start)))
        cache = {}
        def objective(c):
            nonlocal best_value, best_coefficients, best_risk_diagnostics, improvements, best_gradient, first_objective
            key = c.tobytes()
            if key not in cache:
                value, gradient = model.value_gradient(basis.waveform(c))
                cache.clear()
                cache[key] = value, basis.gradient(gradient)
                feasible=bool(np.min(basis.constraints(c, limit))>=-1e-10)
                improved=feasible and value<best_value
                if first_objective is None:first_objective=float(value)
                evaluation_trace.append(dict(evaluation=evaluations,loss=float(value),coefficients=c.tolist(),gradient=cache[key][1].tolist(),gradient_norm=float(np.linalg.norm(cache[key][1])),feasible=feasible,retained_improvement=bool(improved)))
                if improved:
                    if np.isfinite(best_value):improvements+=1
                    best_gradient=cache[key][1].copy()
                    best_value, best_coefficients = value, c.copy()
                    best_risk_diagnostics=getattr(model,"last_risk",None)
            return cache[key]
        def progress(c):
            value, gradient = objective(c)
            history.append({'iteration': len(history), 'restart': restart+1, 'loss': float(value),
                            'coefficient_gradient': gradient.tolist(),
                            'gradient_norm': float(np.linalg.norm(gradient)),
                            'coefficients': c.tolist()})
            if len(history) % 5 == 0:
                print(f"  Fourier-GRAPE start {restart+1}/{options['restarts']}: loss={value:.9g}", flush=True)
        try:
            progress(initial)
            history[-1]['kind'] = 'Fourier start'
            def solver_callback(c):
                nonlocal callback_steps
                callback_steps+=1
                progress(c)
            result = minimize(objective, initial, method='SLSQP', jac=True,
                              constraints={'type': 'ineq',
                                'fun': lambda c: basis.constraints(c, limit),
                                'jac': lambda c: basis.constraint_gradient(c, limit)},
                              callback=solver_callback,
                              options={'maxiter': options['iterations'], 'ftol': options['ftol']})
            value, gradient = objective(result.x)
        except BudgetExhausted as error:
            budget_reason=str(error)
            history.append(dict(iteration=len(history),loss=best_value,optimizer_success=False,message=budget_reason))
            break
        history.append({'iteration': len(history), 'restart': restart+1,
                        'loss': float(result.fun), 'optimizer_success': bool(result.success),
                        'message': str(result.message),
                        'coefficient_gradient': gradient.tolist(),
                        'gradient_norm': float(np.linalg.norm(gradient)),
                        'coefficients': result.x.tolist()})
    history.append({'iteration': len(history), 'loss': best_value, 'kind': 'best feasible Fourier proposal',
                    'a_coefficients': best_coefficients[:options['harmonics']].tolist(),
                    'b_coefficients': best_coefficients[options['harmonics']:].tolist(),
                    'scale_rad_ns': scale})
    try:
        model.value_gradient(basis.waveform(best_coefficients))
    except BudgetExhausted:
        pass
    history[-1].update(objective_evaluations=evaluations,gradient_evaluations=evaluations,
        surrogate_scenario_passes=actual_passes,backend_qualification_production_solves=backend_production_solves,
        acceleration=model.diagnostics() if hasattr(model,'diagnostics') else getattr(model,'backend_info',dict(actual_backend='cpu')),
        budget_exhaustion=budget_reason,elapsed_seconds=perf_counter()-started,risk_diagnostics=best_risk_diagnostics)
    u = basis.waveform(best_coefficients)
    delta=float(np.linalg.norm(best_coefficients-initial_coefficients))
    moved=delta>options.get('meaningful_change_tolerance',1e-8)*(1+float(np.linalg.norm(initial_coefficients)))
    if not moved:
        warnings.warn('GRAPE returned its initial Fourier waveform without meaningful coefficient changes; inspect budgets, trace and termination.',RuntimeWarning)
    def wave_hash(control):
        return hashlib.sha256(np.array([times,control[:,0],control[:,1]]).tobytes()).hexdigest()
    history[-1]['execution_audit']=dict(initial_objective=first_objective,final_objective=best_value,
        initial_coefficients=initial_coefficients.tolist(),unperturbed_fit_coefficients=fitted_coefficients.tolist(),
        final_coefficients=best_coefficients.tolist(),final_gradient_norm=float(np.linalg.norm(best_gradient)) if best_gradient is not None else None,
        solver_callback_steps=callback_steps,retained_feasible_improvements=improvements,coefficient_change_norm=delta,
        meaningful_parameter_change=bool(moved),initial_waveform_sha256=wave_hash(initial_waveform),final_waveform_sha256=wave_hash(u),
        seed_waveform_sha256=hashlib.sha256(np.array(pulse).tobytes()).hexdigest(),initialization_fit=fit_info,evaluation_trace=evaluation_trace)
    return (np.asarray(times).copy(), u[:, 0], u[:, 1]), history
