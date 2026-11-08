"""Training-only optimizer execution audit. Never generates/reads validation devices."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .grape import FourierControls,EnsembleGRAPE,DEFAULTS,refine
from .risk_aware_grape import RiskEnsembleGRAPE,refine as risk_refine,empirical_cvar,smooth_cvar
from .challenge_model import Device,Scenario,monte_carlo_scenarios,evaluate_waveform,hardware_metrics,MHZ_TO_RAD_NS
from .robust_optimization import write_json,write_csv
from .research_benchmark import load_profile,effective_grape_options


def hash_wave(w):return hashlib.sha256(np.asarray(w,dtype=float).tobytes()).hexdigest()


def validate_objective_algebra():
    """Independent 100-scenario algebra test; physical adjoints are checked separately."""
    rng=np.random.default_rng(7140)
    controls=np.array([[0.,0.],[.1,.02],[.08,-.01],[0.,0.]])
    losses=np.linspace(.001,.03,100)
    leakage=np.linspace(1e-5,8e-5,100)
    gf=rng.normal(0.,.002,(100,*controls.shape))
    gl=rng.normal(0.,1e-5,(100,*controls.shape))
    options={**DEFAULTS,'alpha':.95,'risk_weight':1.7,'leakage_weight':2.3,'smoothing':.002}
    model=RiskEnsembleGRAPE.__new__(RiskEnsembleGRAPE)
    model.options=options;model.models=list(range(100))
    def scenario_value_gradient(u,index):
        mu=model.options['leakage_weight'];delta=u-controls
        return (float(losses[index]+np.sum(gf[index]*delta)+mu*(leakage[index]+np.sum(gl[index]*delta))),gf[index]+mu*gl[index])
    model.scenario_value_gradient=scenario_value_gradient
    value,g=model.value_gradient(controls)
    differences=np.diff(controls,axis=0)
    regularization=options['smoothness_weight']*np.sum(differences**2)/len(differences)
    cv,weights,_=smooth_cvar(losses,options['alpha'],options['smoothing'])
    expected=float(losses.mean()+1.7*cv+2.3*leakage.mean()+regularization)
    eps=1e-6;errors=[]
    for index in np.ndindex(controls.shape):
        plus=controls.copy();minus=controls.copy();plus[index]+=eps;minus[index]-=eps
        fd=(model.value_gradient(plus)[0]-model.value_gradient(minus)[0])/(2*eps)
        errors.append(abs(fd-g[index]))
    options['risk_weight']=0.;v0,g0=model.value_gradient(controls)
    options['risk_weight']=2.;v2,g2=model.value_gradient(controls)
    options['leakage_weight']=0.;vm,gm=model.value_gradient(controls)
    exact=empirical_cvar(losses);bound=options['smoothing']*np.log(2)/(1-options['alpha'])
    result=dict(scenarios=100,distinct_infidelities=len(np.unique(losses)),
        objective_formula='mean(1-fidelity) + lambda*smooth_CVaR95(1-fidelity) + mu*mean(leakage) + regularization',
        objective_formula_error=abs(value-expected),regularization=float(regularization),
        all_control_gradient_max_absolute_error=max(errors),
        lambda_objective_error=abs((v2-v0)-2*cv),
        lambda_gradient_error=float(np.max(abs((g2-g0)-2*np.einsum('i,ijk->jk',weights,gf)))),
        mu_objective_error=abs((v2-vm)-2.3*leakage.mean()),
        mu_gradient_error=float(np.max(abs((g2-gm)-2.3*gl.mean(axis=0)))),
        risk_derivatives_nonnegative=bool(np.all(weights>=0)),
        larger_infidelity_has_larger_risk_derivative=bool(np.all(np.diff(weights)>=-1e-14)),
        higher_infidelity_increases_risk=bool(smooth_cvar(losses+.01,.95,.002)[0]>cv),
        exact_cvar=exact,smoothed_cvar=cv,smoothing_gap=cv-exact,smoothing_bound=bound)
    result['passed']=bool(max(result[k] for k in ('objective_formula_error','all_control_gradient_max_absolute_error','lambda_objective_error','lambda_gradient_error','mu_objective_error','mu_gradient_error'))<1e-8 and result['risk_derivatives_nonnegative'] and result['larger_infidelity_has_larger_risk_derivative'] and result['higher_infidelity_increases_risk'] and -1e-12<=cv-exact<=bound+1e-12)
    return result


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    c=json.loads(Path('challenge_config.json').read_text());d=Device(**c['device'])
    old=Path('results/research_benchmark_complete')
    seed=np.asarray(json.loads((old/'references.json').read_text())['calibrated_drag']['waveform'])
    seed_metrics=evaluate_waveform(*seed,d,strict=True)
    if seed_metrics['fidelity']<c['quality_targets']['nominal_fidelity_min'] or seed_metrics['leakage']>c['quality_targets']['leakage_max']:
        raise RuntimeError('Diagnostic starting pulse must qualify')
    config=dict(full=load_profile('research_config.json'),quick=load_profile('research_config.json',True),
        diagnostic=dict(training_seed=7100,gradient_scenarios=100,optimization_scenarios=12,
            iterations=12,objective_cap=40,wall_cap=None,initialization_scale=0.,initialization_strategy='constrained_least_squares',harmonics=5,substeps=2),
        independent_validation_used=False,engineering_thresholds=c['quality_targets'])
    config['resolved_optimizer_options']={mode:{regime:effective_grape_options(c,config[mode],config[mode]['optimizer_seeds'][0],regime)
        for regime in config[mode]['regimes']} for mode in ('full','quick')}
    write_json(out/'effective_research_config.json',config)
    cache={}
    def production(w):
        key=hash_wave(w)
        if key not in cache:cache[key]=evaluate_waveform(*w,d,strict=True)
        return cache[key]
    traces=[];initializations=[];audit=[]
    def initrow(label,w,basis=None,coeff=None):
        metric=production(w);hw=hardware_metrics(*w,d)
        initializations.append(dict(stage=label,fidelity=metric['fidelity'],leakage=metric['leakage'],peak_mhz=hw['peak_amplitude_mhz'],headroom_feasible=hw['peak_amplitude_mhz']*1.03<=d.amplitude_limit_mhz+1e-10,endpoint=hw['endpoint_amplitude_rad_ns'],waveform_sha256=hash_wave(w),relative_fit_error=float(np.linalg.norm(np.array(w)[1:]-seed[1:])/np.linalg.norm(seed[1:])),coefficients=json.dumps(coeff.tolist()) if coeff is not None else ''))
    initrow('qualified calibrated DRAG',seed)
    for harmonics in (2,5):
        basis=FourierControls(seed[0],harmonics,d.amplitude_limit_mhz*MHZ_TO_RAD_NS)
        fit=basis.fit(seed[1:].T);limit=d.amplitude_limit_mhz*MHZ_TO_RAD_NS/1.03
        fit*=min(1.,.999*limit/max(np.linalg.norm(basis.waveform(fit),axis=1).max(),1e-30))
        for label,coeff in [('unperturbed least-squares',fit),('seeded perturbation .05',fit+np.random.default_rng(6200).normal(0.,.05,len(fit)))]:
            coeff=coeff*min(1.,.999*limit/max(np.linalg.norm(basis.waveform(coeff),axis=1).max(),1e-30))
            u=basis.waveform(coeff);initrow(f'{harmonics} harmonics: {label}',np.array([seed[0],u[:,0],u[:,1]]),basis,coeff)
        constrained,info=basis.constrained_fit(seed[1:].T,limit);u=basis.waveform(constrained)
        initrow(f'{harmonics} harmonics: constrained fit',np.array([seed[0],u[:,0],u[:,1]]),basis,constrained)
    # Audit saved executions without accessing any held-out device file.
    for regime in ('equal_evaluations','equal_wall'):
        for name in ('conventional','mu_0','A_mean','B_worst','C_cvar','D_mean_cvar','E_mean_cvar_leak'):
            r=json.loads((old/f'{regime}_{name}_6200.json').read_text());h=r['history'];first=next(t for t in h if 'coefficients'in t);last=h[-1]
            basis=FourierControls(seed[0],len(last['a_coefficients']),last['scale_rad_ns']);coeff=np.array(first['coefficients']);u=basis.waveform(coeff);initial=np.array([seed[0],u[:,0],u[:,1]])
            final=np.asarray(r['candidate']['waveform']);fm=production(final);im=production(initial)
            finalcoeff=np.r_[last['a_coefficients'],last['b_coefficients']]
            audit.append(dict(method=name,regime=regime,initial_objective=first['loss'],final_objective=last['loss'],gradient_norm=first['gradient_norm'],initial_coefficients=coeff.tolist(),final_coefficients=finalcoeff.tolist(),coefficient_change=float(np.linalg.norm(coeff-finalcoeff)),initial_waveform_sha256=hash_wave(initial),final_waveform_sha256=hash_wave(final),waveform_updated=not np.allclose(initial,final,rtol=0,atol=1e-12),initial_fidelity=im['fidelity'],final_fidelity=fm['fidelity'],initial_leakage=im['leakage'],final_leakage=fm['leakage'],objective_evaluations=last['objective_evaluations'],gradient_evaluations=last['gradient_evaluations'],termination=[t['message'] for t in h if 'message'in t],accepted_steps_reconstructible=False))
    for record in audit:
        original=json.loads((old/f"{record['regime']}_{record['method']}_6200.json").read_text())['history'][-1]
        record.update(elapsed_seconds=original['elapsed_seconds'],surrogate_scenario_passes=original['surrogate_scenario_passes'])
    write_json(out/'historical_execution_audit.json',audit)
    from .optimizer_audit_tables import replay
    replay(out)
    # 100 distinct fixed scenarios; full gate duration, production level count.
    scenarios=monte_carlo_scenarios(100,7100,c['coherence_min'],**c['manufacturing_ranges'])
    write_json(out/'diagnostic_training_devices.json',[s.__dict__ for s in scenarios])
    options={**DEFAULTS,'substeps':2,'harmonics':5,'alpha':.95,'risk_weight':1.,'leakage_weight':2.,'smoothing':1e-6}
    basis=FourierControls(seed[0],5,d.amplitude_limit_mhz*MHZ_TO_RAD_NS);coeff,_=basis.constrained_fit(seed[1:].T,d.amplitude_limit_mhz*MHZ_TO_RAD_NS/1.03)
    controls=basis.waveform(coeff);direction=np.random.default_rng(7110).normal(size=len(coeff));direction/=np.linalg.norm(direction)
    checks={}
    for name,cls in [('conventional',EnsembleGRAPE),('risk',RiskEnsembleGRAPE)]:
        model=cls(seed[0],d,scenarios,dict(options))
        value,g=model.value_gradient(controls);analytic=float(basis.gradient(g)@direction);eps=1e-5
        fd=(model.value_gradient(basis.waveform(coeff+eps*direction))[0]-model.value_gradient(basis.waveform(coeff-eps*direction))[0])/(2*eps)
        checks[name]=dict(scenarios=100,value=value,analytic_directional_gradient=analytic,finite_difference=fd,absolute_error=abs(analytic-fd),passed=abs(analytic-fd)<2e-6)
        if name=='risk':
            model.value_gradient(controls)
            diag=model.last_risk.copy();model.options['risk_weight']=0.;v0,g0=model.value_gradient(controls)
            model.options['risk_weight']=2.;v2,g2=model.value_gradient(controls)
            model.options['leakage_weight']=0.;vm,gm=model.value_gradient(controls)
            checks['weighting']=dict(lambda_objective_change=v2-v0,lambda_gradient_change=float(np.linalg.norm(g2-g0)),mu_objective_change=v2-vm,mu_gradient_change=float(np.linalg.norm(g2-gm)),lambda_expected_change=2*diag['smoothed_cvar'],smoothing=diag)
    losses=np.linspace(.001,.02,100);sv,sg,_=smooth_cvar(losses,.95,1e-6);bound=1e-6*np.log(2)/.05
    checks['cvar']=dict(exact=empirical_cvar(losses),smooth=sv,gap=sv-empirical_cvar(losses),bound=bound,penalty_increases=smooth_cvar(losses+.01)[0]>sv,gradient_sum=float(sg.sum()),passed=bool(-1e-12<=sv-empirical_cvar(losses)<=bound+1e-12))
    checks['weighting']['passed']=bool(abs(checks['weighting']['lambda_objective_change']-checks['weighting']['lambda_expected_change'])<1e-10 and checks['weighting']['lambda_gradient_change']>0 and checks['weighting']['mu_gradient_change']>0 and checks['weighting']['mu_objective_change']>=-1e-12)
    checks['objective_algebra']=validate_objective_algebra()
    write_json(out/'gradient_validation.json',checks)
    if not all(item['passed'] for item in checks.values()):raise RuntimeError('gradient verification failed; optimization cancelled')
    runs={}
    for name,optimizer in [('conventional',refine),('risk',risk_refine)]:
        opts={**options,'seed':7120,'iterations':12,'restarts':1,'max_objective_evaluations':40,'initialization_scale':0.,'initialization_strategy':'constrained_least_squares'}
        wave,h=optimizer(seed,d,scenarios[:12],opts,1.03);execution=h[-1]['execution_audit'];runs[name]=dict(history=h,production_initial=production(np.array([seed[0],*basis.waveform(np.array(execution['initial_coefficients'])).T])),production_final=production(wave),hardware=hardware_metrics(*wave,d))
        for row in execution['evaluation_trace']:
            traces.append(dict(method=name,**{k:v if not isinstance(v,list) else json.dumps(v) for k,v in row.items()}))
        np.savetxt(out/f'{name}_final_waveform.csv',np.asarray(wave).T,delimiter=',',header='time_ns,I_rad_ns,Q_rad_ns',comments='')
        print(name,execution['initial_objective'],execution['final_objective'],execution['coefficient_change_norm'],h[-1]['budget_exhaustion'],flush=True)
    write_json(out/'diagnostic_optimizer_runs.json',runs);write_csv(out/'optimizer_traces.csv',traces);write_csv(out/'initialization_comparison.csv',initializations)
    lines=['# Optimizer diagnostic report','','No validation devices were generated or used. No full campaign was executed.','','## Historical execution audit','','| Regime / method | Initial objective | Final objective | Gradient norm | Coefficient change | Stop |','|---|---:|---:|---:|---:|---|']
    for r in audit:lines.append(f"| {r['regime']} / {r['method']} | {r['initial_objective']:.9g} | {r['final_objective']:.9g} | {r['gradient_norm']:.9g} | {r['coefficient_change']:.9g} | {r['termination']} |")
    lines+=['','These are per-method own objectives, not comparable objective scales. Historical traces lack enough line-search detail to reconstruct accepted solver steps; no accepted-step count is fabricated. Direct waveform hashes show identical returned controls, independent of result aggregation. Initial objectives differ, so objective ablations were not accidentally identical.',
        '','## Initialization','', '| Strategy | Nominal F | Leakage | Relative waveform residual |','|---|---:|---:|---:|']
    lines += [f"| {r['stage']} | {r['fidelity']:.10g} | {r['leakage']:.9g} | {r['relative_fit_error']:.9g} |" for r in initializations]
    lines+=['','Least-squares and constrained fits are investigated explicitly. Legacy initialization remains the default; the diagnostic uses opt-in constrained least squares and no random perturbation to isolate optimizer execution from randomized-start difficulty.',
        '','## Gradient verification','','Both dissipative gradients were checked on 100 distinct uncertainty scenarios at the full 15.6 ns gate duration with five levels. See gradient_validation.json for finite differences, weight effects and the exact/smoothed CVaR gap.',
        '','## Usable-budget diagnostic','','| Method | Initial objective | Final objective | Coefficient change | Retained feasible improvements | Initial F | Final F | Final leakage |','|---|---:|---:|---:|---:|---:|---:|---:|']
    terminations=[]
    for name,r in runs.items():
        a=r['history'][-1]['execution_audit'];lines.append(f"| {name} | {a['initial_objective']:.9g} | {a['final_objective']:.9g} | {a['coefficient_change_norm']:.9g} | {a['retained_feasible_improvements']} | {r['production_initial']['fidelity']:.9g} | {r['production_final']['fidelity']:.9g} | {r['production_final']['leakage']:.9g} |")
        terminations.append(f"Termination ({name}): {[t.get('message') for t in r['history'] if 'message' in t]}")
    lines+=['',*terminations]
    lines+=['','## Findings and limits','','The historical budget of three calls includes raw-seed evaluation and the initial Fourier point, leaving essentially one trial evaluation and no adequate line search. Coupled initialization plus budget exhaustion explains identical returned controls; finite nonzero gradients contradict a blanket claim of a broken optimizer. The seeded perturbation must be assessed separately from Fourier approximation error.',
        '','Full research configuration is exported separately from quick and diagnostic budgets. Debugging does not change these budgets or weaken thresholds. Objective reductions on fixed training devices are not independent evidence of risk-method superiority. Numerical interpolation, initialization variability and full-budget behavior remain research questions. A production nominal result below the DRAG reference must be reported, not hidden.']
    (out/'optimizer_diagnostic_report.md').write_text('\n'.join(lines)+'\n')
    from .optimizer_audit_tables import finalize
    finalize(out)
    return checks,runs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/optimizer_diagnostic');a=p.parse_args();run(a.output)
