"""Experimental lambda continuation, with one budget shared by every stage.

No monotonic CVaR or generalization guarantee is asserted. Candidate retention
uses nominal qualification and the final scheduled training objective, never
selection or validation observations.
"""
import numpy as np
from binding_budget_optimizer import ResourceBudget,BudgetExhausted,refine as stage_refine,waveform_hash
from risk_aware_grape import RiskEnsembleGRAPE


def validate_schedule(schedule):
    values=np.asarray(schedule,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all() or values[0]!=0 or np.any(np.diff(values)<=0):
        raise ValueError('continuation must start at zero and strictly increase through finite weights')
    return values.tolist()


def final_objective(metrics,lam,mu):
    return (1-metrics['mean_training_fidelity']+lam*metrics['smooth_cvar95_training_infidelity']+
            mu*metrics['mean_training_leakage']+metrics['regularization'])


def refine(pulse,device,scenarios,options,amplitude_scale):
    options=dict(options);schedule=validate_schedule(options.get('continuation_schedule',[0,.1,.5,1.]))
    if options.get('restarts',5)<len(schedule):raise ValueError('need at least one restart allocation per continuation stage')
    budget=ResourceBudget(objective_limit=options.get('max_objective_evaluations',1000),
        pass_limit=options.get('max_total_scenario_passes',float('inf')),
        wall_limit=options.get('max_wall_seconds',float('inf')),
        iteration_limit=options.get('max_total_iterations',options.get('iterations',150)))
    current=tuple(np.asarray(row).copy() for row in pulse);warm=None;stages=[];best=None;best_score=float('inf')
    restarts_left=options.get('restarts',5)
    for index,lam in enumerate(schedule):
        remaining=len(schedule)-index
        before=budget.snapshot();input_hash=waveform_hash(current)
        allocation=max(1,restarts_left//remaining);restarts_left-=allocation
        stage_options={**options,'risk_weight':lam,'restarts':allocation,
            'initialization_strategy':'qualified_fourier_multistart','seed':options['seed']+index*10000}
        if warm is not None:stage_options['warm_start_coefficients']=warm
        quota=dict(objectives=max(0,(budget.objective_limit-budget.objective_evaluations)//remaining),
            iterations=max(0,(budget.iteration_limit-budget.iterations)//remaining))
        if np.isfinite(budget.pass_limit):quota['passes']=max(0,(budget.pass_limit-budget.passes)//remaining)
        if np.isfinite(budget.wall_limit):quota['wall']=max(0,(budget.wall_limit-budget.elapsed)/remaining)
        try:
            proposed,h=stage_refine(current,device,scenarios,stage_options,amplitude_scale,
                model_class=RiskEnsembleGRAPE,budget=budget,quota=quota)
            result=h[-1];metrics=result['training_metrics']
            if metrics is None:raise RuntimeError('stage had no qualified evaluated candidate')
            current=proposed;warm=result['coefficients']
            # All archived feasible improvements are comparable under final lambda.
            for candidate in result['accepted_steps']:
                score=final_objective(candidate,schedule[-1],options['leakage_weight'])
                if score<best_score:
                    from grape import FourierControls
                    from challenge_model import MHZ_TO_RAD_NS
                    basis=FourierControls(pulse[0],5,device.amplitude_limit_mhz*MHZ_TO_RAD_NS)
                    u=basis.waveform(np.asarray(candidate['coefficients']))
                    best=(tuple(np.array([pulse[0],*u.T])),candidate['coefficients'],candidate)
                    best_score=score
            stages.append(dict(stage=index,lambda_value=lam,status='completed',input_waveform_sha256=input_hash,
                output_waveform_sha256=waveform_hash(current),transferred_coefficients=warm,
                stage_objective=result['loss'],training_cvar=metrics['cvar95_training_infidelity'],
                nominal_fidelity=result['accepted_steps'][-1]['nominal_fidelity'],
                nominal_leakage=result['accepted_steps'][-1]['nominal_leakage'],
                converged=result['converged'],termination=result['message'],
                budget_exhaustion=result['budget_exhaustion'],history=h,
                incremental_cost={k:budget.snapshot()[k]-before[k] for k in before}))
        except (BudgetExhausted,ValueError,RuntimeError) as error:
            stages.append(dict(stage=index,lambda_value=lam,status='failed_previous_waveform_preserved',
                input_waveform_sha256=input_hash,output_waveform_sha256=waveform_hash(current),
                termination=str(error),converged=False,
                incremental_cost={k:budget.snapshot()[k]-before[k] for k in before}))
        if budget.objective_evaluations>=budget.objective_limit or budget.passes>=budget.pass_limit or budget.elapsed>=budget.wall_limit:
            break
    output=best[0] if best is not None else current
    accepted=[]
    for stage in stages:
        for record in stage.get('history',[{}])[-1].get('accepted_steps',[]):
            accepted.append({**record,'stage':stage['stage'],'lambda_value':stage['lambda_value']})
    result=dict(kind='continuation_risk_grape',schedule=schedule,stages=stages,
        completed_stages=len(stages),schedule_complete=len(stages)==len(schedule) and all(s['status']=='completed' for s in stages),
        converged=len(stages)==len(schedule) and all(s['converged'] for s in stages),
        optimizer_success=len(stages)==len(schedule) and all(s['converged'] for s in stages),
        message='Experimental continuation; inspect each stage status',accepted_steps=accepted,
        training_metrics=None if best is None else best[2],loss=None if best is None else best_score,
        coefficients=None if best is None else best[1],initial_waveform_sha256=waveform_hash(pulse),
        final_waveform_sha256=waveform_hash(output),**budget.snapshot())
    return output,[result]
