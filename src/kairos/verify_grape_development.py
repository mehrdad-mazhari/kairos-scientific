"""Verify completed development artifacts without drawing any new scenarios."""
import argparse,csv,hashlib,json
from pathlib import Path
from time import perf_counter
import numpy as np
from .challenge_model import Device,Scenario,evaluate_waveform,hardware_metrics,MHZ_TO_RAD_NS
from .grape import DEFAULTS,FourierControls,EnsembleGRAPE
from .grape_research import waveform_hash
from .risk_aware_grape import empirical_cvar


def verify(output):
    out=Path(output);protocol=json.loads((out/'protocol.json').read_text());base=protocol['base_config'];c=protocol['config']
    device=Device(**base['device']);quality=base['quality_targets'];checks={};started=perf_counter();verification_solves=0
    changes=json.loads((out/'post_execution_hardening.json').read_text()) if (out/'post_execution_hardening.json').exists() else {}
    changed=[]
    for p,value in protocol['source_hashes'].items():
        actual=hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if actual==value:continue
        assert p in ('grape_development.py','grape_development_report.py') and p in changes,'numerical source changed'
        assert hashlib.sha256((out/'source_snapshot'/p).read_bytes()).hexdigest()==value
        assert changes[p]['current_sha256']==actual and changes[p]['executed_sha256']==value
        changed.append(p)
    checks['numerical_sources_match_executed_protocol']=True
    checks['subsequent_provenance_or_rendering_changes']=changed
    provenance=json.loads((out/'input_provenance.json').read_text()) if (out/'input_provenance.json').exists() else None
    if provenance:
        assert hashlib.sha256(Path(provenance['reference_checkpoint']).read_bytes()).hexdigest()==provenance['reference_checkpoint_sha256']
        assert hashlib.sha256(Path(provenance['copied_to_source_snapshot']).read_bytes()).hexdigest()==provenance['reference_checkpoint_sha256']
        checks['reference_checkpoint_matches_executed_snapshot']=True
    scenarios=json.loads((out/'training_devices.json').read_text());selection=json.loads((out/'selection_devices.json').read_text())
    assert len(scenarios)==110 and len(selection)==200
    assert len({json.dumps(s,sort_keys=True) for s in scenarios[:100]})==100
    checks['training_selection_disjoint']=not ({json.dumps(s,sort_keys=True) for s in scenarios[:100]} & {json.dumps(s,sort_keys=True) for s in selection})
    checks['no_validation_dataset']=not any(out.glob('*validation_devices*'))
    gradients=json.loads((out/'gradient_verification.json').read_text());assert all(r['passed'] for r in gradients.values())
    with (out/'initialization_results.csv').open() as stream:initializations=list(csv.DictReader(stream))
    reference=json.loads((out/'shared_reference.json').read_text());times=np.asarray(reference['waveform'][0]);nominal_checks=[]
    for row in initializations:
        coeff=np.asarray(json.loads(row['coefficients']));basis=FourierControls(times,len(coeff)//2,device.amplitude_limit_mhz*MHZ_TO_RAD_NS)
        wave=np.array([times,*basis.waveform(coeff).T]);hw=hardware_metrics(*wave,device)
        assert waveform_hash(wave)==row['waveform_sha256']
        assert hw['endpoint_amplitude_rad_ns']==0 and hw['peak_amplitude_mhz']*base['amplitude_headroom_scale']<=device.amplitude_limit_mhz+1e-8
        if row['strategy']=='qualified_fourier_multistart' and row['qualified']=='True':
            assert float(row['fidelity'])>=quality['nominal_fidelity_min'] and float(row['leakage'])<=quality['leakage_max']
    stage_checks=[];pulse_rows=[]
    for path in sorted(out.glob('equal_*_*.json')):
        if path.name.endswith(('_optimization.json','_progress.json')):continue
        run=json.loads(path.read_text())
        if 'history' not in run:continue
        wave=np.asarray(run['waveform']);last=run['history'][-1]
        assert waveform_hash(wave)==run['waveform_sha256']
        np.testing.assert_array_equal(wave[0],times);np.testing.assert_array_equal(wave[1:,[0,-1]],0.)
        actual=evaluate_waveform(*wave,device,strict=True);verification_solves+=1
        reproduction={k:abs(actual[k]-run['nominal'][k]) for k in ('fidelity','leakage')}
        assert max(reproduction.values())<1e-12
        assert actual['fidelity']>=quality['nominal_fidelity_min'] and actual['leakage']<=quality['leakage_max']
        # Verify saved selection aggregation without evaluating or tuning on new draws.
        observations=[json.loads((out/'selection_channels'/f"{run['waveform_sha256']}_{i}.json").read_text()) for i in range(200)]
        exact=empirical_cvar([1-row['fidelity'] for row in observations])
        assert abs(exact-run['selection']['statistics']['cvar95_infidelity'])<1e-14
        assert last['total_dissipative_passes']==last['surrogate_scenario_passes']+last['production_channel_solves']
        if run['regime']=='equal_objectives':assert last['objective_evaluations']<=c['objective_calls']
        if run['regime']=='equal_passes':assert last['total_dissipative_passes']<=c['scenario_passes']
        assert last['solver_iterations']<=c['iterations']
        for i,stage in enumerate(last.get('stages',[])):
            if i:
                previous=last['stages'][i-1]
                assert stage['input_waveform_sha256']==previous['output_waveform_sha256']
            if stage['status']=='failed_previous_waveform_preserved':assert stage['input_waveform_sha256']==stage['output_waveform_sha256']
        if last.get('stages'):
            for key in ('objective_evaluations','surrogate_scenario_passes','production_channel_solves'):
                assert last[key]==sum(stage['incremental_cost'][key] for stage in last['stages'])
        # Solver checks at nominal + three predeclared deterministic stress cases.
        detail=[]
        for scenario_dict in [scenarios[-10],scenarios[-9],scenarios[-2],scenarios[-1]]:
            scenario=Scenario(**scenario_dict);metric=evaluate_waveform(*wave,device,scenario,strict=True);verification_solves+=1
            errors={}
            for substeps in (2,4):
                model=EnsembleGRAPE(wave[0],device,[scenario],{**DEFAULTS,'substeps':substeps,'leakage_weight':0.})
                model.scenario_value_gradient(wave[1:].T,model.models[0]);surrogate=model.last_scenario_metrics
                errors[substeps]={k:abs(surrogate[k]-metric[k]) for k in ('fidelity','leakage')}
            detail.append(dict(scenario=scenario_dict,errors=errors))
        nominal_checks.append(dict(method=run['method'],seed=run['seed'],reproduction=reproduction,substep_checks=detail))
        first=last['accepted_steps'][0];final=last['training_metrics']
        initial_hashes={r['waveform_sha256'] for r in last.get('initializations',[])}
        stage_initial_hashes={}
        for stage in last.get('stages',[]):
            if stage.get('history'):stage_initial_hashes[stage['stage']]={r['waveform_sha256'] for r in stage['history'][-1]['initializations']}
        retained=next(r for r in reversed(last['accepted_steps']) if r['waveform_sha256']==run['waveform_sha256'])
        pulse_rows.append(dict(method=run['method'],seed=run['seed'],initial_fidelity=first['nominal_fidelity'],
            final_fidelity=actual['fidelity'],initial_leakage=first['nominal_leakage'],final_leakage=actual['leakage'],
            initial_training_cvar=first['cvar95_training_infidelity'],final_training_cvar=final['cvar95_training_infidelity'],
            non_initialization_retained_proposals=sum(r['waveform_sha256'] not in stage_initial_hashes.get(r.get('stage'),initial_hashes) for r in last['accepted_steps']),
            gradient_norm_at_retention=retained['gradient_norm'],converged=last['converged']))
    errors=[error for run in nominal_checks for item in run['substep_checks'] for e in item['errors'].values() for error in e.values()]
    checks.update(completed_trials=len(pulse_rows),gradient_checks_passed=True,
        max_production_reproduction_error=max((max(r['reproduction'].values()) for r in nominal_checks),default=0.),
        maximum_stress_substep_metric_error=max(errors,default=0.),
        stress_substep_checks_passed=max(errors,default=0.)<=quality['solver_metric_tolerance'],
        qualification_and_cost_checks_passed=True,verification_extra_production_channel_solves=verification_solves,
        verification_elapsed_seconds=perf_counter()-started,independent_validation_generated=False,
        full_intermediate_executed=c['mode']=='intermediate',
        pre_experiment_unit_tests=int(__import__('re').search(r'Ran (\d+) tests',(out/'tests.log').read_text()).group(1)),
        final_unit_tests=int(__import__('re').search(r'Ran (\d+) tests',(out/'tests_final.log').read_text()).group(1)) if (out/'tests_final.log').exists() else None)
    with (out/'pulse_improvements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pulse_rows[0]));writer.writeheader();writer.writerows(pulse_rows)
    (out/'verification.json').write_text(json.dumps(dict(checks=checks,production_checks=nominal_checks),indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    return checks


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');args=parser.parse_args();verify(args.output)
