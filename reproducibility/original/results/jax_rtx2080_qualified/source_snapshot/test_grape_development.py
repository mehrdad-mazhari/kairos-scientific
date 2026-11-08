import json,tempfile,unittest,warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
from challenge_model import Device,PulseParameters,make_pulse,MHZ_TO_RAD_NS
from grape import DEFAULTS,FourierControls,EnsembleGRAPE
from risk_aware_grape import RiskEnsembleGRAPE
from challenge_model import Scenario
from grape_research import (ResourceBudget,BudgetExhausted,qualified_fourier_multistart,
    convergence_assessment,refine,waveform_hash)
from continuation_risk_grape import refine as continuation,validate_schedule
from grape_development import load_config,make_ensembles,optimizer_options,load_reference_checkpoint,validate_reference_waveform,amplitude_headroom


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.base=json.loads(Path('challenge_config.json').read_text());self.device=Device(**self.base['device'])
        self.pulse=make_pulse(PulseParameters(15.6,.3,.5,1.,.3),self.device)
        self.options={**DEFAULTS,'harmonics':5,'seed':7500,'restarts':5,'qualified_perturbation_scale':.002,
            'quality_targets':self.base['quality_targets']}

    def test_qualified_five_harmonic_starts_reproducible_and_physical(self):
        first=qualified_fourier_multistart(self.pulse,self.device,self.options,1.03)
        repeat=qualified_fourier_multistart(self.pulse,self.device,self.options,1.03)
        basis,starts,records,_=first
        self.assertEqual(len(starts),5)
        for a,b in zip(starts,repeat[1]):np.testing.assert_array_equal(a,b)
        self.assertEqual(records[0]['perturbation_magnitude'],0.)
        self.assertTrue(all(r['nominal_fidelity']>=.999 and r['nominal_leakage']<=.0001 for r in records if r['accepted']))
        for start in starts:
            self.assertEqual(start.shape,(10,))
            np.testing.assert_array_equal(basis.waveform(start)[[0,-1]],0.)
            self.assertGreaterEqual(basis.constraints(start,self.device.amplitude_limit_mhz*MHZ_TO_RAD_NS/1.03).min(),-1e-10)

    def test_perturbations_adapt_and_bad_sources_rejected(self):
        _,_,records,_=qualified_fourier_multistart(self.pulse,self.device,{**self.options,'qualified_perturbation_scale':.1},1.03)
        self.assertTrue(any(r['attempt']>0 for r in records))
        with self.assertRaises(ValueError):qualified_fourier_multistart(self.pulse,self.device,{**self.options,'harmonics':2},1.03)
        zero=(self.pulse[0],np.zeros(len(self.pulse[0])),np.zeros(len(self.pulse[0])))
        with self.assertRaisesRegex(ValueError,'source pulse'):qualified_fourier_multistart(zero,self.device,self.options,1.03)

    def test_exact_warm_transfer(self):
        basis,starts,_,_=qualified_fourier_multistart(self.pulse,self.device,{**self.options,'restarts':1},1.03)
        wave=tuple(np.array([self.pulse[0],*basis.waveform(starts[0]).T]))
        _,again,records,_=qualified_fourier_multistart(wave,self.device,{**self.options,'restarts':1,'warm_start_coefficients':starts[0].tolist()},1.03)
        np.testing.assert_array_equal(starts[0],again[0]);self.assertEqual(waveform_hash(wave),records[0]['waveform_sha256'])

    def test_budget_checks_and_cache_accounting(self):
        budget=ResourceBudget(objective_limit=2,pass_limit=3)
        with patch('grape_research.evaluate_waveform',return_value={'fidelity':.9999,'leakage':1e-6}):
            budget.nominal(self.pulse,self.device);budget.nominal(self.pulse,self.device)
        self.assertEqual(budget.production_channel_solves,1)
        budget.objective_evaluations=2
        with self.assertRaises(BudgetExhausted):budget.check(objectives=1)
        budget.surrogate_scenario_passes=2
        with self.assertRaises(BudgetExhausted):budget.check(passes=1)
        with patch('grape_research.perf_counter',return_value=2.):
            with self.assertRaises(BudgetExhausted):ResourceBudget(wall_limit=1.,started=0.).check()

    def test_continuation_zero_weight_matches_conventional_with_leakage(self):
        times=np.linspace(0.,1.,5);u=np.column_stack([.1*np.sin(np.pi*times),.02*np.sin(2*np.pi*times)])
        scenarios=[Scenario(),Scenario(amplitude_scale=.97,detuning_mhz=-3.)]
        options={**DEFAULTS,'substeps':1,'alpha':.95,'risk_weight':0.,'leakage_weight':2.,'smoothing':1e-6}
        a,ga=EnsembleGRAPE(times,self.device,scenarios,dict(options)).value_gradient(u)
        b,gb=RiskEnsembleGRAPE(times,self.device,scenarios,dict(options)).value_gradient(u)
        self.assertAlmostEqual(a,b,places=12);np.testing.assert_allclose(ga,gb,atol=1e-12)

    def test_refiner_counts_unsuccessful_restarts_under_one_cap(self):
        class ToyModel:
            def __init__(self,times,device,scenarios,options):pass
            def value_gradient(self,u):
                loss=float(np.sum(u*u))
                self.last_ensemble_metrics=dict(mean_training_fidelity=1-loss,
                    cvar95_training_infidelity=loss,smooth_cvar95_training_infidelity=loss,
                    mean_training_leakage=0.,maximum_training_leakage=0.)
                return loss,2*u
        options={**self.options,'iterations':10,'max_objective_evaluations':8}
        with patch('grape_research.evaluate_waveform',return_value={'fidelity':.9999,'leakage':1e-6}),warnings.catch_warnings():
            warnings.simplefilter('ignore');_,history=refine(self.pulse,self.device,[object(),object()],options,1.03,model_class=ToyModel)
        result=history[-1]
        self.assertLessEqual(result['objective_evaluations'],8)
        self.assertEqual(result['surrogate_scenario_passes'],4*result['objective_evaluations'])
        self.assertEqual(result['total_dissipative_passes'],result['surrogate_scenario_passes']+result['production_channel_solves'])
        self.assertGreater(result['production_channel_solves'],0)
        self.assertGreater(len(result['restart_outcomes']),1)
        json.dumps(result,allow_nan=False)

    def test_convergence_needs_all_criteria(self):
        basis=FourierControls(self.pulse[0],5,.5);c=np.zeros(10)
        assess=lambda success,g,change:convergence_assessment(success,g,c,basis,.4,change,{})
        self.assertTrue(assess(True,np.zeros(10),(0.,0.))['converged'])
        self.assertFalse(assess(False,np.zeros(10),(0.,0.))['converged'])
        self.assertFalse(assess(True,np.ones(10),(0.,0.))['converged'])
        self.assertFalse(assess(True,np.zeros(10),(.1,.1))['converged'])

    def test_schedules(self):
        self.assertEqual(validate_schedule([0,.1,.5,1]),[0.,.1,.5,1.])
        for bad in ([.1,1],[0,0,1],[0,1,.5],[0,float('nan')]):
            with self.assertRaises(ValueError):validate_schedule(bad)

    def fake_stage(self,fail_index=None):
        calls=[];basis=FourierControls(self.pulse[0],5,self.device.amplitude_limit_mhz*MHZ_TO_RAD_NS)
        def stage(pulse,device,scenarios,options,headroom,*,model_class,budget,quota):
            index=len(calls);calls.append((waveform_hash(pulse),options.copy()))
            budget.objective_evaluations+=2;budget.surrogate_scenario_passes+=20;budget.production_channel_solves+=1
            if index==fail_index:raise RuntimeError('Injected stage failure')
            coeff=basis.fit(np.array(pulse)[1:].T);coeff[0]+=.00001
            wave=tuple(np.array([pulse[0],*basis.waveform(coeff).T]))
            record=dict(mean_training_fidelity=.999,mean_training_leakage=1e-5,
                smooth_cvar95_training_infidelity=.002,cvar95_training_infidelity=.002,
                regularization=.000001,nominal_fidelity=.9995,nominal_leakage=1e-5,coefficients=coeff.tolist())
            last=dict(training_metrics=record,loss=.001,coefficients=coeff.tolist(),accepted_steps=[record],
                converged=False,message='Injected iteration limit',budget_exhaustion='iteration limit')
            return wave,[last]
        return calls,stage

    def test_continuation_schedule_transfer_and_total_cost(self):
        calls,stage=self.fake_stage()
        options={**self.options,'iterations':150,'leakage_weight':2.,'max_objective_evaluations':1000}
        with patch('continuation_risk_grape.stage_refine',side_effect=stage):_,h=continuation(self.pulse,self.device,[],options,1.03)
        result=h[-1];self.assertEqual([r[1]['risk_weight'] for r in calls],[0.,.1,.5,1.])
        self.assertEqual(sum(r[1]['restarts'] for r in calls),5)
        for i in range(1,4):
            self.assertEqual(calls[i][0],result['stages'][i-1]['output_waveform_sha256'])
            self.assertEqual(calls[i][1]['warm_start_coefficients'],result['stages'][i-1]['transferred_coefficients'])
        self.assertEqual(result['objective_evaluations'],8)
        self.assertEqual(result['total_dissipative_passes'],84)
        self.assertFalse(result['converged'])

    def test_failed_stage_preserves_wave_and_charges_cost(self):
        calls,stage=self.fake_stage(fail_index=1)
        with patch('continuation_risk_grape.stage_refine',side_effect=stage):_,h=continuation(self.pulse,self.device,[],self.options,1.03)
        result=h[-1];failed=result['stages'][1]
        self.assertEqual(failed['input_waveform_sha256'],failed['output_waveform_sha256'])
        self.assertEqual(calls[2][0],failed['input_waveform_sha256'])
        self.assertEqual(result['objective_evaluations'],8)
        self.assertFalse(result['schedule_complete'])

    def test_training_selection_separation_and_no_validation_generation(self):
        c=load_config('grape_development_config.json',True);calls=[]
        from challenge_model import monte_carlo_scenarios
        def tracked(n,seed,**kwargs):calls.append((n,seed));return monte_carlo_scenarios(n,seed,**kwargs)
        with patch('grape_development.monte_carlo_scenarios',side_effect=tracked):training,selection=make_ensembles(c,self.base)
        self.assertEqual(len(training),110);self.assertEqual(len(selection),200)
        self.assertEqual(calls,[(100,c['training_seed']),(200,c['selection_seed'])])
        self.assertNotIn(c['reserved_validation_seed'],[seed for n,seed in calls])
        with tempfile.TemporaryDirectory() as d:
            bad=json.loads(Path('grape_development_config.json').read_text());bad['selection_seed']=bad['training_seed']
            path=Path(d)/'bad.json';path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load_config(path)

    def test_regimes_clear_other_caps_and_diagnostic_explicit(self):
        full=load_config('grape_development_config.json');short=load_config('grape_development_config.json',True)
        self.assertEqual(full['iterations'],150);self.assertEqual(full['objective_calls'],1000)
        self.assertEqual(short['objective_calls'],24);self.assertEqual(short['mode'],'short_diagnostic')
        for regime,key in [('equal_objectives','max_objective_evaluations'),('equal_passes','max_total_scenario_passes'),('equal_wall','max_wall_seconds')]:
            options=optimizer_options(full,self.base,7500,regime,'risk');self.assertIn(key,options)
            self.assertEqual(options['restarts'],5)
            if regime!='equal_wall':self.assertNotIn('max_wall_seconds',options)

    def test_reference_content_is_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.json'
            path.write_text(json.dumps({'calibrated_drag':{'waveform':[1,2,3]}}))
            first,digest=load_reference_checkpoint(path)
            self.assertEqual(first['waveform'],[1,2,3])
            path.write_text(json.dumps({'calibrated_drag':{'waveform':[1,2,4]}}))
            self.assertNotEqual(load_reference_checkpoint(path)[1],digest)

    def test_fixed_duration_grid_and_manufacturing_headroom(self):
        reference={'waveform':np.asarray(self.pulse).tolist()}
        np.testing.assert_array_equal(validate_reference_waveform(reference,self.base),self.pulse)
        wrong={**self.base,'gate_duration_ns':20.}
        with self.assertRaisesRegex(ValueError,'duration'):validate_reference_waveform(reference,wrong)
        wrong={**self.base,'device':{**self.base['device'],'dt_ns':.1}}
        with self.assertRaisesRegex(ValueError,'sampling'):validate_reference_waveform(reference,wrong)
        wider={**self.base,'manufacturing_ranges':{**self.base['manufacturing_ranges'],'amplitude_fraction':.1}}
        self.assertEqual(amplitude_headroom(wider),1.1)
        with self.assertRaisesRegex(ValueError,'headroom'):validate_reference_waveform(reference,wider)


if __name__=='__main__':unittest.main()
