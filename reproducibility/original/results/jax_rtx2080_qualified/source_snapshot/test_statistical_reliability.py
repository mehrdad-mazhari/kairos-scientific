"""Certification coverage, stage separation, shared controls and adversarial probes."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import binom

from challenge_model import Device,Scenario,monte_carlo_scenarios
from pipeline_settings import resolve_config,validate_reliability_settings,apply_quick_mode
from statistical_certification import tail_bound_details,certify,engineering_assessment
from robust_optimization import summarize_trials,TrainingEvaluator,validate_config,select_robust_candidate


def config():
    return json.loads(Path(__file__).with_name('challenge_config.json').read_text())


class ReliabilityTests(unittest.TestCase):
    def test_exact_binomial_order_statistic_and_coverage(self):
        for n in (59,200,2000):
            values=np.linspace(.98,1.,n)
            bound=tail_bound_details(values);k=bound['order_statistic_index']
            self.assertTrue(bound['supported'])
            self.assertEqual(bound['lower_bound'],values[k-1])
            self.assertLessEqual(binom.cdf(k-1,n,.05),.05+1e-14)
            self.assertGreater(binom.cdf(k,n,.05),.05)
            self.assertEqual(bound['confidence_level'],.95)

    def test_insufficient_samples_do_not_become_observed_zero_fidelity(self):
        for n in (1,8,58):
            stats=summarize_trials([dict(fidelity=.9999,leakage=1e-6)]*n,config()['quality_targets'])
            certified=certify(stats,config()['quality_targets'])
            self.assertEqual(stats['p05_fidelity_lcb95'],0.)
            self.assertFalse(stats['tail_bound_supported'])
            self.assertGreater(stats['worst_sample_fidelity'],.999)
            self.assertGreater(stats['p05_fidelity'],.999)
            self.assertFalse(certified['passed'])
            self.assertIn('Insufficient',certified['reasons'][0])
        self.assertTrue(tail_bound_details([.999]*59)['supported'])

    def test_configured_sample_thresholds_replace_legacy_pdf_yield(self):
        stats=summarize_trials([dict(fidelity=.997,leakage=.001)]*200,config()['quality_targets'])
        self.assertEqual(stats['threshold_pass_fraction'],0.)
        self.assertFalse(certify(stats,config()['quality_targets'])['passed'])
        self.assertEqual(stats['thresholds']['leakage_max'],.0001)
        with self.assertRaises(ValueError):
            summarize_trials([])

    def test_invalid_fidelity_samples_rejected(self):
        for values in ([],[float('nan')],[1.1],[-.1],[[.99,.98]]):
            with self.subTest(values=values),self.assertRaises(ValueError):tail_bound_details(values)

    def assessment(self,**changes):
        c=resolve_config(config())
        cert=certify(summarize_trials([dict(fidelity=.9999,leakage=1e-6)]*2000,c['quality_targets']),c['quality_targets'])
        args=dict(selection_feasible=True,nominal=dict(fidelity=.9999,leakage=1e-6),
            hardware=dict(peak_amplitude_mhz=70.,feasible=True),duration_ns=c['gate_duration_ns'],config=c,
            certification=cert,public=dict(worst_leakage=1e-6),numerical_passed=True,
            truncation=dict(passed=True),worst_case=dict(enabled=False),
            search_costs=[dict(method='Nelder–Mead',success=False)])
        args.update(changes)
        return engineering_assessment(**args)

    def test_good_candidate_can_pass_even_when_optimizer_exhausted_budget(self):
        result=self.assessment()
        self.assertTrue(result['passed'])
        self.assertIn('limit',result['optimizer_status'])
        self.assertEqual(result['converged_restarts'],0)

    def test_nominal_pass_and_statistical_failure_are_separate(self):
        c=config();cert=certify(summarize_trials([dict(fidelity=.9999,leakage=1e-6)]*8),c['quality_targets'])
        result=self.assessment(certification=cert)
        self.assertTrue(result['nominal_quality_passed'])
        self.assertFalse(result['passed'])
        self.assertEqual([r['code'] for r in result['failures']],['STATISTICAL_CERTIFICATION_FAILURE'])

    def test_each_engineering_failure_is_explicit_and_cannot_pass(self):
        cases=[('selection_feasible',False,'OPTIMIZATION_NO_ELIGIBLE_CANDIDATE'),
            ('nominal',dict(fidelity=.998,leakage=1e-6),'NOMINAL_QUALITY_FAILURE'),
            ('hardware',dict(peak_amplitude_mhz=79.,feasible=True),'HARDWARE_CONSTRAINT_FAILURE'),
            ('duration_ns',15.60001,'DURATION_CONSTRAINT_FAILURE'),
            ('numerical_passed',False,'NUMERICAL_CONVERGENCE_FAILURE'),
            ('truncation',dict(passed=False),'HILBERT_SPACE_TRUNCATION_FAILURE'),
            ('public',dict(worst_leakage=.001),'PUBLIC_GRID_LEAKAGE_FAILURE'),
            ('worst_case',dict(enabled=True,violations=['leakage']),'OBSERVED_ROBUSTNESS_VIOLATION')]
        for key,value,code in cases:
            with self.subTest(key=key):
                result=self.assessment(**{key:value})
                self.assertFalse(result['passed'])
                self.assertIn(code,[r['code'] for r in result['failures']])

    def test_reproducible_independent_uniform_samples_and_ranges(self):
        kwargs=dict(coherence_min=.91,amplitude_fraction=.02,detuning_mhz=1.5,alpha_shift_mhz=5.)
        first=monte_carlo_scenarios(3000,2028,**kwargs)
        self.assertEqual(first,monte_carlo_scenarios(3000,2028,**kwargs))
        self.assertNotEqual(first,monte_carlo_scenarios(3000,2029,**kwargs))
        values=np.array([list(asdict(s).values()) for s in first])
        low=[.98,-1.5,-5.,.91,.91,.91,.91];high=[1.02,1.5,5.,1.,1.,1.,1.]
        self.assertTrue(np.all(values>=low) and np.all(values<=high))
        offdiag=np.corrcoef(values.T)-np.eye(7)
        self.assertLess(np.max(np.abs(offdiag)),.08)
        for n in (True,2.5,1):
            with self.assertRaises(ValueError):monte_carlo_scenarios(n,1)

    def test_training_configuration_is_separate_from_test_domain(self):
        c=config();before=copy.deepcopy(c['device'])
        c['training']={'coherence_min':.95,'manufacturing_ranges':dict(amplitude_fraction=.01,detuning_mhz=.5,alpha_shift_mhz=2.)}
        evaluator=TrainingEvaluator(c)
        self.assertEqual(max(s.detuning_mhz for s in evaluator.scenarios),.5)
        self.assertEqual(min(s.t1_1_scale for s in evaluator.scenarios),.95)
        validation=monte_carlo_scenarios(2000,c['mc_seed'],c['coherence_min'],**c['manufacturing_ranges'])
        self.assertGreater(max(abs(s.detuning_mhz) for s in validation),2.)
        self.assertEqual(c['device'],before)

    def test_legacy_configs_preserve_explicit_budgets_device_and_ranges(self):
        c=config()
        for section in ('training','worst_case'):c.pop(section,None)
        c.update(mc_selection_samples=1024,monte_carlo_samples=2048,search_seeds=[7,17])
        before=copy.deepcopy(c)
        resolved=resolve_config(c);validate_config(c)
        self.assertEqual(c,before)
        self.assertEqual(resolved['mc_selection_samples'],1024)
        self.assertEqual(resolved['monte_carlo_samples'],2048)
        self.assertEqual(resolved['training']['manufacturing_ranges'],c['manufacturing_ranges'])
        self.assertEqual(resolved['device'],c['device'])
        self.assertEqual(resolve_config(resolved),resolved)

    def test_seed_collisions_and_invalid_budgets_are_rejected(self):
        for mutate in (lambda c:c.update(mc_seed=c['mc_selection_seed']),
            lambda c:c.update(search_seeds=[c['mc_seed']]),
            lambda c:c['worst_case'].update(seeds=[2041,c['mc_seed']]),
            lambda c:c.update(monte_carlo_samples=20.5),
            lambda c:c['worst_case'].update(max_evaluations=4)):
            c=config();mutate(c)
            with self.assertRaises(ValueError):validate_config(c)

    def test_cli_desktop_and_json_run_controls_agree(self):
        from optimization_inputs import add_input_arguments,apply_input_arguments,build_config,input_command_arguments,run_field_values
        parser=argparse.ArgumentParser();add_input_arguments(parser)
        args=parser.parse_args(['--selection-samples','50','--validation-samples','300',
            '--nm-evaluations','120','--restart-seeds','7','17','27',
            '--worst-case-evaluations','24','--no-worst-case','--selection-metric','worst_sample_fidelity'])
        cli=config();apply_input_arguments(cli,args)
        forwarded=parser.parse_args(input_command_arguments(args));again=config();apply_input_arguments(again,forwarded)
        self.assertEqual(cli,again)
        c=config();inputs={name:str(c['gate_duration_ns'] if name=='tg_ns' else c['device'][name])
            for name in ('tg_ns','detuning_mhz','alpha_mhz','t1_1_ns','t1_2_ns','tphi_1_ns','tphi_2_ns')}
        desktop=build_config(c,inputs,True,True,run_field_values(cli),False)
        self.assertEqual(resolve_config(cli),desktop)
        self.assertEqual(cli['device'],c['device'])

    def test_shared_quick_mode_and_default_budgets(self):
        c=config()
        self.assertEqual(c['mc_selection_samples'],200)
        self.assertEqual(c['monte_carlo_samples'],2000)
        self.assertEqual(c['search_seeds'],[7,17,27,37,47])
        self.assertEqual(c['nelder_mead']['max_evaluations'],500)
        quick=apply_quick_mode(copy.deepcopy(c));validate_config(quick)
        self.assertLess(quick['mc_selection_samples'],c['mc_selection_samples'])
        self.assertLess(quick['monte_carlo_samples'],c['monte_carlo_samples'])
        self.assertEqual(quick['device'],c['device'])
        self.assertEqual(quick['quality_targets'],c['quality_targets'])

    def test_actual_selection_uses_headroom_and_recomputed_waveform(self):
        from challenge_model import PulseParameters,make_pulse
        c=config();c.update(mc_selection_samples=2,mc_selection_candidates=1)
        c['amplitude_headroom_scale']=1.2
        d=Device(**c['device']);p=PulseParameters(c['gate_duration_ns'],.3,.5,1.,0.)
        reference=dict(parameters=asdict(p),waveform=np.array(make_pulse(p,d)).tolist(),
            hardware=dict(peak_amplitude_mhz=1.),nominal=dict(fidelity=.9999,leakage=1e-6),violation=0.,selection_loss=.001)
        with tempfile.TemporaryDirectory() as folder,patch('robust_optimization.evaluate_set',return_value=[dict(fidelity=.9999,leakage=1e-6)]*2),patch('builtins.print'):
            selected=select_robust_candidate([reference],c,d,Path(folder),diagnostic_fallback=True)
        self.assertFalse(selected['mc_selection']['selection_feasible'])
        self.assertTrue(any('amplitude' in r for r in selected['mc_selection']['rejection_reasons']))

    def test_selection_metrics_are_configurable_and_all_methods_share_draws(self):
        c=config();c.update(mc_selection_samples=20,mc_selection_candidates=2,gate_duration_ns=24.)
        candidates=[dict(parameters=dict(duration_ns=24.,sigma_fraction=.3,beta=beta,amplitude=1.,phase_ramp_mhz=0.),
            nominal=dict(fidelity=.9999,leakage=1e-6),violation=0.,hardware=dict(peak_amplitude_mhz=50.),selection_loss=.001)
            for beta in (1.,.5)]
        for criterion,winner in [('p05_fidelity',.5),('mean_fidelity',1.),('worst_sample_fidelity',.5)]:
            c['mc_selection_criterion']=criterion
            seen=[]
            def evaluate(pulse,scenarios,*args):
                seen.append(scenarios)
                distribution=[.94]*2+[.999]*18 if len(seen)==1 else [.992]*20
                return [dict(fidelity=f,leakage=1e-6) for f in distribution]
            with tempfile.TemporaryDirectory() as folder,patch('robust_optimization.evaluate_set',side_effect=evaluate),patch('builtins.print'):
                selected=select_robust_candidate(candidates,c,Device(),Path(folder))
            self.assertEqual(selected['parameters']['beta'],winner)
            self.assertIs(seen[0],seen[1])
            self.assertNotEqual(seen[0],monte_carlo_scenarios(20,c['mc_seed']))

    def test_numerical_failures_persist_explicit_failed_reports(self):
        from contextlib import ExitStack
        import robust_optimization as workflow
        for diagnostic_failure in (False,True):
            c=config();c['risk_aware_grape']['enabled']=False;c.update(mc_selection_samples=2,monte_carlo_samples=2,search_seeds=[7])
            c['calibration']['enabled']=False;c['nelder_mead']['enabled']=False;c['grape']['enabled']=False
            c['worst_case']['enabled']=diagnostic_failure
            def metric(*args,**kwargs):
                fidelity=.998 if kwargs.get('strict') and not diagnostic_failure else .9999
                return dict(fidelity=fidelity,infidelity=1-fidelity,leakage=1e-6,max_trace_error=0.,min_output_eigenvalue=0.)
            with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                stack.enter_context(patch.object(workflow,'evaluate_waveform',side_effect=metric))
                stack.enter_context(patch('study_validation.truncation_check',return_value=dict(passed=True)))
                if diagnostic_failure:
                    stack.enter_context(patch('robustness_diagnostics.diagnose',side_effect=RuntimeError('nonphysical diagnostic solve')))
                for target in ('research_figures.save_research_outputs','challenge_report.save_plots','challenge_report.save_reliability_plots',
                    'stage_report.save_stage_comparisons','pulse_diagnostics.save_pulse_diagnostics',
                    'robust_optimization.print_final_summary','builtins.print'):
                    stack.enter_context(patch(target))
                result=workflow.run(c,folder)
                self.assertFalse(result['numerical_convergence_passed'])
                self.assertFalse(result['quality_passed'])
                failures=[r['code'] for r in result['engineering_assessment']['failures']]
                self.assertIn('NUMERICAL_CONVERGENCE_FAILURE',failures)
                if diagnostic_failure:self.assertIn('ROBUSTNESS_DIAGNOSTIC_FAILURE',failures)
                self.assertTrue((Path(folder)/'summary.json').exists())
                self.assertTrue((Path(folder)/'report.md').exists())

    def test_worst_case_search_is_bounded_reproducible_and_frozen(self):
        from robustness_diagnostics import diagnose
        c=resolve_config(config());c['worst_case'].update(max_evaluations=20,axis_points=3)
        pulse=np.array([[0.,1.],[0.,0.],[0.,0.]]);before=pulse.copy();calls=[]
        def analytic(t,i,q,d,s,**kwargs):
            calls.append(s)
            self.assertTrue(.97<=s.amplitude_scale<=1.03 and abs(s.detuning_mhz)<=3. and abs(s.alpha_shift_mhz)<=8.)
            self.assertTrue(all(.8<=getattr(s,name)<=1. for name in ('t1_1_scale','t1_2_scale','tphi_1_scale','tphi_2_scale')))
            error=(s.amplitude_scale-1)**2+.001*(s.detuning_mhz/3)**2+.001*(s.alpha_shift_mhz/8)**2+.001*(1-s.t1_1_scale)
            return dict(fidelity=.9999-error,leakage=1e-6+.001*(s.amplitude_scale-.97)*(s.detuning_mhz+3))
        result=diagnose(pulse,Device(**c['device']),c,evaluate=analytic)
        repeat=diagnose(pulse,Device(**c['device']),c,evaluate=analytic)
        self.assertEqual(result['observations'],repeat['observations'])
        self.assertEqual(result['minimum_observed'],repeat['minimum_observed'])
        self.assertFalse(result['mathematically_certified_global_bound'])
        self.assertTrue(all(r['objective_calls']<=20 for r in result['search_status']))
        self.assertLess(result['minimum_observed']['fidelity'],.9999)
        self.assertTrue(result['violations'])
        self.assertIn('joint_coherence',result['sensitivities'])
        np.testing.assert_array_equal(pulse,before)

    def test_zero_uncertainty_domain_diagnostic_is_valid(self):
        from robustness_diagnostics import diagnose
        c=resolve_config(config());c['manufacturing_ranges']=dict(amplitude_fraction=0.,detuning_mhz=0.,alpha_shift_mhz=0.);c['coherence_min']=1.
        result=diagnose(np.array([[0.,1.],[0.,0.],[0.,0.]]),Device(),c,
                        evaluate=lambda *a,**k:dict(fidelity=.9999,leakage=1e-6))
        self.assertEqual(result['channel_simulations'],1)
        self.assertEqual(result['sensitivities']['amplitude_scale']['fidelity_span'],0.)
        self.assertFalse(result['violations'])


if __name__=='__main__':unittest.main()
