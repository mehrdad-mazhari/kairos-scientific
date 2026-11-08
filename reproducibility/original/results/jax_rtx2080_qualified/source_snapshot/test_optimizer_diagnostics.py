import unittest,warnings
import numpy as np
from grape import FourierControls,DEFAULTS,refine
from challenge_model import Device,Scenario

class OptimizerDiagnosticTests(unittest.TestCase):
    def test_objective_algebra_on_100_distinct_losses(self):
        from optimizer_diagnostics import validate_objective_algebra
        result=validate_objective_algebra()
        self.assertTrue(result['passed'],result)
        self.assertEqual(result['distinct_infidelities'],100)
        self.assertGreater(result['smoothing_gap'],0.)

    def test_full_profile_does_not_inherit_quick_budget(self):
        from research_benchmark import load_profile,effective_grape_options
        full=load_profile('research_config.json');quick=load_profile('research_config.json',True)
        self.assertEqual(full['mode'],'full')
        self.assertEqual(quick['mode'],'quick')
        for key in ('objective_calls','iterations','wall_seconds','training_samples','selection_samples','validation_samples'):
            self.assertGreater(full[key],quick[key])
        self.assertGreaterEqual(full['training_samples'],100)
        self.assertGreaterEqual(len(full['optimizer_seeds']),10)
        base={'grape':{**DEFAULTS,'max_objective_evaluations':3,'max_wall_seconds':1,'max_surrogate_passes':72}}
        calls=effective_grape_options(base,full,full['optimizer_seeds'][0],'equal_evaluations')
        wall=effective_grape_options(base,full,full['optimizer_seeds'][0],'equal_wall')
        self.assertEqual(calls['max_objective_evaluations'],full['objective_calls'])
        self.assertNotIn('max_wall_seconds',calls)
        self.assertNotIn('max_objective_evaluations',wall)
        self.assertEqual(wall['max_wall_seconds'],full['wall_seconds'])
        self.assertEqual(wall['max_surrogate_passes'],full['scenario_passes'])

    def test_constrained_fit_is_feasible_and_not_worse_than_radial(self):
        t=np.linspace(0,16,81);b=FourierControls(t,3,.5)
        controls=np.column_stack([.7*np.sin(np.pi*t/16),.1*np.sin(2*np.pi*t/16)])
        c=b.fit(controls);c*=min(1,.999*.4/np.linalg.norm(b.waveform(c),axis=1).max())
        fitted,info=b.constrained_fit(controls,.4)
        self.assertGreaterEqual(b.constraints(fitted,.4).min(),-1e-10)
        self.assertLessEqual(np.sum((b.waveform(fitted)-controls)**2),np.sum((b.waveform(c)-controls)**2)+1e-10)
        np.testing.assert_array_equal(b.waveform(fitted)[[0,-1]],0.)
    def test_no_change_warning_and_execution_audit(self):
        t=np.linspace(0,1,4);pulse=(t,.1*np.sin(np.pi*t),.02*np.sin(2*np.pi*t))
        o={**DEFAULTS,'harmonics':2,'substeps':1,'iterations':1,'restarts':1,'max_objective_evaluations':2}
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always');wave,h=refine(pulse,Device(),[Scenario()],o,1.03)
        a=h[-1]['execution_audit']
        self.assertFalse(a['meaningful_parameter_change'])
        self.assertTrue(any('without meaningful' in str(x.message) for x in messages))
        self.assertEqual(a['initial_waveform_sha256'],a['final_waveform_sha256'])
        self.assertEqual(a['retained_feasible_improvements'],0)
        self.assertTrue(a['evaluation_trace'])
        self.assertIsNotNone(a['final_gradient_norm'])
        self.assertEqual(h[-1]['objective_evaluations'],2)

if __name__=='__main__':unittest.main()
