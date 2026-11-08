"""Known optima, bounds, reproducibility and primary-search integration contracts."""
import argparse
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

from nelder_mead import minimize
from optimization_inputs import add_input_arguments, apply_input_arguments, input_command_arguments


class NelderMeadTests(unittest.TestCase):
    def test_rosenbrock_convergence_and_monotone_best_history(self):
        loss=lambda x:100*(x[1]-x[0]**2)**2+(1-x[0])**2
        result=minimize(loss,[(-2.,2.),(-1.,3.)],[-1.2,1.],max_evaluations=600)
        self.assertTrue(result.success)
        self.assertLess(result.fun,1e-8)
        np.testing.assert_allclose(result.x,[1.,1.],atol=1e-3)
        self.assertEqual(len(result.history),result.nfev)
        self.assertTrue(np.all(np.diff([h['best_loss'] for h in result.history])<=0.))

    def test_normalization_with_different_control_units_and_boundary_optimum(self):
        def loss(x):
            self.assertTrue(0. <= x[0] <= 1. and -1000. <= x[1] <= 1000.)
            return (x[0]-2.)**2+((x[1]-250.)/2000.)**2
        result=minimize(loss,[(0.,1.),(-1000.,1000.)],[.3,0.],max_evaluations=400)
        self.assertLess(abs(result.x[0]-1.),1e-7)
        self.assertLess(abs(result.x[1]-250.),.1)

    def test_budget_exhaustion_is_not_reported_as_convergence(self):
        result=minimize(lambda x:np.sum((x-.7)**2),[(0.,1.)]*4,[.2]*4,max_evaluations=5)
        self.assertFalse(result.success)
        self.assertEqual(result.nfev,5)
        self.assertIn('evaluations',result.message)

    def test_restart_seeds_are_reproducible_and_change_start(self):
        args=(lambda x:np.sum((x-.7)**2),[(0.,1.)]*4,[.2]*4)
        first=minimize(*args,seed=17,max_evaluations=30,randomize_start=True)
        repeat=minimize(*args,seed=17,max_evaluations=30,randomize_start=True)
        another=minimize(*args,seed=7,max_evaluations=30,randomize_start=True)
        self.assertEqual(first.history,repeat.history)
        self.assertNotEqual(first.history[0]['parameters'],another.history[0]['parameters'])
        self.assertNotEqual(first.history[0]['parameters'],[.2]*4)

    def test_invalid_budgets_bounds_tolerances_and_objectives(self):
        for kwargs in ({'max_evaluations':2},{'max_evaluations':True},{'xatol':0.},{'fatol':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                minimize(lambda x:0.,[(0.,1.)]*4,[.2]*4,**kwargs)
        with self.assertRaises(ValueError):
            minimize(lambda x:0.,[(0.,1.)],[2.])
        with self.assertRaises(ValueError):
            minimize(lambda x:float('nan'),[(0.,1.)],[.2])

    def test_cli_toggles_and_command_forwarding(self):
        parser=argparse.ArgumentParser()
        add_input_arguments(parser)
        args=parser.parse_args(['--non-interactive','--no-nelder-mead','--no-grape'])
        c=json.loads(Path(__file__).with_name('challenge_config.json').read_text())
        apply_input_arguments(c,args)
        self.assertFalse(c['nelder_mead']['enabled'])
        self.assertFalse(c['grape']['enabled'])
        self.assertEqual(input_command_arguments(args),['--no-nelder-mead','--no-grape'])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--nsga'])

    def test_workflow_import_does_not_load_removed_optimizer_dependency(self):
        subprocess.run([sys.executable,'-c',"import sys; import robust_optimization; assert 'pymoo' not in sys.modules; assert 'nsga3' not in sys.modules"],check=True)


if __name__=='__main__':
    unittest.main()
