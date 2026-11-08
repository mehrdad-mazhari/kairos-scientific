import unittest,warnings
from unittest.mock import patch
import numpy as np
from test_binding_budget_optimizer import DevelopmentTests
from binding_budget_optimizer import refine,ResourceBudget

class BindingCaps(DevelopmentTests):
    def test_resource_caps_bind_before_iteration_guard(self):
        for kind,cap in [('max_objective_evaluations',20),('max_total_scenario_passes',100),('max_wall_seconds',12)]:
            clock=[0.]
            class Model:
                def __init__(self,*args):pass
                def value_gradient(self,u):
                    clock[0]+=1.
                    loss=float(np.sum(u*u))
                    self.last_ensemble_metrics=dict(mean_training_fidelity=1-loss,cvar95_training_infidelity=loss,
                        smooth_cvar95_training_infidelity=loss,mean_training_leakage=0.,maximum_training_leakage=0.)
                    return loss,2*u
            o={**self.options,'iterations':100000,'max_total_iterations':100000,'max_objective_evaluations':10**9,kind:cap}
            with patch('binding_budget_optimizer.perf_counter',side_effect=lambda:clock[0]),patch('binding_budget_optimizer.evaluate_waveform',return_value=dict(fidelity=.9999,leakage=1e-6)),warnings.catch_warnings():
                warnings.simplefilter('ignore');budget=ResourceBudget(started=0.,objective_limit=o.get('max_objective_evaluations',10**9),pass_limit=o.get('max_total_scenario_passes',float('inf')),wall_limit=o.get('max_wall_seconds',float('inf')),iteration_limit=100000);_,history=refine(self.pulse,self.device,[object(),object()],o,1.03,model_class=Model,budget=budget)
            h=history[-1]
            self.assertLess(h['solver_iterations'],100000)
            self.assertNotIn('Iteration limit reached',h['message'])
            self.assertGreater(len(h['restart_outcomes']),1)
            key={'max_objective_evaluations':'objective_evaluations','max_total_scenario_passes':'total_dissipative_passes','max_wall_seconds':'elapsed_seconds'}[kind]
            self.assertLessEqual(h[key],cap+ (1 if kind=='max_wall_seconds' else 0))
            self.assertGreaterEqual(h[key]/cap,.7)

if __name__=='__main__':unittest.main()
