import argparse
import json
from pathlib import Path
import unittest
import numpy as np
from grape import EnsembleGRAPE,DEFAULTS
from risk_aware_grape import empirical_cvar,smooth_cvar,RiskEnsembleGRAPE
from challenge_model import Device,Scenario,monte_carlo_scenarios
from pipeline_settings import resolve_config,validate_reliability_settings

class RiskTests(unittest.TestCase):
    def test_fractional_tail_mass(self):
        self.assertAlmostEqual(empirical_cvar([1.,2.,3.,4.],.625),(4.+.5*3.)/1.5)
        self.assertAlmostEqual(empirical_cvar([1.,2.,3.,4.],0.),2.5)
        self.assertEqual(empirical_cvar([1.,2.],.95),2.)
    def test_identical_small_and_extreme_tail(self):
        for alpha in (0.,.95,1-1e-10):
            for x in ([.1],[.1,.1],[0.,.1,.2]):
                v,g,_=smooth_cvar(x,alpha,1e-6)
                exact=empirical_cvar(x,alpha)
                self.assertTrue(np.isfinite(v) and np.all(np.isfinite(g)))
                self.assertAlmostEqual(g.sum(),1.,places=6)
                self.assertGreaterEqual(v+1e-10,exact)
                self.assertLessEqual(v-exact,1e-6*np.log(2)/(1-alpha)+1e-10)
    def test_cvar_gradient(self):
        x=np.array([.01,.02,.025,.027]);v,g,_=smooth_cvar(x,.8,.002)
        for i in range(len(x)):
            p=x.copy();m=x.copy();p[i]+=1e-7;m[i]-=1e-7
            self.assertAlmostEqual(g[i],(smooth_cvar(p,.8,.002)[0]-smooth_cvar(m,.8,.002)[0])/2e-7,places=6)
    def test_invalid_parameters(self):
        for x,a in [([], .95),([float('nan')],.95),([1.],1.),([1.],-.1)]:
            with self.assertRaises(ValueError):empirical_cvar(x,a)
        with self.assertRaises(ValueError):smooth_cvar([.1],.95,0.)
    def model(self,lam=1.,mu=2.):
        t=np.linspace(0.,1.,5);controls=np.column_stack([.15*np.sin(np.pi*t),.02*np.sin(2*np.pi*t)])
        o={**DEFAULTS,'substeps':1,'alpha':.95,'risk_weight':lam,'leakage_weight':mu,'smoothing':1e-5}
        s=[Scenario(),Scenario(.97,-3.,8.,.8,.8,.8,.8)]
        return RiskEnsembleGRAPE(t,Device(levels=5),s,o),controls,s,o
    def test_dissipative_adjoint_gradient(self):
        model,u,_,_=self.model();_,g=model.value_gradient(u)
        for index in [(1,0),(2,1)]:
            p=u.copy();m=u.copy();p[index]+=1e-6;m[index]-=1e-6
            fd=(model.value_gradient(p)[0]-model.value_gradient(m)[0])/2e-6
            self.assertAlmostEqual(g[index],fd,delta=3e-7)
    def test_lambda_mu_zero_limit(self):
        model,u,s,o=self.model(0.,0.)
        conventional=EnsembleGRAPE(model.times,Device(levels=5),s,o)
        v,g=model.value_gradient(u);cv,cg=conventional.value_gradient(u)
        self.assertAlmostEqual(v,cv,places=12);np.testing.assert_allclose(g,cg,atol=1e-12)
    def test_weights(self):
        model,u,_,_=self.model(0.,0.);base=model.value_gradient(u)[0]
        model.options['risk_weight']=2.;risk=model.value_gradient(u)[0]
        self.assertAlmostEqual(risk-base,2*model.last_risk['smoothed_cvar'],places=12)
        model.options['leakage_weight']=3.;self.assertGreaterEqual(model.value_gradient(u)[0],risk-1e-12)
    def test_training_reproducibility_and_separation(self):
        c=resolve_config(json.loads(Path('challenge_config.json').read_text()));r=c['risk_aware_grape']
        data=monte_carlo_scenarios(r['training_samples'],r['training_seed'],c['coherence_min'],**c['manufacturing_ranges'])
        self.assertEqual(data,monte_carlo_scenarios(r['training_samples'],r['training_seed'],c['coherence_min'],**c['manufacturing_ranges']))
        self.assertNotEqual(data,monte_carlo_scenarios(r['training_samples'],c['mc_seed'],c['coherence_min'],**c['manufacturing_ranges']))
        r['training_seed']=c['mc_seed']
        with self.assertRaises(ValueError):validate_reliability_settings(c)
    def test_cli_risk_and_cvar(self):
        from optimization_inputs import add_input_arguments,apply_input_arguments,input_command_arguments
        p=argparse.ArgumentParser();add_input_arguments(p)
        args=p.parse_args(['--risk-aware-grape','--selection-metric','cvar95_infidelity','--validation-samples','2000'])
        c=json.loads(Path('challenge_config.json').read_text());apply_input_arguments(c,args)
        self.assertTrue(c['risk_aware_grape']['enabled']);self.assertEqual(c['monte_carlo_samples'],2000)
        self.assertEqual(p.parse_args(input_command_arguments(args)).selection_metric,'cvar95_infidelity')

if __name__=='__main__':unittest.main()
