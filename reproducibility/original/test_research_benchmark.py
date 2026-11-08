import json,tempfile,unittest
from pathlib import Path
import numpy as np
from research_benchmark import load_profile,checkpoint,digest
from research_statistics import paired_bootstrap
from risk_aware_grape import RiskEnsembleGRAPE
from grape import DEFAULTS,refine
from challenge_model import Device,Scenario

class ResearchTests(unittest.TestCase):
    def test_profiles_cannot_silently_reduce_full(self):
        full=load_profile('research_config.json');quick=load_profile('research_config.json',True)
        self.assertEqual(full['validation_samples'],2000);self.assertEqual(full['training_samples'],100)
        self.assertEqual(len(full['optimizer_seeds']),10);self.assertEqual(full['worst_calls'],500)
        self.assertEqual(quick['validation_samples'],8)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';bad=json.loads(Path('research_config.json').read_text());bad['validation_samples']=8;p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):load_profile(p)
    def test_checkpoint_does_not_repeat_completed_work(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trial.json';self.assertEqual(checkpoint(p,lambda:dict(value=3)),dict(value=3))
            self.assertEqual(checkpoint(p,lambda:1/0),dict(value=3))
            self.assertFalse(p.with_suffix('.tmp').exists())
    def test_paired_bootstrap_zero_and_reproducibility(self):
        x=np.linspace(.995,1.,30)[None,:]
        r=paired_bootstrap(x,x,repetitions=100)
        self.assertEqual(r['cvar_difference'],0.);self.assertEqual(r['cvar_ci'],[0.,0.]);self.assertFalse(r['superiority_supported'])
        y=x-.001
        a=paired_bootstrap(x,y,repetitions=100);b=paired_bootstrap(x,y,repetitions=100)
        self.assertEqual(a,b);self.assertLess(a['cvar_difference'],0.)
    def test_ablations_gradient(self):
        times=np.linspace(0.,1.,4);u=np.column_stack([.1*np.sin(np.pi*times),.02*np.sin(2*np.pi*times)])
        for mean,risk,worst,leak in [(1,0,0,0),(0,0,1,0),(0,1,0,0),(1,1,0,0),(1,1,0,2)]:
            o={**DEFAULTS,'substeps':1,'alpha':.95,'smoothing':1e-4,'mean_weight':mean,'risk_weight':risk,'worst_weight':worst,'leakage_weight':leak}
            m=RiskEnsembleGRAPE(times,Device(levels=5),[Scenario(),Scenario(.97,-3,8,.8,.8,.8,.8)],o)
            _,g=m.value_gradient(u);p=u.copy();q=u.copy();p[1,0]+=1e-6;q[1,0]-=1e-6
            self.assertAlmostEqual(g[1,0],(m.value_gradient(p)[0]-m.value_gradient(q)[0])/2e-6,delta=5e-7)
    def test_optimizer_seeds_produce_distinct_paired_initializations(self):
        times=np.linspace(0.,1.,4);pulse=(times,.1*np.sin(np.pi*times),.02*np.sin(2*np.pi*times))
        o={**DEFAULTS,'iterations':1,'harmonics':2,'substeps':1,'restarts':1,'max_objective_evaluations':3,'initialization_scale':.05}
        _,first=refine(pulse,Device(),[Scenario()],{**o,'seed':6200},1.03)
        _,second=refine(pulse,Device(),[Scenario()],{**o,'seed':6201},1.03)
        self.assertNotEqual(first[1]['coefficients'],second[1]['coefficients'])
        _,repeat=refine(pulse,Device(),[Scenario()],{**o,'seed':6200},1.03)
        self.assertEqual(first[1]['coefficients'],repeat[1]['coefficients'])

    def test_objective_and_pass_caps_report_exhaustion(self):
        times=np.linspace(0.,1.,4);pulse=(times,.1*np.sin(np.pi*times),.02*np.sin(2*np.pi*times))
        o={**DEFAULTS,'iterations':20,'harmonics':2,'substeps':1,'restarts':1,'max_objective_evaluations':3}
        wave,h=refine(pulse,Device(),[Scenario()],o,1.03)
        self.assertLessEqual(h[-1]['objective_evaluations'],3)
        self.assertTrue(any(r.get('optimizer_success') is False for r in h))
        self.assertTrue(np.isfinite(wave).all())
        o.pop('max_objective_evaluations');o['max_surrogate_passes']=3
        _,h=refine(pulse,Device(),[Scenario()],o,1.03)
        self.assertLessEqual(h[-1]['surrogate_scenario_passes'],3)

if __name__=='__main__':unittest.main()
