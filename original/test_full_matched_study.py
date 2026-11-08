import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from challenge_model import Device
from full_matched_study import resolve_backend,validate_full
from grape_development import load_config
from full_matched_report import termination


class MatchedStudyTests(unittest.TestCase):
    def setUp(self):
        t=np.linspace(0,1,5);self.wave=np.array([t,.1*np.sin(np.pi*t),.01*np.sin(2*np.pi*t)])

    def test_cpu_mode_never_qualifies_or_uses_jax(self):
        with tempfile.TemporaryDirectory() as d,patch('full_matched_study.environment',return_value={}),patch('full_matched_study.qualify',side_effect=AssertionError('GPU must not initialize')):
            settings,info,q=resolve_backend('cpu',self.wave,Device(levels=5),[],{},Path(d))
        self.assertEqual(settings['backend'],'cpu');self.assertEqual(info['actual_backend'],'cpu')

    def test_auto_falls_back_on_failed_qualification(self):
        with tempfile.TemporaryDirectory() as d,patch('full_matched_study.environment',return_value={}),patch('full_matched_study.qualify',side_effect=RuntimeError('accuracy failure')):
            settings,info,q=resolve_backend('auto',self.wave,Device(levels=5),[],{},Path(d))
        self.assertEqual(settings['backend'],'cpu');self.assertIn('accuracy failure',info['fallback_reason'])

    def test_explicit_gpu_never_silently_falls_back(self):
        with tempfile.TemporaryDirectory() as d,patch('full_matched_study.environment',return_value={}),patch('full_matched_study.qualify',side_effect=RuntimeError('unavailable')):
            with self.assertRaisesRegex(RuntimeError,'Explicit GPU mode'):
                resolve_backend('gpu',self.wave,Device(levels=5),[],{},Path(d))

    def test_full_profile_rejects_quick_caps(self):
        c=load_config('full_matched_config.json');base=json.loads(Path(c['base_config']).read_text())
        validate_full(c,base)
        for key,value in [('iterations',24),('objective_calls',3),('substeps',2),('mode','short_diagnostic')]:
            with self.assertRaises(ValueError):validate_full({**c,key:value},base)

    def test_termination_categories_do_not_use_objective_decrease(self):
        for text,label in [('wall-clock budget exhausted','WALL_TIME_LIMIT'),('total dissipative scenario-pass budget exhausted','SCENARIO_PASS_LIMIT'),('objective evaluation budget exhausted','EVALUATION_LIMIT'),('aggregate iteration budget exhausted','ITERATION_LIMIT'),('accuracy gate failed','NUMERICAL_FAILURE')]:
            self.assertEqual(termination(dict(converged=False,message=text)),label)
        self.assertEqual(termination(dict(converged=False,message='Optimization terminated successfully')),'STOPPED_WITHOUT_CONVERGENCE')


if __name__=='__main__':unittest.main()
