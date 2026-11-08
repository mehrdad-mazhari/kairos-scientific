import importlib.util,json,os,tempfile,unittest,warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
from grape import EnsembleGRAPE,DEFAULTS,FourierControls
from risk_aware_grape import RiskEnsembleGRAPE
from challenge_model import Device,Scenario,make_pulse,PulseParameters,evaluate_waveform
from jax_backend import JAXEnsemble,BackendUnavailable,AccuracyError,make_model,plan_batch_size,settings,is_oom,gpu_preflight


class BackendControlTests(unittest.TestCase):
    def test_cpu_default_does_not_initialize_jax(self):
        with patch('jax_backend.load_jax',side_effect=AssertionError('optional import')):
            model=make_model(np.linspace(0,1,3),Device(levels=5),[Scenario()],DEFAULTS,EnsembleGRAPE)
        self.assertIsInstance(model,EnsembleGRAPE)

    def test_missing_gpu_falls_back_without_changing_precision_or_objective(self):
        t=np.linspace(0,1,4);u=np.column_stack([.1*np.sin(np.pi*t),.02*np.sin(2*np.pi*t)])
        options={**DEFAULTS,'acceleration':{'backend':'jax_gpu','precision':'float32'}}
        with patch('jax_backend.load_jax',side_effect=BackendUnavailable('driver unavailable')),warnings.catch_warnings(record=True) as messages:
            model=make_model(t,Device(levels=5),[Scenario()],options,EnsembleGRAPE)
        self.assertEqual(model.backend_info['actual_backend'],'cpu');self.assertTrue(messages)
        v,g=model.value_gradient(u);cv,cg=EnsembleGRAPE(t,Device(levels=5),[Scenario()],options).value_gradient(u)
        self.assertEqual(v,cv);np.testing.assert_array_equal(g,cg)

    def test_research_gpu_fails_closed(self):
        options={**DEFAULTS,'acceleration':{'backend':'jax_gpu'},'require_verified_gpu':True}
        with patch('jax_backend.load_jax',side_effect=BackendUnavailable('no CUDA')):
            with self.assertRaises(BackendUnavailable):make_model(np.linspace(0,1,4),Device(levels=5),[Scenario()],options,EnsembleGRAPE)
        options['acceleration']['require_accuracy_gate']=False
        with self.assertRaises(AccuracyError):make_model(np.linspace(0,1,4),Device(levels=5),[Scenario()],options,EnsembleGRAPE)

    def test_cpu_evidence_cannot_authorize_a_gpu_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qualification.json';path.write_text(json.dumps(dict(passed=True,actual_platform='cpu')))
            options={**DEFAULTS,'acceleration':{'backend':'jax_gpu','qualification_report':str(path)}}
            pulse=(np.linspace(0,1,4),np.zeros(4),np.zeros(4))
            with self.assertRaisesRegex(AccuracyError,'CUDA accuracy'):
                gpu_preflight(pulse,Device(levels=5),[Scenario()],options)

    def test_standard_full_workflow_blocks_gpu_before_expensive_search(self):
        from robust_optimization import run
        config=json.loads(Path('challenge_config.json').read_text());config['grape']['acceleration']={'backend':'jax_gpu'}
        with tempfile.TemporaryDirectory() as directory,patch('robust_optimization.reference_pulses',side_effect=AssertionError('search must not start')):
            with self.assertRaisesRegex(AccuracyError,'qualification report'):run(config,directory)

    def test_memory_planning_and_configuration(self):
        single,_=plan_batch_size(128,312,'float32',2**30)
        double,_=plan_batch_size(128,312,'float64',2**30)
        self.assertGreaterEqual(single,double);self.assertGreater(double,0)
        self.assertEqual(plan_batch_size(128,312,'float64',1)[0],1)
        self.assertLessEqual(plan_batch_size(128,312,'float64',2**33,free_bytes=2**28)[0],double)
        self.assertTrue(is_oom(RuntimeError('RESOURCE_EXHAUSTED: out of memory')))
        for config in ({'precision':'float16'},{'backend':'cuda_fake'},{'batch_size':0},{'max_vram_gb':10}):
            with self.assertRaises(ValueError):settings({'acceleration':config})


@unittest.skipUnless(importlib.util.find_spec('jax') and importlib.util.find_spec('jaxlib'),'optional JAX not installed')
class JAXNumericalTests(unittest.TestCase):
    def setUp(self):
        self.device=Device(levels=5);self.t=np.linspace(0,1,5)
        self.u=np.column_stack([.1*np.sin(np.pi*self.t),.02*np.sin(2*np.pi*self.t)])
        self.scenarios=[Scenario(),Scenario(.97,-3,8,.8,.9,.85,.95),Scenario(1.03,3,-8,.9,.8,.95,.85)]
        self.o={**DEFAULTS,'substeps':2,'alpha':.95,'risk_weight':1.,'smoothing':1e-5,
            'acceleration':{'backend':'jax_cpu','batch_size':2,'precision':'float64','fallback_to_cpu':False,'require_accuracy_gate':False}}

    def test_cpu_analytic_and_jax_gradients_match_for_all_objectives(self):
        for cls,extra in [(EnsembleGRAPE,{}),(RiskEnsembleGRAPE,{}),(RiskEnsembleGRAPE,dict(mean_weight=0.,risk_weight=1.)),(RiskEnsembleGRAPE,dict(mean_weight=0.,risk_weight=0.,worst_weight=1.))]:
            options={**self.o,**extra};cpu=cls(self.t,self.device,self.scenarios,dict(options));jax=JAXEnsemble(self.t,self.device,self.scenarios,options,cls)
            v,g=jax.value_gradient(self.u);cv,cg=cpu.value_gradient(self.u)
            self.assertAlmostEqual(v,cv,places=11);np.testing.assert_allclose(g,cg,atol=1e-10,rtol=1e-8)
            self.assertEqual(jax.info['platform'],'cpu')

    def test_cvar_is_global_and_independent_of_batch_partition(self):
        one=JAXEnsemble(self.t,self.device,self.scenarios,self.o,RiskEnsembleGRAPE)
        many=JAXEnsemble(self.t,self.device,self.scenarios,{**self.o,'acceleration':{**self.o['acceleration'],'batch_size':1}},RiskEnsembleGRAPE)
        a,ga=one.value_gradient(self.u);b,gb=many.value_gradient(self.u)
        self.assertAlmostEqual(a,b,places=12);np.testing.assert_allclose(ga,gb,atol=1e-11)

    def test_finite_difference_and_fourier_constraints(self):
        basis=FourierControls(self.t,2,.5);c=basis.fit(self.u);model=JAXEnsemble(self.t,self.device,self.scenarios,self.o,RiskEnsembleGRAPE)
        _,g=model.value_gradient(basis.waveform(c));cg=basis.gradient(g);eps=1e-5
        for index in range(len(c)):
            step=np.eye(len(c))[index]*eps
            fd=(model.value_gradient(basis.waveform(c+step))[0]-model.value_gradient(basis.waveform(c-step))[0])/(2*eps)
            self.assertAlmostEqual(cg[index],fd,delta=1e-8)
        np.testing.assert_array_equal(basis.waveform(c)[[0,-1]],0.)
        self.assertGreaterEqual(basis.constraints(c,.4).min(),0.)

    def test_float32_and_float64_are_explicit(self):
        for precision,dtype in [('float32',np.float32),('float64',np.float64)]:
            model=JAXEnsemble(self.t,self.device,self.scenarios,{**self.o,'substeps':8,'acceleration':{**self.o['acceleration'],'precision':precision}},EnsembleGRAPE)
            self.assertIs(model.real,dtype)
            result=model.metrics(self.u)
            self.assertEqual(result.dtype,dtype)
            reference=np.array([[1-r['fidelity'],r['leakage']] for r in [evaluate_waveform(self.t,*self.u.T,self.device,s,strict=True) for s in self.scenarios]])
            np.testing.assert_allclose(result,reference,atol=2e-6,rtol=0.)

    def test_physical_states_trace_positivity_and_unconditional_metrics(self):
        model=JAXEnsemble(self.t,self.device,self.scenarios,self.o,EnsembleGRAPE)
        states=model.metrics(self.u,states=True)
        rho=states.transpose(0,2,1).reshape(-1,5,5).transpose(0,2,1)
        np.testing.assert_allclose(np.trace(rho,axis1=1,axis2=2),1.,atol=1e-11)
        np.testing.assert_allclose(rho,np.conj(rho.transpose(0,2,1)),atol=1e-11)
        self.assertGreater(np.linalg.eigvalsh(rho).min(),-1e-11)

    def test_production_accuracy_gate_and_oom_retry(self):
        model=JAXEnsemble(self.t,self.device,self.scenarios,self.o,RiskEnsembleGRAPE)
        with self.assertRaises(AccuracyError):model.validate(self.u)
        model=JAXEnsemble(self.t,self.device,self.scenarios,{**self.o,'substeps':8},RiskEnsembleGRAPE)
        self.assertTrue(model.validate(self.u)['passed'])
        self.assertFalse(model.info['gpu_qualified'])
        model.batch_size=4;model.info.pop('last_execution_batch_size',None);attempts=[]
        def allocation():
            attempts.append(model.batch_size)
            if len(attempts)==1:raise RuntimeError('out of memory')
            return 'ok'
        self.assertEqual(model._with_oom_retry(allocation),'ok')
        self.assertEqual(attempts,[4,2])

    def test_runtime_failure_uses_trusted_cpu_and_records_actual_backend(self):
        options={**self.o,'acceleration':{**self.o['acceleration'],'fallback_to_cpu':True,'require_accuracy_gate':True}}
        model=JAXEnsemble(self.t,self.device,self.scenarios,options,RiskEnsembleGRAPE)
        with patch.object(model,'validate',side_effect=AccuracyError('deliberate rejection')),warnings.catch_warnings(record=True):
            value,g=model.value_gradient(self.u)
        reference,cg=model.cpu.value_gradient(self.u)
        self.assertEqual(value,reference);np.testing.assert_array_equal(g,cg)
        self.assertEqual(model.backend_info['actual_backend'],'cpu');self.assertFalse(model.backend_info['gpu_qualified'])

    def test_optimizer_and_continuation_use_jax_but_nominal_qualification_is_cpu(self):
        from grape_research import refine
        from continuation_risk_grape import refine as continuation
        pulse=make_pulse(PulseParameters(15.6,.3,.5,1.,.3),self.device)
        backend=os.environ.get('KAIROS_TEST_OPTIMIZER_BACKEND','jax_cpu')
        precision=os.environ.get('KAIROS_TEST_OPTIMIZER_PRECISION','float64')
        options={**DEFAULTS,'iterations':4,'max_objective_evaluations':8,'restarts':5,'harmonics':5,'substeps':4,
            'seed':7500,'alpha':.95,'risk_weight':1.,'smoothing':1e-6,
            'acceleration':{'backend':backend,'precision':precision,'batch_size':2,'require_accuracy_gate':True,'fallback_to_cpu':False}}
        for name,optimizer in [('conventional',lambda:refine(pulse,self.device,self.scenarios,options,1.03,model_class=EnsembleGRAPE)),
                               ('risk',lambda:refine(pulse,self.device,self.scenarios,options,1.03,model_class=RiskEnsembleGRAPE)),
                               ('continuation',lambda:continuation(pulse,self.device,self.scenarios,options,1.03))]:
            with warnings.catch_warnings():warnings.simplefilter('ignore');wave,h=optimizer()
            result=h[-1];self.assertLessEqual(result['objective_evaluations'],8)
            np.testing.assert_array_equal(np.asarray(wave)[1:,[0,-1]],0.)
            nominal=evaluate_waveform(*wave,self.device,strict=True)
            self.assertGreaterEqual(nominal['fidelity'],.999)
            self.assertLessEqual(nominal['leakage'],1e-4)
            self.assertLessEqual(np.linalg.norm(np.asarray(wave)[1:],axis=0).max()*1.03,self.device.amplitude_limit_mhz*.002*np.pi+1e-10)
            if name!='continuation':self.assertEqual(result['acceleration']['actual_backend'],backend)
            else:
                self.assertEqual([s['lambda_value'] for s in result['stages']],[0.,.1,.5,1.])
                self.assertTrue(all(s['history'][-1]['acceleration']['actual_backend']==backend for s in result['stages']))


if __name__=='__main__':unittest.main()
