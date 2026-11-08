import importlib,json,hashlib
from pathlib import Path
import numpy as np
from kairos import Device,Scenario
from kairos.challenge_model import model_operators,evaluate_waveform,hardware_metrics
from kairos.grape import FourierControls,EnsembleGRAPE,DEFAULTS
from kairos.risk_aware_grape import empirical_cvar
ROOT=Path(__file__).resolve().parents[1]

def test_package_imports():
    for name in ['challenge_model','grape','risk_aware_grape','continuation_risk_grape','jax_backend','statistical_certification','confirmatory_statistics','robust_optimization']:
        importlib.import_module('kairos.'+name)

def test_five_level_model_and_gradient():
    d=Device(levels=5);assert model_operators(5)[0][0].shape==(5,1)
    t=np.linspace(0,1,4);b=FourierControls(t,2,.1);c=np.array([.3,.05,.02,.01]);u=b.waveform(c)
    m=EnsembleGRAPE(t,d,[Scenario()],dict(DEFAULTS,substeps=1))
    v,g=m.value_gradient(u);anal=b.gradient(g);eps=1e-6
    numeric=[]
    for k in range(4):
        delta=np.eye(4)[k]*eps
        numeric.append((m.value_gradient(b.waveform(c+delta))[0]-m.value_gradient(b.waveform(c-delta))[0])/(2*eps))
    np.testing.assert_allclose(anal,numeric,atol=1e-8,rtol=1e-4)
    assert np.all(u[[0,-1]]==0)

def test_frozen_results_and_configuration():
    orig=ROOT/'reproducibility/original'
    manifest=json.loads((orig/'results/preconfirmatory_checks_v1/freeze_manifest.json').read_text())
    for rel,h in manifest.items():assert hashlib.sha256((orig/'results/preconfirmatory_checks_v1'/rel).read_bytes()).hexdigest()==h
    ledger=json.loads((orig/'results/confirmatory_validation_v1/completion_ledger.json').read_text())
    assert ledger['status']=='COMPLETE'
    assert ledger['completed_candidate_device_evaluations']==10000
    assert empirical_cvar(np.arange(100)/100)==np.mean(np.arange(95,100)/100)

def test_cpu_frozen_pulse():
    orig=ROOT/'reproducibility/original'
    data=json.loads((orig/'results/preconfirmatory_checks_v1/frozen_candidates.json').read_text())
    pulse=tuple(np.array(x) for x in data['conventional']['waveform'])
    m=evaluate_waveform(*pulse,Device(levels=5),strict=True)
    assert m['fidelity']>=.999 and m['leakage']<=1e-4
    assert hardware_metrics(*pulse,Device(levels=5))['peak_amplitude_mhz']*1.03<=80+1e-8


def load_tests(loader, tests, pattern):
    import unittest
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in globals().copy().items() if name.startswith('test_') and callable(value))
