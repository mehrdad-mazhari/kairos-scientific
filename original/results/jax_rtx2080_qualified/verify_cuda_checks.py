"""Additional live CUDA checks; run from the project virtual environment."""
import json,os,sys,warnings
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from challenge_model import Device,Scenario,hardware_metrics
from grape import DEFAULTS,EnsembleGRAPE
from risk_aware_grape import RiskEnsembleGRAPE
from jax_backend import JAXEnsemble,make_model,gpu_preflight

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

protocol=json.loads((OUT/'protocol.json').read_text());c=protocol['config'];base=protocol['base_config']
device=Device(**base['device']);wave=np.asarray(json.loads((OUT/'pulse.json').read_text())['waveform'])
controls=wave[1:].T
scenarios=[Scenario(**s) for s in json.loads((OUT/'benchmark_scenarios.json').read_text())]
options={**DEFAULTS,'substeps':c['substeps'],'alpha':.95,'risk_weight':1.,'smoothing':1e-6,'quality_targets':base['quality_targets']}
state_checks={}
for precision in c['precisions']:
    acceleration=dict(backend='jax_gpu',precision=precision,batch_size=c['batch_size'],fallback_to_cpu=False,require_accuracy_gate=False)
    model=JAXEnsemble(wave[0],device,scenarios,{**options,'acceleration':acceleration},RiskEnsembleGRAPE)
    state=model.metrics(controls,states=True)
    rho=state.transpose(0,2,1).reshape(-1,5,5).transpose(0,2,1)
    trace=float(np.max(abs(np.trace(rho,axis1=1,axis2=2)-1)))
    herm=float(np.max(abs(rho-np.conj(rho.transpose(0,2,1)))))
    eig=float(np.linalg.eigvalsh(rho).min())
    state_checks[precision]=dict(actual_platform=model.jax_device.platform,scenarios=len(scenarios),
        maximum_trace_error=trace,hermiticity_error=herm,minimum_eigenvalue=eig,
        passed=trace<=2e-6 and herm<=2e-6 and eig>=-2e-6,diagnostics=model.diagnostics())
    save('full_ensemble_state_checks.json',state_checks)
assert state_checks['float64']['passed'],state_checks

acceleration=dict(backend='jax_gpu',precision='float64',batch_size=c['batch_size'],fallback_to_cpu=False,
    require_accuracy_gate=True,qualification_report=str(OUT/'gpu_qualification_float64.json'))
preflight=gpu_preflight(wave,device,scenarios[:100],{**options,'acceleration':acceleration})
save('live_gpu_preflight.json',preflight)

# The first actual CUDA float32 benchmark scenario already failed its metric
# tolerance. Reuse this numerical test case to check automatic CPU fallback.
acceleration=dict(backend='jax_gpu',precision='float32',batch_size=c['batch_size'],fallback_to_cpu=True,require_accuracy_gate=True)
model=make_model(wave[0],device,scenarios[:1],{**options,'acceleration':acceleration},RiskEnsembleGRAPE)
with warnings.catch_warnings(record=True) as messages:
    value,gradient=model.value_gradient(controls)
reference,g=RiskEnsembleGRAPE(wave[0],device,scenarios[:1],dict(options)).value_gradient(controls)
fallback=dict(diagnostics=model.diagnostics(),warnings=[str(m.message) for m in messages],
    objective_difference=abs(value-reference),gradient_difference=float(np.max(abs(gradient-g))))
save('accuracy_fallback_smoke.json',fallback)
assert fallback['diagnostics']['actual_backend']=='cpu',fallback
assert value==reference and np.array_equal(gradient,g),fallback

hardware=hardware_metrics(*wave,device)
headroom=max(base['amplitude_headroom_scale'],1+base['manufacturing_ranges']['amplitude_fraction'])
constraints=dict(**hardware,headroom=headroom,
    headroom_amplitude_feasible=hardware['peak_amplitude_mhz']*headroom<=device.amplitude_limit_mhz+1e-10,
    exact_zero_endpoints=bool(np.all(wave[1:,[0,-1]]==0)))
save('constraint_checks.json',constraints)
assert constraints['headroom_amplitude_feasible'] and constraints['exact_zero_endpoints'],constraints
print('All-state float64 checks, live CUDA preflight, actual accuracy fallback and waveform constraints verified.')
