"""Small CPU demonstrations, never publication results or validation samples."""
import argparse,json
from pathlib import Path
import numpy as np
from kairos.challenge_model import Device,Scenario,evaluate_waveform,hardware_metrics,monte_carlo_scenarios
from kairos.grape import FourierControls,DEFAULTS,refine
from kairos.risk_aware_grape import refine as risk_refine,empirical_cvar
from kairos.continuation_risk_grape import refine as continuation_refine

p=argparse.ArgumentParser();p.add_argument('mode',choices=['simulate','evaluate','grape','risk','continuation','robustness']);args=p.parse_args()
# A frozen qualified pulse is reused as demonstration input, never refitted.
root=Path(__file__).resolve().parents[1]
frozen=json.loads((root/'reproducibility/original/results/preconfirmatory_checks_v1/frozen_candidates.json').read_text())
pulse=tuple(np.asarray(row) for row in frozen['conventional']['waveform'])
dev=Device(levels=5)
options={**DEFAULTS,'iterations':1,'restarts':1,'seed':1234,'risk_weight':1.,'alpha':.95,'smoothing':1e-6,'max_objective_evaluations':3,'initialization_scale':0.,'acceleration':{'backend':'cpu'}}
if args.mode in ('grape','risk'):
    pulse,history=(refine if args.mode=='grape' else risk_refine)(pulse,dev,[Scenario()],options,1.03)
    print('DEMONSTRATION optimizer telemetry:',history)
elif args.mode=='continuation':
    # Qualified research continuation requires one restart allocation per stage.
    options.update(restarts=2,continuation_schedule=[0.,1.],max_total_iterations=2,max_wall_seconds=60.,quality_targets={'nominal_fidelity_min':.999,'leakage_max':1e-4})
    pulse,history=continuation_refine(pulse,dev,[Scenario()],options,1.03)
    print('DEMONSTRATION continuation telemetry:',history)
if args.mode=='robustness':
    # Separate demonstration seed; never the reserved confirmatory seed.
    records=[evaluate_waveform(*pulse,dev,s,strict=True) for s in monte_carlo_scenarios(4,123456)]
    print('DEMONSTRATION ONLY',{'mean_fidelity':np.mean([v['fidelity'] for v in records]),'cvar95':empirical_cvar([1-v['fidelity'] for v in records])})
else:
    print('DEMONSTRATION ONLY',evaluate_waveform(*pulse,dev,strict=True))
print('Hardware:',hardware_metrics(*pulse,dev))
