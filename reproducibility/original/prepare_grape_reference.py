"""Prepare a stronger nominal-only DRAG reference without selection/validation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from challenge_model import Device,evaluate_waveform
from reference_calibration import calibrate_reference
from grape import DEFAULTS
from grape_research import qualified_fourier_multistart,ResourceBudget,nominal_qualified
from grape_development import write_json,amplitude_headroom
from research_benchmark import checkpoint,digest


def prepare(output,max_evaluations=500):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    base=json.loads(Path('challenge_config.json').read_text());base['calibration']['max_evaluations']=max_evaluations
    device=Device(**base['device']);source=Path('results/research_benchmark_complete/references.json')
    original=json.loads(source.read_text())
    protocol=dict(base=base,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        calibration_code_sha256=hashlib.sha256(Path('reference_calibration.py').read_bytes()).hexdigest(),
        selection_used=False,validation_used=False)
    identity=digest(protocol);old=checkpoint(out/'protocol.json',lambda:dict(signature=identity,**protocol))
    if old['signature']!=identity:raise ValueError('reference preparation changed; use another output')
    # Persist the expensive nominal search before any export/qualification checks.
    record=checkpoint(out/'calibration_checkpoint.json',lambda:calibrate_reference(original['references']['drag_baseline'],'drag_baseline',base,device))
    wave=np.asarray(record['waveform'],dtype=float)
    nominal=evaluate_waveform(*wave,device,strict=True)
    if not nominal_qualified(nominal,base['quality_targets']):raise RuntimeError('Nominally optimized reference does not qualify; keep previous reference')
    record['strict_nominal']=nominal
    write_json(out/'references.json',dict(calibrated_drag=record))
    config=json.loads(Path('grape_development_config.json').read_text());previous=config['reference_checkpoint']
    config['reference_checkpoint']=str(out/'references.json');config['diagnostic']['reference_checkpoint']=previous
    write_json(out/'full_config.json',config)
    banks={}
    for seed in config['optimizer_seeds']:
        budget=ResourceBudget()
        _,starts,records,fit=qualified_fourier_multistart(wave,device,{**DEFAULTS,'harmonics':5,'restarts':5,
            'seed':seed,'quality_targets':base['quality_targets'],'qualified_perturbation_scale':.002},amplitude_headroom(base),budget)
        banks[str(seed)]=dict(records=records,fit=fit,cost=budget.snapshot(),qualified_starts=len(starts))
    write_json(out/'initialization_banks.json',banks)
    result=dict(calibration=record['calibration'],strict_nominal=nominal,nominal_qualified=True,
        nominal_only=True,selection_used=False,validation_used=False,max_calibration_evaluations=max_evaluations,
        source_hash=protocol['calibration_code_sha256'],qualified_start_counts={seed:bank['qualified_starts'] for seed,bank in banks.items()},
        note='Stronger reference for a future matched-start rerun; the completed pilot retains its original five-call reference.')
    write_json(out/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/grape_development_reference');parser.add_argument('--max-evaluations',type=int,default=500)
    args=parser.parse_args();prepare(args.output,args.max_evaluations)
