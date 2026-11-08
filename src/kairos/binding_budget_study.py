"""Separate binding-budget development study; reads saved devices, never samples."""
import argparse,fcntl,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .challenge_model import Device,Scenario,evaluate_waveform,hardware_metrics
from .grape import EnsembleGRAPE
from .risk_aware_grape import RiskEnsembleGRAPE
from .binding_budget_optimizer import refine,waveform_hash
from .binding_budget_continuation import refine as continuation
from .grape_development import optimizer_options,amplitude_headroom,selection_measure,write_json,write_csv
from .full_matched_study import resolve_backend
from .full_matched_report import verify_trial,all_histories

OLD=Path('results/full_matched_budget')
OUT=Path('results/binding_budget_v2')

def report():
    rows=[]
    for p in sorted(OUT.glob('trial_*.json')):
        if p.stem.endswith(('_progress','_optimization')):continue
        r=json.loads(p.read_text());h=r['history'][-1];stats=r['selection']['statistics']
        caps=r['caps'];key={'equal_objectives':'objective_evaluations','equal_passes':'total_dissipative_passes','equal_wall':'elapsed_seconds'}[r['regime']]
        actual=h[key];cap=caps[key]
        terms=[x['message'] for x in all_histories(h)]
        rows.append(dict(trial=p.stem,regime=r['regime'],level=r['level'],seed=r['seed'],method=r['method'],cap=cap,
            actual_matching_resource=actual,utilization=actual/cap,overshoot=max(0,actual-cap),
            cap_effective=bool(actual/cap>=.90 and not any('Iteration limit reached' in t for t in terms)),
            objective_calls=h['objective_evaluations'],scenario_passes=h['total_dissipative_passes'],
            wall_seconds=h['elapsed_seconds'],external_optimizer_seconds=r['external_optimizer_seconds'],
            gpu_backend_seconds=h['gpu_backend_wall_seconds'],iterations=h['solver_iterations'],
            nominal_fidelity=r['nominal']['fidelity'],nominal_leakage=r['nominal']['leakage'],
            **{k:stats[k] for k in ('mean_fidelity','p05_fidelity','cvar95_infidelity','mean_leakage','worst_sample_leakage')},
            converged=h['converged'],numerical_pass=r['numerical']['passed'],selection_feasible=r['selection_feasible'],
            waveform_sha256=r['waveform_sha256'],termination=json.dumps(terms)))
    if rows:write_csv(OUT/'performance_vs_cost.csv',rows)
    write_json(OUT/'status.json',dict(completed=len(rows),required=54,status='COMPLETE' if len(rows)==54 else 'RUNNING',
        validation_generated=False,efficiency_superiority_established=False))
    lines=['# Binding computational-budget study','',f'{len(rows)}/54 trials complete. Development selection only; no independent validation.',
        '', 'Caps are upper bounds, not proof of equal expenditure. Compare actual resource consumption and utilization in performance_vs_cost.csv. '
        'Wall deadlines are checked between synchronized calls: one in-flight call may overrun. No efficiency superiority is established. '
        'Early convergence or unused cap must be reported rather than called a matched comparison.', '',
        'Two predeclared levels: 24/48 joint objective-gradient calls; 5280/10560 total dissipative passes; 120/240 optimizer seconds. '
        'Three saved seeds, three methods, five restart allocations, unchanged continuation schedule and physical constraints. '
        'The 100000-iteration ceiling is a safety guard, not an intended stopping rule. '
        'Per-stage and per-restart resource shares carry unused work forward. '
        'All qualification and nominal retention inside optimization are charged; shared warmup, post-trial numerical checks, '
        'and CPU selection are separately timed. Pass units are ledger units, not equivalent FLOPs. '
        'Risk uses two forward/gradient passes per device; conventional one. '
        'Interrupted trials are archived and their sunk work is additional campaign cost.', '']
    if rows:
        lines+=['|Regime / level / method|n|Mean CVaR95|Mean fidelity|Mean leakage|Mean actual cost|', '|---|---:|---:|---:|---:|---:|']
        for regime in ('equal_objectives','equal_passes','equal_wall'):
            for level in (0,1):
                for method in ('conventional','risk','continuation'):
                    rs=[r for r in rows if (r['regime'],r['level'],r['method'])==(regime,level,method)]
                    if rs:lines.append(f"|{regime}/{level}/{method}|{len(rs)}|{np.mean([r['cvar95_infidelity'] for r in rs]):.8g}|{np.mean([r['mean_fidelity'] for r in rs]):.9g}|{np.mean([r['mean_leakage'] for r in rs]):.8g}|{np.mean([r['actual_matching_resource'] for r in rs]):.5g}|")
    (OUT/'study_report.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'study.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=json.loads((OLD/'protocol.json').read_text());base=old['base_config'];c=old['config']
    device=Device(**base['device']);pulse=np.array(json.loads((OLD/'common_reference.json').read_text())['waveform'])
    training=[Scenario(**s) for s in json.loads((OLD/'training_devices.json').read_text())]
    selection=[Scenario(**s) for s in json.loads((OLD/'selection_devices.json').read_text())]
    assert len(training)==110 and len(selection)==200
    files=list(old['source_hashes'])+['binding_budget_optimizer.py','binding_budget_continuation.py','binding_budget_study.py']
    protocol=dict(source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        original_protocol=old['signature'],base_config=base,levels=dict(objectives=[24,48],passes=[5280,10560],wall=[120,240]),
        seeds=[7500,7501,7502],validation_generated=False)
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol,'Frozen v2 source changed'
    else:write_json(OUT/'protocol.json',protocol)
    for f in files:
        target=OUT/'source_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(Path(f).read_bytes())
    for label in ('training','selection'):
        data=(OLD/f'{label}_devices.json').read_bytes();(OUT/f'{label}_devices.json').write_bytes(data)
    options=optimizer_options(c,base,7500,'equal_objectives','risk')
    acceleration,hardware,qualification=resolve_backend('gpu',pulse,device,training,options,OUT)
    write_json(OUT/'hardware.json',hardware);write_json(OUT/'qualification.json',qualification)
    c={**c,'acceleration':acceleration};report()
    for level in (0,1):
        for seed in (7500,7501,7502):
            for regime in ('equal_objectives','equal_passes','equal_wall'):
                for method in ('conventional','risk','continuation'):
                    stem=f'trial_{regime}_{level}_{seed}_{method}';path=OUT/f'{stem}.json'
                    if path.exists():continue
                    o=optimizer_options(c,base,seed,regime,method)
                    o.update(iterations=100000,max_total_iterations=100000,max_objective_evaluations=10**9)
                    for k in ('max_total_scenario_passes','max_wall_seconds'):o.pop(k,None)
                    key={'equal_objectives':'objective_evaluations','equal_passes':'total_dissipative_passes','equal_wall':'elapsed_seconds'}[regime]
                    cap={'equal_objectives':[24,48],'equal_passes':[5280,10560],'equal_wall':[120,240]}[regime][level]
                    o[{'equal_objectives':'max_objective_evaluations','equal_passes':'max_total_scenario_passes','equal_wall':'max_wall_seconds'}[regime]]=cap
                    o['progress_checkpoint']=str(OUT/f'{stem}_progress.json')
                    optpath=OUT/f'{stem}_optimization.json';started=time.perf_counter()
                    print('Optimizing',stem,flush=True)
                    if optpath.exists():saved=json.loads(optpath.read_text());wave=np.array(saved['waveform']);history=saved['history'];external=saved['external_optimizer_seconds']
                    else:
                        progress=Path(o['progress_checkpoint'])
                        if progress.exists():
                            archive=OUT/'interrupted_attempts';archive.mkdir(exist_ok=True)
                            progress.rename(archive/f'{progress.stem}_{time.time_ns()}.json')
                        wave,history=(continuation(pulse,device,training,o,amplitude_headroom(base)) if method=='continuation' else
                            refine(pulse,device,training,o,amplitude_headroom(base),model_class=EnsembleGRAPE if method=='conventional' else RiskEnsembleGRAPE))
                        external=time.perf_counter()-started
                        write_json(optpath,dict(waveform=np.asarray(wave).tolist(),history=history,external_optimizer_seconds=external))
                    h=history[-1]
                    if regime!='equal_wall':assert h[key]<=cap,'Hard resource cap exceeded'
                    assert h['solver_iterations']<100000,'Iteration guard bound: comparison invalid'
                    nom=evaluate_waveform(*wave,device,strict=True);hw=hardware_metrics(*wave,device)
                    assert nom['fidelity']>=base['quality_targets']['nominal_fidelity_min'] and nom['leakage']<=base['quality_targets']['leakage_max']
                    assert hw['peak_amplitude_mhz']*amplitude_headroom(base)<=device.amplitude_limit_mhz+1e-8
                    r=dict(regime=regime,level=level,seed=seed,method=method,options=o,caps={key:cap},waveform=np.asarray(wave).tolist(),
                        waveform_sha256=waveform_hash(wave),history=history,external_optimizer_seconds=external,nominal=nom,hardware=hw)
                    r['numerical']=verify_trial(r,c,base)
                    if not r['numerical']['passed']:raise RuntimeError('Numerical gate failed; campaign stopped')
                    r['selection']=selection_measure(OUT,wave,device,selection,base['quality_targets'])
                    r['selection_feasible']=r['selection']['statistics']['worst_sample_leakage']<=base['quality_targets']['leakage_max']
                    write_json(path,r);report()
    manifest=json.loads((OUT/'completed_experiment_manifest.json').read_text())
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in manifest.items()),'Original experiment changed'
    print('COMPLETE; original experiment unchanged',flush=True)

if __name__=='__main__':main()
