"""Prespecified descriptive post-analysis; never launches optimization or validation.

This separate analysis runs only after acquiring the campaign lock. It does not
change the frozen optimizer protocol or any completed trial checkpoint.
"""
import argparse,csv,fcntl,hashlib,json
from pathlib import Path
import numpy as np


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def finalize(out):
    protocol=json.loads((out/'protocol.json').read_text())
    for name,expected in protocol['source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Frozen source changed: '+name)
    summary=json.loads((out/'study_summary.json').read_text())
    with (out/'selection_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    with (out/'computational_costs.csv').open() as f:costs=list(csv.DictReader(f))
    questions={};paired_devices=[]
    for comparison,(first,second) in {'fixed_risk_vs_conventional':('conventional','risk'),
        'continuation_vs_conventional':('conventional','continuation'),
        'continuation_vs_fixed_risk':('risk','continuation')}.items():
        questions[comparison]={}
        for regime in protocol['config']['regimes']:
            pairs=[]
            for seed in protocol['config']['optimizer_seeds']:
                def find(method):return next((r for r in rows if r.get('regime')==regime and r.get('method')==method and r.get('seed')==str(seed)),None)
                a,b=find(first),find(second)
                if a is None or b is None:continue
                delta=float(a['cvar95_infidelity'])-float(b['cvar95_infidelity'])
                def cost(method):return next(r for r in costs if r['regime']==regime and r['method']==method and r['seed']==str(seed))
                ca,cb=cost(first),cost(second)
                pairs.append(dict(seed=seed,delta_cvar=delta,percent_improvement=100*delta/float(a['cvar95_infidelity']),
                    wall_ratio=float(cb['elapsed_seconds'])/float(ca['elapsed_seconds']),
                    pass_ratio=int(cb['total_dissipative_passes'])/int(ca['total_dissipative_passes']),
                    **{k+'_difference':float(b[k])-float(a[k]) for k in
                        ('nominal_fidelity','mean_fidelity','p05_fidelity','mean_leakage','worst_sample_leakage')}))
                aa=json.loads((out/f'{regime}_{first}_{seed}.json').read_text())
                bb=json.loads((out/f'{regime}_{second}_{seed}.json').read_text())
                for i in range(200):
                    def channel(run):return json.loads((out/'selection_channels'/f'{run["waveform_sha256"]}_{i}.json').read_text())
                    va,vb=channel(aa),channel(bb)
                    paired_devices.append(dict(comparison=comparison,regime=regime,seed=seed,selection_index=i,
                        first_fidelity=va['fidelity'],second_fidelity=vb['fidelity'],fidelity_difference=vb['fidelity']-va['fidelity'],
                        first_leakage=va['leakage'],second_leakage=vb['leakage']))
            deltas=[p['delta_cvar'] for p in pairs]
            questions[comparison][regime]=dict(paired_seeds=len(pairs),pairs=pairs,
                mean_delta_cvar=float(np.mean(deltas)) if pairs else None,
                seed_delta_range=[min(deltas),max(deltas)] if pairs else None,
                mean_percent_improvement=float(np.mean([p['percent_improvement'] for p in pairs])) if pairs else None,
                seeds_favoring_second=sum(v>0 for v in deltas),
                interpretation='INCOMPLETE' if len(pairs)<3 else
                    ('Lower empirical mean CVaR for '+second if np.mean(deltas)>0 else 'No lower empirical mean CVaR for '+second),
                statistical_superiority_established=False)
    if paired_devices:
        with (out/'paired_selection_devices.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired_devices[0]));writer.writeheader();writer.writerows(paired_devices)
    summary['direct_comparisons']=questions
    summary['question_3_efficiency_scope']='Read equal_passes and equal_wall separately, with actual wall/pass ratios; equal-call advantage alone does not establish efficiency.'
    summary['interrupted_attempts']=[str(p) for p in (out/'interrupted_attempts').glob('*.json')] if (out/'interrupted_attempts').exists() else []
    if summary['interrupted_attempts']:
        summary['decision']='NO-GO';summary['decision_reason']='Interrupted work requires separate cost audit before a matched-efficiency claim'
    before=out/'study_summary_runner.json'
    if not before.exists():before.write_bytes((out/'study_summary.json').read_bytes())
    save(out/'study_summary.json',summary)
    original=out/'study_report_runner.md'
    if not original.exists():original.write_bytes((out/'study_report.md').read_bytes())
    lines=['','## Prespecified final paired analysis','',
        'The following answers use completed strong-reference checkpoints only. A positive delta favors the second method. The range is observed variation across three optimizer seeds, not a confidence interval. Device-level pairs are in paired_selection_devices.csv; reused devices are not independent extra replications.','',
        '| Comparison | Regime | Paired seeds | Mean CVaR delta | Seed range | Interpretation |',
        '|---|---|---:|---:|---|---|']
    for name,regimes in questions.items():
        for regime,v in regimes.items():lines.append(f'| {name} | {regime} | {v["paired_seeds"]} | {v["mean_delta_cvar"]} | {v["seed_delta_range"]} | {v["interpretation"]} |')
    lines+=['','Fixed-risk versus conventional answers question 1. Continuation versus fixed-risk answers question 2. The separate equal-pass/equal-wall rows and recorded cost ratios answer question 3; no statistical or population-level superiority follows.',
        '',f'Final development decision: **{summary["decision"]}**. {summary["decision_reason"]}.',
        '',f'Completed trials: {summary["completed_trials"]}/27; optimizer GPU-call wall seconds: {summary["gpu_backend_wall_seconds"]:.3f}.',
        '', 'No independent validation was generated. A confirmatory command remains withheld while the final protocol is unfrozen or development limitations remain. Existing research_benchmark.py does not implement this three-method/three-regime continuation confirmation; a command for that different study would be misleading.']
    (out/'study_report.md').write_text(original.read_text()+'\n'.join(lines)+'\n')
    save(out/'post_analysis_provenance.json',dict(script=Path(__file__).name,
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prespecified_comparisons=True,optimizer_protocol_changed=False,completed_trial_checkpoints_changed=False,
        validation_generated=False,statistical_superiority_established=False))
    print('Post-analysis completed:',summary['completed_trials'],'of 27;',summary['decision'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results/full_matched_budget');p.add_argument('--wait',action='store_true')
    args=p.parse_args();out=Path(args.output)
    with (out/'study.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX if args.wait else fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Study active. Use --wait for analysis after the worker finishes.')
        finalize(out)
