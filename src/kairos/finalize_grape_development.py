"""Audit the short equal-call diagnostic; full reports stay separated by regime."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
from .verify_grape_development import verify
from .grape_development import render_outputs,write_json


def finalize(output,timing=None):
    out=Path(output);protocol=json.loads((out/'protocol.json').read_text());c=protocol['config'];base=protocol['base_config']
    if c['mode']!='short_diagnostic' or c['regimes']!=['equal_objectives']:
        raise ValueError('This mean-summary helper is for the short equal-call diagnostic; use the per-regime full runner reports for mixed budgets')
    pulse=np.asarray(json.loads((out/'shared_reference.json').read_text())['waveform'])
    verification=verify(out)
    runs=render_outputs(out,c,base,pulse)
    with (out/'paired_selection_comparisons.csv').open() as stream:comparisons=list(csv.DictReader(stream))
    costs={method:np.mean([r['history'][-1]['elapsed_seconds'] for r in runs if r['method']==method]) for method in ('conventional','risk','continuation')}
    summary=[]
    for method in ('conventional','risk','continuation'):
        group=[r for r in runs if r['method']==method]
        summary.append(dict(method=method,optimizer_seeds=len(group),
            unique_final_waveforms=len({r['waveform_sha256'] for r in group}),
            initial_nominal_fidelity=group[0]['history'][-1]['accepted_steps'][0]['nominal_fidelity'],
            mean_final_nominal_fidelity=float(np.mean([r['nominal']['fidelity'] for r in group])),
            mean_selection_cvar=float(np.mean([r['selection']['statistics']['cvar95_infidelity'] for r in group])),
            mean_selection_p05=float(np.mean([r['selection']['statistics']['p05_fidelity'] for r in group])),
            mean_nominal_leakage=float(np.mean([r['nominal']['leakage'] for r in group])),
            mean_runtime_seconds=float(costs[method]),converged_trials=sum(r['history'][-1]['converged'] for r in group)))
    delta={method:float(np.mean([float(r['delta_cvar']) for r in comparisons if r['method']==method])) for method in ('risk','continuation')}
    verdict=dict(mode=c['mode'],summary=summary,mean_delta_cvar=delta,
        continuation_same_output_across_seeds=len({r['waveform_sha256'] for r in runs if r['method']=='continuation'})==1,
        no_independent_superiority_claim=True,go_for_independent_validation=False,
        reason='Full-budget/equal-pass/equal-wall comparisons and converged reference calibration remain unfinished.',
        calibration_reference_calls=json.loads((out/'shared_reference.json').read_text())['calibration']['objective_calls'])
    write_json(out/'development_verdict.json',verdict)
    lines=['\n## Executed diagnostic verdict\n',
        '| Method | Initial nominal F | Mean final nominal F | Mean selection CVaR95 | Mean nominal leakage | Mean optimizer time (s) | Converged trials |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:lines.append(f"| {row['method']} | {row['initial_nominal_fidelity']:.10f} | {row['mean_final_nominal_fidelity']:.10f} | {row['mean_selection_cvar']:.9g} | {row['mean_nominal_leakage']:.9g} | {row['mean_runtime_seconds']:.2f} | {row['converged_trials']}/{row['optimizer_seeds']} |")
    lines+=['',f"Mean paired selection delta CVaR: fixed risk {delta['risk']:.9g}; continuation {delta['continuation']:.9g}. Positive favors the risk method. These average per-trial CVaRs across optimizer seeds; they do not pool the common 200 devices into additional independent observations.",
        '',f"Continuation minus fixed-risk CVaR advantage (positive favors continuation): {delta['continuation']-delta['risk']:.9g}. This is descriptive, not a statistically established winner.",
        '', 'Both risk methods must also be assessed against mean/P05 fidelity and actual work. Read the signed paired differences above; a CVaR improvement does not imply P05 or mean-fidelity improvement. The risk and continuation adjoints use twice the surrogate passes per call. All trials remain budget-limited; a valid initial pulse does not establish optimization convergence.',
        '', 'Qualification succeeded for the admitted starts. This supports initialization safety, not a general probability of optimizer success. The unperturbed fit itself was already qualified; this experiment does not isolate filtering from the smaller perturbation scale.',
        '',f"Production reproduction error: {verification['max_production_reproduction_error']:.3g}. Maximum stress/substep metric error: {verification['maximum_stress_substep_metric_error']:.3g}; unchanged tolerance {base['quality_targets']['solver_metric_tolerance']:.3g}. These are point checks, not a uniform ensemble convergence proof.",
        '', 'NO-GO for an independent superiority claim or final validation launch yet. Complete the three-regime full-budget development experiment, strengthen nominal reference calibration, assess seed variability and freeze the final research protocol. No final validation dataset was generated.',
        '', 'Numerical source is preserved in source_snapshot/. Input-signature/constraint guards and figure-only hardening applied afterward are recorded in post_execution_hardening.patch/json. The optimizer, physical model, objective and executed effective inputs are unchanged. Current figure exports include terminal cost checkpoints as squares; these are reporting points, not extra accepted steps.']
    with (out/'intermediate_benchmark.md').open('a') as stream:stream.write('\n'.join(lines)+'\n')
    if verdict['continuation_same_output_across_seeds']:
        explanation='\nContinuation selected the same final waveform across seeds. Inspect stage input/output hashes and the differing random-start hashes: common deterministic warm-start improvements can beat every random alternative. Identical final outputs do not imply unchanged coefficients and add no independent device evidence.\n'
        with (out/'intermediate_benchmark.md').open('a') as stream:stream.write(explanation)
        with (out/'continuation_cvar_study.md').open('a') as stream:stream.write(explanation)
    # Include termination explicitly rather than inferring convergence from objective decrease.
    terminations=[]
    for run in runs:
        last=run['history'][-1]
        terminations.append(dict(method=run['method'],seed=run['seed'],regime=run['regime'],
            converged=last['converged'],termination=last['message'],
            stage_terminations=json.dumps([s.get('termination') for s in last.get('stages',[])]),
            objective_evaluations=last['objective_evaluations'],solver_iterations=last['solver_iterations']))
    with (out/'optimizer_termination.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(terminations[0]));writer.writeheader();writer.writerows(terminations)
    initializations=json.loads((out/'initialization_checkpoint.json').read_text())
    ordinary=sum(r['production_channel_solves'] for r in initializations if r['strategy']!='qualified_fourier_multistart')
    bank=sum(max(r['production_channel_solves'] for r in initializations if r['strategy']=='qualified_fourier_multistart' and r['seed']==seed) for seed in c['optimizer_seeds'])
    overhead=dict(initialization_study_production_channel_solves=ordinary+bank,
        gradient_verification_objective_gradient_evaluations=6,gradient_verification_surrogate_passes=900,
        initial_reference_probe_production_solves=1,final_export_nominal_production_solves=len(runs),
        unique_selection_production_channel_solves=len(list((out/'selection_channels').glob('*.json'))),
        verification_extra_production_channel_solves=verification['verification_extra_production_channel_solves'],
        verification_elapsed_seconds=verification['verification_elapsed_seconds'],
        note='Development overhead is separate from method-wide optimization costs; no unrecorded optimization tuning is included.')
    if timing:
        shutil.copyfile(timing,out/'execution_time.txt')
        overhead['main_campaign_timing']={line.split()[0]:float(line.split()[1]) for line in Path(timing).read_text().splitlines()}
    write_json(out/'development_overhead.json',overhead)
    write_json(out/'report_render_sources.json',dict(numerical_protocol_signature=protocol['signature'],
        renderer_sha256=hashlib.sha256(Path('grape_development_report.py').read_bytes()).hexdigest(),
        finalizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        change='Figure export bounds, terminal cost reporting points and explicit calibration-budget caveat; saved numerical trials unchanged.'))
    print(json.dumps(verdict,indent=2))
    return verdict


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--timing')
    args=parser.parse_args();finalize(args.output,args.timing)
