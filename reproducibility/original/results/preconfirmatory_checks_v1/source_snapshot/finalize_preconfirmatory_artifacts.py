"""Finish metadata and immutable manifests after the CPU convergence checks."""
import csv,hashlib,json,os,sys,platform,importlib.metadata
from pathlib import Path
from grape_development import write_csv,write_json

OUT=Path('results/preconfirmatory_checks_v1')

def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='COMPLETE','CPU study still running'
    candidates=json.loads((OUT/'frozen_candidates.json').read_text())
    reference=json.loads(Path('results/full_matched_budget/common_reference.json').read_text())
    assert hashlib.sha256(Path('results/grape_development_reference/references.json').read_bytes()).hexdigest()==reference['checkpoint_sha256']
    assert candidates['calibrated_drag']['waveform_sha256']==reference['waveform_sha256']
    for name,c in candidates.items():
        if name!='calibrated_drag':
            source=json.loads((Path('results/binding_budget_v2')/(c['source_trial']+'.json')).read_text());h=source['history'][-1]
            c['complete_computational_cost']={k:h[k] for k in ('objective_evaluations','gradient_evaluations','surrogate_scenario_passes','production_channel_solves','total_dissipative_passes','gpu_backend_wall_seconds','elapsed_seconds')}
            c['complete_computational_cost']['selection']=source['selection']['selection_elapsed_seconds']
            c['complete_computational_cost']['scope']='Successful trial; exact CPU nominal wall not separately instrumented. Shared setup and historical interrupted work remain separate.'
    write_json(OUT/'frozen_candidates.json',candidates)
    original=OUT/'candidate_eligibility_audit_original_trials.csv'
    if not original.exists():original.write_bytes((OUT/'candidate_eligibility_audit.csv').read_bytes())
    rows=list(csv.DictReader(original.open()));reps=[]
    for name,c in candidates.items():
        s=c['development_metrics'];h=c['hardware']
        reasons=[]
        if s['nominal_fidelity']<.999:reasons.append('nominal fidelity <.999')
        if s['nominal_leakage']>1e-4:reasons.append('nominal leakage >1e-4')
        if s['maximum_leakage']>1e-4:reasons.append('maximum ensemble leakage >1e-4')
        if not c['hilbert_check']['five_level_pass']:reasons.append('N5/N8 development difference > existing numerical tolerance')
        if not c['independent_solver_pass']:reasons.append('CPU numerical gate')
        reps.append(dict(candidate=name,source_trial=c['source_trial'],nominal_fidelity=s['nominal_fidelity'],nominal_leakage=s['nominal_leakage'],
            mean_ensemble_leakage=s['mean_leakage'],maximum_ensemble_leakage=s['maximum_leakage'],cvar95_infidelity=s['cvar95_infidelity'],
            p05_fidelity=s['p05_fidelity'],p05_lower_bound=s['p05_lower_bound'],peak_amplitude_mhz=h['peak_amplitude_mhz'],
            amplitude_with_headroom_mhz=h['peak_amplitude_mhz']*1.03,endpoint_amplitude=h['endpoint_amplitude_rad_ns'],
            numerical_pass=c['independent_solver_pass'],hilbert_pass=c['hilbert_check']['five_level_pass'],
            engineering_eligible=c['engineering_eligible_after_checks'],allowed_primary_comparator=c['allowed_primary_comparator'],
            exclusion_reasons='; '.join(reasons),waveform_sha256=c['waveform_sha256']))
    write_csv(OUT/'candidate_representatives.csv',reps)
    drag=next(r for r in reps if r['candidate']=='calibrated_drag')
    rows.append({k:drag.get(k,'') for k in rows[0]})
    rows[-1].update(method='calibrated_drag',original_runner_eligible='not applicable: calibrated reference',
        audited_existing_criteria_eligible=drag['engineering_eligible'],post_analysis_p05_guard_pass=drag['p05_lower_bound']>=.995)
    write_csv(OUT/'candidate_eligibility_audit.csv',rows)
    convergence=list(csv.DictReader((OUT/'hilbert_convergence.csv').open()))
    for r in convergence:
        for field in ('nominal_leakage','mean_leakage','maximum_leakage'):
            reference=float(r[field])-float(r['difference_from_N8_'+field])
            r['relative_difference_from_N8_'+field]=float(r['difference_from_N8_'+field])/reference if reference else None
    write_csv(OUT/'hilbert_convergence_with_relative_errors.csv',convergence)
    verification=json.loads((OUT/'independent_solver_verification.json').read_text())
    ready=json.loads((OUT/'preconfirmatory_readiness.json').read_text())
    manifest=json.loads((OUT/'development_preservation_manifest.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    original=json.loads(Path('results/binding_budget_v2/completed_experiment_manifest.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in original.items())
    write_json(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','qutip','matplotlib')},
        threads={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}))
    audit=OUT/'audit_source_snapshot';audit.mkdir(exist_ok=True)
    source_names=list(json.loads(Path('results/binding_budget_v2/protocol.json').read_text())['source_hashes'])
    source_names+=['pulse_shapes.py','pulse_methods.py','reference_calibration.py','corrective_study_analysis.py']
    for name in source_names:
        dest=audit/Path(name).name
        if not dest.exists():dest.write_bytes(Path(name).read_bytes())
    # Preserve the computation-stage manifest before the final metadata freeze.
    if not (OUT/'computation_stage_manifest.json').exists():(OUT/'computation_stage_manifest.json').write_bytes((OUT/'freeze_manifest.json').read_bytes())
    report=['# Final preconfirmatory scientific checks','',f'**{ready["decision"]}** for the next scientific stage; independent validation has not been generated or executed.', '',
        'The original 27- and 54-trial experiments, saved ensembles and frozen optimization/source manifests remain unchanged. All 54 corrective checkpoints and aggregations passed integrity checks. New work is confined to this versioned directory.', '',
        'Conventional was selected by the literal minimum feasible CVaR95. Continuation was not excluded: its best pulse differs by only about5.37e-7 in development CVaR, below the existing2e-6 numerical criterion. Fixed risk\'s best unfiltered pulse violates maximum ensemble leakage; it remains a diagnostic comparator. All54 candidates pass the later development P05 guard, so that additional draft guard changed no selection.', '',
        '|Representative|Nominal F|CVaR95|Maximum selection leakage|Engineering eligible|Primary allowed after checks|', '|---|---:|---:|---:|---|---|']
    for r in reps:report.append(f"|{r['candidate']}|{r['nominal_fidelity']:.10g}|{r['cvar95_infidelity']:.9g}|{r['maximum_ensemble_leakage']:.9g}|{r['engineering_eligible']}|{r['allowed_primary_comparator']}|")
    report+=['', 'The Hilbert checks use N5,6,7,8, the same exact sampled waveforms, 200 saved development selection scenarios and 10 saved training stress entries plus nominal. No new device draws or optimization runs occur. Absolute and relative errors are reported; finite sampled agreement is not uniform convergence or validation of the physical/noise assumptions.', '',
        'Observable definitions are unchanged: each device fidelity and leakage is the mean over six computational input states. Maximum ensemble leakage is the maximum of these per-device means across the saved200 devices, not the maximum over arbitrary input states. State-specific results remain in channel records. Fidelity is unconditional; leaked outputs are never renormalized.', '',
        f'Independent CPU metric/physicality verification passed: {verification["passed"] and verification["all_physicality_pass"]}. Maximum observed dt/4 versus dt/8 or propagator versus state-solve metric discrepancy: {max(r["metric_error"] for r in verification["comparisons"]):.9g}. '
        'Both propagation paths use QuTiP and may share an integration backend. This checks channel-versus-state propagation and metric construction, not independence of software implementations. The original GPU4/8 checks remain preserved as supporting evidence.', '',
        'The mathematical dissipator construction, trace-preserving generator, sqrt(n) annihilation ladder and leakage projectors agree across dimensions. All returned density matrices are checked for trace, Hermiticity, positivity and computational-plus-leakage population balance with unchanged thresholds.', '',
        'The frozen confirmatory plan specifies exactly2000 future devices with reserved seed941206, all frozen candidates on identical devices, CPU strict production solves, paired CVaR bootstrap with common index resampling and six-contrast Bonferroni correction. Existing one-sided95% distribution-free P05 certificates are preserved; an additional simultaneous engineering certificate uses the same method. Optimizer-seed variation is never counted as independent device sampling.', '',
        'Protocol/source/candidate hashes are locked in freeze_manifest.json. Candidate and protocol files are read-only to prevent accidental changes; hash checking remains authoritative. This freeze is preparation only and grants no authorization to draw validation devices. Q1 submission readiness still requires confirmation and a defensible novelty/model-scope argument; numerical checks alone do not establish it.', '',
        'Files: candidate_eligibility_audit.csv contains all54 trials plus calibrated DRAG; candidate_representatives.csv contains the five audited representatives. hilbert_convergence.csv preserves signed absolute changes; hilbert_convergence_with_relative_errors.csv adds relative leakage differences. Frozen arrays, coefficients/native parameters, exact source trials, optimization seeds, training configurations and costs are in frozen_candidates.json.']
    (OUT/'preconfirmatory_final_report.md').write_text('\n'.join(report)+'\n')
    snap=OUT/'source_snapshot'/'finalize_preconfirmatory_artifacts.py'
    if not snap.exists():snap.write_bytes(Path(__file__).read_bytes())
    final={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name not in ('execution.log','launcher.json','checks.lock','freeze_manifest.json')}
    write_json(OUT/'freeze_manifest.json',final)
    for p in [OUT/'frozen_candidates.json',OUT/'input_candidates.json',OUT/'confirmatory_protocol.json',OUT/'confirmatory_protocol_frozen.md',OUT/'check_protocol.json',OUT/'freeze_manifest.json',*list((OUT/'source_snapshot').glob('*.py')),*list(audit.glob('*.py'))]:os.chmod(p,0o444)
    print(json.dumps(ready,indent=2))

if __name__=='__main__':main()
