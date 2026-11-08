import hashlib,json,tempfile,unittest,importlib.metadata,copy,concurrent.futures as cf,fcntl
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
from confirmatory_execution_v1 import (FreezeError,verify_release,verify_freeze,load_or_generate_dataset,
    canonical,sha,seal_record,verify_checkpoint,execute,validate_cohort,validate_metrics,
    plan_evaluations,reserve_and_load_cohort,run_pending,result_path,evaluate,execute_locked,finish_validation,DIAGNOSTIC)
from confirmatory_analysis_v1 import secondary_intervals,engineering_result,summarize_claim,overall_reporting
from confirmatory_statistics import PRIMARY,PAIRS,paired_cvar_intervals
from risk_aware_grape import empirical_cvar
from statistical_certification import tail_bound_details
from scipy.stats import binom

def metric_fixture():
    return dict(physicality_pass=True,state_fidelities=[.999]*6,state_leakages=[1e-5]*6,
        fidelity=.999,infidelity=1-.999,leakage=1e-5,max_trace_error=0.,max_hermiticity_error=0.,
        subspace_population_balance_error=0.,min_output_eigenvalue=0.)

def cohort_fixture():
    # Copies existing development rows in a temporary schema fixture, never
    # calls the reserved sampler and never evaluates an independent device.
    rows=json.loads(Path('results/preconfirmatory_checks_v1/selection_devices.json').read_text())*10
    cfg=json.loads(Path('results/preconfirmatory_checks_v1/confirmatory_protocol.json').read_text())
    d=dict(seed=941206,n_devices=2000,scenarios=rows,dataset_sha256=hashlib.sha256(canonical(rows)).hexdigest(),
        implementation_manifest_sha256='unit-test',generator_source_sha256=sha('challenge_model.py'),
        generator='challenge_model.monte_carlo_scenarios / numpy.default_rng',numpy_version=importlib.metadata.version('numpy'),
        distribution=dict(**cfg['physical_model']['manufacturing_ranges'],coherence_min=.8),development_exact_overlaps=0)
    d['cohort_payload_sha256']=hashlib.sha256(canonical(d)).hexdigest()
    return d,cfg

class ConfirmatoryExecutionTests(unittest.TestCase):
    def test_release_is_required_before_sampling(self):
        with tempfile.TemporaryDirectory() as d,patch('confirmatory_execution_v1.verify_release',side_effect=FreezeError('missing review')),patch('confirmatory_execution_v1.load_or_generate_dataset') as generate:
            with self.assertRaises(FreezeError):execute(Path(d),{}, {},{})
            generate.assert_not_called()

    def test_unaccepted_release_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FreezeError):verify_release(Path(d))

    def test_release_binds_exact_source_and_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'fixture.py';source.write_text('# unit test source\n')
            interpretation=p/'interpretation.json';interpretation.write_text('{}')
            manifest=dict(sources={str(source):sha(source)},reviewed_artifacts={str(interpretation):sha(interpretation)},frozen_scientific_manifest_sha256=sha('results/preconfirmatory_checks_v1/freeze_manifest.json'))
            (p/'implementation_manifest.json').write_text(json.dumps(manifest))
            (p/'implementation_review.json').write_text(json.dumps(dict(status='ACCEPTED',implementation_manifest_sha256=sha(p/'implementation_manifest.json'),reviewer='unit-test-only',review_completed_before_validation=True)))
            with patch('confirmatory_execution_v1.SOURCES',(str(source),)):
                verify_release(p);source.write_text('# altered unit test source\n')
                with self.assertRaises(FreezeError):verify_release(p)

    def test_changed_freeze_is_never_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'fixture').write_text('original');h=sha(p/'fixture');(p/'freeze_manifest.json').write_text(json.dumps({'fixture':h}));(p/'fixture').write_text('tampered')
            with self.assertRaises(FreezeError):verify_freeze(p,nominal=False)
            self.assertEqual((p/'fixture').read_text(),'tampered')

    def test_interrupted_generation_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'dataset_generation_started.json').write_text('{}');generator=Mock(side_effect=AssertionError('must not draw'))
            with self.assertRaises(FreezeError):load_or_generate_dataset(p,{},'unit-test',generator)
            generator.assert_not_called()

    def test_cohort_resume_never_calls_generator(self):
        # Repeated existing development cases; not independent validation data.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cohort,cfg=cohort_fixture()
            (p/'dataset_generation_started.json').write_text(json.dumps(dict(seed=941206,count=2000,status='GENERATION_STARTED',implementation_manifest_sha256='unit-test')))
            (p/'validation_dataset.json').write_text(json.dumps(cohort));generator=Mock(side_effect=AssertionError('must not draw'))
            with patch('confirmatory_execution_v1.development_scenario_keys',return_value=set()):self.assertEqual(load_or_generate_dataset(p,cfg,'unit-test',generator),cohort)
            generator.assert_not_called()

    def test_corrupt_dataset_is_not_regenerated(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cohort,cfg=cohort_fixture();cohort['dataset_sha256']='wrong'
            (p/'dataset_generation_started.json').write_text(json.dumps(dict(seed=941206,count=2000,status='GENERATION_STARTED',implementation_manifest_sha256='unit-test')))
            (p/'validation_dataset.json').write_text(json.dumps(cohort))
            generator=Mock(side_effect=AssertionError('must not draw'))
            with self.assertRaises(FreezeError):load_or_generate_dataset(p,cfg,'unit-test',generator)
            generator.assert_not_called()

    def test_checkpoint_corruption_is_detected(self):
        scenario={'unit_test_fixture':True};dataset=dict(dataset_sha256='cohort',scenarios=[scenario]);candidate=dict(waveform_sha256='pulse')
        record=seal_record(dict(candidate='test',index=0,mode='four',candidate_sha256='pulse',dataset_sha256='cohort',scenario_sha256=hashlib.sha256(canonical(scenario)).hexdigest(),status='COMPLETE',metrics={**metric_fixture(),'unit_test_metric':1}))
        verify_checkpoint(record,'test',0,candidate,dataset)
        record['metrics']['unit_test_metric']=2
        with self.assertRaises(FreezeError):verify_checkpoint(record,'test',0,candidate,dataset)

    def test_solver_failure_checkpoint_is_preserved_as_blocker(self):
        scenario={'unit_test_fixture':True};dataset=dict(dataset_sha256='cohort',scenarios=[scenario]);candidate=dict(waveform_sha256='pulse')
        record=seal_record(dict(candidate='test',index=0,mode='four',candidate_sha256='pulse',dataset_sha256='cohort',scenario_sha256=hashlib.sha256(canonical(scenario)).hexdigest(),status='SOLVER_FAILURE',error='unit-test exception'))
        with self.assertRaises(FreezeError):verify_checkpoint(record,'test',0,candidate,dataset)

    def test_tail_uses100_devices(self):
        losses=np.linspace(0,.1,2000)
        self.assertAlmostEqual(empirical_cvar(losses),losses[-100:].mean(),places=14)

    def test_secondary_common_indices_preserve_paired_shifts(self):
        base=np.linspace(.97,.99,20);f={m:base+i*.001 for i,m in enumerate(PRIMARY)};l={m:np.full(20,1e-5+i*1e-6) for i,m in enumerate(PRIMARY)}
        result=secondary_intervals(f,l,repetitions=100)
        for key,ci in result['paired'].items():
            a,c=key.split('/');shift=(f[c]-f[a]).mean()
            self.assertAlmostEqual(ci[0][0],shift,places=13);self.assertAlmostEqual(ci[0][1],shift,places=13)

    def test_diagnostic_never_promoted_by_good_validation(self):
        c=dict(hardware=dict(peak_amplitude_mhz=70.,endpoint_amplitude_rad_ns=0.,duration_ns=15.6),hilbert_check=dict(five_level_pass=True),allowed_primary_comparator=False)
        r=engineering_result('risk_unfiltered_diagnostic',c,dict(lower_bound=.997,maximum_leakage=1e-5),dict(fidelity=.9995,leakage=1e-5),None)
        self.assertEqual(r['status'],'DIAGNOSTIC_ONLY');self.assertFalse(r['passed'])
        c['allowed_primary_comparator']=True
        self.assertFalse(engineering_result('risk_unfiltered_diagnostic',c,dict(lower_bound=.997,maximum_leakage=1e-5),dict(fidelity=.9995,leakage=1e-5),dict(lower_bound=.997))['passed'])

    def test_resolution_flag_is_separate_from_statistical_direction(self):
        row=dict(reference='a',candidate='b',direction='candidate improvement',delta_cvar=5e-7,family_adjusted_ci=[1e-7,1e-6]);before=copy.deepcopy(row)
        r=summarize_claim(row,row,{'a':dict(passed=True),'b':dict(passed=True)},{'a':1e-11,'b':1e-11})
        self.assertEqual(row,before);self.assertTrue(r['numerical_resolution_review_required']);self.assertFalse(r['superiority_claim_supported'])
        self.assertEqual(r['statistical_direction'],'candidate improvement')
        self.assertEqual(r['numerical_reliability'],'NUMERICALLY_UNRESOLVED')

    def test_point_below_resolution_even_if_biased_ci_above_it(self):
        r=dict(reference='a',candidate='b',direction='candidate improvement',delta_cvar=1e-6,family_adjusted_ci=[3e-6,5e-6])
        result=summarize_claim(r,r,{'a':dict(passed=True),'b':dict(passed=True)},{})
        self.assertEqual(result['numerical_reliability'],'NUMERICALLY_UNRESOLVED');self.assertFalse(result['superiority_claim_supported'])

    def test_bootstrap_sensitivity_not_numerical_or_ci_rewrite(self):
        r=dict(reference='a',candidate='b',direction='candidate improvement',delta_cvar=.001,family_adjusted_ci=[.0001,.002])
        s={**r,'direction':'inconclusive'};result=summarize_claim(r,s,{'a':dict(passed=True),'b':dict(passed=True)},{})
        self.assertEqual(result['statistical_direction'],'candidate improvement');self.assertEqual(result['bootstrap_sensitivity_status'],'UNSTABLE')
        self.assertEqual(result['numerical_reliability'],'CHECKED_NUMERICALLY_RESOLVED');self.assertFalse(result['superiority_claim_supported'])

    def test_engineering_failure_does_not_rewrite_statistical_result(self):
        r=dict(reference='a',candidate='b',direction='candidate improvement',delta_cvar=.001,family_adjusted_ci=[.0001,.002])
        result=summarize_claim(r,r,{'a':dict(passed=True),'b':dict(passed=False)},{})
        self.assertEqual(result['statistical_direction'],'candidate improvement');self.assertTrue(result['superiority_claim_supported'])
        self.assertFalse(result['engineering_candidate_pass'])

    def test_metric_structure_and_scalar_consistency(self):
        for change in (dict(state_fidelities=[.999]*5),dict(fidelity=.998),dict(max_trace_error=1e-3),dict(min_output_eigenvalue=-1e-3),dict(state_leakages=[float('nan')]*6)):
            with self.assertRaises(FreezeError):validate_metrics({**metric_fixture(),**change})

    def test_cohort_metadata_versions_and_development_overlap(self):
        d,cfg=cohort_fixture()
        with self.assertRaises(FreezeError):validate_cohort(d,cfg,'unit-test') # reused dev rows are deliberately not independent
        with patch('confirmatory_execution_v1.development_scenario_keys',return_value=set()):
            validate_cohort(d,cfg,'unit-test')
            for change in (dict(seed=941207),dict(numpy_version='wrong'),dict(generator_source_sha256='wrong'),dict(n_devices=1999),dict(distribution={})):
                with self.assertRaises(FreezeError):validate_cohort({**d,**change},cfg,'unit-test')

    def test_primary_bonferroni_and_paired_bootstrap_against_reference(self):
        f={m:np.linspace(.97,.995,12)+i*.0001 for i,m in enumerate(PRIMARY)};b=100;seed=941207
        actual=paired_cvar_intervals(f,repetitions=b,seed=seed);rng=np.random.default_rng(seed);rep=[]
        for _ in range(b):
            indices=rng.integers(12,size=12);v={m:empirical_cvar(1-a[indices]) for m,a in f.items()};rep.append([v[a]-v[c] for a,c in PAIRS])
        for i,r in enumerate(actual['comparisons']):
            np.testing.assert_allclose(r['family_adjusted_ci'],np.quantile(np.array(rep)[:,i],[.05/12,1-.05/12]),atol=0,rtol=0)
            self.assertAlmostEqual(r['adjusted_confidence'],1-.05/6)

    def test_p05_and_simultaneous_order_indices(self):
        # Mathematical vector, not generated devices.
        f=np.linspace(.99,1,2000)
        for confidence in (.95,1-.05/4):
            r=tail_bound_details(f,confidence=confidence);k=int(binom.ppf(1-confidence,2000,.05))
            self.assertEqual(r['order_statistic_index'],k);self.assertEqual(r['lower_bound'],f[k-1])

    def test_overall_convention_keeps_engineering_separate(self):
        eng={m:dict(simultaneous_engineering_pass=True) for m in PRIMARY}
        rows=[dict(reference=a,candidate=c,superiority_claim_supported=True,numerical_resolution_review_required=False) for a,c in PAIRS]
        self.assertEqual(overall_reporting(rows,eng)['classification'],'CONFIRMED')
        rows[0]['superiority_claim_supported']=False;self.assertEqual(overall_reporting(rows,eng)['classification'],'PARTIALLY CONFIRMED')
        for r in rows:r['superiority_claim_supported']=False
        self.assertEqual(overall_reporting(rows,eng)['classification'],'NOT CONFIRMED')
        self.assertTrue(overall_reporting(rows,eng)['publication_readiness']['independent_of_favorable_outcomes'])
        with self.assertRaises(FreezeError):overall_reporting([rows[0]]*6,eng)

    def test_new_generation_helper_requires_review(self):
        with tempfile.TemporaryDirectory() as d,patch('confirmatory_execution_v1.verify_release',side_effect=FreezeError('missing')):
            generator=Mock(side_effect=AssertionError('forbidden draw'))
            with self.assertRaises(FreezeError):load_or_generate_dataset(Path(d),{},'unit-test',generator)
            generator.assert_not_called();self.assertFalse((Path(d)/'dataset_generation_started.json').exists())

    def test_completed_evaluations_are_not_resubmitted(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'channels'/'a').mkdir(parents=True);scenarios=[{'fixture':0},{'fixture':1}]
            cohort=dict(scenarios=scenarios,dataset_sha256='cohort');candidate=dict(waveform_sha256='pulse')
            record=seal_record(dict(candidate='a',index=0,mode='four',candidate_sha256='pulse',dataset_sha256='cohort',scenario_sha256=hashlib.sha256(canonical(scenarios[0])).hexdigest(),status='COMPLETE',metrics=metric_fixture()))
            result_path(out,'a',0).write_text(json.dumps(record))
            pending,complete=plan_evaluations(out,{},dict(a=candidate),cohort)
            self.assertEqual(complete,1);self.assertEqual(len(pending),1);self.assertEqual(pending[0][1],1)

    def test_one_global_cohort_and_no_redraw_when_committed_file_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);package=root/'package';package.mkdir();a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            with patch('confirmatory_execution_v1.load_or_generate_dataset',return_value=dict(cohort_payload_sha256='fixture-only')) as load:
                reserve_and_load_cohort(a,{},'unit-test',package);self.assertEqual(load.call_count,1)
                for output in (a,b):
                    with self.assertRaises(FreezeError):reserve_and_load_cohort(output,{},'unit-test',package)
                self.assertEqual(load.call_count,1)

    def test_failed_run_drains_inflight_results_without_filtering(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'channels'/'a').mkdir(parents=True);tasks=[];futures=[]
            for i in (0,1):
                scenario={'fixture':i};tasks.append(('a',i,scenario,dict(waveform_sha256='pulse'),{},'cohort','four'))
                record=dict(candidate='a',index=i,mode='four',candidate_sha256='pulse',dataset_sha256='cohort',scenario_sha256=hashlib.sha256(canonical(scenario)).hexdigest(),status='SOLVER_FAILURE' if i==0 else 'COMPLETE',metrics=metric_fixture())
                f=cf.Future();f.set_running_or_notify_cancel();f.set_result(seal_record(record));futures.append(f)
            pool=Mock();pool.submit.side_effect=futures;seen=[]
            with patch('confirmatory_execution_v1.cf.as_completed',side_effect=lambda values:iter(values)):
                with self.assertRaises(FreezeError):run_pending(pool,tasks,out,lambda r,failed:seen.append((r['index'],failed)))
            self.assertEqual(seen,[(0,True),(1,True)])
            self.assertEqual(json.loads(result_path(out,'a',1).read_text())['status'],'COMPLETE')

    def test_execution_lock_blocks_duplicate_process(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            with (out/'execution.lock').open('a') as lock,patch('confirmatory_execution_v1.verify_release') as release:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):execute(out,{}, {},{})
                release.assert_not_called()

    def test_adverse_engineering_result_is_not_filtered_as_solver_failure(self):
        m={**metric_fixture(),'fidelity':.998,'infidelity':.002,'leakage':.001,'state_fidelities':[.998]*6,'state_leakages':[.001]*6}
        with patch('confirmatory_execution_v1.solve',return_value=m):
            r=evaluate(('a',0,{'unit_test_fixture':True},dict(waveform=[],waveform_sha256='pulse'),dict(physical_model={}), 'cohort','four'))
        self.assertEqual(r['status'],'COMPLETE');self.assertGreater(r['metrics']['leakage'],1e-4)

    def test_completed_grid_control_flow_never_resubmits_or_draws(self):
        # Mocked state-machine test: no cohort creation or production solver.
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cfg=dict(production_solver={});candidates={m:{} for m in (*PRIMARY,DIAGNOSTIC)}
            audit=dict(freeze_manifest_sha256='unit-fixture');cohort=dict(dataset_sha256='unit-fixture')
            with patch('confirmatory_execution_v1.verify_release',return_value='unit-release'),patch('confirmatory_execution_v1.verify_freeze',return_value=(cfg,candidates,audit)),patch('confirmatory_execution_v1.reserve_and_load_cohort',return_value=cohort),patch('confirmatory_execution_v1.plan_evaluations',return_value=([],10000)),patch('confirmatory_execution_v1.cf.ProcessPoolExecutor') as pool,patch('confirmatory_execution_v1.finish_validation') as finish,patch('confirmatory_execution_v1.monte_carlo_scenarios',side_effect=AssertionError('forbidden draw')):
                execute_locked(out,cfg,candidates,audit)
                pool.return_value.__enter__.return_value.submit.assert_not_called()
                finish.assert_called_once()
                self.assertEqual(finish.call_args.args[3],cohort)

    def test_numerical_repeat_and_analysis_transition_with_math_checkpoints(self):
        # Cache fixtures use repeated existing development rows, not new draws.
        cohort,_=cohort_fixture();candidates={m:dict(waveform_sha256='unit-fixture-'+m) for m in (*PRIMARY,DIAGNOSTIC)}
        indices=np.linspace(0,1999,20,dtype=int).tolist()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for name,c in candidates.items():
                (out/'channels'/name).mkdir(parents=True)
                for i in indices:
                    for mode in ('four','eight','mesolve_eight'):
                        record=seal_record(dict(candidate=name,index=i,mode=mode,candidate_sha256=c['waveform_sha256'],dataset_sha256=cohort['dataset_sha256'],scenario_sha256=hashlib.sha256(canonical(cohort['scenarios'][i])).hexdigest(),status='COMPLETE',metrics=metric_fixture()))
                        result_path(out,name,i,mode).write_text(json.dumps(record))
            statuses=[]
            with patch('confirmatory_execution_v1.evaluate',side_effect=AssertionError('cached records must not be recomputed')),patch('confirmatory_execution_v1.verify_freeze'),patch('confirmatory_execution_v1.verify_release'),patch('confirmatory_analysis_v1.analyze') as analysis:
                finish_validation(out,{},candidates,cohort,{},statuses.append)
                analysis.assert_called_once()
            self.assertEqual(statuses,['NUMERICAL_REPEATS','STATISTICAL_ANALYSIS','COMPLETE'])
            self.assertEqual(len(json.loads((out/'numerical_repeat_checks.json').read_text())),200)

if __name__=='__main__':unittest.main()
