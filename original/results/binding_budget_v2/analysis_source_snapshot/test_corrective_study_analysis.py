import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from corrective_study_analysis import expected_stems,matching,nondominated,paired_rows,draft_protocol

class CorrectiveAnalysisTests(unittest.TestCase):
    def test_exact_required_grid(self):
        self.assertEqual(len(expected_stems()),54)
        self.assertEqual(len(set(expected_stems())),54)

    def test_actual_cost_not_allowance_determines_match(self):
        a=dict(actual_resource=100.,utilization=1.,overshoot_fraction=0.)
        b=dict(a,actual_resource=106.)
        self.assertFalse(matching(a,b)[0])
        self.assertTrue(matching(a,dict(a,actual_resource=104.))[0])
        self.assertFalse(matching(a,dict(a,utilization=.6))[0])
        self.assertFalse(matching(a,dict(a,overshoot_fraction=.07))[0])

    def test_pareto_dominance_and_duplicates(self):
        np.testing.assert_array_equal(nondominated([[1,2],[2,1],[2,2],[1,2]]),[True,True,False,True])

    def test_signs_and_seed_pairing(self):
        def row(method,cvar,seed=7500):
            return dict(regime='equal_objectives',level=0,seed=seed,method=method,
                cvar95_infidelity=cvar,mean_fidelity=.99,p05_fidelity=.98,nominal_fidelity=.9995,
                nominal_leakage=1e-5,mean_leakage=2e-5,worst_sample_leakage=4e-5,
                actual_resource=24,utilization=1,overshoot_fraction=0,objective_calls=24,
                scenario_passes=5000,optimizer_seconds=120,engineering_selection_eligible=True)
        p=paired_rows([row('conventional',.03),row('risk',.02),row('continuation',.04),row('risk',.001,7501)])
        self.assertEqual(len(p),3)
        self.assertAlmostEqual(p[0]['delta_cvar'],.01)
        self.assertLess(p[1]['delta_cvar'],0)
        self.assertLess(p[2]['delta_cvar'],0)
        self.assertFalse(any(x['statistical_superiority_established'] for x in p))

    def test_confirmatory_draft_does_not_draw_devices(self):
        with tempfile.TemporaryDirectory() as d,patch('challenge_model.monte_carlo_scenarios',side_effect=AssertionError('must not draw')):
            protocol=json.loads(Path('results/binding_budget_v2/protocol.json').read_text())
            draft_protocol(Path(d),protocol,{})
            m=json.loads((Path(d)/'confirmatory_protocol_manifest.json').read_text())
            self.assertFalse(m['validation_generated'])
            self.assertEqual(m['validation_devices'],2000)
            self.assertNotIn(m['validation_seed'],(7400,7401,7402))
            self.assertEqual(len(set(m['numerical_repeat_indices'])),20)

if __name__=='__main__':unittest.main()
