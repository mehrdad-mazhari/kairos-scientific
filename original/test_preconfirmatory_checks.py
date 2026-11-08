import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from risk_aware_grape import empirical_cvar
from confirmatory_statistics import paired_cvar_intervals,freeze_protocol,PRIMARY
from preconfirmatory_checks import dissipator_checks

class ScientificCheckTests(unittest.TestCase):
    def test_2000_device_cvar_tail_mass(self):
        losses=np.linspace(0,.1,2000)
        self.assertAlmostEqual(empirical_cvar(losses),np.sort(losses)[-100:].mean(),places=14)
        self.assertAlmostEqual(empirical_cvar([0,1,2],alpha=.5),(2+.5)/1.5)

    def test_paired_bootstrap_constant_offset(self):
        base=np.linspace(.96,.99,30)
        data={m:base+i*.001 for i,m in enumerate(PRIMARY)}
        r=paired_cvar_intervals(data,repetitions=100)
        self.assertEqual(len(r['comparisons']),6)
        for p in r['comparisons']:
            self.assertAlmostEqual(p['family_adjusted_ci'][0],p['delta_cvar'],places=13)
            self.assertAlmostEqual(p['family_adjusted_ci'][1],p['delta_cvar'],places=13)
            self.assertEqual(p['family_size'],6)

    def test_dimension_extensions_trace_preserving(self):
        base=json.loads(Path('challenge_config.json').read_text())
        r=dissipator_checks(base)
        self.assertEqual(set(r),{5,6,7,8})
        self.assertTrue(all(x['passed'] for x in r.values()))
        self.assertEqual(r[8]['operator_count'],14)

    def test_preparation_never_draws_validation(self):
        with tempfile.TemporaryDirectory() as d,patch('challenge_model.monte_carlo_scenarios',side_effect=AssertionError('forbidden')):
            p=dict(base_config=json.loads(Path('challenge_config.json').read_text()),source_hashes={})
            freeze_protocol(Path(d),p,{},'NO-GO')
            r=json.loads((Path(d)/'confirmatory_protocol.json').read_text())
            self.assertFalse(r['validation_generated']);self.assertEqual(r['validation_count'],2000)
            self.assertEqual(len(r['primary_pairs']),6)

if __name__=='__main__':unittest.main()
