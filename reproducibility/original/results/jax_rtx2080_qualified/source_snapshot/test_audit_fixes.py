"""Independent regressions for the scientific audit findings."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from qutip import basis, to_super, qeye, liouvillian, expect
from scipy.linalg import expm

from challenge_model import (Device, PulseParameters, Scenario, MHZ_TO_RAD_NS,
    model_operators, channel_metrics, hamiltonian, collapse_operators,
    make_pulse, evaluate_waveform, hardware_metrics, training_scenarios)
from pulse_methods import reference_pulses, waveform_from_spec


def config():
    return json.loads(Path(__file__).with_name('challenge_config.json').read_text())


class AuditFixTests(unittest.TestCase):
    def test_all_leakage_levels_counted_and_hamiltonian_matches_duffing(self):
        for size in (3,4,5,6):
            device = Device(levels=size)
            kets, states, target, lowering, number, dx, dy, leakage = model_operators(size)
            self.assertAlmostEqual(channel_metrics(to_super(qeye(size)),size)['fidelity'],1/3)
            # Swap |1> into the HIGHEST modeled level, which a P2-only metric misses.
            swap = qeye(size)-kets[1].proj()-kets[-1].proj()+kets[1]*kets[-1].dag()+kets[-1]*kets[1].dag()
            self.assertAlmostEqual(channel_metrics(to_super(swap),size)['leakage'],.5)
            h = hamiltonian([0.,.2],np.array([.1,.1]),np.array([.03,.03]),device,Scenario(detuning_mhz=2.))(.1).full()
            n = np.arange(size)
            expected = np.diag((2*n-220*n*(n-1)/2)*MHZ_TO_RAD_NS).astype(complex)
            for j in range(1,size):
                expected[j-1,j] = np.sqrt(j)*(.1-.03j)/2
                expected[j,j-1] = expected[j-1,j].conjugate()
            np.testing.assert_allclose(h,expected,atol=1e-14)

    def test_higher_level_dissipation_preserves_trace_and_supplied_rates(self):
        device=Device(levels=6)
        kets=model_operators(6)[0]
        dissipator=liouvillian(0*qeye(6),collapse_operators(device))
        for n in range(1,6):
            derivative=dissipator(kets[n].proj())
            expected=1/device.t1_1_ns if n==1 else n/(2*device.t1_2_ns)
            self.assertAlmostEqual(expect(kets[n].proj(),derivative),-expected)
            self.assertAlmostEqual(derivative.tr(),0.)

    def test_reference_hardware_headroom_and_exact_recipes(self):
        c=config();device=Device(**c['device'])
        for key, reference in reference_pulses(c,device).items():
            pulse=np.asarray(reference['waveform'])
            self.assertLessEqual(hardware_metrics(*pulse,device)['peak_amplitude_mhz']*1.03,80.+1e-10)
            np.testing.assert_allclose(waveform_from_spec(reference['pulse_spec']),pulse,atol=1e-13)

    def test_actual_corpse_waveform_has_equal_rates_and_detuning_protection(self):
        c=config();d=Device(**c['device']);r=reference_pulses(c,d)['composite_corpse']
        pulse=np.asarray(r['waveform']);spec=r['pulse_spec']
        rates=[]
        for segment in spec['segments']:
            a,b=segment['start_sample'],segment['end_sample']
            area=np.trapezoid((pulse[1,a:b+1]+1j*pulse[2,a:b+1])*np.exp(-1j*segment['phase_rad']),pulse[0,a:b+1])
            self.assertAlmostEqual(area.real,segment['rotation_rad'],places=12)
            rates.append(segment['amplitude_mhz'])
        np.testing.assert_allclose(rates,rates[0],atol=1e-12)
        sx=np.array([[0.,1.],[1.,0.]]);sz=np.diag([-1.,1.])
        def error(offset):
            u=np.eye(2,dtype=complex)
            rate=rates[0]*MHZ_TO_RAD_NS
            # Integrate the EXPORTED linearly interpolated waveform independently.
            for j in range(len(pulse[0])-1):
                for sub in range(8):
                    fraction=(sub+.5)/8
                    i=(1-fraction)*pulse[1,j]+fraction*pulse[1,j+1]
                    h=(i*sx+offset*rate*sz)/2
                    u=expm(-1j*d.dt_ns/8*h)@u
            return 1-abs(np.trace(sx@u)/2)**2
        plain=np.sin(np.pi/2*np.sqrt(1+.01**2))**2/(1+.01**2)
        self.assertLess(abs(error(0.)),1e-11)
        self.assertLess(error(.01),(1-plain)/100)

    def test_calibration_improves_nominal_and_reconstructs_without_mc(self):
        from reference_calibration import calibrate_reference
        c=config();c['calibration']['max_evaluations']=120
        d=Device(**c['device']);raw=reference_pulses(c,d)['drag_baseline']
        before=evaluate_waveform(*np.asarray(raw['waveform']),d)
        with patch('robust_optimization.monte_carlo_scenarios',side_effect=AssertionError('test data accessed')):
            calibrated=calibrate_reference(raw,'drag_baseline',c,d)
        pulse=np.asarray(calibrated['waveform'])
        np.testing.assert_allclose(waveform_from_spec(calibrated['pulse_spec']),pulse,atol=1e-14)
        after=evaluate_waveform(*pulse,d)
        self.assertLess(after['infidelity'],.8*before['infidelity'])
        self.assertLessEqual(hardware_metrics(*pulse,d)['peak_amplitude_mhz']*1.03,80.+1e-10)

    def test_multilevel_grape_gradient_matches_finite_difference(self):
        from grape import EnsembleGRAPE,DEFAULTS
        times=np.linspace(0.,1.,6)
        controls=np.column_stack([.2*np.sin(np.pi*times),.03*np.sin(2*np.pi*times)])
        model=EnsembleGRAPE(times,Device(levels=5),[Scenario()],{**DEFAULTS,'substeps':2})
        value,gradient=model.value_gradient(controls)
        for index in [(1,0),(3,1)]:
            plus,minus=controls.copy(),controls.copy()
            plus[index]+=1e-6;minus[index]-=1e-6
            finite=(model.value_gradient(plus)[0]-model.value_gradient(minus)[0])/2e-6
            self.assertAlmostEqual(gradient[index],finite,delta=2e-8)

    def test_training_ranges_follow_configuration(self):
        scenarios=training_scenarios(.9,amplitude_fraction=.07,detuning_mhz=5.,alpha_shift_mhz=12.)
        self.assertAlmostEqual(min(s.amplitude_scale for s in scenarios),.93)
        self.assertEqual(max(s.detuning_mhz for s in scenarios),5.)
        self.assertEqual(min(s.alpha_shift_mhz for s in scenarios),-12.)

    def test_simple_searches_share_loss_bounds_and_respect_call_budget(self):
        from search_comparisons import run_comparisons
        class Quadratic:
            def __init__(self,c): self.cache={};self.simulations=0
            def scalar(self,x):
                self.assert_bounds(x)
                loss=float(np.sum((x-[.3,.5,1.,0.])**2))
                self.cache[tuple(x)]={'parameters':dict(zip(('sigma_fraction','beta','amplitude','phase_ramp_mhz'),x)), 'selection_loss':loss}
                self.simulations+=1
                return loss
            def assert_bounds(self,x):
                if np.any(x<np.array(c['bounds'])[:,0]) or np.any(x>np.array(c['bounds'])[:,1]): raise AssertionError('bounds')
        c=config();c['search_seeds']=[7];c['random_search']['max_evaluations']=40
        records,costs=run_comparisons(c,Quadratic,[.32,.55,1.,0.])
        self.assertEqual({r['method'] for r in records},{'Random search'})
        self.assertTrue(all(r['objective_calls']<=40 for r in costs))

    def test_objective_redundancy_detects_one_dimensional_front(self):
        from study_validation import objective_diagnostics
        result=objective_diagnostics([dict(training=True,objectives=[i,2*i,3*i]) for i in range(1,10)])
        self.assertGreater(result['correlation'][0][1],.999)
        self.assertLess(result['normalized_singular_values'][1],1e-12)

    def test_truncation_failure_is_detected_at_production_dimension(self):
        from study_validation import truncation_check
        def metric(*args,**kwargs):
            size=args[3].levels
            return dict(fidelity=1.-1e-3/size,leakage=1e-3/size)
        with patch('study_validation.evaluate_waveform',side_effect=metric):
            result=truncation_check(([0.,1.],[0.,0.],[0.,0.]),Device(levels=3),Scenario(),{'metric_tolerance':2e-6})
        self.assertFalse(result['passed'])
        self.assertEqual([r['levels'] for r in result['rows']],[3,4,5,6])

    def test_multilevel_chevron_conserves_total_population(self):
        from pulse_diagnostics import rabi_chevron
        device=Device(levels=5)
        pulse=make_pulse(PulseParameters(16.,.3,.5,1.,0.),device)
        populations=rabi_chevron(pulse,device,[0.,16.],[0.])
        np.testing.assert_allclose(populations.sum(axis=2),1.,atol=2e-6)
        np.testing.assert_array_equal(populations[0,0],[1.,0.,0.])
        self.assertGreaterEqual(populations[0,1,2],0.)

    def test_three_level_model_cannot_pass_truncation_gate(self):
        from study_validation import truncation_check
        metrics=dict(fidelity=.9999,leakage=1e-6)
        with patch('study_validation.evaluate_waveform',return_value=metrics):
            checked=truncation_check(([0.,1.],[0.,0.],[0.,0.]),Device(levels=3),Scenario(),{})
        self.assertFalse(checked['passed'])

    def test_configuration_import_does_not_load_tkinter(self):
        subprocess.run([sys.executable,'-c',"import sys; from optimization_inputs import build_config; assert 'tkinter' not in sys.modules"],check=True)


if __name__=='__main__':
    unittest.main()
