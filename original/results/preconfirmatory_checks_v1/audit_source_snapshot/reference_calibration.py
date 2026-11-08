"""Nominal-device calibration only: never consumes selection or held-out scenarios."""
from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from challenge_model import evaluate_waveform, hardware_metrics, MHZ_TO_RAD_NS
from pulse_methods import waveform_from_spec, METHOD_LABELS


def calibrate_reference(reference, key, config, device):
    options = config.get('calibration', {})
    headroom = max(config['amplitude_headroom_scale'],
                   1+config['manufacturing_ranges']['amplitude_fraction'])
    source = deepcopy(reference['pulse_spec'])
    tune_beta = source['representation'] in ('tapered_gaussian_drag', 'hermite_drag', 'wah_wah', 'composite_drag') and key != 'gaussian_baseline'
    original_beta = source.get('beta', 0.)
    bounds = [(.85, 1.15), (-8., 8.)] + ([(-1., 2.)] if tune_beta else [])
    initial = [1., 0.] + ([original_beta] if tune_beta else [])
    cache, best = {}, None
    simulations = 0
    started = perf_counter()

    def build(x):
        base = deepcopy(source)
        if tune_beta:
            base['beta'] = float(x[2])
            for segment in base.get('segments', []):
                segment['beta'] = float(x[2])
            base['notes'] = 'Underlying reference shape; beta fitted by the enclosing nominal calibration.'
        spec = dict(method=METHOD_LABELS[key+'_calibrated'], representation='calibrated_reference',
                    duration_ns=source['duration_ns'], dt_ns=device.dt_ns,
                    base_spec=base, gain=float(x[0]), phase_ramp_mhz=float(x[1]),
                    notes='Gain, constant phase ramp and (where applicable) DRAG beta calibrated on the nominal device only. Shape and duration held fixed; no held-out data used.')
        pulse = waveform_from_spec(spec)
        base_pulse = waveform_from_spec(base)
        base['peak_amplitude_mhz'] = hardware_metrics(*base_pulse, device)['peak_amplitude_mhz']
        for segment in base.get('segments', []):
            a, b = segment['start_sample'], segment['end_sample']
            segment['amplitude_mhz'] = float(np.hypot(*base_pulse[1:, a:b+1]).max()/MHZ_TO_RAD_NS)
        return spec, pulse

    def objective(x):
        nonlocal best, simulations
        k = tuple(x)
        if k in cache:
            return cache[k]
        spec, pulse = build(x)
        hardware = hardware_metrics(*pulse, device)
        violation = hardware['peak_amplitude_mhz']*headroom/device.amplitude_limit_mhz-1
        if violation > 1e-12:
            loss = 1+violation
        else:
            metric = evaluate_waveform(*pulse, device)
            simulations += 1
            loss = metric['infidelity']
            if best is None or loss < best[0]:
                best = loss, spec, pulse, metric
        cache[k] = loss
        return loss

    objective(initial)
    result = minimize(objective, initial, method='Nelder-Mead', bounds=bounds,
                      options={'maxfev': options.get('max_evaluations', 180),
                               'xatol': 1e-5, 'fatol': 1e-10,
                               # Explicit steps: SciPy's zero-coordinate default is too small.
                               'initial_simplex': np.array([initial]+[
                                   np.array(initial)+np.eye(len(initial))[j]*[.02, .3, .1][j]
                                   for j in range(len(initial))])})
    if best is None:
        raise RuntimeError('No feasible calibration start')
    _, spec, pulse, metric = best
    spec['peak_amplitude_mhz'] = hardware_metrics(*pulse, device)['peak_amplitude_mhz']
    spec['sample_count'] = len(pulse[0])
    parameters = dict(reference['parameters'])
    parameters['amplitude'] *= spec['gain']
    parameters['beta'] = spec['base_spec'].get('beta', parameters['beta'])
    parameters['phase_ramp_mhz'] = spec['phase_ramp_mhz']
    return dict(parameters=parameters, method=spec['method'], waveform=pulse.tolist(), pulse_spec=spec,
                calibration=dict(nominal=metric, raw_method=reference['method'],
                                 objective='nominal infidelity', channel_simulations=simulations,
                                 objective_calls=len(cache), elapsed_seconds=perf_counter()-started,
                                 optimizer_success=bool(result.success), message=str(result.message)))
