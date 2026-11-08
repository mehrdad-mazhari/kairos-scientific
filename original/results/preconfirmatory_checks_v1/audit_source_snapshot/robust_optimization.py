"""Compare baselines -> optional Nelder–Mead -> optional GRAPE -> final Monte Carlo.

Usage: python robust_optimization.py --config challenge_config.json
Outputs include exact pulse samples, a full evaluation archive, configuration,
nominal/public/Monte Carlo metrics, convergence/Pareto plots and a short report.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from time import perf_counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from qutip.solver.integrator.integrator import IntegratorException

from pulse_shapes import PAPER_DRAG_BETA
from pulse_coefficients import pulse_coefficients
from pulse_methods import reference_pulses, analytic_spec, fourier_spec, parameter_rows
from nelder_mead import minimize as nm_minimize
from study_validation import nondominated_fronts
from pipeline_settings import resolve_config, validate_reliability_settings
from risk_aware_grape import empirical_cvar
from statistical_certification import tail_bound_details, certify, engineering_assessment
from challenge_model import (Device, PulseParameters, Scenario, evaluate_waveform,
                             hardware_metrics, make_pulse, monte_carlo_scenarios,
                             public_scenarios, training_scenarios)

PARAMETER_NAMES = ("sigma_fraction", "beta", "amplitude", "phase_ramp_mhz")
OBJECTIVE_NAMES = ("training_mean_infidelity", "training_worst_infidelity", "training_worst_leakage")


class TrainingEvaluator:
    """Cache exact quantized parameters; reuse fixed scenarios for every candidate."""

    def __init__(self, config):
        config = resolve_config(config)
        self.device = Device(**config["device"])
        self.duration_ns = config["gate_duration_ns"]
        self.scenarios = training_scenarios(config["training"]["coherence_min"], **config["training"]["manufacturing_ranges"])
        risk=config['risk_aware_grape']
        if risk['enabled']:
            self.scenarios+=monte_carlo_scenarios(risk['training_samples'],risk['training_seed'],config['training']['coherence_min'],**config['training']['manufacturing_ranges'])
        self.headroom = max(config["amplitude_headroom_scale"],
                            1+config["manufacturing_ranges"]["amplitude_fraction"])
        self.weights = config["selection_weights"]
        self.quality = config["quality_targets"]
        self.cache = {}
        self.simulations = 0
        self.objective_calls = 0

    def evaluate(self, x):
        self.objective_calls += 1
        x = np.asarray(x, dtype=float)
        if x.shape != (4,) or not np.all(np.isfinite(x)):
            raise ValueError("optimize four pulse controls; Tg is a fixed input")
        p = PulseParameters(self.duration_ns, **dict(zip(PARAMETER_NAMES, x)))
        key = tuple(p.as_array())
        if key in self.cache:
            return self.cache[key]
        pulse = make_pulse(p, self.device)
        record = self.evaluate_samples(pulse, asdict(p))
        self.cache[key] = record
        return record

    def evaluate_samples(self, pulse, parameters):
        """Same objective and constraints for arbitrary reference I/Q samples."""
        hardware = hardware_metrics(*pulse, self.device)
        cv = max(0.0, self.headroom * hardware["peak_amplitude_mhz"]
                 / self.device.amplitude_limit_mhz - 1,
                 hardware.get("endpoint_amplitude_rad_ns", 0.)/1e-10-1)
        record = {"parameters": parameters, "hardware": hardware, "violation": 1 + cv if cv > 0 else 0.0}
        if cv > 0:
            record.update(objectives=[1.0, 1.0, 1.0], selection_loss=10 + cv,
                          nominal=None, training=None)
        else:
            nominal = evaluate_waveform(*pulse, self.device, self.scenarios[0])
            self.simulations += 1
            nominal_cv = (max(0., (self.quality["nominal_fidelity_min"] - nominal["fidelity"])
                              / (1 - self.quality["nominal_fidelity_min"]))
                          + max(0., nominal["leakage"] / self.quality["leakage_max"] - 1))
            if nominal_cv > 0:
                # Keep hardware-feasible candidates ahead of all hardware failures.
                # Raw nominal violations can greatly exceed tiny amplitude violations.
                record.update(violation=nominal_cv/(1+nominal_cv), nominal=nominal, training=None,
                              objectives=[nominal["infidelity"], nominal["infidelity"],
                                          nominal["leakage"]],
                              selection_loss=10 + nominal_cv)
                return record
            metrics = [nominal] + [evaluate_waveform(*pulse, self.device, s) for s in self.scenarios[1:]]
            self.simulations += len(metrics) - 1
            errors = np.array([m["infidelity"] for m in metrics])
            leaks = np.array([m["leakage"] for m in metrics])
            summary = {"mean_infidelity": float(errors.mean()),
                       "worst_infidelity": float(errors.max()),
                       "worst_leakage": float(leaks.max())}
            record.update(objectives=[summary["mean_infidelity"], summary["worst_infidelity"],
                                      summary["worst_leakage"]],
                          selection_loss=sum(self.weights[k] * v for k, v in summary.items()),
                          nominal=metrics[0], training=summary)
        return record

    def objective(self, x):
        record = self.evaluate(x)
        return record["objectives"], record["violation"]

    def scalar(self, x):
        return self.evaluate(x)["selection_loss"]


def passes_nominal(record, quality=None):
    quality = quality or {"nominal_fidelity_min": .999, "leakage_max": .0001}
    return (record["violation"] == 0 and record["nominal"] is not None
            and record["nominal"]["fidelity"] >= quality["nominal_fidelity_min"]
            and record["nominal"]["leakage"] <= quality["leakage_max"])


def lower_tail_confidence_bound(fidelities, quantile=.05, confidence=.95):
    """Distribution-free one-sided lower bound from a binomial order statistic.

    For continuous iid samples, P(X_(k) <= true q-quantile) >= confidence.
    A sample too small for a nontrivial bound returns the physical bound zero.
    """
    return tail_bound_details(fidelities, quantile, confidence)['lower_bound']


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_trials(rows, quality=None):
    """Empirical tails; mean CI is a Student-t approximation, yield CI is Wilson."""
    if not rows:
        raise ValueError("at least one trial is required")
    quality = quality or {"nominal_fidelity_min": .990, "leakage_max": .020}
    fidelity = np.array([r["fidelity"] for r in rows])
    leakage = np.array([r["leakage"] for r in rows])
    n = len(rows)
    se = float(fidelity.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    half = float(student_t.ppf(0.975, n - 1) * se) if n > 1 else 0.0
    bound = tail_bound_details(fidelity)
    successes = (fidelity >= quality["nominal_fidelity_min"]) & (leakage <= quality["leakage_max"])
    rate = float(successes.mean())
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (rate + z*z/(2*n)) / denominator
    radius = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / denominator
    return {"n_samples": n, "mean_fidelity": float(fidelity.mean()),
            "std_fidelity": float(fidelity.std(ddof=1)) if n > 1 else 0.0,
            "mean_fidelity_ci95": [max(0., float(fidelity.mean()-half)),
                                    min(1., float(fidelity.mean()+half))],
            "p05_fidelity": float(np.quantile(fidelity, .05)),
            "p05_fidelity_lcb95": bound['lower_bound'],
            "tail_bound_supported": bound['supported'], "tail_bound_status": bound['status'],
            "tail_bound_order_statistic_index": bound['order_statistic_index'],
            "confidence_level": .95, "tail_quantile": .05,
            "thresholds": {"fidelity_min": quality['nominal_fidelity_min'], "leakage_max": quality['leakage_max']},
            "worst_sample_fidelity": float(fidelity.min()),
            "mean_infidelity": float(1-fidelity.mean()), "cvar95_infidelity": empirical_cvar(1-fidelity),
            "mean_leakage": float(leakage.mean()),
            "p95_leakage": float(np.quantile(leakage, .95)),
            "worst_sample_leakage": float(leakage.max()),
            "threshold_pass_fraction": rate,
            "threshold_pass_fraction_wilson_ci95": [max(0., center-radius), min(1., center+radius)]}


def evaluate_set(pulse, scenarios, device, label):
    rows = []
    for i, scenario in enumerate(scenarios):
        m = evaluate_waveform(*pulse, device, scenario)
        rows.append({"sample": i, **asdict(scenario),
                     "actual_detuning_mhz": device.detuning_mhz + scenario.detuning_mhz,
                     "actual_alpha_mhz": device.alpha_mhz + scenario.alpha_shift_mhz,
                     **{k: m[k] for k in ["fidelity", "infidelity", "leakage",
                                          "max_trace_error", "min_output_eigenvalue"]}})
        if (i + 1) % 2000 == 0:
            print(f"  {label}: {i+1}/{len(scenarios)} samples", flush=True)
    return rows


def validate_config(config):
    validate_reliability_settings(config)
    from pulse_methods import settings as method_settings
    method_settings(config)
    bounds = np.asarray(config["bounds"], dtype=float)
    if (bounds.shape != (4, 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 1] <= bounds[:, 0])
            or bounds[0, 0] <= 0):
        raise ValueError("provide four increasing bounds: sigma_fraction, beta, amplitude, phase_ramp_mhz; Tg is fixed")
    device = Device(**config["device"])
    if (not np.all(np.isfinite(list(asdict(device).values()))) or device.alpha_mhz >= 0
            or min(device.t1_1_ns, device.t1_2_ns, device.tphi_1_ns,
                   device.tphi_2_ns, device.amplitude_limit_mhz) <= 0
            or not np.isclose(device.dt_ns, .2, atol=1e-12, rtol=0)):
        raise ValueError("require negative alpha, positive physical parameters and PDF dt=0.2 ns")
    from grape import settings
    g = settings(config)
    if (g['iterations'] < 1 or g['samples'] < 2 or g['substeps'] < 1
            or not 0 < g['tail_fraction'] <= 1
            or not np.all(np.isfinite([g['leakage_weight'], g['smoothness_weight'], g['restart_scale'], g['ftol']]))
            or min(g['leakage_weight'], g['smoothness_weight']) < 0
            or g['harmonics'] < 1 or g['restarts'] < 1 or g['restart_scale'] < 0
            or g['amplitude_grid_points'] < 2 or g['detuning_grid_points'] < 2
            or not 0 <= g['tail_weight'] <= 1 or g['ftol'] <= 0
            or g['seed'] in (config['mc_seed'], config['mc_selection_seed'])):
        raise ValueError('invalid GRAPE settings; training seed must differ from MC selection/validation')
    duration = config["gate_duration_ns"]
    if (not np.isfinite(duration) or not 10 <= duration <= 60
            or not np.isclose(round(duration/device.dt_ns)*device.dt_ns, duration,
                              atol=1e-10, rtol=0)):
        raise ValueError("fixed Tg must be 10-60 ns and an exact multiple of 0.2 ns")
    if (config["monte_carlo_samples"] < 2
            or not 0 < config["coherence_min"] <= 1
            or config["amplitude_headroom_scale"] < 1):
        raise ValueError("invalid optimization budget or robustness configuration")
    expected = {"mean_infidelity", "worst_infidelity", "worst_leakage"}
    weights = config["selection_weights"]
    if (set(weights) != expected or not all(np.isfinite(w) and w >= 0 for w in weights.values())
            or weights["mean_infidelity"] <= 0):
        raise ValueError("selection_weights must be finite nonnegative with positive mean error weight")
    if (config["mc_selection_samples"] < 2 or config["mc_selection_candidates"] < 1
            or config["mc_selection_seed"] == config["mc_seed"]
            or config["mc_selection_criterion"] not in
            {"p05_fidelity", "mean_fidelity", "worst_sample_fidelity", "mean_infidelity", "cvar95_infidelity"}):
        raise ValueError("invalid MC selection settings; selection and validation seeds must differ")
    for section, field in (('calibration', 'max_evaluations'), ('nelder_mead', 'max_evaluations'), ('random_search', 'max_evaluations')):
        value = config.get(section, {}).get(field, 180)
        if isinstance(value, bool) or not isinstance(value, int) or value < 5:
            raise ValueError(f'{section}.{field} must be an integer >= 5')
    tolerance = config.get('truncation', {}).get('metric_tolerance', 2e-6)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 2e-5:
        raise ValueError('truncation.metric_tolerance must be in (0, 2e-5]')
    from nelder_mead import settings as nm_settings
    nm = nm_settings(config)
    if not all(np.isfinite(nm[k]) and nm[k] > 0 for k in ('xatol', 'fatol')):
        raise ValueError('Nelder–Mead tolerances must be finite and positive')
    if (not config['search_seeds'] or len(set(config['search_seeds'])) != len(config['search_seeds'])
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in config['search_seeds'])):
        raise ValueError('require distinct nonnegative integer search seeds')
    q = config["quality_targets"]
    if (not .990 <= q["nominal_fidelity_min"] < 1 or not 0 < q["leakage_max"] <= .020
            or not 0 < q["p05_validation_lcb_min"] < 1
            or not 0 < q["solver_metric_tolerance"] <= 2e-5):
        raise ValueError("quality targets must be at least as strict as the PDF minimums")
    ranges = config['manufacturing_ranges']
    if (not all(np.isfinite(v) and v >= 0 for v in ranges.values())
            or ranges['amplitude_fraction'] >= 1):
        raise ValueError('invalid manufacturing ranges')


def candidate_pulse(record, device):
    """Preserve sampled GRAPE controls; analytic parameters only describe its seed."""
    if 'waveform' in record:
        return np.asarray(record['waveform'], dtype=float)
    return np.asarray(make_pulse(PulseParameters(**record['parameters']), device))


def select_robust_candidate(candidates, config, device, output, *, diagnostic_fallback=False):
    """Compare fixed pulses on common perturbed devices, not the luckiest device.

    Shortlist by optimizer training loss, then rank by the requested Monte Carlo
    fidelity statistic. Maximum sampled leakage and amplitude remain constraints.
    All candidates see identical draws. Independent validation cannot change this choice.
    """
    shortlist, seen = [], set()
    for candidate in sorted(candidates, key=lambda r: r["selection_loss"]):
        key = candidate_pulse(candidate, device).tobytes()
        if key not in seen:
            shortlist.append(candidate)
            seen.add(key)
        if len(shortlist) == config["mc_selection_candidates"]:
            break
    scenarios = monte_carlo_scenarios(config["mc_selection_samples"], config["mc_selection_seed"],
                                      config["coherence_min"], **config["manufacturing_ranges"])
    criterion = config["mc_selection_criterion"]
    write_json(output / "mc_selection_scenarios.json", [asdict(s) for s in scenarios])
    rankings = []
    for index, candidate in enumerate(shortlist):
        pulse = candidate_pulse(candidate, device)
        rows = evaluate_set(pulse, scenarios, device, f"candidate {index+1}")
        stats = summarize_trials(rows, config["quality_targets"])
        actual_hardware = hardware_metrics(*pulse,device)
        max_amplitude = actual_hardware["peak_amplitude_mhz"] * (
            max(config["amplitude_headroom_scale"], 1 + config["manufacturing_ranges"]["amplitude_fraction"]))
        reasons = []
        if not actual_hardware.get("feasible",True):
            reasons.append("waveform violates hardware magnitude or endpoint constraints")
        if not np.isclose(pulse[0,-1], config["gate_duration_ns"], rtol=0, atol=1e-10):
            reasons.append("comparison only: actual duration exceeds requested gate duration")
        nominal = candidate.get('nominal')
        if nominal is None:
            reasons.append('nominal qualification skipped because training hardware headroom failed')
        elif nominal['fidelity'] < config['quality_targets']['nominal_fidelity_min']:
            reasons.append('nominal fidelity below target')
        if nominal is not None and nominal['leakage'] > config['quality_targets']['leakage_max']:
            reasons.append('nominal leakage above limit')
        if stats['worst_sample_leakage'] > config['quality_targets']['leakage_max']:
            reasons.append('selection MC leakage above limit')
        if max_amplitude > device.amplitude_limit_mhz + 1e-10:
            reasons.append('amplitude exceeds hardware limit with manufacturing headroom')
        if candidate['violation'] != 0 and not reasons:
            reasons.append('training hardware/quality constraints failed')
        feasible = not reasons
        rankings.append({"candidate": index, "method": candidate.get("method", "Nelder–Mead"), "parameters": candidate["parameters"],
                         "training_loss": candidate["selection_loss"], "statistics": stats,
                         "waveform_sha256": hashlib.sha256(pulse.tobytes()).hexdigest(),
                         "rejection_reasons": reasons,
                         "selection_feasible": feasible, "selection_metric": criterion, "score": -stats[criterion] if criterion in ("mean_infidelity","cvar95_infidelity") else stats[criterion]})
        write_csv(output / f"candidate_{index:02d}_mc_selection.csv", rows)
        print(f"  candidate {index+1}/{len(shortlist)}: {criterion}="
              f"{stats[criterion]:.8f}, eligible={feasible}", flush=True)
    write_json(output / "mc_selection_ranking.json", rankings)
    eligible = [r for r in rankings if r["selection_feasible"]]
    if not eligible and not diagnostic_fallback:
        raise OptimizationFailure("No shortlisted pulse passes the nominal, sampled-leakage and amplitude "
                           "constraints. Selection diagnostics saved; no final pulse exported.")
    duration_matched = [r for r in rankings if np.isclose(candidate_pulse(shortlist[r["candidate"]], device)[0,-1], config["gate_duration_ns"], rtol=0, atol=1e-10)]
    if not eligible and not duration_matched:
        raise OptimizationFailure("No pulse matches requested gate duration")
    winner = max(eligible or duration_matched, key=lambda r: (r["score"], -r["training_loss"]))
    selected = dict(shortlist[winner["candidate"]])
    selected["mc_selection"] = winner
    return selected


def print_final_summary(manifest, output=None):
    """Print all final numerical results and PDF-required report content."""
    from terminal_report import print_results
    print_results(manifest, output)


class OptimizationFailure(RuntimeError):
    """A completed search did not find an acceptable design, not a software crash."""


def report_search_failure(candidates, config, output):
    measured = [r for r in candidates if r["nominal"] is not None]
    closest = min(measured or candidates, key=lambda r: r["violation"])
    diagnostic = {"status": "NO_ACCEPTABLE_PULSE", "config": config,
                  "candidates_tested": len(candidates), "nominal_simulations": len(measured),
                  "closest_candidate": closest,
                  "highest_fidelity_candidate": max(measured, key=lambda r: r["nominal"]["fidelity"]) if measured else None,
                  "lowest_leakage_candidate": min(measured, key=lambda r: r["nominal"]["leakage"]) if measured else None}
    write_json(Path(output) / "failure_summary.json", diagnostic)
    print("\nNO ACCEPTABLE PULSE FOUND (search result, not a dependency error)", flush=True)
    print(f"  Candidates tested: {len(candidates)}; nominally simulated: {len(measured)}")
    print(f"  Fixed device inputs: {config['device']}; Tg={config['gate_duration_ns']} ns")
    print(f"  Closest tested candidate parameters: {closest['parameters']}")
    print(f"  Peak amplitude including headroom: {closest['hardware']['peak_amplitude_mhz']*config['amplitude_headroom_scale']:.8f} MHz "
          f"(limit {config['device']['amplitude_limit_mhz']} MHz)")
    if closest["nominal"]:
        m = closest["nominal"]
        q = config['quality_targets']
        print(f"  Closest candidate fidelity: {100*m['fidelity']:.6f}% (required >= {100*q['nominal_fidelity_min']:.6f}%)")
        print(f"  Closest candidate leakage: {100*m['leakage']:.6f}% (required <= {100*q['leakage_max']:.6f}%)")
        print(f"  Highest tested fidelity: {100*diagnostic['highest_fidelity_candidate']['nominal']['fidelity']:.6f}%")
        print(f"  Lowest tested leakage: {100*diagnostic['lowest_leakage_candidate']['nominal']['leakage']:.6f}% (may be a different candidate)")
    print("  Your inputs and quality targets were preserved. No accepted final pulse was exported.")
    print("  This finite search does not prove impossibility. Review the search bounds/budget,")
    print("  or choose a different Tg/device input if your design allows it.")
    print(f"  Diagnostics: {(Path(output)/'failure_summary.json').resolve()}", flush=True)


def run(config, output):
    run_started = perf_counter()
    config = resolve_config(config)
    from grape import settings
    config['grape'] = settings(config)
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new --output")
    print('Effective budgets:',json.dumps({k:config[k] for k in ('mc_selection_samples','monte_carlo_samples','search_seeds','nelder_mead','risk_aware_grape','worst_case')}),flush=True)
    if config['monte_carlo_samples']<59:
        print('WARNING: INSUFFICIENT_EVIDENCE: fewer than 59 held-out devices cannot support a nontrivial 95% P05 lower bound. Zero is a bound, not observed fidelity.',flush=True)
    write_json(output / "config.json", config)
    evaluator = TrainingEvaluator(config)
    device = evaluator.device
    # A large independent campaign cannot treat CPU fallback as CUDA validation.
    if config.get('run_mode')!='quick_smoke_not_full_validation' and (config['monte_carlo_samples']>=2000 or config.get('run_mode')=='full_research'):
        from jax_backend import gpu_preflight
        for label,options in [('conventional',config['grape']),('risk',config['risk_aware_grape'])]:
            if options.get('enabled') and options.get('acceleration',{}).get('backend')=='jax_gpu':
                options['require_verified_gpu']=True
                probe=make_pulse(PulseParameters(config['gate_duration_ns'],.3,.5,1.,.3),device)
                write_json(output/f'gpu_preflight_{label}.json',gpu_preflight(probe,device,evaluator.scenarios,{**options,'quality_targets':config['quality_targets']}))
    print("Fixed physical inputs (not optimized):", flush=True)
    print(f"  detuning = {device.detuning_mhz:g} MHz; alpha = {device.alpha_mhz:g} MHz")
    print(f"  T1(1) = {device.t1_1_ns:g} ns; T1(2) = {device.t1_2_ns:g} ns")
    print(f"  Tphi(1) = {device.tphi_1_ns:g} ns; Tphi(2) = {device.tphi_2_ns:g} ns")
    print(f"  Tg = {config['gate_duration_ns']:g} ns", flush=True)
    bounds = np.asarray(config["bounds"])
    write_json(output / "training_scenarios.json", [asdict(s) for s in evaluator.scenarios])

    # References share device inputs and sample spacing; composites take longer.
    references = reference_pulses(config, device)
    if config.get('calibration', {}).get('enabled', True):
        from reference_calibration import calibrate_reference
        for key, reference in list(references.items()):
            print(f"  Nominally calibrating {reference['method']}", flush=True)
            references[key+'_calibrated'] = calibrate_reference(reference, key, config, device)
    write_json(output/'reference_calibration.json',
               {key: r['calibration'] for key,r in references.items() if 'calibration' in r})
    baseline_records = []
    for key, reference in references.items():
        print(f"  Building and evaluating {reference['method']} reference", flush=True)
        record = evaluator.evaluate_samples(candidate_pulse(reference, device), reference['parameters'])
        reference.update(record)
        baseline_records.append(reference)
    from nelder_mead import settings as nm_settings
    nm = nm_settings(config)
    all_history, search_costs, records = [], [], []
    primitive = [r for r in baseline_records if np.isclose(r['parameters']['duration_ns'], config['gate_duration_ns'])]
    initial_record = references.get('drag_baseline_calibrated', references['drag_baseline'])
    initial = np.clip([initial_record['parameters'][k] for k in PARAMETER_NAMES], bounds[:,0], bounds[:,1])
    print(f"Stage 1/4: baseline comparison; Nelder–Mead {'enabled' if nm['enabled'] else 'disabled'}", flush=True)
    if nm['enabled']:
        for restart, search_seed in enumerate(config['search_seeds']):
            started = perf_counter()
            before_simulations, before_candidates = evaluator.simulations, len(evaluator.cache)
            def progress(row):
                if row['evaluations'] == 1 or row['evaluations'] % 50 == 0:
                    print(f"  Nelder–Mead seed {search_seed}: {row['evaluations']} evaluations; best loss={row['best_loss']:.9g}", flush=True)
            result = nm_minimize(evaluator.scalar, bounds, initial, seed=search_seed,
                max_evaluations=nm['max_evaluations'], xatol=nm['xatol'], fatol=nm['fatol'],
                randomize_start=restart > 0, callback=progress)
            search_costs.append(dict(method='Nelder–Mead', seed=search_seed,
                budget_evaluations=nm['max_evaluations'], objective_calls=int(result.nfev),
                unique_candidates=len(evaluator.cache)-before_candidates,
                channel_simulations=evaluator.simulations-before_simulations,
                elapsed_seconds=perf_counter()-started, success=bool(result.success), message=str(result.message),
                cache_scope='Nelder–Mead restarts share the exact-candidate cache'))
            all_history.extend(result.history)
            write_json(output/f'search_seed_{search_seed}.json', result.history)
        for record in evaluator.cache.values():
            spec = analytic_spec(PulseParameters(**record['parameters']), device, 'Nelder–Mead')
            records.append(dict(record, method='Nelder–Mead', pulse_spec=spec))
    write_json(output/'nelder_mead_history.json', all_history)
    if config.get('random_search', {}).get('enabled', False):
        from search_comparisons import run_comparisons
        comparisons, costs = run_comparisons(config, TrainingEvaluator, initial)
        for record in comparisons:
            record['pulse_spec'] = analytic_spec(PulseParameters(**record['parameters']), device, record['method'])
        records.extend(comparisons)
        search_costs.extend(costs)
    write_json(output/'search_costs.json', search_costs)
    baseline_waveforms = {candidate_pulse(r, device).tobytes() for r in baseline_records}
    records = baseline_records + [r for r in records if candidate_pulse(r, device).tobytes() not in baseline_waveforms]
    write_json(output / 'optimization_archive.json', records)
    if nm['enabled']:
        write_json(output/'nelder_mead_archive.json', [r for r in records if r['method'] == 'Nelder–Mead'])
    fixed_duration = [r for r in records if np.isclose(r['parameters']['duration_ns'], config['gate_duration_ns'])]
    feasible = [r for r in fixed_duration if passes_nominal(r, config['quality_targets'])]
    pareto = ([feasible[i] for i in nondominated_fronts(np.array([r['objectives'] for r in feasible]))[0]]
              if feasible else [])
    write_json(output / 'pareto.json', pareto)
    from study_validation import objective_diagnostics
    redundancy = objective_diagnostics(feasible)
    write_json(output/'objective_diagnostics.json', redundancy)
    seed_record = min(feasible or fixed_duration, key=lambda r: r['selection_loss'])
    seed_pulse = candidate_pulse(seed_record, device)
    # Limit numerical finalists while always comparing every reference family.
    candidates = sorted(records, key=lambda r: r['selection_loss'])[:config['mc_selection_candidates']]
    # Preserve each simpler method finalist, even if the combined top-k is
    # dominated by many near-identical points from another search.
    comparison_best = {}
    for method, key in [('Nelder–Mead', 'nelder_mead'), ('Random search', 'random_search')]:
        pool = [r for r in records if r['method'] == method]
        if pool:
            comparison_best[key] = min(pool, key=lambda r:r['selection_loss'])
    for record in baseline_records + list(comparison_best.values()):
        if record not in candidates:
            candidates.append(record)
    from grape import refine
    g = config['grape']
    grape_result = {'enabled': g['enabled'], 'accepted': False, 'seed_method': seed_record['method']}
    risk=config['risk_aware_grape']
    risk_result={'enabled':risk['enabled']}
    risk_candidate=None
    if g['enabled'] or risk['enabled']:
        np.savetxt(output/'pre_grape_pulse.csv',np.asarray(seed_pulse).T,delimiter=',',header='time_ns,omega_i_rad_ns,omega_q_rad_ns',comments='')
        ranges=config['manufacturing_ranges'];stress=config['training']['manufacturing_ranges']
        if risk['enabled']:
            training=list(evaluator.scenarios)
        else:
            training=[Scenario(amplitude_scale=float(a),detuning_mhz=float(d))
                for a in np.linspace(1-stress['amplitude_fraction'],1+stress['amplitude_fraction'],g['amplitude_grid_points'])
                for d in np.linspace(-stress['detuning_mhz'],stress['detuning_mhz'],g['detuning_grid_points'])]
        if not risk['enabled']:training+=evaluator.scenarios
        write_json(output/'grape_training_scenarios.json',[asdict(s) for s in training])
        from risk_aware_grape import refine as risk_refine
        for label,enabled,options,optimizer in [('GRAPE',g['enabled'],g,refine),('Risk-aware GRAPE',risk['enabled'],risk,risk_refine)]:
            if not enabled:continue
            proposals=[];runs=[]
            seeds=risk['optimizer_seeds'] if risk['enabled'] else [g['seed']]
            for seed in seeds:
                resolved={**options,'seed':seed}
                proposal,history=optimizer(seed_pulse,device,training,resolved,max(config['amplitude_headroom_scale'],1+ranges['amplitude_fraction']))
                proposal=np.asarray(proposal)
                record=evaluator.evaluate_samples(proposal,seed_record['parameters'])
                spec=fourier_spec(proposal,history,seed_record['pulse_spec'],device);spec['method']=label
                candidate=dict(record,method=label,waveform=proposal.tolist(),pulse_spec=spec)
                candidates.append(candidate);proposals.append(candidate)
                runs.append(dict(seed=seed,history=history,production_training_loss=record['selection_loss'],nominal=record['nominal']))
            representative=min(proposals,key=lambda r:r['selection_loss'])
            result=dict(enabled=True,runs=runs,history=runs[proposals.index(representative)]['history'],
                        seed_method=seed_record['method'],accepted=False,selected=False)
            if label=='GRAPE':
                grape_candidate=representative;grape_result=result
                write_json(output/'grape_history.json',result['history'])
                np.savetxt(output/'grape_proposal.csv',np.asarray(representative['waveform']).T,delimiter=',',header='time_ns,omega_i_rad_ns,omega_q_rad_ns',comments='')
            else:risk_candidate=representative;risk_result=result
        write_json(output/'grape_benchmark_runs.json',dict(conventional=grape_result,risk_aware=risk_result))
    print('Stage 3/4: final Monte Carlo selection of completed method candidates', flush=True)
    selection_config = {**config, 'mc_selection_candidates': len(candidates)}
    best = select_robust_candidate(candidates, selection_config, device, output, diagnostic_fallback=True)
    chosen = PulseParameters(**best['parameters'])
    selected_spec = {**best['pulse_spec'], 'waveform_file': 'pulse.csv',
                     'peak_amplitude_mhz': best['hardware']['peak_amplitude_mhz']}
    write_json(output / 'selected_pulse_parameters.json', selected_spec)
    final_pulse = np.array(candidate_pulse(best, device), dtype=float, copy=True)
    final_pulse.setflags(write=False)
    grape_result['accepted'] = best['method'] == 'GRAPE' and best['mc_selection']['selection_feasible']
    grape_result['selected'] = best['method'] == 'GRAPE'
    risk_result['selected']=best['method']=='Risk-aware GRAPE'
    write_json(output / 'grape_summary.json', grape_result)
    # Freeze selection and export before generating/looking at held-out MC data.
    write_json(output / "selected_pulse.json", {**best, "risk_aware_grape": risk_result, "grape": grape_result, "final_waveform_file": "pulse.csv"})
    np.savetxt(output / "pulse.csv", np.column_stack(final_pulse), delimiter=",",
               header="time_ns,omega_i_rad_ns,omega_q_rad_ns", comments="", fmt="%.17g")
    reloaded = np.loadtxt(output / "pulse.csv", delimiter=",", skiprows=1).T
    np.testing.assert_allclose(reloaded, np.array(final_pulse), rtol=1e-14, atol=1e-14)

    frozen_hash = hashlib.sha256(final_pulse.tobytes()).hexdigest()
    write_json(output/'frozen_waveform.json', dict(waveform_sha256=frozen_hash,
        file_sha256=hashlib.sha256((output/'pulse.csv').read_bytes()).hexdigest(),
        selection_seed=config['mc_selection_seed'], validation_seed=config['mc_seed']))
    print("Stage 4/4: frozen-pulse public grid and independent Monte Carlo validation", flush=True)
    mc_scenarios = monte_carlo_scenarios(config["monte_carlo_samples"], config["mc_seed"],
                                        config["coherence_min"], **config["manufacturing_ranges"])
    write_json(output/'mc_validation_scenarios.json', [asdict(s) for s in mc_scenarios])
    public = public_scenarios()
    stage_records = {**references, **comparison_best}
    if risk_candidate is not None:stage_records['risk_aware_grape']=risk_candidate
    if g['enabled']:
        stage_records['grape_proposal'] = grape_candidate
    stage_records['robust_selected'] = best
    variants = {name: PulseParameters(**record['parameters']) for name, record in stage_records.items()}
    stage_pulses = {name: candidate_pulse(record, device) for name, record in stage_records.items()}
    rankings = json.loads((output/'mc_selection_ranking.json').read_text())
    validation_cache = {}
    summaries, mc_rows, public_rows = {}, {}, {}
    for name, parameters in variants.items():
        print(f"  Validating {name} on common public and held-out MC devices", flush=True)
        pulse = stage_pulses[name]
        np.savetxt(output / f"{name}_pulse.csv", np.column_stack(pulse), delimiter=",",
                   header="time_ns,omega_i_rad_ns,omega_q_rad_ns", comments="", fmt="%.17g")
        waveform_key = np.array(pulse).tobytes()
        if waveform_key not in validation_cache:
            validation_cache[waveform_key] = (evaluate_waveform(*pulse, device),
                evaluate_set(pulse, public, device, name + ' public'),
                evaluate_set(pulse, mc_scenarios, device, name + ' MC'))
        nominal, public_rows[name], mc_rows[name] = validation_cache[waveform_key]
        write_csv(output / f"{name}_public.csv", public_rows[name])
        write_csv(output / f"{name}_monte_carlo.csv", mc_rows[name])
        public_f = [r["fidelity"] for r in public_rows[name]]
        public_l = [r["leakage"] for r in public_rows[name]]
        matches = [r for r in rankings if r.get('waveform_sha256') == hashlib.sha256(np.asarray(pulse).tobytes()).hexdigest()]
        selection = matches[0] if matches else None
        summaries[name] = {"parameters": asdict(parameters), "nominal": nominal,
                           "pulse_spec": stage_records[name]['pulse_spec'],
                           "selection": selection,
                           "hardware": hardware_metrics(*pulse, device),
                           "monte_carlo": summarize_trials(mc_rows[name], config["quality_targets"]),
                           "public": {"n_samples": len(public),
                                      "mean_fidelity": float(np.mean(public_f)),
                                      "worst_fidelity": float(np.min(public_f)),
                                      "worst_leakage": float(np.max(public_l))}}


    # Certification consumes held-out results only AFTER selection is frozen.
    certification = certify(summaries['robust_selected']['monte_carlo'], config['quality_targets'])
    write_json(output/'statistical_certification.json', certification)
    print('Post-validation: statistical certification, then frozen-pulse robustness diagnostics', flush=True)
    from robustness_diagnostics import diagnose
    diagnostic_started=perf_counter()
    diagnostic_calls=0
    def diagnostic_evaluate(*args,**kwargs):
        nonlocal diagnostic_calls
        diagnostic_calls+=1
        return evaluate_waveform(*args,**kwargs)
    try:
        worst_case=diagnose(reloaded,device,config,evaluate=diagnostic_evaluate)
    except (RuntimeError,IntegratorException) as error:
        worst_case=dict(enabled=True,error=str(error),violations=[],channel_simulations=diagnostic_calls,
                        elapsed_seconds=perf_counter()-diagnostic_started,
                        mathematically_certified_global_bound=False)
    write_json(output/'worst_case_robustness.json', worst_case)

    # Independent tighter solves on the exported waveform; no resampling.
    selected_rows=mc_rows['robust_selected']
    worst_index=int(np.argmin([r['fidelity'] for r in selected_rows]))
    leakage_index=int(np.argmax([r['leakage'] for r in selected_rows]))
    numerical_errors=[worst_case["error"]] if "error" in worst_case else []
    numerical_checks={}
    strict_nominal=summaries['robust_selected']['nominal']
    try:
        strict_nominal=evaluate_waveform(*reloaded,device,strict=True)
        cases=[('nominal',Scenario(),summaries['robust_selected']['nominal']),
               ('worst_mc',mc_scenarios[worst_index],selected_rows[worst_index]),
               ('max_leakage_mc',mc_scenarios[leakage_index],selected_rows[leakage_index])]
        if worst_case['enabled'] and 'error' not in worst_case:
            for label,key in [('diagnostic_minimum','minimum_observed'),('diagnostic_leakage','maximum_leakage_observed')]:
                row=worst_case[key]
                cases.append((label,Scenario(**row['scenario']),row))
        for label,scenario,reference in cases:
            strict=strict_nominal if label=='nominal' else evaluate_waveform(*reloaded,device,scenario,strict=True)
            for metric in ('fidelity','leakage'):
                numerical_checks[f'{label}_{metric}_difference']=abs(strict[metric]-reference[metric])
    except (RuntimeError, IntegratorException) as error:
        numerical_errors.append(str(error))
    numerical_passed=bool(not numerical_errors and numerical_checks and max(numerical_checks.values())<=config['quality_targets']['solver_metric_tolerance'])
    from study_validation import truncation_check, paired_comparisons
    print('  Checking selected-pulse Hilbert-space convergence (3–6 levels and production+1)', flush=True)
    try:
        truncation_options={**config.get('truncation',{})}
        if worst_case['enabled'] and 'error' not in worst_case:
            truncation_options['diagnostic_scenarios']=[worst_case[key]['scenario'] for key in ('minimum_observed','maximum_leakage_observed')]
        truncation=truncation_check(reloaded,device,mc_scenarios[worst_index],truncation_options)
    except (RuntimeError, IntegratorException) as error:
        truncation=dict(passed=False,error=str(error))
    write_json(output/'truncation_check.json',truncation)
    paired=paired_comparisons(mc_rows)
    write_json(output/'paired_comparisons.json',paired)
    assessment=engineering_assessment(selection_feasible=best['mc_selection']['selection_feasible'],
        nominal=strict_nominal,hardware=hardware_metrics(*reloaded,device),duration_ns=float(reloaded[0,-1]),
        config=config,certification=certification,public=summaries['robust_selected']['public'],
        numerical_passed=numerical_passed,truncation=truncation,worst_case=worst_case,search_costs=search_costs)
    accepted=assessment['nominal_acceptance_passed']
    quality_passed=assessment['passed']
    if hashlib.sha256(reloaded.tobytes()).hexdigest()!=frozen_hash or hashlib.sha256((output/'pulse.csv').read_bytes()).hexdigest()!=json.loads((output/'frozen_waveform.json').read_text())['file_sha256']:
        raise RuntimeError('Frozen waveform changed during validation/diagnostics')
    manifest = {"statistical_certification": certification, "engineering_assessment": assessment,
                "worst_case_robustness": worst_case, "numerical_convergence_passed": numerical_passed,
                "numerical_errors": numerical_errors, "frozen_waveform_sha256": frozen_hash,
                "dataset_provenance": {"training": "deterministic stress cases", "selection_seed": config['mc_selection_seed'],
                    "validation_seed": config['mc_seed'], "risk_training_seed": config["risk_aware_grape"]["training_seed"] if risk["enabled"] else None, "grape_training_sha256": hashlib.sha256((output/"grape_training_scenarios.json").read_bytes()).hexdigest() if (output/"grape_training_scenarios.json").exists() else None, "distribution": "independent uniforms; unchanged",
                    "selection_scenarios_sha256": hashlib.sha256((output/'mc_selection_scenarios.json').read_bytes()).hexdigest(),
                    "validation_scenarios_sha256": hashlib.sha256((output/'mc_validation_scenarios.json').read_bytes()).hexdigest(),
                    "training_scenarios_sha256": hashlib.sha256((output/'training_scenarios.json').read_bytes()).hexdigest()},
                "computational_cost": {"elapsed_seconds_before_plotting": perf_counter()-run_started,
                    "selection_channel_solves": len(rankings)*config['mc_selection_samples'],
                    "validation_and_public_channel_solves": len(validation_cache)*(1+len(public)+len(mc_scenarios)),
                    "diagnostic_channel_solves": worst_case['channel_simulations']},
                "truncation_check": truncation, "objective_diagnostics": redundancy,
                "paired_comparisons": paired, "search_costs": search_costs,
                "grape_vs_calibrated_drag": {"paired_mean_comparison": paired.get("grape_proposal"), "superiority_claim": False, "scope": "Exploratory paired mean CI; not proof of superiority or a tail comparison."},
                "scientific_claim": "Finite simulation comparison only; no assumed optimizer advantage or experimental validation.",
                "config": config, "variants": summaries, "numerical_checks": numerical_checks,
                "nominal_acceptance_passed": accepted, "quality_passed": quality_passed,
                "optimizer": "Calibrated references and enabled numerical searches",
                "selected_method": best["method"],
                "selected_pulse_spec": selected_spec,
                "workflow": ["physical_configuration", "reference_generation", "nominal_calibration", "optional_nelder_mead", "optional_random_search", "optional_grape", "mc_selection", "freeze_waveform", "mc_validation", "statistical_certification", "worst_case_diagnostics", "numerical_convergence", "hilbert_space_truncation", "engineering_assessment"],
                "noise_model": "T1/Tphi Lindblad (challenge PDF page 5)",
                "risk_aware_grape": risk_result, "grape": grape_result,
                "pulse_coefficients": pulse_coefficients(asdict(chosen), device.dt_ns),
                "pulse_coefficients_scope": "Gaussian reference only; selected_pulse_spec defines the actual final pulse",
                "drag_reference": {"source": "Motzoi et al., arXiv:0901.0534; Gambetta et al., arXiv:1011.1949 (Y-only correction)",
                                   "lambda": float(np.sqrt(2)), "baseline_beta": PAPER_DRAG_BETA},
                "training_channel_simulations": evaluator.simulations + sum(c['channel_simulations'] for c in search_costs if c['method'] != 'Nelder–Mead'),
                "calibration_channel_simulations": sum(r.get('calibration', {}).get('channel_simulations', 0) for r in references.values()),
                "simulation_count_scope": "Production-channel training solves only; calibration separately counted. Excludes GRAPE matrix-exponential surrogate, selection, validation and truncation solves.",
                "unique_optimization_candidates": len(evaluator.cache) + sum(c['unique_candidates'] for c in search_costs if c['method'] != 'Nelder–Mead'),
                "candidate_count_scope": "Sum of method-local unique candidates; cross-method duplicates may be counted more than once. Excludes reference calibration.",
                "selection_loss": best["selection_loss"],
                "mc_selection": best["mc_selection"],
                "versions": {n: importlib.metadata.version(n) for n in
                             ["numpy", "scipy", "matplotlib", "qutip", "scqubits"]},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")}}
    write_json(output / "summary.json", manifest)
    parameter_text = '# Selected pulse parameters\n\n' + '\n'.join(
        f'- **{label}:** {value}' for label, value in parameter_rows(selected_spec))
    (output / 'selected_pulse_parameters.md').write_text(parameter_text+'\n')
    from research_figures import save_research_outputs
    save_research_outputs(output,manifest,mc_rows,public_rows)
    from challenge_report import save_plots, write_report, save_reliability_plots
    save_plots(output, manifest, pareto, all_history, variants, mc_rows, public_rows, device)
    save_reliability_plots(output, manifest, mc_rows)
    write_report(output, manifest)
    from stage_report import save_stage_comparisons
    save_stage_comparisons(output, manifest, stage_pulses, public_rows, all_history)
    from pulse_diagnostics import save_pulse_diagnostics
    if config.get("diagnostics", {}).get("rabi_chevron", True):
        save_pulse_diagnostics(output, manifest, stage_pulses)
    manifest["computational_cost"]["total_elapsed_seconds"] = perf_counter()-run_started
    write_json(output/"summary.json",manifest)
    print_final_summary(manifest, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("challenge_config.json"))
    parser.add_argument("--output", type=Path, default=Path("results/nelder_mead"))
    parser.add_argument("--quick", action="store_true", help="small end-to-end smoke run, not convergence evidence")
    from optimization_inputs import add_input_arguments, apply_input_arguments, prompt_input_arguments
    add_input_arguments(parser)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        prompt_input_arguments(config, args)
    except EOFError:
        parser.error("Input ended before completion; supply seven answers or use --non-interactive.")
    except KeyboardInterrupt:
        raise SystemExit(130)
    apply_input_arguments(config, args)
    if args.quick:
        from pipeline_settings import apply_quick_mode
        apply_quick_mode(config)
    try:
        manifest = run(config, args.output)
    except IntegratorException as error:
        diagnostic = {"status": "SOLVER_FAILURE", "config": config, "error": str(error)}
        if args.output.is_dir():
            write_json(args.output / "solver_failure.json", diagnostic)
        print(f"\nNumerical integration failed: {error}", flush=True)
        print(f"Input detuning: {config['device']['detuning_mhz']:g} MHz; "
              f"anharmonicity: {config['device']['alpha_mhz']:g} MHz; "
              f"Tg: {config['gate_duration_ns']:g} ns.")
        print("Check frequency units first: 300000 Hz = 0.3 MHz, not 300000 MHz.")
        print("If units are intentional, the solver settings/model require investigation; no valid final result was produced.")
        raise SystemExit(3)
    except OptimizationFailure as error:
        print(f"\nOptimization stopped: {error}", flush=True)
        raise SystemExit(2)
    if not manifest["quality_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
