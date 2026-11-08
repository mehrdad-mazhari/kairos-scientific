"""Static scientific figures and a self-contained Markdown technical report."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qutip import mesolve

from .challenge_model import (KETS, MHZ_TO_RAD_NS, STATE_NAMES, TEST_STATES,
                             collapse_operators, hamiltonian, make_pulse, model_operators)

from .pulse_methods import METHOD_LABELS as LABELS, parameter_rows
from itertools import cycle
COLORS = list(plt.get_cmap('tab20').colors)


def chart_label(name):
    return (LABELS[name].replace(' (nominally calibrated)', ' (cal.)')
            .replace('Composite ', '').replace('Final selected pulse', 'Selected'))


def save_plots(output, manifest, pareto, search_history, variants, mc_rows,
               public_rows, device):
    output = Path(output)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    final = variants["robust_selected"]
    times, pulse_i, pulse_q = np.loadtxt(output / "pulse.csv", delimiter=",", skiprows=1).T
    axes[0, 0].plot(times, pulse_i / MHZ_TO_RAD_NS, label="I")
    axes[0, 0].plot(times, pulse_q / MHZ_TO_RAD_NS, label="Q")
    axes[0, 0].plot(times, np.hypot(pulse_i, pulse_q) / MHZ_TO_RAD_NS, "k--", label="Magnitude")
    axes[0, 0].axhline(device.amplitude_limit_mhz, color="red", ls=":", label="Limit")
    axes[0, 0].set(title="Final exported pulse", xlabel="Time (ns)", ylabel="Amplitude / 2 pi (MHz)")
    axes[0, 0].legend(fontsize=8)

    if pareto:
        f = np.array([r["objectives"] for r in pareto])
        sc = axes[0, 1].scatter(f[:, 1], f[:, 0], c=np.log10(np.maximum(f[:, 2], 1e-12)),
                               s=28, cmap="viridis")
        axes[0, 1].set(title="Candidate Pareto set", xlabel="Training worst gate infidelity",
                       ylabel="Training mean gate infidelity", yscale="log")
        from matplotlib.ticker import MaxNLocator
        axes[0, 1].xaxis.set_major_locator(MaxNLocator(4))
        axes[0, 1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=True)
        fig.colorbar(sc, ax=axes[0, 1], label="log10 worst training leakage")

    else:
        axes[0, 1].text(.5, .5, 'No feasible Pareto archive', ha='center', transform=axes[0, 1].transAxes)
    for seed in sorted({h["seed"] for h in search_history}):
        history = [h for h in search_history if h["seed"] == seed]
        axes[0, 2].plot([h["evaluations"] for h in history],
                        [h["best_loss"] for h in history], marker="o", ms=3, label=f"Seed {seed}")
    axes[0, 2].set(title="Nelder–Mead search diagnostics", xlabel="Objective evaluations",
                   ylabel="Best training loss")
    if search_history:
        axes[0, 2].legend(fontsize=8)
    else:
        axes[0, 2].set_title('Nelder–Mead disabled')

    positions = np.arange(len(variants))
    nominal_errors = [1-manifest["variants"][name]["nominal"]["fidelity"] for name in variants]
    nominal_leakage = [manifest["variants"][name]["nominal"]["leakage"] for name in variants]
    axes[1, 0].barh(positions-.18, np.maximum(nominal_errors, 1e-12), height=.36, label="Gate infidelity")
    axes[1, 0].barh(positions+.18, np.maximum(nominal_leakage, 1e-12), height=.36, label="Mean leakage")
    axes[1, 0].set_yticks(positions, [chart_label(n) for n in variants], fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(title="Nominal six-state metrics", xlabel="Error / population", xscale="log")
    axes[1, 0].legend(fontsize=8)

    for (name, rows), color in zip(mc_rows.items(), cycle(COLORS)):
        fidelity = np.sort([r["fidelity"] for r in rows])
        axes[1, 1].step(fidelity, np.arange(1, len(rows)+1)/len(rows), where="post",
                        label=chart_label(name), color=color)
    axes[1, 1].axvline(.990, color="black", ls=":", label="PDF nominal minimum")
    axes[1, 1].set(title="Held-out Monte Carlo", xlabel="Average X-gate fidelity",
                   ylabel="Empirical cumulative probability")
    axes[1, 1].legend(fontsize=6, loc="upper left", ncol=2, framealpha=.7)

    rows = [r for r in public_rows["robust_selected"] if r["alpha_shift_mhz"] == 0]
    grid = np.array([r["fidelity"] for r in rows]).reshape(3, 5)
    im = axes[1, 2].imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    axes[1, 2].set_xticks(range(5), [device.detuning_mhz + d for d in [-3, -1, 0, 1, 3]])
    axes[1, 2].set_yticks(range(3), [.97, 1, 1.03])
    axes[1, 2].set(title=f"Public-grid slice: alpha / 2 pi = {device.alpha_mhz:g} MHz",
                   xlabel="Residual detuning (MHz)", ylabel="Amplitude scale")
    fig.colorbar(im, ax=axes[1, 2], label="Average X-gate fidelity")
    for ax in axes.flat[:5]:
        ax.grid(alpha=.2)
    fig.suptitle(f"{device.levels}-level X gate: method comparison and manufacturing robustness", fontsize=15)
    fig.savefig(output / "optimization_dashboard.png", dpi=170)
    fig.savefig(output / "optimization_dashboard.svg")
    plt.close(fig)

    if "mc_selection" in manifest:
        rankings = json.loads((output / "mc_selection_ranking.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        positions = np.arange(1, len(rankings)+1)
        for metric, label, marker in [("mean_fidelity", "Mean fidelity", "o"),
                                       ("p05_fidelity", "5th-percentile fidelity", "s"),
                                       ("worst_sample_fidelity", "Lowest sampled fidelity", "v")]:
            values = [100*r["statistics"][metric] for r in rankings]
            axes[0].plot(positions, values, marker=marker, ms=5, label=label)
        winner = manifest["mc_selection"]["candidate"] + 1
        axes[0].axvline(winner, color="black", ls=":", label=f"Selected candidate {winner}")
        axes[0].set_xticks(positions)
        axes[0].set(title="Manufacturing-range candidate selection",
                     xlabel="Candidate (ordered by optimizer training loss)", ylabel="X-gate fidelity (%)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=.2)
        rows = mc_rows["robust_selected"]
        scatter = axes[1].scatter([device.detuning_mhz + r["detuning_mhz"] for r in rows],
                                  [100*r["fidelity"] for r in rows],
                                  c=[100*(r["amplitude_scale"]-1) for r in rows],
                                  s=12, alpha=.65, cmap="coolwarm")
        axes[1].set(title="Selected pulse: independent manufactured-device samples",
                     xlabel="Residual detuning (MHz)", ylabel="X-gate fidelity (%)")
        fig.colorbar(scatter, ax=axes[1], label="Amplitude error (%)")
        axes[1].grid(alpha=.2)
        fig.savefig(output / "manufacturing_selection.png", dpi=170)
        plt.close(fig)

    kets, states, *_ = model_operators(device.levels)
    h = hamiltonian(times, pulse_i, pulse_q, device)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, state, name in zip(axes.flat, states, STATE_NAMES):
        result = mesolve(h, state, times, c_ops=collapse_operators(device),
                         e_ops=[ket.proj() for ket in kets],
                         options={"atol": 1e-10, "rtol": 1e-8, "max_step": .1,
                                  "normalize_output": False})
        for level, population in enumerate(result.expect):
            ax.plot(times, population, label=f"P{level}")
        ax.set(title=f"Input |{name}>", xlabel="Time (ns)", ylabel="Population", ylim=(-.02, 1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Final pulse: all six input-state population trajectories", fontsize=15)
    fig.savefig(output / "six_state_dynamics.png", dpi=170)
    plt.close(fig)



def write_report(output, manifest):
    """Self-contained report for the actual scalar-search and validation workflow."""
    output = Path(output)
    config = manifest['config']
    final = manifest['variants']['robust_selected']
    status = 'PASS' if manifest['quality_passed'] else 'FAIL — exported pulse is a diagnostic candidate'
    lines = ['# Transmon X-gate comparison', '', f'**Engineering quality: {status}.**', '',
        f"Selected method: **{manifest['selected_method']}**.", '',
        'Workflow: textbook references → nominal calibration → optional multistart Nelder–Mead '
        '→ optional GRAPE → Monte Carlo selection → frozen-pulse independent validation '
        '→ Hilbert-space truncation check.', '',
        '## Fixed device and quality inputs', '',
        f"Device: `{config['device']}`. Gate duration: {config['gate_duration_ns']} ns.", '',
        f"Quality targets: `{config['quality_targets']}`.", '',
        '## Selected pulse', '']
    lines.extend(f'- **{label}:** {value}' for label,value in parameter_rows(manifest['selected_pulse_spec']))
    lines += ['', 'Full precision: [recipe](selected_pulse_parameters.json), [waveform](pulse.csv).', '',
        '## Method comparison', '',
        '| Method | Actual duration ns | Nominal F | Nominal leakage | Held-out mean F | Held-out P05 F | P05 lower bound |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in manifest['variants'].items():
        n,m = row['nominal'],row['monte_carlo']
        lines.append(f"| {LABELS.get(name,name)} | {row['hardware']['duration_ns']:.6g} | {n['fidelity']:.9f} | {n['leakage']:.5g} | {m['mean_fidelity']:.9f} | {m['p05_fidelity']:.9f} | {m['p05_fidelity_lcb95']:.9f} |")
    lines += ['', '## Selection eligibility', '', '| Method | Eligible | Reason |', '|---|---|---|']
    for name,row in manifest['variants'].items():
        selection = row.get('selection')
        reason = '; '.join(selection['rejection_reasons']) if selection else 'Duplicate waveform or not shortlisted'
        lines.append(f"| {LABELS.get(name,name)} | {selection['selection_feasible'] if selection else 'N/A'} | {reason or 'Passed selection constraints'} |")
    lines += ['', '## Nelder–Mead search', '',
        f"Settings: `{config.get('nelder_mead', {})}`. Restart seeds: `{config['search_seeds']}`.", '',
        'The bounded simplex searches sigma/Tg, DRAG beta, amplitude scale and constant I/Q phase ramp '
        'in normalized coordinates. The first restart starts from nominally calibrated DRAG when available; '
        'later restarts perturb that start reproducibly. Fixed duration and device inputs are never optimized.', '',
        'The scalar loss is the configured weighted sum of mean training infidelity, worst training '
        'infidelity and worst training leakage. Hardware and nominal quality failures receive penalties; '
        'they remain ineligible for final acceptance. Fixed training stress cases use configured manufacturing ranges.', '',
        f"Weights: `{config['selection_weights']}`.", '',
        'Budget exhaustion is reported separately from optimizer convergence. A converged local simplex '
        'does not prove global optimality. All feasible evaluated points may enter the finalist shortlist.', '',
        '| Method | Seed | Calls | Channel solves | Seconds | Converged/completed |',
        '|---|---:|---:|---:|---:|---|']
    for row in manifest.get('search_costs', []):
        lines.append(f"| {row['method']} | {row['seed']} | {row['objective_calls']} | {row['channel_simulations']} | {row['elapsed_seconds']:.3f} | {row['success']} |")
    lines += ['', '[Search history](nelder_mead_history.json), [search costs](search_costs.json), '
        '[candidate archive](optimization_archive.json), [calibration costs](reference_calibration.json).', '',
        'Calibration minimizes nominal infidelity only, at fixed reference shape/duration, fitting gain, '
        'phase ramp and applicable beta. Textbook and calibrated references are shown separately. '
        'Raw references are projected into the hardware disk including manufacturing headroom; '
        'the projection scale is explicit and can reduce rotation area below pi.', '',
        'BB1 uses DRAG-shaped segments. CORPSE uses equal-rate plateaus with one-sample edge ramps '
        'and exact sampled segment areas. Composite pulses include decoherence for their full longer '
        'duration and cannot win fixed-duration selection. Hermite DRAG is an envelope variant, not '
        'the higher-order adiabatic DRAG expansion. Wah-Wah is evaluated without a spectator-qubit model.', '',
        '## GRAPE and independent validation', '',
        f"GRAPE enabled: {manifest.get('grape', {}).get('enabled', False)}; selected: {manifest.get('grape', {}).get('selected', False)}.", '',
        'Optional Fourier GRAPE uses dissipative ensemble training, analytic Fréchet derivatives, '
        'constrained SLSQP and the same production-model acceptance checks. Its final samples define '
        'the waveform; Gaussian parameters describe its seed only. Odd sine harmonics define I, '
        'even sine harmonics define Q. Leakage and smoothness weights are explicit surrogate terms.', '',
        f"Selection uses {config['mc_selection_samples']} devices (seed {config['mc_selection_seed']}); "
        f"independent validation uses {config['monte_carlo_samples']} devices (seed {config['mc_seed']}).", '',
        'All methods share identical device draws. Selection is frozen before held-out validation; '
        'held-out performance cannot change the winner. If no candidate is eligible, an explicitly '
        'failed diagnostic candidate is exported. The public 45-case grid retains challenge-defined '
        'offsets around nominal inputs. Monte Carlo uses independent uniform quasi-static errors '
        'and independent coherence scales in the configured interval.', '',
        'P05 lower bounds use a one-sided 95% binomial order statistic; small samples can only '
        'support the physical lower bound zero. Mean intervals use a Student-t approximation. '
        'Sampled extremes do not certify all possible devices.', '',
        '## Physical model and numerical checks', '',
        '`H = Delta*n + alpha*n*(n-1)/2 + I*(a+a†)/2 - i*Q*(a-a†)/2`.', '',
        'Time is ns; MHz converts to rad/ns by 2*pi*1e-3. The ladder coupling is sqrt(n). '
        'The six Pauli eigenstates define unconditional X fidelity; leaked output is never renormalized. '
        'Leakage sums all populations n>=2. Samples use linear interpolation and zero endpoints.', '',
        'T1/Tphi Lindblad operators preserve supplied n=1,2 lifetimes. For n>2, relaxation rates '
        'scale as n/2 relative to level 2; dephasing projector rates scale as (n/2)^2. '
        'This Duffing/noise extrapolation is an assumption, not measured transmon data.', '',
        f"Strict-solver differences: `{manifest['numerical_checks']}`.", '',
        f"Selected-pulse truncation check passed: **{manifest.get('truncation_check', {}).get('passed', False)}**.", '',
        '[Truncation diagnostics](truncation_check.json) compare 3–6 levels plus production and '
        'production+1 for nominal, worst-fidelity held-out and coherent nominal propagation. '
        'Quality requires production-dimension convergence; a three-level production model cannot pass. '
        'These checks do not establish uniform convergence over the uncertainty domain.', '',
        '[Objective correlations](objective_diagnostics.json) and any nondominated candidate set '
        'are diagnostics of the scalar-search archive, not a multi-objective optimization algorithm. '
        '[Paired held-out differences](paired_comparisons.json) compare to calibrated DRAG. '
        'These exploratory mean-difference intervals have no multiple-comparison correction.', '',
        '## Figures', '', '![Comparison dashboard](optimization_dashboard.png)', '',
        '![Selection](manufacturing_selection.png)', '', '![Six-state dynamics](six_state_dynamics.png)', '',
        'Additional tables and plots: [stage comparisons](stage_comparisons.md).', '',
        '## Reproduction and limits', '',
        f"`python run_all.py --non-interactive --config {output.as_posix()}/config.json --output results/reproduction_nelder_mead`", '',
        'Software versions and source hashes are in summary.json. Passing tests verifies specific '
        'implementation properties, not scientific novelty or hardware performance. No optimizer '
        'advantage or global optimum is assumed; short runs are smoke tests.', '',
        'References: [DRAG](https://arxiv.org/abs/0901.0534), '
        '[analytic control](https://arxiv.org/abs/1011.1949), '
        '[composite rotations](https://arxiv.org/abs/quant-ph/0208092), '
        '[SciPy Nelder–Mead](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html), '
        '[QuTiP](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-propagator.html).']
    if 'statistical_certification' in manifest:
        stats=final['monte_carlo'];cert=manifest['statistical_certification'];assessment=manifest['engineering_assessment']
        lines += ['', '## Independent validation and statistical certification', '',
            '| Quantity | Value |', '|---|---:|',
            f"| Nominal fidelity | {final['nominal']['fidelity']:.10g} |",
            f"| Nominal leakage | {final['nominal']['leakage']:.10g} |",
            f"| Independent validation samples | {stats['n_samples']} |",
            f"| Mean fidelity | {stats['mean_fidelity']:.10g} |",
            f"| Mean fidelity 95% CI | {stats['mean_fidelity_ci95']} |",
            f"| Empirical P05 fidelity | {stats['p05_fidelity']:.10g} |",
            f"| One-sided 95% lower bound for true P05 | {stats['p05_fidelity_lcb95']:.10g} |",
            f"| Minimum OBSERVED fidelity | {stats['worst_sample_fidelity']:.10g} |",
            f"| Maximum OBSERVED leakage | {stats['worst_sample_leakage']:.10g} |",
            f"| Fraction meeting both configured sample thresholds | {stats['threshold_pass_fraction']:.10g} |",
            f"| Thresholds | {stats['thresholds']} |",
            f"| Confidence level | {cert['confidence_level']} |", '',
            f"Tail-bound status: **{stats['tail_bound_status']}**. Certification: **{'PASS' if cert['passed'] else 'FAIL'}**.", '',
            'The bound is the exact binomial order-statistic construction, not a normal approximation. '
            'A zero confidence lower bound is NOT an observed gate fidelity of zero. '
            'Selection-set statistics are descriptive because that dataset chose the pulse; '
            'only the independent held-out dataset certifies the frozen selection.', '',
            *[f'- {reason}' for reason in cert['reasons']], '',
            '[Certification data](statistical_certification.json), [validation device draws](mc_validation_scenarios.json), '
            '[frozen waveform hashes](frozen_waveform.json).', '',
            '![Independent certification](statistical_certification.png)', '',
            '## Final engineering assessment', '',
            f"Nominal quality constraints: **{'PASS' if assessment['nominal_quality_passed'] else 'FAIL'}**. "
            f"Independent statistical certification: **{'PASS' if cert['passed'] else 'FAIL'}**.", '',
            f"Optimizer status: `{assessment['optimizer_status']}`. Evaluation-budget exhaustion is not global convergence.", '',
            '| Check | Status |', '|---|---|',
            *[f"| {name} | {'PASS' if passed else 'FAIL'} |" for name,passed in assessment['checks'].items()], '',
            *[f"- **{failure['code']}**" for failure in assessment['failures']], '',
            f"Numerical errors: `{manifest.get('numerical_errors',[])}`.", '',
            '## Worst-case robustness diagnostics', '']
        worst=manifest['worst_case_robustness']
        if 'error' in worst:
            lines += [f"**Diagnostic numerical failure:** {worst['error']}. No worst-case result is claimed."]
        elif worst['enabled']:
            lines += [f"Minimum observed fidelity: **{worst['minimum_observed']['fidelity']:.10g}**. "
                f"Maximum observed leakage: **{worst['maximum_leakage_observed']['leakage']:.10g}**.", '',
                'These are finite axis/boundary probes and bounded local searches, NOT mathematically certified '
                'global worst-case bounds. They do not modify the frozen pulse. A P05 requirement does not '
                'require every possible device to exceed that fidelity; a separate minimum-fidelity requirement '
                'applies only when worst_case.fidelity_min is explicitly set. Observed hard-leakage violations cause FAIL.', '',
                '| Perturbation (others nominal) | Minimum observed F | Maximum observed leakage | Fidelity span |',
                '|---|---:|---:|---:|',
                *[f"| {name} | {row['min_observed_fidelity']:.9g} | {row['max_observed_leakage']:.9g} | {row['fidelity_span']:.9g} |" for name,row in worst['sensitivities'].items()], '',
                'Joint uncertainty searches and witness scenarios: [diagnostic data](worst_case_robustness.json).', '',
                f"Cost: {worst['channel_simulations']} unique channel solves, {worst['elapsed_seconds']:.3f} s.", '',
                *[f"- {row['metric']} search: success={row['success']}, calls={row['objective_calls']}, {row['message']}" for row in worst['search_status']], '',
                '![Worst-case probes](worst_case_robustness.png)']
        else:
            lines += ['Optional worst-case diagnostic search was disabled. No domain-wide worst-case certificate is claimed.']
        comparison=manifest.get('grape_vs_calibrated_drag',{}).get('paired_mean_comparison')
        lines += ['', '## GRAPE versus calibrated DRAG', '',
            f"Paired mean-fidelity difference and 95% CI: `{comparison}`.", '',
            'The methods see identical independent devices. This exploratory mean comparison has no '
            'multiplicity correction and does not establish GRAPE superiority, a tail-fidelity advantage '
            'or hardware performance.', '',
            f"Frozen waveform SHA256: `{manifest['frozen_waveform_sha256']}`."]
    if 'research_comparisons' in manifest:
        lines += ['', '## Risk-aware GRAPE benchmark', '',
            'Shared frozen training ensemble; common selection and independent validation draws. No confirmatory superiority claim. Empirical CVaR95 is a loss statistic, not the 95% certification confidence.', '',
            '[Machine-readable paired comparisons](research_comparisons.json)', '',
            *[f'![Research figure {i}](research_figure_{i}.png)' for i in range(1,8)]]
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def save_reliability_plots(output, manifest, mc_rows):
    """Keep confidence bounds visually distinct from observed fidelities."""
    if 'statistical_certification' not in manifest:
        return
    output=Path(output);stats=manifest['variants']['robust_selected']['monte_carlo']
    values=np.sort([r['fidelity'] for r in mc_rows['robust_selected']])
    fig,axes=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    axes[0].step(values,np.arange(1,len(values)+1)/len(values),where='post')
    axes[0].axvline(manifest['config']['quality_targets']['p05_validation_lcb_min'],ls=':',color='k',label='P05 target')
    axes[0].set(title='Independent observed fidelities',xlabel='Observed fidelity',ylabel='Empirical cumulative probability')
    axes[0].legend(fontsize=8)
    points=[stats['worst_sample_fidelity'],stats['p05_fidelity'],stats['p05_fidelity_lcb95']]
    axes[1].scatter(points,[2,1,0],c=['#3572a5','#3572a5','#c98239'],s=60)
    axes[1].set_yticks([2,1,0],['Observed minimum','Empirical P05','95% lower bound for P05'])
    axes[1].set(title='Observation versus certification',xlabel='Fidelity / statistical lower bound',ylim=(-.6,2.6))
    if not stats['tail_bound_supported']:
        axes[1].annotate('Insufficient samples; zero is a bound, not an observation',
            (0,0),xytext=(.05,.2),textcoords='axes fraction',fontsize=8,wrap=True)
    checks=manifest['engineering_assessment']['checks'];names=list(checks)
    axes[2].barh(range(len(names)),[1]*len(names),color=['#468b6b' if checks[n] else '#bd5a50' for n in names])
    axes[2].set_yticks(range(len(names)),[n.replace('_',' ') for n in names],fontsize=8)
    axes[2].set_xticks([]);axes[2].invert_yaxis();axes[2].set_title('Engineering checks (green PASS / red FAIL)')
    fig.savefig(output/'statistical_certification.png',dpi=160);plt.close(fig)
    worst=manifest['worst_case_robustness']
    if worst['enabled'] and 'error' not in worst:
        names=list(worst['sensitivities']);rows=[worst['sensitivities'][n] for n in names]
        fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
        for ax,key,title in zip(axes,['fidelity_span','max_observed_leakage'],['Single-axis fidelity span','Single-axis maximum observed leakage']):
            ax.barh(range(len(names)),[r[key] for r in rows]);ax.set_yticks(range(len(names)),names,fontsize=8)
            ax.set_title(title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        axes[1].axvline(manifest['config']['quality_targets']['leakage_max'],color='r',ls=':',label='Leakage limit');axes[1].legend()
        fig.suptitle('Finite frozen-pulse probes — no global worst-case certificate')
        fig.savefig(output/'worst_case_robustness.png',dpi=160);plt.close(fig)
