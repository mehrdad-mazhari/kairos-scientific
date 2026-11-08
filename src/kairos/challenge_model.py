"""PQO-2026-02 model, following the detailed equations on PDF pages 4-6.

All times are ns, frequencies in the public API are MHz, and the Hamiltonian
uses rad/ns with hbar=1. Delta has the PDF's PLUS sign (unlike legacy scripts).
"""

from dataclasses import dataclass, replace
from itertools import product
from functools import lru_cache

import numpy as np
from qutip import QobjEvo, basis, expect, propagator, qeye, destroy

from .pulse_shapes import smooth_drag_pulse

MHZ_TO_RAD_NS = 2 * np.pi * 1e-3
STATE_NAMES = ("0", "1", "+", "-", "+i", "-i")
@lru_cache(maxsize=None)
def model_operators(levels=3):
    """Duffing ladder; leakage is the sum of ALL noncomputational populations."""
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) or levels < 3:
        raise ValueError("levels must be an integer >= 3")
    kets = [basis(levels, i) for i in range(levels)]
    states = [kets[0], kets[1], (kets[0]+kets[1]).unit(),
              (kets[0]-kets[1]).unit(), (kets[0]+1j*kets[1]).unit(),
              (kets[0]-1j*kets[1]).unit()]
    target = kets[0]*kets[1].dag()+kets[1]*kets[0].dag()
    lowering = destroy(levels)
    return (kets, states, target, lowering, lowering.dag()*lowering,
            (lowering+lowering.dag())/2, -.5j*(lowering-lowering.dag()),
            sum(k.proj() for k in kets[2:]))


# Compatibility aliases for legacy three-level notebooks, never used to set Device size.
KETS, TEST_STATES, TARGET_X, LOWERING, NUMBER, DRIVE_X, DRIVE_Y, LEAKAGE = model_operators()


@dataclass(frozen=True)
class Device:
    alpha_mhz: float = -220.0
    t1_1_ns: float = 30_000.0
    t1_2_ns: float = 18_000.0
    tphi_1_ns: float = 40_000.0
    tphi_2_ns: float = 25_000.0
    dt_ns: float = 0.2
    amplitude_limit_mhz: float = 80.0
    detuning_mhz: float = 0.0
    levels: int = 3

    def __post_init__(self):
        model_operators(self.levels)


@dataclass(frozen=True)
class Scenario:
    amplitude_scale: float = 1.0
    detuning_mhz: float = 0.0  # additive error relative to Device.detuning_mhz
    alpha_shift_mhz: float = 0.0
    t1_1_scale: float = 1.0
    t1_2_scale: float = 1.0
    tphi_1_scale: float = 1.0
    tphi_2_scale: float = 1.0


@dataclass(frozen=True)
class PulseParameters:
    duration_ns: float = 24.0
    sigma_fraction: float = 0.3
    beta: float = 1.0
    amplitude: float = 1.0
    phase_ramp_mhz: float = 0.0

    @classmethod
    def from_array(cls, values, dt_ns=0.2):
        values = np.asarray(values, dtype=float).copy()
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("pulse requires five finite parameters")
        values[0] = np.rint(values[0] / dt_ns) * dt_ns
        return cls(*values)

    def as_array(self):
        return np.array([self.duration_ns, self.sigma_fraction, self.beta,
                         self.amplitude, self.phase_ramp_mhz])


def make_pulse(parameters, device=Device()):
    """Quantized gate duration; exported samples are the actual simulated pulse."""
    duration = parameters.duration_ns
    steps = int(np.rint(duration / device.dt_ns))
    if (not 10 <= duration <= 60 or steps < 2
            or not np.isclose(steps * device.dt_ns, duration, atol=1e-10, rtol=0)):
        raise ValueError("gate duration must be 10-60 ns and a multiple of dt")
    times = np.arange(steps + 1) * device.dt_ns
    in_phase, quadrature = smooth_drag_pulse(
        times, device.alpha_mhz * MHZ_TO_RAD_NS, parameters.beta,
        parameters.sigma_fraction * duration, parameters.amplitude,
        parameters.phase_ramp_mhz)
    return times, in_phase, quadrature


def hardware_metrics(times, pulse_i, pulse_q, device=Device()):
    magnitude = np.hypot(pulse_i, pulse_q)
    peak_mhz = float(magnitude.max() / MHZ_TO_RAD_NS)
    endpoint = float(magnitude[[0, -1]].max())
    # A linear interpolant stays inside the convex amplitude disk at all times.
    return {"peak_amplitude_mhz": peak_mhz,
            "endpoint_amplitude_rad_ns": endpoint,
            "max_slew_rad_ns2": float(np.hypot(np.diff(pulse_i), np.diff(pulse_q)).max()
                                      / device.dt_ns),
            "duration_ns": float(times[-1]),
            "feasible": bool(peak_mhz <= device.amplitude_limit_mhz + 1e-10
                             and endpoint <= 1e-10)}


def collapse_operators(device=Device(), scenario=Scenario()):
    """Page 5: sqrt(1/T1) transitions and sqrt(2/Tphi) projectors.

    Page 2 has different factors. This explicit choice also gives the familiar
    rho01 decay rate 1/(2*T1_1) + 1/Tphi_1.
    """
    lifetimes = np.array([device.t1_1_ns, device.t1_2_ns,
                          device.tphi_1_ns, device.tphi_2_ns])
    scales = np.array([scenario.t1_1_scale, scenario.t1_2_scale,
                       scenario.tphi_1_scale, scenario.tphi_2_scale])
    if np.any(lifetimes <= 0) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("lifetimes and coherence scales must be positive")
    rates = 1 / (lifetimes * scales)
    kets = model_operators(device.levels)[0]
    # Preserve the supplied n=1,2 lifetimes exactly. For n>2, assume
    # transition rate n/2 times the n=2 rate and dephasing rate (n/2)^2
    # times the n=2 projector rate. These are model assumptions, not measurements.
    operators = []
    for n in range(1, device.levels):
        relaxation = rates[0] if n == 1 else rates[1]*n/2
        dephasing = rates[2] if n == 1 else rates[3]*(n/2)**2
        operators.extend([np.sqrt(relaxation)*kets[n-1]*kets[n].dag(),
                          np.sqrt(2*dephasing)*kets[n].proj()])
    return operators



def hamiltonian(times, pulse_i, pulse_q, device=Device(), scenario=Scenario()):
    """PDF page 4, rotating frame and rotating-wave approximation (hbar=1).

    H0 = Delta*n + alpha*n*(n-1)/2
    Hctrl = Omega_I*(a+a.dag())/2 - i*Omega_Q*(a-a.dag())/2

    In the ordered basis |0>, |1>, |2>, the diagonal is
    [0, Delta, 2*Delta+alpha]. Upper off-diagonal entries are
    (Omega_I-i*Omega_Q)/2 and sqrt(2)*(Omega_I-i*Omega_Q)/2.
    There is no direct |0><2| coupling. All coefficients are rad/ns.
    Device inputs are nominal; scenario errors are additive in MHz.
    T1/Tphi enter the page-5 Lindblad operators, not this Hamiltonian.
    """
    alpha = (device.alpha_mhz + scenario.alpha_shift_mhz) * MHZ_TO_RAD_NS
    delta = (device.detuning_mhz + scenario.detuning_mhz) * MHZ_TO_RAD_NS
    _, _, _, _, number, drive_x, drive_y, _ = model_operators(device.levels)
    h0 = delta * number + 0.5 * alpha * number * (number - qeye(device.levels))
    return QobjEvo([h0, [drive_x, scenario.amplitude_scale * pulse_i],
                   [drive_y, scenario.amplitude_scale * pulse_q]],
                   tlist=times, order=1)


def channel_metrics(channel, levels=3):
    """Six-state, unconditional X fidelity; leaked output is never normalized."""
    fidelities, leakages, trace_errors, min_eigenvalues = [], [], [], []
    _, states, target_x, _, _, _, _, leakage = model_operators(levels)
    for state in states:
        out = channel(state.proj())
        target = target_x * state
        fidelities.append(float(np.real(target.dag() * out * target)))
        leakages.append(float(expect(leakage, out)))
        trace_errors.append(float(abs(out.tr() - 1)))
        min_eigenvalues.append(float(np.linalg.eigvalsh(out.full()).min()))
    if (max(trace_errors) > 2e-6 or min(min_eigenvalues) < -2e-6
            or min(fidelities + leakages) < -2e-6
            or max(fidelities + leakages) > 1 + 2e-6):
        raise RuntimeError("nonphysical output: tighten solver tolerances")
    return {"fidelity": float(np.mean(fidelities)),
            "infidelity": float(1 - np.mean(fidelities)),
            "leakage": float(np.mean(leakages)),
            "state_fidelities": fidelities, "state_leakages": leakages,
            "max_trace_error": max(trace_errors),
            "min_output_eigenvalue": min(min_eigenvalues)}


def evaluate_waveform(times, pulse_i, pulse_q, device=Device(), scenario=Scenario(),
                      *, strict=False, coherent=False):
    """One Lindblad channel propagation evaluates all six input states."""
    h = hamiltonian(times, pulse_i, pulse_q, device, scenario)
    options = {"atol": 1e-12 if strict else 1e-10,
               "rtol": 1e-10 if strict else 1e-8,
               "max_step": device.dt_ns / (4 if strict else 2),
               "nsteps": 100_000, "normalize_output": False}
    channel = propagator(h, float(times[-1]), c_ops=[] if coherent else collapse_operators(device, scenario),
                         options=options)
    # Without collapse operators QuTiP returns a unitary rather than a channel.
    if coherent:
        from qutip import to_super
        channel = to_super(channel)
    return channel_metrics(channel, device.levels)


def training_scenarios(coherence_min=0.8, *, amplitude_fraction=.03, detuning_mhz=3., alpha_shift_mhz=8.):
    """Fixed nominal, single-axis limits and two joint limits; not MC test data."""
    scenarios = [Scenario()]
    for value in [1-amplitude_fraction, 1+amplitude_fraction]:
        scenarios.append(Scenario(amplitude_scale=value))
    for value in [-detuning_mhz, detuning_mhz]:
        scenarios.append(Scenario(detuning_mhz=value))
    for value in [-alpha_shift_mhz, alpha_shift_mhz]:
        scenarios.append(Scenario(alpha_shift_mhz=value))
    scales = dict(t1_1_scale=coherence_min, t1_2_scale=coherence_min,
                  tphi_1_scale=coherence_min, tphi_2_scale=coherence_min)
    scenarios.append(Scenario(**scales))
    scenarios.extend([Scenario(1-amplitude_fraction, -detuning_mhz, -alpha_shift_mhz, **scales),
                      Scenario(1+amplitude_fraction, detuning_mhz, alpha_shift_mhz, **scales)])
    return scenarios


def public_scenarios():
    """Full 3 x 5 x 3 Cartesian product from PDF page 8 (45 cases)."""
    return [Scenario(a, d, h) for a, d, h in
            product([0.97, 1, 1.03], [-3, -1, 0, 1, 3], [-8, 0, 8])]


def monte_carlo_scenarios(n_samples, seed, coherence_min=0.8, *,
                          amplitude_fraction=0.03, detuning_mhz=3.0, alpha_shift_mhz=8.0):
    """Independent uniform quasi-static uncertainties, sampled once per gate.

    The PDF specifies ranges, not probability laws or coherence degradation.
    Independent uniforms and 80-100% lifetimes are explicit modeling assumptions.
    """
    if (isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer))
            or n_samples < 2 or isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or seed < 0 or not 0 < coherence_min <= 1
            or not np.all(np.isfinite([amplitude_fraction, detuning_mhz, alpha_shift_mhz]))
            or not 0 <= amplitude_fraction < 1 or min(detuning_mhz, alpha_shift_mhz) < 0):
        raise ValueError("require at least 2 samples and 0 < coherence_min <= 1")
    draws = np.random.default_rng(seed).uniform(
        [1-amplitude_fraction, -detuning_mhz, -alpha_shift_mhz, *([coherence_min] * 4)],
        [1+amplitude_fraction, detuning_mhz, alpha_shift_mhz, 1, 1, 1, 1],
        size=(n_samples, 7))
    return [Scenario(*row) for row in draws]
