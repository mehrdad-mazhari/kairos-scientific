"""Bounded Nelder–Mead in normalized coordinates, with reproducible restarts."""
import numpy as np
from scipy.optimize import minimize as scipy_minimize

DEFAULTS = dict(enabled=True, max_evaluations=500, xatol=1e-5, fatol=1e-10)


def settings(config):
    return {**DEFAULTS, **config.get('nelder_mead', {})}


def minimize(objective, bounds, x0, *, seed=7, max_evaluations=500,
             xatol=1e-5, fatol=1e-10, randomize_start=False, callback=None):
    """Minimize a scalar loss, retaining a complete per-evaluation trace.

    Bounds apply in physical units. The simplex operates on [0,1]^d so MHz
    and dimensionless controls have comparable step sizes. Later restarts
    perturb the calibrated start deterministically; all searches see training
    data only. Budget exhaustion is reported separately from convergence.
    """
    bounds = np.asarray(bounds, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) != len(x0)
            or not np.all(np.isfinite(bounds)) or not np.all(np.isfinite(x0))
            or np.any(bounds[:,1] <= bounds[:,0]) or np.any(x0 < bounds[:,0])
            or np.any(x0 > bounds[:,1])):
        raise ValueError('require finite increasing bounds and an in-bounds starting point')
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int)
            or max_evaluations < len(x0)+1 or not np.isfinite(xatol) or xatol <= 0
            or not np.isfinite(fatol) or fatol <= 0):
        raise ValueError('invalid Nelder–Mead budget or tolerances')
    low, high = bounds.T
    rng = np.random.default_rng(seed)
    u0 = (x0-low)/(high-low)
    if randomize_start:
        u0 = np.clip(u0+rng.normal(0., .12, len(x0)), 0., 1.)
    simplex = np.tile(u0, (len(x0)+1, 1))
    for j, step in enumerate(rng.uniform(.03, .07, len(x0))):
        simplex[j+1,j] += step if u0[j] <= .9 else -step
    history = []
    best_loss = float('inf')

    def evaluate(u):
        nonlocal best_loss
        x = low+np.clip(u, 0., 1.)*(high-low)
        loss = float(objective(x))
        if not np.isfinite(loss):
            raise ValueError('Nelder–Mead objective must return a finite scalar')
        best_loss = min(best_loss, loss)
        row = dict(seed=seed, evaluations=len(history)+1, loss=loss,
                   best_loss=best_loss, parameters=x.tolist())
        history.append(row)
        if callback:
            callback(row)
        return loss

    result = scipy_minimize(evaluate, u0, method='Nelder-Mead', bounds=[(0.,1.)]*len(x0),
        options=dict(maxfev=max_evaluations, xatol=xatol, fatol=fatol, initial_simplex=simplex))
    result.x = low+result.x*(high-low)
    result.history = history
    return result
