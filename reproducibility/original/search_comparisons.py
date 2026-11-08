"""Optional random-search benchmark against the primary Nelder–Mead search."""
from time import perf_counter
import numpy as np


def run_comparisons(config, evaluator_class, initial):
    bounds = np.asarray(config['bounds'], dtype=float)
    low, high = bounds.T
    budget = config.get('random_search', {}).get('max_evaluations', 500)
    records, costs = [], []
    for seed in config['search_seeds']:
        evaluator = evaluator_class(config)
        started = perf_counter()
        evaluator.scalar(np.clip(np.asarray(initial), low, high))
        for u in np.random.default_rng(seed).uniform(size=(budget-1, len(low))):
            evaluator.scalar(low+u*(high-low))
        records.extend(dict(r, method='Random search', search_seed=seed) for r in evaluator.cache.values())
        costs.append(dict(method='Random search', seed=seed, budget_evaluations=budget,
            unique_candidates=len(evaluator.cache), channel_simulations=evaluator.simulations,
            elapsed_seconds=perf_counter()-started, success=True,
            message='Completed fixed random-search budget', objective_calls=budget))
    return records, costs
