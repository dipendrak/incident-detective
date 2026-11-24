"""Approximate inference by likelihood weighting.

Each sample draws R uniformly, forward simulates the failures F from the
prior, and is weighted by P(A observed | F). The posterior over R is the
share of total weight carried by samples with each root.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diagnose.graph import Graph
from diagnose.model import Params, log_alarm_likelihood, sample_failures


@dataclass
class LWResult:
    """Posterior estimate and diagnostics from likelihood weighting."""

    posterior: np.ndarray
    ess: float
    n_samples: int


def likelihood_weighting(
    graph: Graph,
    params: Params,
    alarms: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> LWResult:
    """Estimate P(R | A) with n_samples weighted forward samples.

    In plain terms: imagine many random outages, each starting at a random
    service. Score each imagined outage by how well it matches the real
    alarms. The posterior for service r is r's share of the total score.
    """
    n = graph.n
    roots = rng.integers(0, n, size=n_samples)
    failures = sample_failures(graph, params, roots, rng)
    log_w = log_alarm_likelihood(failures, alarms, params)

    m = np.max(log_w)
    if not np.isfinite(m):
        # Every sample is impossible under the evidence: no information.
        return LWResult(np.full(n, 1.0 / n), 0.0, n_samples)
    w = np.exp(log_w - m)  # rescaling does not change normalized weights
    # Add up the weight of every sample that started at each root.
    posterior = np.bincount(roots, weights=w, minlength=n) / w.sum()
    # Effective sample size: how many equally weighted samples the uneven
    # weights are worth. Close to n_samples is good; close to 1 is bad.
    ess = float(w.sum() ** 2 / np.sum(w**2))
    return LWResult(posterior, ess, n_samples)
