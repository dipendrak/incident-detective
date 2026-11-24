"""The Bayesian network: parameters, conditional probabilities, forward sampling.

Variables:
    R    root cause, uniform over the N services (single fault).
    F_i  failure of service i. F_R = 1. For i != R the failure follows a
         noisy-OR over failed parents plus a small leak:
         P(F_i = 1 | parents) = 1 - (1 - leak) * (1 - p_prop) ** (#failed parents)
    A_i  alarm on service i. P(A_i = 1 | F_i = 1) = 1 - fn_rate,
         P(A_i = 1 | F_i = 0) = fp_rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diagnose.graph import Graph


@dataclass(frozen=True)
class Params:
    """Model parameters. Defaults match the project specification."""

    p_prop: float = 0.7
    leak: float = 0.01
    fp_rate: float = 0.05
    fn_rate: float = 0.1


def fail_prob(n_failed_parents: np.ndarray, params: Params) -> np.ndarray:
    """Chance that a non-root service fails, given how many of its parents failed.

    Each failed parent gets an independent chance p_prop to knock the service
    over, and the leak is a small chance it fails on its own. The service
    stays healthy only if every one of those chances misses.
    """
    return 1.0 - (1.0 - params.leak) * (1.0 - params.p_prop) ** n_failed_parents


def safe_log(x: np.ndarray) -> np.ndarray:
    """Elementwise log that maps 0 to -inf without warnings."""
    with np.errstate(divide="ignore"):
        return np.log(x)


def log_alarm_likelihood(failures: np.ndarray, alarms: np.ndarray, params: Params) -> np.ndarray:
    """How well each failure pattern explains the observed alarms, as log P(A | F).

    `failures` has one failure pattern per row (shape (S, N)). For every row we
    ask: if exactly these services were down, how likely is the alarm pattern
    we actually saw? Alarms are independent given the failures, so the log
    likelihood is a sum of one term per service.
    """
    f = failures.astype(bool)
    a = alarms.astype(bool)[None, :]
    p_alarm = np.where(f, 1.0 - params.fn_rate, params.fp_rate)
    p_obs = np.where(a, p_alarm, 1.0 - p_alarm)
    return safe_log(p_obs).sum(axis=1)


def sample_failures(graph: Graph, params: Params, roots: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate how failures spread, for a whole batch of root causes at once.

    We walk the services in topological order, so every parent is decided
    before its children. A service fails if it is the root, or if the
    noisy-OR coin flip (driven by its failed parents) comes up "fail".

    Args:
        roots: integer array of shape (S,), the root cause of each sample.
    Returns:
        Boolean array of shape (S, N).
    """
    s = len(roots)
    f = np.zeros((s, graph.n), dtype=bool)
    for i in range(graph.n):
        ps = list(graph.parents[i])
        k = f[:, ps].sum(axis=1) if ps else np.zeros(s, dtype=np.int64)
        knocked_over = rng.random(s) < fail_prob(k, params)
        f[:, i] = (roots == i) | knocked_over
    return f


def sample_alarms(failures: np.ndarray, params: Params, rng: np.random.Generator) -> np.ndarray:
    """Draw noisy alarms: failed services usually alarm, healthy ones rarely do."""
    p_alarm = np.where(failures, 1.0 - params.fn_rate, params.fp_rate)
    return rng.random(failures.shape) < p_alarm
