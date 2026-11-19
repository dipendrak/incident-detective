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
    """Noisy-OR probability that a non-root service fails."""
    return 1.0 - (1.0 - params.leak) * (1.0 - params.p_prop) ** n_failed_parents


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Elementwise log that maps 0 to -inf without warnings."""
    with np.errstate(divide="ignore"):
        return np.log(x)


def log_alarm_likelihood(failures: np.ndarray, alarms: np.ndarray, params: Params) -> np.ndarray:
    """log P(A | F) for each row of a (S, N) failure matrix.

    Alarms are independent given the failures, so the log likelihood is a
    sum over services.
    """
    f = failures.astype(bool)
    a = alarms.astype(bool)[None, :]
    p_alarm = np.where(f, 1.0 - params.fn_rate, params.fp_rate)
    p_obs = np.where(a, p_alarm, 1.0 - p_alarm)
    return _safe_log(p_obs).sum(axis=1)


def sample_failures(graph: Graph, params: Params, roots: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Forward simulate failures for a batch of root causes.

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
        spontaneous = rng.random(s) < fail_prob(k, params)
        f[:, i] = (roots == i) | spontaneous
    return f


def sample_alarms(failures: np.ndarray, params: Params, rng: np.random.Generator) -> np.ndarray:
    """Draw noisy alarms for a (S, N) failure matrix."""
    p_alarm = np.where(failures, 1.0 - params.fn_rate, params.fp_rate)
    return rng.random(failures.shape) < p_alarm
