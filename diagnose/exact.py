"""Exact inference by enumerating every failure configuration.

P(R = r | A) is proportional to P(R = r) * sum over F of P(F | R = r) * P(A | F).
We list all 2^N configurations F as rows of a matrix and do the sum with
numpy. Cost grows as N * 2^N, which is fine up to roughly N = 16 to 18.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from diagnose.graph import Graph
from diagnose.model import Params, _safe_log, fail_prob, log_alarm_likelihood


@lru_cache(maxsize=4)
def all_configurations(n: int) -> np.ndarray:
    """Boolean matrix of shape (2^n, n): row k is the binary expansion of k."""
    codes = np.arange(2**n, dtype=np.int64)[:, None]
    return ((codes >> np.arange(n, dtype=np.int64)) & 1).astype(bool)


def _logsumexp(x: np.ndarray) -> float:
    """Stable log(sum(exp(x))) that returns -inf for an all -inf input."""
    m = np.max(x)
    if not np.isfinite(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(x - m))))


def exact_posterior(graph: Graph, params: Params, alarms: np.ndarray) -> np.ndarray:
    """Return the exact posterior P(R = r | A) as an array of length N."""
    n = graph.n
    f = all_configurations(n)
    # Number of failed parents of every node in every configuration.
    k = f.astype(np.int32) @ graph.parent_matrix()
    p = fail_prob(k, params)
    # node_log[s, i] = log P(F_i | parents) if i is NOT the root.
    node_log = np.where(f, _safe_log(p), _safe_log(1.0 - p))
    log_lik = log_alarm_likelihood(f, alarms, params)
    # Sum of all node terms, kept as (finite part, number of -inf terms) so
    # that one node's term can be removed later without computing inf - inf.
    finite = np.isfinite(node_log)
    finite_sum = np.where(finite, node_log, 0.0).sum(axis=1)
    n_impossible = (~finite).sum(axis=1)

    log_post = np.full(n, -np.inf)
    for r in range(n):
        # If r is the root, F_r = 1 with certainty, so configurations with
        # F_r = 0 are impossible and node r's own term is replaced by log 1 = 0.
        rows = f[:, r]
        own_finite = finite[rows, r]
        rest_sum = finite_sum[rows] - np.where(own_finite, node_log[rows, r], 0.0)
        rest_impossible = n_impossible[rows] - (~own_finite)
        log_prior_f = np.where(rest_impossible > 0, -np.inf, rest_sum)
        log_post[r] = _logsumexp(log_prior_f + log_lik[rows])  # uniform P(R) cancels

    z = _logsumexp(log_post)
    if not np.isfinite(z):
        return np.full(n, 1.0 / n)
    return np.exp(log_post - z)
