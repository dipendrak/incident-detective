"""Incident simulator used by every experiment and by the demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diagnose.graph import Graph
from diagnose.model import Params, sample_alarms, sample_failures


@dataclass
class Incident:
    """One simulated incident: the true root, the failures, and the alarms."""

    root: int
    failures: np.ndarray
    alarms: np.ndarray


def simulate_incident(
    graph: Graph,
    params: Params,
    rng: np.random.Generator,
    root: int | None = None,
) -> Incident:
    """Create one fake outage: break a service, let the failure spread, raise alarms.

    The root is random unless `root` is given. Only `alarms` would be visible
    to an operator; `root` and `failures` are the hidden ground truth we
    score against.
    """
    if root is None:
        root = int(rng.integers(0, graph.n))
    failures = sample_failures(graph, params, np.array([root]), rng)
    alarms = sample_alarms(failures, params, rng)
    return Incident(root=root, failures=failures[0], alarms=alarms[0])
