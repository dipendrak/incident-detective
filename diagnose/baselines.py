"""Simple heuristic baselines. Each returns a full ranking of suspects."""

from __future__ import annotations

import numpy as np

from diagnose.graph import Graph


def most_upstream(graph: Graph, alarms: np.ndarray) -> list[int]:
    """Blame the alarmed service that sits highest up in the graph.

    Idea: a failure flows downstream, so the first alarm in topological order
    is the likeliest source. Alarmed services come first (upstream to
    downstream), then the silent ones.
    """
    alarmed = [i for i in range(graph.n) if alarms[i]]
    silent = [i for i in range(graph.n) if not alarms[i]]
    return alarmed + silent


def most_alarmed_descendants(graph: Graph, alarms: np.ndarray) -> list[int]:
    """Blame the alarmed service whose downstream services alarm the most.

    Idea: a true root should "explain" many alarms below it. Ties are broken
    by topological order. Silent services come last.
    """
    desc = graph.descendants()
    score = [sum(bool(alarms[d]) for d in desc[i]) for i in range(graph.n)]
    alarmed = sorted((i for i in range(graph.n) if alarms[i]), key=lambda i: (-score[i], i))
    silent = sorted((i for i in range(graph.n) if not alarms[i]), key=lambda i: (-score[i], i))
    return alarmed + silent


def random_ranking(graph: Graph, rng: np.random.Generator) -> list[int]:
    """A uniformly random ranking, showing what pure chance scores."""
    return [int(i) for i in rng.permutation(graph.n)]
