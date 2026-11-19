"""Random service dependency graphs.

Services are numbered 0..N-1 in topological order. An edge j -> i means
service i depends on service j, so a failure of j can flow to i. Every
parent of i has a smaller index than i.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    """A directed acyclic graph stored as a parent list per node."""

    parents: tuple[tuple[int, ...], ...]

    @property
    def n(self) -> int:
        """Number of services."""
        return len(self.parents)

    def names(self) -> list[str]:
        """Display names for the services."""
        return [f"S{i}" for i in range(self.n)]

    def children(self) -> list[list[int]]:
        """Child list per node (the services that depend on it)."""
        kids: list[list[int]] = [[] for _ in range(self.n)]
        for i, ps in enumerate(self.parents):
            for j in ps:
                kids[j].append(i)
        return kids

    def descendants(self) -> list[set[int]]:
        """Set of all downstream services for each node."""
        kids = self.children()
        desc: list[set[int]] = [set() for _ in range(self.n)]
        # Reverse topological order: children are finished before parents.
        for i in reversed(range(self.n)):
            for c in kids[i]:
                desc[i].add(c)
                desc[i] |= desc[c]
        return desc

    def parent_matrix(self) -> np.ndarray:
        """Adjacency matrix M with M[j, i] = 1 when j is a parent of i."""
        m = np.zeros((self.n, self.n), dtype=np.int32)
        for i, ps in enumerate(self.parents):
            for j in ps:
                m[j, i] = 1
        return m

    def layers(self) -> list[int]:
        """Depth of each node: 0 for sources, else 1 + max parent depth."""
        depth = [0] * self.n
        for i, ps in enumerate(self.parents):
            if ps:
                depth[i] = 1 + max(depth[j] for j in ps)
        return depth


def generate_graph(n: int, seed: int, max_parents: int = 3) -> Graph:
    """Generate a random DAG where each node has 0 to max_parents parents.

    Node i picks its parents uniformly from the nodes before it, so the
    numbering is already a topological order.
    """
    rng = np.random.default_rng(seed)
    parents: list[tuple[int, ...]] = []
    for i in range(n):
        k = int(rng.integers(0, min(max_parents, i) + 1))
        chosen = rng.choice(i, size=k, replace=False) if k > 0 else []
        parents.append(tuple(sorted(int(j) for j in chosen)))
    return Graph(parents=tuple(parents))


def chain_graph(n: int) -> Graph:
    """A simple chain 0 -> 1 -> ... -> n-1 (used in tests)."""
    return Graph(parents=tuple(() if i == 0 else (i - 1,) for i in range(n)))
