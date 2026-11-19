"""Scoring helpers: rankings, top-k hits, distances, confidence intervals."""

from __future__ import annotations

import numpy as np


def ranking_from_scores(scores: np.ndarray) -> list[int]:
    """Services sorted by descending score. Ties go to the lower index."""
    return [int(i) for i in np.argsort(-np.asarray(scores), kind="stable")]


def top_k_hit(ranking: list[int], true_root: int, k: int) -> bool:
    """True when the true root is among the first k suspects."""
    return true_root in ranking[:k]


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two distributions."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95 percent Wilson score interval for a proportion."""
    if trials == 0:
        return 0.0, 1.0
    phat = successes / trials
    denom = 1 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2)) / denom
    return float(centre - half), float(centre + half)


def mean_ci(values: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    """Mean and normal-approximation 95 percent interval."""
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    half = z * float(v.std(ddof=1)) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, m - half, m + half
