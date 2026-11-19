"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from diagnose.baselines import most_alarmed_descendants, most_upstream, random_ranking  # noqa: E402
from diagnose.exact import exact_posterior  # noqa: E402
from diagnose.graph import Graph  # noqa: E402
from diagnose.metrics import ranking_from_scores  # noqa: E402
from diagnose.model import Params  # noqa: E402
from diagnose.sampling import likelihood_weighting  # noqa: E402

DEFAULT_OUT = ROOT / "results"
SEED = 560  # base seed for every experiment

METHODS = ["exact", "likelihood weighting", "most upstream", "most alarmed descendants", "random"]
COLORS = {
    "exact": "#2a78d6",
    "likelihood weighting": "#eb6834",
    "most upstream": "#1baf7a",
    "most alarmed descendants": "#eda100",
    "random": "#8a8984",
}


def setup_style() -> None:
    """Readable, quiet matplotlib defaults for all figures."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    )


def rank_all_methods(
    graph: Graph, params: Params, alarms: np.ndarray, n_samples: int, rng: np.random.Generator
) -> dict[str, list[int]]:
    """Run every method on one incident and return each ranking."""
    post = exact_posterior(graph, params, alarms)
    lw = likelihood_weighting(graph, params, alarms, n_samples, rng)
    return {
        "exact": ranking_from_scores(post),
        "likelihood weighting": ranking_from_scores(lw.posterior),
        "most upstream": most_upstream(graph, alarms),
        "most alarmed descendants": most_alarmed_descendants(graph, alarms),
        "random": random_ranking(graph, rng),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to CSV with the keys of the first row as header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
