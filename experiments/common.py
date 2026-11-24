"""Shared helpers for the experiment scripts.

Every experiment script imports this module first. Importing it puts the repo
root on sys.path (so `import diagnose` works when a script is run directly)
and switches matplotlib to a non-interactive backend that just writes files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

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
    """Run every method on one incident and return each one's ranking of suspects.

    A ranking is a list of service indices, most suspicious first.
    """
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


def cli(run: Callable[[Path, bool], object], doc: str | None) -> None:
    """Command-line entry point shared by the experiment scripts (--quick, --out)."""
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--quick", action="store_true", help="fewer incidents and samples")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()
    run(args.out, args.quick)
