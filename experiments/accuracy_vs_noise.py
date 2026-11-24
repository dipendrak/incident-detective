"""Experiment 1: top-1 and top-3 accuracy of every method as alarm noise grows.

Two sweeps at N = 12: the false-positive rate (fn fixed at 0.1) and the
false-negative rate (fp fixed at 0.05). Incident k uses graph seed SEED + k
and the same random stream in every setting, so settings are compared on
the same graphs.
"""

from __future__ import annotations

from pathlib import Path

# `common` must be imported first: it puts the repo root on sys.path.
from common import COLORS, DEFAULT_OUT, METHODS, SEED, cli, rank_all_methods, setup_style, write_csv

import matplotlib.pyplot as plt
import numpy as np

from diagnose.graph import generate_graph
from diagnose.metrics import top_k_hit, wilson_ci
from diagnose.model import Params
from diagnose.simulate import simulate_incident

N = 12
FP_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3]
FN_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4]


def evaluate(params: Params, n_incidents: int, n_samples: int) -> dict[str, dict[str, int]]:
    """Count top-1 and top-3 hits per method over n_incidents simulated incidents."""
    hits = {m: {"top1": 0, "top3": 0} for m in METHODS}
    for k in range(n_incidents):
        # Same graph and random stream for incident k in every noise setting.
        graph = generate_graph(N, seed=SEED + k)
        rng = np.random.default_rng(SEED + k)
        inc = simulate_incident(graph, params, rng)
        for m, ranking in rank_all_methods(graph, params, inc.alarms, n_samples, rng).items():
            hits[m]["top1"] += top_k_hit(ranking, inc.root, 1)
            hits[m]["top3"] += top_k_hit(ranking, inc.root, 3)
    return hits


def run(out_dir: Path = DEFAULT_OUT, quick: bool = False) -> list[dict]:
    """Run both sweeps, write CSV and figure, return the rows."""
    n_incidents = 60 if quick else 300
    n_samples = 1000 if quick else 5000
    # Each setting is (which sweep, fp rate, fn rate).
    settings = [("fp", fp, 0.1) for fp in FP_VALUES] + [("fn", 0.05, fn) for fn in FN_VALUES]

    rows: list[dict] = []
    for sweep, fp, fn in settings:
        hits = evaluate(Params(fp_rate=fp, fn_rate=fn), n_incidents, n_samples)
        for m in METHODS:
            row = {"sweep": sweep, "fp_rate": fp, "fn_rate": fn, "method": m, "n_incidents": n_incidents}
            for key in ("top1", "top3"):
                lo, hi = wilson_ci(hits[m][key], n_incidents)
                row[key] = hits[m][key] / n_incidents
                row[f"{key}_lo"], row[f"{key}_hi"] = lo, hi
            rows.append(row)
        print(f"  {sweep} sweep fp={fp} fn={fn}: exact top1 {hits['exact']['top1'] / n_incidents:.3f}")

    write_csv(out_dir / "accuracy_vs_noise.csv", rows)
    plot(rows, out_dir / "accuracy_vs_noise.png", n_incidents)
    return rows


def plot(rows: list[dict], path: Path, n_incidents: int) -> None:
    """Two by two grid: rows are top-1 and top-3, columns are the two sweeps."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row")
    for col, (sweep, xkey, xlabel) in enumerate(
        [("fp", "fp_rate", "false-positive rate (fn = 0.1)"), ("fn", "fn_rate", "false-negative rate (fp = 0.05)")]
    ):
        for row_i, key in enumerate(("top1", "top3")):
            ax = axes[row_i, col]
            for m in METHODS:
                sel = [r for r in rows if r["sweep"] == sweep and r["method"] == m]
                x = [r[xkey] for r in sel]
                y = [r[key] for r in sel]
                ax.plot(x, y, marker="o", color=COLORS[m], label=m, linestyle="--" if m == "random" else "-")
                ax.fill_between(x, [r[f"{key}_lo"] for r in sel], [r[f"{key}_hi"] for r in sel],
                                color=COLORS[m], alpha=0.15, linewidth=0)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("top-1 accuracy" if key == "top1" else "top-3 accuracy")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Root-cause accuracy vs alarm noise (simulated, N = 12, {n_incidents} incidents per point, 95% CI)")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    cli(run, __doc__)
