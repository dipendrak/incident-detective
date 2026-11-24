"""Experiment 2: how fast likelihood weighting converges to the exact posterior.

For each sample budget we report the total variation distance to the exact
posterior and how often the top suspect agrees with exact, averaged over
simulated incidents at N = 12 with default noise.
"""

from __future__ import annotations

from pathlib import Path

# `common` must be imported first: it puts the repo root on sys.path.
from common import COLORS, DEFAULT_OUT, SEED, cli, setup_style, write_csv

import matplotlib.pyplot as plt
import numpy as np

from diagnose.exact import exact_posterior
from diagnose.graph import generate_graph
from diagnose.metrics import mean_ci, total_variation, wilson_ci
from diagnose.model import Params
from diagnose.sampling import likelihood_weighting
from diagnose.simulate import simulate_incident

N = 12
SAMPLE_SIZES = [100, 300, 1000, 3000, 10000, 30000, 100000]


def run(out_dir: Path = DEFAULT_OUT, quick: bool = False) -> list[dict]:
    """Compare LW to exact over a range of sample sizes; write CSV and figure."""
    n_incidents = 40 if quick else 200
    sizes = SAMPLE_SIZES[:5] if quick else SAMPLE_SIZES
    params = Params()

    # Simulate the incidents and their exact posteriors once, then reuse them
    # for every sample budget.
    incidents = []
    for k in range(n_incidents):
        graph = generate_graph(N, seed=SEED + k)
        inc = simulate_incident(graph, params, np.random.default_rng(SEED + k))
        incidents.append((graph, inc, exact_posterior(graph, params, inc.alarms)))

    rows: list[dict] = []
    for s in sizes:
        tvs, agree, ess = [], 0, []
        for k, (graph, inc, post) in enumerate(incidents):
            lw = likelihood_weighting(graph, params, inc.alarms, s, np.random.default_rng([SEED, s, k]))
            tvs.append(total_variation(post, lw.posterior))
            agree += int(np.argmax(lw.posterior) == np.argmax(post))
            ess.append(lw.ess)
        tv_m, tv_lo, tv_hi = mean_ci(np.array(tvs))
        ag_lo, ag_hi = wilson_ci(agree, n_incidents)
        rows.append({
            "n_samples": s, "n_incidents": n_incidents,
            "tv_mean": tv_m, "tv_lo": tv_lo, "tv_hi": tv_hi,
            "top1_agreement": agree / n_incidents, "agree_lo": ag_lo, "agree_hi": ag_hi,
            "ess_mean": float(np.mean(ess)),
        })
        print(f"  samples={s}: TV {tv_m:.4f}, top-1 agreement {agree / n_incidents:.3f}, mean ESS {np.mean(ess):.0f}")

    write_csv(out_dir / "sampling_convergence.csv", rows)
    plot(rows, out_dir / "sampling_convergence.png", n_incidents)
    return rows


def plot(rows: list[dict], path: Path, n_incidents: int) -> None:
    """TV distance and top-1 agreement vs number of samples, log x-axis."""
    setup_style()
    color = COLORS["likelihood weighting"]
    x = [r["n_samples"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(x, [r["tv_mean"] for r in rows], marker="o", color=color)
    ax1.fill_between(x, [r["tv_lo"] for r in rows], [r["tv_hi"] for r in rows], color=color, alpha=0.2, linewidth=0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("number of samples")
    ax1.set_ylabel("total variation distance to exact")
    ax1.set_title("Posterior error shrinks with more samples")
    ax2.plot(x, [r["top1_agreement"] for r in rows], marker="o", color=color)
    ax2.fill_between(x, [r["agree_lo"] for r in rows], [r["agree_hi"] for r in rows], color=color, alpha=0.2, linewidth=0)
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("number of samples")
    ax2.set_ylabel("top-1 agreement with exact")
    ax2.set_title("Top suspect matches exact inference")
    fig.suptitle(f"Likelihood weighting vs exact (simulated, N = 12, {n_incidents} incidents, 95% CI)")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    cli(run, __doc__)
