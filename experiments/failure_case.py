"""Experiment 4: when likelihood weighting breaks down.

Likelihood weighting proposes failures from the prior and ignores the alarms
until it weights the samples. When the observed alarm pattern is unlikely
under the prior (many alarms and a very low false-positive rate), almost all
weight falls on a handful of samples, the effective sample size (ESS)
collapses, and the estimate becomes noisy.

We simulate many incidents at N = 12, group them by the number of alarms,
and compare a default noise setting with a low-noise one. LW uses a modest
1,000 samples so the effect is visible.
"""

from __future__ import annotations

from pathlib import Path

# `common` must be imported first: it puts the repo root on sys.path.
from common import DEFAULT_OUT, SEED, cli, setup_style, write_csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from diagnose.exact import exact_posterior
from diagnose.graph import generate_graph
from diagnose.metrics import total_variation
from diagnose.model import Params
from diagnose.sampling import likelihood_weighting
from diagnose.simulate import simulate_incident

N = 12
N_SAMPLES = 1000
SETTINGS = {
    "default noise (fp 0.05, fn 0.1)": Params(fp_rate=0.05, fn_rate=0.1),
    "low noise (fp 0.001, fn 0.01)": Params(fp_rate=0.001, fn_rate=0.01),
}
BUCKETS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, N)]  # inclusive ranges of alarm counts
SETTING_COLORS = ["#2a78d6", "#eb6834"]


def bucket_label(lo: int, hi: int) -> str:
    """Human-readable label for an alarm-count bucket."""
    return f"{lo}+" if hi == N else f"{lo}-{hi}"


def run(out_dir: Path = DEFAULT_OUT, quick: bool = False) -> list[dict]:
    """Collect ESS, TV distance and accuracy per alarm bucket; write CSV, figure and text."""
    n_incidents = 300 if quick else 2000
    rows: list[dict] = []
    for name, params in SETTINGS.items():
        records = []
        for k in range(n_incidents):
            graph = generate_graph(N, seed=SEED + k)
            rng = np.random.default_rng(SEED + k)
            inc = simulate_incident(graph, params, rng)
            post = exact_posterior(graph, params, inc.alarms)
            lw = likelihood_weighting(graph, params, inc.alarms, N_SAMPLES, rng)
            # One record per incident: alarm count, ESS, TV distance, exact hit, LW hit.
            records.append((
                int(inc.alarms.sum()), lw.ess, total_variation(post, lw.posterior),
                int(np.argmax(post) == inc.root), int(np.argmax(lw.posterior) == inc.root),
            ))
        rec = np.array(records, dtype=float)
        for lo, hi in BUCKETS:
            sel = rec[(rec[:, 0] >= lo) & (rec[:, 0] <= hi)]
            if len(sel) == 0:
                continue
            # Summarize every incident whose alarm count falls in this bucket.
            rows.append({
                "setting": name, "alarms": bucket_label(lo, hi), "n_incidents": len(sel),
                "median_ess": float(np.median(sel[:, 1])), "ess_fraction": float(np.median(sel[:, 1]) / N_SAMPLES),
                "mean_tv": float(sel[:, 2].mean()),
                "exact_top1": float(sel[:, 3].mean()), "lw_top1": float(sel[:, 4].mean()),
            })
            r = rows[-1]
            print(f"  {name}, alarms {r['alarms']}: n={r['n_incidents']}, median ESS {r['median_ess']:.1f}, "
                  f"TV {r['mean_tv']:.3f}, exact {r['exact_top1']:.2f}, LW {r['lw_top1']:.2f}")

    write_csv(out_dir / "failure_case.csv", rows)
    plot(rows, out_dir / "failure_case.png", n_incidents)
    write_explanation(rows, out_dir / "failure_case.txt", n_incidents)
    return rows


def plot(rows: list[dict], path: Path, n_incidents: int) -> None:
    """Median ESS, TV distance, and the accuracy gap, by number of alarms."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, color in zip(SETTINGS, SETTING_COLORS):
        sel = [r for r in rows if r["setting"] == name]
        x = [r["alarms"] for r in sel]
        axes[0].plot(x, [r["median_ess"] for r in sel], marker="o", color=color, label=name)
        axes[1].plot(x, [r["mean_tv"] for r in sel], marker="o", color=color, label=name)
        axes[2].plot(x, [r["exact_top1"] - r["lw_top1"] for r in sel], marker="o", color=color, label=name)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel(f"median ESS (of {N_SAMPLES} samples)")
    axes[0].set_title("Effective sample size collapses")
    axes[1].set_ylabel("mean TV distance to exact")
    axes[1].set_title("Posterior estimate gets worse")
    axes[2].axhline(0, color="#8a8984", linewidth=1)
    axes[2].set_ylabel("exact top-1 minus LW top-1")
    axes[2].set_title("Accuracy lost to sampling error")
    for ax in axes:
        ax.set_xlabel("number of alarms in the incident")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(f"Likelihood weighting failure case (simulated, N = 12, {n_incidents} incidents per setting)")
    fig.savefig(path)
    plt.close(fig)


def write_explanation(rows: list[dict], path: Path, n_incidents: int) -> None:
    """Short plain-text explanation with the measured numbers filled in."""
    low = [r for r in rows if r["setting"].startswith("low")]
    few, many = low[0], low[-1]
    text = f"""Likelihood weighting failure case (simulated data, N = 12, {n_incidents} incidents per setting, {N_SAMPLES} samples)

What we see
  In the low-noise setting (fp 0.001, fn 0.01), incidents with {few['alarms']} alarms keep a median
  effective sample size of {few['median_ess']:.1f} out of {N_SAMPLES} samples, with a mean total
  variation distance to the exact posterior of {few['mean_tv']:.3f}. Incidents with {many['alarms']}
  alarms drop to a median ESS of {many['median_ess']:.1f}, the distance grows to {many['mean_tv']:.3f},
  and top-1 accuracy is {many['exact_top1']:.2f} for exact vs {many['lw_top1']:.2f} for LW.
  See failure_case.csv for every bucket and both noise settings.

What is happening
  Likelihood weighting draws the root and the failures from the prior, without looking
  at the alarms, and only afterwards weights each sample by P(alarms | failures). That
  weight is a product of one factor per service. A sample that leaves an alarmed service
  healthy pays a factor of fp (0.001 here), and a sample that fails a silent service pays
  a factor of fn (0.01). With many alarms, a sample has to reproduce the whole pattern of
  failures almost exactly to avoid these penalties, and forward simulation from the prior
  rarely does that. A few lucky samples then carry nearly all of the weight, so the
  posterior estimate is effectively built from a handful of samples. The ESS,
  (sum w)^2 / sum w^2, measures exactly this.

  Lower noise makes the penalty factors smaller, so the weights become more uneven for
  the same number of alarms. The model is not wrong here: exact inference on the same
  evidence stays accurate. The failure is purely in the sampler.

What would help
  More samples (cost grows linearly), a proposal that looks at the alarms (for example,
  drawing the root near alarmed services and correcting with importance weights), or an
  MCMC method such as Gibbs sampling. A low ESS is a useful warning sign to report.
"""
    path.write_text(text)


if __name__ == "__main__":
    cli(run, __doc__)
