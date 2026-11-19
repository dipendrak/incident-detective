"""Experiment 3: run time and accuracy of exact vs likelihood weighting as N grows.

Exact inference is stopped at the first graph size where its mean time per
incident (measured on a few probe incidents) exceeds TIME_LIMIT_S. The probe
time is still recorded so the figure shows where the limit was crossed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import COLORS, DEFAULT_OUT, SEED, setup_style, write_csv

import matplotlib.pyplot as plt
import numpy as np

from diagnose.exact import exact_posterior
from diagnose.graph import generate_graph
from diagnose.metrics import top_k_hit, ranking_from_scores, wilson_ci
from diagnose.model import Params
from diagnose.sampling import likelihood_weighting
from diagnose.simulate import simulate_incident

SIZES = [6, 8, 10, 12, 14, 16, 18]
N_SAMPLES = 5000
TIME_LIMIT_S = 0.1  # per incident
N_PROBE = 3


def _timed(fn, *args):
    """Call fn(*args) and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def run(out_dir: Path = DEFAULT_OUT, quick: bool = False) -> list[dict]:
    """Time both methods for every graph size; write CSV and figure."""
    n_incidents = 20 if quick else 100
    params = Params()
    rows: list[dict] = []
    exact_alive = True

    for n in SIZES:
        cases = []
        for k in range(n_incidents):
            graph = generate_graph(n, seed=SEED + k)
            inc = simulate_incident(graph, params, np.random.default_rng(SEED + k))
            cases.append((graph, inc))

        # Likelihood weighting: always runs.
        times, hits = [], 0
        for k, (graph, inc) in enumerate(cases):
            lw, dt = _timed(likelihood_weighting, graph, params, inc.alarms, N_SAMPLES, np.random.default_rng([SEED, n, k]))
            times.append(dt)
            hits += top_k_hit(ranking_from_scores(lw.posterior), inc.root, 1)
        rows.append(_row(n, "likelihood weighting", times, hits, n_incidents, "ok"))

        # Exact: probe first, stop if over the time limit.
        if not exact_alive:
            rows.append({"n": n, "method": "exact", "mean_time_s": float("nan"), "top1": float("nan"),
                         "top1_lo": float("nan"), "top1_hi": float("nan"), "n_incidents": 0, "status": "skipped"})
            continue
        times, hits = [], 0
        for k, (graph, inc) in enumerate(cases):
            post, dt = _timed(exact_posterior, graph, params, inc.alarms)
            times.append(dt)
            hits += top_k_hit(ranking_from_scores(post), inc.root, 1)
            if k + 1 == N_PROBE and np.mean(times) > TIME_LIMIT_S:
                exact_alive = False
                break
        if exact_alive:
            rows.append(_row(n, "exact", times, hits, n_incidents, "ok"))
        else:
            rows.append({"n": n, "method": "exact", "mean_time_s": float(np.mean(times)), "top1": float("nan"),
                         "top1_lo": float("nan"), "top1_hi": float("nan"), "n_incidents": len(times),
                         "status": f"stopped: over {TIME_LIMIT_S} s per incident"})
        print(f"  N={n}: " + ", ".join(f"{r['method']} {r['mean_time_s'] * 1000:.1f} ms" for r in rows[-2:]))

    write_csv(out_dir / "scaling.csv", rows)
    plot(rows, out_dir / "scaling.png", n_incidents)
    return rows


def _row(n: int, method: str, times: list[float], hits: int, n_incidents: int, status: str) -> dict:
    lo, hi = wilson_ci(hits, n_incidents)
    return {"n": n, "method": method, "mean_time_s": float(np.mean(times)), "top1": hits / n_incidents,
            "top1_lo": lo, "top1_hi": hi, "n_incidents": n_incidents, "status": status}


def plot(rows: list[dict], path: Path, n_incidents: int) -> None:
    """Run time on a log scale and top-1 accuracy, both vs graph size."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in ("exact", "likelihood weighting"):
        sel = [r for r in rows if r["method"] == m and r["status"] != "skipped"]
        ok = [r for r in sel if r["status"] == "ok"]
        ax1.plot([r["n"] for r in sel], [r["mean_time_s"] for r in sel], marker="o", color=COLORS[m], label=m)
        ax2.plot([r["n"] for r in ok], [r["top1"] for r in ok], marker="o", color=COLORS[m], label=m)
        ax2.fill_between([r["n"] for r in ok], [r["top1_lo"] for r in ok], [r["top1_hi"] for r in ok],
                         color=COLORS[m], alpha=0.15, linewidth=0)
        for r in sel:
            if r["status"].startswith("stopped"):
                ax1.annotate("time limit hit,\nexact stopped", (r["n"], r["mean_time_s"]), textcoords="offset points",
                             xytext=(-12, -14), ha="right", va="top", fontsize=10)
    ax1.axhline(TIME_LIMIT_S, color="#8a8984", linestyle=":", linewidth=1.2)
    ax1.text(SIZES[0], TIME_LIMIT_S * 1.2, f"time limit {TIME_LIMIT_S} s per incident", ha="left", fontsize=9, color="#52514e")
    ax1.set_yscale("log")
    ax1.set_xlabel("number of services N")
    ax1.set_ylabel("seconds per incident")
    ax1.set_title(f"Run time (LW uses {N_SAMPLES} samples)")
    ax1.legend(loc="lower right")
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("number of services N")
    ax2.set_ylabel("top-1 accuracy")
    ax2.set_title("Accuracy")
    fig.suptitle(f"Scaling with graph size (simulated, {n_incidents} incidents per size, 95% CI)")
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    run(args.out, args.quick)
