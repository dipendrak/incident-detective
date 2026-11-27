"""Redraw the results as compact figures sized for the 4-page paper.

Reads the CSVs that run_all.py writes to results/ (it does not rerun any
experiment) and saves PDF figures to paper/figures/.

    python experiments/paper_figures.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# `common` must be imported first: it puts the repo root on sys.path.
from common import COLORS, DEFAULT_OUT, METHODS, ROOT

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter

TEXT_WIDTH = 5.5  # inches, the NeurIPS text width
LABELS = {
    "exact": "Exact inference",
    "likelihood weighting": "Likelihood weighting",
    "most upstream": "Most upstream alarm",
    "most alarmed descendants": "Most alarmed descendants",
    "random": "Random",
}


def paper_style() -> None:
    """Small serif fonts so the figures match the Times text of the paper."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def read_csv(path: Path) -> list[dict]:
    """Load a CSV, turning every numeric field into a float."""
    rows = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def accuracy_figure(res: Path, out: Path) -> None:
    """Top-1 accuracy of every method as false and missed alarms increase."""
    rows = read_csv(res / "accuracy_vs_noise.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.85), sharey=True)
    panels = [("fp", "fp_rate", "False-alarm rate (missed-alarm rate 0.1)"),
              ("fn", "fn_rate", "Missed-alarm rate (false-alarm rate 0.05)")]
    for ax, (sweep, xkey, xlabel) in zip(axes, panels):
        for m in METHODS:
            sel = [r for r in rows if r["sweep"] == sweep and r["method"] == m]
            x = [r[xkey] for r in sel]
            ax.plot(x, [r["top1"] for r in sel], marker="o", color=COLORS[m], label=LABELS[m],
                    linestyle="--" if m == "random" else "-")
            ax.fill_between(x, [r["top1_lo"] for r in sel], [r["top1_hi"] for r in sel],
                            color=COLORS[m], alpha=0.12, linewidth=0)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Top-1 accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.95),
               columnspacing=1.0, handlelength=1.8)
    fig.savefig(out / "accuracy.pdf")
    plt.close(fig)


def cost_figure(res: Path, out: Path) -> None:
    """Left: sampling error vs number of samples. Right: run time vs graph size."""
    conv = read_csv(res / "sampling_convergence.csv")
    scale = read_csv(res / "scaling.csv")
    color = COLORS["likelihood weighting"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.75))

    x = [r["n_samples"] for r in conv]
    ax1.plot(x, [r["tv_mean"] for r in conv], marker="o", color=color)
    ax1.fill_between(x, [r["tv_lo"] for r in conv], [r["tv_hi"] for r in conv], color=color, alpha=0.2, linewidth=0)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of samples")
    ax1.set_ylabel("TV distance to exact")
    ax1.set_title("(a) Likelihood weighting error, N = 12")

    for m in ("exact", "likelihood weighting"):
        sel = [r for r in scale if r["method"] == m and r["status"] != "skipped"]
        ax2.plot([r["n"] for r in sel], [r["mean_time_s"] * 1000 for r in sel], marker="o",
                 color=COLORS[m], label=LABELS[m])
    limit_ms = 100
    ax2.axhline(limit_ms, color="#8a8984", linestyle=":", linewidth=1)
    ax2.text(6, limit_ms * 1.25, "time limit", fontsize=7, color="#52514e")
    ax2.set_yscale("log")
    ax2.set_yticks([0.1, 1, 10, 100])
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("Number of services N")
    ax2.set_ylabel("Milliseconds per incident")
    ax2.set_title("(b) Run time")
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), handlelength=1.2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "cost.pdf")
    plt.close(fig)


def failure_figure(res: Path, out: Path) -> None:
    """Effective sample size and posterior error, grouped by number of alarms."""
    rows = read_csv(res / "failure_case.csv")
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    colors = [COLORS["exact"], COLORS["likelihood weighting"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.75))
    for name, color in zip(settings, colors):
        sel = [r for r in rows if r["setting"] == name]
        x = [r["alarms"] for r in sel]
        label = name.replace(" noise", "").capitalize()  # e.g. "Default (fp 0.05, fn 0.1)"
        ax1.plot(x, [r["median_ess"] for r in sel], marker="o", color=color, label=label)
        ax2.plot(x, [r["mean_tv"] for r in sel], marker="o", color=color, label=label)
    ax1.set_yscale("log")
    ax1.set_yticks([5, 10, 20, 50, 100])
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_ylabel("Median ESS (of 1,000)")
    ax1.set_title("(a) Effective sample size")
    ax2.set_ylabel("Mean TV distance to exact")
    ax2.set_title("(b) Posterior error")
    for ax in (ax1, ax2):
        ax.set_xlabel("Number of alarms in the incident")
    ax2.legend(loc="upper left", handlelength=1.2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out / "failure.pdf")
    plt.close(fig)


def main() -> None:
    """Draw all paper figures from the CSVs in results/."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_OUT, help="folder with the experiment CSVs")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "figures", help="where to save the PDFs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paper_style()
    accuracy_figure(args.results, args.out)
    cost_figure(args.results, args.out)
    failure_figure(args.results, args.out)
    print(f"Paper figures written to {args.out}")


if __name__ == "__main__":
    main()
