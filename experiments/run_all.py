"""Regenerate every CSV and figure in results/ with one command.

    python experiments/run_all.py           # full run
    python experiments/run_all.py --quick   # smaller run, a few seconds
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# `common` must be imported first: it puts the repo root on sys.path.
from common import DEFAULT_OUT

import accuracy_vs_noise
import failure_case
import sampling_convergence
import scaling

EXPERIMENTS = [
    ("accuracy_vs_noise", accuracy_vs_noise.run),
    ("sampling_convergence", sampling_convergence.run),
    ("scaling", scaling.run),
    ("failure_case", failure_case.run),
]


def main() -> None:
    """Run all experiments in order and report timing."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer incidents and samples")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for name, fn in EXPERIMENTS:
        t0 = time.perf_counter()
        print(f"[{name}]")
        fn(args.out, args.quick)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f} s")
    print(f"All experiments finished in {time.perf_counter() - start:.1f} s. Output: {args.out}")


if __name__ == "__main__":
    main()
