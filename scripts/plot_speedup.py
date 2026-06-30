#!/usr/bin/env python
"""Plot a JLPredictor speedup sweep from a benchmark_predictors.py JSON file.

Usage:
    python scripts/plot_speedup.py reports/data/benchmark_th1-2020-64.json \
        --out reports/figures/speedup.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    data = json.loads(args.json.read_text())
    # num_proc == 0 is the serial baseline; the rest are the fast-predictor sweep.
    fast = [r for r in data["runs"] if r["num_proc"] > 0]
    nproc = [r["num_proc"] for r in fast]
    speedup = [r["speedup"] for r in fast]
    wall = [r["wall_s"] for r in fast]

    nx, ny, nz = data["grid"]
    title = (
        f"JLPredictor parallel speedup\n{nx}x{ny}x{nz} grid "
        f"({data['n_points']:,} points), OMP_NUM_THREADS={data['omp_num_threads']}"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(nproc, nproc, "--", color="gray", label="ideal (linear)")
    ax1.plot(nproc, speedup, "o-", color="C0", label="measured")
    ax1.set_xlabel("num_proc (worker processes)")
    ax1.set_ylabel("speedup vs serial")
    ax1.set_title("Speedup")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(nproc, wall, "o-", color="C1")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("num_proc (worker processes)")
    ax2.set_ylabel("wall time (s)")
    ax2.set_title("Wall time")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
