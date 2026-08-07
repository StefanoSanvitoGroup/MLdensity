#!/usr/bin/env python
"""Plot a JLGridFingerprints sweep from a benchmark_fingerprints.py JSON file.

Left panel: speedup vs ``num_proc`` against the ideal linear line.
Right panel: speedup vs point-set size, with the break-even line at 1.0 --
where the curve crosses it is the smallest point set worth parallelising.

Panels are drawn only for the sweeps present in the JSON.

Usage:
    python scripts/benchmark_fingerprints/plot_speedup.py results.json --out speedup.png
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
    procs = data.get("procs_sweep")
    centers = data.get("centers_sweep")
    panels = [s for s in (procs, centers) if s is not None]
    if not panels:
        raise SystemExit("JSON contains neither procs_sweep nor centers_sweep")

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5))
    axes = [axes] if len(panels) == 1 else list(axes)
    ax = iter(axes)

    if procs is not None:
        a = next(ax)
        # num_proc == 0 is the serial baseline; the rest are the fast sweep.
        fast = [r for r in procs["runs"] if r["num_proc"] > 0]
        nproc = [r["num_proc"] for r in fast]
        a.plot(nproc, nproc, "--", color="gray", label="ideal (linear)")
        a.plot(nproc, [r["speedup"] for r in fast], "o-", color="C0", label="measured")
        a.set_xscale("log", base=2)
        a.set_yscale("log", base=2)
        a.set_xlabel("num_proc (worker processes)")
        a.set_ylabel("speedup vs serial")
        a.set_title(
            f"Strong scaling\n{procs['n_centers']:,} centers, "
            f"batch_size={procs['batch_size']:,}"
        )
        a.legend()
        a.grid(True, which="both", alpha=0.3)

    if centers is not None:
        a = next(ax)
        runs = centers["runs"]
        a.axhline(1.0, ls="--", color="gray", label="break-even")
        a.plot(
            [r["n_centers"] for r in runs],
            [r["speedup"] for r in runs],
            "o-",
            color="C1",
            label="measured",
        )
        a.set_xscale("log")
        a.set_xlabel("centers per create() call")
        a.set_ylabel("speedup vs serial")
        a.set_title(f"Crossover\nnum_proc={centers['num_proc']}")
        a.legend()
        a.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"JLGridFingerprints.create parallel scaling — {data.get('host', 'unknown')}"
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
