#!/usr/bin/env python
"""Benchmark the serial vs parallel JLPredictor on a real aluminium frame.

Times :class:`jlgridfingerprints.predictor.JLPredictor` (serial) against
:class:`jlgridfingerprints.fast_predictor.JLPredictor` at a range of ``num_proc``
values and prints a wall-time + speedup table. Descriptor + prediction time only
(``write_chgcar=False``).

The parallel path only wins on full-size grids, where per-batch fingerprint work
dominates process-spawn/pickling overhead; on small grids it is slower. Use
``--grid 140 140 140`` (the aluminium example grid) for a representative number.

IMPORTANT: set ``OMP_NUM_THREADS=1`` before running with ``num_proc > 1``.
``JLGridFingerprints.create`` uses OpenMP internally, so without it each worker
process spawns its own thread pool and oversubscribes the cores, making the
parallel path *much* slower than serial (measured ~20-30x slower).

Run from anywhere; defaults to the repo's bundled aluminium inputs. Requires the
compiled Cython extensions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ase.io import read

from jlgridfingerprints.fast_predictor import JLPredictor as FastPredictor
from jlgridfingerprints.predictor import JLPredictor as SerialPredictor

# Aluminium example settings (matches aluminium/ml_model/predict_chgcar).
SETTINGS = {
    "rcut": 4.08,
    "nmax": [15, 6],
    "lmax": 6,
    "alpha": [7.875386069413652, 5.875090883472657],
    "beta": [3.6238075908648106, 1.7505953204305842],
    "rmin": -0.74,
    "species": ["Al"],
    "body": "1+2",
    "periodic": True,
    "double_shifted": True,
}

REPO = Path(__file__).resolve().parent.parent
DEFAULT_POSCAR = REPO / "aluminium/data_scf/test_scf/10/POSCAR"
DEFAULT_MODEL = REPO / "aluminium/ml_model/train_ml/scikit_linear_model_chg.p"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poscar", type=Path, default=DEFAULT_POSCAR)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument(
        "--grid",
        type=int,
        nargs=3,
        default=(56, 56, 56),
        metavar=("NX", "NY", "NZ"),
        help="FFT grid size. Use 140 140 140 for a representative full-grid number.",
    )
    p.add_argument("--num-proc", type=int, nargs="+", default=(1, 2, 4))
    p.add_argument("--batch-size", type=int, default=100000)
    p.add_argument("--nelect", type=float, default=96.0)
    p.add_argument(
        "--json", type=Path, default=None, help="Write the results to this JSON file."
    )
    args = p.parse_args()

    if max(args.num_proc) > 1 and os.environ.get("OMP_NUM_THREADS") != "1":
        print(
            "WARNING: OMP_NUM_THREADS is not 1. With num_proc > 1 each worker "
            "spawns its own OpenMP threads and oversubscribes the cores, which "
            "makes the parallel path much slower. Re-run with OMP_NUM_THREADS=1.\n",
            flush=True,
        )

    atoms = read(str(args.poscar))
    grid = tuple(args.grid)
    npts = grid[0] * grid[1] * grid[2]
    print(f"Structure: {args.poscar}  ({len(atoms)} atoms)")
    print(f"Grid: {grid[0]}x{grid[1]}x{grid[2]} = {npts} points\n")

    # Serial reference.
    serial = SerialPredictor(
        jl_settings=SETTINGS, model_path=str(args.model), grid_size=grid
    )
    t0 = time.time()
    serial.predict_chgcar(
        atoms, nelect=args.nelect, batch_size=args.batch_size, write_chgcar=False
    )
    serial_t = time.time() - t0

    fast = FastPredictor(
        jl_settings=SETTINGS, model_path=str(args.model), grid_size=grid
    )

    print(f"{'predictor':<28}{'wall (s)':>12}{'speedup':>12}")
    print("-" * 52)
    print(f"{'serial JLPredictor':<28}{serial_t:>12.3f}{1.0:>12.2f}")

    runs = [{"num_proc": 0, "wall_s": serial_t, "speedup": 1.0, "label": "serial"}]
    for nproc in args.num_proc:
        t0 = time.time()
        fast.predict_chgcar(
            atoms,
            nelect=args.nelect,
            batch_size=args.batch_size,
            num_proc=nproc,
            write_chgcar=False,
        )
        dt = time.time() - t0
        print(f"{'fast num_proc=' + str(nproc):<28}{dt:>12.3f}{serial_t / dt:>12.2f}")
        runs.append(
            {
                "num_proc": nproc,
                "wall_s": dt,
                "speedup": serial_t / dt,
                "label": f"fast num_proc={nproc}",
            }
        )

    if args.json is not None:
        payload = {
            "grid": list(grid),
            "n_points": npts,
            "batch_size": args.batch_size,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "n_atoms": len(atoms),
            "poscar": str(args.poscar),
            "runs": runs,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.json}", flush=True)


if __name__ == "__main__":
    main()
