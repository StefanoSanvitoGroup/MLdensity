#!/usr/bin/env python
"""Benchmark the serial vs parallel JLGridFingerprints on a real aluminium frame.

Times :class:`jlgridfingerprints.fingerprints.JLGridFingerprints` (serial)
against :class:`jlgridfingerprints.fast_fingerprints.JLGridFingerprints` and
prints wall-time + speedup tables. Descriptor construction only -- no model, no
CHGCAR I/O.

Two sweeps, selected with ``--sweep``:

``procs``
    Fixed number of centers, varying ``num_proc``. The scaling curve, directly
    comparable to ``scripts/benchmark_predictors``.
``centers``
    Fixed ``num_proc``, varying number of centers. Locates the *crossover*: the
    point set below which process-spawn and pickling overhead outweighs the
    parallel work. This matters because the training-data pipelines
    (``*/ml_model/data_ml/create_data.py``) sample only ~0.5% of the voxels,
    which may well sit below it.
``both``
    Run both (default).

Unlike ``scripts/benchmark_predictors``, ``OMP_NUM_THREADS`` is *not* required
here: as of v0.1.5 the Cython kernels no longer link OpenMP. The value is still
recorded in the JSON so old and new runs stay comparable.

Memory note: the descriptor matrix is dense ``(n_centers, n_features)`` float64
(~1 kB per center for these settings), so ``--n-centers 2744000`` needs ~2.6 GB
resident plus the per-batch copies. Size the job accordingly.

Run from anywhere; defaults to the repo's bundled aluminium inputs. Requires the
compiled Cython extensions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from ase.io import read

from jlgridfingerprints.fast_fingerprints import JLGridFingerprints as FastFingerprints
from jlgridfingerprints.fingerprints import JLGridFingerprints as SerialFingerprints
from jlgridfingerprints.tools import create_grid_coords

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

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POSCAR = REPO / "aluminium/data_scf/test_scf/10/POSCAR"


def sample_centers(atoms, n_centers: int, grid: tuple[int, int, int]) -> np.ndarray:
    """Take the first ``n_centers`` points of the ``grid`` in Cartesian coords.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure providing the cell.
    n_centers : int
        Number of centers to return. Capped at the grid size.
    grid : tuple of int
        FFT grid the centers are drawn from.

    Returns
    -------
    numpy.ndarray
        ``(n_centers, 3)`` Cartesian coordinates.
    """
    coords = create_grid_coords(
        grid_size=grid,
        return_cartesian_coords=True,
        a_vectors=atoms.get_cell().array,
    )
    return coords[:n_centers]


def time_create(jl, atoms, centers, batch_size=None, num_proc=1) -> float:
    """Return the wall time of one ``create`` call, in seconds."""
    t0 = time.perf_counter()
    if isinstance(jl, FastFingerprints):
        jl.create(atoms, centers, batch_size=batch_size, num_proc=num_proc)
    else:
        jl.create(atoms, centers)
    return time.perf_counter() - t0


def sweep_procs(atoms, centers, num_proc, batch_size) -> list[dict]:
    """Time the serial class, then the fast class at each ``num_proc``."""
    serial_t = time_create(SerialFingerprints(**SETTINGS), atoms, centers)
    runs = [{"num_proc": 0, "wall_s": serial_t, "speedup": 1.0, "label": "serial"}]

    print(f"\n{len(centers)} centers, batch_size={batch_size}")
    print(f"{'variant':<28}{'wall (s)':>12}{'speedup':>12}")
    print("-" * 52)
    print(f"{'serial JLGridFingerprints':<28}{serial_t:>12.3f}{1.0:>12.2f}")

    fast = FastFingerprints(**SETTINGS)
    for nproc in num_proc:
        dt = time_create(fast, atoms, centers, batch_size=batch_size, num_proc=nproc)
        print(f"{'fast num_proc=' + str(nproc):<28}{dt:>12.3f}{serial_t / dt:>12.2f}")
        runs.append(
            {
                "num_proc": nproc,
                "wall_s": dt,
                "speedup": serial_t / dt,
                "label": f"fast num_proc={nproc}",
            }
        )
    return runs


def sweep_centers(atoms, sizes, grid, num_proc, batch_frac) -> list[dict]:
    """Time serial vs fast at each point-set size, to find the crossover."""
    print(f"\ncrossover sweep, num_proc={num_proc}")
    print(f"{'n_centers':>12}{'serial (s)':>14}{'fast (s)':>12}{'speedup':>12}")
    print("-" * 50)

    fast = FastFingerprints(**SETTINGS)
    runs = []
    for n in sizes:
        centers = sample_centers(atoms, n, grid)
        # Keep the batch count proportional so every size uses all the workers.
        batch_size = max(1, len(centers) // (num_proc * batch_frac))
        serial_t = time_create(SerialFingerprints(**SETTINGS), atoms, centers)
        fast_t = time_create(fast, atoms, centers, batch_size, num_proc)
        print(
            f"{len(centers):>12}{serial_t:>14.3f}{fast_t:>12.3f}"
            f"{serial_t / fast_t:>12.2f}"
        )
        runs.append(
            {
                "n_centers": len(centers),
                "batch_size": batch_size,
                "serial_s": serial_t,
                "fast_s": fast_t,
                "speedup": serial_t / fast_t,
            }
        )
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poscar", type=Path, default=DEFAULT_POSCAR)
    p.add_argument(
        "--grid",
        type=int,
        nargs=3,
        default=(140, 140, 140),
        metavar=("NX", "NY", "NZ"),
        help="FFT grid the centers are drawn from (aluminium example: 140^3).",
    )
    p.add_argument("--sweep", choices=("procs", "centers", "both"), default="both")
    p.add_argument(
        "--n-centers",
        type=int,
        default=500000,
        help="Point-set size for the --sweep procs run.",
    )
    p.add_argument("--num-proc", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32))
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument(
        "--centers-sizes",
        type=int,
        nargs="+",
        default=(1000, 5000, 13000, 50000, 200000, 500000),
        help="Point-set sizes for the --sweep centers run. 13000 ~ one "
        "create_data.py frame.",
    )
    p.add_argument(
        "--centers-num-proc",
        type=int,
        default=8,
        help="num_proc held fixed during the --sweep centers run.",
    )
    p.add_argument(
        "--batches-per-proc",
        type=int,
        default=4,
        help="Batches per worker in the --sweep centers run, so every size "
        "keeps all workers busy.",
    )
    p.add_argument(
        "--json", type=Path, default=None, help="Write the results to this JSON file."
    )
    args = p.parse_args()

    atoms = read(str(args.poscar))
    grid = tuple(args.grid)
    n_features = SerialFingerprints(**SETTINGS)._n_features
    print(f"Structure: {args.poscar}  ({len(atoms)} atoms)")
    print(f"Grid: {grid[0]}x{grid[1]}x{grid[2]} = {np.prod(grid)} points")
    print(f"Features per center: {n_features}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} (no longer relevant)")

    payload = {
        "host": platform.node(),
        "n_features": n_features,
        "grid": list(grid),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "n_atoms": len(atoms),
        "poscar": str(args.poscar),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
    }

    if args.sweep in ("procs", "both"):
        centers = sample_centers(atoms, args.n_centers, grid)
        payload["procs_sweep"] = {
            "n_centers": len(centers),
            "batch_size": args.batch_size,
            "runs": sweep_procs(atoms, centers, args.num_proc, args.batch_size),
        }

    if args.sweep in ("centers", "both"):
        payload["centers_sweep"] = {
            "num_proc": args.centers_num_proc,
            "batches_per_proc": args.batches_per_proc,
            "runs": sweep_centers(
                atoms,
                args.centers_sizes,
                grid,
                args.centers_num_proc,
                args.batches_per_proc,
            ),
        }

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.json}", flush=True)


if __name__ == "__main__":
    main()
