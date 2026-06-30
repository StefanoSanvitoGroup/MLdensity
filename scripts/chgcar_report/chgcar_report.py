"""
Print summary statistics for a set of CHGCAR / POSCAR pairs.

For each matched folder, reports cell info (primitive lattice-vector length,
volume, number of atoms), grid info (shape, voxel size), and the per-grid-point
distribution of n = chgcar.data["total"] / volume — which is what
JLGridFingerprints' sample_charge actually sees.

Usage example (run from project root):

    python scripts/chgcar_report/chgcar_report.py 'aluminium/data_scf/train_scf/*'

(The pattern is a Python glob; quote it on the shell so the shell doesn't expand it itself).

Expected output:

folder                           nat  a_prim      vol           grid       voxel             n[min/mean/max]  |n|[1/25/50/75/99 %]
----------------------------------------------------------------------------------------------------------------------------------
0                                 32   8.078  527.141    140x140x140   1.921e-04        -0.011/+0.182/+0.265  0.0568/0.165/0.189/0.205/0.239
1                                 32   8.078  527.141    140x140x140   1.921e-04       -0.0107/+0.182/+0.267  0.0569/0.166/0.188/0.204/0.24
2                                 32   8.078  527.141    140x140x140   1.921e-04       -0.0107/+0.182/+0.284  0.0567/0.165/0.188/0.205/0.242
3                                 32   8.078  527.141    140x140x140   1.921e-04       -0.0109/+0.182/+0.276  0.0568/0.166/0.189/0.204/0.239
4                                 32   8.078  527.141    140x140x140   1.921e-04       -0.0108/+0.182/+0.257  0.0568/0.165/0.188/0.205/0.238
5                                 32   8.078  527.141    140x140x140   1.921e-04       -0.0107/+0.182/+0.272  0.0568/0.165/0.188/0.205/0.241
6                                 32   8.078  527.141    140x140x140   1.921e-04        -0.0108/+0.182/+0.27  0.0568/0.167/0.189/0.204/0.239
7                                 32   8.078  527.141    140x140x140   1.921e-04       -0.0107/+0.182/+0.273  0.0568/0.166/0.189/0.204/0.238
8                                 32   8.078  527.141    140x140x140   1.921e-04       -0.0106/+0.182/+0.266  0.0568/0.166/0.189/0.204/0.238
9                                 32   8.078  527.141    140x140x140   1.921e-04       -0.0105/+0.182/+0.261  0.0567/0.166/0.189/0.205/0.237
"""

import argparse
import glob
import os
import sys

import numpy as np
from ase.io import read
from pymatgen.io.vasp.outputs import Chgcar


def report_one(folder, chgcar_name):
    poscar_path = os.path.join(folder, "POSCAR")
    chgcar_path = os.path.join(folder, chgcar_name)
    if not (os.path.isfile(poscar_path) and os.path.isfile(chgcar_path)):
        return None
    atoms = read(poscar_path)
    chgcar = Chgcar.from_file(chgcar_path)
    raw = chgcar.data["total"]
    vol = atoms.get_volume()
    a_prim = np.linalg.norm(atoms.get_cell()[0])
    n = raw / vol
    abs_n = np.abs(n.ravel())
    p = np.percentile(abs_n, [1, 25, 50, 75, 99])
    return {
        "folder": os.path.basename(folder.rstrip("/")),
        "natoms": len(atoms),
        "a_prim": a_prim,
        "vol": vol,
        "grid": raw.shape,
        "voxel": vol / raw.size,
        "n_min": float(n.min()),
        "n_mean": float(n.mean()),
        "n_max": float(n.max()),
        "abs_n_pcts": p,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pattern",
        help="Glob matching folders, each containing POSCAR and a CHGCAR file",
    )
    parser.add_argument(
        "--chgcar-name",
        default="CHGCAR",
        help="Filename of the CHGCAR within each folder (default: CHGCAR)",
    )
    args = parser.parse_args()

    folders = sorted(f for f in glob.glob(args.pattern) if os.path.isdir(f))
    if not folders:
        print(f"No folders matched: {args.pattern}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in folders:
        r = report_one(f, args.chgcar_name)
        if r is None:
            print(
                f"Skip (missing POSCAR or {args.chgcar_name}): {f}",
                file=sys.stderr,
            )
            continue
        rows.append(r)

    if not rows:
        sys.exit(1)

    header = (
        f"{'folder':32s} {'nat':>3s} {'a_prim':>7s} {'vol':>8s} "
        f"{'grid':>14s} {'voxel':>11s}  "
        f"{'n[min/mean/max]':>26s}  |n|[1/25/50/75/99 %]"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        grid_str = "x".join(str(x) for x in r["grid"])
        n_str = f"{r['n_min']:+.3g}/{r['n_mean']:+.3g}/{r['n_max']:+.3g}"
        p = r["abs_n_pcts"]
        p_str = f"{p[0]:.3g}/{p[1]:.3g}/{p[2]:.3g}/{p[3]:.3g}/{p[4]:.3g}"
        print(
            f"{r['folder']:32s} {r['natoms']:>3d} {r['a_prim']:>7.3f} "
            f"{r['vol']:>8.3f} {grid_str:>14s} {r['voxel']:>11.3e}  "
            f"{n_str:>26s}  {p_str}"
        )


if __name__ == "__main__":
    main()
