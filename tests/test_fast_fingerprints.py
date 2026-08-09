"""Equivalence tests for the parallel ``fast_fingerprints.JLGridFingerprints``.

These assert the fast descriptor produces the *same* array as the serial
:class:`jlgridfingerprints.fingerprints.JLGridFingerprints` across its serial,
batched and multiprocessing paths. Equality is exact: each row depends only on
its own center and the neighbours within ``rcut``, so batching cannot change a
single bit, and since v0.1.5 the Cython kernels no longer use OpenMP. A failure
here is a real bug, not floating-point noise.

They make no timing claims: at this size the parallel path is slower than serial
(process overhead dominates), so the speedup is measured separately in
``scripts/benchmark_fingerprints/benchmark_fingerprints.py``.

Like ``test_fingerprints.py`` these require the compiled Cython extensions and
are skipped if they are not built.
"""

import numpy as np
import pytest

# Skip the whole module unless the compiled extensions import.
pytest.importorskip("jlgridfingerprints.lib.jlcontraction")

from ase.build import bulk  # noqa: E402

from jlgridfingerprints.fast_fingerprints import (  # noqa: E402
    JLGridFingerprints as FastFingerprints,
)
from jlgridfingerprints.fingerprints import (  # noqa: E402
    JLGridFingerprints as SerialFingerprints,
)
from jlgridfingerprints.tools import create_grid_coords  # noqa: E402

# Known-good settings from the aluminium example pipeline.
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

GRID = (4, 4, 4)


@pytest.fixture
def atoms():
    return bulk("Al", "fcc", a=4.05)


@pytest.fixture
def centers(atoms):
    return create_grid_coords(
        grid_size=GRID,
        return_cartesian_coords=True,
        a_vectors=atoms.get_cell().array,
    )


@pytest.fixture
def serial_X(atoms, centers):
    """Reference descriptors from the serial class."""
    return SerialFingerprints(**SETTINGS).create(atoms, centers)


def test_serial_path_matches_serial(atoms, centers, serial_X):
    """num_proc=1, batch_size=None delegates straight to the parent."""
    X = FastFingerprints(**SETTINGS).create(atoms, centers)
    assert np.array_equal(X, serial_X)


def test_batched_matches_serial(atoms, centers, serial_X):
    """Batched but single-process: exercises the worker functions in-process."""
    X = FastFingerprints(**SETTINGS).create(atoms, centers, batch_size=8)
    assert np.array_equal(X, serial_X)


def test_multiprocess_matches_serial(atoms, centers, serial_X):
    """The real parallel path, including spawn, pickling and reassembly order."""
    X = FastFingerprints(**SETTINGS).create(atoms, centers, batch_size=8, num_proc=2)
    assert np.array_equal(X, serial_X)


def test_ragged_batch_matches_serial(atoms, centers, serial_X):
    """A batch_size that does not divide n_centers evenly."""
    assert len(centers) % 7 != 0
    X = FastFingerprints(**SETTINGS).create(atoms, centers, batch_size=7, num_proc=2)
    assert np.array_equal(X, serial_X)


def test_default_positions_matches_serial(atoms):
    """positions=None falls back to the atom positions, as the parent does."""
    X = FastFingerprints(**SETTINGS).create(atoms, batch_size=1, num_proc=1)
    assert np.array_equal(X, SerialFingerprints(**SETTINGS).create(atoms))


def test_num_proc_without_batch_size_raises(atoms, centers):
    with pytest.raises(ValueError, match="requires batch_size"):
        FastFingerprints(**SETTINGS).create(atoms, centers, num_proc=2)


def test_nonpositive_batch_size_raises(atoms, centers):
    with pytest.raises(ValueError, match="positive integer"):
        FastFingerprints(**SETTINGS).create(atoms, centers, batch_size=0)
