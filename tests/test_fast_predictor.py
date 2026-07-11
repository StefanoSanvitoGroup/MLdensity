"""Equivalence tests for the parallel ``fast_predictor.JLPredictor``.

These assert the fast predictor produces the *same* density as the serial
:class:`jlgridfingerprints.predictor.JLPredictor` across its serial, batched and
multiprocessing paths. They make no timing claims: at this grid size the parallel
path is slower than serial (process overhead dominates), so the speedup is measured
separately in ``scripts/benchmark_predictors/benchmark_predictors.py``.

Like ``test_fingerprints.py`` these require the compiled Cython extensions and are
skipped if they are not built.
"""

import pickle

import numpy as np
import pytest

# Skip the whole module unless the compiled extensions import.
pytest.importorskip("jlgridfingerprints.lib.jlcontraction")

from ase.build import bulk  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from jlgridfingerprints.fast_predictor import JLPredictor as FastPredictor  # noqa: E402
from jlgridfingerprints.fingerprints import JLGridFingerprints  # noqa: E402
from jlgridfingerprints.predictor import JLPredictor as SerialPredictor  # noqa: E402
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
NELECT = 1.0


@pytest.fixture
def atoms():
    return bulk("Al", "fcc", a=4.05)


@pytest.fixture
def model_path(atoms, tmp_path):
    """A trivial fitted Ridge model, pickled, to exercise the predict path."""
    jl = JLGridFingerprints(**SETTINGS)
    centers = create_grid_coords(
        grid_size=GRID,
        return_cartesian_coords=True,
        a_vectors=atoms.get_cell().array,
    )
    X = jl.create(atoms, centers)
    model = Ridge(fit_intercept=False).fit(X, np.ones(len(X)))
    path = tmp_path / "model.p"
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return str(path)


@pytest.fixture
def serial_chg(atoms, model_path):
    """Reference density from the serial predictor (always normalises)."""
    predictor = SerialPredictor(
        jl_settings=SETTINGS, model_path=model_path, grid_size=GRID
    )
    return predictor.predict_chgcar(
        atoms, nelect=NELECT, write_chgcar=False, return_chg=True
    )


def test_fast_serial_path_matches_serial(atoms, model_path, serial_chg):
    fast = FastPredictor(jl_settings=SETTINGS, model_path=model_path, grid_size=GRID)
    chg = fast.predict_chgcar(
        atoms, nelect=NELECT, normalize=True, write_chgcar=False, return_chg=True
    )
    assert chg.shape == GRID
    assert np.allclose(chg, serial_chg)


def test_fast_batched_matches_serial(atoms, model_path, serial_chg):
    fast = FastPredictor(jl_settings=SETTINGS, model_path=model_path, grid_size=GRID)
    chg = fast.predict_chgcar(
        atoms,
        nelect=NELECT,
        batch_size=8,
        num_proc=1,
        normalize=True,
        write_chgcar=False,
        return_chg=True,
    )
    assert np.allclose(chg, serial_chg)


def test_fast_multiprocess_matches_serial(atoms, model_path, serial_chg):
    fast = FastPredictor(jl_settings=SETTINGS, model_path=model_path, grid_size=GRID)
    chg = fast.predict_chgcar(
        atoms,
        nelect=NELECT,
        batch_size=8,
        num_proc=2,
        normalize=True,
        write_chgcar=False,
        return_chg=True,
    )
    assert np.allclose(chg, serial_chg)


def test_predict_key_chgcar_matches_predict_chgcar(atoms, model_path):
    fast = FastPredictor(jl_settings=SETTINGS, model_path=model_path, grid_size=GRID)
    chg_full = fast.predict_chgcar(
        atoms, nelect=NELECT, write_chgcar=False, return_chg=True
    )
    centers = create_grid_coords(
        grid_size=GRID,
        return_cartesian_coords=True,
        a_vectors=atoms.get_cell().array,
    )
    chg_key = fast.predict_key_chgcar(
        atoms,
        cart_positions=centers,
        gridsize=GRID,
        nelect=NELECT,
        write_chgcar=False,
        return_chg=True,
    )
    assert np.allclose(chg_key, chg_full)
