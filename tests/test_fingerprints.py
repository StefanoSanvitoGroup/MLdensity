"""Smoke tests for the fingerprint + predictor pipeline.

These require the compiled Cython extensions; if they are not built (e.g. on a
host where only a wheel-less checkout is present) the module is skipped. They
are exercised for real in CI, which builds the extensions first.
"""

import pickle

import numpy as np
import pytest

# Skip the whole module unless the compiled extensions import.
pytest.importorskip("jlgridfingerprints.lib.jlcontraction")

from ase.build import bulk  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from jlgridfingerprints.fingerprints import JLGridFingerprints  # noqa: E402
from jlgridfingerprints.predictor import JLPredictor  # noqa: E402
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


@pytest.fixture
def atoms():
    return bulk("Al", "fcc", a=4.05)


def test_alpha_beta_at_domain_boundary_raises():
    settings = dict(SETTINGS, alpha=[7.875386069413652, -1.0])
    with pytest.raises(ValueError, match="alpha and beta must be > -1"):
        JLGridFingerprints(**settings)


def test_alpha_beta_below_domain_boundary_raises():
    settings = dict(SETTINGS, beta=[3.6238075908648106, -2.5])
    with pytest.raises(ValueError, match="alpha and beta must be > -1"):
        JLGridFingerprints(**settings)


def test_fingerprint_shape(atoms):
    jl = JLGridFingerprints(**SETTINGS)
    centers = create_grid_coords(
        grid_size=(4, 4, 4),
        return_cartesian_coords=True,
        a_vectors=atoms.get_cell().array,
    )
    X = jl.create(atoms, centers)
    assert X.shape == (len(centers), jl._n_features)
    assert np.isfinite(X).all()


def test_predictor_roundtrip(atoms, tmp_path):
    jl = JLGridFingerprints(**SETTINGS)
    centers = create_grid_coords(
        grid_size=(4, 4, 4),
        return_cartesian_coords=True,
        a_vectors=atoms.get_cell().array,
    )
    X = jl.create(atoms, centers)

    # Trivial fitted model just to exercise the predict path.
    model = Ridge(fit_intercept=False).fit(X, np.ones(len(X)))
    model_path = tmp_path / "model.p"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    predictor = JLPredictor(
        jl_settings=SETTINGS, model_path=str(model_path), grid_size=(4, 4, 4)
    )
    chg = predictor.predict_chgcar(
        atoms, nelect=1.0, write_chgcar=False, return_chg=True
    )
    assert chg.shape == (4, 4, 4)
    assert np.isfinite(chg).all()
