"""Smoke tests for the fingerprint + predictor pipeline.

These require the compiled Cython extensions; if they are not built (e.g. on a
host where only a wheel-less checkout is present) the module is skipped. They
are exercised for real in CI, which builds the extensions first.
"""

import pickle
import warnings

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


# --- Absent configured species -------------------------------------------------
#
# A configured species with no atom in the structure must give identically-zero
# blocks rather than an error. The layout these tests index into is not exposed
# by the library, so it is rebuilt here from the same rules create_2b_jl and
# create_3b_jl concatenate by; that is deliberate, since the layout is what is
# under test.

WIDE = ["Al", "Cu", "Ni"]
NARROW = ["Al", "Cu"]


def _block_slices(jl):
    """Map each block of the descriptor vector to its column slice.

    Parameters
    ----------
    jl : JLGridFingerprints
        A constructed descriptor, whose ``species`` order fixes the layout.

    Returns
    -------
    dict
        Keys are ``frozenset`` of the chemical symbols the block involves --
        one symbol for a 1B block, one or two for a 2B block -- prefixed by the
        term name, e.g. ``("1b", frozenset({"Al"}))`` and
        ``("2b", frozenset({"Al", "Ni"}))``. Values are ``slice`` objects.
    """

    slices = {}
    offset = 0

    for symbol in jl.species:
        slices[("1b", frozenset({symbol}))] = slice(offset, offset + jl._nmax_1b)
        offset += jl._nmax_1b

    for ispec, jspec in jl.species_pair_index:
        if ispec == jspec:
            width = jl._n_upper_terms * (jl._lmax + 1)
        else:
            width = jl._n_full_terms * (jl._lmax + 1)
        key = ("2b", frozenset({jl.species[ispec], jl.species[jspec]}))
        slices[key] = slice(offset, offset + width)
        offset += width

    assert offset == jl._n_features
    return slices


@pytest.fixture
def alcu():
    """Two-element cell: the minimal stand-in for a single-impurity embedding."""
    atoms = bulk("Al", "fcc", a=4.05).repeat((2, 1, 1))
    atoms.symbols[1] = "Cu"
    return atoms


def _centers(atoms):
    return create_grid_coords(
        grid_size=(4, 4, 4),
        return_cartesian_coords=True,
        a_vectors=atoms.get_cell().array,
    )


def test_absent_species_gives_zero_blocks(alcu):
    settings = {k: v for k, v in SETTINGS.items() if k != "species"}
    centers = _centers(alcu)

    wide = JLGridFingerprints(species=WIDE, **settings)
    with pytest.warns(UserWarning, match="Ni"):
        X_wide = wide.create(alcu, centers)

    assert X_wide.shape == (len(centers), wide._n_features)
    assert np.isfinite(X_wide).all()

    narrow = JLGridFingerprints(species=NARROW, **settings)
    X_narrow = narrow.create(alcu, centers)

    wide_slices = _block_slices(wide)
    narrow_slices = _block_slices(narrow)

    for key, sl in wide_slices.items():
        _term, symbols = key
        if "Ni" in symbols:
            # Absent species: the block must be exactly zero, not merely small.
            assert np.count_nonzero(X_wide[:, sl]) == 0, f"{key} is not zero"
        else:
            # Present species: bitwise unchanged by the wider configuration.
            assert np.array_equal(X_wide[:, sl], X_narrow[:, narrow_slices[key]]), (
                f"{key} differs between the wide and narrow configurations"
            )

    # Every column is accounted for: nothing outside the Ni blocks is zeroed.
    assert np.count_nonzero(X_wide) > 0


def test_all_but_one_species_absent():
    """Two of three configured species absent -- the ordinary impurity case."""
    atoms = bulk("Al", "fcc", a=4.05)
    settings = {k: v for k, v in SETTINGS.items() if k != "species"}
    centers = _centers(atoms)

    jl = JLGridFingerprints(species=WIDE, **settings)
    with pytest.warns(UserWarning, match="Cu"):
        X = jl.create(atoms, centers)

    assert np.isfinite(X).all()

    for key, sl in _block_slices(jl).items():
        if key[1] == frozenset({"Al"}):
            assert np.count_nonzero(X[:, sl]) > 0, f"{key} should not be zero"
        else:
            assert np.count_nonzero(X[:, sl]) == 0, f"{key} is not zero"


def test_warn_absent_species_can_be_disabled(alcu):
    """The toggle silences the warning without touching the descriptor.

    A ``warnings`` filter is not equivalent for callers of
    ``fast_fingerprints``, whose spawned workers do not inherit filter state,
    which is why this is a constructor argument rather than documentation.
    """
    settings = {k: v for k, v in SETTINGS.items() if k != "species"}
    centers = _centers(alcu)

    with pytest.warns(UserWarning, match="Ni"):
        loud = JLGridFingerprints(species=WIDE, **settings).create(alcu, centers)

    quiet_jl = JLGridFingerprints(species=WIDE, warn_absent_species=False, **settings)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        quiet = quiet_jl.create(alcu, centers)

    assert np.array_equal(loud, quiet)
