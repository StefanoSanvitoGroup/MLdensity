"""Smoke tests for the fingerprint + predictor pipeline.

These require the compiled Cython extensions; if they are not built (e.g. on a
host where only a wheel-less checkout is present) the module is skipped. They
are exercised for real in CI, which builds the extensions first.
"""

import json
import pickle

import numpy as np
import pytest

# Skip the whole module unless the compiled extensions import.
pytest.importorskip("jlgridfingerprints.lib.jlcontraction")

from ase.build import bulk  # noqa: E402
from jlgridfingerprints.lib.polynomials import expand_jacobi  # noqa: E402
from scipy.special import eval_jacobi  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from jlgridfingerprints.fast_fingerprints import (  # noqa: E402
    JLGridFingerprints as FastJLGridFingerprints,
)
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

# 2B-term exponents from SETTINGS, used directly by the radial_map / slope_shifted
# tests below so they exercise the exponents an actual example pipeline uses.
AL_ALPHA_2B = SETTINGS["alpha"][1]
AL_BETA_2B = SETTINGS["beta"][1]
AL_NMAX_2B = SETTINGS["nmax"][1]
RCUT = SETTINGS["rcut"]

# An arbitrary softening length (Angstrom) for radial_map="log", well below RCUT so
# the map has room to be logarithmic before its linear region takes over near r = 0.
RSOFT = 2.121e-4

# A regression fixture for the double_shifted anchor fix, captured once from this
# implementation at gamma = 1.0 (r = [0, rcut/3, rcut], nmax = 4, AL_ALPHA_2B/BETA_2B).
# Frozen here so a future refactor of the anchor cannot silently change gamma = 1
# behaviour without this test catching it -- see the fix's own commit message for
# why gamma = 1 is the one value at which the pre-fix and fixed anchors coincide.
GOLDEN_DOUBLE_SHIFT_GAMMA1 = np.array(
    [
        [0.0, -11.581021266349028, 0.0],
        [0.0, -44.003327520891105, 0.0],
        [0.0, -139.31777932824966, 0.0],
    ]
)


@pytest.fixture
def atoms():
    return bulk("Al", "fcc", a=4.05)


@pytest.fixture
def centers(atoms):
    return create_grid_coords(
        grid_size=(4, 4, 4),
        return_cartesian_coords=True,
        a_vectors=atoms.get_cell().array,
    )


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


# --------------------------------------------------------------------------- #
# radial_map: the logarithmic map added alongside the existing cosine map.
# --------------------------------------------------------------------------- #


def test_cosine_map_is_default(atoms, centers):
    default = JLGridFingerprints(**SETTINGS).create(atoms, centers)
    explicit = JLGridFingerprints(**dict(SETTINGS, radial_map="cosine")).create(
        atoms, centers
    )
    assert np.array_equal(default, explicit)


def test_log_map_endpoints():
    # expand_jacobi does not expose the mapped x itself, so check the map's
    # endpoints indirectly: the map contract says r=0 -> x=+gamma and
    # r=rcut -> x=-gamma, so the *raw* (unshifted) Jacobi values there must
    # equal an independent evaluation of the same polynomial at x = +-gamma.
    r = np.ascontiguousarray([0.0, RCUT])
    raw = expand_jacobi(
        r,
        AL_NMAX_2B,
        AL_ALPHA_2B,
        AL_BETA_2B,
        RCUT,
        0.0,
        1.0,
        shifted=0,
        double_shifted=0,
        slope_shifted=0,
        radial_map="log",
        rsoft=RSOFT,
    )
    expected = np.array(
        [
            [eval_jacobi(n, AL_ALPHA_2B, AL_BETA_2B, x) for x in (1.0, -1.0)]
            for n in range(1, AL_NMAX_2B + 1)
        ]
    )
    assert np.allclose(raw, expected, atol=1e-10)


def test_log_map_monotone():
    # Order 1 is affine in x (see jacobi_eval's deg==1 branch), so it is
    # monotonic in x for alpha, beta > -1; checking it is monotonic in r is
    # therefore a check that the map x(r) itself is monotonic, without needing
    # to invert the map to recover x directly.
    r = np.ascontiguousarray(np.linspace(1e-6, RCUT, 500))
    order1 = expand_jacobi(
        r,
        AL_NMAX_2B,
        AL_ALPHA_2B,
        AL_BETA_2B,
        RCUT,
        0.0,
        1.0,
        shifted=0,
        radial_map="log",
        rsoft=RSOFT,
    )[0]
    diffs = np.diff(order1)
    assert np.all(diffs < 0) or np.all(diffs > 0)


def test_log_map_linear_limit():
    # As rsoft -> infinity, log1p(u)/log1p(v) -> u/v, so the map becomes
    # linear in r: x -> gamma*(1 - 2*r/rcut). Check the implementation
    # approaches that limit, via an independent scipy evaluation at the
    # limiting x.
    r = np.ascontiguousarray(np.linspace(0.0, RCUT, 2001))
    rsoft = 1.0e6
    raw = expand_jacobi(
        r,
        1,
        AL_ALPHA_2B,
        AL_BETA_2B,
        RCUT,
        0.0,
        1.0,
        shifted=0,
        radial_map="log",
        rsoft=rsoft,
    )[0]
    x_linear = 1.0 - 2.0 * r / RCUT
    expected = eval_jacobi(1, AL_ALPHA_2B, AL_BETA_2B, x_linear)
    assert np.abs(raw - expected).max() < 1e-5


def test_log_map_rejects_bad_settings():
    r = np.ascontiguousarray([1.0])
    with pytest.raises(ValueError, match="requires a positive 'rsoft'"):
        expand_jacobi(
            r, AL_NMAX_2B, AL_ALPHA_2B, AL_BETA_2B, RCUT, radial_map="log", rsoft=0.0
        )
    with pytest.raises(ValueError, match="requires rmin == 0"):
        expand_jacobi(
            r,
            AL_NMAX_2B,
            AL_ALPHA_2B,
            AL_BETA_2B,
            RCUT,
            -0.74,
            radial_map="log",
            rsoft=1e-3,
        )
    with pytest.raises(ValueError, match="has no meaning for radial_map='cosine'"):
        expand_jacobi(
            r,
            AL_NMAX_2B,
            AL_ALPHA_2B,
            AL_BETA_2B,
            RCUT,
            radial_map="cosine",
            rsoft=1e-3,
        )
    with pytest.raises(ValueError, match="Unknown 'radial_map'"):
        expand_jacobi(r, AL_NMAX_2B, AL_ALPHA_2B, AL_BETA_2B, RCUT, radial_map="bogus")


# --------------------------------------------------------------------------- #
# double_shifted: the upper-anchor fix (x = +gamma, not the hard-coded x = +1).
# --------------------------------------------------------------------------- #


def test_double_shift_vanishes_both_ends():
    # The anchor fix operates on x, not on the map, so both endpoints should
    # vanish under either map and any gamma -- not just the cosine map at the
    # default gamma = 1, which is all the published examples exercise.
    r = np.ascontiguousarray([0.0, RCUT])
    for gamma in (0.5, 1.0, 2.0):
        for kwargs in (
            dict(radial_map="cosine"),
            dict(radial_map="log", rsoft=RSOFT),
        ):
            vals = expand_jacobi(
                r,
                AL_NMAX_2B,
                AL_ALPHA_2B,
                AL_BETA_2B,
                RCUT,
                0.0,
                gamma,
                shifted=1,
                double_shifted=1,
                **kwargs,
            )
            assert np.abs(vals).max() < 1e-9, (gamma, kwargs)


def test_double_shift_unchanged_at_unit_gamma():
    r = np.ascontiguousarray([0.0, RCUT / 3.0, RCUT])
    ours = expand_jacobi(
        r, 4, AL_ALPHA_2B, AL_BETA_2B, RCUT, 0.0, 1.0, shifted=1, double_shifted=1
    )
    assert np.array_equal(ours, GOLDEN_DOUBLE_SHIFT_GAMMA1)


# --------------------------------------------------------------------------- #
# Propagation: the two new keywords must reach every consumer of __init__.
# --------------------------------------------------------------------------- #


def test_log_map_serial_parallel_agree(atoms, centers):
    settings = dict(
        SETTINGS, double_shifted=False, radial_map="log", rsoft=RSOFT, rmin=0.0
    )
    serial = FastJLGridFingerprints(**settings).create(atoms, centers)
    parallel = FastJLGridFingerprints(**settings).create(
        atoms, centers, batch_size=8, num_proc=2
    )
    assert np.array_equal(serial, parallel)


def test_log_map_predictor_roundtrip(atoms, centers, tmp_path):
    settings = dict(
        SETTINGS, double_shifted=False, radial_map="log", rsoft=RSOFT, rmin=0.0
    )
    jl = JLGridFingerprints(**settings)
    X = jl.create(atoms, centers)

    model = Ridge(fit_intercept=False).fit(X, np.ones(len(X)))
    model_path = tmp_path / "model.p"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(settings))

    predictor = JLPredictor(
        jl_settings=str(settings_path), model_path=str(model_path), grid_size=(4, 4, 4)
    )
    chg = predictor.predict_chgcar(
        atoms, nelect=1.0, write_chgcar=False, return_chg=True
    )
    assert chg.shape == (4, 4, 4)
    assert np.isfinite(chg).all()


# --------------------------------------------------------------------------- #
# slope_shifted: value AND first derivative vanishing at rcut.
# --------------------------------------------------------------------------- #


def test_slope_shift_double_root():
    eps = np.array([1e-3, 1e-4, 1e-5])
    r = np.ascontiguousarray(RCUT - eps)
    common = dict(
        shifted=1, double_shifted=0, slope_shifted=1, radial_map="log", rsoft=RSOFT
    )
    vals = expand_jacobi(
        r, AL_NMAX_2B, AL_ALPHA_2B, AL_BETA_2B, RCUT, 0.0, 1.0, **common
    )
    at_rcut = expand_jacobi(
        np.ascontiguousarray([RCUT]),
        AL_NMAX_2B,
        AL_ALPHA_2B,
        AL_BETA_2B,
        RCUT,
        0.0,
        1.0,
        **common,
    )
    assert np.abs(at_rcut).max() < 1e-9

    mags = np.abs(vals).max(axis=0)
    # A double root (value AND slope vanish) falls ~100x per decade of
    # approach; a single root (value alone, i.e. `shifted` without
    # `slope_shifted`) would fall only ~10x. A tolerance on |f| cannot tell
    # the two apart -- only the scaling can.
    assert 90.0 < mags[0] / mags[1] < 110.0
    assert 90.0 < mags[1] / mags[2] < 110.0


def test_slope_shift_block_width():
    r = np.ascontiguousarray(np.linspace(0.0, RCUT, 10))
    common = dict(rmin=0.0, gamma=1.0, shifted=1, radial_map="log", rsoft=RSOFT)
    for nmax, expected_shifted in ((6, 6), (15, 15)):
        shifted = expand_jacobi(
            r,
            nmax,
            AL_ALPHA_2B,
            AL_BETA_2B,
            RCUT,
            double_shifted=0,
            slope_shifted=0,
            **common,
        )
        slope = expand_jacobi(
            r,
            nmax,
            AL_ALPHA_2B,
            AL_BETA_2B,
            RCUT,
            double_shifted=0,
            slope_shifted=1,
            **common,
        )
        slope_plus1 = expand_jacobi(
            r,
            nmax + 1,
            AL_ALPHA_2B,
            AL_BETA_2B,
            RCUT,
            double_shifted=0,
            slope_shifted=1,
            **common,
        )
        assert shifted.shape[0] == expected_shifted
        assert slope.shape[0] == expected_shifted - 1
        assert slope_plus1.shape[0] == expected_shifted


def test_slope_shift_rejects_bad_settings():
    r = np.ascontiguousarray([1.0])
    with pytest.raises(ValueError, match="mutually exclusive"):
        expand_jacobi(
            r,
            AL_NMAX_2B,
            AL_ALPHA_2B,
            AL_BETA_2B,
            RCUT,
            shifted=1,
            double_shifted=1,
            slope_shifted=1,
        )
    with pytest.raises(ValueError, match="requires 'shifted'"):
        expand_jacobi(
            r, AL_NMAX_2B, AL_ALPHA_2B, AL_BETA_2B, RCUT, shifted=0, slope_shifted=1
        )
    with pytest.raises(ValueError, match="nmax >= 2"):
        expand_jacobi(r, 1, AL_ALPHA_2B, AL_BETA_2B, RCUT, shifted=1, slope_shifted=1)


def test_slope_shift_off_is_bit_identical(atoms, centers):
    settings = dict(
        SETTINGS, double_shifted=False, radial_map="log", rsoft=RSOFT, rmin=0.0
    )
    default = JLGridFingerprints(**settings).create(atoms, centers)
    explicit = JLGridFingerprints(**dict(settings, slope_shifted=False)).create(
        atoms, centers
    )
    assert np.array_equal(default, explicit)
