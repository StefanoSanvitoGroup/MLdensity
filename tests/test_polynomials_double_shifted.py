"""The ``double_shifted`` basis must vanish at both ends of its own interval.

``double_shifted`` removes a multiple of the linear order so that the radial basis
vanishes at the upper end of the mapped interval as well as the lower one. The
interval is ``[-gamma, +gamma]``, so both anchors have to follow ``gamma``. These
tests pin that at values of ``gamma`` away from the default, which is where the two
anchors stop coinciding and where the property used to be lost.

Integer ``alpha``/``beta`` throughout, deliberately: ``expand_jacobi`` still declares
those exponents as ``int``, so float values would be truncated toward zero and a test
written with them would not be exercising the exponents it names.
"""

import numpy as np
import pytest

# Skip the whole module unless the compiled extensions import.
pytest.importorskip("jlgridfingerprints.lib.polynomials")

from jlgridfingerprints.lib.polynomials import expand_jacobi  # noqa: E402

ALPHA, BETA, NMAX, RCUT = 5, 1, 6, 4.08
GAMMAS = (0.5, 0.75, 1.0, 1.5, 2.0)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_double_shifted_vanishes_at_both_endpoints(gamma):
    """Both ends of [0, rcut] map to the interval ends, so both must vanish."""
    r = np.ascontiguousarray([0.0, RCUT])
    basis = expand_jacobi(
        r, NMAX, ALPHA, BETA, RCUT, 0.0, gamma, shifted=1, double_shifted=1
    )
    assert np.abs(basis).max() < 1e-9


@pytest.mark.parametrize("gamma", (1.5, 2.0))
def test_double_shifted_has_no_spurious_interior_node(gamma):
    """The upper vanishing point must not sit at an interior radius.

    Anchoring at the fixed ``x = +1`` instead of ``x = +gamma`` put the upper
    vanishing point inside the reachable range whenever ``gamma > 1``, at
    ``r* = (rcut - rmin) / pi * arccos(1 / gamma) + rmin``. Every returned order was
    pinned to zero there, so a caller sweeping ``gamma`` was silently imposing a hard
    node at a fixed fraction of the cutoff rather than sweeping a smooth family.
    """
    frac = np.arccos(1.0 / gamma) / np.pi
    r_star = np.ascontiguousarray([frac * RCUT])
    basis = expand_jacobi(
        r_star, NMAX, ALPHA, BETA, RCUT, 0.0, gamma, shifted=1, double_shifted=1
    )
    assert np.abs(basis).max() > 1.0


def test_double_shifted_unchanged_at_unit_gamma():
    """gamma = 1 is the one value where the two anchors coincide.

    Everything published from this repository runs at the default gamma = 1 -- it is
    set in no example settings dict and in no other test -- so this is the case that
    has to stay bit-identical, and it is what makes this fix safe to apply
    unconditionally. Frozen with ``array_equal``, not ``allclose``: the claim is
    identity, not closeness.
    """
    r = np.ascontiguousarray([0.0, RCUT / 3.0, RCUT])
    basis = expand_jacobi(
        r, 4, ALPHA, BETA, RCUT, 0.0, 1.0, shifted=1, double_shifted=1
    )
    expected = np.array(
        [
            [0.0, -8.437500000000004, 0.0],
            [0.0, -30.937500000000007, 0.0],
            [0.0, -88.55859375000001, 0.0],
        ]
    )
    assert np.array_equal(basis, expected)
