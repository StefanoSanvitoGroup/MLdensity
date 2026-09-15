"""Regression tests for the alpha/beta exponents of ``expand_jacobi``.

``expand_jacobi`` used to declare its Jacobi weight exponents as ``int alpha,
int beta`` while every worker beneath it took ``double``. Cython's int coercion
truncates a non-integer argument toward zero instead of raising, so a fitted
float exponent silently became the integer it truncated to. Widening the
signature to ``double`` fixed that; see issue #6 and PR #7 for the
full write-up.

These tests were first landed as a characterization of the defective behaviour
and then inverted in the commit that widened the signature, so the numerical
consequence of the fix appears in the diff rather than silently. They now assert
that float exponents are honoured, that the historical truncated basis stays
reachable by passing the integers explicitly, and that already-integral settings
were never affected either way.
"""

import numpy as np
import pytest

pytest.importorskip("jlgridfingerprints.lib.polynomials")

from jlgridfingerprints.lib.polynomials import expand_jacobi  # noqa: E402

# Neighbour distances spanning the cutoff; the values only need to not all map
# to the same point of the cosine map. Shared with repro_alpha_beta_truncation.py.
R = np.linspace(0.05, 4.0, 64)

# A fixed harness for every alpha/beta pair below, taking rcut/rmin from the
# aluminium example. Deliberately not a copy of either expand_jacobi call site --
# create_2b_jl forces double_shifted=0 and create_3b_jl forces rmin=0.0 with
# nmax_2b, so no single call path matches this combination. It does not need to:
# expand_jacobi's alpha/beta type handling is independent of the radial map.
NMAX = 15
MAP = dict(rcut=4.08, rmin=-0.74, gamma=1.0, shifted=1, double_shifted=1)


def _jacobi(alpha, beta):
    """``expand_jacobi`` at the fixed distances and map settings above."""
    return expand_jacobi(R, NMAX, alpha, beta, **MAP)


# Published (float) alpha/beta against the integers they used to truncate to,
# from handover Table 2's affected (non-integral) parameter sets. Each id is
# "<example>-<channel>"; alpha and beta are paired by channel, not by table row,
# since fingerprints.py calls expand_jacobi once per channel with its own
# alpha/beta component.
AFFECTED_PAIRS = [
    pytest.param(7.875386069413652, 3.6238075908648106, 7, 3, id="aluminium-1B"),
    pytest.param(5.875090883472657, 1.7505953204305842, 5, 1, id="aluminium-2B"),
    pytest.param(6.72, 6.97, 6, 6, id="2d_mos2-1B"),
    pytest.param(5.07, 2.69, 5, 2, id="2d_mos2-2B"),
    pytest.param(4.016939668269249, 5.4622394140294785, 4, 5, id="molybdenium-1B"),
    pytest.param(-0.0827444142560686, 2.378398890338807, 0, 2, id="molybdenium-2B"),
    pytest.param(8.834136312569242, -0.15337566191456764, 8, 0, id="kkr-host-campaign"),
]


@pytest.mark.parametrize("a_float,b_float,a_trunc,b_trunc", AFFECTED_PAIRS)
def test_float_settings_differ_from_their_truncation(
    a_float, b_float, a_trunc, b_trunc
):
    # Inverted at the fix: before the widening these two calls agreed, which was
    # the defect. A fractional exponent must now reach the recurrence intact.
    assert not np.array_equal(_jacobi(a_float, b_float), _jacobi(a_trunc, b_trunc))


@pytest.mark.parametrize("a_float,b_float,a_trunc,b_trunc", AFFECTED_PAIRS)
def test_truncated_integers_reproduce_the_pre_fix_basis(
    a_float, b_float, a_trunc, b_trunc
):
    # The historical basis stays reachable: passing the integers that used to be
    # in force reproduces the pre-fix output exactly, because integral values
    # take the identical arithmetic path through the recurrence whether they
    # arrive as int or float. This is what lets the published models be
    # reconstructed from a fixed library, and is the unit-level counterpart of
    # the byte-identity check on the aluminium pipeline.
    del a_float, b_float  # named for parametrize id parity with the test above
    assert np.array_equal(
        _jacobi(a_trunc, b_trunc), _jacobi(float(a_trunc), float(b_trunc))
    )


def test_fractional_beta_is_not_collapsed_to_either_neighbouring_integer():
    # Before the fix, a beta in (-1, 0) truncated toward zero -- to 0, not to -1
    # as floor would give. Both are now wrong answers: the fractional value must
    # match neither of its neighbouring integers.
    alpha, beta = 8.834136312569242, -0.15337566191456764
    assert not np.array_equal(_jacobi(alpha, beta), _jacobi(8, 0))
    assert not np.array_equal(_jacobi(alpha, beta), _jacobi(8, -1))
    # ...and it must still be finite: beta > -1 is inside the Jacobi domain, so
    # widening must not have opened a singular branch of the recurrence.
    assert np.isfinite(_jacobi(alpha, beta)).all()


def test_already_integral_settings_are_unaffected():
    # benzene's alpha=[7, 7], beta=[0, 0] -- both channels identical and already
    # integral, so a pure type widening must leave this pair untouched. Held
    # before the fix and must hold after; it is what distinguishes widening the
    # signature from altering the recurrence.
    assert np.array_equal(_jacobi(7.0, 0.0), _jacobi(7, 0))
