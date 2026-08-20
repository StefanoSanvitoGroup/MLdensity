"""Characterization test for the alpha/beta truncation in ``expand_jacobi``.

``jlgridfingerprints/src/polynomials.pyx:13`` declares ``expand_jacobi``'s Jacobi
weight exponents as ``int alpha, int beta``, while every worker beneath it takes
``double``. Cython's int coercion truncates a non-integer argument toward zero
instead of raising, so a fitted float exponent silently becomes the integer it
truncates to. See ``reports/2026-08-19-alpha-beta-truncation/`` for the full
write-up and ``repro_alpha_beta_truncation.py`` for a standalone version of the
same checks.

On ``stable`` every assertion below passes: the truncation is present. Once the
signature is widened to ``double alpha, double beta``, assertions 1 and 2 are
expected to invert -- that inversion, plus a new assertion that the truncated
integers still reproduce the pre-fix output exactly, lands in the same commit as
the fix. Assertion 3 (the control) must hold unchanged before and after.
"""

import numpy as np
import pytest

pytest.importorskip("jlgridfingerprints.lib.polynomials")

from jlgridfingerprints.lib.polynomials import expand_jacobi  # noqa: E402

# Neighbour distances spanning the cutoff; the values only need to not all map
# to the same point of the cosine map. Shared with repro_alpha_beta_truncation.py.
R = np.linspace(0.05, 4.0, 64)

# Aluminium example's radial-map settings, reused as a fixed harness for every
# alpha/beta pair below -- this test is about expand_jacobi's type handling,
# not about reproducing each example pipeline end-to-end.
NMAX_1B = 15
MAP = dict(rcut=4.08, rmin=-0.74, gamma=1.0, shifted=1, double_shifted=1)


def _jacobi(alpha, beta):
    """``expand_jacobi`` at the fixed distances and map settings above."""
    return expand_jacobi(R, NMAX_1B, alpha, beta, **MAP)


# Published (float) alpha/beta against the integers they truncate to, from
# handover Table 2's affected (non-integral) parameter sets. Each id is
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
def test_float_settings_match_their_truncation(a_float, b_float, a_trunc, b_trunc):
    assert np.array_equal(_jacobi(a_float, b_float), _jacobi(a_trunc, b_trunc))


def test_truncation_direction_is_toward_zero_not_floor():
    # beta in (-1, 0) truncates to 0 (toward zero), not -1 (floor). Getting this
    # backwards would misidentify every affected negative exponent.
    alpha, beta = 8.834136312569242, -0.15337566191456764
    assert np.array_equal(_jacobi(alpha, beta), _jacobi(8, 0))
    assert not np.array_equal(_jacobi(alpha, beta), _jacobi(8, -1))


def test_already_integral_settings_are_unaffected():
    # benzene's alpha=[7, 7], beta=[0, 0] -- both channels identical and already
    # integral, so a pure type widening must leave this pair untouched.
    assert np.array_equal(_jacobi(7.0, 0.0), _jacobi(7, 0))
