"""Regression tests for the 2B (two-body) upper-triangle packing.

``calculate_3b_upper`` packs the symmetric ``nmax x nmax`` radial index pair
into ``nmax * (nmax + 1) / 2`` slots per angular order. The packing formula is
only a bijection over the upper triangle ``n2 >= n1``; pairing it with the
opposite loop guard silently drops pairs onto slots that already hold another
pair and leaves ``floor((nmax - 1)**2 / 4)`` slots per angular order
permanently at zero. These tests pin the bijection itself, so that failure mode
cannot return unnoticed.
"""

import numpy as np
import pytest

# Skip the whole module unless the compiled extensions import.
pytest.importorskip("jlgridfingerprints.lib.jlcontraction")

from ase.build import bulk  # noqa: E402

from jlgridfingerprints.fingerprints import JLGridFingerprints  # noqa: E402
from jlgridfingerprints.lib.jlcontraction import calculate_3b_upper  # noqa: E402

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


def _slots_for_pair(nmax, n1, n2, lmax=2, n_neigh=3):
    """Return the output slots the kernel writes for the radial pair (n1, n2).

    Drives the kernel with one-hot Jacobi expansions, so exactly one internal
    ``(i1, i2)`` iteration can contribute and the nonzero slots of the result
    are precisely the ones that pair is packed into. The Legendre block is
    strictly positive, so a slot that is written is never zero by cancellation.
    """

    a = np.zeros((nmax, n_neigh))
    a[n1] = 1.0
    b = np.zeros((nmax, n_neigh))
    b[n2] = 1.0
    c = np.ascontiguousarray(np.ones((lmax, n_neigh, n_neigh)))

    out = calculate_3b_upper(a, b, c)
    return set(np.flatnonzero(np.abs(out) > 1e-12).tolist())


@pytest.mark.parametrize("nmax", [2, 3, 4, 5, 8, 11])
def test_upper_packing_is_a_bijection(nmax):
    """Every upper-triangle pair gets its own slot, and no slot is left dead."""

    lmax = 2
    n_pairs = nmax * (nmax + 1) // 2

    pair_of_slot = {}
    for n1 in range(nmax):
        for n2 in range(n1, nmax):
            slots = _slots_for_pair(nmax, n1, n2, lmax=lmax)

            # One slot per angular order, at the documented stride.
            assert len(slots) == lmax, (
                f"pair ({n1},{n2}) wrote {len(slots)} slots, expected {lmax}"
            )
            base = min(slots)
            assert slots == {base + l * n_pairs for l in range(lmax)}, (
                f"pair ({n1},{n2}) is not packed at stride {n_pairs}: {sorted(slots)}"
            )

            assert base not in pair_of_slot, (
                f"pair ({n1},{n2}) collides with {pair_of_slot[base]} at slot {base}"
            )
            pair_of_slot[base] = (n1, n2)

    # Onto, not merely injective: no slot is left permanently at zero.
    assert set(pair_of_slot) == set(range(n_pairs)), (
        f"dead slots: {sorted(set(range(n_pairs)) - set(pair_of_slot))}"
    )


@pytest.mark.parametrize("nmax", [4, 5, 11])
def test_kernel_matches_an_explicit_contraction(nmax):
    """Every slot holds the coefficient of the radial pair it is packed for.

    The bijection test above pins *where* each pair lands; this pins *what*
    lands there, against an independent contraction in NumPy. Called the way
    the library calls it -- ``a is b``, and a Legendre block symmetric in its
    two neighbour axes -- so the upper triangle carries the whole block.
    """

    lmax, n_neigh = 3, 6
    n_pairs = nmax * (nmax + 1) // 2

    rng = np.random.default_rng(0)
    a = np.ascontiguousarray(rng.standard_normal((nmax, n_neigh)))
    c = rng.standard_normal((lmax, n_neigh, n_neigh))
    c = np.ascontiguousarray((c + c.transpose(0, 2, 1)) / 2)

    out = calculate_3b_upper(a, a, c)
    expected = np.einsum("in,jm,lnm->ijl", a, a, c)

    for n1 in range(nmax):
        for n2 in range(n1, nmax):
            slot = n1 * nmax - (n1 * (n1 - 1)) // 2 + (n2 - n1)
            for l in range(lmax):
                assert out[slot + l * n_pairs] == pytest.approx(
                    expected[n1, n2, l]
                ), f"slot for pair ({n1},{n2}) at l={l} holds the wrong coefficient"


def test_no_identically_zero_descriptor_columns():
    """A real structure exercises every column of the descriptor.

    With the packing broken, ``(lmax + 1) * floor((nmax_2b - 2)**2 / 4)``
    columns are zero for every center of every structure -- 28 of these 120
    under the aluminium settings.
    """

    atoms = bulk("Al", "fcc", a=4.05).repeat((2, 2, 2))
    atoms.rattle(stdev=0.1, seed=42)

    jl = JLGridFingerprints(**SETTINGS)

    rng = np.random.default_rng(42)
    centers = rng.random((12, 3)) @ atoms.get_cell().array

    X = jl.create(atoms, centers)

    dead = np.flatnonzero(np.all(np.abs(X) < 1e-12, axis=0))
    assert dead.size == 0, f"{dead.size} identically-zero columns: {dead.tolist()}"
