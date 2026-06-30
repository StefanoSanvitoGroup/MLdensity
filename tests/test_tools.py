"""Smoke tests for jlgridfingerprints.tools (no compiled extensions needed)."""

import numpy as np

from jlgridfingerprints.tools import create_grid_coords, sample_charge


def test_create_grid_coords_fractional_shape():
    coords = create_grid_coords(grid_size=(2, 3, 4), return_cartesian_coords=False)
    assert coords.shape == (2 * 3 * 4, 3)
    assert coords.min() >= 0.0 and coords.max() < 1.0


def test_create_grid_coords_cartesian_scales_with_cell():
    cell = np.diag([2.0, 2.0, 2.0])
    frac = create_grid_coords(grid_size=(2, 2, 2), return_cartesian_coords=False)
    cart = create_grid_coords(
        grid_size=(2, 2, 2), return_cartesian_coords=True, a_vectors=cell
    )
    assert np.allclose(cart, frac @ cell)


def test_sample_charge_returns_unique_in_range():
    chg = np.linspace(0.1, 1.0, 100)
    idx = sample_charge(chg, sigma=1.0, n_samples=10, uniform_ratio=0.5, seed=42)
    assert idx.shape == (10,)
    assert len(np.unique(idx)) == 10
    assert idx.min() >= 0 and idx.max() < chg.size


def test_sample_charge_is_deterministic_with_seed():
    chg = np.linspace(0.1, 1.0, 100)
    a = sample_charge(chg, sigma=1.0, n_samples=10, uniform_ratio=0.5, seed=7)
    b = sample_charge(chg, sigma=1.0, n_samples=10, uniform_ratio=0.5, seed=7)
    assert np.array_equal(a, b)
