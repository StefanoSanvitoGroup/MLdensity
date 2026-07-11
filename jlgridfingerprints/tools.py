"""Helper utilities for building grids and sampling charge-density points."""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng


def create_grid_coords(
    grid_size: tuple[int, int, int] = (160, 160, 160),
    return_cartesian_coords: bool = False,
    a_vectors: np.ndarray = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ),
) -> np.ndarray:
    """Build a regular grid of coordinates spanning the unit cell.

    Parameters
    ----------
    grid_size : tuple of int
        Number of grid points along each cell axis ``(nx, ny, nz)``.
    return_cartesian_coords : bool
        If ``True``, return Cartesian coordinates (fractional coordinates
        projected onto ``a_vectors``); otherwise return fractional coordinates.
    a_vectors : numpy.ndarray
        ``(3, 3)`` lattice vectors stored as rows, used only when
        ``return_cartesian_coords`` is ``True``.

    Returns
    -------
    numpy.ndarray
        ``(nx * ny * nz, 3)`` array of grid coordinates in C order.
    """

    ngxf, ngyf, ngzf = grid_size

    xx, yy, zz = np.meshgrid(
        np.arange(0, 1, 1 / ngxf),
        np.arange(0, 1, 1 / ngyf),
        np.arange(0, 1, 1 / ngzf),
        indexing="ij",
    )

    fcoords = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    if return_cartesian_coords:
        return np.dot(fcoords, a_vectors)
    else:
        return fcoords


def sample_charge(
    chg: np.ndarray,
    sigma: float,
    n_samples: int,
    uniform_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """Select sample indices from a charge-density grid.

    Combines density-weighted sampling with uniform sampling. The density
    weight is ``exp(-(1 / chg) ** 2 / (2 * sigma ** 2))``, which emphasises
    high-density voxels; the remaining fraction is drawn uniformly. The pool of
    density-weighted draws is grown until enough unique indices are collected.

    Parameters
    ----------
    chg : numpy.ndarray
        Charge-density grid (any shape; it is flattened internally).
    sigma : float
        Width of the Gaussian density weight in ``1 / chg``.
    n_samples : int
        Number of unique flat indices to return.
    uniform_ratio : float
        Fraction of ``n_samples`` drawn uniformly; the rest are
        density-weighted.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    numpy.ndarray
        ``(n_samples,)`` array of shuffled flat indices into ``chg.ravel()``.
    """

    rng = default_rng(seed)

    chg = chg.ravel()

    if n_samples > chg.size:
        raise ValueError(
            f"n_samples ({n_samples}) exceeds the number of voxels "
            f"({chg.size}); unique sampling without replacement is impossible"
        )

    n_prob = int(np.ceil(n_samples * (1 - uniform_ratio)))
    n_uniform = n_samples - n_prob

    prob_chg = np.exp(-((1 / chg) ** 2) / (2 * sigma**2)) / (
        (2 * np.pi * (sigma**2)) ** 0.5
    )
    prob_chg /= sum(prob_chg)

    selected_index = np.array([])
    m = 1.0
    while len(selected_index) < n_samples:
        selected_chg = rng.choice(
            np.arange(len(chg)),
            size=int(np.ceil(n_prob * m)),
            p=prob_chg,
            replace=False,
        )
        selected_uniform = rng.choice(
            np.arange(len(chg)), size=n_uniform, replace=False
        )
        selected_index = np.unique(np.append(selected_chg, selected_uniform))
        m += 0.1

    rng.shuffle(selected_index)

    return selected_index[:n_samples]
