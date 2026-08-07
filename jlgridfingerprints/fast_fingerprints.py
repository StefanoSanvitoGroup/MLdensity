"""Parallel :class:`JLGridFingerprints` variant.

This is a faster sibling of
:class:`jlgridfingerprints.fingerprints.JLGridFingerprints`. It subclasses the
serial descriptor and inherits everything except :meth:`create`, so the numerics
are identical: the center loop in the serial ``create`` is embarrassingly
parallel (row ``io`` reads only its own neighbour slice and writes only its own
row, with no accumulation across centers), so splitting it into batches changes
nothing but the wall clock.

How it differs from the serial
:class:`~jlgridfingerprints.fingerprints.JLGridFingerprints`:

* **Batching.** :meth:`create` gains ``batch_size`` and ``num_proc``. The
  centers are sliced into contiguous ``batch_size`` blocks and farmed out to
  ``num_proc`` worker processes, then reassembled in order. ``num_proc > 1``
  requires ``batch_size`` to be set.
* **Construction.** Must be constructed with keyword arguments. The settings are
  stashed so each worker can rebuild the descriptor under the ``spawn`` start
  method; ``__init__`` is pure index bookkeeping, so this is cheap.

Everything else, including the default ``num_proc=1, batch_size=None`` call
signature, behaves exactly as the serial class.

When to use which
-----------------
This class and :mod:`jlgridfingerprints.fast_predictor` parallelise the **same
axis** -- the grid points. ``fast_predictor`` already calls the *serial*
``create`` inside each of its workers, so:

* For predicting a charge density, use ``fast_predictor``. It also parallelises
  the model evaluation, and it never materialises the descriptor matrix.
* For anything that needs the descriptors themselves -- training-data
  generation, analysis -- use this class.
* **Do not nest the two.** It is the same parallelism twice over;
  :class:`multiprocessing.Pool` workers are daemonic and cannot spawn children,
  so it raises rather than silently oversubscribing.

Memory
------
The return value is a dense ``(n_centers, n_features)`` float64 array. For the
aluminium settings (120 features) that is ~1 kB per center, so a full 140^3 grid
would be ~2.6 GB. This API is meant for sampled-point workloads -- the
``create_data.py`` pipelines use ~0.5% of the voxels -- not for full grids.

The speedup only materialises once the per-batch descriptor work dominates the
process-spawn and pickling overhead; on small point sets the parallel path is
*slower*. See ``scripts/benchmark_fingerprints/benchmark_fingerprints.py``.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np

from jlgridfingerprints.fingerprints import JLGridFingerprints as _SerialFingerprints

if TYPE_CHECKING:
    from ase import Atoms


# Per-process state for the multiprocessing workers. A module-level dict + a
# top-level worker function are picklable under the "spawn" start method (the
# default on macOS/Windows), unlike a closure over ``self``.
_WORKER: dict = {}


def _worker_init(jl_settings: dict, system) -> None:
    """Initialise per-process worker state (called once per pool process)."""
    _WORKER["jl"] = _SerialFingerprints(**jl_settings)
    _WORKER["system"] = system


def _create_batch(positions: np.ndarray) -> np.ndarray:
    """Evaluate the descriptor for one batch of centers."""
    return _WORKER["jl"].create(_WORKER["system"], positions)


class JLGridFingerprints(_SerialFingerprints):
    """Grid fingerprints with an optional process-parallel :meth:`create`.

    Takes the same keyword arguments as
    :class:`jlgridfingerprints.fingerprints.JLGridFingerprints`, but they must
    be passed by keyword (they are stashed for reconstruction in the workers).
    """

    def __init__(self, **kwargs):
        """Configure the descriptor and remember the settings for the workers.

        Parameters
        ----------
        **kwargs
            Forwarded verbatim to
            :meth:`jlgridfingerprints.fingerprints.JLGridFingerprints.__init__`.
            Positional arguments are not accepted.
        """
        self._jl_settings = dict(kwargs)
        super().__init__(**kwargs)

    def create(
        self,
        system: Atoms,
        positions: np.ndarray = None,
        batch_size: int = None,
        num_proc: int = 1,
    ) -> np.ndarray:
        """Evaluate the JL fingerprints at each center, optionally in parallel.

        Parameters
        ----------
        system : ase.Atoms
            Structure providing the atoms and (if periodic) the cell.
        positions : numpy.ndarray, optional
            ``(n_centers, 3)`` Cartesian coordinates of the points to describe.
            Defaults to the atom positions of ``system``.
        batch_size : int, optional
            Number of centers evaluated per batch. ``None`` (default) evaluates
            every center in one call. Required when ``num_proc > 1``.
        num_proc : int
            Number of worker processes. Default ``1`` (serial).

        Returns
        -------
        numpy.ndarray
            ``(n_centers, n_features)`` array of descriptors, in input order.

        Raises
        ------
        ValueError
            If ``num_proc > 1`` without a ``batch_size``, or if ``batch_size``
            is not positive.
        """
        if num_proc > 1 and not batch_size:
            raise ValueError("num_proc > 1 requires batch_size to be set")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        if num_proc == 1 and batch_size is None:
            return super().create(system, positions)

        centers = np.asarray(system.get_positions() if positions is None else positions)
        batches = [
            centers[i : i + batch_size] for i in range(0, len(centers), batch_size)
        ]
        init_args = (self._jl_settings, system)

        if num_proc > 1:
            # Force "spawn": it is portable, and the worker design above is
            # built for it. "fork" additionally risks inheriting locked thread
            # state from any OpenMP runtime already initialised in the parent
            # (scikit-learn still brings one, even though our own extensions no
            # longer link OpenMP).
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=num_proc, initializer=_worker_init, initargs=init_args
            ) as pool:
                # imap (not map) streams results lazily and preserves order,
                # which is what makes the concatenation below correct.
                blocks = list(pool.imap(_create_batch, batches))
        else:
            # Same worker functions, run in-process: keeps the serial batched
            # path on exactly the code the parallel path uses.
            _worker_init(*init_args)
            blocks = [_create_batch(batch) for batch in batches]

        return np.concatenate(blocks, axis=0)
