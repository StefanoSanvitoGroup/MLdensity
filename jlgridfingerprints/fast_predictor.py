"""Parallel :class:`JLPredictor` variant.

This is a faster sibling of :class:`jlgridfingerprints.predictor.JLPredictor`. It
subclasses the serial predictor and inherits ``__init__``, ``get_chgcar_grid`` and
``normalize_nelect`` unchanged, so the numerics of the grid derivation and the
renormalisation are identical. It overrides the prediction step to evaluate the
fingerprints and the model over a ``multiprocessing.Pool``.

How it differs from the serial :class:`~jlgridfingerprints.predictor.JLPredictor`:

* **Grid evaluation.** The serial predictor evaluates every grid point in one
  ``jl.create`` call (or a sequential chunk loop with ``np.append``). This variant
  splits the grid into batches and farms them out to ``num_proc`` worker processes,
  streaming the per-batch predictions back through ``np.fromiter``.
* **``num_proc``.** New argument to :meth:`predict_chgcar`. ``num_proc > 1`` requires
  ``batch_size`` to be set.
* **Normalisation.** The serial predictor *always* renormalises the density to
  ``nelect``. Here it is opt-in via ``normalize=False`` (default). This is a
  behaviour difference to be aware of: pass ``normalize=True`` to match the serial
  predictor.
* **Explicit point sets.** Adds :meth:`predict_key_chgcar`, which predicts on a
  caller-supplied set of Cartesian points and grid shape instead of the full FFT
  grid.

The speedup over the serial predictor only materialises on full-size grids, where
the per-batch fingerprint work dominates the process-spawn and pickling overhead.
On small grids the parallel path is *slower*. See
``scripts/benchmark_predictors/benchmark_predictors.py`` for a way to measure it on
real data.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from itertools import chain, islice
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Chgcar

from jlgridfingerprints.fingerprints import JLGridFingerprints
from jlgridfingerprints.predictor import JLPredictor as _SerialPredictor

if TYPE_CHECKING:
    from ase import Atoms


# ponytail: hand-rolled to avoid a more_itertools dependency for one call;
# replace with itertools.batched once the supported floor is Python 3.12.
def _batched(iterable, n):
    """Yield successive ``n``-sized chunks from ``iterable`` as tuples."""
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


# Per-process state for the multiprocessing workers. A module-level dict + a
# top-level worker function are picklable under the ``spawn`` start method
# (the default on macOS/Windows), unlike a closure over ``self``.
_WORKER: dict = {}


def _worker_init(jl_settings, model, scaler, atoms, use_scaler) -> None:
    """Initialise per-process worker state (called once per pool process)."""
    _WORKER["jl"] = JLGridFingerprints(**jl_settings)
    _WORKER["model"] = model
    _WORKER["scaler"] = scaler
    _WORKER["atoms"] = atoms
    _WORKER["use_scaler"] = use_scaler


def _predict_batch(positions) -> np.ndarray:
    """Create fingerprints for one batch of points and predict on them."""
    X = _WORKER["jl"].create(_WORKER["atoms"], np.asarray(positions))
    if _WORKER["use_scaler"]:
        X = _WORKER["scaler"].transform(X)
    return _WORKER["model"].predict(X)


class JLPredictor(_SerialPredictor):
    """Multiprocessing charge-density predictor.

    Inherits construction, FFT-grid derivation (:meth:`get_chgcar_grid`) and
    renormalisation (:meth:`normalize_nelect`) from
    :class:`jlgridfingerprints.predictor.JLPredictor`; see that class for the
    constructor parameters.
    """

    def _evaluate(
        self,
        atoms: Atoms,
        cart_positions: np.ndarray,
        batch_size: int | None,
        num_proc: int,
        use_scaler: bool,
    ) -> np.ndarray:
        """Evaluate fingerprints and the model over a set of Cartesian points.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure to predict the density for.
        cart_positions : numpy.ndarray
            ``(N, 3)`` Cartesian points to predict at.
        batch_size : int or None
            Points per batch. If ``None`` (and ``num_proc == 1``), all points are
            evaluated in a single call.
        num_proc : int
            Number of worker processes. ``> 1`` requires ``batch_size``.
        use_scaler : bool
            Apply the loaded feature scaler before prediction.

        Returns
        -------
        numpy.ndarray
            Flat ``(N,)`` array of predicted point densities.
        """

        if num_proc > 1 and not batch_size:
            raise ValueError("num_proc > 1 requires batch_size to be set")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        scaler = getattr(self, "scaler", None)
        if use_scaler and scaler is None:
            raise ValueError(
                "use_scaler=True but no scaler was loaded; "
                "pass scaler_path to the constructor"
            )

        if num_proc == 1 and batch_size is None:
            X = self.jl.create(atoms, cart_positions)
            if use_scaler:
                X = scaler.transform(X)
            return self.model.predict(X)

        init_args = (self.jl_settings, self.model, scaler, atoms, use_scaler)
        batches = _batched(cart_positions, batch_size)

        if num_proc > 1:
            with mp.Pool(
                processes=num_proc, initializer=_worker_init, initargs=init_args
            ) as pool:
                # imap (not map) streams results lazily, keeping peak memory low.
                results = pool.imap(_predict_batch, batches)
                return np.fromiter(chain.from_iterable(results), dtype=np.float64)

        _worker_init(*init_args)
        results = map(_predict_batch, batches)
        return np.fromiter(chain.from_iterable(results), dtype=np.float64)

    def _write_chgcar(
        self,
        atoms: Atoms,
        data: np.ndarray,
        save_path: str | None,
        name: str | None,
    ) -> None:
        """Write a density grid to a VASP CHGCAR file."""
        chgcar = Chgcar(
            poscar=Poscar(AseAtomsAdaptor.get_structure(atoms)),
            data={"total": data},
            data_aug=None,
        )
        chgcar.data_aug["total"] = []

        if save_path is None:
            save_path = ""
        elif not save_path.endswith("/"):
            save_path += "/"
        chgcar.write_file(save_path + (name if name is not None else "CHGCAR"))

    def predict_chgcar(
        self,
        atoms: Atoms,
        nelect: float,
        batch_size: int | None = None,
        num_proc: int = 1,
        verbose: bool = False,
        save_path: str | None = None,
        name: str | None = None,
        return_chg: bool = False,
        write_chgcar: bool = True,
        use_scaler: bool = False,
        normalize: bool = False,
    ) -> np.ndarray | None:
        """Predict the charge density of a structure on the FFT grid.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure to predict the density for.
        nelect : float
            Target number of electrons. Only used when ``normalize`` is ``True``.
        batch_size : int, optional
            If set, evaluate fingerprints and predict in chunks of this many grid
            points. Required when ``num_proc > 1``.
        num_proc : int
            Number of worker processes used to evaluate the grid in parallel.
            ``> 1`` requires ``batch_size``.
        verbose : bool
            Print grid size and timing information.
        save_path : str, optional
            Directory to write the CHGCAR into (default: current directory).
        name : str, optional
            Output filename (default ``"CHGCAR"``).
        return_chg : bool
            If ``True``, return the predicted density array.
        write_chgcar : bool
            If ``True``, write the density to a VASP CHGCAR file.
        use_scaler : bool
            Apply the loaded feature scaler before prediction.
        normalize : bool
            If ``True``, renormalise the predicted density to ``nelect`` when it
            deviates by more than ``1e-6``. Unlike the serial predictor, this is
            off by default.

        Returns
        -------
        numpy.ndarray or None
            The ``(nx, ny, nz)`` density grid if ``return_chg`` is ``True``,
            otherwise ``None``.
        """

        vol = atoms.cell.volume

        if self.grid_size is not None and len(self.grid_size) == 3:
            ngxf, ngyf, ngzf = np.array(self.grid_size, dtype=int)
        else:
            alats = np.linalg.norm(atoms.get_cell().array, axis=-1)
            ngxf, ngyf, ngzf = self.get_chgcar_grid(
                alats, self._encut, self._prec_factor
            )

        if verbose:
            print(f"Grid size is: {ngxf}x{ngyf}x{ngzf}", flush=True)

        xx, yy, zz = np.meshgrid(
            np.arange(0, 1, 1 / ngxf),
            np.arange(0, 1, 1 / ngyf),
            np.arange(0, 1, 1 / ngzf),
            indexing="ij",
        )

        frac_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        cart_positions = np.dot(frac_points, atoms.get_cell().array)
        del frac_points

        t_init = time.time()
        ml_chg_points = self._evaluate(
            atoms, cart_positions, batch_size, num_proc, use_scaler
        )
        time_descriptor = time.time() - t_init

        ml_chg_points = ml_chg_points.reshape((ngxf, ngyf, ngzf), order="C") * vol
        ml_nelect = ml_chg_points.sum() / ml_chg_points.size

        if normalize and abs(ml_nelect - nelect) > 1e-6:
            ml_chg_points = self.normalize_nelect(
                ml_chg_points, nelect=nelect, volume=vol
            )

        time_write = 0.0
        if write_chgcar:
            t_init = time.time()
            self._write_chgcar(atoms, ml_chg_points, save_path, name)
            time_write = time.time() - t_init

        if verbose:
            print(
                f"JL coeff    : {time_descriptor:>5.3f} sec for "
                f"{ngxf * ngyf * ngzf} points ({ngxf}x{ngyf}x{ngzf} grid) ",
                flush=True,
            )
            print(f"ML number of electrons is {ml_nelect}", flush=True)
            if write_chgcar:
                print(f"Write files : {time_write:>5.3f} sec", flush=True)

        if return_chg:
            return ml_chg_points

    def predict_key_chgcar(
        self,
        atoms: Atoms,
        cart_positions: np.ndarray,
        gridsize: tuple[int, int, int],
        nelect: float,
        batch_size: int | None = None,
        num_proc: int = 1,
        verbose: bool = False,
        save_path: str | None = None,
        name: str | None = None,
        return_chg: bool = False,
        write_chgcar: bool = True,
        use_scaler: bool = False,
        normalize: bool = False,
    ) -> np.ndarray | None:
        """Predict the density on a caller-supplied set of points.

        Like :meth:`predict_chgcar` but evaluates at the explicit ``cart_positions``
        (reshaped to ``gridsize``) instead of deriving the FFT grid internally.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure to predict the density for.
        cart_positions : numpy.ndarray
            ``(N, 3)`` Cartesian points to predict at; ``N`` must equal the product
            of ``gridsize``.
        gridsize : tuple of int
            Grid shape ``(nx, ny, nz)`` to reshape the predictions into.
        nelect : float
            Target number of electrons. Only used when ``normalize`` is ``True``.
        batch_size : int, optional
            Points per batch. Required when ``num_proc > 1``.
        num_proc : int
            Number of worker processes. ``> 1`` requires ``batch_size``.
        verbose : bool
            Print the predicted electron count.
        save_path : str, optional
            Directory to write the CHGCAR into (default: current directory).
        name : str, optional
            Output filename (default ``"CHGCAR"``).
        return_chg : bool
            If ``True``, return the predicted density array.
        write_chgcar : bool
            If ``True``, write the density to a VASP CHGCAR file.
        use_scaler : bool
            Apply the loaded feature scaler before prediction.
        normalize : bool
            If ``True``, renormalise the predicted density to ``nelect`` when it
            deviates by more than ``1e-6``.

        Returns
        -------
        numpy.ndarray or None
            The density grid (shape ``gridsize``) if ``return_chg`` is ``True``,
            otherwise ``None``.
        """

        n_expected = int(np.prod(gridsize))
        if len(cart_positions) != n_expected:
            raise ValueError(
                f"cart_positions has {len(cart_positions)} points but gridsize "
                f"{tuple(gridsize)} implies {n_expected}"
            )

        vol = atoms.get_volume()

        ml_chg_points = self._evaluate(
            atoms, cart_positions, batch_size, num_proc, use_scaler
        )

        ml_chg_points = ml_chg_points.reshape(gridsize, order="C") * vol
        ml_nelect = ml_chg_points.sum() / ml_chg_points.size

        if normalize and abs(ml_nelect - nelect) > 1e-6:
            ml_chg_points = self.normalize_nelect(
                ml_chg_points, nelect=nelect, volume=vol
            )

        if verbose:
            print(f"ML number of electrons is {ml_nelect}", flush=True)

        if write_chgcar:
            self._write_chgcar(atoms, ml_chg_points, save_path, name)

        if return_chg:
            return ml_chg_points
