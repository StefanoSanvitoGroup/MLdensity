"""Predict charge densities on an FFT grid from JL fingerprints and a model."""

from __future__ import annotations

import json
import pickle
import time
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Bohr, Rydberg
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Chgcar

from jlgridfingerprints.fingerprints import JLGridFingerprints

if TYPE_CHECKING:
    from ase import Atoms


class JLPredictor:
    """Predict the charge density of a structure from a trained model.

    Wraps a :class:`JLGridFingerprints` descriptor and a pickled regressor:
    fingerprints are evaluated on a regular FFT grid, fed to the model, and the
    predicted density is renormalised to the target electron count and
    optionally written as a VASP CHGCAR.
    """

    def __init__(
        self,
        jl_settings: str | dict,
        model_path: str,
        grid_size: tuple[int, int, int] | None = None,
        encut: float | None = None,
        prec: str = "Accurate",
        scaler_path: str | None = None,
    ) -> None:
        """Load the JL settings, trained model, and optional feature scaler.

        Parameters
        ----------
        jl_settings : str or dict
            :class:`JLGridFingerprints` keyword arguments, either as a dict or
            as a path to a ``.json`` file containing them.
        model_path : str
            Path to a pickled fitted regressor exposing ``predict``.
        grid_size : tuple of int, optional
            Explicit FFT grid ``(nx, ny, nz)``. If omitted, the grid is derived
            from ``encut`` per structure via :meth:`get_chgcar_grid`.
        encut : float, optional
            Plane-wave cutoff energy (eV) used to derive the grid when
            ``grid_size`` is not given.
        prec : str
            VASP-like precision (e.g. ``"Accurate"``/``"Normal"``); sets the
            grid precision factor.
        scaler_path : str, optional
            Path to a pickled feature scaler exposing ``transform``.
        """

        if isinstance(jl_settings, str) and jl_settings.endswith(".json"):
            self.jl_settings = json.load(open(jl_settings))
        elif isinstance(jl_settings, dict):
            self.jl_settings = jl_settings
        else:
            raise ValueError(
                "jl_settings argument needs to be a dict or path to json file"
            )

        print("Using jl_settings: ", self.jl_settings, flush=True)

        self.jl = JLGridFingerprints(**self.jl_settings)
        print("Number of JL coefficients: ", self.jl._n_features, flush=True)

        print("Loading model from: ", model_path, flush=True)
        self.model = pickle.load(open(model_path, "rb"))

        if scaler_path is not None:
            self.scaler = pickle.load(open(scaler_path, "rb"))

        self.grid_size = grid_size

        self._encut = encut
        # I need to check the proper multiplication for encut with respect to prec

        if prec.lower().startswith("a"):
            self._prec_factor = 2.0
        elif prec.lower().startswith("n"):
            self._prec_factor = 2.0
        else:
            self._prec_factor = 1.0

    def predict_chgcar(
        self,
        atoms: Atoms,
        nelect: float,
        batch_size: int | None = None,
        verbose: bool = False,
        save_path: str | None = None,
        name: str | None = None,
        return_chg: bool = False,
        write_chgcar: bool = True,
        use_scaler: bool = False,
    ) -> np.ndarray | None:
        """Predict the charge density of a structure on the FFT grid.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure to predict the density for.
        nelect : float
            Target number of electrons; the predicted density is renormalised
            to this value when it deviates by more than ``1e-6``.
        batch_size : int, optional
            If set, evaluate fingerprints and predict in chunks of this many
            grid points instead of all at once (lower peak memory).
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

        Returns
        -------
        numpy.ndarray or None
            The ``(nx, ny, nz)`` density grid if ``return_chg`` is ``True``,
            otherwise ``None``.
        """

        time_descriptor = 0.0
        time_write = 0.0

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
        if batch_size is None:
            X_batch = self.jl.create(atoms, cart_positions)
            if use_scaler:
                X_batch = self.scaler.transform(X_batch)
            ml_chg_points = self.model.predict(X_batch)
        else:
            nchunks = int(np.ceil(len(cart_positions) / batch_size))
            for ib in range(nchunks):
                X_batch = self.jl.create(
                    atoms, cart_positions[ib * batch_size : (ib + 1) * batch_size]
                )
                if use_scaler:
                    X_batch = self.scaler.transform(X_batch)
                if ib == 0:
                    ml_chg_points = self.model.predict(X_batch)
                else:
                    ml_chg_points = np.append(
                        ml_chg_points, self.model.predict(X_batch), axis=0
                    )
            del X_batch
        time_descriptor += time.time() - t_init

        ml_chg_points = ml_chg_points.reshape((ngxf, ngyf, ngzf), order="C") * vol
        ml_nelect = ml_chg_points.sum() / ml_chg_points.size
        if abs(ml_nelect - nelect) > 1e-6:
            ml_chg_points = self.normalize_nelect(
                ml_chg_points, nelect=nelect, volume=vol
            )

        if write_chgcar:
            t_init = time.time()
            chgcar = Chgcar(
                poscar=Poscar(AseAtomsAdaptor.get_structure(atoms)),
                data={"total": ml_chg_points},
                data_aug=None,
            )
            chgcar.data_aug["total"] = []

            if save_path is None:
                save_path = ""
            else:
                if not save_path.endswith("/"):
                    save_path += "/"
            if name is None:
                chgcar.write_file(save_path + "CHGCAR")
            else:
                chgcar.write_file(save_path + name)

            time_write += time.time() - t_init

        if verbose:
            print(
                f"JL coeff    : {time_descriptor:>5.3f} sec for {ngxf * ngyf * ngzf} points ({ngxf}x{ngyf}x{ngzf} grid) ",
                flush=True,
            )
            if write_chgcar:
                print(f"Write files : {time_write:>5.3f} sec", flush=True)

        if return_chg:
            return ml_chg_points

    def get_chgcar_grid(
        self,
        alats: np.ndarray,
        encut: float,
        prec_factor: float,
        wfact: int = 4,
    ) -> tuple[int, int, int]:
        """Derive a VASP-compatible FFT grid from the plane-wave cutoff.

        Each grid dimension is rounded up to an FFT-friendly size (factorisable
        into 2, 3, 5, 7 with at least one factor of 2) and scaled by the
        precision factor.

        Parameters
        ----------
        alats : numpy.ndarray
            Cell edge lengths ``(a, b, c)``.
        encut : float
            Plane-wave cutoff energy (eV).
        prec_factor : float
            Grid multiplier set by the precision option.
        wfact : int
            Wavevector factor controlling the base grid density.

        Returns
        -------
        tuple of int
            FFT grid sizes ``(ngxf, ngyf, ngzf)``.
        """

        def fftchk(grid):

            def fftchk_legal(nin):
                ifact = [2, 3, 5, 7]
                n2div = 0
                n = nin
                for fact in ifact:
                    while n % fact == 0:
                        n = n / fact
                        if fact == 2:
                            n2div += 1
                if n == 1 and n2div != 0:
                    return True
                else:
                    return False

            for i in range(3):
                while not fftchk_legal(grid[i]):
                    grid[i] += 1

            return grid

        ngx, ngy, ngz = np.floor(
            (encut / Rydberg) ** 0.5 / (2 * np.pi / (alats / Bohr)) * wfact + 0.5
        ).astype(int)
        ngx, ngy, ngz = fftchk([ngx, ngy, ngz])
        ngxf, ngyf, ngzf = prec_factor * np.asarray([ngx, ngy, ngz])

        return int(ngxf), int(ngyf), int(ngzf)

    def normalize_nelect(
        self,
        chg: np.ndarray,
        nelect: float,
        volume: float,
    ) -> np.ndarray:
        """Rescale a density grid to integrate to ``nelect`` electrons.

        Sets the ``G = 0`` Fourier component so the mean density matches the
        target electron count, then transforms back to real space.

        Parameters
        ----------
        chg : numpy.ndarray
            Charge-density grid.
        nelect : float
            Target number of electrons.
        volume : float
            Cell volume.

        Returns
        -------
        numpy.ndarray
            The renormalised density grid (same shape as ``chg``).
        """

        from scipy import fft

        ngxf, ngyf, ngzf = chg.shape

        chg /= volume

        chg_g = fft.fftn(chg)
        chg_g[0, 0, 0] = nelect / volume * (ngxf * ngyf * ngzf)
        chg_r = fft.ifftn(chg_g).real

        chg_r *= volume

        return chg_r
