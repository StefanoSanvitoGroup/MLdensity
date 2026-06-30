"""Linear Jacobi-Legendre grid fingerprints of the local atomic environment."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from .lib.geometry import get_nn_in_sphere
from .lib.jlcontraction import calculate_2b, calculate_3b, calculate_3b_upper
from .lib.polynomials import expand_jacobi, expand_legendre
from .lib.utils import get_versors

if TYPE_CHECKING:
    from ase import Atoms


class JLGridFingerprints:
    """Linear Jacobi-Legendre (JL) fingerprints evaluated on grid points.

    Each center (an arbitrary point in space, typically a charge-density grid
    point) is described by an expansion of its local atomic environment in
    Jacobi polynomials (radial part) and Legendre polynomials (angular part),
    following the Jacobi-Legendre Charge Density Model (JLCDM,
    DOI:10.1038/s41524-023-01053-0).

    Notes
    -----
    Body-order terminology: the JL expansion is formally extendable to any body
    order; this implementation provides the first two. In the grid-centered
    JLCDM convention the central grid point is *not* counted as a body, so the
    term that is two-body in the atom-centered JL potential becomes the
    one-body (``1B``) term here, and the three-body term becomes the two-body
    (``2B``) term. The public API uses this grid-centered (1B/2B) convention;
    some internal names (the :meth:`create_2b_jl` / :meth:`create_3b_jl`
    methods and the ``_do_2b_jl`` / ``_do_3b_jl`` flags) retain the 
    atom-centered labels.

    - **1B term**: radial expansion in (vanishing-)Jacobi polynomials over the
      grid-point-to-atom distances.
    - **2B term**: radial-angular expansion combining (double-vanishing-)Jacobi
      polynomials (distances) with Legendre polynomials of the angle between
      pairs of neighbours.
    """

    def __init__(
        self,
        rcut: float = None,
        nmax: list[int] = None,
        lmax: int = None,
        alpha: list[float] = None,
        beta: list[float] = None,
        species: list[str] = None,
        body: str = "1+2",
        rmin: float = 0.0,
        gamma: float = 1.0,
        vector: bool = True,
        periodic: bool = True,
        shifted: bool = True,
        double_shifted: bool = False,
        nn_leaf_size: int = 2,
    ):
        """Configure the descriptor and precompute the feature layout.

        Parameters
        ----------
        rcut : float
            Neighbour cutoff radius in Angstrom (required, > 0).
        nmax : list of int
            Maximum radial (Jacobi) order per body term: a single value shared
            by both terms, or ``[n_1B, n_2B]``.
        lmax : int
            Maximum angular (Legendre) order for the 2B term.
        alpha, beta : list of float
            Jacobi polynomial weighting-function parameters (both > -1),
            parallel to ``nmax`` (one or two entries); ``alpha`` and ``beta``
            must have matching length.
        species : list of str
            Chemical symbols of the species to include.
        body : str
            ``"+"``-joined tokens selecting which terms to compute: ``"1"`` for
            the 1B (radial) term and ``"2"`` for the 2B (radial-angular) term.
            Default ``"1+2"`` enables both.
        rmin : float
            Lower edge of the radial cosine map in Angstrom, applied to the 1B
            term only (the 2B term always uses ``rmin = 0``). May be negative
            to push the point of vanishing derivative to inaccessible negative
            distances, letting the model fit steep density variations close to
            the atoms. Default ``0.0``.
        gamma : float
            Scaling factor applied to the cosine-mapped radial variable in the
            Jacobi expansion. Default ``1.0``.
        vector : bool
            If ``True`` (default), return descriptors as a concatenated 1-D
            vector; otherwise as a list of per-species / per-pair blocks.
        periodic : bool
            Apply periodic boundary conditions in the neighbour search. Default
            ``True``.
        shifted : bool
            Use vanishing-Jacobi radial polynomials (vanishing at ``rcut``).
            Default ``True``.
        double_shifted : bool
            Use double-vanishing-Jacobi radial polynomials (vanishing at both
            edges of the interval). Applies to the 2B term, where it preserves
            continuity near the atoms and removes inter-body redundancy.
            Default ``False``.
        nn_leaf_size : int
            Leaf size of the KD-tree used for the neighbour search. Default
            ``2``.
        """

        self._vector = vector

        self._periodic = periodic

        # Setup the involved chemical species
        self.species = species
        self._n_species = len(species)

        if rcut <= 0:
            raise ValueError("Only positive 'rcut' are allowed.")
        self._rcut = float(rcut)

        self._do_2b_jl = "1" in body.split("+")
        self._do_3b_jl = "2" in body.split("+")

        if self._do_3b_jl:
            if len(nmax) == 0:
                self._same_nmax = True
                self._nmax_1b = self._nmax_2b = nmax
                self._alpha_1b = self._alpha_2b = alpha
                self._beta_1b = self._beta_2b = beta
            elif len(nmax) == 1:
                self._same_nmax = True
                self._nmax_1b = self._nmax_2b = nmax[0]
                self._alpha_1b = self._alpha_2b = alpha[0]
                self._beta_1b = self._beta_2b = beta[0]
            elif len(nmax) == 2:
                self._same_nmax = False
                self._nmax_1b = nmax[0]
                self._nmax_2b = nmax[1]
                self._alpha_1b, self._alpha_2b = alpha
                self._beta_1b, self._beta_2b = beta
            else:
                raise ValueError(
                    "size of nmax should correspond to the number of bodys for expansion"
                )
        else:
            if len(nmax) == 0:
                self._nmax_1b = nmax
                self._same_nmax = True
                self._alpha_1b = alpha
                self._beta_1b = beta
            elif len(nmax) == 1:
                self._nmax_1b = nmax[0]
                self._same_nmax = True
                self._alpha_1b = alpha[0]
                self._beta_1b = beta[0]
            else:
                raise ValueError(
                    "size of nmax should correspond to the number of bodys for expansion"
                )

        if len(alpha) != len(beta):
            raise ValueError("size of alpha and beta should match")

        if self._do_3b_jl:
            if len(alpha) == 0:
                self._alpha_1b = self._alpha_2b = alpha
                self._beta_1b = self._beta_2b = beta
            elif len(alpha) == 1:
                self._alpha_1b = self._alpha_2b = alpha[0]
                self._beta_1b = self._beta_2b = beta[0]
            elif len(alpha) == 2:
                self._alpha_1b, self._alpha_2b = alpha
                self._beta_1b, self._beta_2b = beta
            else:
                raise ValueError(
                    "size of alpha and beta should correspond to the number of bodys for expansion"
                )
        else:
            if len(alpha) == 0:
                self._alpha_1b = alpha
                self._beta_1b = beta
            elif len(alpha) == 1:
                self._alpha_1b = alpha[0]
                self._beta_1b = beta[0]
            else:
                raise ValueError(
                    "size of alpha and beta should correspond to the number of bodys for expansion"
                )

        self._lmax = lmax
        self._gamma = gamma
        self._rmin = rmin
        self._shifted = shifted
        self._double_shifted = double_shifted

        self._nn_leaf_size = nn_leaf_size

        pair_index = np.unique(
            np.sort(
                list(
                    itertools.product(
                        np.arange(self._n_species), np.arange(self._n_species)
                    )
                )
            ),
            axis=0,
        )

        if self._n_species > 1:
            self.species_pair_index = np.append(
                [(i, j) for i, j in pair_index if i == j],
                [(i, j) for i, j in pair_index if i != j],
                axis=0,
            )

            self.species_ij = np.append(
                [
                    (self.species[i], self.species[j])
                    for i, j in self.species_pair_index
                    if i == j
                ],
                [
                    (self.species[i], self.species[j])
                    for i, j in self.species_pair_index
                    if i != j
                ],
                axis=0,
            )

            self._n_unique_symb = 0
            for symb_a, symb_b in self.species_ij:
                if symb_a != symb_b:
                    self._n_unique_symb += 1

        else:
            self.species_pair_index = pair_index
            self.species_ij = np.asarray(
                [(self.species[i], self.species[j]) for i, j in self.species_pair_index]
            )

        self._n_features = self.number_of_features()

    def number_of_features(self) -> int:
        """Return the total descriptor length for the current settings.

        Also caches the per-term feature counts (``_n_2b_features``,
        ``_n_3b_features_mono``, ``_n_3b_features_duo``) as a side effect.

        Returns
        -------
        int
            Number of features in the descriptor vector.
        """

        n_features = 0
        if self._do_2b_jl:
            n_features += self._n_species * (self._nmax_1b)
            self._n_2b_features = self._n_species * (self._nmax_1b)
        if self._do_3b_jl:
            if self._double_shifted:
                n_full_terms = (self._nmax_2b - 1) * (self._nmax_2b - 1)
                n_upper_terms = (n_full_terms - (self._nmax_2b - 1)) // 2 + (
                    self._nmax_2b - 1
                )
            else:
                n_full_terms = (self._nmax_2b) * (self._nmax_2b)
                n_upper_terms = (n_full_terms - (self._nmax_2b)) // 2 + (self._nmax_2b)

            self._n_full_terms = n_full_terms
            self._n_upper_terms = n_upper_terms

            if self._n_species == 1:
                n_features += len(self.species_ij) * n_upper_terms * (self._lmax + 1)
                self._n_3b_features_mono = (
                    len(self.species_ij) * n_upper_terms * (self._lmax + 1)
                )
            else:
                n_features += self._n_species * n_upper_terms * (self._lmax + 1)
                n_features += self._n_unique_symb * n_full_terms * (self._lmax + 1)

                self._n_3b_features_mono = (
                    self._n_species * n_upper_terms * (self._lmax + 1)
                )
                self._n_3b_features_duo = (
                    self._n_unique_symb * n_full_terms * (self._lmax + 1)
                )

        return n_features

    def create(self, system: Atoms, positions: np.ndarray = None) -> np.ndarray:
        """Evaluate the JL fingerprints at each center.

        Parameters
        ----------
        system : ase.Atoms
            Structure providing the atoms and (if periodic) the cell.
        positions : numpy.ndarray, optional
            ``(n_centers, 3)`` Cartesian coordinates of the points to describe.
            Defaults to the atom positions of ``system``.

        Returns
        -------
        numpy.ndarray
            ``(n_centers, n_features)`` array of descriptors.
        """

        system.set_pbc(self._periodic)

        if positions is None:
            centers = np.array(system.get_positions())
        else:
            centers = np.array(positions)

        self._n_centers = len(centers)
        self.centers = centers

        self._initialize_distances(system=system)

        jl_descriptors = np.zeros((self._n_centers, self._n_features))
        for io in range(self._n_centers):
            if self._do_2b_jl:
                jl_2b_descriptors = self.create_2b_jl(io)
            if self._do_3b_jl:
                jl_3b_descriptors = self.create_3b_jl(io)

            if self._do_2b_jl and not self._do_3b_jl:
                jl_descriptors[io] = jl_2b_descriptors
            elif not self._do_2b_jl and self._do_3b_jl:
                jl_descriptors[io] = jl_3b_descriptors
            else:
                jl_descriptors[io] = np.append(jl_2b_descriptors, jl_3b_descriptors)

        return jl_descriptors

    def _initialize_distances(self, system: Atoms) -> None:
        """Build per-species neighbour lists within ``rcut`` for all centers.

        Populates the ``nn_elem_*`` attributes consumed by the per-term
        builders.

        Parameters
        ----------
        system : ase.Atoms
            Structure to search for neighbours of each center.
        """

        nn_elem_list = []
        nn_elem_dist = []
        nn_elem_vec = []
        nn_elem_start = []
        nn_elem_end = []
        nn_elem_num = []

        chem_symbols = np.asarray(system.get_chemical_symbols())

        for elem in self.species:
            elem_system = system.copy()
            del elem_system[(chem_symbols != elem).nonzero()[0]]

            nn_index, nn_dist, nn_vec, nn_start, nn_end, nn_num = get_nn_in_sphere(
                self.centers,
                elem_system.get_positions(),
                elem_system.get_cell().array,
                elem_system.get_pbc().astype(int),
                self._rcut,
                self._nn_leaf_size,
            )

            nn_elem_list.append(nn_index.copy())
            nn_elem_dist.append(nn_dist.copy())
            nn_elem_vec.append(nn_vec.copy())
            nn_elem_start.append(nn_start.copy())
            nn_elem_end.append(nn_end.copy())
            nn_elem_num.append(nn_num.copy())

        self.nn_elem_list = nn_elem_list
        self.nn_elem_dist = nn_elem_dist
        self.nn_elem_vec = nn_elem_vec
        self.nn_elem_start = nn_elem_start
        self.nn_elem_end = nn_elem_end
        self.nn_elem_num = nn_elem_num

    def create_2b_jl(self, grid_point_index: int) -> np.ndarray | list:
        """One-body (1B) radial JL coefficients for a single center.

        The method name keeps the older atom-centered ``2b`` label; in the
        grid-centered JLCDM convention this is the 1B term.

        Parameters
        ----------
        grid_point_index : int
            Index of the center (into ``self.centers``) to describe.

        Returns
        -------
        numpy.ndarray or list
            Concatenated coefficient vector if ``vector`` is ``True``, else a
            list of per-species blocks.
        """

        coeff_2b_matrix = []

        for ispec in range(self._n_species):
            if self.nn_elem_num[ispec][grid_point_index] == 0:
                coeff_2b_matrix.append(np.zeros(self._nmax_1b))

            else:
                distance_centers_i = self.nn_elem_dist[ispec][
                    self.nn_elem_start[ispec][grid_point_index] : self.nn_elem_end[
                        ispec
                    ][grid_point_index]
                ]

                jacobi_ni = expand_jacobi(
                    distance_centers_i,
                    self._nmax_1b,
                    self._alpha_1b,
                    self._beta_1b,
                    self._rcut,
                    self._rmin,
                    self._gamma,
                    shifted=int(self._shifted),
                    double_shifted=0,
                )

                coeff_2b_matrix.append(calculate_2b(jacobi_ni).copy())

        if not self._vector:
            return coeff_2b_matrix
        else:
            return np.concatenate(coeff_2b_matrix, axis=0)

    def create_3b_jl(self, grid_point_index: int) -> np.ndarray | list:
        """Two-body (2B) radial-angular JL coefficients for a single center.

        The method name keeps the older atom-centered ``3b`` label; in the
        grid-centered JLCDM convention this is the 2B term.

        Parameters
        ----------
        grid_point_index : int
            Index of the center (into ``self.centers``) to describe.

        Returns
        -------
        numpy.ndarray or list
            Concatenated coefficient vector if ``vector`` is ``True``, else a
            list of per-species-pair blocks.
        """

        coeff_3b_matrix = []

        for ispec, jspec in self.species_pair_index:
            if not (
                self.nn_elem_num[ispec][grid_point_index] > 0
                and self.nn_elem_num[jspec][grid_point_index] > 0
            ):
                if ispec == jspec:
                    coeff_3b_matrix.append(
                        np.zeros(self._n_upper_terms * (self._lmax + 1))
                    )
                else:
                    coeff_3b_matrix.append(
                        np.zeros(self._n_full_terms * (self._lmax + 1))
                    )

            else:
                vector_centers_i = self.nn_elem_vec[ispec][
                    self.nn_elem_start[ispec][grid_point_index] : self.nn_elem_end[
                        ispec
                    ][grid_point_index]
                ]
                distance_centers_i = self.nn_elem_dist[ispec][
                    self.nn_elem_start[ispec][grid_point_index] : self.nn_elem_end[
                        ispec
                    ][grid_point_index]
                ]

                vector_centers_i = get_versors(vector_centers_i, distance_centers_i)

                jacobi_ni_2b = expand_jacobi(
                    distance_centers_i,
                    self._nmax_2b,
                    self._alpha_2b,
                    self._beta_2b,
                    self._rcut,
                    0.0,
                    self._gamma,
                    shifted=int(self._shifted),
                    double_shifted=int(self._double_shifted),
                )

                if ispec == jspec:
                    legendre_lij = expand_legendre(
                        self._lmax, vector_centers_i, vector_centers_i, zero_diag=1
                    )
                    coeff_3b_jl = calculate_3b_upper(
                        jacobi_ni_2b, jacobi_ni_2b, legendre_lij
                    )

                else:
                    vector_centers_j = self.nn_elem_vec[jspec][
                        self.nn_elem_start[jspec][grid_point_index] : self.nn_elem_end[
                            jspec
                        ][grid_point_index]
                    ]
                    distance_centers_j = self.nn_elem_dist[jspec][
                        self.nn_elem_start[jspec][grid_point_index] : self.nn_elem_end[
                            jspec
                        ][grid_point_index]
                    ]

                    vector_centers_j = get_versors(vector_centers_j, distance_centers_j)

                    jacobi_nj_2b = expand_jacobi(
                        distance_centers_j,
                        self._nmax_2b,
                        self._alpha_2b,
                        self._beta_2b,
                        self._rcut,
                        0.0,
                        self._gamma,
                        shifted=int(self._shifted),
                        double_shifted=int(self._double_shifted),
                    )
                    legendre_lij = expand_legendre(
                        self._lmax, vector_centers_i, vector_centers_j, zero_diag=0
                    )
                    coeff_3b_jl = calculate_3b(jacobi_ni_2b, jacobi_nj_2b, legendre_lij)

                coeff_3b_matrix.append(coeff_3b_jl)

        if not self._vector:
            return coeff_3b_matrix
        else:
            return np.concatenate(coeff_3b_matrix, axis=0)
