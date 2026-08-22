# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import numpy as np
cimport numpy as np
cimport cython

from jlgridfingerprints.lib.utils import vector_dot
from libc.math cimport pi,cos,log1p

def expand_jacobi(double[::1] rgi, int nmax, double alpha, double beta, double rcut, double rmin=0, double gamma=1, bint shifted=1, bint double_shifted=0, bint slope_shifted=0, str radial_map='cosine', double rsoft=0.0):
    """Vanishing-Jacobi radial expansion of neighbour distances.

    See the JLCDM paper (DOI:10.1038/s41524-023-01053-0) for the definitions.

    Parameters
    ----------
    rgi : numpy.ndarray
        Neighbour distances (1-D, length n_neighbours).
    nmax : int
        Maximum Jacobi order.
    alpha, beta : float
        Jacobi weighting-function parameters, real-valued with domain
        ``alpha > -1`` and ``beta > -1``. Not validated here: this function
        runs once per grid center, so the domain check belongs to the caller.
    rcut : float
        Cutoff radius. Every radial map sends ``rcut`` to ``x = -gamma``, and
        orders ``>= 1`` are zeroed beyond it.
    rmin : float, optional
        Lower edge of the cosine map, i.e. the distance sent to ``x = +gamma``.
        Negative values push that map's point of vanishing derivative to
        inaccessible negative distances. Must be 0 for ``radial_map='log'``,
        which has no such point; use ``rsoft`` there instead. Default 0.
    gamma : float, optional
        Scaling factor applied to the mapped variable, i.e. the half-width of
        the interval ``[-gamma, +gamma]`` the expansion runs over. Must be
        strictly positive. Default 1.
    shifted : bool, optional
        Use the vanishing-Jacobi polynomials. Default True.
    double_shifted : bool, optional
        Use the double-vanishing-Jacobi polynomials, whose orders start at 2
        instead of 1. Default False.
    slope_shifted : bool, optional
        Constrain the basis to vanish at ``rcut`` in value *and* first
        derivative with respect to distance, rather than in value alone (see
        Notes). Like ``double_shifted`` it consumes two orders and returns
        orders 2..nmax, and it is mutually exclusive with it: the two impose
        different conditions at different points. Requires ``shifted``, which
        it extends rather than replaces, and ``nmax >= 2``. Default False.
    radial_map : str, optional
        Which coordinate carries distance to the Jacobi variable: ``'cosine'``
        or ``'log'`` (see Notes). Default ``'cosine'``.
    rsoft : float, optional
        Softening length, in the units of ``rcut``: the radius below which
        ``radial_map='log'`` is linear in distance rather than logarithmic.
        Required (``> 0``) for that map and rejected for ``'cosine'``.
        Default 0.0, which means "unset".

    Returns
    -------
    numpy.ndarray
        Jacobi expansion of shape ``(n_orders, n_neighbours)``, where
        ``n_orders`` is ``nmax`` (or ``nmax - 1`` when ``double_shifted`` or
        ``slope_shifted``, each of which consumes two orders).

    Raises
    ------
    ValueError
        If ``radial_map`` is not one of the two accepted names; if ``rsoft`` is
        set for ``'cosine'`` or unset for ``'log'``; if ``rmin`` is non-zero
        for ``'log'``; if any distance is negative under ``'log'``; or if
        ``slope_shifted`` is combined with ``double_shifted``, set without
        ``shifted``, or used with ``nmax < 2``.

    Notes
    -----
    Both maps satisfy one contract, which is all that anything downstream of
    the map depends on, and which a future third map must also satisfy:

        A radial map carries a distance on ``[0, rcut]`` to ``x`` on
        ``[-gamma, +gamma]``, monotonically *decreasing*, with
        ``x(rmin) = +gamma`` and ``x(rcut) = -gamma``.

    The orientation is the load-bearing half: ``x`` decreases as the distance
    grows, which is why ``shifted`` subtracting ``P_n(-gamma)`` is what makes
    the basis vanish at the cutoff, under either map and with no map-specific
    code of its own. The two maps are::

        cosine:  x(r) = gamma * cos(pi * (r - rmin) / (rcut - rmin))
        log:     x(r) = gamma * (1 - 2 * log1p(r / rsoft) / log1p(rcut / rsoft))

    They differ in where they spend their range. The cosine map concentrates it
    in the middle of the interval and is stationary at both ends, so a region
    close to ``rmin`` receives almost none of it -- fine for a field that varies
    in the interstitial region, useless for one with structure at the nucleus.
    The logarithmic map gives equal ratios of distance equal stretches of ``x``,
    so the near-origin region receives a large share. ``rsoft`` is what keeps it
    finite at ``r = 0``, where ``log r`` is not, and tunes the trade: large
    ``rsoft`` approaches a map linear in distance, small ``rsoft`` one linear in
    log distance.

    That difference is what ``slope_shifted`` exists for. ``shifted`` makes the
    basis vanish in value at ``rcut`` under either map, but a vanishing *radial
    slope* there is a different condition: ``df/dr = (df/dx)(dx/dr)``, so it
    holds for free wherever the map's own derivative vanishes at the cutoff.
    The cosine map is stationary at both ends and supplies it silently for any
    ``rmin``; the logarithmic map is strictly monotone and does not. A caller
    who needs a basis that joins something else at ``rcut`` without a kink --
    a model split at a radius, say -- therefore gets it for free under the
    cosine map and must ask for it under the logarithmic one.

    ``slope_shifted`` imposes both conditions at the single anchor ``x = -gamma``
    by building ``(gamma + x)**2 * P_k^(alpha, beta+2)(x)`` for
    ``k = 0..nmax-2``. Any polynomial with a double root at ``x = -gamma``
    factors that way, so this spans the same space as subtracting orders 0 and 1
    of the raw basis would, but it is far better behaved per function: the
    subtractive form multiplies order 1 by a coefficient growing like ``n**2``,
    which swamps the polynomial itself and distorts any penalty on the fitted
    coefficients.
    """

    # gamma is the half-width of the interval the expansion lives on, so it has
    # to be strictly positive: gamma = 0 collapses [-gamma, +gamma] to the single
    # point 0, and gamma < 0 reverses the interval and breaks the map's
    # orientation. Checked here rather than left to the arithmetic because
    # cdivision=True means the double_shifted normalisation divides by 2*gamma
    # without raising -- at gamma = 0 every returned order would be a silent NaN.
    if gamma <= 0:
        raise ValueError("Only positive 'gamma' are allowed.")

    cdef int ndist = rgi.shape[0]
    cdef double[::1] dist = rgi
    cdef int deg_max = nmax + 1

    cdef int i = 0
    cdef int n = 0
    cdef double d = 0
    cdef double theta0 = 0
    cdef int map_code = 0
    cdef double log_norm = 0.0

    cdef np.ndarray[dtype=double,ndim=1] cos_theta = np.empty(ndist,dtype=np.double)
    cdef double[::1] cos_theta0 = cos_theta

    cdef np.ndarray[dtype=double,ndim=2] pjacobi = np.empty((deg_max,ndist),dtype=np.double)
    cdef double[:,::1] vj = pjacobi

    if slope_shifted:
        if double_shifted:
            raise ValueError(
                "'slope_shifted' and 'double_shifted' are mutually exclusive: they impose "
                "different conditions (value+slope at rcut, versus value at each end)."
            )
        if not shifted:
            raise ValueError(
                "'slope_shifted' requires 'shifted': the slope condition extends the value "
                "condition rather than replacing it."
            )
        if nmax < 2:
            raise ValueError(
                "'slope_shifted' consumes two orders, so it requires nmax >= 2; "
                f"got nmax={nmax}."
            )

    if radial_map == 'cosine':
        map_code = 0
        if rsoft != 0.0:
            raise ValueError("'rsoft' has no meaning for radial_map='cosine'; leave it unset.")
    elif radial_map == 'log':
        map_code = 1
        if rsoft <= 0.0:
            raise ValueError("radial_map='log' requires a positive 'rsoft' (the softening length).")
        if rmin != 0.0:
            raise ValueError("radial_map='log' requires rmin == 0; use 'rsoft' to set the near-core resolution.")
        log_norm = log1p(rcut / rsoft)
    else:
        raise ValueError(f"Unknown 'radial_map' {radial_map!r}: expected 'cosine' or 'log'.")

    if map_code == 0:
        for i in range(ndist):
            theta0 = pi * (dist[i] - rmin) / (rcut - rmin)
            cos_theta0[i] = gamma * cos(theta0)
    else:
        for i in range(ndist):
            # Negative distances are undefined under this map and would fail
            # silently: between -rsoft and 0 they land outside [-gamma, +gamma],
            # where the Jacobi polynomials are no longer orthogonal under their
            # own weight and grow fast; at or below -rsoft they are -inf or NaN,
            # which cdivision=True will not catch either.
            if dist[i] < 0.0:
                raise ValueError("radial_map='log' requires non-negative distances.")
            cos_theta0[i] = gamma * (1.0 - 2.0 * log1p(dist[i] / rsoft) / log_norm)

    calculate_jacobi(nmax, alpha, beta, cos_theta0, gamma, shifted, double_shifted, slope_shifted, vj)

    for i in range(ndist):
        if dist[i] > rcut:
            for n in range(1,deg_max):
                vj[n,i] = d

    if double_shifted or slope_shifted:
        return pjacobi[2:,:]
    else:
        return pjacobi[1:,:]

def expand_legendre(int lmax, double[:,::1] hat_rgi, double[:,::1] hat_rgj, bint zero_diag=1):
    """Legendre expansion of the angles between pairs of neighbour versors.

    See the JLCDM paper (DOI:10.1038/s41524-023-01053-0) for the definitions.

    Parameters
    ----------
    lmax : int
        Maximum Legendre order.
    hat_rgi : numpy.ndarray
        Unit vectors from the center to the species-i neighbours, shape
        ``(n_i, 3)``.
    hat_rgj : numpy.ndarray
        Unit vectors from the center to the species-j neighbours, shape
        ``(n_j, 3)``.
    zero_diag : bool, optional
        Zero the diagonal of the pair matrix (used when i and j are the same
        neighbour set, to drop self-pairs). Default True.

    Returns
    -------
    numpy.ndarray
        Legendre expansion of shape ``(lmax + 1, n_i, n_j)`` evaluated on the
        scalar products of the neighbour versors.
    """

    cdef int num_n = hat_rgi.shape[0]
    cdef int num_m = hat_rgj.shape[0]
    cdef int deg_max = lmax + 1

    cdef int io = 0
    cdef int jo = 0

    cdef np.ndarray[dtype=double,ndim=2] rhatdot = np.empty((num_n,num_m),dtype=np.double)
    cdef double[:,::1] prod = rhatdot

    cdef np.ndarray[dtype=double,ndim=3] plegendre = np.empty((deg_max,num_n,num_m),dtype=np.double)
    cdef double[:,:,::1] vl = plegendre

    # maybe we could do this inside the calculate legendre so that we don't need the double loop twice
    # need testing
    vector_dot(hat_rgi, hat_rgj, prod)

    for io in range(num_n):
        for jo in range(num_m):
            if prod[io,jo] > 1.0: prod[io,jo] = 1.0
            elif prod[io,jo] < -1.0: prod[io,jo] = -1.0

    legendre_eval(lmax, rhatdot, zero_diag, vl)

    return plegendre

cdef void calculate_jacobi(int nmax,double alpha,double beta,double [::1]x, double gamma, bint shifted, bint double_shifted, bint slope_shifted, double[:,::1] jac):

    cdef int i = 0
    cdef int deg = 1
    cdef int deg_max = nmax + 1
    cdef int ndist = x.shape[0]

    cdef np.ndarray[dtype=np.double_t,ndim=1] pjacobi0 = np.empty((deg_max),dtype=np.double)
    cdef double[::1] pj0 = pjacobi0

    cdef np.ndarray[dtype=np.double_t,ndim=1] pjacobi1 = np.empty((deg_max),dtype=np.double)
    cdef double[::1] pj1 = pjacobi1

    cdef double s = 0
    cdef double p1x = 0
    cdef double gfac = 0
    cdef double w = 0

    if slope_shifted:
        # Value and first derivative both vanish at the anchor x = -gamma, built
        # as (gamma+x)**2 * P_k^(alpha,beta+2) for k = 0..nmax-2. Every polynomial
        # with a double root at -gamma factors that way, so this spans the same
        # space as the subtractive form Phat_n = P_n - a_n - b_n*P_1, but without
        # its conditioning problem: there b_n = P_n'(-gamma)/P_1' grows like n**2
        # (-4.14 at n=2, -5282 at n=24) and b_n*P_1 swamps P_n itself. Plain least
        # squares cannot tell the two apart; a coefficient penalty can, and this
        # is the form whose penalty is not distorted by an n**2 factor.
        #
        # The k-th function lands in row k+2, so the caller's existing
        # `return pjacobi[2:,:]` yields orders 2..nmax -- the same block width
        # double_shifted returns. Rows are filled descending so that jac[deg-2]
        # is still the raw value when it is read.
        jacobi_eval(nmax-2, alpha, beta+2.0, x, jac)

        for i in range(ndist):
            w = gamma + x[i]
            w = w * w
            for deg in range(deg_max-1, 1, -1):
                jac[deg,i] = w * jac[deg-2,i]

        # Return before the `shifted` block below: `shifted` is subsumed here,
        # not forgotten. The (gamma+x)**2 factor already makes every function
        # vanish in value at x = -gamma, so subtracting P_n(-gamma) as well
        # would subtract zero from a basis built to be zero there -- and would
        # break it, since the subtraction is applied to jac rows that no longer
        # hold raw P_n. `slope_shifted` still *requires* `shifted` so that the
        # contradictory request (slope condition without value condition) is
        # rejected rather than silently reinterpreted.
        return

    jacobi_eval(nmax, alpha, beta, x, jac)

    if shifted:
        jacobi_eval_single(nmax, alpha, beta,-1 * gamma, pj0)

        for i in range(ndist):
            for deg in range(1,deg_max):
                s = jac[deg,i] - pj0[deg]
                jac[deg,i] = s

        if double_shifted:
            # Anchor the second vanishing point at the interval's own upper end,
            # x = +gamma, not at x = +1. The two coincide only at gamma = 1.
            # pj1 and the gfac denominator are two halves of one anchor: gfac is
            # the closed form of P1~(x)/P1~(x_up) = (x+gamma)/(x_up+gamma), so
            # changing either alone gives a basis vanishing at neither endpoint.
            jacobi_eval_single(nmax, alpha, beta, gamma, pj1)

            for i in range(ndist):
                for deg in range(2,deg_max):
                    gfac = (gamma+x[i])/(2.0*gamma)
                    p1x = pj1[deg]-pj0[deg]
                    s = jac[deg,i] - gfac * p1x
                    jac[deg,i] = s


cdef void jacobi_eval_single(int nmax,double alpha, double beta, double x, double[::1] jac):
    
    cdef int deg_max = nmax + 1

    cdef double val = 1
    cdef double a = 0
    cdef double b = 0
    cdef double c = 0
    cdef double v1 = 0
    cdef double v2 = 0
    cdef int deg = 0

    jac[0] = val
    
    for deg in range(1,deg_max):
        a = deg + alpha
        b = deg + beta
        c = a + b
        if deg == 1:
            val = 0.5*(x-1.0)
            val *= c
            val += a
        else:
            v1 = -2.0*c * (a-1.0)*(b-1.0) * jac[deg-2]
            v2 = c*(c-2.0)*x + (a-b)*(c-2.0*deg)
            v2 *= (c-1.0)
            v2 *= jac[deg-1]
        
            v1 += v2
            v2 = 2.0*deg*(c-deg)*(c-2.0)
            val = v1/v2

        jac[deg] = val


cdef void jacobi_eval(int nmax,double alpha, double beta, double [::1]x, double[:,::1] jac):
    
    cdef int deg_max = nmax + 1

    cdef int i = 0
    cdef int ndist = x.shape[0]

    cdef double val = 1
    cdef double a = 0
    cdef double b = 0
    cdef double c = 0
    cdef double v1 = 0
    cdef double v2 = 0
    cdef double xi = 0
    cdef int deg = 0

    for i in range(ndist):
        jac[0,i] = val

    for i in range(ndist):
        val = 1
        v1 = 0
        v2 = 0
        xi = x[i]
        for deg in range(1,deg_max):
            a = deg + alpha
            b = deg + beta
            c = a + b
            if deg == 1:
                val = 0.5*(xi-1.0)
                val *= c
                val += a
            else:
                v1 = -2.0*c * (a-1.0)*(b-1.0) * jac[deg-2,i]
                v2 = c*(c-2.0)*xi + (a-b)*(c-2.0*deg)
                v2 *= (c-1.0)
                v2 *= jac[deg-1,i]
            
                v1 += v2
                v2 = 2.0*deg*(c-deg)*(c-2.0)
                val = v1/v2

            jac[deg,i] = val


cdef void legendre_eval(int lmax, double [:,::1]x, bint zero_diag, double[:,:,::1] jac):
    
    cdef int deg_max = lmax + 1

    cdef int i = 0
    cdef int j = 0
    cdef int idist = x.shape[0]
    cdef int jdist = x.shape[1]

    cdef double val = 1
    cdef double a = 0
    cdef double b = 0
    cdef double c = 0
    cdef double v1 = 0
    cdef double v2 = 0
    cdef double xij = 0
    cdef int deg = 0

    for i in range(idist):
        for j in range(jdist):
            jac[0,i,j] = val

    for i in range(idist):
        for j in range(jdist):
            if zero_diag and j <= i:
                continue
            val = 1
            v1 = 0
            v2 = 0
            xij = x[i,j]
            for deg in range(1,deg_max):
                a = deg
                b = deg
                c = a + b
                if deg == 1:
                    val = 0.5*(xij-1.0)
                    val *= c
                    val += a
                else:
                    v1 = -2.0*c * (a-1.0)*(b-1.0) * jac[deg-2,i,j]
                    v2 = c*(c-2.0)*xij + (a-b)*(c-2.0*deg)
                    v2 *= (c-1.0)
                    v2 *= jac[deg-1,i,j]
                
                    v1 += v2
                    v2 = 2.0*deg*(c-deg)*(c-2.0)
                    val = v1/v2

                jac[deg,i,j] = val

    if zero_diag:
        for i in range(idist):
            for j in range(jdist):
                for deg in range(deg_max):
                    if i == j:
                        jac[deg,i,j] = 0.0
                    elif j < i:
                        jac[deg,i,j] = jac[deg,j,i]
            