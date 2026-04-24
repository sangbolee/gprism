"""Log-likelihood computation for the beta-binomial mixture model."""

import numpy as np
from scipy.special import betaln, logsumexp

from gprism.config import DEFAULT_PHI, DEFAULT_EPSILON
from gprism.utils.math import (
    generate_center_dict,
    generate_popaf_dict,
    log_nCk,
)


def log_likelihood(a_vec, X, N_vec, PopAF, phi=DEFAULT_PHI, epsilon=DEFAULT_EPSILON):
    """Compute the total log-likelihood of observed data under the mixture.

    Parameters
    ----------
    a_vec : array-like, shape (K,)
        Mixture proportion vector.
    X : array-like, shape (N,)
        Alternate allele counts at each locus.
    N_vec : array-like, shape (N,)
        Total read depths at each locus.
    PopAF : array-like, shape (N,)
        Population allele frequencies at each locus.
    phi : float or None
        Dispersion (concentration) parameter. If *None*, defaults to the
        median read depth (matches the manuscript definition).
    epsilon : float
        Numerical stability constant.

    Returns
    -------
    float
        Total log-likelihood across all loci.
    """
    if phi is None:
        phi = float(np.median(np.asarray(N_vec))) if len(N_vec) else DEFAULT_PHI

    centers = generate_center_dict(a_vec)
    K = len(a_vec)
    total_ll = 0.0

    for x_i, n_i, p_i in zip(X, N_vec, PopAF):
        popaf_center = generate_popaf_dict(p_i, K)
        log_pmf = []

        for combo, center in centers.items():
            alpha = phi * center + epsilon
            beta = phi * (1 - center) + epsilon

            ll = (
                np.log(popaf_center[combo])
                + log_nCk(n_i, x_i)
                + betaln(x_i + alpha, n_i - x_i + beta)
                - betaln(alpha, beta)
            )
            log_pmf.append(ll)

        total_ll += logsumexp(log_pmf)

    return total_ll
