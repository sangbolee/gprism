"""Posterior probability computation for genotype reconstruction."""

import numpy as np
from scipy.special import betaln, logsumexp

from gprism.config import DEFAULT_PHI, DEFAULT_EPSILON
from gprism.utils.math import (
    generate_center_dict,
    generate_popaf_dict,
    log_nCk,
)


def compute_posterior(a_vec, X, N_vec, PopAF, epsilon=DEFAULT_EPSILON, phi=DEFAULT_PHI):
    """Compute posterior genotype-cluster probabilities for each variant.

    Parameters
    ----------
    a_vec : array-like, shape (K,)
        Mixture proportions.
    X, N_vec, PopAF : array-like, shape (N,)
        Alternate counts, depths, and population allele frequencies.
    epsilon : float
        Numerical stability constant.
    phi : float or None
        Dispersion parameter. If *None*, defaults to the median read depth.

    Returns
    -------
    np.ndarray, shape (N, M)
        Posterior probability of each genotype cluster at each locus,
        where M = 3^K.
    """
    if phi is None:
        phi = float(np.median(np.asarray(N_vec))) if len(N_vec) else DEFAULT_PHI

    center_dict = generate_center_dict(a_vec)
    combos = list(center_dict.keys())
    combo_to_idx = {c: idx for idx, c in enumerate(combos)}
    centers = np.array([center_dict[c] for c in combos])
    M = len(centers)
    K = len(a_vec)

    alpha = phi * centers + epsilon
    beta = phi * (1 - centers) + epsilon

    N = len(X)
    posterior = np.zeros((N, M))

    for i in range(N):
        # Prior
        popaf_dict = generate_popaf_dict(PopAF[i], K)
        priors = np.zeros(M)
        for combo, prob in popaf_dict.items():
            priors[combo_to_idx[combo]] = prob
        log_prior = np.where(priors > 0, np.log(priors), -np.inf)

        # Likelihood
        log_lik = (
            log_nCk(N_vec[i], X[i])
            + betaln(X[i] + alpha, N_vec[i] - X[i] + beta)
            - betaln(alpha, beta)
        )

        # Normalised posterior. Guard against -inf priors that would make
        # logsumexp underflow and propagate NaN through the subtraction.
        log_joint = log_prior + log_lik
        finite = np.isfinite(log_joint)
        if finite.any():
            norm = logsumexp(log_joint[finite])
            normed = log_joint - norm
            posterior[i] = np.where(finite, np.exp(normed), 0.0)
        else:
            posterior[i] = 0.0

    return posterior
