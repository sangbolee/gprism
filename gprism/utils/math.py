"""Shared mathematical utilities for gPRISM.

Contains all beta-binomial, genotype combination, and population allele
frequency functions used across the likelihood, EM, and posterior modules.
"""

import math
import itertools
import numpy as np
from scipy.special import betaln, gammaln


# ---------------------------------------------------------------------------
# 1. Genotype combination enumeration
# ---------------------------------------------------------------------------

GENOTYPE_STATES = ("ref", "hetero", "homo")


def generate_center_dict(a_vec):
    """Generate the expected VAF for every possible genotype combination.

    For K contributors with mixture proportions *a_vec*, each contributor
    can carry one of three diploid genotype states. The expected VAF for
    combination *combo* is the weighted sum of allelic dosages:
        0 for ref, a_k/2 for hetero, a_k for homo.

    Returns
    -------
    dict[tuple[str,...], float]
        Mapping from genotype-state tuple (length K) to expected VAF.
    """
    K = len(a_vec)
    center_dict = {}
    for combo in itertools.product(GENOTYPE_STATES, repeat=K):
        value = sum(
            0.0 if state == "ref"
            else a_vec[i] / 2 if state == "hetero"
            else a_vec[i]
            for i, state in enumerate(combo)
        )
        center_dict[combo] = value
    return center_dict


def generate_popaf_dict(popaf, K):
    """Compute Hardy-Weinberg genotype priors for K independent contributors.

    Parameters
    ----------
    popaf : float
        Population minor-allele frequency at a given SNP locus.
    K : int
        Number of contributors.

    Returns
    -------
    dict[tuple[str,...], float]
        Prior probability for each genotype combination.
    """
    p = popaf
    single_locus = {
        "ref":    (1 - p) ** 2,
        "hetero": 2 * p * (1 - p),
        "homo":   p ** 2,
    }
    prob_dict = {}
    for combo in itertools.product(GENOTYPE_STATES, repeat=K):
        prob_dict[combo] = math.prod(single_locus[s] for s in combo)
    return prob_dict


# ---------------------------------------------------------------------------
# 2. Beta-binomial functions
# ---------------------------------------------------------------------------

def log_nCk(n, k):
    """Log binomial coefficient log(C(n, k)) via the log-gamma function."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def beta_binomial_logpmf(x, n, alpha, beta):
    """Log probability mass function of the beta-binomial distribution."""
    return (
        log_nCk(n, x)
        + betaln(x + alpha, n - x + beta)
        - betaln(alpha, beta)
    )


# ---------------------------------------------------------------------------
# 3. Combination matrix builders (vectorised, for EM / posterior)
# ---------------------------------------------------------------------------

def build_combination_matrix(K):
    """Build the (M, K) allelic-dosage matrix for all 3^K genotype combos.

    Entry (j, k) is 0.0 (ref), 0.5 (hetero), or 1.0 (homo).

    Returns
    -------
    combos : list[tuple[str,...]]
    comb_matrix : np.ndarray, shape (M, K), dtype float32
    """
    combos = list(itertools.product(GENOTYPE_STATES, repeat=K))
    M = len(combos)
    comb_matrix = np.zeros((M, K), dtype=np.float32)
    for j, combo in enumerate(combos):
        for k, state in enumerate(combo):
            if state == "hetero":
                comb_matrix[j, k] = 0.5
            elif state == "homo":
                comb_matrix[j, k] = 1.0
    return combos, comb_matrix


def build_popaf_matrix(PopAF, K):
    """Build the (N, M) prior-probability matrix for N loci and 3^K combos.

    Parameters
    ----------
    PopAF : np.ndarray, shape (N,)
    K : int

    Returns
    -------
    np.ndarray, shape (N, M), dtype float32
    """
    combos = list(itertools.product(GENOTYPE_STATES, repeat=K))
    M = len(combos)
    N = len(PopAF)

    p = PopAF.astype(np.float32)
    P0 = (1 - p) ** 2  # ref
    P1 = 2 * p * (1 - p)  # hetero
    P2 = p ** 2  # homo
    single_probs = np.stack([P0, P1, P2], axis=1)  # (N, 3)

    state_idx = np.zeros((M, K), dtype=np.int64)
    for j, combo in enumerate(combos):
        for k, state in enumerate(combo):
            state_idx[j, k] = GENOTYPE_STATES.index(state)

    popaf_mat = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        sp = single_probs[i]
        for j in range(M):
            popaf_mat[i, j] = np.prod(sp[state_idx[j, :]])

    return popaf_mat


# ---------------------------------------------------------------------------
# 4. Genotype index lookup (for posterior marginalisation)
# ---------------------------------------------------------------------------

def build_genotype_indices(combos, K):
    """Pre-compute per-contributor genotype index lists.

    Returns
    -------
    ref_indices, hetero_indices, homo_indices : list[list[int]]
        Each is length K; element k is the list of combo indices where
        contributor k has that genotype state.
    """
    ref_indices = [[] for _ in range(K)]
    hetero_indices = [[] for _ in range(K)]
    homo_indices = [[] for _ in range(K)]

    for comp_idx, combo in enumerate(combos):
        for k, gt in enumerate(combo):
            if gt == "ref":
                ref_indices[k].append(comp_idx)
            elif gt == "hetero":
                hetero_indices[k].append(comp_idx)
            elif gt == "homo":
                homo_indices[k].append(comp_idx)

    return ref_indices, hetero_indices, homo_indices
