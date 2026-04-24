"""Generalized EM algorithm for beta-binomial mixture deconvolution."""

import warnings

import numpy as np
import torch
from torch.distributions import Dirichlet

from gprism.config import DEFAULT_CONFIG
from gprism.utils.math import build_combination_matrix, build_popaf_matrix


_LOW_PHI_WARNED: set = set()


def _resolve_phi_and_min_depth(all_depths, config, file_path=None):
    """Compute phi (user-fixed or median of unfiltered depths), warn if low,
    and return the adaptive min_depth threshold.

    The warning fires at most once per (file_path, threshold) tuple within
    a single Python process to avoid flooding stderr on repeated calls.
    """
    phi = config.effective_phi(all_depths)
    if phi <= config.phi_low_warning_threshold:
        key = (file_path, config.phi_low_warning_threshold)
        if key not in _LOW_PHI_WARNED:
            warnings.warn(
                f"phi={phi:.1f} <= {config.phi_low_warning_threshold}: "
                "too low depth for the algorithm; results may be unreliable.",
                UserWarning,
                stacklevel=3,
            )
            _LOW_PHI_WARNED.add(key)
    min_depth = config.resolve_min_depth(phi)
    return phi, min_depth


def load_filtered_data(file_path, config=None):
    """Load and filter variant data from a .biallelic.mpileup file.

    Returns
    -------
    X, N_vec, PopAF : np.ndarray
        Filtered alternate counts, depths, and population AFs.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Pass 1: read all depths (no filtering) to resolve phi and adaptive min_depth.
    all_depths = []
    with open(file_path, mode="r") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            all_depths.append(int(line.strip().split("\t")[4]))
    all_depths_np = np.asarray(all_depths, dtype=np.int64)

    _, min_depth = _resolve_phi_and_min_depth(all_depths_np, config, file_path)

    # Pass 2: apply filters using the adaptive min_depth.
    X, N_vec, PopAF = [], [], []
    with open(file_path, mode="r") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            toks = line.strip().split("\t")
            depth = int(toks[4])
            paf = float(toks[7])

            if depth < min_depth:
                continue
            if not (config.popaf_range[0] < paf < config.popaf_range[1]):
                continue

            X.append(int(toks[5]))
            N_vec.append(depth)
            PopAF.append(paf)

    return (
        np.array(X, dtype=np.int64),
        np.array(N_vec, dtype=np.int64),
        np.array(PopAF, dtype=np.float32),
    )


def load_filtered_data_with_contigs(file_path, config=None):
    """Load variant data with genomic coordinates.

    Returns
    -------
    X, N_vec, PopAF : np.ndarray
    contigs : list of (chrom, pos) tuples
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Pass 1: read all depths (no filtering) to resolve phi and adaptive min_depth.
    all_depths = []
    with open(file_path, mode="r") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            all_depths.append(int(line.strip().split("\t")[4]))
    all_depths_np = np.asarray(all_depths, dtype=np.int64)

    _, min_depth = _resolve_phi_and_min_depth(all_depths_np, config, file_path)

    # Pass 2: apply filters using the adaptive min_depth.
    X, N_vec, PopAF, contigs = [], [], [], []
    with open(file_path, mode="r") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            toks = line.strip().split("\t")
            depth = int(toks[4])
            paf = float(toks[7])

            if depth < min_depth:
                continue
            if not (config.popaf_range[0] < paf < config.popaf_range[1]):
                continue

            contigs.append((toks[0], toks[1]))
            X.append(int(toks[5]))
            N_vec.append(depth)
            PopAF.append(paf)

    return (
        np.array(X, dtype=np.int64),
        np.array(N_vec, dtype=np.int64),
        np.array(PopAF, dtype=np.float32),
        contigs,
    )


def run_em(file_path, K, config=None):
    """Run the generalised EM algorithm for a given number of contributors.

    Parameters
    ----------
    file_path : str
        Path to the ``.biallelic.mpileup`` file.
    K : int
        Number of contributors to fit.
    config : GPRISMConfig, optional

    Returns
    -------
    dict or None
        ``{'a_vec', 'centers', 'weights', 'bic', 'loglike'}`` on success,
        *None* when the file yields no usable variants.
    """
    if config is None:
        config = DEFAULT_CONFIG

    epsilon = config.epsilon
    lr = config.lr
    max_em_iter = config.max_em_iter

    # --- Device ---
    if config.device:
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    X_np, N_vec_np, PopAF_np = load_filtered_data(file_path, config)
    if len(X_np) == 0:
        return None
    N = len(X_np)

    # Dispersion: fixed or median-depth (per manuscript)
    phi = config.effective_phi(N_vec_np)

    # Compute resolution from mean depth
    mean_depth = float(N_vec_np.mean())
    resolution = config.resolution(mean_depth)

    X_t = torch.from_numpy(X_np).to(device)
    N_vec_t = torch.from_numpy(N_vec_np).to(device)

    # --- Build combination and popaf matrices ---
    combos, comb_matrix_np = build_combination_matrix(K)
    popaf_mat_np = build_popaf_matrix(PopAF_np, K)

    comb_matrix_t = torch.from_numpy(comb_matrix_np).to(device)
    popaf_mat_t = torch.from_numpy(popaf_mat_np).to(device)

    eps = 1e-30

    # --- Initialise theta ---
    alpha_prior = torch.ones(K, device=device, dtype=torch.float32)
    dist = Dirichlet(alpha_prior)
    theta = dist.sample().log().clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([theta], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # --- EM loop ---
    decrease_count = 0
    prev_L = None
    L = None
    a_vec = None
    weight = None

    for em_iter in range(max_em_iter):
        # a_vec with resolution floor
        p = torch.softmax(theta, dim=0)
        a_vec = resolution + (1 - K * resolution) * p

        # E-step
        centers = comb_matrix_t.matmul(a_vec)
        alpha = phi * centers + epsilon
        beta = phi * (1 - centers) + epsilon

        X_ext = X_t.view(N, 1).float()
        NmX_ext = (N_vec_t - X_t).view(N, 1).float()
        alpha_ext = alpha.view(1, -1)
        beta_ext = beta.view(1, -1)

        log_nCk = (
            torch.lgamma(N_vec_t.float().view(N, 1) + 1)
            - torch.lgamma(X_ext + 1)
            - torch.lgamma(NmX_ext + 1)
        ).expand(N, alpha_ext.shape[1])

        term1 = (
            torch.lgamma(X_ext + alpha_ext)
            + torch.lgamma(NmX_ext + beta_ext)
            - torch.lgamma((X_ext + alpha_ext) + (NmX_ext + beta_ext))
        )
        term2 = (
            torch.lgamma(alpha_ext)
            + torch.lgamma(beta_ext)
            - torch.lgamma(alpha_ext + beta_ext)
        ).expand(N, alpha_ext.shape[1])

        log_bb = log_nCk + term1 - term2
        log_popaf = torch.log(popaf_mat_t + eps)
        log_resp = log_popaf + log_bb
        log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
        resp = torch.exp(log_resp - log_norm)

        # M-step
        optimizer.zero_grad()
        Q = -torch.sum(resp.detach() * log_bb)
        Q.backward()
        optimizer.step()

        weight = resp.sum(dim=0) / N

        # Monitor convergence
        with torch.no_grad():
            L = torch.sum(torch.logsumexp(log_popaf + log_bb, dim=1))

        if prev_L is not None:
            if L < prev_L - 1:
                decrease_count += 1
            else:
                decrease_count = 0
            rel_change = torch.abs((L - prev_L) / (prev_L + 1e-300))
            if rel_change < config.convergence_tol or decrease_count >= config.max_decrease_count:
                break
        prev_L = L

    # Final output
    a_final = a_vec.detach().cpu().numpy()
    centers_final = comb_matrix_t.matmul(a_vec).detach().cpu().numpy()
    weights_final = weight.detach().cpu().numpy()
    loglike = L.item()
    bic = (-loglike + ((K - 1) / 2) * np.log(N)) * 2

    return {
        "a_vec": a_final,
        "centers": centers_final,
        "weights": weights_final,
        "bic": bic,
        "loglike": loglike,
    }
