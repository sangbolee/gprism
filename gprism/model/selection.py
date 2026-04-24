"""BIC-based model selection across candidate numbers of contributors."""

import time
import numpy as np

from gprism.config import DEFAULT_CONFIG
from gprism.model.likelihood import log_likelihood
from gprism.model.em import run_em, load_filtered_data


def select_optimal_k(file_path, sample_name, config=None):
    """Test K=1..max_k and select the optimal number of contributors.

    Parameters
    ----------
    file_path : str
        Path to the ``.biallelic.mpileup`` file.
    sample_name : str
        Sample identifier (for logging).
    config : GPRISMConfig, optional

    Returns
    -------
    dict[int, dict]
        Results keyed by K, each containing ``'a_vec'``, ``'bic'``, and
        (for K > 1) ``'centers'``, ``'weights'``, ``'loglike'``.
    """
    if config is None:
        config = DEFAULT_CONFIG

    start = time.time()
    results = {}

    # K = 1: closed-form solution
    X, N_vec, PopAF = load_filtered_data(file_path, config)
    n = len(X)
    if n == 0:
        print(f"[{sample_name}] No variants after filtering.")
        return results

    phi = config.effective_phi(N_vec)

    a = np.array([1.0])
    ll = log_likelihood(a, X, N_vec, PopAF, phi=phi, epsilon=config.epsilon)
    bic = (-float(ll) + (0 / 2) * np.log(n)) * 2  # p = K-1 = 0
    results[1] = {"a_vec": a, "bic": bic}
    print(f"[{sample_name}, K=1] BIC: {bic:.2f}")

    # K = 2..max_k: EM with multiple initialisations
    for K in range(2, config.max_k + 1):
        best_result = None
        best_bic = float("inf")

        for trial in range(config.n_init):
            result = run_em(file_path, K, config)
            if result is None:
                continue
            if result["bic"] < best_bic:
                best_bic = result["bic"]
                best_result = result

        if best_result is not None:
            results[K] = best_result
            print(f"[{sample_name}, K={K}] BIC: {best_result['bic']:.2f}")

    elapsed = (time.time() - start) / 60
    print(f"[{sample_name}] Completed in {elapsed:.2f} min")
    return results
