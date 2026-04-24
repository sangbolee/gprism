"""Unified genotype reconstruction: hard calls and posterior probabilities."""

import numpy as np

from gprism.config import DEFAULT_CONFIG
from gprism.utils.math import (
    generate_center_dict,
    build_genotype_indices,
)
from gprism.genotype.posterior import compute_posterior
from gprism.model.em import load_filtered_data_with_contigs
from gprism.io.writer import (
    load_mixture_results,
    write_membership_hard,
    write_membership_posterior,
)


def reconstruct_genotypes(data_path, result_path, sample_name, output_path,
                          mode="posterior", config=None):
    """Reconstruct contributor genotypes from mixture deconvolution results.

    Parameters
    ----------
    data_path : str
        Path to ``<sample>.biallelic.mpileup``.
    result_path : str
        Path to ``<sample>.Mixture_Results.txt``.
    sample_name : str
        Sample identifier.
    output_path : str
        Path for the output ``<sample>.Membership.txt``.
    mode : str
        ``"hard"`` for discrete genotype calls, ``"posterior"`` for
        probability triplets (ref/hetero/homo).
    config : GPRISMConfig, optional
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Load mixture results and select optimal K
    results, optimal_k = load_mixture_results(result_path)
    a_vec = results[optimal_k]["a_vec"]
    K = len(a_vec)

    # Load variant data
    X, N_vec, PopAF, contigs = load_filtered_data_with_contigs(data_path, config)

    # Compute posteriors (phi resolved based on config mode + these depths)
    phi = config.effective_phi(N_vec)
    post = compute_posterior(a_vec, X, N_vec, PopAF,
                             epsilon=config.epsilon, phi=phi)
    n_samples, M = post.shape

    # Build genotype indices for marginalisation
    center_dict = generate_center_dict(a_vec)
    combos = list(center_dict.keys())
    ref_idx, het_idx, hom_idx = build_genotype_indices(combos, K)

    if mode == "posterior":
        _reconstruct_posterior(
            post, contigs, ref_idx, het_idx, hom_idx,
            K, n_samples, sample_name, output_path,
        )
    else:
        _reconstruct_hard(
            post, contigs, ref_idx, het_idx, hom_idx,
            K, n_samples, sample_name, output_path, config,
        )


def _reconstruct_posterior(post, contigs, ref_idx, het_idx, hom_idx,
                           K, n_samples, sample_name, output_path):
    """Output normalised posterior probabilities per contributor."""
    ref_prob = np.zeros((n_samples, K))
    het_prob = np.zeros((n_samples, K))
    hom_prob = np.zeros((n_samples, K))

    for j in range(n_samples):
        pj = post[j]
        for k in range(K):
            pr = pj[ref_idx[k]].sum()
            ph = pj[het_idx[k]].sum()
            pm = pj[hom_idx[k]].sum()
            total = pr + ph + pm
            if total > 0:
                pr /= total
                ph /= total
                pm /= total
            ref_prob[j, k] = pr
            het_prob[j, k] = ph
            hom_prob[j, k] = pm

    write_membership_posterior(output_path, sample_name, contigs,
                               ref_prob, het_prob, hom_prob, K)

    print(f"[{sample_name}] Posterior genotypes written ({n_samples} loci, K={K})")


def _reconstruct_hard(post, contigs, ref_idx, het_idx, hom_idx,
                      K, n_samples, sample_name, output_path, config):
    """Output discrete genotype calls with confidence scores."""
    genotypes = np.empty((n_samples, K), dtype=object)
    genotype_probs = np.empty((n_samples, K), dtype=object)

    rat = config.ref_alt_threshold
    hht = config.homo_hetero_threshold

    for j in range(n_samples):
        pj = post[j]
        for k in range(K):
            sum_ref = pj[ref_idx[k]].sum()
            sum_het = pj[het_idx[k]].sum()
            sum_hom = pj[hom_idx[k]].sum()

            alt_total = sum_het + sum_hom + 1e-10
            ref_alt_ratio = sum_ref / alt_total

            if ref_alt_ratio >= rat:
                genotypes[j, k] = "ref"
                genotype_probs[j, k] = sum_ref
            elif ref_alt_ratio <= 1.0 / rat:
                if sum_hom / (sum_het + 1e-10) >= hht:
                    genotypes[j, k] = "alt_homo"
                    genotype_probs[j, k] = sum_hom
                elif sum_het / (sum_hom + 1e-10) >= hht:
                    genotypes[j, k] = "alt_hetero"
                    genotype_probs[j, k] = sum_het
                else:
                    genotypes[j, k] = "alt_ambi"
                    genotype_probs[j, k] = sum_het + sum_hom
            else:
                genotypes[j, k] = "ambiguous"
                genotype_probs[j, k] = "."

    amb = sum(1 for j in range(n_samples) if "ambiguous" in genotypes[j])
    print(f"[{sample_name}] Ambiguous: {amb}, Non-ambiguous: {n_samples - amb}")

    write_membership_hard(output_path, sample_name, contigs,
                          genotypes, genotype_probs, K)
