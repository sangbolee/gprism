"""Output file writers for gPRISM results."""

from typing import Dict
import numpy as np


def write_mixture_results(output_path: str, sample_name: str,
                          results: Dict[int, dict]):
    """Write the BIC model-selection results for K=1..max_k.

    Parameters
    ----------
    output_path : str
        Full file path for the output TSV.
    sample_name : str
        Sample identifier.
    results : dict[int, dict]
        ``{K: {'a_vec': ..., 'bic': ..., 'loglike': ..., ...}}``.
    """
    with open(output_path, mode="w") as fh:
        fh.write("Sample\tNoC\tMixtureProp\tCenter\tWeight\tBIC\tLogL\n")
        for k in sorted(results):
            if k == "time":
                continue
            r = results[k]
            a_str = ",".join(map(str, r["a_vec"]))
            if k == 1:
                fh.write(f"{sample_name}\t{k}\t{a_str}\t{r['bic']}\n")
            else:
                c_str = ",".join(map(str, r["centers"]))
                w_str = ",".join(map(str, r["weights"]))
                fh.write(
                    f"{sample_name}\t{k}\t{a_str}\t{c_str}\t{w_str}\t"
                    f"{r['bic']}\t{r['loglike']}\n"
                )


def load_mixture_results(result_path: str):
    """Load Mixture_Results.txt and return parsed dict + optimal K.

    Returns
    -------
    results : dict[int, dict]
    optimal_k : int
    """
    results = {}
    with open(result_path, mode="r") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            fields = line.strip().split("\t")
            noc = fields[1]
            if noc == "time":
                continue
            k = int(noc)
            if k == 1:
                results[k] = {"a_vec": [1.0], "bic": float(fields[3])}
            else:
                props = list(map(float, fields[2].split(",")))
                props.sort(reverse=True)
                results[k] = {
                    "a_vec": props,
                    "centers": list(map(float, fields[3].split(","))),
                    "weights": list(map(float, fields[4].split(","))),
                    "bic": float(fields[5]),
                    "loglike": float(fields[6]),
                }

    bics = {k: v["bic"] for k, v in results.items()}
    optimal_k = min(bics, key=bics.get)
    return results, optimal_k


def write_membership_hard(output_path: str, sample_name: str,
                          contigs: list, genotypes: np.ndarray,
                          genotype_probs: np.ndarray, K: int):
    """Write hard genotype calls per contributor per locus."""
    with open(output_path, mode="w") as fh:
        header_parts = ["Sample", "Chr", "Position"]
        for i in range(K):
            header_parts += [f"Contributor#{i+1}_GT", f"Contributor#{i+1}_Prob"]
        fh.write("\t".join(header_parts) + "\n")

        for j, locus in enumerate(contigs):
            parts = [sample_name, locus[0], locus[1]]
            for i in range(K):
                gt = genotypes[j, i]
                prob = genotype_probs[j, i]
                prob_str = f"{prob:.5f}" if isinstance(prob, float) else str(prob)
                parts += [gt, prob_str]
            fh.write("\t".join(parts) + "\n")


def write_membership_posterior(output_path: str, sample_name: str,
                               contigs: list, ref_prob: np.ndarray,
                               het_prob: np.ndarray, hom_prob: np.ndarray,
                               K: int):
    """Write posterior genotype probabilities per contributor per locus."""
    with open(output_path, mode="w") as fh:
        header_parts = ["Sample", "Chr", "Position"]
        for i in range(K):
            header_parts += [
                f"Contributor#{i+1}_ref_prob",
                f"Contributor#{i+1}_hetero_prob",
                f"Contributor#{i+1}_homo_prob",
            ]
        fh.write("\t".join(header_parts) + "\n")

        for j, locus in enumerate(contigs):
            parts = [sample_name, locus[0], locus[1]]
            for i in range(K):
                parts += [
                    f"{ref_prob[j, i]:.5f}",
                    f"{het_prob[j, i]:.5f}",
                    f"{hom_prob[j, i]:.5f}",
                ]
            fh.write("\t".join(parts) + "\n")
