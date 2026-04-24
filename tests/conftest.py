"""Pytest fixtures for gPRISM tests.

Provides a deterministic synthetic two-person DNA mixture in the
``.biallelic.mpileup`` format that the production code consumes, generated
once at session scope so integration tests can share it without rebuilding.
"""

from pathlib import Path

import numpy as np
import pytest


FIXTURE_DIR = Path(__file__).parent / "fixtures"
TINY_PATH = FIXTURE_DIR / "tiny.biallelic.mpileup"


def _synthesize_tiny_mpileup(path: Path, n_sites: int = 400,
                             a_vec=(0.7, 0.3), depth: int = 200,
                             seed: int = 20260417):
    """Write a deterministic synthetic two-person mixture file.

    Each locus has randomly drawn population AF in (0.1, 0.9), diploid
    genotypes sampled under Hardy-Weinberg for each of the K contributors,
    and beta-binomial read counts at the specified depth.
    """
    rng = np.random.default_rng(seed)
    a_vec = np.asarray(a_vec, dtype=np.float64)
    K = len(a_vec)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("#CHROM\tPOS\tREF\tALT\tDepth\tAC\tVAF\tPAF\tBases\tBQ\n")
        chrom = "chr1"
        for i in range(n_sites):
            paf = float(rng.uniform(0.15, 0.85))
            # Sample each contributor's dosage under HWE
            dosages = []
            for _ in range(K):
                r = rng.random()
                if r < (1 - paf) ** 2:
                    d = 0.0
                elif r < (1 - paf) ** 2 + 2 * paf * (1 - paf):
                    d = 0.5
                else:
                    d = 1.0
                dosages.append(d)
            mu = float(np.dot(a_vec, dosages))
            # Beta-binomial draw with phi=depth
            phi = float(depth)
            alpha = phi * mu + 0.2
            beta = phi * (1 - mu) + 0.2
            p = rng.beta(alpha, beta)
            x = int(rng.binomial(depth, p))
            vaf = x / depth
            pos = 1_000_000 + i * 1000
            fh.write(
                f"{chrom}\t{pos}\tA\tG\t{depth}\t{x}\t{vaf:.6f}\t{paf:.6f}\t.\t.\n"
            )


@pytest.fixture(scope="session")
def tiny_mpileup_path():
    """Absolute path to a synthetic 400-site two-person mixture fixture.

    Regenerated only if missing so the file stays stable across runs.
    """
    if not TINY_PATH.exists():
        _synthesize_tiny_mpileup(TINY_PATH)
    return str(TINY_PATH)
