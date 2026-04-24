"""Unit tests for gprism.genotype.posterior."""

import numpy as np
import pytest

from gprism.genotype.posterior import compute_posterior


def test_posterior_rows_sum_to_one():
    rng = np.random.default_rng(42)
    n = 50
    pafs = rng.uniform(0.2, 0.8, size=n).astype(np.float32)
    depths = np.full(n, 200, dtype=np.int64)
    counts = rng.integers(0, 200, size=n).astype(np.int64)

    post = compute_posterior([0.6, 0.4], counts, depths, pafs,
                             epsilon=0.2, phi=200.0)
    assert post.shape == (n, 9)  # M = 3^2
    for row in post:
        assert row.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(row >= 0)


def test_posterior_auto_phi():
    """phi=None should be equivalent to phi=median(depths)."""
    rng = np.random.default_rng(7)
    n = 40
    pafs = rng.uniform(0.2, 0.8, size=n).astype(np.float32)
    depths = rng.integers(150, 250, size=n).astype(np.int64)
    counts = rng.integers(0, 200, size=n).astype(np.int64)

    post_auto = compute_posterior([0.5, 0.5], counts, depths, pafs, phi=None)
    post_fixed = compute_posterior([0.5, 0.5], counts, depths, pafs,
                                   phi=float(np.median(depths)))
    assert np.allclose(post_auto, post_fixed)
