"""Unit tests for gprism.model.likelihood."""

import numpy as np
import pytest

from gprism.model.likelihood import log_likelihood


def _synthetic_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    pafs = rng.uniform(0.2, 0.8, size=n).astype(np.float32)
    depths = np.full(n, 200, dtype=np.int64)
    # Single-contributor HWE: vaf ~ (0, 0.5, 1.0) by HWE dosage
    vafs = []
    for p in pafs:
        r = rng.random()
        if r < (1 - p) ** 2:
            vafs.append(0.0)
        elif r < (1 - p) ** 2 + 2 * p * (1 - p):
            vafs.append(0.5)
        else:
            vafs.append(1.0)
    xs = np.array([int(rng.binomial(200, v * 0.95 + 0.025)) for v in vafs],
                  dtype=np.int64)
    return xs, depths, pafs


class TestLogLikelihood:
    def test_single_contributor_finite(self):
        X, N, paf = _synthetic_data()
        ll = log_likelihood([1.0], X, N, paf, phi=200.0, epsilon=0.2)
        assert np.isfinite(ll)

    def test_phi_none_uses_median_depth(self):
        X, N, paf = _synthetic_data()
        ll_auto = log_likelihood([1.0], X, N, paf, phi=None, epsilon=0.2)
        ll_fixed = log_likelihood([1.0], X, N, paf, phi=200.0, epsilon=0.2)
        # median(N) == 200 so the two should be identical
        assert ll_auto == pytest.approx(ll_fixed)

    def test_two_contributor_prefers_true_prop(self):
        """A 0.5/0.5 mixture should have higher LL than an extreme 0.99/0.01."""
        rng = np.random.default_rng(1)
        n = 300
        paf = rng.uniform(0.2, 0.8, size=n).astype(np.float32)
        depth = np.full(n, 200, dtype=np.int64)
        # Simulate 50/50 mixture
        xs = []
        for p in paf:
            d1 = rng.choice([0.0, 0.5, 1.0],
                            p=[(1 - p) ** 2, 2 * p * (1 - p), p ** 2])
            d2 = rng.choice([0.0, 0.5, 1.0],
                            p=[(1 - p) ** 2, 2 * p * (1 - p), p ** 2])
            mu = 0.5 * d1 + 0.5 * d2
            xs.append(int(rng.binomial(200, mu * 0.95 + 0.025)))
        xs = np.array(xs, dtype=np.int64)

        ll_balanced = log_likelihood([0.5, 0.5], xs, depth, paf)
        ll_skewed = log_likelihood([0.99, 0.01], xs, depth, paf)
        assert ll_balanced > ll_skewed
