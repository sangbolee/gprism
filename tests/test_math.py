"""Unit tests for gprism.utils.math."""

import math
import numpy as np
import pytest

from gprism.utils.math import (
    generate_center_dict,
    generate_popaf_dict,
    log_nCk,
    beta_binomial_logpmf,
    build_combination_matrix,
    build_popaf_matrix,
    build_genotype_indices,
)


class TestCenterDict:
    def test_single_contributor(self):
        d = generate_center_dict([0.5])
        assert d[("ref",)] == pytest.approx(0.0)
        assert d[("hetero",)] == pytest.approx(0.25)
        assert d[("homo",)] == pytest.approx(0.5)

    def test_two_contributors(self):
        d = generate_center_dict([0.6, 0.4])
        assert len(d) == 9  # 3^2
        assert d[("ref", "ref")] == pytest.approx(0.0)
        assert d[("homo", "homo")] == pytest.approx(1.0)
        assert d[("hetero", "ref")] == pytest.approx(0.3)

    def test_symmetry(self):
        d = generate_center_dict([0.5, 0.5])
        assert d[("homo", "ref")] == pytest.approx(d[("ref", "homo")])


class TestPopAFDict:
    def test_probabilities_sum_to_one(self):
        d = generate_popaf_dict(0.3, 2)
        assert sum(d.values()) == pytest.approx(1.0)

    def test_single_contributor_hwe(self):
        p = 0.2
        d = generate_popaf_dict(p, 1)
        assert d[("ref",)] == pytest.approx((1 - p) ** 2)
        assert d[("hetero",)] == pytest.approx(2 * p * (1 - p))
        assert d[("homo",)] == pytest.approx(p ** 2)


class TestLogNCk:
    def test_basic(self):
        assert log_nCk(10, 3) == pytest.approx(math.log(120), rel=1e-6)

    def test_edge(self):
        assert log_nCk(5, 0) == pytest.approx(0.0, abs=1e-10)
        assert log_nCk(5, 5) == pytest.approx(0.0, abs=1e-10)


class TestBetaBinomial:
    def test_valid_logpmf(self):
        val = beta_binomial_logpmf(3, 10, 2.0, 5.0)
        assert np.isfinite(val)
        assert val <= 0  # log probability

    def test_sum_approximately_one(self):
        """PMF over all x=0..n should sum to ~1."""
        n = 20
        alpha, beta = 3.0, 5.0
        total = sum(
            np.exp(beta_binomial_logpmf(x, n, alpha, beta))
            for x in range(n + 1)
        )
        assert total == pytest.approx(1.0, abs=1e-6)


class TestCombinationMatrix:
    def test_shape(self):
        combos, mat = build_combination_matrix(2)
        assert mat.shape == (9, 2)
        assert len(combos) == 9

    def test_values(self):
        _, mat = build_combination_matrix(1)
        assert set(mat[:, 0].tolist()) == {0.0, 0.5, 1.0}


class TestPopAFMatrix:
    def test_rows_sum_to_one(self):
        pafs = np.array([0.1, 0.3, 0.5], dtype=np.float32)
        m = build_popaf_matrix(pafs, K=2)
        assert m.shape == (3, 9)
        for row in m:
            assert row.sum() == pytest.approx(1.0, abs=1e-4)


class TestGenotypeIndices:
    def test_coverage(self):
        combos, _ = build_combination_matrix(2)
        ref_idx, het_idx, hom_idx = build_genotype_indices(combos, 2)
        for k in range(2):
            all_idx = set(ref_idx[k] + het_idx[k] + hom_idx[k])
            assert all_idx == set(range(9))
