"""Centralized configuration for gPRISM parameters and paths."""

from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Package-wide default constants (single source of truth).
# Any function-level default for phi, epsilon, or popaf_range should be
# imported from here rather than spelled as a literal.
# ---------------------------------------------------------------------------
DEFAULT_PHI: float = 200.0
DEFAULT_EPSILON: float = 0.2
DEFAULT_POPAF_RANGE: Tuple[float, float] = (0.1, 0.9)


@dataclass
class GPRISMConfig:
    """All tunable parameters for the gPRISM pipeline."""

    # Beta-binomial model
    # phi controls the concentration of the beta-binomial distribution.
    # - A positive float: fixed dispersion (default DEFAULT_PHI).
    # - None: phi is set to the median read depth at run time, matching the
    #   description in the manuscript Methods section.
    phi: Optional[float] = DEFAULT_PHI
    epsilon: float = DEFAULT_EPSILON

    # Data filtering
    # min_depth is the "legacy" threshold used when phi >= 100. For phi < 100
    # the effective threshold is max(phi/2, 10) (see resolve_min_depth).
    min_depth: int = 50
    popaf_range: Tuple[float, float] = DEFAULT_POPAF_RANGE

    # Warn when phi is at/below this value — algorithm is unreliable there.
    phi_low_warning_threshold: float = 20.0

    # EM algorithm
    max_k: int = 5
    n_init: int = 10
    max_em_iter: int = 1000
    lr: float = 1e-2
    convergence_tol: float = 1e-6
    max_decrease_count: int = 3

    # Resolution floor: resolution = min(resolution_constant / mean_depth,
    #                                    resolution_ceiling)
    # 4.04 is the inverse-model fit from Supplementary Fig. 2.
    # Ceiling at 0.05 keeps K * resolution << 1 at very low depth
    # (e.g. mean_depth=10 -> 4.04/10 = 0.404 would break a_vec for K >= 3).
    resolution_constant: float = 4.04
    resolution_ceiling: float = 0.05

    # Genotype classification thresholds
    ref_alt_threshold: float = 4.0
    homo_hetero_threshold: float = 9.0

    # Device
    device: Optional[str] = None  # None = auto-detect

    def resolve_min_depth(self, phi: float) -> int:
        """Return the effective min_depth filter threshold for a given phi.

        - phi >= 100  → self.min_depth (default 50, legacy behaviour)
        - phi <  100  → max(phi / 2, 10), rounded down to int

        Rationale: when phi is set to the median read depth of the data
        (manuscript definition), requiring depth >= 50 discards most sites
        for low-coverage datasets. Scaling the floor with phi keeps a
        reasonable fraction of sites while still excluding extremely shallow
        ones that contribute mostly noise.
        """
        if phi >= 100:
            return int(self.min_depth)
        return int(max(phi / 2.0, 10))

    def resolution(self, mean_depth: float) -> float:
        """Compute the resolution floor for a given mean depth.

        Capped at ``resolution_ceiling`` so that K * resolution stays well
        below 1 even at very low depth.
        """
        if mean_depth <= 0:
            return min(0.02, self.resolution_ceiling)
        return min(self.resolution_constant / mean_depth, self.resolution_ceiling)

    def effective_phi(self, depths) -> float:
        """Return the dispersion to use for a dataset with the given depths.

        When ``self.phi`` is *None*, phi is set to the median read depth of
        *depths*, matching the manuscript definition. Otherwise returns the
        fixed configured value.
        """
        if self.phi is not None:
            return float(self.phi)
        import numpy as np
        depths = np.asarray(depths)
        if depths.size == 0:
            return DEFAULT_PHI
        return float(np.median(depths))


DEFAULT_CONFIG = GPRISMConfig()
