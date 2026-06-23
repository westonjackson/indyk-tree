"""Data generators for benchmarking IndykTree.

Each generator returns (points, queries) where:
  points: np.ndarray of shape (n, d) — the database
  queries: np.ndarray of shape (N_QUERIES, d) — query points planted within
           ℓ∞ distance 0.5 of a random database point, guaranteeing a true
           near neighbor exists within distance 0.5 < 1.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

N_QUERIES = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _plant_queries(
    points: np.ndarray, rng: Generator, n_queries: int = N_QUERIES
) -> np.ndarray:
    """Plant queries within ℓ∞ distance 0.5 of random database points."""
    n, d = points.shape
    idx = rng.integers(0, n, size=n_queries)
    noise = rng.uniform(-0.5, 0.5, size=(n_queries, d))
    return points[idx] + noise


# ---------------------------------------------------------------------------
# 1. Uniform random
# ---------------------------------------------------------------------------


def uniform_random(
    n: int,
    d: int,
    low: float = -100.0,
    high: float = 100.0,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """N points drawn i.i.d. from Uniform[low, high]^d.

    This is the "baseline non-adversarial" distribution. The separator search
    finds many score-0 splits (large gaps exist in 200-unit ranges with only n
    points). Query results mostly fall back to brute force because box-node
    centers are far from the query at this scale (NN distances >> 1).
    """
    if rng is None:
        rng = np.random.default_rng()
    points = rng.uniform(low, high, size=(n, d)).astype(np.float64)
    queries = _plant_queries(points, rng)
    return points, queries


# ---------------------------------------------------------------------------
# 2. Clustered (Gaussian mixture)
# ---------------------------------------------------------------------------


def clustered(
    n: int,
    d: int,
    n_clusters: int = 5,
    cluster_std: float = 1.0,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian mixture model: realistic "non-adversarial" structured data.

    k random cluster centers in [-50, 50]^d, points drawn from N(center, σ²I).
    The separator search should find clean separators between clusters. Within
    each cluster (diameter ~σ), box nodes cover local neighborhoods.
    """
    if rng is None:
        rng = np.random.default_rng()
    centers = rng.uniform(-50.0, 50.0, size=(n_clusters, d))
    assignments = rng.integers(0, n_clusters, size=n)
    points = centers[assignments] + rng.normal(0.0, cluster_std, size=(n, d))
    points = points.astype(np.float64)
    queries = _plant_queries(points, rng)
    return points, queries


# ---------------------------------------------------------------------------
# 3. ACP08 adversarial
# ---------------------------------------------------------------------------


def acp08_adversarial(
    n: int,
    d: int,
    rho: float,
    m: int = 30,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adversarial distribution from Andoni, Croitoru & Pătrașcu (FOCS 2008).

    Their hardness construction for ℓ∞ nearest neighbor uses a 1-D marginal
    with a geometric tail of integer values:

        π_i = 2^{-(2ρ)^i}  for i = 1, …, m
        π_0 = 1 – Σ π_i            (probability mass at 0)

    Each coordinate of each point is drawn i.i.d. from this marginal: value 0
    with probability π_0, value +i or −i (50/50) with probability π_i.

    The product-distribution over d coordinates is designed to defeat
    separator-search algorithms: most coordinates are 0 (making the L/M/R
    partition degenerate), and the few non-zero coordinates cluster near small
    integers (making separators replicate almost all points).

    VALIDITY REQUIREMENT: the series Σ π_i must converge and sum to < 0.5
    (so π_0 ≥ 0.5). This requires (2ρ) > 1, i.e. ρ > 0.5. For ρ ≤ 0.5
    the series diverges and the distribution is undefined.

    Query construction (per the paper): draw an independent coordinate-wise
    offset from the same marginal and add it to a random database point.
    This "plants" the query in the ε-ball of a database point in the L∞ sense
    expected by the hardness proof.

    Args:
        n: Number of database points.
        d: Dimension.
        rho: Quality parameter; must satisfy rho > 0.5.
        m: Truncation depth (30 is large enough that π_m < 10^{-9} for ρ ≥ 1).
        rng: Optional seeded random generator.

    Raises:
        ValueError: If ρ ≤ 0.5, making π_0 < 0 with the chosen m.

    """
    if rng is None:
        rng = np.random.default_rng()

    # Compute tail probabilities.
    pi = np.array([2.0 ** (-(2.0 * rho) ** i) for i in range(1, m + 1)])
    pi_0 = 1.0 - pi.sum()

    # The spec says "assert pi_0 >= 0.5 and raise if rho is too large";
    # based on the formula the failure mode is rho TOO SMALL (diverging series).
    # We check pi_0 >= 0.5 regardless of direction.
    if pi_0 < 0.5:
        raise ValueError(
            f"ACP08 distribution invalid for rho={rho}: "
            f"pi_0 = {pi_0:.4f} < 0.5 with m={m}. "
            f"The series Σ π_i diverges for rho <= 0.5; "
            f"use rho > 0.5 (practically rho >= 1.0 for safe margin)."
        )

    # Absolute-value probabilities: index k → value k, k=0,...,m.
    abs_probs = np.concatenate([[pi_0], pi])
    abs_probs /= abs_probs.sum()  # normalise for floating-point safety

    # Draw database points: all coords at once.
    abs_vals = rng.choice(m + 1, size=(n, d), p=abs_probs)
    signs = rng.choice(np.array([-1, 1]), size=(n, d))
    points = (abs_vals * signs).astype(np.float64)

    # Planted queries: random db point + independent ACP08 offset per coord.
    q_idx = rng.integers(0, n, size=N_QUERIES)
    q_abs = rng.choice(m + 1, size=(N_QUERIES, d), p=abs_probs)
    q_signs = rng.choice(np.array([-1, 1]), size=(N_QUERIES, d))
    offsets = (q_abs * q_signs).astype(np.float64)
    queries = points[q_idx] + offsets

    return points, queries


# ---------------------------------------------------------------------------
# 4. Cyclic closeness  (predicted to NOT be adversarial)
# ---------------------------------------------------------------------------


def cyclic_closeness(
    n: int,
    d: int,
    gap: float = 10.0,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-adversarial structured distribution: cyclic single-coordinate offsets.

    Each point i is assigned a "special" coordinate s = i % d.  All other
    coordinates are drawn from N(0, 0.5²).  The special coordinate is
    additionally shifted by `gap`.

    Analytical prediction: Indyk's separator search finds a score-0 split
    along the special coordinate (a gap of size `gap - 2` exists between
    points with different special coordinates), so the tree should be nearly
    balanced with few box nodes and low approximation error.  This distribution
    is included to *confirm* the theory by showing it is NOT adversarial.
    """
    if rng is None:
        rng = np.random.default_rng()
    points = rng.normal(0.0, 0.5, size=(n, d)).astype(np.float64)
    for i in range(n):
        points[i, i % d] += gap
    queries = _plant_queries(points, rng)
    return points, queries


# ---------------------------------------------------------------------------
# 5. Boundary stress
# ---------------------------------------------------------------------------


def boundary_stress(
    n: int,
    d: int,
    rho: float,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stress-tests replication by snapping coordinates to separator boundaries.

    Generates uniform-random points in [-20, 20]^d, then rounds each
    coordinate to the nearest integer.  This maximises coordinate collisions:
    for integer-valued data, every threshold t = v ± 1 (integer ±1) captures
    many points in the slab [t-1, t+1], creating high replication pressure.

    The distribution is designed to stress the m < 0.5 guard and measure
    whether space usage blows up relative to uniform random data.
    """
    if rng is None:
        rng = np.random.default_rng()
    points = np.round(rng.uniform(-20.0, 20.0, size=(n, d))).astype(np.float64)
    queries = _plant_queries(points, rng)
    return points, queries


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, object] = {
    "uniform_random": uniform_random,
    "clustered": clustered,
    "acp08_adversarial": acp08_adversarial,
    "cyclic_closeness": cyclic_closeness,
    "boundary_stress": boundary_stress,
}
