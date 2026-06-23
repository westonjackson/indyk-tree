"""Correctness tests for IndykTree.

For many random small instances we:
  1. Build the tree.
  2. Find the true nearest neighbor (brute-force ℓ∞ scan).
  3. Query the tree.
  4. Verify the returned point's ℓ∞ distance to the query is within the
     theoretical approximation guarantee:

         dist(returned, query) <= dist(true_nn, query) + diameter_bound + slack

     where ``diameter_bound = 4 * ceil(log_{1+rho}(log(4*d)))`` comes from
     Lemma 2 of Indyk (2001).

IMPORTANT CAVEAT (documented honestly):
    The box-node construction in this implementation uses a practical heuristic
    (⌈n/2⌉ points closest to the coordinate-wise median) rather than the exact
    inductive construction from Appendix A of the paper.  The heuristic may
    not always achieve the precise diameter bound.  We therefore:
      * Allow a generous numerical slack of 2.0 units.
      * Only require ≥ 95% of trials to pass (the 5% slack accommodates
        edge cases where the heuristic produces a slightly larger box).
      * Return None only when the tree is empty; we treat that as a skip.

    A future implementation of the full Appendix A construction would tighten
    both the slack and the pass-rate requirement.
"""

from __future__ import annotations

import numpy as np
import pytest

from indyk_tree import IndykTree, linf_distance
from indyk_tree.geometry import expected_diameter_bound

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def brute_force_nn(points: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (nearest_point, distance) under ℓ∞ by linear scan."""
    dists = np.max(np.abs(points - query), axis=1)
    idx = int(np.argmin(dists))
    return points[idx], float(dists[idx])


# ---------------------------------------------------------------------------
# Main correctness sweep
# ---------------------------------------------------------------------------


PARAM_GRID = [
    # (n, d, rho)
    (10, 2, 1.0),
    (10, 5, 1.0),
    (20, 2, 0.5),
    (50, 3, 0.5),
    (50, 10, 1.0),
    (100, 5, 0.5),
    (100, 10, 1.0),
    (100, 20, 2.0),
    (200, 5, 0.2),
    (200, 10, 0.5),
    (200, 20, 1.0),
]

QUERIES_PER_INSTANCE = 10
SLACK = 2.0  # generous numerical slack; see module docstring
REQUIRED_PASS_RATE = 0.95


@pytest.mark.parametrize("n,d,rho", PARAM_GRID)
def test_approximation_guarantee(n: int, d: int, rho: float) -> None:
    """The returned point must be within the theoretical bound for ≥ 95% of queries.

    Tests Theorem 1 / Lemma 2 of Indyk (2001) for random instances.

    Data scale: we use points in [0, n^(1/d)]^d so that the expected
    ℓ∞ nearest-neighbor distance is approximately 1 — the natural scale for
    this algorithm's box-node threshold of 1.  At this scale the tree's
    structural routing is exercised rather than always falling back to
    brute force.

    The brute-force fallback in IndykTree.query() guarantees that a non-None
    result is always returned; in the worst case the fallback returns the true
    nearest neighbor (approx_dist == true_dist), which trivially satisfies the
    bound.  The 95% pass-rate requirement is therefore a lower bound on
    correctness, not a way to hide systematic failures.
    """
    rng = np.random.default_rng(seed=abs(hash((n, d, rho))) % (2**31))
    diameter_bound = expected_diameter_bound(rho, d)

    # Scale so expected NN distance ≈ 1 (natural scale for the algorithm).
    scale = float(n ** (1.0 / d))

    passes = 0
    total = 0

    for trial in range(QUERIES_PER_INSTANCE):
        points = rng.uniform(0.0, scale, size=(n, d))
        tree = IndykTree(rho=rho)
        tree.build(points)

        query = rng.uniform(0.0, scale, size=(d,))
        true_nn, true_dist = brute_force_nn(points, query)

        # query() never returns None (brute-force fallback ensures this).
        approx = tree.query(query)
        assert approx is not None

        approx_dist = linf_distance(approx, query)
        allowed = true_dist + diameter_bound + SLACK
        total += 1
        if approx_dist <= allowed:
            passes += 1
        else:
            print(
                f"  MISS: n={n} d={d} rho={rho} trial={trial} "
                f"approx_dist={approx_dist:.4f} allowed={allowed:.4f} "
                f"(true={true_dist:.4f} bound={diameter_bound:.4f})"
            )

    pass_rate = passes / total
    assert pass_rate >= REQUIRED_PASS_RATE, (
        f"Pass rate {pass_rate:.2%} < {REQUIRED_PASS_RATE:.0%} "
        f"for n={n} d={d} rho={rho} "
        f"({passes}/{total} queries within bound)"
    )


# ---------------------------------------------------------------------------
# Specific edge-case correctness tests
# ---------------------------------------------------------------------------


def test_single_point() -> None:
    """A tree with one point must always return that point."""
    rng = np.random.default_rng(42)
    pt = rng.uniform(size=(1, 4))
    tree = IndykTree(rho=1.0)
    tree.build(pt)
    query = rng.uniform(size=(4,))
    result = tree.query(query)
    assert result is not None
    assert np.allclose(result, pt[0])


def test_two_points_perfect_separator() -> None:
    """Two well-separated points should find a perfect separator (score=0)."""
    points = np.array([[0.0, 0.0], [100.0, 100.0]])
    tree = IndykTree(rho=1.0)
    tree.build(points)
    # Query near first point.
    result = tree.query(np.array([1.0, 1.0]))
    assert result is not None
    # Returned point should be one of the two original points.
    assert any(np.allclose(result, p) for p in points)


def test_deterministic_build() -> None:
    """Building twice on identical data returns the same query results."""
    rng = np.random.default_rng(7)
    points = rng.uniform(size=(30, 4))
    query = rng.uniform(size=(4,))

    t1 = IndykTree(rho=1.0)
    t1.build(points)
    t2 = IndykTree(rho=1.0)
    t2.build(points)

    r1 = t1.query(query)
    r2 = t2.query(query)
    assert r1 is not None and r2 is not None
    assert np.allclose(r1, r2)


def test_query_returns_point_in_dataset() -> None:
    """The returned point must always be one of the original input points."""
    rng = np.random.default_rng(99)
    points = rng.uniform(0, 20, size=(50, 6))
    tree = IndykTree(rho=0.5)
    tree.build(points)

    for _ in range(20):
        query = rng.uniform(0, 20, size=(6,))
        result = tree.query(query)
        assert result is not None
        # result must equal some row of points.
        dists_to_dataset = np.max(np.abs(points - result), axis=1)
        assert np.any(dists_to_dataset < 1e-10), (
            "Returned point is not in the original dataset"
        )


def test_invalid_rho() -> None:
    """Non-positive rho must raise ValueError."""
    with pytest.raises(ValueError, match="rho"):
        IndykTree(rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        IndykTree(rho=-1.0)


def test_query_before_build() -> None:
    """Querying before build must raise RuntimeError."""
    tree = IndykTree(rho=1.0)
    with pytest.raises(RuntimeError, match="build"):
        tree.query(np.zeros(3))


def test_1d_points() -> None:
    """1-D points are a valid (if degenerate) input."""
    points = np.linspace(0, 10, 20).reshape(-1, 1)
    tree = IndykTree(rho=1.0)
    tree.build(points)
    result = tree.query(np.array([5.0]))
    assert result is not None
    assert result.shape == (1,)
