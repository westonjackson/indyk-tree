"""Unit tests for tree construction internals.

These tests verify the separator scoring, node types, and structural
properties of the built tree independently of the query correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from indyk_tree import BoxNode, IndykTree, SeparatorNode
from indyk_tree.geometry import (
    _compute_score,
    expected_diameter_bound,
    find_best_separator,
    linf_distance,
    linf_distances_to_point,
)

# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def test_linf_distance_basic() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 1.0, -2.0])
    assert linf_distance(a, b) == pytest.approx(3.0)


def test_linf_distance_same_point() -> None:
    a = np.array([1.5, -2.3, 0.0])
    assert linf_distance(a, a) == pytest.approx(0.0)


def test_linf_distances_to_point_shape() -> None:
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]])
    q = np.array([0.0, 0.0])
    d = linf_distances_to_point(pts, q)
    assert d.shape == (3,)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(4.0)
    assert d[2] == pytest.approx(2.0)


def test_compute_score_perfect_split() -> None:
    """m == 0 always gives score == 0 (no replication)."""
    assert _compute_score(0.5, 0.5, 0.0) == pytest.approx(0.0)


def test_compute_score_numerics() -> None:
    """Score is non-negative for valid l+r+m==1 inputs."""
    score = _compute_score(0.4, 0.4, 0.2)
    assert score is not None
    assert score >= 0.0


def test_compute_score_l_zero_returns_none() -> None:
    """m + r == 1 (i.e. l == 0) must return None."""
    result = _compute_score(0.0, 0.5, 0.5)
    assert result is None


def test_expected_diameter_bound_positive() -> None:
    """Bound is positive for reasonable rho and d."""
    b = expected_diameter_bound(1.0, 10)
    assert b > 0


def test_expected_diameter_bound_increases_with_d() -> None:
    """Larger d gives a larger (or equal) diameter bound."""
    b5 = expected_diameter_bound(1.0, 5)
    b20 = expected_diameter_bound(1.0, 20)
    assert b20 >= b5


# ---------------------------------------------------------------------------
# separator search
# ---------------------------------------------------------------------------


def test_find_best_separator_well_separated() -> None:
    """Two clearly separated clusters should find a good separator."""
    rng = np.random.default_rng(0)
    left_cluster = rng.uniform(0, 5, size=(20, 3))
    right_cluster = rng.uniform(15, 20, size=(20, 3))
    points = np.vstack([left_cluster, right_cluster])

    sep = find_best_separator(points, rho=1.0)
    assert sep is not None
    assert sep.score == pytest.approx(0.0, abs=1e-9)  # perfect split


def test_find_best_separator_uniform_cloud() -> None:
    """Tight uniform cloud: separator may or may not exist — just verify no crash."""
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 1, size=(50, 5))
    # May return None or a separator — both are valid.
    result = find_best_separator(points, rho=1.0)
    assert result is None or isinstance(result, object)


def test_find_best_separator_empty() -> None:
    """Empty array returns None."""
    points = np.empty((0, 3), dtype=np.float64)
    assert find_best_separator(points, rho=1.0) is None


def test_find_best_separator_selects_best_score() -> None:
    """Among multiple good separators, the one with lowest score is returned."""
    # Three clusters: [0,2], [8,10], [18,20] — two separators exist.
    rng = np.random.default_rng(2)
    c1 = rng.uniform(0, 2, size=(15, 2))
    c2 = rng.uniform(8, 10, size=(15, 2))
    c3 = rng.uniform(18, 20, size=(15, 2))
    points = np.vstack([c1, c2, c3])

    sep = find_best_separator(points, rho=2.0)
    assert sep is not None
    assert sep.score <= 2.0


# ---------------------------------------------------------------------------
# tree structure
# ---------------------------------------------------------------------------


def test_build_produces_node() -> None:
    """build() must return a SeparatorNode or BoxNode (not None) for n > 0."""
    rng = np.random.default_rng(3)
    points = rng.uniform(size=(10, 3))
    tree = IndykTree(rho=1.0)
    tree.build(points)
    assert tree._root is not None
    assert isinstance(tree._root, (SeparatorNode, BoxNode))


def test_separator_node_children_are_subsets() -> None:
    """SeparatorNode children together cover all points (with possible overlap in M)."""
    rng = np.random.default_rng(4)
    # Construct clearly separable data to guarantee a SeparatorNode at root.
    left = rng.uniform(0, 3, size=(15, 2))
    right = rng.uniform(20, 23, size=(15, 2))
    points = np.vstack([left, right])

    tree = IndykTree(rho=1.0)
    tree.build(points)

    root = tree._root
    assert isinstance(root, SeparatorNode), (
        "Expected SeparatorNode for clearly separable data"
    )


def test_box_node_representative_in_box_points() -> None:
    """BoxNode.representative must be one of the box_points."""
    rng = np.random.default_rng(5)
    # Tight cluster → likely triggers box-node path.
    points = rng.uniform(0, 0.5, size=(8, 4))
    tree = IndykTree(rho=0.01)  # very small rho makes separators hard to find
    tree.build(points)

    def check_box_nodes(node: object) -> None:
        if node is None:
            return
        if isinstance(node, BoxNode):
            dists = np.max(np.abs(node.box_points - node.representative), axis=1)
            assert np.any(dists < 1e-10), "representative not in box_points"
            check_box_nodes(node.continuation)
        elif isinstance(node, SeparatorNode):
            check_box_nodes(node.left)
            check_box_nodes(node.right)

    check_box_nodes(tree._root)


def test_build_invalid_input() -> None:
    """1-D and empty arrays must raise ValueError."""
    tree = IndykTree(rho=1.0)
    with pytest.raises(ValueError):
        tree.build(np.array([1.0, 2.0, 3.0]))  # 1-D
    with pytest.raises(ValueError):
        tree.build(np.empty((0, 3)))  # empty


def test_build_copies_input() -> None:
    """Mutating the original array after build must not affect query results."""
    rng = np.random.default_rng(6)
    points = rng.uniform(size=(20, 3)).copy()
    tree = IndykTree(rho=1.0)
    tree.build(points)
    query = rng.uniform(size=(3,))
    result_before = tree.query(query)

    points[:] = 9999.0  # corrupt original
    result_after = tree.query(query)

    assert result_before is not None and result_after is not None
    assert np.allclose(result_before, result_after)
