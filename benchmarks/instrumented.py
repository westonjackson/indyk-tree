"""InstrumentedIndykTree: IndykTree subclass with build and query statistics.

After the iterative refactor of IndykTree._build and _query, instrumentation
is done via explicit hook methods (_on_build_entry, _on_sep_node_built, etc.)
that IndykTree calls at each logical step.  InstrumentedIndykTree overrides
those hooks to accumulate statistics, replacing the earlier MRO-intercept
pattern which required recursive _build/_query calls and doubled stack depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from indyk_tree import IndykTree
from indyk_tree.geometry import linf_distances_to_point
from indyk_tree.nodes import BoxNode, SeparatorNode

FloatArray = npt.NDArray[np.float64]


@dataclass
class BuildStats:
    """Accumulated statistics from one call to IndykTree.build()."""

    total_nodes: int = 0
    separator_nodes: int = 0
    box_nodes: int = 0
    total_memberships: int = 0  # Σ n_i across all non-empty build work items
    max_depth: int = 0
    build_time: float = 0.0


@dataclass
class QueryStats:
    """Per-query statistics collected during one call to IndykTree.query()."""

    nodes_visited: int = 0
    sep_visits: int = 0
    box_visits: int = 0
    used_fallback: bool = False
    query_time: float = 0.0
    true_dist: float = 0.0
    approx_dist: float = 0.0
    approx_ratio: float = 1.0  # approx_dist / true_dist (1.0 when true_dist=0)
    returned_point: FloatArray = field(default_factory=lambda: np.array([]))
    true_nn: FloatArray = field(default_factory=lambda: np.array([]))


class InstrumentedIndykTree(IndykTree):
    """IndykTree with per-build and per-query instrumentation.

    Statistics are accumulated in ``build_stats`` during construction and
    reset/collected per call in ``last_query_stats`` during querying.

    The base-class iterative build calls ``_on_build_entry``,
    ``_on_sep_node_built``, and ``_on_box_node_built`` at each logical step;
    the iterative query calls ``_on_sep_node_visited`` and
    ``_on_box_node_visited``.  This subclass overrides all five hooks to
    populate ``build_stats`` and ``last_query_stats`` without duplicating any
    algorithm logic.
    """

    def __init__(self, rho: float) -> None:
        super().__init__(rho)
        self.build_stats = BuildStats()
        self.last_query_stats = QueryStats()

    # ------------------------------------------------------------------
    # Build instrumentation
    # ------------------------------------------------------------------

    def build(self, points: FloatArray) -> None:
        """Build the tree and reset all build statistics."""
        self.build_stats = BuildStats()
        super().build(points)
        self.build_stats.max_depth = self._compute_tree_depth()

    def _on_build_entry(self, n: int) -> None:
        """Count memberships and total nodes for each non-empty build work item."""
        self.build_stats.total_memberships += n
        self.build_stats.total_nodes += 1

    def _on_sep_node_built(self, node: SeparatorNode) -> None:
        """Tally separator nodes as they are created."""
        self.build_stats.separator_nodes += 1

    def _on_box_node_built(self, node: BoxNode) -> None:
        """Tally box nodes as they are created."""
        self.build_stats.box_nodes += 1

    def _compute_tree_depth(self) -> int:
        """Compute tree depth iteratively to avoid hitting the recursion limit."""
        stack: list[tuple[Any, int]] = [(self._root, 0)]
        max_d = 0
        while stack:
            node, d = stack.pop()
            if node is None:
                max_d = max(max_d, d)
            elif isinstance(node, SeparatorNode):
                stack.append((node.left, d + 1))
                stack.append((node.right, d + 1))
            elif isinstance(node, BoxNode):
                stack.append((node.continuation, d + 1))
        return max_d

    # ------------------------------------------------------------------
    # Query instrumentation
    # ------------------------------------------------------------------

    def query(self, y: FloatArray) -> FloatArray:  # type: ignore[override]
        """Query with full stats collection; always returns a point."""
        if self._root is None and self._points is None:
            raise RuntimeError("Call build() before query()")

        y = np.asarray(y, dtype=np.float64)

        qs = QueryStats()
        self.last_query_stats = qs

        result = self._query(y, self._root)

        if result is None:
            qs.used_fallback = True
            assert self._points is not None
            dists = linf_distances_to_point(self._points, y)
            result = self._points[int(np.argmin(dists))]

        return result

    def _on_sep_node_visited(self, node: SeparatorNode) -> None:
        """Count separator-node visits during query."""
        self.last_query_stats.nodes_visited += 1
        self.last_query_stats.sep_visits += 1

    def _on_box_node_visited(self, node: BoxNode) -> None:
        """Count box-node visits during query."""
        self.last_query_stats.nodes_visited += 1
        self.last_query_stats.box_visits += 1


# ---------------------------------------------------------------------------
# Pluggable interface for future ANN implementations
# ---------------------------------------------------------------------------


class ANNInterface:
    """Minimal interface any ANN implementation must satisfy.

    To plug in a different data structure, subclass this and implement
    ``build`` and ``query``.  The harness only calls these two methods.
    """

    def build(self, points: FloatArray) -> None:
        """Index the given point set."""
        raise NotImplementedError

    def query(self, y: FloatArray) -> FloatArray:
        """Return the approximate nearest neighbor of y."""
        raise NotImplementedError


class IndykTreeAdapter(ANNInterface):
    """Wraps InstrumentedIndykTree behind the pluggable interface."""

    def __init__(self, rho: float) -> None:
        self._tree = InstrumentedIndykTree(rho=rho)

    def build(self, points: FloatArray) -> None:
        """Delegate to the wrapped InstrumentedIndykTree."""
        self._tree.build(points)

    def query(self, y: FloatArray) -> FloatArray:
        """Delegate to the wrapped InstrumentedIndykTree."""
        return self._tree.query(y)

    @property
    def build_stats(self) -> BuildStats:
        """Build statistics from the last build() call."""
        return self._tree.build_stats

    @property
    def last_query_stats(self) -> QueryStats:
        """Query statistics from the last query() call."""
        return self._tree.last_query_stats
