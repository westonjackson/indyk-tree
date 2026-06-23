"""InstrumentedIndykTree: IndykTree subclass with build and query statistics.

Relies on Python's MRO: both _build and _query in IndykTree call ``self.*``,
so overriding them in this subclass instruments the entire recursion without
duplicating any logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    total_memberships: int = 0  # Σ n_i across all _build(points_i) calls
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
    """

    def __init__(self, rho: float) -> None:
        super().__init__(rho)
        self.build_stats = BuildStats()
        self.last_query_stats = QueryStats()
        self._current_depth: int = 0

    # ------------------------------------------------------------------
    # Build instrumentation
    # ------------------------------------------------------------------

    def build(self, points: FloatArray) -> None:
        """Build the tree and reset all build statistics."""
        self.build_stats = BuildStats()
        self._current_depth = 0
        super().build(points)
        self.build_stats.max_depth = self._compute_tree_depth()

    def _build(self, points: FloatArray) -> object:  # type: ignore[override]
        n = len(points)
        if n == 0:
            return None

        self.build_stats.total_memberships += n
        self.build_stats.total_nodes += 1

        # Delegate to parent; recursive calls within IndykTree._build dispatch
        # back to this override via Python's MRO (self._build is us).
        node = super()._build(points)

        if isinstance(node, SeparatorNode):
            self.build_stats.separator_nodes += 1
        elif isinstance(node, BoxNode):
            self.build_stats.box_nodes += 1

        return node

    def _compute_tree_depth(self) -> int:
        def walk(node: object, d: int) -> int:
            if node is None:
                return d
            if isinstance(node, SeparatorNode):
                return max(walk(node.left, d + 1), walk(node.right, d + 1))
            if isinstance(node, BoxNode):
                return walk(node.continuation, d + 1)
            return d

        return walk(self._root, 0)

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

        # Run tree traversal (our _query override counts nodes).
        result = self._query(y, self._root)

        if result is None:
            qs.used_fallback = True
            assert self._points is not None
            dists = linf_distances_to_point(self._points, y)
            result = self._points[int(np.argmin(dists))]

        return result

    def _query(self, y: FloatArray, node: object) -> FloatArray | None:
        if node is None:
            return None

        qs = self.last_query_stats
        if isinstance(node, SeparatorNode):
            qs.nodes_visited += 1
            qs.sep_visits += 1
        elif isinstance(node, BoxNode):
            qs.nodes_visited += 1
            qs.box_visits += 1

        # Dispatch to parent logic; recursive calls come back here via MRO.
        return super()._query(y, node)  # type: ignore[return-value]


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
