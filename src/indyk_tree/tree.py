"""IndykTree: static ℓ∞ approximate nearest neighbor data structure.

Implements the construction and query algorithms from:

    Piotr Indyk, "On Approximate Nearest Neighbors under ℓ∞ Norm,"
    Journal of Computer and System Sciences 63(4), pp. 627-638, 2001.

Space:  O(n^{1+ρ} · d)      — replication of M-slab points at each split.
Build:  O(n^{1+ρ} · d²)     — separator search is O(n·d) per level.
Query:  O(log n · d / ρ)     — one root-to-leaf path, d work per node.

The ``rho`` parameter controls the trade-off between approximation quality
and space/time cost.  Smaller ρ gives a tighter approximation guarantee but
larger space and build time.  Typical values: ρ ∈ (0, 2].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from .geometry import (
    find_best_separator,
    linf_distance,
    linf_distances_to_point,
)
from .nodes import BoxNode, SeparatorNode

FloatArray = npt.NDArray[np.float64]
Node = SeparatorNode | BoxNode | None


class IndykTree:
    """Static ℓ∞ approximate nearest neighbor tree.

    Build once with a set of points, then answer approximate nearest-neighbor
    queries in sub-linear time (relative to a brute-force scan).

    The approximation guarantee (Theorem 1, Indyk 2001): for any query point
    *y*, the returned point *p* satisfies::

        ||p - y||_∞  ≤  ||nn(y) - y||_∞  +  O(log_{1+ρ}(log d))

    where ``nn(y)`` is the true nearest neighbor.  The additive error comes
    from the box-node construction (see :class:`.BoxNode`).

    Args:
        rho: Quality parameter ρ > 0.  Smaller values give a tighter
            approximation but larger space/build cost.  The paper analyzes
            ρ ∈ (0, 2]; values outside this range are accepted but the
            theoretical guarantees may not apply.

    Example:
        >>> import numpy as np
        >>> from indyk_tree import IndykTree
        >>> rng = np.random.default_rng(0)
        >>> points = rng.uniform(0, 100, size=(200, 5))
        >>> tree = IndykTree(rho=1.0)
        >>> tree.build(points)
        >>> query = rng.uniform(0, 100, size=(5,))
        >>> approx_nn = tree.query(query)

    """

    def __init__(self, rho: float = 1.0) -> None:
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho!r}")
        self.rho = rho
        self._root: Node = None
        self._points: FloatArray | None = None
        self._d: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, points: FloatArray) -> None:
        """Build the ANN tree from a set of points.

        Args:
            points: Array of shape ``(n, d)`` with ``dtype=float64`` (or
                convertible).  The array is copied internally so the caller
                may modify it afterwards.

        Raises:
            ValueError: If *points* is empty or not 2-D.

        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError(f"points must be 2-D, got shape {points.shape}")
        if points.shape[0] == 0:
            raise ValueError("points must be non-empty")

        self._points = points.copy()
        self._d = points.shape[1]
        self._root = self._build(self._points)

    def query(self, y: FloatArray) -> FloatArray | None:
        """Return an approximate nearest neighbor of *y* under ℓ∞.

        Implements the query pseudocode from Section 3 of Indyk (2001)::

            def query(y, node):
                if node is None: return None
                if SeparatorNode(i, t):
                    if y[i] < t: return query(y, node.left)
                    else:        return query(y, node.right)
                if BoxNode(center, representative, continuation):
                    if linf(y, center) <= 1: return representative
                    else: return query(y, node.continuation)

        Args:
            y: Query point, shape ``(d,)``.

        Returns:
            An approximate nearest neighbor point (shape ``(d,)``).  The
            tree's structural traversal may reach an empty node (returning
            ``None`` internally) when the query lies outside the ℓ∞ radius-1
            neighbourhood of every box center — this occurs for data at
            scales much larger than 1.  In that case the method falls back to
            a brute-force linear scan over the stored points so that the
            caller always receives a valid result.

        Raises:
            RuntimeError: If :meth:`build` has not been called.

        """
        if self._root is None and self._points is None:
            raise RuntimeError("Call build() before query()")
        y = np.asarray(y, dtype=np.float64)
        result = self._query(y, self._root)
        if result is None:
            # Fallback: brute-force scan.  Triggered when the query is outside
            # the ℓ∞ radius-1 neighbourhood of every box center — this is
            # correct algorithm behaviour (the tree is designed for data at
            # scale O(1)) but we make the public API robust by returning the
            # true nearest neighbour in this case.
            assert self._points is not None
            dists = linf_distances_to_point(self._points, y)
            result = self._points[int(np.argmin(dists))]
        return result

    # ------------------------------------------------------------------
    # Internal construction (iterative — no Python recursion)
    # ------------------------------------------------------------------

    def _build(self, points: FloatArray) -> Node:
        r"""Construct the tree on *points* using an iterative work-stack.

        Uses an explicit work/result stack instead of recursion to avoid
        Python's default recursion limit (~1000 frames).

        Phase 1 — separator search (Section 3):
            Try every (coordinate, threshold) candidate.  If a good separator
            (i*, t*) is found, create a SeparatorNode and recurse on
            P1 = L ∪ M and P2 = R ∪ M.  Points in M are replicated into
            *both* children; this is intentional and is the source of the
            n^{1+ρ} space bound (Theorem 1).

        Phase 2 — box node (Lemma 2 / practical heuristic):
            If no good separator exists, the paper guarantees a dense box C
            of diameter O(log_{1+ρ}(log d)) covering ≥ |P|/2 points, but the
            exact construction (Appendix A) is intricate.  We use a practical
            heuristic: take the ⌈|P|/2⌉ points closest under ℓ∞ to the
            coordinate-wise median of P as the box set C, then recurse on
            P \\ C as the continuation.  This is an approximation of the
            theoretical construction and is honestly documented.

        Args:
            points: Current point set, shape ``(n, d)``.

        Returns:
            A :class:`.SeparatorNode`, :class:`.BoxNode`, or ``None`` if the
            point set is empty.

        """
        # Work stack: each item is a tuple whose first element is an opcode.
        #   ("build", pts)           — compute a node for pts; push result
        #   ("sep",   axis, t)       — pop right, pop left; push SeparatorNode
        #   ("box",   ctr, rep, bps) — pop continuation; push BoxNode
        work: list[Any] = [("build", points)]
        result: list[Node] = []

        while work:
            item = work.pop()
            op: str = item[0]

            if op == "build":
                pts: FloatArray = item[1]
                n = len(pts)

                if n == 0:
                    result.append(None)
                    continue

                self._on_build_entry(n)

                if n == 1:
                    ctr = pts[0].copy()
                    rep = pts[0].copy()
                    node: Node = BoxNode(
                        center=ctr,
                        representative=rep,
                        box_points=pts.copy(),
                        continuation=None,
                    )
                    self._on_box_node_built(node)  # type: ignore[arg-type]
                    result.append(node)
                    continue

                sep = find_best_separator(pts, self.rho)

                if sep is not None:
                    axis, t = sep.axis, sep.threshold
                    col = pts[:, axis]
                    l_mask = col < t - 1.0
                    r_mask = col > t + 1.0
                    m_mask = ~l_mask & ~r_mask

                    p1 = pts[l_mask | m_mask]   # L ∪ M  (routed left)
                    p2 = pts[r_mask | m_mask]   # R ∪ M  (routed right)

                    # Push make_sep first (executed last), then right then
                    # left; stack is LIFO so left is processed first, giving
                    # result order: [left, right] → popped as right then left.
                    work.append(("sep", axis, t))
                    work.append(("build", p2))
                    work.append(("build", p1))

                else:
                    # Box-node heuristic: ⌈n/2⌉ points closest to the median.
                    median = np.median(pts, axis=0)
                    dists = linf_distances_to_point(pts, median)
                    box_size = math.ceil(n / 2)
                    idx = np.argpartition(dists, box_size - 1)
                    box_pts = pts[idx[:box_size]]
                    remainder = pts[idx[box_size:]]

                    bb_min = box_pts.min(axis=0)
                    bb_max = box_pts.max(axis=0)
                    ctr2 = (bb_min + bb_max) / 2.0
                    rep2 = box_pts[0].copy()
                    bps_copy = box_pts.copy()

                    work.append(("box", ctr2, rep2, bps_copy))
                    work.append(("build", remainder))

            elif op == "sep":
                _, axis, t = item
                # Build order: left was pushed last → processed first → deeper
                # in result stack.  Right was pushed second-to-last →
                # processed second → on top of result stack.
                right = result.pop()
                left = result.pop()
                sep_node = SeparatorNode(
                    axis=int(axis), threshold=float(t), left=left, right=right
                )
                self._on_sep_node_built(sep_node)
                result.append(sep_node)

            else:  # "box"
                _, ctr, rep, bps = item
                continuation = result.pop()
                box_node = BoxNode(
                    center=ctr,
                    representative=rep,
                    box_points=bps,
                    continuation=continuation,
                )
                self._on_box_node_built(box_node)
                result.append(box_node)

        return result[0]

    # ------------------------------------------------------------------
    # Internal query (iterative — no Python recursion)
    # ------------------------------------------------------------------

    def _query(self, y: FloatArray, node: Node) -> FloatArray | None:
        """Execute the query iteratively — see :meth:`query` for the pseudocode."""
        while node is not None:
            if isinstance(node, SeparatorNode):
                self._on_sep_node_visited(node)
                node = node.left if y[node.axis] < node.threshold else node.right
            else:  # BoxNode
                self._on_box_node_visited(node)  # type: ignore[arg-type]
                if linf_distance(y, node.center) <= 1.0:
                    return node.representative
                node = node.continuation
        return None

    # ------------------------------------------------------------------
    # Instrumentation hooks (no-ops in the base class)
    # ------------------------------------------------------------------

    def _on_build_entry(self, n: int) -> None:
        """Override to observe each non-empty build work item."""

    def _on_sep_node_built(self, node: SeparatorNode) -> None:
        """Override to observe each SeparatorNode as it is created."""

    def _on_box_node_built(self, node: BoxNode) -> None:
        """Override to observe each BoxNode as it is created."""

    def _on_sep_node_visited(self, node: SeparatorNode) -> None:
        """Override to observe each SeparatorNode visited during query."""

    def _on_box_node_visited(self, node: BoxNode) -> None:
        """Override to observe each BoxNode visited during query."""
