"""Node types for Indyk's ℓ∞ ANN tree.

Two node types arise from the recursive construction in Indyk (JCSS 2001):

* SeparatorNode — a coordinate/threshold split found by the separator-search
  subroutine (Section 3 of the paper).  Points in M (the "middle slab") are
  replicated into *both* children, which is the source of the n^(1+ρ) space
  bound stated in Theorem 1.

* BoxNode — created when no good separator exists.  In that case the paper
  (Lemma 2 / Appendix A) guarantees that at least half of the remaining points
  lie inside a box of ℓ∞ diameter O(log_{1+ρ}(log d)).  The implementation
  here uses a practical heuristic to find that box (see IndykTree._build for
  details) rather than the full inductive construction from Appendix A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Forward-declared union; `from __future__ import annotations` makes all
# annotations strings, so the self-referential type is resolved lazily.
Node = "SeparatorNode | BoxNode | None"


@dataclass
class SeparatorNode:
    """A node that splits the point set along coordinate *axis* at threshold *t*.

    The query algorithm routes to ``left`` when ``y[axis] < t`` and to
    ``right`` otherwise.  This matches the query pseudocode in Section 3 of
    Indyk (2001) exactly.

    Attributes:
        axis: Coordinate index used for the split (0-based).
        threshold: Split value *t*.  Left child contains L ∪ M
            (x[axis] <= t+1); right child contains R ∪ M (x[axis] >= t-1).
        left: Subtree for the left/middle slab (P1 = L ∪ M).
        right: Subtree for the right/middle slab (P2 = R ∪ M).

    """

    axis: int
    threshold: float
    left: SeparatorNode | BoxNode | None
    right: SeparatorNode | BoxNode | None


@dataclass
class BoxNode:
    r"""A node that represents a dense "box" of nearby points.

    Created when the separator-search subroutine fails to find any good
    separator.  The paper guarantees (Lemma 2) that a box C of diameter
    O(log_{1+ρ}(log d)) containing at least |P|/2 points exists, but does not
    give a polynomial-time algorithm to find it exactly.  The implementation
    uses a practical heuristic: the |P|/2 points closest (under ℓ∞) to the
    coordinate-wise median of P.

    The ``continuation`` field is the tree built recursively on P \\ C; it
    handles queries whose nearest neighbor is not inside the box.

    Note: the separate "balanced binary tree over C" mentioned in the
    space-accounting lemma of Indyk (2001) is *not* built here — that
    structure exists only to make a theoretical space proof go through and is
    not required for correct query behavior.

    Attributes:
        center: Coordinate-wise midpoint of C's bounding box.  A query point
            whose ℓ∞ distance to ``center`` is ≤ 1 is answered with
            ``representative``.
        representative: A single point from C returned as the approximate
            nearest neighbor for queries near the box.
        box_points: All points in C (stored for inspection and testing).
        continuation: Subtree built on P \\ C.

    """

    center: FloatArray
    representative: FloatArray
    box_points: FloatArray
    continuation: SeparatorNode | BoxNode | None
