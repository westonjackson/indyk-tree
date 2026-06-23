"""Geometric helpers for Indyk's ℓ∞ ANN construction.

All distance and separator-scoring logic lives here so that ``tree.py`` stays
focused on the recursive structure.

Reference: Piotr Indyk, "On Approximate Nearest Neighbors under ℓ∞ Norm,"
Journal of Computer and System Sciences 63(4), pp. 627-638, 2001.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def linf_distance(a: FloatArray, b: FloatArray) -> float:
    """Return the ℓ∞ (Chebyshev) distance between two vectors.

    Args:
        a: First point, shape ``(d,)``.
        b: Second point, shape ``(d,)``.

    Returns:
        ``max_i |a[i] - b[i]|``.

    """
    return float(np.max(np.abs(a - b)))


def linf_distances_to_point(points: FloatArray, query: FloatArray) -> FloatArray:
    """Return ℓ∞ distances from every row of *points* to *query*.

    Args:
        points: Array of shape ``(n, d)``.
        query: Array of shape ``(d,)``.

    Returns:
        Array of shape ``(n,)`` with ℓ∞ distances.

    """
    result: FloatArray = np.max(np.abs(points - query), axis=1)
    return result


class SeparatorCandidate(NamedTuple):
    """A (coordinate, threshold) pair together with its separator score.

    Attributes:
        axis: Coordinate index.
        threshold: Split threshold *t*.
        score: Separator score (lower is better; 0 means perfect split).
        m_count: Number of points in the middle slab ``[t-1, t+1]``.

    """

    axis: int
    threshold: float
    score: float
    m_count: int


def _compute_score(l_frac: float, r_frac: float, m_frac: float) -> float | None:
    """Compute the separator score for a given L/R/M partition.

    The score is defined in Section 3 of Indyk (2001) as::

        score = log(1/m - 1) / log(1/(m+r))   if m > 0
        score = 0                               if m == 0

    where l, r, m are the fractions of points in the left, right, and middle
    slabs respectively (l + r + m = 1).

    A separator is "good" if ``score <= rho``.

    Args:
        l_frac: Fraction of points in L (left slab).
        r_frac: Fraction of points in R (right slab).
        m_frac: Fraction of points in M (middle slab).

    Returns:
        The score, or ``None`` if the candidate should be skipped (e.g.
        ``m + r == 1`` which means ``l == 0`` — guarded for numerical safety
        even though the ``l > 1/(4d)`` pre-filter already excludes this).

    """
    if m_frac == 0.0:
        return 0.0

    # The score formula log(1/m-1)/log(1/(m+r)) is only meaningful for m < 0.5.
    # When m >= 0.5, the numerator log(1/m-1) becomes <= 0, making score <= 0,
    # which looks "good" (< rho) but reflects catastrophic replication: both
    # children contain at least m*n >= n/2 replicated points, so the recursion
    # produces ~2^(31) nodes for n=200.  Reject such candidates explicitly.
    if m_frac >= 0.5:
        return None

    mr = m_frac + r_frac
    # Guard: log(1/(m+r)) == 0 when m+r == 1, i.e. l == 0.
    if mr >= 1.0 - 1e-12:
        return None

    numerator = math.log(1.0 / m_frac - 1.0)
    denominator = math.log(1.0 / mr)
    return numerator / denominator


def find_best_separator(
    points: FloatArray,
    rho: float,
) -> SeparatorCandidate | None:
    """Search all coordinates and candidate thresholds for a good separator.

    Implements the separator-search subroutine from Section 3 of Indyk (2001).

    **Candidate thresholds** for coordinate *i*:
    1. ``v - 1`` and ``v + 1`` for each distinct value *v* — the breakpoints
       where a data point crosses an L/M/R slab boundary as *t* varies.
    2. Midpoints between consecutive sorted breakpoints — probing the interior
       of each stable L/M/R region catches ``m = 0`` separators that lie in a
       data-free gap (e.g., two well-separated clusters).  These are never
       themselves breakpoints yet have score 0 and are the best possible split.

    A candidate ``(i, t)`` is **good** if:
    * ``l > 1/(4d)``  — enough points on the left,
    * ``r > 1/(4d)``  — enough points on the right,
    * ``score <= rho``.

    Among all good candidates the one with the **smallest score** is returned
    (ties broken by smallest ``|M|``).  This selection rule is an
    implementation decision: the theorem only guarantees the *existence* of a
    good separator, not a unique choice.  Picking the smallest score tends to
    minimise replication and thus space usage.

    Implementation: candidate evaluation is fully vectorised using
    ``np.searchsorted`` on the pre-sorted column, reducing per-axis work from
    O(n × candidates) to O(n log n + candidates × log n).

    Args:
        points: Array of shape ``(n, d)``.
        rho: Quality parameter (typically in ``(0, 2]``).

    Returns:
        The best :class:`SeparatorCandidate`, or ``None`` if no good separator
        exists for any coordinate or threshold.

    """
    n, d = points.shape
    if n == 0:
        return None

    threshold_l = 1.0 / (4.0 * d)
    best: SeparatorCandidate | None = None

    for axis in range(d):
        col = points[:, axis]
        col_sorted = np.sort(col)
        distinct = np.unique(col)

        # Breakpoint candidates (mandated by the paper) plus midpoints of
        # stable intervals between consecutive breakpoints (to catch m=0 gaps).
        breakpoints = np.concatenate([distinct - 1.0, distinct + 1.0])
        bp_sorted = np.sort(np.unique(breakpoints))
        midpoints = (bp_sorted[:-1] + bp_sorted[1:]) / 2.0
        ts = np.unique(np.concatenate([bp_sorted, midpoints]))

        # Vectorised L/R counts via binary search: O(|ts| * log n).
        # l_counts[k] = #{x : x < ts[k] - 1}
        l_counts = np.searchsorted(col_sorted, ts - 1.0, side="left")
        # r_counts[k] = #{x : x > ts[k] + 1}
        r_counts = n - np.searchsorted(col_sorted, ts + 1.0, side="right")

        l_fracs = l_counts / n
        r_fracs = r_counts / n
        m_counts = n - l_counts - r_counts
        m_fracs = np.clip(m_counts / n, 0.0, 1.0)

        # Pre-filter: both sides must be non-trivial.
        valid = (l_fracs > threshold_l) & (r_fracs > threshold_l)
        if not valid.any():
            continue

        # Compute scores only for valid candidates.
        for k in np.where(valid)[0]:
            l_f = float(l_fracs[k])
            r_f = float(r_fracs[k])
            m_f = float(m_fracs[k])

            score = _compute_score(l_f, r_f, m_f)
            if score is None or score > rho:
                continue

            mc = int(m_counts[k])
            candidate = SeparatorCandidate(
                axis=axis,
                threshold=float(ts[k]),
                score=score,
                m_count=mc,
            )

            if (
                best is None
                or score < best.score
                or (score == best.score and mc < best.m_count)
            ):
                best = candidate

    return best


def expected_diameter_bound(rho: float, d: int) -> float:
    """Return the theoretical ℓ∞ diameter bound for a box node.

    From Lemma 2 of Indyk (2001): when no good separator exists, at least
    half the points lie in a box of ℓ∞ diameter at most::

        4 * ceil(log_{1+rho}(log(4*d)))

    This is the quantity used in correctness tests to validate the
    approximation guarantee.

    Args:
        rho: Quality parameter.
        d: Dimension.

    Returns:
        The diameter bound as a float.

    """
    if d <= 0 or rho <= 0:
        return float("inf")
    inner = math.log(4.0 * d)
    if inner <= 0:
        return 0.0
    base = 1.0 + rho
    log_val = math.log(inner) / math.log(base)
    return 4.0 * math.ceil(log_val)
