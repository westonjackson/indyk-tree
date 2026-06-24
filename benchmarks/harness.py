"""Measurement harness: run the benchmark grid and write results to CSV.

Grid: n × d × rho × distribution, with N_QUERIES queries per instance.
Each row in the output CSV is one query trial (failure_reason="ok") or one
sentinel row per failed configuration (failure_reason ∈ {"timeout",
"recursion_limit", "invalid_params", "other_error"}).
"""

from __future__ import annotations

import csv
import hashlib
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .generators import GENERATORS
from .instrumented import InstrumentedIndykTree

# Per-configuration build timeout in seconds.
BUILD_TIMEOUT_S = 30


class _BuildTimeoutError(Exception):
    """Raised by SIGALRM when a build exceeds BUILD_TIMEOUT_S."""


def _alarm_handler(signum: int, frame: object) -> None:  # noqa: ANN001
    raise _BuildTimeoutError()


# ---------------------------------------------------------------------------
# Deterministic seed derivation
# ---------------------------------------------------------------------------


def _deterministic_seed(
    n: int, d: int, rho: float, dist: str, base_seed: int
) -> int:
    """Derive a fully deterministic per-config seed, independent of PYTHONHASHSEED.

    Uses SHA-256 so the output is the same in every Python process regardless
    of the random hash-seed that Python injects for string objects.

    Args:
        n: Dataset size.
        d: Dimension.
        rho: Quality parameter.
        dist: Distribution name (a string, hence hash()-unsafe).
        base_seed: Top-level seed passed to run_grid.

    Returns:
        A non-negative integer suitable for use as a numpy random seed.

    """
    key = f"{n}|{d}|{rho}|{dist}|{base_seed}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big") % (2**31)


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

N_VALUES = [100, 500, 2000]
D_VALUES = [2, 5, 10, 20, 40]
RHO_VALUES = [0.3, 0.6, 1.0]
DIST_NAMES = list(GENERATORS.keys())

CSV_FIELDS = [
    "n", "d", "rho", "distribution", "seed",
    "failure_reason",
    # Build metrics (empty string in failure rows)
    "build_time_s", "total_nodes", "sep_nodes", "box_nodes",
    "total_memberships", "max_depth", "box_node_frac",
    # Per-query metrics (empty string in failure rows; query_id=-1)
    "query_id", "query_time_s", "nodes_visited", "sep_visits", "box_visits",
    "box_visit_frac", "used_fallback",
    "true_dist", "approx_dist", "approx_ratio",
]

_EMPTY_BUILD: dict[str, Any] = {
    "build_time_s": "", "total_nodes": "", "sep_nodes": "", "box_nodes": "",
    "total_memberships": "", "max_depth": "", "box_node_frac": "",
}
_EMPTY_QUERY: dict[str, Any] = {
    "query_id": -1, "query_time_s": "", "nodes_visited": "", "sep_visits": "",
    "box_visits": "", "box_visit_frac": "", "used_fallback": "",
    "true_dist": "", "approx_dist": "", "approx_ratio": "",
}


def _failure_row(
    n: int,
    d: int,
    rho: float,
    dist_name: str,
    seed: int,
    failure_reason: str,
    build_time_s: str | float = "",
) -> dict[str, Any]:
    """Build a sentinel CSV row for a configuration that could not be measured."""
    row: dict[str, Any] = {
        "n": n, "d": d, "rho": rho,
        "distribution": dist_name, "seed": seed,
        "failure_reason": failure_reason,
    }
    row.update(_EMPTY_BUILD)
    row.update(_EMPTY_QUERY)
    if build_time_s != "":
        row["build_time_s"] = build_time_s
    return row


# ---------------------------------------------------------------------------
# Brute-force nearest neighbor
# ---------------------------------------------------------------------------


def brute_force_nn(
    points: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return the exact ℓ∞ nearest neighbor and its distance."""
    dists = np.max(np.abs(points - query), axis=1)
    idx = int(np.argmin(dists))
    return points[idx], float(dists[idx])


# ---------------------------------------------------------------------------
# Single instance runner
# ---------------------------------------------------------------------------


def run_instance(
    n: int,
    d: int,
    rho: float,
    dist_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Run one (n, d, rho, distribution) configuration.

    Returns a list of row dicts.  On success this is one row per query
    (all with failure_reason="ok").  On failure it is exactly one sentinel
    row with the appropriate failure_reason.
    """
    rng = np.random.default_rng(seed)
    gen_fn = GENERATORS[dist_name]

    try:
        if dist_name in ("acp08_adversarial", "boundary_stress"):
            points, queries = gen_fn(n, d, rho, rng=rng)  # type: ignore[call-arg]
        else:
            points, queries = gen_fn(n, d, rng=rng)  # type: ignore[call-arg]
    except ValueError as exc:
        print(f"  SKIP {dist_name} n={n} d={d} rho={rho}: {exc}")
        return [_failure_row(n, d, rho, dist_name, seed, "invalid_params")]
    except Exception:
        print(f"  ERROR generating {dist_name} n={n} d={d} rho={rho}")
        traceback.print_exc()
        return [_failure_row(n, d, rho, dist_name, seed, "other_error")]

    # Build tree with a hard wall-clock timeout (SIGALRM, Unix only).
    tree = InstrumentedIndykTree(rho=rho)
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(BUILD_TIMEOUT_S)
    t0 = time.perf_counter()
    try:
        tree.build(points)
    except _BuildTimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        build_time = time.perf_counter() - t0
        print(f"  TIMEOUT building {dist_name} n={n} d={d} rho={rho} "
              f"after {build_time:.1f}s")
        return [_failure_row(n, d, rho, dist_name, seed, "timeout", build_time)]
    except RecursionError:
        # Caught separately so it is never conflated with a wall-clock timeout.
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        build_time = time.perf_counter() - t0
        print(f"  RECURSION LIMIT building {dist_name} n={n} d={d} rho={rho} "
              f"after {build_time:.3f}s")
        return [_failure_row(n, d, rho, dist_name, seed, "recursion_limit", build_time)]
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        build_time = time.perf_counter() - t0
        print(f"  ERROR building {dist_name} n={n} d={d} rho={rho}")
        traceback.print_exc()
        return [_failure_row(n, d, rho, dist_name, seed, "other_error", build_time)]
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)
    build_time = time.perf_counter() - t0

    bs = tree.build_stats
    bs.build_time = build_time
    box_node_frac = (
        bs.box_nodes / bs.total_nodes if bs.total_nodes > 0 else 0.0
    )

    rows: list[dict[str, Any]] = []

    for qid, query in enumerate(queries):
        true_nn, true_dist = brute_force_nn(points, query)

        t_q = time.perf_counter()
        try:
            approx = tree.query(query)
        except Exception:
            continue
        query_time = time.perf_counter() - t_q

        qs = tree.last_query_stats
        approx_dist = float(np.max(np.abs(approx - query)))

        if true_dist > 1e-12:
            ratio = approx_dist / true_dist
        else:
            ratio = 1.0 if approx_dist < 1e-12 else float("inf")

        bvf = (
            qs.box_visits / qs.nodes_visited if qs.nodes_visited > 0 else 0.0
        )

        rows.append({
            "n": n, "d": d, "rho": rho,
            "distribution": dist_name, "seed": seed,
            "failure_reason": "ok",
            "build_time_s": build_time,
            "total_nodes": bs.total_nodes,
            "sep_nodes": bs.separator_nodes,
            "box_nodes": bs.box_nodes,
            "total_memberships": bs.total_memberships,
            "max_depth": bs.max_depth,
            "box_node_frac": box_node_frac,
            "query_id": qid,
            "query_time_s": query_time,
            "nodes_visited": qs.nodes_visited,
            "sep_visits": qs.sep_visits,
            "box_visits": qs.box_visits,
            "box_visit_frac": bvf,
            "used_fallback": int(qs.used_fallback),
            "true_dist": true_dist,
            "approx_dist": approx_dist,
            "approx_ratio": ratio,
        })

    return rows


# ---------------------------------------------------------------------------
# Full grid
# ---------------------------------------------------------------------------


def run_grid(
    results_dir: Path,
    n_values: list[int] = N_VALUES,
    d_values: list[int] = D_VALUES,
    rho_values: list[float] = RHO_VALUES,
    dist_names: list[str] = DIST_NAMES,
    base_seed: int = 42,
) -> Path:
    """Run the full parameter grid and write results to CSV.

    Returns the path to the written CSV file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "benchmark_raw.csv"

    total = len(n_values) * len(d_values) * len(rho_values) * len(dist_names)
    done = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for rho in rho_values:
            for n in n_values:
                for d in d_values:
                    for dist in dist_names:
                        done += 1
                        seed = _deterministic_seed(n, d, rho, dist, base_seed)
                        print(
                            f"[{done}/{total}] n={n:>5} d={d:>3} rho={rho} "
                            f"dist={dist:<20}",
                            end="  ",
                            flush=True,
                        )
                        t0 = time.perf_counter()
                        rows = run_instance(n, d, rho, dist, seed)
                        elapsed = time.perf_counter() - t0

                        writer.writerows(rows)
                        f.flush()

                        ok_rows = [r for r in rows if r["failure_reason"] == "ok"]
                        if ok_rows:
                            build_t = ok_rows[0]["build_time_s"]
                            avg_ratio = float(
                                np.mean([r["approx_ratio"] for r in ok_rows
                                         if r["approx_ratio"] != float("inf")])
                            )
                            n_fallback = sum(r["used_fallback"] for r in ok_rows)
                            print(
                                f"build={build_t:.3f}s  "
                                f"ratio={avg_ratio:.2f}  "
                                f"fallback={n_fallback}/{len(ok_rows)}  "
                                f"total={elapsed:.2f}s"
                            )
                        else:
                            reason = rows[0]["failure_reason"] if rows else "unknown"
                            print(f"({reason})")

    print(f"\nResults written to {csv_path}")
    return csv_path
