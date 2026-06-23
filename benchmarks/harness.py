"""Measurement harness: run the benchmark grid and write results to CSV.

Grid: n × d × rho × distribution, with N_QUERIES queries per instance.
Each row in the output CSV is one query trial.
"""

from __future__ import annotations

import csv
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .generators import GENERATORS
from .instrumented import InstrumentedIndykTree

# Per-configuration build timeout in seconds.  Configurations that take longer
# are recorded as skipped so the grid keeps moving.
BUILD_TIMEOUT_S = 30


class _BuildTimeoutError(Exception):
    """Raised by SIGALRM when a build exceeds BUILD_TIMEOUT_S."""


def _alarm_handler(signum: int, frame: object) -> None:  # noqa: ANN001
    raise _BuildTimeoutError()

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

N_VALUES = [100, 500, 2000]
D_VALUES = [2, 5, 10, 20, 40]
RHO_VALUES = [0.3, 0.6, 1.0]
DIST_NAMES = list(GENERATORS.keys())

CSV_FIELDS = [
    "n", "d", "rho", "distribution", "seed",
    # Build metrics
    "build_time_s", "total_nodes", "sep_nodes", "box_nodes",
    "total_memberships", "max_depth", "box_node_frac",
    # Per-query metrics
    "query_id", "query_time_s", "nodes_visited", "sep_visits", "box_visits",
    "box_visit_frac", "used_fallback",
    "true_dist", "approx_dist", "approx_ratio",
]


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

    Returns a list of row dicts (one per query), or an empty list on error.
    """
    rng = np.random.default_rng(seed)
    gen_fn = GENERATORS[dist_name]

    # Some generators need rho; pass it as a kwarg; ignore if not accepted.
    try:
        if dist_name in ("acp08_adversarial", "boundary_stress"):
            points, queries = gen_fn(n, d, rho, rng=rng)  # type: ignore[call-arg]
        else:
            points, queries = gen_fn(n, d, rng=rng)  # type: ignore[call-arg]
    except ValueError as exc:
        # ACP08 raises for rho < 1.
        print(f"  SKIP {dist_name} n={n} d={d} rho={rho}: {exc}")
        return []
    except Exception:
        print(f"  ERROR generating {dist_name} n={n} d={d} rho={rho}")
        traceback.print_exc()
        return []

    # Build tree with a hard wall-clock timeout.
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
        return []
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        print(f"  ERROR building {dist_name} n={n} d={d} rho={rho}")
        traceback.print_exc()
        return []
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
        # Brute-force ground truth.
        true_nn, true_dist = brute_force_nn(points, query)

        # Tree query with instrumentation.
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
                        seed = abs(hash((n, d, rho, dist, base_seed))) % (2**31)
                        print(
                            f"[{done}/{total}] n={n:>5} d={d:>3} rho={rho} "
                            f"dist={dist:<20}",
                            end="  ",
                            flush=True,
                        )
                        t0 = time.perf_counter()
                        rows = run_instance(n, d, rho, dist, seed)
                        elapsed = time.perf_counter() - t0
                        if rows:
                            writer.writerows(rows)
                            f.flush()
                            # Quick summary for the terminal.
                            build_t = rows[0]["build_time_s"]
                            avg_ratio = float(
                                np.mean([r["approx_ratio"] for r in rows
                                         if r["approx_ratio"] != float("inf")])
                            )
                            n_fallback = sum(r["used_fallback"] for r in rows)
                            print(
                                f"build={build_t:.3f}s  "
                                f"ratio={avg_ratio:.2f}  "
                                f"fallback={n_fallback}/{len(rows)}  "
                                f"total={elapsed:.2f}s"
                            )
                        else:
                            print("(skipped)")

    print(f"\nResults written to {csv_path}")
    return csv_path
