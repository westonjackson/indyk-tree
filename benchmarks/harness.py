"""Measurement harness: run the benchmark grid and write results to CSV.

Grid: n × d × rho × distribution × repeat, with N_QUERIES queries per instance.
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

# Default number of independent random seeds per (n, d, rho, distribution).
N_REPEATS = 5


class _BuildTimeoutError(Exception):
    """Raised by SIGALRM when a build exceeds BUILD_TIMEOUT_S."""


def _alarm_handler(signum: int, frame: object) -> None:  # noqa: ANN001
    raise _BuildTimeoutError()


# ---------------------------------------------------------------------------
# Deterministic seed derivation
# ---------------------------------------------------------------------------


def _deterministic_seed(
    n: int,
    d: int,
    rho: float,
    dist: str,
    base_seed: int,
    repeat: int = 0,
) -> int:
    """Derive a fully deterministic per-config seed, independent of PYTHONHASHSEED.

    Uses SHA-256 so the output is the same in every Python process regardless
    of the random hash-seed that Python injects for string objects.  The
    ``repeat`` index is folded into the key so that independent runs for the
    same (n, d, rho, dist) config receive genuinely independent data.

    Args:
        n: Dataset size.
        d: Dimension.
        rho: Quality parameter.
        dist: Distribution name (a string, hence hash()-unsafe).
        base_seed: Top-level seed passed to run_grid.
        repeat: Independent-run index (0 … n_repeats-1).

    Returns:
        A non-negative integer suitable for use as a numpy random seed.

    """
    key = f"{n}|{d}|{rho}|{dist}|{base_seed}|{repeat}".encode()
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
    "n", "d", "rho", "distribution", "seed", "repeat",
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
    repeat: int = 0,
    build_time_s: str | float = "",
) -> dict[str, Any]:
    """Build a sentinel CSV row for a configuration that could not be measured."""
    row: dict[str, Any] = {
        "n": n, "d": d, "rho": rho,
        "distribution": dist_name, "seed": seed, "repeat": repeat,
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
    repeat: int = 0,
) -> list[dict[str, Any]]:
    """Run one (n, d, rho, distribution, repeat) configuration.

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
        print(f"  SKIP {dist_name} n={n} d={d} rho={rho} rep={repeat}: {exc}")
        return [_failure_row(n, d, rho, dist_name, seed, "invalid_params", repeat)]
    except Exception:
        print(f"  ERROR generating {dist_name} n={n} d={d} rho={rho} rep={repeat}")
        traceback.print_exc()
        return [_failure_row(n, d, rho, dist_name, seed, "other_error", repeat)]

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
        print(f"  TIMEOUT {dist_name} n={n} d={d} rho={rho} rep={repeat} "
              f"after {build_time:.1f}s")
        return [_failure_row(n, d, rho, dist_name, seed, "timeout", repeat, build_time)]
    except RecursionError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        build_time = time.perf_counter() - t0
        print(f"  RECURSION LIMIT {dist_name} n={n} d={d} rho={rho} rep={repeat} "
              f"after {build_time:.3f}s")
        return [_failure_row(
            n, d, rho, dist_name, seed, "recursion_limit", repeat, build_time
        )]
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        build_time = time.perf_counter() - t0
        print(f"  ERROR building {dist_name} n={n} d={d} rho={rho} rep={repeat}")
        traceback.print_exc()
        return [_failure_row(
            n, d, rho, dist_name, seed, "other_error", repeat, build_time
        )]
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
            "distribution": dist_name, "seed": seed, "repeat": repeat,
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
# Full grid  (multi-seed by default)
# ---------------------------------------------------------------------------


def run_grid(
    results_dir: Path,
    n_values: list[int] = N_VALUES,
    d_values: list[int] = D_VALUES,
    rho_values: list[float] = RHO_VALUES,
    dist_names: list[str] = DIST_NAMES,
    base_seed: int = 42,
    n_repeats: int = N_REPEATS,
) -> Path:
    """Run the full parameter grid and write results to CSV.

    Each (n, d, rho, distribution) combination is run ``n_repeats`` times
    with independent seeds, with a "repeat" column identifying each run.
    Progress (elapsed, ETA) is printed every 20 configurations.

    Returns the path to the written CSV file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "benchmark_raw.csv"

    n_configs = (
        len(n_values) * len(d_values) * len(rho_values)
        * len(dist_names) * n_repeats
    )
    done = 0
    run_start = time.perf_counter()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for rho in rho_values:
            for n in n_values:
                for d in d_values:
                    for dist in dist_names:
                        for repeat in range(n_repeats):
                            done += 1
                            seed = _deterministic_seed(
                                n, d, rho, dist, base_seed, repeat
                            )
                            print(
                                f"[{done}/{n_configs}] "
                                f"n={n:>5} d={d:>3} rho={rho} "
                                f"rep={repeat} dist={dist:<20}",
                                end="  ",
                                flush=True,
                            )
                            t0 = time.perf_counter()
                            rows = run_instance(n, d, rho, dist, seed, repeat)
                            elapsed_item = time.perf_counter() - t0

                            writer.writerows(rows)
                            f.flush()

                            ok_rows = [
                                r for r in rows if r["failure_reason"] == "ok"
                            ]
                            if ok_rows:
                                build_t = ok_rows[0]["build_time_s"]
                                finite = [
                                    r["approx_ratio"] for r in ok_rows
                                    if r["approx_ratio"] != float("inf")
                                ]
                                avg_ratio = (
                                float(np.mean(finite)) if finite else float("nan")
                            )
                                n_fallback = sum(r["used_fallback"] for r in ok_rows)
                                print(
                                    f"build={build_t:.3f}s  "
                                    f"ratio={avg_ratio:.2f}  "
                                    f"fb={n_fallback}/{len(ok_rows)}  "
                                    f"t={elapsed_item:.2f}s"
                                )
                            else:
                                reason = (
                                    rows[0]["failure_reason"] if rows else "unknown"
                                )
                                print(f"({reason})")

                            # Progress report every 20 configs.
                            if done % 20 == 0 or done == n_configs:
                                total_elapsed = time.perf_counter() - run_start
                                rate = total_elapsed / done
                                eta_s = rate * (n_configs - done)
                                print(
                                    f"  ── progress {done}/{n_configs} configs  "
                                    f"elapsed={total_elapsed/60:.1f}m  "
                                    f"ETA={eta_s/60:.1f}m ──"
                                )

    print(f"\nResults written to {csv_path}")
    return csv_path
