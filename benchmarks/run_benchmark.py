"""CLI entry point for the IndykTree benchmarking harness.

Usage:
    python -m benchmarks.run_benchmark [OPTIONS]

Options:
    --results-dir PATH  Directory for CSV and plots [default: results/]
    --n      INT ...    Override n values (space-separated)
    --d      INT ...    Override d values
    --rho    FLOAT ...  Override rho values
    --dist   STR  ...   Override distribution names
    --seed   INT        Base random seed [default: 42]
    --no-plots          Skip generating plots (useful for quick smoke tests)
    --plots-only PATH   Only generate plots from an existing CSV at PATH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .harness import D_VALUES, DIST_NAMES, N_VALUES, RHO_VALUES, run_grid
from .plots import generate_all_plots


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run IndykTree benchmarks and generate plots."
    )
    p.add_argument("--results-dir", default="results", help="Output directory")
    p.add_argument("--n", nargs="+", type=int, default=None)
    p.add_argument("--d", nargs="+", type=int, default=None)
    p.add_argument("--rho", nargs="+", type=float, default=None)
    p.add_argument("--dist", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plots-only", metavar="CSV_PATH", default=None)
    return p.parse_args()


def main() -> None:
    """Parse arguments and run the benchmark grid and/or plot generation."""
    args = _parse_args()
    results_dir = Path(args.results_dir)

    if args.plots_only:
        csv_path = Path(args.plots_only)
        if not csv_path.exists():
            print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
            sys.exit(1)
        generate_all_plots(csv_path, results_dir)
        return

    csv_path = run_grid(
        results_dir=results_dir,
        n_values=args.n or N_VALUES,
        d_values=args.d or D_VALUES,
        rho_values=args.rho or RHO_VALUES,
        dist_names=args.dist or DIST_NAMES,
        base_seed=args.seed,
    )

    if not args.no_plots:
        generate_all_plots(csv_path, results_dir)


if __name__ == "__main__":
    main()
