"""Generate benchmark plots from benchmark_raw.csv.

Plot outputs are written to results/ as PNG files:
  (a) space_vs_n.png              — total_memberships vs n (log-log), one line per rho,
                                     with n^{1+rho} reference overlaid
  (b) approx_ratio_vs_d.png       — mean approx ratio vs d, one line per distribution
  (c) box_visit_frac.png          — mean box-visit fraction per distribution (bar chart)
  (d) query_time_vs_d_logn.png    — mean query time vs d·log(n), one line per rho
  (e) clustered_scatter.png       — scatter: approx_ratio (y, log) vs true_dist (x, log)
                                     for clustered distribution only
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

DIST_COLORS = {
    "uniform_random": "#1f77b4",
    "clustered": "#ff7f0e",
    "acp08_adversarial": "#d62728",
    "cyclic_closeness": "#2ca02c",
    "boundary_stress": "#9467bd",
}

RHO_MARKERS = {0.3: "o", 0.6: "s", 1.0: "^"}


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Helper: aggregate build-level rows (one per config × repeat)
# ---------------------------------------------------------------------------


def _build_level(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to one row per configuration (drop per-query variation)."""
    group_keys = ["n", "d", "rho", "distribution", "seed"]
    if "repeat" in df.columns:
        group_keys.append("repeat")
    return (
        df.groupby(group_keys, as_index=False)
        .agg(
            total_memberships=("total_memberships", "first"),
            total_nodes=("total_nodes", "first"),
            box_node_frac=("box_node_frac", "first"),
            build_time_s=("build_time_s", "first"),
            max_depth=("max_depth", "first"),
        )
    )


# ---------------------------------------------------------------------------
# Plot (a): space vs n  (log-log)
# ---------------------------------------------------------------------------


def plot_space_vs_n(df: pd.DataFrame, results_dir: Path) -> None:
    """Log-log plot of total_memberships vs n, one curve per rho.

    Overlays the theoretical n^{1+rho} reference for each rho.
    Averaged over all distributions and d values to show the main trend.
    """
    bdf = _build_level(df)
    agg = (
        bdf.groupby(["n", "rho"], as_index=False)
        .agg(mean_mem=("total_memberships", "mean"))
    )

    rhos = sorted(agg["rho"].unique())
    fig, ax = plt.subplots(figsize=(6, 5))

    for rho in rhos:
        sub = agg[agg["rho"] == rho].sort_values("n")
        ns = sub["n"].values
        mems = sub["mean_mem"].values
        m = RHO_MARKERS.get(rho, "o")
        (line,) = ax.loglog(ns, mems, marker=m, label=f"ρ={rho}", linewidth=2)
        ref = ns[0] ** (1 + rho)
        scale = mems[0] / ref
        ax.loglog(
            ns,
            scale * ns ** (1 + rho),
            linestyle="--",
            color=line.get_color(),
            alpha=0.5,
            linewidth=1,
        )

    ax.set_xlabel("n (database size)")
    ax.set_ylabel("Total point-memberships (log)")
    ax.set_title("Space usage vs n  (dashed = n^{1+ρ} reference)")
    ax.legend()
    _savefig(fig, results_dir / "space_vs_n.png")


# ---------------------------------------------------------------------------
# Plot (b): approx ratio vs d
# ---------------------------------------------------------------------------


def plot_approx_ratio_vs_d(df: pd.DataFrame, results_dir: Path) -> None:
    """Mean approximation ratio vs dimension, one line per distribution."""
    clean = df[df["approx_ratio"] != float("inf")].copy()

    agg = (
        clean.groupby(["d", "distribution"], as_index=False)
        .agg(mean_ratio=("approx_ratio", "mean"))
    )

    dists = sorted(agg["distribution"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))

    for dist in dists:
        sub = agg[agg["distribution"] == dist].sort_values("d")
        ax.plot(
            sub["d"],
            sub["mean_ratio"],
            marker="o",
            label=dist,
            color=DIST_COLORS.get(dist),
            linewidth=2,
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="exact (=1)")
    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Mean approximation ratio (approx_dist / true_dist)")
    ax.set_title("Approximation quality vs dimension")
    ax.legend(fontsize=8)
    _savefig(fig, results_dir / "approx_ratio_vs_d.png")


# ---------------------------------------------------------------------------
# Plot (c): box-visit fraction per distribution  (bar chart)
# ---------------------------------------------------------------------------


def plot_box_visit_frac(df: pd.DataFrame, results_dir: Path) -> None:
    """Mean fraction of node visits that are box-node visits, per distribution."""
    agg = (
        df.groupby("distribution", as_index=False)
        .agg(mean_bvf=("box_visit_frac", "mean"))
        .sort_values("mean_bvf", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [DIST_COLORS.get(d, "#888888") for d in agg["distribution"]]
    ax.bar(agg["distribution"], agg["mean_bvf"], color=colors, edgecolor="black")
    ax.set_xlabel("Distribution")
    ax.set_ylabel("Mean box-visit fraction")
    ax.set_title("Box-node visit fraction by distribution")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    _savefig(fig, results_dir / "box_visit_frac.png")


# ---------------------------------------------------------------------------
# Plot (d): query time vs d * log(n)
# ---------------------------------------------------------------------------


def plot_query_time_vs_d_logn(df: pd.DataFrame, results_dir: Path) -> None:
    """Mean query time vs d·log(n), one line per rho."""
    tmp = df.copy()
    tmp["d_logn"] = tmp["d"] * np.log(tmp["n"])

    agg = (
        tmp.groupby(["d_logn", "rho"], as_index=False)
        .agg(mean_qt=("query_time_s", "mean"))
        .sort_values("d_logn")
    )

    rhos = sorted(agg["rho"].unique())
    fig, ax = plt.subplots(figsize=(6, 5))

    for rho in rhos:
        sub = agg[agg["rho"] == rho].sort_values("d_logn")
        m = RHO_MARKERS.get(rho, "o")
        ax.plot(
            sub["d_logn"],
            sub["mean_qt"] * 1e3,
            marker=m,
            label=f"ρ={rho}",
            linewidth=2,
        )

    ax.set_xlabel("d · log(n)")
    ax.set_ylabel("Mean query time (ms)")
    ax.set_title("Query time vs d·log(n)")
    ax.legend()
    _savefig(fig, results_dir / "query_time_vs_d_logn.png")


# ---------------------------------------------------------------------------
# Plot (e): clustered scatter — approx_ratio vs true_dist
# ---------------------------------------------------------------------------


def plot_clustered_scatter(df: pd.DataFrame, results_dir: Path) -> None:
    """Scatter of approx_ratio (y, log) vs true_dist (x, log) for clustered data.

    Restricted to rows where distribution='clustered', failure_reason='ok',
    and true_dist > 1e-12 (to avoid log-of-zero).  Points are coloured by rho.
    """
    sub = df[
        (df["distribution"] == "clustered")
        & (df["failure_reason"] == "ok")
        & (df["approx_ratio"] != float("inf"))
        & (df["true_dist"] > 1e-12)
    ].copy()

    if sub.empty:
        print("  (no clustered rows — skipping clustered_scatter.png)")
        return

    rhos = sorted(sub["rho"].unique())
    rho_colors = {0.3: "#1f77b4", 0.6: "#ff7f0e", 1.0: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 5))

    for rho in rhos:
        s = sub[sub["rho"] == rho]
        ax.scatter(
            s["true_dist"],
            s["approx_ratio"],
            alpha=0.25,
            s=8,
            color=rho_colors.get(rho, "#888888"),
            label=f"ρ={rho}",
            rasterized=True,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, label="exact (=1)")
    ax.set_xlabel("True ℓ∞ distance to nearest neighbor (log)")
    ax.set_ylabel("Approximation ratio (log)")
    ax.set_title("Clustered distribution: approx_ratio vs true_dist")
    ax.legend(fontsize=8)
    _savefig(fig, results_dir / "clustered_scatter.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate_all_plots(csv_path: Path, results_dir: Path) -> None:
    """Load CSV and generate all five plots (ok rows only)."""
    df_raw = pd.read_csv(csv_path)
    print(f"Loaded {len(df_raw):,} rows from {csv_path}")

    df = df_raw[df_raw["failure_reason"] == "ok"].copy()
    print(f"  → {len(df):,} rows with failure_reason='ok' used for plots")

    results_dir.mkdir(parents=True, exist_ok=True)

    plot_space_vs_n(df, results_dir)
    plot_approx_ratio_vs_d(df, results_dir)
    plot_box_visit_frac(df, results_dir)
    plot_query_time_vs_d_logn(df, results_dir)
    plot_clustered_scatter(df_raw, results_dir)

    print("All plots generated.")
