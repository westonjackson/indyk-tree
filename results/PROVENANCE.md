# Benchmark Provenance

## Purpose

This file documents the exact grid parameters, expected row counts, and
reconciliation against the previous run, so that the resulting benchmark
numbers in `benchmark_raw.csv` can be cited without caveats.

---

## Grid parameters (verbatim from `benchmarks/harness.py`)

```python
N_VALUES   = [100, 500, 2000]
D_VALUES   = [2, 5, 10, 20, 40]
RHO_VALUES = [0.3, 0.6, 1.0]
DIST_NAMES = ["uniform_random", "clustered", "acp08_adversarial",
              "cyclic_closeness", "boundary_stress"]
N_REPEATS  = 5   # independent seeds per (n, d, rho, dist) config
BASE_SEED  = 42
BUILD_TIMEOUT_S = 30
```

---

## Configuration count

| Factor          | Levels                     | Count |
|-----------------|----------------------------|-------|
| n               | 100, 500, 2000             | 3     |
| d               | 2, 5, 10, 20, 40           | 5     |
| rho             | 0.3, 0.6, 1.0              | 3     |
| distribution    | 5 distributions            | 5     |
| repeat          | 0 … 4 (5 independent seeds)| 5     |
| **Total**       |                            | **1125** |

Expected outcomes per config:

- **ACP08-invalid** (failure_reason="invalid_params"): the `acp08_adversarial`
  generator raises `ValueError` for ρ ≤ 0.5 because the marginal series
  Σ 2^{-(2ρ)^i} diverges, making the distribution ill-defined.  Both ρ=0.3
  and ρ=0.6 are invalid.  That is 3 n × 5 d × 2 invalid rho × 5 repeats =
  **150 sentinel rows** expected.

- **Timeouts** (failure_reason="timeout"): the previous single-seed run had 13
  clustered configs time out.  With 5 repeats those same configs are expected
  to time out on each repeat, yielding up to **65 timeout sentinels**
  (13 configs × 5 repeats), modulo any seed-dependent variation.

- **Remaining** (failure_reason="ok"): 1125 − 150 − (up to 65) ≈ **910–1125**
  ok configs, each producing N_QUERIES=50 rows, so **~45,500–56,250 ok rows**.

---

## Git state at run time

- **Commit hash**: `7d1beac`
- **Working tree**: clean (only `results_archive/` is untracked; all
  benchmarking code changes are committed)
- **Verification**: `git status --porcelain` output at run time:
  `?? results_archive/`

---

## Reconciliation against previous run ("182 configurations, 30 invalid ACP08 skips")

The previous run (archived at `results_archive/7d1beac_20260624T020810Z/`)
used a single seed per config (no `repeat` dimension):

| Category               | Previous run | This run (×5 repeats) |
|------------------------|:------------:|:---------------------:|
| Total configs          | 225          | 1125                  |
| ACP08-invalid skips    | 30           | 150                   |
| Timeout configs        | 13           | up to 65              |
| Active (ok) configs    | 182          | up to 910             |

Previous "182 active configurations" = 225 total − 30 invalid_params −
13 timeout = **182 ✓**.

This run adds the `repeat` dimension (5×) and a `repeat` column to every CSV
row.  The seed derivation uses SHA-256 keyed on
`f"{n}|{d}|{rho}|{dist}|{base_seed}|{repeat}"`, so repeat=0 for this run
uses different seeds than the prior run's single-seed derivation (the prior
run did not include `|{repeat}` in the key).  All prior results are archived
and should not be mixed with this run's data.

---

## Cross-repeat variance methodology

For each (n, d, rho, distribution) configuration, compute the mean
`approx_ratio` across the 50 queries within each repeat, yielding 5 per-repeat
mean ratios.  The standard deviation of those 5 numbers is the cross-repeat
variance.  A configuration is flagged **UNSTABLE** if std/mean > 0.25.

---

*Generated: 2026-06-23*
*Run command: `python -m benchmarks.run_benchmark --repeats 5`*
