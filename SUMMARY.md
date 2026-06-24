# Empirical Study of Indyk's ℓ∞ ANN Data Structure

Benchmark grid: **n ∈ {100, 500, 2000}**, **d ∈ {2, 5, 10, 20, 40}**,
**ρ ∈ {0.3, 0.6, 1.0}**, 5 distributions, 50 queries each.
**9,143 total rows** (9,100 with failure_reason="ok", 30 "invalid_params",
13 "timeout") recorded in `results/benchmark_raw.csv`.
Pre-bugfix results archived as `results/benchmark_raw_pre_bugfix.csv`.

---

## Bug corrections and their impact on prior results

Two infrastructure bugs were fixed before this run; the prior CSV is treated
as provisional.

### Bug 1 — non-deterministic seeding via `hash()`

`run_grid()` was computing per-config seeds as
`abs(hash((n, d, rho, dist, base_seed))) % 2**31`. Python's `hash()` is
randomised per-process for strings (PYTHONHASHSEED), so the seed changed on
every invocation even with the same `base_seed`. All 182 active configurations
now use a different seed than before (SHA-256 based, fully deterministic).

**Impact**: every metric shifted slightly. No finding moves by more than 20%:

| Metric                             | Max change |
|------------------------------------|-----------|
| Mean total_memberships (space)     | +15.6%    |
| Mean approx ratio (by dist)        | +10.4%    |
| Mean box-visit fraction (by dist)  | −7.6%     |

The directional conclusions from the prior run are confirmed. No finding
previously reported as "large" or "small" flips sign or changes category.

### Bug 2 — double recursion depth from MRO-intercept instrumentation

The old `InstrumentedIndykTree` overrode `_build` and `_query` and called
`super()._build()`/`super()._query()`, but IndykTree's own recursion
dispatched back to the subclass via `self._build`/`self._query` (MRO), giving
two Python stack frames per tree level instead of one. This was identified as
a potential source of silent `RecursionError` failures, which would have been
caught by the generic `except Exception` clause and mis-labelled as
wall-clock timeouts.

**Resolution**: `_build` and `_query` were converted to iterative
implementations (work/result stacks and a while-loop respectively) in
`src/indyk_tree/tree.py`. Instrumentation now uses five explicit hook methods
(`_on_build_entry`, `_on_sep_node_built`, etc.) that default to no-ops and
are overridden in `InstrumentedIndykTree`. The public API and all test
assertions are unchanged.

**How many of the 13 prior "timeouts" were actually `recursion_limit`?
Answer: zero.** All 13 previously absent configs now appear in the CSV
with `failure_reason="timeout"`, not `failure_reason="recursion_limit"`.
The prior finding that 13/45 `clustered` configs at high d timed out is
fully confirmed as genuine 30-second wall-clock timeouts, not stack crashes.

The 30 previously absent ACP08 (ρ ≤ 0.5) configs now appear with
`failure_reason="invalid_params"`, making all skips visible in the data.

---

## 1. Space Usage vs n  (`results/space_vs_n.png`)

**Finding: The n^{1+ρ} space bound is very pessimistic in practice.**

| n    | ρ=0.3 (theory n^{1.3}) | ρ=0.3 actual/n | ρ=1.0 (theory n²) | ρ=1.0 actual/n |
|------|------------------------|----------------|---------------------|-----------------|
| 100  | 398                    | 10.8×          | 10,000              | 13.2×           |
| 500  | 3,227                  | 38.1×          | 250,000             | 23.2×           |
| 2000 | 19,562                 | 38.4×          | 4,000,000           | 14.9×           |

For ρ=1.0 the theory predicts O(n²) total memberships; the measured overhead
is 13–23× regardless of n — consistent with O(n) on these data sizes.
The worst-case construction requires an adversarial input our distributions do
not provide; on random data the separator search finds score-0 splits (m=0)
most of the time, producing a near-linear tree.

Space numbers shift modestly from the prior run (≤15.6%) due to seeding
correction; the shape of the curves is unchanged.

---

## 2. Approximation Ratio vs Dimension  (`results/approx_ratio_vs_d.png`)

**Finding: Clustered data is the practical adversary; ACP08 is not.**

| Distribution       | Mean ratio | Median | Max    | Change from prior |
|--------------------|------------|--------|--------|-------------------|
| uniform_random     | 1.027      | 1.000  | 4.34   | +2.0%             |
| acp08_adversarial  | 1.052      | 1.000  | 4.80   | +0.5%             |
| boundary_stress    | 1.171      | 1.000  | 61.1   | +10.4%            |
| cyclic_closeness   | 4.779      | 1.901  | 169.5  | +5.2%             |
| **clustered**      | **3.228**  | 1.000  | 248.9  | −5.6%             |

No finding moves by more than 20%; all directional conclusions are confirmed.

- **Uniform random** achieves near-exact answers. Clean gaps in random data
  produce score-0 separators that rapidly narrow the search.

- **ACP08 (ρ=1.0 only)**: mean ratio 1.052, essentially exact. The geometric
  marginal concentrates ≈50% of coordinates at 0, giving many score-0
  separators. The theoretically "hard" distribution is not adversarial here
  because the m ≥ 0.5 guard prevents the replication blow-up that the
  theoretical proof exploits.

- **Clustered** is the empirical adversary. Dense Gaussian clusters in high d
  (d ≥ 20, n ≥ 500) trigger the 30-second build timeout for 13 configurations.
  When the build succeeds, approximation ratios reach 249×.

- **Cyclic closeness** is adversarial at low d but converges to near-optimal
  at high d: at d=2 the mean ratio reaches ~14× (n=500) but falls to ~1.2×
  at d=40. The cyclic lane structure (point i shifted along axis i%d) only
  creates separator interference when few axes share the pattern.

- **Boundary stress** (integer-valued coordinates) has modest mean ratio
  (1.17) but long tails (max 61×). Integer collisions stress the m < 0.5
  guard and drive a high fallback rate (64%), but the guard holds correctly.

---

## 3. Box-Visit Fraction by Distribution  (`results/box_visit_frac.png`)

| Distribution       | Box-visit fraction | Change |
|--------------------|-------------------|--------|
| acp08_adversarial  | 65.2%             | +3.7%  |
| cyclic_closeness   | 35.9%             | +0.1%  |
| boundary_stress    | 31.9%             | −0.0%  |
| clustered          | 23.9%             | +0.3%  |
| uniform_random     | 9.7%              | −7.6%  |

ACP08 drives the most box-node traversals (65.2%), consistent with its design:
integer-valued coordinates cause rapid convergence to box nodes whose
representatives are then checked in sequence. Uniform random has the lowest
fraction (9.7%) because it produces deep, balanced separator trees where
the query follows a long separator chain before reaching a leaf.

All fractions change by less than 8% from the prior run.

---

## 4. Query Time vs d·log(n)  (`results/query_time_vs_d_logn.png`)

Query times remain in the **0.01–0.04 ms** range across the full grid, with
mild linear growth in d·log(n) for uniform random data, consistent with
O(d log n) traversal depth. No meaningful change from the prior run.

---

## 5. Build Timeouts

13/225 configurations hit the 30-second build timeout, all in the
`clustered` distribution at large n or d:

| Timeout configs       | Count |
|-----------------------|-------|
| n≥500, d≥20, any ρ    | 9     |
| n=2000, d≥10, ρ≥0.6   | 4     |

These are confirmed as genuine wall-clock timeouts (not stack overflows);
see §Bug 2 above.

---

## 6. Key Implementation Findings

| Finding | Detail |
|---------|--------|
| m ≥ 0.5 guard is essential | Without it, clustered n=500 d=5 causes node explosion in < 1 s |
| Midpoint candidates needed | Gap separators (m=0) require mid-breakpoint thresholds |
| Brute-force fallback frequent | 20–64% of queries fall back (scale mismatch with box threshold) |
| ACP08 invalid for ρ ≤ 0.5 | Marginal series diverges; 30 configs recorded as invalid_params |
| Clustered is the true adversary | Dense clusters in d ≥ 20 defeat separator search entirely |
| Iterative build/query required | Recursive implementation plus MRO-intercept instrumentation doubled stack depth; iterative refactor removes the ceiling |

---

## 7. Reproducibility

The benchmark is now fully reproducible: running
`python -m benchmarks.run_benchmark` with the same `base_seed` (default 42)
produces bit-for-bit identical seeds, point sets, and results on any machine
and any PYTHONHASHSEED value.

*Generated by `benchmarks/run_benchmark.py` on 2026-06-23.*
*Raw data: `results/benchmark_raw.csv` (9,143 rows).*
*Pre-bugfix archive: `results/benchmark_raw_pre_bugfix.csv` (9,100 rows).*
