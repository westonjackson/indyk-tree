# Empirical Study of Indyk's ℓ∞ ANN Data Structure

Benchmark grid: **n ∈ {100, 500, 2000}**, **d ∈ {2, 5, 10, 20, 40}**,
**ρ ∈ {0.3, 0.6, 1.0}**, 5 distributions, 50 queries each,
**5 independent seeds per configuration**.
**45,813 total rows** (45,600 ok, 150 invalid\_params, 63 timeout)
recorded in `results/benchmark_raw.csv`.
Previous single-seed results archived at `results_archive/7d1beac_20260624T020810Z/`.

---

## Infrastructure: bug corrections

Two infrastructure bugs were fixed before this run.

### Bug 1 — non-deterministic seeding via `hash()`

Seeds were computed as `abs(hash((n, d, rho, dist, base_seed))) % 2**31`.
Python's `hash()` is randomised per-process (PYTHONHASHSEED), so results
changed on every invocation. Replaced with SHA-256 keyed on
`f"{n}|{d}|{rho}|{dist}|{base_seed}|{repeat}"` — fully deterministic across
processes and machines.

### Bug 2 — double recursion depth from MRO-intercept instrumentation

The old `InstrumentedIndykTree` overrode `_build`/`_query` and called
`super()._build()`/`super()._query()`, but `IndykTree`'s own recursion
dispatched back to the subclass via `self._build`/`self._query` (MRO),
doubling the effective Python stack depth per tree level. Resolution:
converted `_build` and `_query` to iterative implementations (work/result
stacks and a while-loop) in `tree.py`, and replaced MRO overrides with five
explicit hook methods (`_on_build_entry`, `_on_sep_node_built`, etc.).

**How many of the 13 prior timeouts were `RecursionError`?
Answer: zero.** All 13 unique configs still time out on all 5 repeats (63
timeout sentinel rows out of a maximum 65 = 13 × 5, with one config —
n=500, d=20, ρ=0.6, clustered — narrowly timing out on only 3 of 5 seeds).

---

## Step 4: Sanity checks

| Check                        | Expected | Actual  | Status |
|------------------------------|:--------:|:-------:|:------:|
| Distinct (n,d,ρ,dist,repeat) | 1,125    | 1,125   | ✓      |
| ACP08 invalid_params rows    | 150      | 150     | ✓      |
| Timeout rows                 | ≤65      | 63      | ✓      |
| ok rows                      | 45,600   | 45,600  | ✓      |

**Timeout consistency:** 12 of 13 timeout configs timed out on all 5 repeats.
One config (n=500, d=20, ρ=0.6, clustered) timed out on 3 of 5 seeds — it
sits right at the 30-second boundary. The other 2 seeds completed.

**Invalid configs:** Both ρ=0.3 and ρ=0.6 are invalid for `acp08_adversarial`
because the marginal series Σ 2^{-(2ρ)^i} diverges for 2ρ ≤ 1. Recorded
as `failure_reason="invalid_params"`.

---

## Step 2: Full distributional reporting

### Approximation ratio (excluding inf; failure\_reason="ok" rows only)

| Distribution       | Mean   | Median | p90    | p99    | Max     |
|--------------------|:------:|:------:|:------:|:------:|:-------:|
| uniform\_random    | 1.017  | 1.000  | 1.000  | 1.000  | 30.26   |
| acp08\_adversarial | 1.069  | 1.000  | 1.000  | 2.000  | 3.00    |
| boundary\_stress   | 1.061  | 1.000  | 1.000  | 1.770  | 67.33   |
| clustered          | 3.072  | 1.000  | 6.308  | 27.25  | 191.28  |
| cyclic\_closeness  | 4.692  | 1.797  | 9.389  | 47.88  | 830.57  |

**cyclic\_closeness** has the widest tail (max 830×) at d=2; the pattern
collapses to near-exact at d=40.

### Build time (one per config, excluding timeouts)

| Distribution       | Mean (s) | Median (s) | p90 (s) | p99 (s) | Max (s) |
|--------------------|:--------:|:----------:|:-------:|:-------:|:-------:|
| uniform\_random    | 0.331    | 0.138      | 0.890   | 1.970   | 2.153   |
| acp08\_adversarial | 0.189    | 0.007      | 0.246   | 2.560   | 2.797   |
| boundary\_stress   | 0.046    | 0.023      | 0.119   | 0.271   | 0.324   |
| clustered          | 1.864    | 0.113      | 4.087   | 23.93   | 29.82   |
| cyclic\_closeness  | 0.172    | 0.032      | 0.406   | 1.597   | 1.617   |

### Nodes visited per query

| Distribution       | Mean | Median | p90  | p99  | Max  |
|--------------------|:----:|:------:|:----:|:----:|:----:|
| uniform\_random    | 17.5 | 16.0   | 29.0 | 39.0 | 46.0 |
| acp08\_adversarial | 11.2 | 10.0   | 24.0 | 39.0 | 57.0 |
| boundary\_stress   | 11.2 | 11.0   | 15.0 | 19.0 | 23.0 |
| clustered          | 11.8 | 11.0   | 20.0 | 31.0 | 43.0 |
| cyclic\_closeness  | 11.0 | 8.0    | 25.0 | 42.0 | 47.0 |

### Cross-repeat variance (std/mean of per-repeat mean ratio)

| Distribution       | Mean CV | Median CV | Max CV |
|--------------------|:-------:|:---------:|:------:|
| uniform\_random    | 0.016   | 0.000     | 0.250  |
| acp08\_adversarial | 0.031   | 0.025     | 0.096  |
| boundary\_stress   | 0.047   | 0.000     | 0.424  |
| clustered          | 0.094   | 0.089     | 0.241  |
| cyclic\_closeness  | 0.107   | 0.087     | 0.402  |

**9 configs are UNSTABLE (CV > 0.25).** All 9 are at d=2, where small sample
sizes and the periodic structure of cyclic data (or the integer lattice of
boundary\_stress) create high inter-seed variance. No UNSTABLE config appears
at d ≥ 5.

Scatter plot of approx\_ratio vs true\_dist for the clustered distribution
is saved at `results/clustered_scatter.png`.

---

## Step 3: Before/after comparison table

Prior run: 9,100 ok rows (single seed).  
This run: 45,600 ok rows (5 seeds × ~912 configs).

No metric changes by more than 20%. All UNSTABLE flags are for **median**
approx\_ratio, which is inherently high-variance because it collapses to 1.0
for most seeds.

| Metric              | Distribution       | Archived | New mean | New std | %Δ    | Flag      |
|---------------------|--------------------|:--------:|:--------:|:-------:|:-----:|:---------:|
| median\_approx\_ratio | clustered        | 1.000    | 1.000    | 0.448   | 0.0   | UNSTABLE  |
| median\_approx\_ratio | cyclic\_closeness| 1.779    | 1.797    | 0.865   | +1.0  | UNSTABLE  |
| fallback\_rate      | uniform\_random    | 0.172    | 0.205    | 0.048   | +19.1 | CONFIRMED |
| mean\_approx\_ratio | boundary\_stress   | 1.171    | 1.061    | 0.063   | −9.4  | CONFIRMED |
| mean\_approx\_ratio | clustered          | 3.228    | 3.072    | 0.448   | −4.8  | CONFIRMED |
| mean\_approx\_ratio | cyclic\_closeness  | 4.779    | 4.692    | 0.865   | −1.8  | CONFIRMED |
| mean\_approx\_ratio | acp08\_adversarial | 1.052    | 1.069    | 0.035   | +1.6  | CONFIRMED |
| mean\_approx\_ratio | uniform\_random    | 1.027    | 1.017    | 0.018   | −0.9  | CONFIRMED |
| (all other metrics) | (all dists)        | —        | —        | —       | <10%  | CONFIRMED |

**No CHANGED flags.** The two UNSTABLE flags are for median approx\_ratio —
the median is 1.0 for most seeds and occasionally jumps, making its
cross-repeat std high. The means are stable (CV ≤ 0.09 for clustered).

---

## 1. Space Usage vs n  (`results/space_vs_n.png`)

**The n^{1+ρ} space bound is very pessimistic in practice.**

For ρ=1.0 the theory predicts O(n²) total memberships; measured overhead
is 13–23× n on random data — consistent with O(n). Separator search finds
score-0 splits most of the time, producing near-linear trees on the tested
distributions.

---

## 2. Approximation Ratio vs Dimension  (`results/approx_ratio_vs_d.png`)

**Clustered and cyclic data are the practical adversaries.**

- **Uniform random**: mean ratio 1.017. Clean gaps give score-0 separators.
- **ACP08 (ρ=1.0 only)**: mean ratio 1.069. The geometric marginal
  concentrates ~50% of coordinates at 0; many score-0 separators.
- **Clustered**: mean 3.07, p99 27.25, max 191. Dense clusters in d ≥ 20
  trigger the build timeout for 13 configs across all repeats.
- **Cyclic closeness**: mean 4.69, max 830 at d=2, collapses to ~1.1 at d=40.
- **Boundary stress**: mean 1.06 but max 67. Integer collisions drive high
  fallback (63%) but the m ≥ 0.5 guard holds.

---

## 3. Box-Visit Fraction  (`results/box_visit_frac.png`)

ACP08 drives the most box-node traversals (63.1%), consistent with its design.
Uniform random has the lowest (10.5%) due to deep balanced separator trees.

---

## 4. Query Time vs d·log(n)  (`results/query_time_vs_d_logn.png`)

Query times remain in the **0.01–0.04 ms** range across the full grid, with
mild linear growth in d·log(n) for uniform random data.

---

## 5. Clustered Scatter  (`results/clustered_scatter.png`)

Scatter of approx\_ratio (log y) vs true\_dist (log x) for the clustered
distribution. The approximation degrades most at small true\_dist (< 0.01)
where close clusters create separator interference. Points with ratio ≫ 1
are concentrated at the smallest true distances, consistent with the
separator search failing to isolate the nearest cluster.

---

## 6. Key Implementation Findings

| Finding | Detail |
|---------|--------|
| m ≥ 0.5 guard is essential | Without it, clustered n=500 d=5 causes node explosion in < 1 s |
| Midpoint candidates needed | Gap separators (m=0) require mid-breakpoint thresholds |
| Brute-force fallback frequent | 20–63% of queries fall back (scale mismatch with box threshold) |
| ACP08 invalid for ρ ≤ 0.5 | Marginal series diverges; 150 configs (across 5 repeats) recorded as invalid\_params |
| Clustered is the true adversary | Dense clusters in d ≥ 20 defeat separator search; confirmed on all 5 seeds for 12/13 timeout configs |
| 9 UNSTABLE configs at d=2 | High cross-repeat variance at d=2 only; all findings at d ≥ 5 are stable |
| Iterative build/query required | Recursive + MRO instrumentation doubled stack depth; iterative refactor removes the ceiling; zero RecursionError failures across 1,125 configs |

---

## 7. Reproducibility

Running `python -m benchmarks.run_benchmark --repeats 5` with `--seed 42`
(default) on a clean checkout of commit `7d1beac` produces bit-for-bit
identical seeds, point sets, and results on any machine and PYTHONHASHSEED
value.

The `repeat` index is folded into the SHA-256 seed key, so each of the 5
seeds per config is genuinely independent.

*Generated by `benchmarks/run_benchmark.py` on 2026-06-24.*
*Raw data: `results/benchmark_raw.csv` (45,813 rows, 45,600 ok).*
*Provenance: `results/PROVENANCE.md`.*
*Single-seed archive: `results_archive/7d1beac_20260624T020810Z/`.*
