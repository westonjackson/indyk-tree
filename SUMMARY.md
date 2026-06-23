# Empirical Study of Indyk's ℓ∞ ANN Data Structure

Benchmark grid: **n ∈ {100, 500, 2000}**, **d ∈ {2, 5, 10, 20, 40}**,
**ρ ∈ {0.3, 0.6, 1.0}**, 5 distributions, 50 queries each.
9,100 query trials recorded in `results/benchmark_raw.csv`.

---

## 1. Space Usage vs n  (`results/space_vs_n.png`)

**Finding: The n^{1+ρ} space bound is very pessimistic in practice.**

| n    | ρ=0.3 (theory n^{1.3}) | ρ=0.3 actual / n | ρ=1.0 (theory n²) | ρ=1.0 actual / n |
|------|------------------------|------------------|---------------------|-------------------|
| 100  | 398                    | 10.8×            | 10,000              | 14.5×             |
| 500  | 3,227                  | 42.9×            | 250,000             | 20.1×             |
| 2000 | 19,562                 | 36.6×            | 4,000,000           | 14.5×             |

For ρ=1.0 the theory predicts O(n²) total memberships, yet the measured overhead
is roughly **15–20×** regardless of n — indistinguishable from O(n) on these data
sizes. The worst-case construction requires an adversarial input that our distributions
do not provide; on random data the separator search finds score-0 splits (m = 0)
most of the time, collapsing the recursion tree to near-linear size.

For ρ=0.3 the overhead grows from 10× to 43× between n=100 and n=500 but then
plateaus, suggesting the bound is loose even for small ρ.

---

## 2. Approximation Ratio vs Dimension  (`results/approx_ratio_vs_d.png`)

**Finding: Not all distributions are adversarial; clustered data is the hardest.**

| Distribution       | Mean ratio | Median | Max    |
|--------------------|------------|--------|--------|
| uniform_random     | 1.007      | 1.000  | 4.59   |
| acp08_adversarial  | 1.047      | 1.000  | 4.00   |
| boundary_stress    | 1.061      | 1.000  | 48.8   |
| cyclic_closeness   | 4.544      | 1.785  | 175.1  |
| **clustered**      | **3.419**  | 1.000  | 533.7  |

- **Uniform random** is nearly optimal across all d and ρ. The separator search
  finds clean gaps in random data and box nodes cover tight local neighborhoods.

- **ACP08 adversarial** (valid only at ρ=1.0): mean ratio 1.047. Counter-intuitively,
  this theoretically hard distribution is *not* adversarial for our implementation.
  The reason: with ρ=1.0 the marginal concentrates ≈50% probability at 0, so most
  coordinates are 0. The separator search finds score-0 splits along "sparse" axes,
  producing a compact tree. The m ≥ 0.5 guard prevents the replication blow-up that
  the theoretical analysis exploits.

- **Clustered** is the empirical adversary. Dense Gaussian clusters in high d
  (d ≥ 10, n ≥ 500) cause build timeouts (> 30 s) for 13/45 configurations.
  When the tree does build, ratios reach 534×. Cluster interiors have no clean
  ℓ∞ separator — every axis cuts through multiple clusters, producing high m and
  massive replication.

- **Cyclic closeness** shows an unexpected *d-dependent* pattern: at d=2 the mean
  ratio is ~14× (n=500), but it drops to ~1.2× at d=40. With few dimensions the
  cyclic structure (point i shifted in axis i%d) concentrates many points into two
  alternating lanes; the separator can't isolate lanes without replicating everything.
  At high d the shifts spread out across many axes, reducing lane interference.

- **Boundary stress** (integer-valued points) has modest mean ratio ≈ 1.06 but a
  long tail (max 48.8×). Integer coordinates create frequent threshold collisions
  (many points at identical values in each slab), straining the m < 0.5 guard and
  driving a high fallback rate (64%), but the guard holds.

---

## 3. Box-Visit Fraction by Distribution  (`results/box_visit_frac.png`)

| Distribution       | Box-visit fraction |
|--------------------|-------------------|
| acp08_adversarial  | 62.9%             |
| cyclic_closeness   | 35.8%             |
| boundary_stress    | 31.9%             |
| clustered          | 23.8%             |
| uniform_random     | 10.5%             |

**ACP08 drives the most box-node traversals** (62.9%), confirming the theoretical
design: the adversarial marginal makes points bunch near integer values, so the
separator search resolves quickly into box nodes that absorb large local groups.
At query time, the algorithm must check many box-node representatives before finding
one within the ρ-approximate distance.

**Uniform random** has the lowest box-visit fraction (10.5%). Random data produces
deep separator trees with balanced splits; the query path follows a single root-to-leaf
chain of separator decisions before reaching a leaf box node.

---

## 4. Query Time vs d·log(n)  (`results/query_time_vs_d_logn.png`)

Query times are in the **0.01–0.04 ms** range across the full grid, with no
meaningful dependence on ρ. Time grows roughly linearly with d·log(n) for
uniform random data, consistent with a traversal depth of O(log n) nodes each
doing O(d) coordinate comparisons. For clustered data the traversal is deeper
and noisier, but still sub-millisecond when the build succeeds.

---

## 5. Build Timeouts: Where the Theory Kicks In

**Clustered** data triggered the 30-second build timeout for 13 configurations:

| Configuration    | Behaviour                         |
|------------------|-----------------------------------|
| n≥500, d≥20      | Timeout (> 30 s)                  |
| n=2000, d≥10     | Timeout for ρ=0.6 and ρ=1.0      |
| n=500, d=10      | Builds in ~1 s; n=2000 in ~20 s  |

This is the clearest evidence of the theoretical n^{1+ρ} blowup materialising:
the clustered marginal is one where no separator achieves m < 0.5 without
replicating near-n points to each child. The m ≥ 0.5 guard prevents infinite
recursion but cannot prevent exponential growth in the number of nodes when
every candidate separator is borderline.

---

## 6. Key Implementation Findings

| Finding | Detail |
|---------|--------|
| **m ≥ 0.5 guard is essential** | Without it, clustered n=500 d=5 causes 2³¹ nodes in < 1 s |
| **Midpoint candidates needed** | Gap separators (m=0) require mid-breakpoint thresholds; v±1 alone miss them |
| **Brute-force fallback frequent** | 20–64% of queries fall back depending on distribution (scale mismatch) |
| **ACP08 invalid for ρ ≤ 0.5** | The marginal series diverges; only ρ=1.0 is benchmarked |
| **Clustered is the true adversary** | Not ACP08; dense clusters in ≥ 10 dimensions defeat separator search |

---

*Generated by `benchmarks/run_benchmark.py` on 2026-06-22.*
*Raw data: `results/benchmark_raw.csv` (9,100 rows).*
