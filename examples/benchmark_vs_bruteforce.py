"""Benchmark: IndykTree query time vs. brute-force ℓ∞ scan.

Measures wall-clock time for Q queries at various (n, d) scales and
prints a comparison table.
"""

import time

import numpy as np

from indyk_tree import IndykTree

Q = 200  # queries per configuration
CONFIGS = [
    (200, 5),
    (500, 10),
    (1000, 15),
    (2000, 20),
]
RHO = 1.0

rng = np.random.default_rng(0)

print(f"{'n':>6}  {'d':>4}  {'tree_ms':>10}  {'brute_ms':>10}  {'speedup':>8}")
print("-" * 50)

for n, d in CONFIGS:
    points = rng.uniform(0, 100, size=(n, d))
    queries = rng.uniform(0, 100, size=(Q, d))

    tree = IndykTree(rho=RHO)
    tree.build(points)

    # Warm up.
    _ = tree.query(queries[0])

    t0 = time.perf_counter()
    for q in queries:
        tree.query(q)
    tree_ms = (time.perf_counter() - t0) * 1000 / Q

    t0 = time.perf_counter()
    for q in queries:
        np.argmin(np.max(np.abs(points - q), axis=1))
    brute_ms = (time.perf_counter() - t0) * 1000 / Q

    speedup = brute_ms / max(tree_ms, 1e-9)
    print(f"{n:>6}  {d:>4}  {tree_ms:>10.3f}  {brute_ms:>10.3f}  {speedup:>8.2f}x")
