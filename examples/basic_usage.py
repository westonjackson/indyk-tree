"""Basic usage example for indyk_tree.

Demonstrates building a tree and querying approximate nearest neighbors
under the ℓ∞ norm.
"""

import numpy as np

from indyk_tree import IndykTree, linf_distance

rng = np.random.default_rng(42)

# Build a dataset: 500 points in R^10.
n, d = 500, 10
points = rng.uniform(0.0, 100.0, size=(n, d))

# Construct the tree with rho=1.0 (moderate approximation).
tree = IndykTree(rho=1.0)
tree.build(points)

# Issue a few queries.
for _ in range(5):
    query = rng.uniform(0.0, 100.0, size=(d,))

    # Approximate nearest neighbor via the tree.
    approx_nn = tree.query(query)

    # True nearest neighbor via brute-force scan (for comparison).
    dists = np.max(np.abs(points - query), axis=1)
    true_nn = points[np.argmin(dists)]
    true_dist = float(np.min(dists))

    approx_dist = linf_distance(approx_nn, query)

    print(
        f"true_dist={true_dist:.3f}  approx_dist={approx_dist:.3f}  "
        f"ratio={approx_dist / max(true_dist, 1e-9):.2f}"
    )
