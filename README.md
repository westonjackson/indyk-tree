# indyk-tree

[![CI](https://github.com/westonjackson/indyk-tree/actions/workflows/ci.yml/badge.svg)](https://github.com/westonjackson/indyk-tree/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A clean Python implementation of Indyk's static **ℓ∞ approximate nearest neighbor** (ANN) data structure, as described in:

> Piotr Indyk, "On Approximate Nearest Neighbors under ℓ∞ Norm,"  
> *Journal of Computer and System Sciences* **63**(4), pp. 627–638, 2001.  
> DOI: [10.1006/jcss.2001.1781](https://doi.org/10.1006/jcss.2001.1781)

---

## The Problem

Given a set *P* of *n* points in ℝ^d, the **nearest neighbor** problem asks: for a query point *y*, find the point in *P* closest to *y*. Under the **ℓ∞ norm** (Chebyshev distance, `max_i |y_i - p_i|`), a brute-force scan takes *O(nd)* per query — which is prohibitive when *n* or *d* is large.

The **curse of dimensionality** means that most known exact data structures degrade to near-brute-force in high dimensions. Indyk's result sidesteps this by giving a data structure for **approximate** ℓ∞ nearest neighbors: the returned point is guaranteed to be within an additive *O(log_{1+ρ}(log d))* of the true nearest neighbor distance, with sub-linear query time and space *O(n^{1+ρ} · d)* controlled by a parameter *ρ > 0*.

---

## Quickstart

```python
import numpy as np
from indyk_tree import IndykTree, linf_distance

rng = np.random.default_rng(42)

# Build: 500 points in R^10
points = rng.uniform(0.0, 100.0, size=(500, 10))
tree = IndykTree(rho=1.0)
tree.build(points)

# Query
query = rng.uniform(0.0, 100.0, size=(10,))
approx_nn = tree.query(query)

print(f"Approximate NN distance: {linf_distance(approx_nn, query):.3f}")

# Compare with brute force
true_dist = float(np.min(np.max(np.abs(points - query), axis=1)))
print(f"True NN distance:        {true_dist:.3f}")
```

---

## Install

```bash
pip install -e .
```

(PyPI release coming soon.)

---

## Complexity

| Operation     | Complexity                          | Notes                                           |
|---------------|-------------------------------------|-------------------------------------------------|
| **Space**     | O(n^{1+ρ} · d)                      | Replication of M-slab points at each split      |
| **Build**     | O(n^{1+ρ} · d²)                     | Separator search is O(n·d) per recursive level  |
| **Query**     | O(log n · d / ρ)                    | Single root-to-leaf traversal                   |
| **Approx error** | +O(log_{1+ρ}(log d)) additive    | From Theorem 1 / Lemma 2 of Indyk (2001)        |

Smaller *ρ* = tighter approximation + more space/time. Larger *ρ* = looser approximation + less space/time. Typical values: *ρ ∈ (0, 2]*.

---

## How It Works

### Construction

At each recursive call on a point set *P*:

1. **Separator search** — for each coordinate *i* and threshold *t*, partition *P* into left (L), right (R), and middle (M) slabs. A separator is *good* if it puts a 1/(4d) fraction of points on each side and has separator score ≤ ρ (where the score measures how much replication M causes). If found, create a `SeparatorNode` and recurse on `L ∪ M` and `R ∪ M` — points in M are replicated into *both* children (the source of the n^{1+ρ} space bound).

2. **Box node** — if no good separator exists, Lemma 2 guarantees a dense "box" C of diameter *O(log_{1+ρ}(log d))* containing ≥ |P|/2 points. Create a `BoxNode` and recurse on *P \ C*.

### Query

```
query(y, node):
    if node is None:       return None
    if SeparatorNode(i,t): return query(y, left if y[i] < t else right)
    if BoxNode(center, r): return r if linf(y, center) ≤ 1 else query(y, continuation)
```

---

## Limitations

- **Box-node heuristic**: The full inductive construction of the box set C (Appendix A of Indyk 2001) is not implemented. Instead, the ⌈|P|/2⌉ points closest to the coordinate-wise median are used. This is a practical approximation that may exceed the precise theoretical diameter bound in edge cases. The correctness tests allow a slack of 2.0 and require ≥ 95% of random trials to pass.

- **Pure Python + NumPy**: No C extension. Query throughput is limited; for production high-throughput ANN, consider FAISS or ScaNN.

- **Static**: The data structure does not support insertions or deletions. Rebuild for updated datasets.

- **Theoretical regime**: The approximation guarantees apply most cleanly when *n* and *d* are large. For very small *n* or *d*, brute force is faster.

---

## Development

```bash
pip install -e ".[dev]"
pytest            # run tests
ruff check src/   # lint
mypy src/         # type-check
```

---

## Citation

If you use this software in research, please cite the original paper:

```bibtex
@article{indyk2001approximate,
  author  = {Indyk, Piotr},
  title   = {On Approximate Nearest Neighbors under $\ell_\infty$ Norm},
  journal = {Journal of Computer and System Sciences},
  volume  = {63},
  number  = {4},
  pages   = {627--638},
  year    = {2001},
  doi     = {10.1006/jcss.2001.1781},
}
```

See also [`CITATION.cff`](CITATION.cff).

---

## License

MIT — see [`LICENSE`](LICENSE).
