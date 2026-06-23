"""indyk_tree — static ℓ∞ approximate nearest neighbor data structure.

Implements the construction and query algorithms from:

    Piotr Indyk, "On Approximate Nearest Neighbors under ℓ∞ Norm,"
    Journal of Computer and System Sciences 63(4), pp. 627-638, 2001.

Public API
----------
IndykTree
    Build once, query many times.

linf_distance
    Standalone ℓ∞ distance helper.

SeparatorNode, BoxNode
    Node types (useful for introspection and testing).
"""

from .geometry import linf_distance
from .nodes import BoxNode, SeparatorNode
from .tree import IndykTree

__all__ = [
    "IndykTree",
    "SeparatorNode",
    "BoxNode",
    "linf_distance",
]

__version__ = "0.1.0"
