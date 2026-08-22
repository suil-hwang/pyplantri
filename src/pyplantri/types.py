# src/pyplantri/types.py
from __future__ import annotations

from enum import Enum

# Endpoint-sorted undirected support-edge key.
SupportEdge = tuple[int, int]

# Vertex-index face-boundary sequence.
FaceCycle = tuple[int, ...]

# Vertex-indexed exterior-view CW cyclic adjacency (0-based)
Embedding = tuple[tuple[int, ...], ...]


class QuadrangulationDualClass(str, Enum):
    """Plane-dual families selected by supported simple-quadrangulation modes.

    ``QUARTIC_MULTIGRAPH`` uses primal flags ``-q -c2 -m2`` and denotes
    loop-free, 4-regular, 4-edge-connected plane multigraphs. Parallel edges
    are permitted, and simple members are included.

    ``SIMPLE_QUARTIC`` uses primal flags ``-q -c2`` and restricts the primal
    quadrangulation to minimum degree at least 3; its duals are simple,
    4-regular, 4-edge-connected plane graphs.

    This names a request, not a plantri invocation, so it lives beside the other
    leaf types: the cache reader labels stored graphs with it without importing
    the subprocess wrapper.
    """

    QUARTIC_MULTIGRAPH = "quartic_multigraph"
    SIMPLE_QUARTIC = "simple_quartic"
