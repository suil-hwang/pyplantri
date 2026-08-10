# src/pyplantri/types.py
from __future__ import annotations

# Endpoint-sorted undirected support-edge key.
SupportEdge = tuple[int, int]

# Vertex-index face-boundary sequence.
FaceCycle = tuple[int, ...]

# Vertex-indexed exterior-view CW cyclic adjacency (0-based)
Embedding = tuple[tuple[int, ...], ...]
