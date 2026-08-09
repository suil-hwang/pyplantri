# src/pyplantri/types.py
from __future__ import annotations

# Half-edge: (vertex_index, slot_position_in_embedding)
HalfEdge = tuple[int, int]

# Endpoint-sorted undirected support-edge key.
SupportEdge = tuple[int, int]

# Vertex-index face-boundary sequence.
FaceCycle = tuple[int, ...]

# Vertex-indexed exterior-view CW cyclic adjacency (0-based)
Embedding = tuple[tuple[int, ...], ...]
