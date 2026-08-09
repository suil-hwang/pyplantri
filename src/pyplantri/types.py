# src/pyplantri/types.py
from __future__ import annotations

# Half-edge: (vertex_index, slot_position_in_embedding)
HalfEdge = tuple[int, int]

# Vertex-indexed exterior-view CW cyclic adjacency (0-based)
Embedding = tuple[tuple[int, ...], ...]
