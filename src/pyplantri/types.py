# src/pyplantri/types.py
from __future__ import annotations

# plantri edge labels: str for -a format, int (byte value) for -T format
EdgeLabel = str | int

# Half-edge: (vertex_index, slot_position_in_embedding)
# This is a position-based model matching plantri's -T double_code output,
# NOT a (source, target) pair as in NetworkX PlanarEmbedding.
HalfEdge = tuple[int, int]

# Edge label -> pair of half-edges sharing that label
EdgeLabelPairs = dict[EdgeLabel, tuple[HalfEdge, HalfEdge]]

# Immutable serialized edge label entries kept on PlaneGraph for reconstruction.
EdgeLabelPairEntries = tuple[tuple[EdgeLabel, HalfEdge, HalfEdge], ...]

# Vertex-indexed exterior-view CW cyclic adjacency (0-based)
Embedding = tuple[tuple[int, ...], ...]
