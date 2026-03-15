# src/pyplantri/types.py
from __future__ import annotations

from typing import Dict, Tuple, Union

# ASCII parsing constant
ORD_LOWER_A: int = ord("a")

# plantri edge labels: str for -a format, int (byte value) for -T format
EdgeLabel = Union[str, int]

# Half-edge: (vertex_index, slot_position_in_embedding)
# This is a position-based model matching plantri's -T double_code output,
# NOT a (source, target) pair as in NetworkX PlanarEmbedding.
HalfEdge = Tuple[int, int]

# Edge label -> pair of half-edges sharing that label
EdgeLabelPairs = Dict[EdgeLabel, Tuple[HalfEdge, HalfEdge]]

# Canonical embedding: vertex -> CW-ordered neighbor tuple (0-based)
Embedding = Tuple[Tuple[int, ...], ...]
