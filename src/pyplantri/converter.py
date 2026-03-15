# src/pyplantri/converter.py
from __future__ import annotations

from typing import Dict, List, Tuple


class GraphConverter:
    """Utility for converting plantri output to various formats."""

    @staticmethod
    def to_zero_based_embedding(
        adjacency_list: Dict[int, List[int]],
    ) -> Dict[int, Tuple[int, ...]]:
        """Converts 1-based adjacency list to 0-based embedding."""
        embedding: Dict[int, Tuple[int, ...]] = {}
        for vertex, neighbors in adjacency_list.items():
            vertex_idx = vertex - 1
            neighbor_tuple = tuple(u - 1 for u in neighbors)
            embedding[vertex_idx] = neighbor_tuple
        return embedding

    @staticmethod
    def extract_faces_with_twins(
        embedding: Dict[int, Tuple[int, ...]],
        twin_map: Dict[Tuple[int, int], Tuple[int, int]],
    ) -> Tuple[Tuple[int, ...], ...]:
        """Extract faces accurately using position-based half-edge traversal."""
        visited: set = set()
        faces: List[Tuple[int, ...]] = []

        # Calculate max iterations for infinite loop detection
        if not embedding:
            return tuple(faces)

        max_iterations = len(embedding) * max(
            len(neighbors) for neighbors in embedding.values()
        )

        for v in sorted(embedding.keys()):
            deg_v = len(embedding[v])
            for i in range(deg_v):
                if (v, i) in visited:
                    continue

                face: List[int] = []
                curr_v, curr_i = v, i
                iterations = 0

                while (curr_v, curr_i) not in visited:
                    iterations += 1
                    if iterations > max_iterations:
                        raise RuntimeError(
                            f"Face traversal exceeded {max_iterations} iterations. "
                            f"Possible infinite loop or invalid twin_map. "
                            f"Current face: {face}"
                        )

                    visited.add((curr_v, curr_i))
                    face.append(curr_v)

                    # Twin
                    if (curr_v, curr_i) not in twin_map:
                        raise ValueError(
                            f"Half-edge ({curr_v}, {curr_i}) not found in twin_map. "
                            f"Embedding may be invalid or twin_map incomplete. "
                            f"Face so far: {face}"
                        )
                    twin_v, twin_i = twin_map[(curr_v, curr_i)]

                    # Predecessor in CW order
                    deg = len(embedding[twin_v])
                    curr_v = twin_v
                    curr_i = (twin_i - 1) % deg

                if len(face) >= 2:
                    faces.append(tuple(face))

        return tuple(faces)

    @staticmethod
    def is_4_regular(adjacency_list: Dict[int, List[int]]) -> bool:
        """Checks if graph is 4-regular."""
        if not adjacency_list:
            return False
        return all(len(neighbors) == 4 for neighbors in adjacency_list.values())
