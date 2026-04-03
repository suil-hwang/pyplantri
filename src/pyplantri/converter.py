# src/pyplantri/converter.py
from __future__ import annotations

from .types import HalfEdge


class GraphConverter:
    """Utility for converting plantri output to various formats."""

    @staticmethod
    def to_zero_based_embedding(
        adjacency_list: dict[int, list[int]],
    ) -> dict[int, tuple[int, ...]]:
        """Converts 1-based adjacency list to 0-based embedding."""
        embedding: dict[int, tuple[int, ...]] = {}
        for vertex, neighbors in adjacency_list.items():
            vertex_idx = vertex - 1
            neighbor_tuple = tuple(u - 1 for u in neighbors)
            embedding[vertex_idx] = neighbor_tuple
        return embedding

    @staticmethod
    def extract_faces_with_twins(
        embedding: dict[int, tuple[int, ...]],
        twin_map: dict[tuple[int, int], tuple[int, int]],
        *,
        graph_name: str = "graph",
    ) -> tuple[tuple[int, ...], ...]:
        """Extract face vertex cycles using the shared half-edge walker."""
        face_cycles = GraphConverter.extract_face_half_edge_cycles(
            embedding,
            twin_map,
            graph_name=graph_name,
        )
        return tuple(
            tuple(vertex for vertex, _ in face_cycle)
            for face_cycle in face_cycles
        )

    @staticmethod
    def extract_face_half_edge_cycles(
        embedding: dict[int, tuple[int, ...]],
        twin_map: dict[HalfEdge, HalfEdge],
        *,
        graph_name: str = "graph",
    ) -> tuple[tuple[HalfEdge, ...], ...]:
        """Extract face half-edge cycles from a plane embedding and twin map."""
        visited: set[HalfEdge] = set()
        face_cycles: list[tuple[HalfEdge, ...]] = []

        if not embedding:
            return tuple()

        max_deg = max((len(neighbors) for neighbors in embedding.values()), default=0)
        max_iterations = max(1, len(embedding) * max_deg)

        for vertex in sorted(embedding):
            degree = len(embedding[vertex])
            for slot_idx in range(degree):
                start_half_edge = (vertex, slot_idx)
                if start_half_edge in visited:
                    continue

                face_cycle: list[HalfEdge] = []
                curr_v, curr_i = start_half_edge
                iterations = 0

                while (curr_v, curr_i) not in visited:
                    iterations += 1
                    if iterations > max_iterations:
                        raise RuntimeError(
                            f"{graph_name} face traversal overflow: {max_iterations}"
                        )

                    half_edge = (curr_v, curr_i)
                    visited.add(half_edge)
                    face_cycle.append(half_edge)

                    twin = twin_map.get(half_edge)
                    if twin is None:
                        raise ValueError(f"{graph_name} twin_map missing: {half_edge}")
                    twin_v, twin_i = twin
                    curr_v = twin_v
                    curr_i = (twin_i - 1) % len(embedding[twin_v])

                if len(face_cycle) < 2:
                    raise ValueError(f"{graph_name} face too short: {len(face_cycle)}")
                face_cycles.append(tuple(face_cycle))

        return tuple(face_cycles)

    @staticmethod
    def is_4_regular(adjacency_list: dict[int, list[int]]) -> bool:
        """Check whether every vertex has degree 4(quartic)."""
        if not adjacency_list:
            return False
        return all(len(neighbors) == 4 for neighbors in adjacency_list.values())
