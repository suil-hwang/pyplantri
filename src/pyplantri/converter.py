# src/pyplantri/converter.py
from __future__ import annotations

from .types import HalfEdge


class GraphConverter:
    """Utility for converting plantri output to various formats."""

    @staticmethod
    def _all_half_edges(
        embedding: dict[int, tuple[int, ...]],
    ) -> set[HalfEdge]:
        return {
            (vertex, slot_idx)
            for vertex, neighbors in embedding.items()
            for slot_idx in range(len(neighbors))
        }

    @staticmethod
    def validate_twin_map(
        embedding: dict[int, tuple[int, ...]],
        twin_map: dict[HalfEdge, HalfEdge],
        *,
        graph_name: str = "graph",
    ) -> None:
        """Validate that a twin map is a complete involution on embedding slots."""
        expected_half_edges = GraphConverter._all_half_edges(embedding)
        twin_domain = set(twin_map)

        missing_half_edges = expected_half_edges - twin_domain
        extra_half_edges = twin_domain - expected_half_edges
        if missing_half_edges:
            missing_half_edge = min(missing_half_edges)
            raise ValueError(
                f"{graph_name} twin_map missing: {missing_half_edge}"
            )
        if extra_half_edges:
            extra_half_edge = min(extra_half_edges)
            raise ValueError(
                f"{graph_name} twin_map out-of-range source: {extra_half_edge}"
            )

        for half_edge, twin_half_edge in twin_map.items():
            if twin_half_edge not in expected_half_edges:
                raise ValueError(
                    f"{graph_name} twin_map out-of-range target: {half_edge} -> {twin_half_edge}"
                )
            if half_edge == twin_half_edge:
                raise ValueError(
                    f"{graph_name} twin_map self-twin: {half_edge}"
                )
            if twin_map.get(twin_half_edge) != half_edge:
                raise ValueError(
                    f"{graph_name} twin_map not involutive: {half_edge} -> {twin_half_edge}"
                )
            vertex, slot = half_edge
            twin_vertex, _ = twin_half_edge
            if embedding[vertex][slot] != twin_vertex:
                raise ValueError(
                    f"{graph_name} twin_map endpoint mismatch: {half_edge} -> {twin_half_edge}"
                )

    @staticmethod
    def to_zero_based_embedding(
        adjacency_list: dict[int, list[int]],
    ) -> dict[int, tuple[int, ...]]:
        """Converts 1-based adjacency list to 0-based embedding."""
        return {
            vertex - 1: tuple(neighbor - 1 for neighbor in neighbors)
            for vertex, neighbors in adjacency_list.items()
        }

    @staticmethod
    def extract_faces_with_twins(
        embedding: dict[int, tuple[int, ...]],
        twin_map: dict[tuple[int, int], tuple[int, int]],
        *,
        graph_name: str = "graph",
    ) -> tuple[tuple[int, ...], ...]:
        """Project face half-edge orbits to vertex walks, losing edge identity."""
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
        """Return the right-face orbits of rotation^-1 composed with twin."""
        GraphConverter.validate_twin_map(
            embedding,
            twin_map,
            graph_name=graph_name,
        )

        visited: set[HalfEdge] = set()
        face_cycles: list[tuple[HalfEdge, ...]] = []

        for vertex in sorted(embedding):
            degree = len(embedding[vertex])
            for slot_idx in range(degree):
                start_half_edge = (vertex, slot_idx)
                if start_half_edge in visited:
                    continue

                face_cycle: list[HalfEdge] = []
                half_edge = start_half_edge

                while True:
                    visited.add(half_edge)
                    face_cycle.append(half_edge)

                    twin_v, twin_i = twin_map[half_edge]
                    twin_neighbors = embedding[twin_v]
                    # For exterior-view CW rotations, the face successor precedes the twin slot.
                    half_edge = (
                        twin_v,
                        (twin_i - 1) % len(twin_neighbors),
                    )
                    if half_edge == start_half_edge:
                        break

                face_cycles.append(tuple(face_cycle))

        return tuple(face_cycles)

    @staticmethod
    def is_4_regular(adjacency_list: dict[int, list[int]]) -> bool:
        """Check whether a nonempty adjacency list is 4-regular."""
        return bool(adjacency_list) and all(
            len(neighbors) == 4
            for neighbors in adjacency_list.values()
        )
