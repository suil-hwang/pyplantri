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
            raise ValueError(f"{graph_name} twin_map missing: {missing_half_edge}")
        if extra_half_edges:
            extra_half_edge = min(extra_half_edges)
            raise ValueError(f"{graph_name} twin_map out-of-range source: {extra_half_edge}")

        for half_edge, twin_half_edge in twin_map.items():
            if twin_half_edge not in expected_half_edges:
                raise ValueError(
                    f"{graph_name} twin_map out-of-range target: "
                    f"{half_edge} -> {twin_half_edge}"
                )
            if half_edge == twin_half_edge:
                raise ValueError(f"{graph_name} twin_map self-twin: {half_edge}")
            if twin_map.get(twin_half_edge) != half_edge:
                raise ValueError(
                    f"{graph_name} twin_map not involutive: "
                    f"{half_edge} -> {twin_half_edge}"
                )

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
        GraphConverter.validate_twin_map(
            embedding,
            twin_map,
            graph_name=graph_name,
        )

        visited: set[HalfEdge] = set()
        face_cycles: list[tuple[HalfEdge, ...]] = []

        if not embedding:
            return tuple()

        max_iterations = max(
            1,
            sum(len(neighbors) for neighbors in embedding.values()),
        )

        for vertex in sorted(embedding):
            degree = len(embedding[vertex])
            for slot_idx in range(degree):
                start_half_edge = (vertex, slot_idx)
                if start_half_edge in visited:
                    continue

                face_cycle: list[HalfEdge] = []
                face_cycle_seen: set[HalfEdge] = set()
                curr_v, curr_i = start_half_edge
                iterations = 0

                while True:
                    if (curr_v, curr_i) in face_cycle_seen:
                        if (curr_v, curr_i) != start_half_edge:
                            raise ValueError(
                                f"{graph_name} face traversal repeated non-start "
                                f"half-edge: {(curr_v, curr_i)}"
                            )
                        break
                    if (curr_v, curr_i) in visited:
                        raise ValueError(
                            f"{graph_name} face traversal crossed visited "
                            f"half-edge before closure: {(curr_v, curr_i)}"
                        )

                    iterations += 1
                    if iterations > max_iterations:
                        raise RuntimeError(
                            f"{graph_name} face traversal overflow: {max_iterations}"
                        )

                    half_edge = (curr_v, curr_i)
                    face_cycle_seen.add(half_edge)
                    visited.add(half_edge)
                    face_cycle.append(half_edge)

                    twin_v, twin_i = twin_map[half_edge]
                    twin_neighbors = embedding.get(twin_v)
                    if twin_neighbors is None:
                        raise ValueError(
                            f"{graph_name} twin_map references missing vertex: {twin_v}"
                        )
                    if len(twin_neighbors) == 0:
                        raise ValueError(
                            f"{graph_name} twin_map references empty vertex: {twin_v}"
                        )
                    curr_v = twin_v
                    curr_i = (twin_i - 1) % len(twin_neighbors)

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
