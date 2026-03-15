# src/pyplantri/converter.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from .types import ORD_LOWER_A


class GraphConverter:
    """Utility for converting plantri output to various formats."""

    @staticmethod
    def parse_ascii_to_edge_map(
        ascii_line: str,
        is_one_based: bool = False
    ) -> Dict[Tuple[int, int], int]:
        """Converts ASCII output to edge multiplicity map."""
        parts = ascii_line.strip().split()
        if not parts or not parts[0].isdigit():
            return {}

        if len(parts) < 2:
            return {}

        adjacency_str = parts[1]
        vertex_neighbor_lists = adjacency_str.split(",")

        edge_multiplicity: Dict[Tuple[int, int], int] = defaultdict(int)
        index_offset = 0 if is_one_based else 1

        for vertex_idx, neighbors_str in enumerate(vertex_neighbor_lists):
            source_vertex = vertex_idx + (1 if is_one_based else 0)
            for char in neighbors_str:
                target_vertex = ord(char) - ORD_LOWER_A + 1 - index_offset
                # Canonical edge key (u < v) - count one direction only.
                if source_vertex < target_vertex:
                    edge_multiplicity[(source_vertex, target_vertex)] += 1

        return dict(edge_multiplicity)

    @staticmethod
    def parse_ascii_to_adjacency_list(
        ascii_line: str,
        is_one_based: bool = False
    ) -> Dict[int, List[int]]:
        """Converts ASCII output to adjacency list."""
        parts = ascii_line.strip().split()
        if not parts or not parts[0].isdigit():
            return {}

        if len(parts) < 2:
            return {}

        adjacency_str = parts[1]
        vertex_neighbor_lists = adjacency_str.split(",")

        adjacency_list: Dict[int, List[int]] = {}
        index_offset = 0 if is_one_based else 1

        for vertex_idx, neighbors_str in enumerate(vertex_neighbor_lists):
            vertex = vertex_idx + (1 if is_one_based else 0)
            adjacency_list[vertex] = [
                ord(c) - ORD_LOWER_A + 1 - index_offset for c in neighbors_str
            ]

        return adjacency_list

    @staticmethod
    def adjacency_to_edge_multiplicity(
        adjacency_list: Dict[int, List[int]],
        is_one_based: Optional[bool] = None,
        *,
        output_one_based: bool = False,
    ) -> Dict[Tuple[int, int], int]:
        """Convert adjacency list to edge multiplicity map.

        Args:
            adjacency_list: Vertex -> neighbor list adjacency map.
            is_one_based: Whether input vertex indices are 1-based.
                When ``None`` (default), infer indexing from observed vertex
                labels: presence of ``0`` means 0-based, otherwise presence of
                ``1`` means 1-based.
            output_one_based: Whether output edge keys should be 1-based.
                Historical default behavior is 0-based output regardless of
                ``is_one_based``.
        """
        edge_multiplicity: Dict[Tuple[int, int], int] = defaultdict(int)
        if is_one_based is None:
            observed_indices: Set[int] = set(int(v) for v in adjacency_list.keys())
            for neighbors in adjacency_list.values():
                observed_indices.update(int(u) for u in neighbors)

            if not observed_indices:
                is_one_based = False
            elif min(observed_indices) < 0:
                raise ValueError(
                    "adjacency_list contains negative vertex indices, cannot infer index base."
                )
            elif 0 in observed_indices:
                is_one_based = False
            elif 1 in observed_indices:
                is_one_based = True
            else:
                raise ValueError(
                    "Cannot infer index base from adjacency_list. "
                    "Pass is_one_based=True or is_one_based=False explicitly."
                )

        input_offset = 1 if is_one_based else 0
        output_offset = 1 if output_one_based else 0

        for vertex, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                u = min(vertex - input_offset, neighbor - input_offset)
                v = max(vertex - input_offset, neighbor - input_offset)
                if output_offset:
                    u += output_offset
                    v += output_offset
                edge_multiplicity[(u, v)] += 1

        normalized_edge_multiplicity: Dict[Tuple[int, int], int] = {}
        for edge, half_edge_count in edge_multiplicity.items():
            if half_edge_count % 2 != 0:
                raise ValueError(
                    f"Asymmetric adjacency for edge {edge}: "
                    f"half-edge count {half_edge_count}"
                )
            normalized_edge_multiplicity[edge] = half_edge_count // 2

        return normalized_edge_multiplicity

    @staticmethod
    def to_adjacency_matrix(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ) -> np.ndarray:
        """Converts edge multiplicity map to dense adjacency matrix."""
        if not edge_multiplicity:
            return np.zeros((vertex_count, vertex_count), dtype=np.int32)

        # Vectorized indexing for performance.
        edges = list(edge_multiplicity.keys())
        sources = np.array([e[0] for e in edges], dtype=np.intp)
        targets = np.array([e[1] for e in edges], dtype=np.intp)
        multiplicities = np.array(list(edge_multiplicity.values()), dtype=np.int32)

        adjacency_matrix = np.zeros((vertex_count, vertex_count), dtype=np.int32)
        # Assign symmetric matrix at once using fancy indexing.
        adjacency_matrix[sources, targets] = multiplicities
        adjacency_matrix[targets, sources] = multiplicities
        return adjacency_matrix

    @staticmethod
    def to_sparse_adjacency_matrix(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ) -> csr_matrix:
        """Converts edge multiplicity map to CSR sparse matrix."""
        if not edge_multiplicity:
            return csr_matrix((vertex_count, vertex_count), dtype=np.int32)

        # Prepare COO format data (add both directions for symmetric matrix).
        edges = list(edge_multiplicity.keys())
        multiplicities = list(edge_multiplicity.values())

        row = [e[0] for e in edges] + [e[1] for e in edges]
        col = [e[1] for e in edges] + [e[0] for e in edges]
        data = multiplicities + multiplicities

        return csr_matrix(
            (data, (row, col)),
            shape=(vertex_count, vertex_count),
            dtype=np.int32
        )

    @staticmethod
    def get_graph_stats(
        edge_multiplicity: Dict[Tuple[int, int], int],
        vertex_count: int
    ) -> dict:
        """Computes graph statistics from edge multiplicity map."""
        degrees = [0] * vertex_count
        single_edge_count = 0
        double_edge_count = 0
        total_edge_count = 0

        for (source, target), multiplicity in edge_multiplicity.items():
            degrees[source] += multiplicity
            degrees[target] += multiplicity
            total_edge_count += multiplicity
            if multiplicity == 1:
                single_edge_count += 1
            elif multiplicity == 2:
                double_edge_count += 1

        is_regular = len(set(degrees)) == 1 if degrees else False
        regularity = degrees[0] if is_regular and degrees else None

        return {
            "vertex_count": vertex_count,
            "edge_count": total_edge_count,
            "single_edge_count": single_edge_count,
            "double_edge_count": double_edge_count,
            "degrees": degrees,
            "is_regular": is_regular,
            "regularity": regularity,
        }

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

    @staticmethod
    def is_loop_free(adjacency_list: Dict[int, List[int]]) -> bool:
        """Checks if graph has no self-loops."""
        return all(v not in neighbors for v, neighbors in adjacency_list.items())
