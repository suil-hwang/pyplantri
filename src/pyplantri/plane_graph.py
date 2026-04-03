# src/pyplantri/plane_graph.py
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .types import Embedding

# Internal normalization input type.
EmbeddingInput = dict[int, tuple[int, ...]] | Embedding | list[tuple[int, ...]]


class FrozenEdgeMultiplicity(Mapping[tuple[int, int], int]):
    """Immutable mapping wrapper for edge multiplicities."""

    __slots__ = ("_data", "_items")

    def __init__(
        self,
        edge_multiplicity: (
            Mapping[tuple[int, int], int]
            | list[tuple[tuple[int, int], int]]
            | tuple[tuple[tuple[int, int], int], ...]
        ),
    ) -> None:
        if isinstance(edge_multiplicity, FrozenEdgeMultiplicity):
            self._data = edge_multiplicity._data
            self._items = edge_multiplicity._items
            return

        # Fast path for plain dict from trusted internal code:
        # skip per-element isinstance checks and int() conversions.
        if type(edge_multiplicity) is dict:
            ordered_items = tuple(sorted(edge_multiplicity.items()))
            self._items = ordered_items
            self._data = dict(ordered_items)
            return

        items_iter: Iterator[tuple[tuple[int, int], int]]
        if isinstance(edge_multiplicity, Mapping):
            items_iter = iter(edge_multiplicity.items())
        else:
            items_iter = iter(edge_multiplicity)

        normalized: dict[tuple[int, int], int] = {}
        for raw_edge, raw_multiplicity in items_iter:
            if not isinstance(raw_edge, tuple) or len(raw_edge) != 2:
                raise TypeError(
                    "edge_multiplicity keys must be 2-tuples; "
                    f"got {raw_edge!r}."
                )
            raw_u, raw_v = raw_edge
            if isinstance(raw_u, bool) or isinstance(raw_v, bool):
                raise TypeError(
                    "edge_multiplicity keys must be integer vertex indices; "
                    f"got {raw_edge!r}."
                )
            if isinstance(raw_multiplicity, bool):
                raise TypeError(
                    "edge_multiplicity values must be integers; "
                    f"got {raw_multiplicity!r} on edge {raw_edge!r}."
                )
            u = int(raw_u)
            v = int(raw_v)
            multiplicity = int(raw_multiplicity)
            edge = (u, v)
            if edge in normalized:
                raise ValueError(f"Duplicate edge key encountered: {edge}")
            normalized[edge] = multiplicity

        ordered_items = tuple(sorted(normalized.items()))
        self._items = ordered_items
        self._data = dict(ordered_items)

    def __getitem__(self, edge: tuple[int, int]) -> int:
        return self._data[edge]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for edge, _ in self._items:
            yield edge

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenEdgeMultiplicity):
            return self._items == other._items
        if isinstance(other, Mapping):
            return self._data == dict(other.items())
        return False

    def __hash__(self) -> int:
        return hash(self._items)

    def __reduce__(self) -> tuple[Any, tuple[tuple[tuple[int, int], int], ...]]:
        return (self.__class__, (self._items,))

    def __repr__(self) -> str:
        return f"FrozenEdgeMultiplicity({dict(self._items)!r})"

    def to_dict(self) -> dict[tuple[int, int], int]:
        return dict(self._items)


@dataclass(frozen=True, slots=True)
class PlaneGraph:
    """Immutable plane graph with fixed clockwise embedding.

    Dual graph (Q*):
        - 4-regular plane multigraph.
        - Allows double edges, but no loops.
        - dual_num_vertices = n and dual_faces = n + 2.

    Primal graph (Q):
        - Simple quadrangulation.
        - primal_num_vertices = n + 2 and primal_faces = n.
        - dual_vertex_to_primal_face[i] maps dual vertex i to its primal face.
        - primal_vertex_to_dual_face[j] maps primal vertex j to its dual face.
    """

    dual_num_vertices: int
    # Canonical undirected support-edge pairs. Parallel copies are encoded only
    # in dual_edge_multiplicity, so this field has size s + d rather than |E*|.
    dual_support_edges: tuple[tuple[int, int], ...]
    dual_edge_multiplicity: Mapping[tuple[int, int], int]
    dual_embedding: Embedding  # CW cyclic order at each vertex.
    dual_faces: tuple[tuple[int, ...], ...]

    primal_num_vertices: int
    primal_embedding: Embedding
    primal_faces: tuple[tuple[int, ...], ...]
    dual_vertex_to_primal_face: tuple[int, ...] = tuple()
    primal_vertex_to_dual_face: tuple[int, ...] = tuple()

    graph_id: int = 0
    _double_edges_cache: frozenset[tuple[int, int]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @staticmethod
    def _coerce_support_edges(
        edges: Iterable[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        if isinstance(edges, tuple) and all(
            isinstance(edge, tuple)
            and len(edge) == 2
            and isinstance(edge[0], int)
            and isinstance(edge[1], int)
            for edge in edges
        ):
            return edges
        return tuple((int(u), int(v)) for u, v in edges)

    @staticmethod
    def _coerce_faces(
        faces: Iterable[Iterable[int]],
    ) -> tuple[tuple[int, ...], ...]:
        if isinstance(faces, tuple) and all(
            isinstance(face, tuple)
            and all(isinstance(v, int) for v in face)
            for face in faces
        ):
            return tuple(tuple(v for v in face) for face in faces)
        return tuple(tuple(int(v) for v in face) for face in faces)

    @staticmethod
    def _coerce_index_tuple(indices: Iterable[int]) -> tuple[int, ...]:
        if isinstance(indices, tuple) and all(
            isinstance(idx, int) for idx in indices
        ):
            return indices
        return tuple(int(idx) for idx in indices)

    def __post_init__(self) -> None:
        """Normalize mutable inputs to immutable internal representations."""
        _set = object.__setattr__

        _set(self, "dual_num_vertices", int(self.dual_num_vertices))
        _set(self, "primal_num_vertices", int(self.primal_num_vertices))
        _set(self, "graph_id", int(self.graph_id))

        _set(
            self,
            "dual_support_edges",
            self._coerce_support_edges(self.dual_support_edges),
        )

        if not isinstance(self.dual_edge_multiplicity, FrozenEdgeMultiplicity):
            _set(self, "dual_edge_multiplicity", FrozenEdgeMultiplicity(self.dual_edge_multiplicity))

        _set(
            self,
            "dual_embedding",
            self._normalize_embedding(
                self.dual_embedding,
                expected_size=self.dual_num_vertices,
            ),
        )
        _set(
            self,
            "primal_embedding",
            self._normalize_embedding(
                self.primal_embedding,
                expected_size=self.primal_num_vertices,
            ),
        )

        _set(self, "dual_faces", self._coerce_faces(self.dual_faces))
        _set(self, "primal_faces", self._coerce_faces(self.primal_faces))
        _set(
            self,
            "dual_vertex_to_primal_face",
            self._coerce_index_tuple(self.dual_vertex_to_primal_face),
        )
        _set(
            self,
            "primal_vertex_to_dual_face",
            self._coerce_index_tuple(self.primal_vertex_to_dual_face),
        )

    @staticmethod
    def _normalize_embedding(
        embedding: EmbeddingInput,
        *,
        expected_size: int = 0,
    ) -> Embedding:
        """Convert sparse/dict embedding into dense 0..n-1 tuple-of-tuples."""
        if isinstance(embedding, dict):
            size = max(expected_size, 0)
            if embedding:
                max_index = max(int(v) for v in embedding.keys()) + 1
                size = max(size, max_index)
            dense: list[tuple[int, ...]] = [tuple() for _ in range(size)]
            for vertex, neighbors in embedding.items():
                idx = int(vertex)
                if idx < 0:
                    continue
                if idx >= len(dense):
                    dense.extend(tuple() for _ in range(idx + 1 - len(dense)))
                dense[idx] = tuple(int(u) for u in neighbors)
            return tuple(dense)

        # Fast path: already-normalized tuple from a prior _normalize_embedding call.
        if isinstance(embedding, tuple) and len(embedding) >= expected_size and all(
            isinstance(neighbors, tuple)
            and all(isinstance(u, int) for u in neighbors)
            for neighbors in embedding
        ):
            return embedding

        dense_embedding: Embedding = tuple(
            tuple(int(u) for u in neighbors) for neighbors in embedding
        )
        if expected_size > len(dense_embedding):
            dense_embedding = dense_embedding + tuple(
                tuple() for _ in range(expected_size - len(dense_embedding))
            )
        return dense_embedding

    @staticmethod
    def _neighbors_of(
        embedding: Embedding,
        vertex: int,
    ) -> tuple[int, ...]:
        """Get neighbors for vertex from dense tuple embedding."""
        if 0 <= vertex < len(embedding):
            return embedding[vertex]
        return tuple()

    @staticmethod
    def _scan_embedding(
        embedding: Embedding,
        *,
        vertex_count: int,
        errors: list[str],
        vertex_label: str,
        loop_label: str,
        embedding_name: str,
        expected_degree: int | None = None,
        check_neighbor_bounds: bool = False,
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
        directed_counts: dict[tuple[int, int], int] = {}
        undirected_half_edge_counts: dict[tuple[int, int], int] = {}

        for v in range(vertex_count):
            if v >= len(embedding):
                errors.append(f"{vertex_label} {v} missing from embedding")
                continue

            neighbors = embedding[v]
            if expected_degree is not None and len(neighbors) != expected_degree:
                errors.append(
                    f"{vertex_label} {v} has degree {len(neighbors)}, expected {expected_degree}"
                )
            if v in neighbors:
                errors.append(f"{loop_label} at vertex {v}")

            for u in neighbors:
                if check_neighbor_bounds and (u < 0 or u >= vertex_count):
                    errors.append(
                        f"{embedding_name} contains out-of-range neighbor {u} at vertex {v}"
                    )
                    continue
                directed = (v, u)
                directed_counts[directed] = directed_counts.get(directed, 0) + 1
                edge = (v, u) if v <= u else (u, v)
                undirected_half_edge_counts[edge] = (
                    undirected_half_edge_counts.get(edge, 0) + 1
                )

        return directed_counts, undirected_half_edge_counts

    @staticmethod
    def _validate_bijection(
        mapping: tuple[int, ...],
        *,
        errors: list[str],
        mapping_name: str,
        expected_size: int,
        source_label: str,
        target_count: int,
        duplicate_target_label: str,
        onto_label: str,
    ) -> None:
        if len(mapping) != expected_size:
            errors.append(
                f"{mapping_name} length mismatch: {len(mapping)} != {expected_size}"
            )
            return

        mapped_targets: set[int] = set()
        for source_idx, target_idx in enumerate(mapping):
            if target_idx < 0 or target_idx >= target_count:
                errors.append(
                    f"{mapping_name}[{source_idx}] out of range: {target_idx}"
                )
            mapped_targets.add(target_idx)

        if len(mapped_targets) != len(mapping):
            errors.append(
                f"{mapping_name} has duplicate {duplicate_target_label} targets"
            )

        if mapped_targets != set(range(target_count)):
            errors.append(f"{mapping_name} is not a bijection onto {onto_label}")

    def _validate_dual_support_edges(self, errors: list[str]) -> None:
        if len(set(self.dual_support_edges)) != len(self.dual_support_edges):
            errors.append("dual_support_edges field contains duplicates")

        for u, v in self.dual_support_edges:
            if u < 0 or v < 0 or u >= self.dual_num_vertices or v >= self.dual_num_vertices:
                errors.append(
                    "dual_support_edges out of range: "
                    f"({u}, {v}) for n={self.dual_num_vertices}"
                )
            if u > v:
                errors.append(
                    f"dual_support_edges not canonical: ({u}, {v})"
                )

        expected_edges = tuple(sorted(self.dual_edge_multiplicity.keys()))
        if self.dual_support_edges != expected_edges:
            errors.append(
                "dual_support_edges field mismatch: "
                f"got={self.dual_support_edges!r}, expected={expected_edges!r}"
            )

    def _validate_dual_faces(self, errors: list[str]) -> None:
        expected_faces = self.dual_num_vertices + 2
        if self.dual_num_faces != expected_faces:
            errors.append(
                f"dual_faces count mismatch: {self.dual_num_faces} != {expected_faces}"
            )

        for face_idx, face in enumerate(self.dual_faces):
            if len(face) < 2:
                errors.append(f"Dual face {face_idx} has size {len(face)}, expected >= 2")
            if len(face) > 2 and len(set(face)) != len(face):
                errors.append(f"Dual face {face_idx} repeats vertices: {face}")
            for vertex in face:
                if vertex < 0 or vertex >= self.dual_num_vertices:
                    errors.append(
                        f"Dual face {face_idx} out-of-range vertex: {vertex}"
                    )

    def _validate_dual_edge_multiplicity(
        self,
        directed_counts: dict[tuple[int, int], int],
        undirected_half_edge_counts: dict[tuple[int, int], int],
        errors: list[str],
    ) -> int:
        for (u, v), multiplicity in self.dual_edge_multiplicity.items():
            if u < 0 or v < 0 or u >= self.dual_num_vertices or v >= self.dual_num_vertices:
                errors.append(
                    "dual_edge_multiplicity out of range: "
                    f"({u}, {v}) for n={self.dual_num_vertices}"
                )
            if u > v:
                errors.append(f"dual_edge_multiplicity not canonical: ({u}, {v})")
            if multiplicity not in (1, 2):
                errors.append(
                    f"Edge ({u}, {v}) multiplicity mismatch: {multiplicity} != 1|2"
                )
            if u == v:
                continue

            count_uv = directed_counts.get((u, v), 0)
            count_vu = directed_counts.get((v, u), 0)
            if count_uv != multiplicity or count_vu != multiplicity:
                errors.append(
                    f"Edge ({u}, {v}) embedding/multiplicity mismatch: "
                    f"u->v={count_uv}, v->u={count_vu}, m={multiplicity}"
                )

        for edge, half_edge_count in undirected_half_edge_counts.items():
            u, v = edge
            if u == v:
                continue
            edge_multiplicity = self.dual_edge_multiplicity.get(edge)
            if edge_multiplicity is None:
                errors.append(
                    f"Edge {edge} missing from dual_edge_multiplicity"
                )
                continue
            if half_edge_count != 2 * edge_multiplicity:
                errors.append(
                    f"Edge {edge} half-edge mismatch: "
                    f"{half_edge_count} != {2 * edge_multiplicity}"
                )

        return sum(self.dual_edge_multiplicity.values())

    def _validate_primal_faces(self, errors: list[str]) -> None:
        expected_primal_faces = self.dual_num_vertices
        if len(self.primal_faces) != expected_primal_faces:
            errors.append(
                "primal_faces count mismatch: "
                f"{len(self.primal_faces)} != {expected_primal_faces}"
            )

        for face_idx, face in enumerate(self.primal_faces):
            if len(face) != 4:
                errors.append(
                    f"Primal face {face_idx} has size {len(face)}, expected 4"
                )
            if len(set(face)) != len(face):
                errors.append(f"Primal face {face_idx} repeats vertices: {face}")
            for vertex in face:
                if vertex < 0 or vertex >= self.primal_num_vertices:
                    errors.append(
                        f"Primal face {face_idx} out-of-range vertex: {vertex}"
                    )

    @staticmethod
    def _validate_primal_simple_edges(
        directed_counts: dict[tuple[int, int], int],
        undirected_half_edge_counts: dict[tuple[int, int], int],
        errors: list[str],
    ) -> int:
        primal_edge_count = 0
        for edge, half_edge_count in undirected_half_edge_counts.items():
            u, v = edge
            if u == v:
                continue
            count_uv = directed_counts.get((u, v), 0)
            count_vu = directed_counts.get((v, u), 0)
            if count_uv != 1 or count_vu != 1:
                errors.append(
                    f"Primal edge {edge} not simple: u->v={count_uv}, v->u={count_vu}"
                )
            if half_edge_count != 2:
                errors.append(
                    f"Primal edge {edge} half-edge mismatch: {half_edge_count} != 2"
                )
            primal_edge_count += half_edge_count // 2

        return primal_edge_count

    def _has_primal_data(self) -> bool:
        return (
            self.primal_num_vertices > 0
            or bool(self.primal_embedding)
            or bool(self.primal_faces)
            or bool(self.dual_vertex_to_primal_face)
            or bool(self.primal_vertex_to_dual_face)
        )

    def _validate_dual_contract(self, errors: list[str]) -> None:
        if len(self.dual_embedding) != self.dual_num_vertices:
            errors.append(
                f"dual_embedding size mismatch: {len(self.dual_embedding)} != {self.dual_num_vertices}"
            )

        self._validate_dual_support_edges(errors)
        self._validate_dual_faces(errors)

        directed_counts, undirected_half_edge_counts = self._scan_embedding(
            self.dual_embedding,
            vertex_count=self.dual_num_vertices,
            errors=errors,
            vertex_label="Vertex",
            loop_label="Self-loop",
            embedding_name="Dual embedding",
            expected_degree=4,
        )
        edge_count = self._validate_dual_edge_multiplicity(
            directed_counts,
            undirected_half_edge_counts,
            errors,
        )

        euler_lhs = self.dual_num_vertices - edge_count + self.dual_num_faces
        if euler_lhs != 2:
            errors.append(
                f"dual Euler mismatch: V-E+F={euler_lhs} != 2 "
                f"(V={self.dual_num_vertices}, E={edge_count}, F={self.dual_num_faces})"
            )

    def _validate_primal_contract(self, errors: list[str]) -> None:
        expected_primal_vertices = self.dual_num_vertices + 2
        if self.primal_num_vertices != expected_primal_vertices:
            errors.append(
                "primal_num_vertices mismatch: "
                f"{self.primal_num_vertices} != {expected_primal_vertices}"
            )

        if len(self.primal_embedding) != self.primal_num_vertices:
            errors.append(
                "primal_embedding size mismatch: "
                f"{len(self.primal_embedding)} != {self.primal_num_vertices}"
            )

        self._validate_primal_faces(errors)

        primal_directed_counts, primal_undirected_half_edge_counts = self._scan_embedding(
            self.primal_embedding,
            vertex_count=self.primal_num_vertices,
            errors=errors,
            vertex_label="Primal vertex",
            loop_label="Primal self-loop",
            embedding_name="Primal embedding",
            check_neighbor_bounds=True,
        )
        primal_edge_count = self._validate_primal_simple_edges(
            primal_directed_counts,
            primal_undirected_half_edge_counts,
            errors,
        )

        primal_euler_lhs = (
            self.primal_num_vertices - primal_edge_count + len(self.primal_faces)
        )
        if primal_euler_lhs != 2:
            errors.append(
                f"primal Euler mismatch: V-E+F={primal_euler_lhs} != 2 "
                f"(V={self.primal_num_vertices}, E={primal_edge_count}, "
                f"F={len(self.primal_faces)})"
            )

        self._validate_bijection(
            self.dual_vertex_to_primal_face,
            errors=errors,
            mapping_name="dual_vertex_to_primal_face",
            expected_size=self.dual_num_vertices,
            source_label="dual vertex",
            target_count=len(self.primal_faces),
            duplicate_target_label="primal face",
            onto_label="primal_faces",
        )
        self._validate_bijection(
            self.primal_vertex_to_dual_face,
            errors=errors,
            mapping_name="primal_vertex_to_dual_face",
            expected_size=self.primal_num_vertices,
            source_label="primal vertex",
            target_count=len(self.dual_faces),
            duplicate_target_label="dual face",
            onto_label="dual faces",
        )

    @property
    def dual_num_faces(self) -> int:
        """Face count (should be n + 2 by Euler's formula)."""
        return len(self.dual_faces)

    @property
    def double_edges(self) -> frozenset[tuple[int, int]]:
        """Parallel-edge pairs (digons) in the dual embedding."""
        cache = getattr(self, "_double_edges_cache", None)
        if cache is None:
            cache = frozenset(e for e, m in self.dual_edge_multiplicity.items() if m == 2)
            object.__setattr__(
                self,
                "_double_edges_cache",
                cache,
            )
        return cache

    @property
    def is_4_regular(self) -> bool:
        """Whether the dual graph is quartic, i.e. every vertex has degree 4."""
        return all(len(neighbors) == 4 for neighbors in self.dual_embedding)

    @property
    def is_loop_free(self) -> bool:
        """Whether graph has no self-loops."""
        return all(v not in neighbors for v, neighbors in enumerate(self.dual_embedding))

    def neighbors_cw(self, vertex: int) -> tuple[int, ...]:
        """CW-ordered neighbors of a vertex."""
        return self._neighbors_of(self.dual_embedding, vertex)

    def neighbors_ccw(self, vertex: int) -> tuple[int, ...]:
        """CCW-ordered neighbors of a vertex."""
        return tuple(reversed(self.neighbors_cw(vertex)))

    def validate(self) -> tuple[bool, list[str]]:
        """Validates graph invariants."""
        errors: list[str] = []
        self._validate_dual_contract(errors)
        if self._has_primal_data():
            self._validate_primal_contract(errors)

        return len(errors) == 0, errors

    def to_dict(self) -> dict:
        """Converts to dictionary for JSON serialization."""
        return {
            "dual_num_vertices": self.dual_num_vertices,
            "dual_support_edges": list(self.dual_support_edges),
            "dual_edge_multiplicity": {
                f"{u},{v}": m for (u, v), m in self.dual_edge_multiplicity.items()
            },
            "dual_embedding": {
                str(v): list(neighbors)
                for v, neighbors in enumerate(self.dual_embedding)
            },
            "dual_faces": [list(f) for f in self.dual_faces],
            "primal_num_vertices": self.primal_num_vertices,
            "primal_embedding": {
                str(v): list(neighbors)
                for v, neighbors in enumerate(self.primal_embedding)
            },
            "primal_faces": [list(f) for f in self.primal_faces],
            "dual_vertex_to_primal_face": list(self.dual_vertex_to_primal_face),
            "primal_vertex_to_dual_face": list(self.primal_vertex_to_dual_face),
            "graph_id": self.graph_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlaneGraph:
        """Creates PlaneGraph from the canonical dual_/primal_-prefixed dictionary."""
        required_keys = (
            "dual_num_vertices",
            "dual_support_edges",
            "dual_edge_multiplicity",
            "dual_embedding",
            "dual_faces",
        )
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise KeyError(f"PlaneGraph.from_dict missing keys: {missing}")

        dual_num_vertices = int(data["dual_num_vertices"])
        raw_support_edges = data["dual_support_edges"]
        raw_edge_mult = data["dual_edge_multiplicity"]
        raw_embedding = data["dual_embedding"]
        raw_faces = data["dual_faces"]
        primal_num_vertices = int(data.get("primal_num_vertices", 0))
        primal_embedding_payload = data.get("primal_embedding", {})
        parsed_edge_multiplicity: dict[tuple[int, int], int] = {
            (int(parts[0]), int(parts[1])): int(v)
            for k, v in raw_edge_mult.items()
            for parts in [k.split(",")]
        }
        return cls(
            dual_num_vertices=dual_num_vertices,
            dual_support_edges=cls._coerce_support_edges(raw_support_edges),
            dual_edge_multiplicity=parsed_edge_multiplicity,
            dual_embedding=cls._normalize_embedding(
                raw_embedding,
                expected_size=dual_num_vertices,
            ),
            dual_faces=cls._coerce_faces(raw_faces),
            primal_num_vertices=primal_num_vertices,
            primal_embedding=cls._normalize_embedding(
                primal_embedding_payload,
                expected_size=primal_num_vertices,
            ),
            primal_faces=cls._coerce_faces(data.get("primal_faces", [])),
            dual_vertex_to_primal_face=cls._coerce_index_tuple(
                data.get("dual_vertex_to_primal_face", [])
            ),
            primal_vertex_to_dual_face=cls._coerce_index_tuple(
                data.get("primal_vertex_to_dual_face", [])
            ),
            graph_id=int(data.get("graph_id", 0)),
        )
