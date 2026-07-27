# src/pyplantri/plane_graph.py
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from .converter import GraphConverter
from .types import EdgeLabel, EdgeLabelPairEntries, Embedding, HalfEdge

LabelSignature = tuple[str, ...]
_EMPTY_DOUBLE_EDGES: frozenset[tuple[int, int]] = frozenset()
_PLANE_GRAPH_SCALAR_FIELDS = (
    "dual_num_vertices",
    "primal_num_vertices",
    "graph_id",
)
_PLANE_GRAPH_STATE_FIELDS = (
    "dual_num_vertices",
    "dual_support_edges",
    "dual_edge_multiplicity",
    "dual_embedding",
    "dual_faces",
    "primal_num_vertices",
    "primal_embedding",
    "primal_faces",
    "dual_vertex_to_primal_face",
    "primal_vertex_to_dual_face",
    "dual_edge_label_pairs",
    "primal_edge_label_pairs",
    "graph_id",
)


def _edge_label_token(edge_label: EdgeLabel) -> str:
    """Encode an edge label as a stable, type-preserving token."""
    if isinstance(edge_label, int):
        return f"i:{edge_label}"
    return f"s:{edge_label}"


def _label_signature(labels: Iterable[EdgeLabel]) -> LabelSignature:
    """Canonicalize an oriented cyclic label sequence up to rotation."""
    tokens = tuple(_edge_label_token(label) for label in labels)
    return min(
        (tokens[index:] + tokens[:index] for index in range(len(tokens))),
        default=(),
    )


def _match_label_signatures(
    source_signatures: tuple[LabelSignature, ...],
    target_signatures: tuple[LabelSignature, ...],
    *,
    source_name: str,
    target_name: str,
) -> tuple[int, ...]:
    """Match entities by oriented cyclic edge-label signature."""
    if len(source_signatures) != len(target_signatures):
        raise ValueError(
            f"signature count mismatch: {source_name} -> {target_name} "
            f"({len(source_signatures)} != {len(target_signatures)})"
        )

    target_by_signature: dict[LabelSignature, list[int]] = {}
    for target_idx, signature in enumerate(target_signatures):
        target_by_signature.setdefault(signature, []).append(target_idx)

    used_targets: set[int] = set()
    mapping: list[int] = []
    for source_idx, signature in enumerate(source_signatures):
        candidates = [
            target_idx
            for target_idx in target_by_signature.get(signature, [])
            if target_idx not in used_targets
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"signature map ambiguous: {source_name} {source_idx} "
                f"-> {target_name}"
            )
        target_idx = candidates[0]
        used_targets.add(target_idx)
        mapping.append(target_idx)

    return tuple(mapping)


class FrozenEdgeMultiplicity(Mapping[tuple[int, int], int]):
    """Immutable mapping wrapper for edge multiplicities."""

    __slots__ = ("_data", "_items")
    _data: Mapping[tuple[int, int], int]
    _items: tuple[tuple[tuple[int, int], int], ...]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __init__(
        self,
        edge_multiplicity: (
            Mapping[tuple[int, int], int]
            | tuple[tuple[tuple[int, int], int], ...]
        ),
    ) -> None:
        _set = object.__setattr__
        if isinstance(edge_multiplicity, FrozenEdgeMultiplicity):
            _set(self, "_data", edge_multiplicity._data)
            _set(self, "_items", edge_multiplicity._items)
            return

        if isinstance(edge_multiplicity, Mapping):
            raw_items = tuple(edge_multiplicity.items())
        elif type(edge_multiplicity) is tuple:
            raw_items = edge_multiplicity
        else:
            raise TypeError(
                "edge_multiplicity must be a mapping or canonical tuple"
            )

        normalized: dict[tuple[int, int], int] = {}
        for raw_item in raw_items:
            if type(raw_item) is not tuple or len(raw_item) != 2:
                raise TypeError(
                    "edge_multiplicity entries must be "
                    "((int, int), int) tuples"
                )
            raw_edge, raw_multiplicity = raw_item
            if type(raw_edge) is not tuple or len(raw_edge) != 2:
                raise TypeError(
                    f"edge_multiplicity keys must be 2-tuples; got {raw_edge!r}."
                )
            u, v = raw_edge
            if type(u) is not int or type(v) is not int:
                raise TypeError(
                    f"edge_multiplicity keys must be integer vertex indices; got {raw_edge!r}."
                )
            if type(raw_multiplicity) is not int:
                raise TypeError(
                    f"edge_multiplicity values must be integers; got {raw_multiplicity!r} on edge {raw_edge!r}."
                )
            edge = (u, v)
            if edge in normalized:
                raise ValueError(f"Duplicate edge key encountered: {edge}")
            normalized[edge] = raw_multiplicity

        ordered_items = tuple(sorted(normalized.items()))
        _set(self, "_items", ordered_items)
        _set(self, "_data", MappingProxyType(normalized))

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


@dataclass(frozen=True, slots=True)
class PlaneGraph:
    """Immutable plane graph with fixed exterior-view clockwise embedding.

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
    # Canonical undirected support-edge pairs (size s + d); parallel copies live in dual_edge_multiplicity.
    dual_support_edges: tuple[tuple[int, int], ...]
    dual_edge_multiplicity: Mapping[tuple[int, int], int]
    dual_embedding: Embedding  # Exterior-view CW cyclic order at each vertex.
    dual_faces: tuple[tuple[int, ...], ...]

    primal_num_vertices: int
    primal_embedding: Embedding
    primal_faces: tuple[tuple[int, ...], ...]
    dual_vertex_to_primal_face: tuple[int, ...] = tuple()
    primal_vertex_to_dual_face: tuple[int, ...] = tuple()
    dual_edge_label_pairs: EdgeLabelPairEntries = field(
        default=tuple(),
        repr=False,
        compare=False,
    )
    primal_edge_label_pairs: EdgeLabelPairEntries = field(
        default=tuple(),
        repr=False,
        compare=False,
    )

    graph_id: int = 0
    _double_edges_cache: frozenset[tuple[int, int]] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @staticmethod
    def _coerce_support_edges(
        edges: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        if type(edges) is not tuple or any(
            type(edge) is not tuple
            or len(edge) != 2
            or type(edge[0]) is not int
            or type(edge[1]) is not int
            for edge in edges
        ):
            raise TypeError(
                "dual_support_edges must be tuple[tuple[int, int], ...]"
            )
        return edges

    @staticmethod
    def _coerce_nested_int_tuples(
        values: tuple[tuple[int, ...], ...],
        *,
        field_name: str,
    ) -> tuple[tuple[int, ...], ...]:
        if type(values) is not tuple or any(
            type(row) is not tuple or any(type(value) is not int for value in row)
            for row in values
        ):
            raise TypeError(
                f"{field_name} must be tuple[tuple[int, ...], ...]"
            )
        return values

    @staticmethod
    def _coerce_index_tuple(
        indices: tuple[int, ...],
        *,
        field_name: str,
    ) -> tuple[int, ...]:
        if type(indices) is not tuple or any(type(index) is not int for index in indices):
            raise TypeError(f"{field_name} must be tuple[int, ...]")
        return indices

    @staticmethod
    def _require_int(value: int, *, field_name: str) -> int:
        if type(value) is not int:
            raise TypeError(f"{field_name} must be int")
        return value

    @staticmethod
    def _coerce_edge_label(label: EdgeLabel) -> EdgeLabel:
        if type(label) is int:
            return label
        if type(label) is str:
            return label
        raise TypeError(f"edge label must be int|str, got {type(label).__name__}")

    @staticmethod
    def _coerce_half_edge(half_edge: HalfEdge) -> HalfEdge:
        if (
            type(half_edge) is not tuple
            or len(half_edge) != 2
            or type(half_edge[0]) is not int
            or type(half_edge[1]) is not int
        ):
            raise TypeError(
                f"half-edge must be tuple[int, int]; got {half_edge!r}"
            )
        return half_edge

    @staticmethod
    def _edge_label_sort_key(label: EdgeLabel) -> tuple[int, int | str]:
        if type(label) is int:
            return (0, label)
        return (1, label)

    @classmethod
    def _normalize_edge_label_entry(
        cls,
        label: EdgeLabel,
        half_edge_a: HalfEdge,
        half_edge_b: HalfEdge,
    ) -> tuple[EdgeLabel, HalfEdge, HalfEdge]:
        normalized_label = cls._coerce_edge_label(label)
        normalized_half_edge_a, normalized_half_edge_b = sorted(
            (cls._coerce_half_edge(half_edge_a), cls._coerce_half_edge(half_edge_b))
        )
        return normalized_label, normalized_half_edge_a, normalized_half_edge_b

    @classmethod
    def _coerce_edge_label_pairs(
        cls,
        edge_label_pairs: EdgeLabelPairEntries,
    ) -> EdgeLabelPairEntries:
        if type(edge_label_pairs) is not tuple:
            raise TypeError(
                "edge label pairs must be "
                "tuple[tuple[int | str, HalfEdge, HalfEdge], ...]"
            )

        normalized_entries: list[tuple[EdgeLabel, HalfEdge, HalfEdge]] = []
        seen_labels: set[EdgeLabel] = set()
        for raw_entry in edge_label_pairs:
            if type(raw_entry) is not tuple or len(raw_entry) != 3:
                raise TypeError(
                    "edge label entry must be "
                    "tuple[int | str, HalfEdge, HalfEdge]"
                )
            raw_label, raw_h1, raw_h2 = raw_entry
            normalized_entry = cls._normalize_edge_label_entry(
                raw_label,
                raw_h1,
                raw_h2,
            )
            label = normalized_entry[0]
            if label in seen_labels:
                raise ValueError(f"duplicate edge label encountered: {label!r}")
            seen_labels.add(label)
            normalized_entries.append(normalized_entry)

        return tuple(
            sorted(
                normalized_entries,
                key=lambda entry: (
                    cls._edge_label_sort_key(entry[0]),
                    entry[1],
                    entry[2],
                ),
            )
        )

    def __post_init__(self) -> None:
        """Validate canonical inputs and freeze edge multiplicities."""
        _set = object.__setattr__

        _set(
            self,
            "dual_num_vertices",
            self._require_int(
                self.dual_num_vertices,
                field_name="dual_num_vertices",
            ),
        )
        _set(
            self,
            "primal_num_vertices",
            self._require_int(
                self.primal_num_vertices,
                field_name="primal_num_vertices",
            ),
        )
        _set(
            self,
            "graph_id",
            self._require_int(self.graph_id, field_name="graph_id"),
        )

        _set(
            self,
            "dual_support_edges",
            self._coerce_support_edges(self.dual_support_edges),
        )

        if not isinstance(self.dual_edge_multiplicity, FrozenEdgeMultiplicity):
            _set(
                self,
                "dual_edge_multiplicity",
                FrozenEdgeMultiplicity(self.dual_edge_multiplicity),
            )

        _set(
            self,
            "dual_embedding",
            self._coerce_nested_int_tuples(
                self.dual_embedding,
                field_name="dual_embedding",
            ),
        )
        _set(
            self,
            "primal_embedding",
            self._coerce_nested_int_tuples(
                self.primal_embedding,
                field_name="primal_embedding",
            ),
        )

        _set(
            self,
            "dual_faces",
            self._coerce_nested_int_tuples(
                self.dual_faces,
                field_name="dual_faces",
            ),
        )
        _set(
            self,
            "primal_faces",
            self._coerce_nested_int_tuples(
                self.primal_faces,
                field_name="primal_faces",
            ),
        )
        _set(
            self,
            "dual_vertex_to_primal_face",
            self._coerce_index_tuple(
                self.dual_vertex_to_primal_face,
                field_name="dual_vertex_to_primal_face",
            ),
        )
        _set(
            self,
            "primal_vertex_to_dual_face",
            self._coerce_index_tuple(
                self.primal_vertex_to_dual_face,
                field_name="primal_vertex_to_dual_face",
            ),
        )
        _set(
            self,
            "dual_edge_label_pairs",
            self._coerce_edge_label_pairs(self.dual_edge_label_pairs),
        )
        _set(
            self,
            "primal_edge_label_pairs",
            self._coerce_edge_label_pairs(self.primal_edge_label_pairs),
        )

    def __getstate__(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _PLANE_GRAPH_STATE_FIELDS}

    def __setstate__(self, state: Any) -> None:
        if type(state) is not dict:
            raise TypeError(
                f"PlaneGraph pickle state must be dict; got {type(state).__name__}"
            )

        expected_keys = set(_PLANE_GRAPH_STATE_FIELDS)
        actual_keys = set(state)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"PlaneGraph pickle state keys mismatch: missing={missing}, extra={extra}"
            )

        for name in _PLANE_GRAPH_STATE_FIELDS:
            object.__setattr__(self, name, state[name])
        object.__setattr__(self, "_double_edges_cache", None)
        self.__post_init__()

    def _validate_field_types(self, errors: list[str]) -> bool:
        """Reject state that bypassed the canonical constructor boundary."""
        try:
            for field_name in _PLANE_GRAPH_SCALAR_FIELDS:
                self._require_int(
                    getattr(self, field_name),
                    field_name=field_name,
                )
            self._coerce_support_edges(self.dual_support_edges)
            if not isinstance(
                self.dual_edge_multiplicity,
                FrozenEdgeMultiplicity,
            ):
                raise TypeError(
                    "dual_edge_multiplicity must be FrozenEdgeMultiplicity"
                )
            for field_name in (
                "dual_embedding",
                "primal_embedding",
                "dual_faces",
                "primal_faces",
            ):
                self._coerce_nested_int_tuples(
                    getattr(self, field_name),
                    field_name=field_name,
                )
            for field_name in (
                "dual_vertex_to_primal_face",
                "primal_vertex_to_dual_face",
            ):
                self._coerce_index_tuple(
                    getattr(self, field_name),
                    field_name=field_name,
                )
            for field_name in (
                "dual_edge_label_pairs",
                "primal_edge_label_pairs",
            ):
                self._coerce_edge_label_pairs(getattr(self, field_name))
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            return False
        return True

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
                if u < 0 or u >= vertex_count:
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

        reached: set[int] = set()
        pending = [0] if vertex_count > 0 and embedding else []
        while pending:
            vertex = pending.pop()
            if vertex in reached:
                continue
            reached.add(vertex)
            if vertex >= len(embedding):
                continue
            pending.extend(
                neighbor
                for neighbor in embedding[vertex]
                if 0 <= neighbor < vertex_count and neighbor not in reached
            )
        if vertex_count > 0 and len(reached) != vertex_count:
            errors.append(
                f"{embedding_name} is disconnected: "
                f"reached {len(reached)}/{vertex_count} vertices"
            )

        return directed_counts, undirected_half_edge_counts

    def _reconstruct_half_edge_maps(
        self,
        *,
        graph_name: str,
        embedding: Embedding,
        edge_label_pairs: EdgeLabelPairEntries,
        errors: list[str],
    ) -> tuple[dict[HalfEdge, HalfEdge] | None, dict[HalfEdge, EdgeLabel] | None]:
        if not edge_label_pairs:
            return None, None

        twin_map: dict[HalfEdge, HalfEdge] = {}
        half_edge_labels: dict[HalfEdge, EdgeLabel] = {}
        seen_edge_labels: set[EdgeLabel] = set()
        expected_half_edge_count = sum(len(neighbors) for neighbors in embedding)

        for edge_label, half_edge_a, half_edge_b in edge_label_pairs:
            if edge_label in seen_edge_labels:
                errors.append(f"{graph_name} duplicate edge label: {edge_label!r}")
            seen_edge_labels.add(edge_label)

            for half_edge in (half_edge_a, half_edge_b):
                vertex, slot_idx = half_edge
                if vertex < 0 or vertex >= len(embedding):
                    errors.append(
                        f"{graph_name} edge-label pair out-of-range vertex: {half_edge}"
                    )
                    continue
                if slot_idx < 0 or slot_idx >= len(embedding[vertex]):
                    errors.append(
                        f"{graph_name} edge-label pair out-of-range slot: {half_edge}"
                    )

            for src, dst in ((half_edge_a, half_edge_b), (half_edge_b, half_edge_a)):
                existing_twin = twin_map.get(src)
                if existing_twin is not None and existing_twin != dst:
                    errors.append(
                        f"{graph_name} twin_map conflict at {src}: {existing_twin} != {dst}"
                    )
                twin_map[src] = dst

            for half_edge in (half_edge_a, half_edge_b):
                existing_label = half_edge_labels.get(half_edge)
                if existing_label is not None and existing_label != edge_label:
                    errors.append(
                        f"{graph_name} half-edge label conflict at {half_edge}: "
                        f"{existing_label!r} != {edge_label!r}"
                    )
                half_edge_labels[half_edge] = edge_label

        if len(half_edge_labels) != expected_half_edge_count:
            errors.append(
                f"{graph_name} edge-label coverage mismatch: "
                f"{len(half_edge_labels)} != {expected_half_edge_count}"
            )
        if len(twin_map) != expected_half_edge_count:
            errors.append(
                f"{graph_name} twin_map coverage mismatch: "
                f"{len(twin_map)} != {expected_half_edge_count}"
            )

        return twin_map, half_edge_labels

    def _extract_reconstructed_faces(
        self,
        *,
        graph_name: str,
        embedding: Embedding,
        edge_label_pairs: EdgeLabelPairEntries,
        errors: list[str],
    ) -> tuple[
        tuple[tuple[int, ...], ...] | None,
        tuple[LabelSignature, ...] | None,
        dict[HalfEdge, EdgeLabel] | None,
        dict[EdgeLabel, tuple[int, ...]] | None,
    ]:
        twin_map, half_edge_labels = self._reconstruct_half_edge_maps(
            graph_name=graph_name,
            embedding=embedding,
            edge_label_pairs=edge_label_pairs,
            errors=errors,
        )
        if twin_map is None or half_edge_labels is None:
            return None, None, None, None

        try:
            face_cycles = GraphConverter.extract_face_half_edge_cycles(
                dict(enumerate(embedding)),
                twin_map,
                graph_name=graph_name,
            )
        except Exception as exc:
            errors.append(f"{graph_name} face reconstruction failed: {exc}")
            return None, None, half_edge_labels, None

        reconstructed_faces: list[tuple[int, ...]] = []
        face_label_signatures: list[LabelSignature] = []
        edge_label_faces: dict[EdgeLabel, list[int]] = {}
        for face_index, face_cycle in enumerate(face_cycles):
            labels: list[EdgeLabel] = []
            for half_edge in face_cycle:
                label = half_edge_labels.get(half_edge)
                if label is None:
                    errors.append(
                        f"{graph_name} reconstructed face uses unlabeled half-edge: {half_edge}"
                    )
                    continue
                labels.append(label)
                edge_label_faces.setdefault(label, []).append(face_index)
            reconstructed_faces.append(tuple(vertex for vertex, _ in face_cycle))
            face_label_signatures.append(_label_signature(labels))

        return (
            tuple(reconstructed_faces),
            tuple(face_label_signatures),
            half_edge_labels,
            {
                edge_label: tuple(face_indices)
                for edge_label, face_indices in edge_label_faces.items()
            },
        )

    @staticmethod
    def _validate_reconstructed_primal_simplicity(
        edge_label_faces: dict[EdgeLabel, tuple[int, ...]],
        errors: list[str],
    ) -> None:
        # Each dual edge's incident faces become its reconstructed primal endpoints.
        seen_face_pairs: set[tuple[int, int]] = set()
        for face_indices in edge_label_faces.values():
            if len(face_indices) != 2:
                continue
            face_a, face_b = face_indices
            if face_a == face_b:
                errors.append(f"reconstructed primal has loop at face {face_a}")
                continue
            face_pair = (face_a, face_b) if face_a < face_b else (face_b, face_a)
            if face_pair in seen_face_pairs:
                errors.append(
                    "reconstructed primal has parallel edge between faces "
                    f"{face_pair[0]} and {face_pair[1]}"
                )
            seen_face_pairs.add(face_pair)

    def _vertex_label_signatures(
        self,
        *,
        graph_name: str,
        embedding: Embedding,
        half_edge_labels: dict[HalfEdge, EdgeLabel],
        errors: list[str],
    ) -> tuple[LabelSignature, ...] | None:
        signatures: list[LabelSignature] = []
        for vertex, neighbors in enumerate(embedding):
            labels: list[EdgeLabel] = []
            for slot_idx in range(len(neighbors)):
                half_edge = (vertex, slot_idx)
                label = half_edge_labels.get(half_edge)
                if label is None:
                    errors.append(
                        f"{graph_name} vertex uses unlabeled half-edge: {half_edge}"
                    )
                    return None
                labels.append(label)
            signatures.append(_label_signature(labels))
        return tuple(signatures)

    @staticmethod
    def _validate_bijection(
        mapping: tuple[int, ...],
        *,
        errors: list[str],
        mapping_name: str,
        expected_size: int,
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
            if (
                u < 0
                or v < 0
                or u >= self.dual_num_vertices
                or v >= self.dual_num_vertices
            ):
                errors.append(
                    "dual_support_edges out of range: "
                    f"({u}, {v}) for n={self.dual_num_vertices}"
                )
            if u > v:
                errors.append(f"dual_support_edges not canonical: ({u}, {v})")

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
                errors.append(
                    f"Dual face {face_idx} has size {len(face)}, expected >= 2"
                )
            if len(set(face)) != len(face):
                errors.append(f"Dual face {face_idx} repeats vertices: {face}")
            for vertex in face:
                if vertex < 0 or vertex >= self.dual_num_vertices:
                    errors.append(f"Dual face {face_idx} out-of-range vertex: {vertex}")

    def _validate_dual_edge_multiplicity(
        self,
        directed_counts: dict[tuple[int, int], int],
        undirected_half_edge_counts: dict[tuple[int, int], int],
        errors: list[str],
    ) -> int:
        for (u, v), multiplicity in self.dual_edge_multiplicity.items():
            if (
                u < 0
                or v < 0
                or u >= self.dual_num_vertices
                or v >= self.dual_num_vertices
            ):
                errors.append(
                    "dual_edge_multiplicity out of range: "
                    f"({u}, {v}) for n={self.dual_num_vertices}"
                )
            if u > v:
                errors.append(f"dual_edge_multiplicity not canonical: ({u}, {v})")
            if type(multiplicity) is not int:
                errors.append(
                    f"Edge ({u}, {v}) multiplicity must be an integer; "
                    f"got {type(multiplicity).__name__}"
                )
                continue
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
                errors.append(f"Edge {edge} missing from dual_edge_multiplicity")
                continue
            if half_edge_count != 2 * edge_multiplicity:
                errors.append(
                    f"Edge {edge} half-edge mismatch: "
                    f"{half_edge_count} != {2 * edge_multiplicity}"
                )

        return sum(
            multiplicity
            for multiplicity in self.dual_edge_multiplicity.values()
            if type(multiplicity) is int
        )

    def _validate_dual_digon_correspondence(self, errors: list[str]) -> None:
        """Require a one-to-one correspondence between digons and double edges."""
        digon_edge_counts: dict[tuple[int, int], int] = {}
        for face in self.dual_faces:
            if len(face) != 2 or face[0] == face[1]:
                continue
            u, v = face
            edge = (u, v) if u < v else (v, u)
            digon_edge_counts[edge] = digon_edge_counts.get(edge, 0) + 1

        expected_counts = {edge: 1 for edge in self.double_edges}
        if digon_edge_counts != expected_counts:
            errors.append(
                "dual digon/double-edge correspondence mismatch: "
                f"digons={tuple(sorted(digon_edge_counts.items()))!r}, "
                f"double_edges={tuple(sorted(self.double_edges))!r}"
            )

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
            self.primal_num_vertices != 0
            or bool(self.primal_embedding)
            or bool(self.primal_faces)
            or bool(self.dual_vertex_to_primal_face)
            or bool(self.primal_vertex_to_dual_face)
            or bool(self.primal_edge_label_pairs)
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
        self._validate_dual_digon_correspondence(errors)

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

        primal_directed_counts, primal_undirected_half_edge_counts = (
            self._scan_embedding(
                self.primal_embedding,
                vertex_count=self.primal_num_vertices,
                errors=errors,
                vertex_label="Primal vertex",
                loop_label="Primal self-loop",
                embedding_name="Primal embedding",
            )
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
            target_count=len(self.primal_faces),
            duplicate_target_label="primal face",
            onto_label="primal_faces",
        )
        self._validate_bijection(
            self.primal_vertex_to_dual_face,
            errors=errors,
            mapping_name="primal_vertex_to_dual_face",
            expected_size=self.primal_num_vertices,
            target_count=len(self.dual_faces),
            duplicate_target_label="dual face",
            onto_label="dual faces",
        )

    def _validate_topological_consistency(self, errors: list[str]) -> None:
        if not self.dual_edge_label_pairs:
            errors.append("dual topology metadata missing: dual_edge_label_pairs")
            return
        (
            dual_faces_reconstructed,
            dual_face_signatures,
            dual_half_edge_labels,
            dual_edge_label_faces,
        ) = self._extract_reconstructed_faces(
            graph_name="dual",
            embedding=self.dual_embedding,
            edge_label_pairs=self.dual_edge_label_pairs,
            errors=errors,
        )
        if dual_edge_label_faces is not None:
            self._validate_reconstructed_primal_simplicity(
                dual_edge_label_faces,
                errors,
            )
        if (
            dual_faces_reconstructed is not None
            and dual_faces_reconstructed != self.dual_faces
        ):
            errors.append("dual_faces topology mismatch")

        if not self._has_primal_data():
            return
        if not self.primal_edge_label_pairs:
            errors.append("primal topology metadata missing: primal_edge_label_pairs")
            return

        (
            primal_faces_reconstructed,
            primal_face_signatures,
            primal_half_edge_labels,
            _,
        ) = self._extract_reconstructed_faces(
            graph_name="primal",
            embedding=self.primal_embedding,
            edge_label_pairs=self.primal_edge_label_pairs,
            errors=errors,
        )
        if (
            primal_faces_reconstructed is not None
            and primal_faces_reconstructed != self.primal_faces
        ):
            errors.append("primal_faces topology mismatch")

        if (
            dual_face_signatures is None
            or primal_face_signatures is None
            or dual_half_edge_labels is None
            or primal_half_edge_labels is None
        ):
            return

        dual_labels = {label for label, _, _ in self.dual_edge_label_pairs}
        primal_labels = {label for label, _, _ in self.primal_edge_label_pairs}
        if dual_labels != primal_labels:
            errors.append("primal/dual edge label sets mismatch")
            return

        dual_vertex_signatures = self._vertex_label_signatures(
            graph_name="dual",
            embedding=self.dual_embedding,
            half_edge_labels=dual_half_edge_labels,
            errors=errors,
        )
        primal_vertex_signatures = self._vertex_label_signatures(
            graph_name="primal",
            embedding=self.primal_embedding,
            half_edge_labels=primal_half_edge_labels,
            errors=errors,
        )
        if dual_vertex_signatures is None or primal_vertex_signatures is None:
            return

        try:
            reconstructed_dual_vertex_to_primal_face = _match_label_signatures(
                dual_vertex_signatures,
                primal_face_signatures,
                source_name="dual vertex",
                target_name="primal face",
            )
            if (
                reconstructed_dual_vertex_to_primal_face
                != self.dual_vertex_to_primal_face
            ):
                errors.append("dual_vertex_to_primal_face topology mismatch")

            reconstructed_primal_vertex_to_dual_face = _match_label_signatures(
                primal_vertex_signatures,
                dual_face_signatures,
                source_name="primal vertex",
                target_name="dual face",
            )
            if (
                reconstructed_primal_vertex_to_dual_face
                != self.primal_vertex_to_dual_face
            ):
                errors.append("primal_vertex_to_dual_face topology mismatch")
        except ValueError as exc:
            errors.append(str(exc))

    @property
    def dual_num_faces(self) -> int:
        """Face count (should be n + 2 by Euler's formula)."""
        return len(self.dual_faces)

    @property
    def double_edges(self) -> frozenset[tuple[int, int]]:
        """Parallel-edge pairs (digons) in the dual embedding."""
        cache = self._double_edges_cache
        if cache is None:
            cache = frozenset(
                e for e, m in self.dual_edge_multiplicity.items() if m == 2
            ) or _EMPTY_DOUBLE_EDGES
            object.__setattr__(self, "_double_edges_cache", cache)
        return cache

    def validate(self) -> tuple[bool, list[str]]:
        """Validates graph invariants."""
        # Sub-validators append to `errors` rather than raising, so one call reports everything.
        errors: list[str] = []
        if not self._validate_field_types(errors):
            return False, errors

        # Mirrors plantri.MIN_DUAL_VERTEX_COUNT, the smallest enumerable dual.
        if self.dual_num_vertices < 3:
            errors.append(
                f"dual_num_vertices must be >= 3, got {self.dual_num_vertices}"
            )
        if self.primal_num_vertices < 0:
            errors.append(
                f"primal_num_vertices must be non-negative, got {self.primal_num_vertices}"
            )
        if self.graph_id < 0:
            errors.append(
                f"graph_id must be non-negative, got {self.graph_id}"
            )

        self._validate_dual_contract(errors)
        # Primal fields are legitimately empty when built with include_primal=False.
        if self._has_primal_data():
            self._validate_primal_contract(errors)
        # Runs even for dual-only graphs: re-derives faces, checks the implied primal.
        self._validate_topological_consistency(errors)

        return len(errors) == 0, errors
