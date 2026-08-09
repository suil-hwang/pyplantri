# src/pyplantri/builder.py
from __future__ import annotations

from dataclasses import dataclass

from .plane_graph import (
    LabelSignature,
    PlaneGraph,
    _extract_right_face_half_edge_cycles,
    _label_signature,
    _match_label_signatures,
)
from .plantri import ParsedGraphSection
from .types import EdgeLabel, EdgeLabelPairs, Embedding, HalfEdge


@dataclass(frozen=True, slots=True)
class _ZeroBasedGraphSection:
    vertex_count: int
    cyclic_adjacency: dict[int, tuple[int, ...]]
    twin_map: dict[HalfEdge, HalfEdge]
    half_edge_pair_by_label: EdgeLabelPairs
    edge_label_by_half_edge: dict[HalfEdge, EdgeLabel]


def _project_face_half_edge_cycles(
    face_half_edge_cycles: tuple[tuple[HalfEdge, ...], ...],
    edge_label_by_half_edge: dict[HalfEdge, EdgeLabel],
    *,
    graph_name: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[LabelSignature, ...]]:
    """Project oriented face half-edge cycles without identifying reflections."""
    face_vertex_cycles: list[tuple[int, ...]] = []
    face_label_signatures: list[LabelSignature] = []

    for face_half_edges in face_half_edge_cycles:
        boundary_edge_labels: list[EdgeLabel] = []
        for half_edge in face_half_edges:
            edge_label = edge_label_by_half_edge.get(half_edge)
            if edge_label is None:
                raise ValueError(f"{graph_name}: unlabeled half-edge {half_edge}")
            boundary_edge_labels.append(edge_label)
        face_vertex_cycles.append(tuple(vertex for vertex, _ in face_half_edges))
        face_label_signatures.append(_label_signature(boundary_edge_labels))

    return tuple(face_vertex_cycles), tuple(face_label_signatures)


def _vertex_edge_label_signatures(
    cyclic_adjacency: dict[int, tuple[int, ...]],
    edge_label_by_half_edge: dict[HalfEdge, EdgeLabel],
    *,
    vertex_count: int,
    graph_name: str,
) -> tuple[LabelSignature, ...]:
    """Build vertex-indexed oriented edge-label signatures modulo cyclic shift."""
    vertex_edge_label_signatures: list[LabelSignature] = []
    for vertex in range(vertex_count):
        cyclic_neighbors = cyclic_adjacency.get(vertex)
        if cyclic_neighbors is None:
            raise ValueError(
                f"{graph_name}: missing embedding vertex {vertex}"
            )
        cyclic_edge_labels: list[EdgeLabel] = []
        for slot in range(len(cyclic_neighbors)):
            half_edge = (vertex, slot)
            edge_label = edge_label_by_half_edge.get(half_edge)
            if edge_label is None:
                raise ValueError(
                    f"{graph_name}: unlabeled half-edge {half_edge}"
                )
            cyclic_edge_labels.append(edge_label)
        vertex_edge_label_signatures.append(_label_signature(cyclic_edge_labels))
    return tuple(vertex_edge_label_signatures)


def _to_zero_based_graph_section(
    parsed_section: ParsedGraphSection,
    *,
    graph_name: str,
) -> _ZeroBasedGraphSection:
    """Convert a parsed section to zero-based indices and verify label/twin alignment."""
    cyclic_adjacency = {
        vertex - 1: tuple(neighbor - 1 for neighbor in cyclic_neighbors)
        for vertex, cyclic_neighbors in parsed_section.cyclic_adjacency.items()
    }
    # plantri vertex ids are 1-based, while cyclic-order slots are already 0-based.
    twin_map = {
        (vertex - 1, slot): (twin_vertex - 1, twin_slot)
        for (vertex, slot), (twin_vertex, twin_slot) in parsed_section.twin_map.items()
    }
    half_edge_pair_by_label = {
        edge_label: ((vertex_a - 1, slot_a), (vertex_b - 1, slot_b))
        for edge_label, ((vertex_a, slot_a), (vertex_b, slot_b))
        in parsed_section.edge_label_pairs.items()
    }
    edge_label_by_half_edge: dict[HalfEdge, EdgeLabel] = {}
    for edge_label, (half_edge_a, half_edge_b) in half_edge_pair_by_label.items():
        if twin_map.get(half_edge_a) != half_edge_b or twin_map.get(half_edge_b) != half_edge_a:
            raise ValueError(f"{graph_name}: label/twin mismatch {edge_label!r}")
        for half_edge in (half_edge_a, half_edge_b):
            if half_edge in edge_label_by_half_edge:
                raise ValueError(f"{graph_name}: multiple labels {half_edge}")
            edge_label_by_half_edge[half_edge] = edge_label
    if set(edge_label_by_half_edge) != set(twin_map):
        raise ValueError(f"{graph_name}: label/twin domain mismatch")

    return _ZeroBasedGraphSection(
        vertex_count=parsed_section.vertex_count,
        cyclic_adjacency=cyclic_adjacency,
        twin_map=twin_map,
        half_edge_pair_by_label=half_edge_pair_by_label,
        edge_label_by_half_edge=edge_label_by_half_edge,
    )


def _to_dense_embedding(
    cyclic_adjacency: dict[int, tuple[int, ...]],
    *,
    vertex_count: int,
    graph_name: str,
) -> Embedding:
    """Convert exact zero-based cyclic adjacency to vertex-indexed tuple form."""
    expected_vertex_indices = set(range(vertex_count))
    actual_vertex_indices = set(cyclic_adjacency)
    if actual_vertex_indices != expected_vertex_indices:
        raise ValueError(
            f"{graph_name}: embedding vertices {tuple(sorted(actual_vertex_indices))!r}"
        )
    return tuple(cyclic_adjacency[vertex_index] for vertex_index in range(vertex_count))


def _build_plane_graph_from_sections(
    parsed_primal: ParsedGraphSection,
    parsed_dual: ParsedGraphSection,
    graph_id: int,
    *,
    include_primal: bool = True,
) -> PlaneGraph:
    """Build PlaneGraph from parsed primal/dual double_code sections."""
    dual_section = _to_zero_based_graph_section(parsed_dual, graph_name="dual")
    dual_vertex_count = dual_section.vertex_count
    dual_edge_multiplicity: dict[tuple[int, int], int] = {}
    for (vertex_a, _), (vertex_b, _) in dual_section.half_edge_pair_by_label.values():
        edge = (vertex_a, vertex_b) if vertex_a <= vertex_b else (vertex_b, vertex_a)
        dual_edge_multiplicity[edge] = dual_edge_multiplicity.get(edge, 0) + 1
    dual_support_edges = tuple(sorted(dual_edge_multiplicity))
    dual_face_half_edge_cycles = _extract_right_face_half_edge_cycles(
        dual_section.cyclic_adjacency,
        dual_section.twin_map,
        graph_name="dual",
    )
    dual_face_vertex_cycles, dual_face_label_signatures = _project_face_half_edge_cycles(
        dual_face_half_edge_cycles,
        dual_section.edge_label_by_half_edge,
        graph_name="dual",
    )

    primal_num_vertices = 0
    primal_cyclic_adjacency: dict[int, tuple[int, ...]] = {}
    primal_face_vertex_cycles: tuple[tuple[int, ...], ...] = tuple()
    primal_edge_label_entries: tuple[tuple[EdgeLabel, HalfEdge, HalfEdge], ...] = tuple()
    dual_vertex_to_primal_face: tuple[int, ...] = tuple()
    primal_vertex_to_dual_face: tuple[int, ...] = tuple()

    if include_primal:
        primal_section = _to_zero_based_graph_section(parsed_primal, graph_name="primal")
        primal_num_vertices = primal_section.vertex_count
        primal_cyclic_adjacency = primal_section.cyclic_adjacency
        primal_edge_label_entries = tuple(
            (edge_label, half_edge_a, half_edge_b)
            for edge_label, (half_edge_a, half_edge_b) in primal_section.half_edge_pair_by_label.items()
        )
        primal_face_half_edge_cycles = _extract_right_face_half_edge_cycles(
            primal_section.cyclic_adjacency,
            primal_section.twin_map,
            graph_name="primal",
        )
        primal_face_vertex_cycles, primal_face_label_signatures = _project_face_half_edge_cycles(
            primal_face_half_edge_cycles,
            primal_section.edge_label_by_half_edge,
            graph_name="primal",
        )
        dual_vertex_edge_label_signatures = _vertex_edge_label_signatures(
            dual_section.cyclic_adjacency,
            dual_section.edge_label_by_half_edge,
            vertex_count=dual_vertex_count,
            graph_name="dual",
        )
        primal_vertex_edge_label_signatures = _vertex_edge_label_signatures(
            primal_section.cyclic_adjacency,
            primal_section.edge_label_by_half_edge,
            vertex_count=primal_num_vertices,
            graph_name="primal",
        )
        # Match vertices to faces by oriented cyclic edge labels, up to rotation.
        dual_vertex_to_primal_face = _match_label_signatures(
            dual_vertex_edge_label_signatures,
            primal_face_label_signatures,
            source_name="dual vertex",
            target_name="primal face",
        )
        primal_vertex_to_dual_face = _match_label_signatures(
            primal_vertex_edge_label_signatures,
            dual_face_label_signatures,
            source_name="primal vertex",
            target_name="dual face",
        )

    dual_embedding = _to_dense_embedding(
        dual_section.cyclic_adjacency,
        vertex_count=dual_vertex_count,
        graph_name="dual",
    )
    primal_embedding = _to_dense_embedding(
        primal_cyclic_adjacency,
        vertex_count=primal_num_vertices,
        graph_name="primal",
    )
    return PlaneGraph(
        dual_num_vertices=dual_vertex_count,
        dual_support_edges=dual_support_edges,
        dual_edge_multiplicity=dual_edge_multiplicity,
        dual_edge_label_pairs=tuple(
            (edge_label, half_edge_a, half_edge_b)
            for edge_label, (half_edge_a, half_edge_b) in dual_section.half_edge_pair_by_label.items()
        ),
        dual_embedding=dual_embedding,
        dual_faces=dual_face_vertex_cycles,
        primal_num_vertices=primal_num_vertices,
        primal_edge_label_pairs=primal_edge_label_entries,
        primal_embedding=primal_embedding,
        primal_faces=primal_face_vertex_cycles,
        dual_vertex_to_primal_face=dual_vertex_to_primal_face,
        primal_vertex_to_dual_face=primal_vertex_to_dual_face,
        graph_id=graph_id,
    )
