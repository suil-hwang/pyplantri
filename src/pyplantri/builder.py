# src/pyplantri/builder.py
from __future__ import annotations

from dataclasses import dataclass

from .converter import GraphConverter
from .plane_graph import (
    LabelSignature,
    PlaneGraph,
    _label_signature,
    _match_label_signatures,
)
from .plantri import ParsedGraphSection
from .types import EdgeLabel, EdgeLabelPairs, HalfEdge


@dataclass(frozen=True, slots=True)
class _PreparedSectionData:
    vertex_count: int
    embedding: dict[int, tuple[int, ...]]
    twin_map: dict[HalfEdge, HalfEdge]
    edge_label_pairs: EdgeLabelPairs
    half_edge_labels: dict[HalfEdge, EdgeLabel]


def _build_half_edge_label_map(
    edge_label_pairs: EdgeLabelPairs,
    twin_map: dict[HalfEdge, HalfEdge],
    *,
    graph_name: str,
) -> dict[HalfEdge, EdgeLabel]:
    """Invert edge labels and require them to encode the twin involution."""
    half_edge_labels: dict[HalfEdge, EdgeLabel] = {}
    for edge_label, (h1, h2) in edge_label_pairs.items():
        if twin_map.get(h1) != h2 or twin_map.get(h2) != h1:
            raise ValueError(
                f"{graph_name}: label/twin mismatch {edge_label!r}"
            )

        for half_edge in (h1, h2):
            if half_edge in half_edge_labels:
                raise ValueError(
                    f"{graph_name}: multiple labels {half_edge}"
                )
            half_edge_labels[half_edge] = edge_label

    if set(half_edge_labels) != set(twin_map):
        raise ValueError(
            f"{graph_name}: label/twin domain mismatch"
        )

    return half_edge_labels


def _extract_faces_and_label_signatures(
    face_cycles: tuple[tuple[HalfEdge, ...], ...],
    half_edge_labels: dict[HalfEdge, EdgeLabel],
    *,
    graph_name: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[LabelSignature, ...]]:
    """Extract face vertex cycles and label signatures from face half-edge cycles."""
    faces: list[tuple[int, ...]] = []
    signatures: list[LabelSignature] = []

    for face_cycle in face_cycles:
        face_labels: list[EdgeLabel] = []
        for half_edge in face_cycle:
            label = half_edge_labels.get(half_edge)
            if label is None:
                raise ValueError(
                    f"{graph_name}: unlabeled half-edge {half_edge}"
                )
            face_labels.append(label)
        faces.append(tuple(vertex for vertex, _ in face_cycle))
        signatures.append(_label_signature(face_labels))

    return tuple(faces), tuple(signatures)


def _vertex_label_signatures(
    embedding: dict[int, tuple[int, ...]],
    half_edge_labels: dict[HalfEdge, EdgeLabel],
    *,
    vertex_count: int,
    graph_name: str,
) -> tuple[LabelSignature, ...]:
    """Build edge-label signatures for all vertices in index order."""
    signatures: list[LabelSignature] = []
    for v in range(vertex_count):
        neighbors = embedding.get(v)
        if neighbors is None:
            raise ValueError(
                f"{graph_name}: missing embedding vertex {v}"
            )
        labels: list[EdgeLabel] = []
        for i in range(len(neighbors)):
            half_edge = (v, i)
            label = half_edge_labels.get(half_edge)
            if label is None:
                raise ValueError(
                    f"{graph_name}: unlabeled half-edge {half_edge}"
                )
            labels.append(label)
        signatures.append(_label_signature(labels))
    return tuple(signatures)


def _prepare_section_data(
    section_data: ParsedGraphSection,
    *,
    graph_name: str,
) -> _PreparedSectionData:
    """Convert one parsed double_code section to zero-based builder data."""
    embedding = GraphConverter.to_zero_based_embedding(section_data.cyclic_adjacency)
    # plantri vertex ids are 1-based, while cyclic-order slots are already 0-based.
    twin_map = {
        (v - 1, i): (u - 1, j)
        for (v, i), (u, j) in section_data.twin_map.items()
    }
    edge_label_pairs = {
        edge_label: ((u1 - 1, i1), (u2 - 1, i2))
        for edge_label, ((u1, i1), (u2, i2)) in section_data.edge_label_pairs.items()
    }
    half_edge_labels = _build_half_edge_label_map(
        edge_label_pairs,
        twin_map,
        graph_name=graph_name,
    )
    return _PreparedSectionData(
        vertex_count=section_data.vertex_count,
        embedding=embedding,
        twin_map=twin_map,
        edge_label_pairs=edge_label_pairs,
        half_edge_labels=half_edge_labels,
    )


def _dense_embedding(
    embedding: dict[int, tuple[int, ...]],
    *,
    vertex_count: int,
    graph_name: str,
) -> tuple[tuple[int, ...], ...]:
    """Convert a complete zero-based adjacency map to its canonical tuple."""
    expected_vertices = set(range(vertex_count))
    actual_vertices = set(embedding)
    if actual_vertices != expected_vertices:
        raise ValueError(
            f"{graph_name}: embedding vertices {tuple(sorted(actual_vertices))!r}"
        )
    return tuple(embedding[vertex] for vertex in range(vertex_count))


def _edge_multiplicity_from_edge_label_pairs(
    edge_label_pairs: EdgeLabelPairs,
) -> dict[tuple[int, int], int]:
    """Build support-edge multiplicities from edge-labeled half-edge pairs."""
    edge_multiplicity: dict[tuple[int, int], int] = {}
    for half_edge_a, half_edge_b in edge_label_pairs.values():
        vertex_a, _ = half_edge_a
        vertex_b, _ = half_edge_b
        edge: tuple[int, int] = (vertex_a, vertex_b) if vertex_a <= vertex_b else (vertex_b, vertex_a)
        edge_multiplicity[edge] = edge_multiplicity.get(edge, 0) + 1
    return edge_multiplicity


def _build_plane_graph_from_sections(
    primal_data: ParsedGraphSection,
    dual_data: ParsedGraphSection,
    graph_id: int,
    *,
    include_primal: bool = True,
) -> PlaneGraph:
    """Build PlaneGraph from parsed primal/dual double_code sections."""
    dual = _prepare_section_data(dual_data, graph_name="dual")
    dual_vertex_count = dual.vertex_count
    edge_multiplicity = _edge_multiplicity_from_edge_label_pairs(dual.edge_label_pairs)
    edges = tuple(sorted(edge_multiplicity))
    dual_face_cycles = GraphConverter.extract_face_half_edge_cycles(dual.embedding, dual.twin_map, graph_name="dual")
    faces, dual_face_label_signatures = _extract_faces_and_label_signatures(dual_face_cycles, dual.half_edge_labels, graph_name="dual")

    primal_num_vertices = 0
    primal_embedding: dict[int, tuple[int, ...]] = {}
    primal_faces: tuple[tuple[int, ...], ...] = tuple()
    primal_edge_label_entries: tuple[tuple[EdgeLabel, HalfEdge, HalfEdge], ...] = tuple()
    dual_vertex_to_primal_face: tuple[int, ...] = tuple()
    primal_vertex_to_dual_face: tuple[int, ...] = tuple()

    if include_primal:
        primal = _prepare_section_data(primal_data, graph_name="primal")
        primal_num_vertices = primal.vertex_count
        primal_embedding = primal.embedding
        primal_edge_label_entries = tuple(
            (edge_label, half_edge_a, half_edge_b)
            for edge_label, (half_edge_a, half_edge_b) in primal.edge_label_pairs.items()
        )
        primal_face_cycles = GraphConverter.extract_face_half_edge_cycles(primal.embedding, primal.twin_map, graph_name="primal")
        primal_faces, primal_face_label_signatures = _extract_faces_and_label_signatures(primal_face_cycles, primal.half_edge_labels, graph_name="primal")
        dual_vertex_label_signatures = _vertex_label_signatures(
            dual.embedding,
            dual.half_edge_labels,
            vertex_count=dual_vertex_count,
            graph_name="dual",
        )
        primal_vertex_label_signatures = _vertex_label_signatures(
            primal.embedding,
            primal.half_edge_labels,
            vertex_count=primal_num_vertices,
            graph_name="primal",
        )
        # Match vertices to faces by oriented cyclic edge labels, up to rotation.
        dual_vertex_to_primal_face = _match_label_signatures(
            dual_vertex_label_signatures,
            primal_face_label_signatures,
            source_name="dual vertex",
            target_name="primal face",
        )
        primal_vertex_to_dual_face = _match_label_signatures(
            primal_vertex_label_signatures,
            dual_face_label_signatures,
            source_name="primal vertex",
            target_name="dual face",
        )

    normalized_embedding = _dense_embedding(
        dual.embedding,
        vertex_count=dual_vertex_count,
        graph_name="dual",
    )
    normalized_primal_embedding = _dense_embedding(
        primal_embedding,
        vertex_count=primal_num_vertices,
        graph_name="primal",
    )
    return PlaneGraph(
        dual_num_vertices=dual_vertex_count,
        dual_support_edges=edges,
        dual_edge_multiplicity=edge_multiplicity,
        dual_edge_label_pairs=tuple(
            (edge_label, half_edge_a, half_edge_b)
            for edge_label, (half_edge_a, half_edge_b) in dual.edge_label_pairs.items()
        ),
        dual_embedding=normalized_embedding,
        dual_faces=faces,
        primal_num_vertices=primal_num_vertices,
        primal_edge_label_pairs=primal_edge_label_entries,
        primal_embedding=normalized_primal_embedding,
        primal_faces=primal_faces,
        dual_vertex_to_primal_face=dual_vertex_to_primal_face,
        primal_vertex_to_dual_face=primal_vertex_to_dual_face,
        graph_id=graph_id,
    )
