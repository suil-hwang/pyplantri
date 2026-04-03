# src/pyplantri/builder.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .converter import GraphConverter
from .plane_graph import PlaneGraph
from .plantri import ParsedGraphSection
from .types import EdgeLabel, EdgeLabelPairs, HalfEdge

LabelSignature = tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _PreparedSectionData:
    vertex_count: int
    embedding: dict[int, tuple[int, ...]]
    twin_map: dict[HalfEdge, HalfEdge]
    edge_label_pairs: EdgeLabelPairs
    half_edge_labels: dict[HalfEdge, EdgeLabel]


def _to_zero_based_twin_map(
    twin_map_1based: dict[tuple[int, int], tuple[int, int]],
    embedding: dict[int, tuple[int, ...]],
    *,
    graph_name: str,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Convert and validate twin_map completeness for -T based enumeration."""
    if not twin_map_1based:
        raise ValueError(f"{graph_name} twin_map missing")

    twin_map_0based: dict[tuple[int, int], tuple[int, int]] = {
        (v - 1, i): (u - 1, j)
        for (v, i), (u, j) in twin_map_1based.items()
    }

    expected_half_edges = sum(len(neighbors) for neighbors in embedding.values())
    if len(twin_map_0based) != expected_half_edges:
        raise ValueError(
            f"{graph_name} twin_map size mismatch: {len(twin_map_0based)} != {expected_half_edges}"
        )

    GraphConverter.validate_twin_map(
        embedding,
        twin_map_0based,
        graph_name=graph_name,
    )

    return twin_map_0based


def _to_zero_based_edge_label_pairs(
    edge_label_pairs_1based: EdgeLabelPairs,
) -> EdgeLabelPairs:
    """Convert edge-label to half-edge pair map from 1-based to 0-based."""
    return {
        edge_label: ((u1 - 1, i1), (u2 - 1, i2))
        for edge_label, ((u1, i1), (u2, i2)) in edge_label_pairs_1based.items()
    }


def _build_half_edge_label_map(edge_label_pairs: EdgeLabelPairs) -> dict[HalfEdge, EdgeLabel]:
    """Build half-edge -> edge-label mapping from edge-label pair map."""
    half_edge_labels: dict[HalfEdge, EdgeLabel] = {}
    for edge_label, (h1, h2) in edge_label_pairs.items():
        prev = half_edge_labels.get(h1)
        if prev is not None and prev != edge_label:
            raise ValueError(f"half-edge label conflict: {h1}")
        half_edge_labels[h1] = edge_label

        prev = half_edge_labels.get(h2)
        if prev is not None and prev != edge_label:
            raise ValueError(f"half-edge label conflict: {h2}")
        half_edge_labels[h2] = edge_label
    return half_edge_labels


def _edge_label_token(edge_label: EdgeLabel) -> str:
    """Normalize edge-label key for multiset signature matching."""
    if isinstance(edge_label, int):
        return f"i:{edge_label}"
    return f"s:{edge_label}"


def _label_signature(labels: Iterable[EdgeLabel]) -> LabelSignature:
    """Convert edge-label multiset into a canonical signature tuple."""
    counts: dict[str, int] = defaultdict(int)
    for edge_label in labels:
        counts[_edge_label_token(edge_label)] += 1
    return tuple(sorted(counts.items()))


def _extract_faces_and_label_signatures(
    face_cycles: tuple[tuple[HalfEdge, ...], ...],
    half_edge_labels: dict[HalfEdge, EdgeLabel],
    *,
    graph_name: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[LabelSignature, ...]]:
    """Extract face vertex cycles and label signatures from face half-edge cycles."""
    faces: list[tuple[int, ...]] = []
    signatures: list[LabelSignature] = []

    if not face_cycles:
        return tuple(), tuple()

    for face_cycle in face_cycles:
        face_labels: list[EdgeLabel] = []
        for half_edge in face_cycle:
            label = half_edge_labels.get(half_edge)
            if label is None:
                raise ValueError(f"{graph_name} half-edge unlabeled: {half_edge}")
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
            raise ValueError(f"{graph_name} embedding missing vertex: {v}")
        labels: list[EdgeLabel] = []
        for i in range(len(neighbors)):
            half_edge = (v, i)
            label = half_edge_labels.get(half_edge)
            if label is None:
                raise ValueError(f"{graph_name} half-edge unlabeled: {half_edge}")
            labels.append(label)
        signatures.append(_label_signature(labels))
    return tuple(signatures)


def _match_label_signatures(
    source_signatures: tuple[LabelSignature, ...],
    target_signatures: tuple[LabelSignature, ...],
    *,
    source_name: str,
    target_name: str,
) -> tuple[int, ...]:
    """Match entities by edge-label multiset signature."""
    if len(source_signatures) != len(target_signatures):
        raise ValueError(
            f"signature count mismatch: {source_name} -> {target_name} ({len(source_signatures)} != {len(target_signatures)})"
        )

    target_by_signature: dict[LabelSignature, list[int]] = defaultdict(list)
    for target_idx, signature in enumerate(target_signatures):
        target_by_signature[signature].append(target_idx)

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
                f"signature map ambiguous: {source_name} {source_idx} -> {target_name}"
            )
        target_idx = candidates[0]
        used_targets.add(target_idx)
        mapping.append(target_idx)

    if len(used_targets) != len(target_signatures):
        raise ValueError(f"signature map not bijective: {source_name} -> {target_name}")

    return tuple(mapping)


def _prepare_section_data(
    section_data: ParsedGraphSection,
    *,
    graph_name: str,
) -> _PreparedSectionData:
    """Convert one parsed double_code section to zero-based builder data."""
    embedding = GraphConverter.to_zero_based_embedding(section_data.cyclic_adjacency)
    twin_map = _to_zero_based_twin_map(
        section_data.twin_map,
        embedding,
        graph_name=graph_name,
    )
    edge_label_pairs = _to_zero_based_edge_label_pairs(section_data.edge_label_pairs)
    half_edge_labels = _build_half_edge_label_map(edge_label_pairs)
    return _PreparedSectionData(
        vertex_count=section_data.vertex_count,
        embedding=embedding,
        twin_map=twin_map,
        edge_label_pairs=edge_label_pairs,
        half_edge_labels=half_edge_labels,
    )


def _edge_multiplicity_from_edge_label_pairs(
    edge_label_pairs: EdgeLabelPairs,
) -> dict[tuple[int, int], int]:
    """Build support-edge multiplicities from edge-labeled half-edge pairs."""
    edge_multiplicity: dict[tuple[int, int], int] = {}
    for half_edge_a, half_edge_b in edge_label_pairs.values():
        vertex_a, _ = half_edge_a
        vertex_b, _ = half_edge_b
        edge: tuple[int, int] = (
            (vertex_a, vertex_b)
            if vertex_a <= vertex_b
            else (vertex_b, vertex_a)
        )
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
    edges = tuple(sorted(edge_multiplicity.keys()))
    dual_face_cycles = GraphConverter.extract_face_half_edge_cycles(
        dual.embedding,
        dual.twin_map,
        graph_name="dual",
    )
    faces, dual_face_label_signatures = _extract_faces_and_label_signatures(
        dual_face_cycles,
        dual.half_edge_labels,
        graph_name="dual",
    )

    primal_num_vertices = 0
    primal_embedding: dict[int, tuple[int, ...]] = {}
    primal_faces: tuple[tuple[int, ...], ...] = tuple()
    dual_vertex_to_primal_face: tuple[int, ...] = tuple()
    primal_vertex_to_dual_face: tuple[int, ...] = tuple()

    if include_primal:
        primal = _prepare_section_data(primal_data, graph_name="primal")
        primal_num_vertices = primal.vertex_count
        primal_embedding = primal.embedding
        primal_face_cycles = GraphConverter.extract_face_half_edge_cycles(
            primal.embedding,
            primal.twin_map,
            graph_name="primal",
        )
        primal_faces, primal_face_label_signatures = _extract_faces_and_label_signatures(
            primal_face_cycles,
            primal.half_edge_labels,
            graph_name="primal",
        )
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

    normalized_embedding = PlaneGraph._normalize_embedding(
        dual.embedding,
        expected_size=dual_vertex_count,
    )
    normalized_primal_embedding = PlaneGraph._normalize_embedding(
        primal_embedding,
        expected_size=primal_num_vertices,
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
        primal_edge_label_pairs=(
            tuple(
                (edge_label, half_edge_a, half_edge_b)
                for edge_label, (half_edge_a, half_edge_b) in primal.edge_label_pairs.items()
            )
            if include_primal
            else tuple()
        ),
        primal_embedding=normalized_primal_embedding,
        primal_faces=primal_faces,
        dual_vertex_to_primal_face=dual_vertex_to_primal_face,
        primal_vertex_to_dual_face=primal_vertex_to_dual_face,
        graph_id=graph_id,
    )
