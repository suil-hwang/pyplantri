# # src/pyplantri/builder.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .converter import GraphConverter
from .plane_graph import PlaneGraph
from .plantri import ParsedGraphSection
from .types import EdgeLabel, EdgeLabelPairs, HalfEdge
LabelSignature = Tuple[Tuple[str, int], ...]


def _to_zero_based_twin_map(
    twin_map_1based: Dict[Tuple[int, int], Tuple[int, int]],
    embedding: Dict[int, Tuple[int, ...]],
    *,
    graph_name: str,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Convert and validate twin_map completeness for -T based enumeration."""
    if not twin_map_1based:
        raise ValueError(
            f"{graph_name} twin_map is missing. "
            "Current plantri pipeline requires -T double_code with full twin labels."
        )

    twin_map_0based: Dict[Tuple[int, int], Tuple[int, int]] = {
        (v - 1, i): (u - 1, j)
        for (v, i), (u, j) in twin_map_1based.items()
    }

    expected_half_edges = sum(len(neighbors) for neighbors in embedding.values())
    if len(twin_map_0based) != expected_half_edges:
        raise ValueError(
            f"{graph_name} twin_map size mismatch: "
            f"{len(twin_map_0based)} entries for {expected_half_edges} half-edges"
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


def _build_half_edge_label_map(edge_label_pairs: EdgeLabelPairs) -> Dict[HalfEdge, EdgeLabel]:
    """Build half-edge -> edge-label mapping from edge-label pair map."""
    half_edge_labels: Dict[HalfEdge, EdgeLabel] = {}
    for edge_label, (h1, h2) in edge_label_pairs.items():
        prev = half_edge_labels.get(h1)
        if prev is not None and prev != edge_label:
            raise ValueError(
                f"Conflicting labels for half-edge {h1}: {prev!r} vs {edge_label!r}"
            )
        half_edge_labels[h1] = edge_label

        prev = half_edge_labels.get(h2)
        if prev is not None and prev != edge_label:
            raise ValueError(
                f"Conflicting labels for half-edge {h2}: {prev!r} vs {edge_label!r}"
            )
        half_edge_labels[h2] = edge_label
    return half_edge_labels


def _edge_label_token(edge_label: EdgeLabel) -> str:
    """Normalize edge-label key for multiset signature matching."""
    if isinstance(edge_label, int):
        return f"i:{edge_label}"
    return f"s:{edge_label}"


def _label_signature(labels: Iterable[EdgeLabel]) -> LabelSignature:
    """Convert edge-label multiset into a canonical signature tuple."""
    counts: Dict[str, int] = defaultdict(int)
    for edge_label in labels:
        counts[_edge_label_token(edge_label)] += 1
    return tuple(sorted(counts.items()))


def _extract_faces_and_label_signatures(
    embedding: Dict[int, Tuple[int, ...]],
    twin_map: Dict[HalfEdge, HalfEdge],
    half_edge_labels: Dict[HalfEdge, EdgeLabel],
    *,
    graph_name: str,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[LabelSignature, ...]]:
    """Extract faces and edge-label signatures from half-edge traversal."""
    visited: set[HalfEdge] = set()
    faces: List[Tuple[int, ...]] = []
    signatures: List[LabelSignature] = []

    if not embedding:
        return tuple(), tuple()

    max_deg = max((len(neighbors) for neighbors in embedding.values()), default=0)
    max_iterations = max(1, len(embedding) * max_deg)

    for v in sorted(embedding.keys()):
        deg_v = len(embedding[v])
        for i in range(deg_v):
            if (v, i) in visited:
                continue

            face: List[int] = []
            face_labels: List[EdgeLabel] = []
            curr_v, curr_i = v, i
            iterations = 0

            while (curr_v, curr_i) not in visited:
                iterations += 1
                if iterations > max_iterations:
                    raise RuntimeError(
                        f"{graph_name} face traversal exceeded {max_iterations} iterations. "
                        "Possible invalid twin_map or embedding."
                    )

                half_edge = (curr_v, curr_i)
                label = half_edge_labels.get(half_edge)
                if label is None:
                    raise ValueError(
                        f"{graph_name} half-edge {half_edge} has no edge label."
                    )

                visited.add(half_edge)
                face.append(curr_v)
                face_labels.append(label)

                twin = twin_map.get(half_edge)
                if twin is None:
                    raise ValueError(
                        f"{graph_name} half-edge {half_edge} missing twin_map entry."
                    )
                twin_v, twin_i = twin

                deg = len(embedding[twin_v])
                curr_v = twin_v
                curr_i = (twin_i - 1) % deg

            if len(face) >= 2:
                faces.append(tuple(face))
                signatures.append(_label_signature(face_labels))

    return tuple(faces), tuple(signatures)


def _vertex_label_signatures(
    embedding: Dict[int, Tuple[int, ...]],
    half_edge_labels: Dict[HalfEdge, EdgeLabel],
    *,
    vertex_count: int,
    graph_name: str,
) -> Tuple[LabelSignature, ...]:
    """Build edge-label signatures for all vertices in index order."""
    signatures: List[LabelSignature] = []
    for v in range(vertex_count):
        neighbors = embedding.get(v)
        if neighbors is None:
            raise ValueError(f"{graph_name} embedding missing vertex {v}.")
        labels: List[EdgeLabel] = []
        for i in range(len(neighbors)):
            half_edge = (v, i)
            label = half_edge_labels.get(half_edge)
            if label is None:
                raise ValueError(
                    f"{graph_name} half-edge {half_edge} has no edge label."
                )
            labels.append(label)
        signatures.append(_label_signature(labels))
    return tuple(signatures)


def _match_label_signatures(
    source_signatures: Tuple[LabelSignature, ...],
    target_signatures: Tuple[LabelSignature, ...],
    *,
    source_name: str,
    target_name: str,
) -> Tuple[int, ...]:
    """Match source entities to target entities by edge-label multiset signature."""
    if len(source_signatures) != len(target_signatures):
        raise ValueError(
            f"Cannot map {source_name} to {target_name}: "
            f"count mismatch {len(source_signatures)} vs {len(target_signatures)}."
        )

    target_by_signature: Dict[LabelSignature, List[int]] = defaultdict(list)
    for target_idx, signature in enumerate(target_signatures):
        target_by_signature[signature].append(target_idx)

    used_targets: set[int] = set()
    mapping: List[int] = []
    for source_idx, signature in enumerate(source_signatures):
        candidates = [
            target_idx
            for target_idx in target_by_signature.get(signature, [])
            if target_idx not in used_targets
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely map {source_name} {source_idx} "
                f"to {target_name} using edge-label signature."
            )
        target_idx = candidates[0]
        used_targets.add(target_idx)
        mapping.append(target_idx)

    if len(used_targets) != len(target_signatures):
        raise ValueError(
            f"Mapping {source_name}->{target_name} is not bijective."
        )

    return tuple(mapping)


def _build_plane_graph(
    primal_data: ParsedGraphSection,
    dual_data: ParsedGraphSection,
    graph_id: int,
    *,
    include_primal: bool = True,
) -> PlaneGraph:
    """Builds PlaneGraph from primal and dual data."""
    dual_vertex_count = dual_data.vertex_count
    dual_adj_1based = dual_data.adjacency_list
    twin_map_1based = dual_data.twin_map
    dual_edge_label_pairs_1based: EdgeLabelPairs = dual_data.edge_label_pairs

    embedding = GraphConverter.to_zero_based_embedding(dual_adj_1based)
    twin_map_0based = _to_zero_based_twin_map(
        twin_map_1based,
        embedding,
        graph_name="dual",
    )
    dual_edge_label_pairs_0based = _to_zero_based_edge_label_pairs(
        dual_edge_label_pairs_1based
    )
    dual_half_edge_labels = _build_half_edge_label_map(dual_edge_label_pairs_0based)

    edge_multiplicity_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for u, neighbors in embedding.items():
        for v in neighbors:
            if u <= v:
                edge_multiplicity_counts[(u, v)] += 1

    edge_multiplicity: Dict[Tuple[int, int], int] = dict(edge_multiplicity_counts)
    edges = tuple(sorted(edge_multiplicity.keys()))
    if dual_half_edge_labels:
        faces, dual_face_label_signatures = _extract_faces_and_label_signatures(
            embedding,
            twin_map_0based,
            dual_half_edge_labels,
            graph_name="dual",
        )
    else:
        faces = GraphConverter.extract_faces_with_twins(
            embedding,
            twin_map_0based,
        )
        dual_face_label_signatures = tuple()

    primal_num_vertices = 0
    primal_embedding: Dict[int, Tuple[int, ...]] = {}
    primal_faces: Tuple[Tuple[int, ...], ...] = tuple()
    dual_vertex_to_primal_face: Tuple[int, ...] = tuple()
    primal_vertex_to_dual_face: Tuple[int, ...] = tuple()

    if include_primal:
        primal_adj_1based = primal_data.adjacency_list
        primal_num_vertices = primal_data.vertex_count
        primal_twin_map_1based = primal_data.twin_map
        primal_edge_label_pairs_1based: EdgeLabelPairs = primal_data.edge_label_pairs
        primal_embedding = GraphConverter.to_zero_based_embedding(primal_adj_1based)
        primal_twin_map_0based = _to_zero_based_twin_map(
            primal_twin_map_1based,
            primal_embedding,
            graph_name="primal",
        )
        primal_edge_label_pairs_0based = _to_zero_based_edge_label_pairs(
            primal_edge_label_pairs_1based
        )
        primal_half_edge_labels = _build_half_edge_label_map(
            primal_edge_label_pairs_0based
        )
        if not dual_half_edge_labels or not primal_half_edge_labels:
            raise ValueError(
                "Missing edge labels in -T output. Cannot build dual/primal mapping."
            )

        primal_faces, primal_face_label_signatures = _extract_faces_and_label_signatures(
            primal_embedding,
            primal_twin_map_0based,
            primal_half_edge_labels,
            graph_name="primal",
        )
        dual_vertex_label_signatures = _vertex_label_signatures(
            embedding,
            dual_half_edge_labels,
            vertex_count=dual_vertex_count,
            graph_name="dual",
        )
        primal_vertex_label_signatures = _vertex_label_signatures(
            primal_embedding,
            primal_half_edge_labels,
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
        embedding,
        expected_size=dual_vertex_count,
    )
    normalized_primal_embedding = PlaneGraph._normalize_embedding(
        primal_embedding,
        expected_size=primal_num_vertices,
    )
    return PlaneGraph(
        num_vertices=dual_vertex_count,
        edges=edges,
        edge_multiplicity=edge_multiplicity,
        embedding=normalized_embedding,
        faces=faces,
        primal_num_vertices=primal_num_vertices,
        primal_embedding=normalized_primal_embedding,
        primal_faces=primal_faces,
        dual_vertex_to_primal_face=dual_vertex_to_primal_face,
        primal_vertex_to_dual_face=primal_vertex_to_dual_face,
        graph_id=graph_id,
    )
