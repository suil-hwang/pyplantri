# src/pyplantri/plane_graph_enumeration.py
from __future__ import annotations

import multiprocessing
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .core import GraphConverter, Plantri, QuadrangulationEnumerator

if TYPE_CHECKING:
    from .plane_graph import PlaneGraph


EdgeLabel = Union[str, int]
HalfEdge = Tuple[int, int]
EdgeLabelPairs = Dict[EdgeLabel, Tuple[HalfEdge, HalfEdge]]
LabelSignature = Tuple[Tuple[str, int], ...]


@lru_cache(maxsize=1)
def _get_plane_graph_cls():
    """Resolve PlaneGraph lazily to avoid import cycles."""
    from .plane_graph import PlaneGraph

    return PlaneGraph


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
    primal_data: Dict,
    dual_data: Dict,
    graph_id: int,
    *,
    include_primal: bool = True,
) -> PlaneGraph:
    """Builds PlaneGraph from primal and dual data."""
    dual_vertex_count = dual_data["vertex_count"]
    dual_adj_1based = dual_data["adjacency_list"]
    twin_map_1based = dual_data.get("twin_map", {})
    dual_edge_label_pairs_1based: EdgeLabelPairs = dual_data.get("edge_label_pairs", {})

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
        faces = GraphConverter.extract_faces(
            embedding,
            twin_map=twin_map_0based,
        )
        dual_face_label_signatures = tuple()

    primal_num_vertices = 0
    primal_embedding: Dict[int, Tuple[int, ...]] = {}
    primal_faces: Tuple[Tuple[int, ...], ...] = tuple()
    dual_vertex_to_primal_face: Tuple[int, ...] = tuple()
    primal_vertex_to_dual_face: Tuple[int, ...] = tuple()

    if include_primal:
        primal_adj_1based = primal_data["adjacency_list"]
        primal_num_vertices = primal_data["vertex_count"]
        primal_twin_map_1based = primal_data.get("twin_map", {})
        primal_edge_label_pairs_1based: EdgeLabelPairs = primal_data.get(
            "edge_label_pairs", {}
        )
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

    plane_graph_cls = _get_plane_graph_cls()
    normalized_embedding = plane_graph_cls._normalize_embedding(
        embedding,
        expected_size=dual_vertex_count,
    )
    normalized_primal_embedding = plane_graph_cls._normalize_embedding(
        primal_embedding,
        expected_size=primal_num_vertices,
    )
    return plane_graph_cls(
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


def enumerate_plane_graphs(
    dual_vertex_count: int,
    max_count: Optional[int] = None,
    validate: bool = True,
    verbose: bool = False,
    include_primal: bool = True,
) -> List[PlaneGraph]:
    """Enumerates all n-vertex 4-regular planar multigraphs."""
    if verbose:
        print(f"[Plantri] Enumerating {dual_vertex_count}-vertex 4-regular planar multigraphs...")

    enumerator = QuadrangulationEnumerator()
    graphs: List[PlaneGraph] = []

    for graph_id, (primal_data, dual_data) in enumerate(enumerator.generate_pairs(dual_vertex_count)):
        graph = _build_plane_graph(
            primal_data,
            dual_data,
            graph_id,
            include_primal=include_primal,
        )

        if validate:
            is_valid, errors = graph.validate()
            if not is_valid:
                if verbose:
                    print(f"  [!] Graph {graph_id} validation failed: {errors}")
                continue

        graphs.append(graph)

        if verbose and (graph_id + 1) % 100 == 0:
            print(f"  Processed {graph_id + 1} graphs...")

        if max_count and len(graphs) >= max_count:
            break

    if verbose:
        print(f"[Plantri] Found {len(graphs)} valid graphs")

    return graphs


def iter_plane_graphs(
    dual_vertex_count: int,
    validate: bool = True,
    include_primal: bool = True,
) -> Iterator[PlaneGraph]:
    """Iterates over PlaneGraph objects."""
    enumerator = QuadrangulationEnumerator()

    for graph_id, (primal_data, dual_data) in enumerate(
        enumerator.generate_pairs(dual_vertex_count)
    ):
        graph = _build_plane_graph(
            primal_data,
            dual_data,
            graph_id,
            include_primal=include_primal,
        )

        if validate:
            is_valid, _ = graph.validate()
            if not is_valid:
                continue

        yield graph


def _close_if_possible(obj: object) -> None:
    """Close an iterator/generator if it exposes a close() method."""
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _iter_raw_double_code_lines(
    raw_lines: Iterable[Union[str, bytes]],
) -> Iterator[bytes]:
    """Yield valid double_code lines from a raw line stream."""
    raw_iter = iter(raw_lines)
    try:
        for line in raw_iter:
            if isinstance(line, bytes):
                stripped = line.strip()
            else:
                stripped = line.strip().encode("latin-1")
            if stripped and 48 <= stripped[0] <= 57:
                yield stripped
    finally:
        _close_if_possible(raw_iter)


def _iter_prefixed_lines(
    prefix: List[bytes],
    raw_lines: Iterable[bytes],
) -> Iterator[bytes]:
    """Yield prefetched lines first, then continue streaming from raw_lines."""
    raw_iter = iter(raw_lines)
    try:
        for line in prefix:
            yield line
        for line in raw_iter:
            yield line
    finally:
        _close_if_possible(raw_iter)


def _iter_chunk_args(
    raw_lines: Iterable[bytes],
    *,
    chunk_size: int,
    validate: bool,
    include_primal: bool,
) -> Iterator[Tuple[List[bytes], int, bool, bool]]:
    """Create chunk arguments lazily from a raw line stream."""
    raw_iter = iter(raw_lines)
    start_id = 0
    chunk: List[bytes] = []

    try:
        for line in raw_iter:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                current_chunk = chunk
                yield (current_chunk, start_id, validate, include_primal)
                start_id += len(current_chunk)
                chunk = []
        if chunk:
            yield (chunk, start_id, validate, include_primal)
    finally:
        _close_if_possible(raw_iter)


def _has_digon_from_dual_adjacency(dual_adjacency: Dict[int, List[int]]) -> bool:
    """Return True if dual adjacency has any parallel edge (digon)."""
    half_edge_counts: Dict[Tuple[int, int], int] = {}
    for vertex, neighbors in dual_adjacency.items():
        for neighbor in neighbors:
            edge = (vertex, neighbor) if vertex <= neighbor else (neighbor, vertex)
            count = half_edge_counts.get(edge, 0) + 1
            if count >= 4:
                return True
            half_edge_counts[edge] = count
    return False


def _default_parallel_workers(raw_line_count: Optional[int] = None) -> int:
    """Choose a conservative worker count to reduce spawn overhead."""
    cpu_count = os.cpu_count() or 4
    if raw_line_count is None:
        target = 8
    elif raw_line_count < 120_000:
        target = 4
    elif raw_line_count < 600_000:
        target = 8
    else:
        target = 12
    return max(2, min(cpu_count, target))


def _default_chunk_size(raw_line_count: Optional[int] = None) -> int:
    """Choose chunk size for balanced IPC overhead and parallelism."""
    if raw_line_count is None:
        return 5_000
    if raw_line_count < 50_000:
        return 2_000
    if raw_line_count < 600_000:
        return 5_000
    return 10_000


def _build_graphs_from_raw_lines(
    raw_lines: Iterable[Union[str, bytes]],
    *,
    max_count: Optional[int],
    validate: bool,
    include_primal: bool,
    digon_zero_only: bool = False,
) -> Tuple[List[PlaneGraph], int]:
    """Build PlaneGraph objects from raw double_code lines."""
    graphs: List[PlaneGraph] = []
    generated_count = 0

    raw_iter = iter(raw_lines)
    try:
        for graph_id, line in enumerate(raw_iter):
            generated_count += 1

            primal_data, dual_data = QuadrangulationEnumerator.parse_double_code(line)
            if not primal_data["adjacency_list"] or not dual_data["adjacency_list"]:
                continue

            if digon_zero_only and _has_digon_from_dual_adjacency(dual_data["adjacency_list"]):
                continue

            graph = _build_plane_graph(
                primal_data,
                dual_data,
                graph_id,
                include_primal=include_primal,
            )

            if validate:
                is_valid, _ = graph.validate()
                if not is_valid:
                    continue

            graphs.append(graph)
            if max_count and len(graphs) >= max_count:
                break
    finally:
        _close_if_possible(raw_iter)

    return graphs, generated_count


def _process_graph_chunk(
    args: Tuple[List[bytes], int, bool, bool]
) -> List[PlaneGraph]:
    """Process a chunk of raw lines into PlaneGraph objects."""
    lines, start_id, validate, include_primal = args
    graphs: List[PlaneGraph] = []

    for i, line in enumerate(lines):
        graph_id = start_id + i
        try:
            primal_data, dual_data = QuadrangulationEnumerator.parse_double_code(line)
            if not primal_data["adjacency_list"] or not dual_data["adjacency_list"]:
                continue

            graph = _build_plane_graph(
                primal_data,
                dual_data,
                graph_id,
                include_primal=include_primal,
            )

            if validate:
                is_valid, _ = graph.validate()
                if not is_valid:
                    continue

            graphs.append(graph)
        except (ValueError, RuntimeError, KeyError, IndexError):
            continue  # Skip malformed graph lines and continue chunk processing

    return graphs


@dataclass
class EnumerationTiming:
    """Timing breakdown for enumeration."""

    plantri_s: float
    parse_build_s: float
    total_s: float
    graph_count: int


@dataclass
class FilteredEnumerationResult:
    """Enumeration result with source count and timing details."""

    graphs: List[PlaneGraph]
    generated_count: int
    timing: EnumerationTiming


def enumerate_plane_graphs_filtered(
    dual_vertex_count: int,
    *,
    max_count: Optional[int] = None,
    validate: bool = True,
    include_primal: bool = True,
    digon_zero_only: bool = False,
    verbose: bool = False,
) -> FilteredEnumerationResult:
    """Enumerate plane graphs with optional filtering in a single pass."""
    import time

    t_start = time.perf_counter()
    primal_vertex_count = dual_vertex_count + 2
    plantri = Plantri()
    raw_stream = _iter_raw_double_code_lines(
        plantri.iter_stdout_lines(
            primal_vertex_count,
            QuadrangulationEnumerator.OPTIONS,
        )
    )
    raw_iter = iter(raw_stream)

    prefetched: List[bytes] = []
    try:
        prefetched.append(next(raw_iter))
    except StopIteration:
        pass
    # In streaming mode, plantri generation and parsing overlap.
    # Keep plantri_s as startup latency until first valid raw line.
    t_plantri = time.perf_counter() - t_start

    graphs, generated_count = _build_graphs_from_raw_lines(
        _iter_prefixed_lines(prefetched, raw_iter),
        max_count=max_count,
        validate=validate,
        include_primal=include_primal,
        digon_zero_only=digon_zero_only,
    )
    t_total = time.perf_counter() - t_start
    timing = EnumerationTiming(
        plantri_s=t_plantri,
        parse_build_s=t_total - t_plantri,
        total_s=t_total,
        graph_count=len(graphs),
    )

    if verbose:
        mode = "digon=0 " if digon_zero_only else ""
        print(
            f"[Plantri] Filtered enumeration ({mode}n={dual_vertex_count}): "
            f"{len(graphs)}/{generated_count} graphs"
        )

    return FilteredEnumerationResult(
        graphs=graphs,
        generated_count=generated_count,
        timing=timing,
    )


def enumerate_plane_graphs_parallel(
    dual_vertex_count: int,
    max_count: Optional[int] = None,
    validate: bool = True,
    verbose: bool = False,
    n_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    include_primal: bool = True,
    return_timing: bool = False,
) -> Union[List[PlaneGraph], Tuple[List[PlaneGraph], EnumerationTiming]]:
    """Parallel enumeration of plane graphs."""
    import time

    t_start = time.perf_counter()

    # Step 1: Start streaming plantri output.
    primal_vertex_count = dual_vertex_count + 2
    plantri = Plantri()
    raw_stream = _iter_raw_double_code_lines(
        plantri.iter_stdout_lines(
            primal_vertex_count,
            QuadrangulationEnumerator.OPTIONS,
        )
    )
    raw_iter = iter(raw_stream)

    prefetched: List[bytes] = []
    try:
        prefetched.append(next(raw_iter))
    except StopIteration:
        pass

    # In streaming mode, plantri generation and parsing overlap.
    # Keep plantri_s as startup latency until first valid raw line.
    t_plantri = time.perf_counter() - t_start

    if n_workers is None:
        n_workers = _default_parallel_workers()
    effective_chunk_size = (
        chunk_size if chunk_size is not None else _default_chunk_size()
    )

    if not prefetched:
        if return_timing:
            t_total = time.perf_counter() - t_start
            timing = EnumerationTiming(
                plantri_s=t_plantri,
                parse_build_s=t_total - t_plantri,
                total_s=t_total,
                graph_count=0,
            )
            return [], timing
        return []

    if verbose:
        print(
            f"[Plantri] Parallel enumeration (n={dual_vertex_count}, "
            f"workers={n_workers}, chunk={effective_chunk_size})..."
        )

    # Bounded warmup to decide if we should stay sequential for small inputs.
    warmup_limit = effective_chunk_size * 2
    while len(prefetched) < warmup_limit:
        try:
            prefetched.append(next(raw_iter))
        except StopIteration:
            break

    if verbose:
        if len(prefetched) < warmup_limit:
            print(f"[Plantri] {len(prefetched)} raw graphs from plantri (small input)")
        else:
            print(
                f"[Plantri] Streaming mode enabled "
                f"(observed first {len(prefetched)}+ raw graphs)"
            )

    # For small inputs, use sequential processing to avoid multiprocessing overhead.
    if len(prefetched) < warmup_limit or n_workers <= 1:
        if verbose:
            print("[Plantri] Using sequential processing (small input)")
        graphs, _ = _build_graphs_from_raw_lines(
            _iter_prefixed_lines(prefetched, raw_iter),
            max_count=max_count,
            validate=validate,
            include_primal=include_primal,
            digon_zero_only=False,
        )
        if return_timing:
            t_total = time.perf_counter() - t_start
            timing = EnumerationTiming(
                plantri_s=t_plantri,
                parse_build_s=t_total - t_plantri,
                total_s=t_total,
                graph_count=len(graphs),
            )
            return graphs, timing
        return graphs

    # Step 2: Stream chunks directly to worker pool.
    all_graphs: List[PlaneGraph] = []
    chunk_args = _iter_chunk_args(
        _iter_prefixed_lines(prefetched, raw_iter),
        chunk_size=effective_chunk_size,
        validate=validate,
        include_primal=include_primal,
    )

    # Use spawn context for Windows compatibility
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        try:
            for chunk_graphs in pool.imap(_process_graph_chunk, chunk_args):
                all_graphs.extend(chunk_graphs)
                if max_count and len(all_graphs) >= max_count:
                    all_graphs = all_graphs[:max_count]
                    break
        finally:
            _close_if_possible(chunk_args)

    # Re-assign graph_ids sequentially
    plane_graph_cls = _get_plane_graph_cls()
    for i, graph in enumerate(all_graphs):
        # PlaneGraph is frozen, so we need to create new instances
        all_graphs[i] = plane_graph_cls(
            num_vertices=graph.num_vertices,
            edges=graph.edges,
            edge_multiplicity=graph.edge_multiplicity,
            embedding=graph.embedding,
            faces=graph.faces,
            primal_num_vertices=graph.primal_num_vertices,
            primal_embedding=graph.primal_embedding,
            primal_faces=graph.primal_faces,
            dual_vertex_to_primal_face=graph.dual_vertex_to_primal_face,
            primal_vertex_to_dual_face=graph.primal_vertex_to_dual_face,
            graph_id=i,
        )

    if verbose:
        print(f"[Plantri] Found {len(all_graphs)} valid graphs")

    if return_timing:
        t_total = time.perf_counter() - t_start
        timing = EnumerationTiming(
            plantri_s=t_plantri,
            parse_build_s=t_total - t_plantri,
            total_s=t_total,
            graph_count=len(all_graphs),
        )
        return all_graphs, timing

    return all_graphs
