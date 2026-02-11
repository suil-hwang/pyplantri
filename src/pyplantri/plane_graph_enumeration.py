# src/pyplantri/plane_graph_enumeration.py
from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .core import GraphConverter, Plantri, QuadrangulationEnumerator

if TYPE_CHECKING:
    from .plane_graph import PlaneGraph


@lru_cache(maxsize=1)
def _get_plane_graph_cls():
    """Resolve PlaneGraph lazily to avoid import cycles."""
    from .plane_graph import PlaneGraph

    return PlaneGraph


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

    embedding = GraphConverter.to_zero_based_embedding(dual_adj_1based)

    # Convert twin_map to 0-based indexing.
    # Note: vertex: 1-based -> 0-based (subtract 1)
    #       position: already 0-based, keep as-is
    twin_map_0based: Dict[Tuple[int, int], Tuple[int, int]] = {
        (v - 1, i): (u - 1, j)
        for (v, i), (u, j) in twin_map_1based.items()
    }

    edge_multiplicity: Dict[Tuple[int, int], int] = {}
    unique_edges = set()

    for v, neighbors in embedding.items():
        for u in neighbors:
            edge = (min(v, u), max(v, u))
            unique_edges.add(edge)

    for edge in unique_edges:
        u, v = edge
        count = embedding[u].count(v)
        edge_multiplicity[edge] = count

    edges = tuple(sorted(unique_edges))

    # Use position-based extraction if twin_map available, else fallback to old method.
    if twin_map_0based:
        faces = GraphConverter.extract_faces_with_twins(
            embedding, twin_map_0based
        )
    else:
        faces = GraphConverter.extract_faces(embedding, edge_multiplicity)

    primal_num_vertices = 0
    primal_embedding: Dict[int, Tuple[int, ...]] = {}
    primal_faces: Tuple[Tuple[int, ...], ...] = tuple()

    if include_primal:
        primal_adj_1based = primal_data["adjacency_list"]
        primal_num_vertices = primal_data["vertex_count"]
        primal_twin_map_1based = primal_data.get("twin_map", {})
        primal_embedding = GraphConverter.to_zero_based_embedding(primal_adj_1based)

        primal_twin_map_0based: Dict[Tuple[int, int], Tuple[int, int]] = {
            (v - 1, i): (u - 1, j)
            for (v, i), (u, j) in primal_twin_map_1based.items()
        }
        if primal_twin_map_0based:
            primal_faces = GraphConverter.extract_faces_with_twins(
                primal_embedding, primal_twin_map_0based
            )
        else:
            primal_faces = GraphConverter.extract_faces(primal_embedding)

    plane_graph_cls = _get_plane_graph_cls()
    return plane_graph_cls(
        num_vertices=dual_vertex_count,
        edges=edges,
        edge_multiplicity=edge_multiplicity,
        embedding=embedding,
        faces=faces,
        primal_num_vertices=primal_num_vertices,
        primal_embedding=primal_embedding,
        primal_faces=primal_faces,
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


def _iter_raw_double_code_lines(output: bytes) -> Iterator[str]:
    """Yield valid double_code lines from plantri raw output."""
    for line in output.decode(errors="replace").splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            yield stripped


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


def _default_parallel_workers(raw_line_count: int) -> int:
    """Choose a conservative worker count to reduce spawn overhead."""
    cpu_count = os.cpu_count() or 4
    if raw_line_count < 120_000:
        target = 4
    elif raw_line_count < 600_000:
        target = 8
    else:
        target = 12
    return max(2, min(cpu_count, target))


def _default_chunk_size(raw_line_count: int) -> int:
    """Choose chunk size for balanced IPC overhead and parallelism."""
    if raw_line_count < 50_000:
        return 2_000
    if raw_line_count < 600_000:
        return 5_000
    return 10_000


def _build_graphs_from_raw_lines(
    raw_lines: Iterable[str],
    *,
    max_count: Optional[int],
    validate: bool,
    include_primal: bool,
    digon_zero_only: bool = False,
) -> Tuple[List[PlaneGraph], int]:
    """Build PlaneGraph objects from raw double_code lines."""
    graphs: List[PlaneGraph] = []
    generated_count = 0

    for graph_id, line in enumerate(raw_lines):
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

    return graphs, generated_count


def _process_graph_chunk(args: Tuple[List[str], int, bool, bool]) -> List[PlaneGraph]:
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
    output = plantri.run(primal_vertex_count, QuadrangulationEnumerator.OPTIONS)
    t_plantri = time.perf_counter() - t_start

    graphs, generated_count = _build_graphs_from_raw_lines(
        _iter_raw_double_code_lines(output),
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

    # Step 1: Run plantri and collect raw output
    primal_vertex_count = dual_vertex_count + 2
    plantri = Plantri()
    output = plantri.run(primal_vertex_count, QuadrangulationEnumerator.OPTIONS)
    t_plantri = time.perf_counter() - t_start

    # Parse raw lines
    raw_lines = list(_iter_raw_double_code_lines(output))

    if n_workers is None:
        n_workers = _default_parallel_workers(len(raw_lines))
    effective_chunk_size = (
        chunk_size if chunk_size is not None else _default_chunk_size(len(raw_lines))
    )

    if not raw_lines:
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
        print(f"[Plantri] {len(raw_lines)} raw graphs from plantri")

    # For small inputs, use sequential processing
    if len(raw_lines) < effective_chunk_size * 2 or n_workers <= 1:
        if verbose:
            print("[Plantri] Using sequential processing (small input)")
        graphs, _ = _build_graphs_from_raw_lines(
            raw_lines,
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

    # Step 2: Split into chunks for parallel processing
    chunks = []
    start_id = 0
    for i in range(0, len(raw_lines), effective_chunk_size):
        chunk_lines = raw_lines[i:i + effective_chunk_size]
        chunks.append((chunk_lines, start_id, validate, include_primal))
        start_id += len(chunk_lines)

    if verbose:
        print(f"[Plantri] Processing {len(chunks)} chunks...")

    # Step 3: Parallel processing
    all_graphs: List[PlaneGraph] = []

    # Use spawn context for Windows compatibility
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_process_graph_chunk, chunks)

    for chunk_graphs in results:
        all_graphs.extend(chunk_graphs)
        if max_count and len(all_graphs) >= max_count:
            all_graphs = all_graphs[:max_count]
            break

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

