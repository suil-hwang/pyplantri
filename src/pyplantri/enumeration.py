# src/pyplantri/enumeration.py
from __future__ import annotations

import multiprocessing
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .builder import _build_plane_graph
from .plane_graph import PlaneGraph
from .plantri import Plantri, QuadrangulationEnumerator


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
    digon_zero_only: bool,
) -> Iterator[Tuple[List[bytes], int, bool, bool, bool]]:
    """Create chunk arguments lazily from a raw line stream."""
    raw_iter = iter(raw_lines)
    start_id = 0
    chunk: List[bytes] = []

    try:
        for line in raw_iter:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                current_chunk = chunk
                yield (
                    current_chunk,
                    start_id,
                    validate,
                    include_primal,
                    digon_zero_only,
                )
                start_id += len(current_chunk)
                chunk = []
        if chunk:
            yield (chunk, start_id, validate, include_primal, digon_zero_only)
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
            try:
                primal_data, dual_data = QuadrangulationEnumerator.parse_double_code(line)
                if not primal_data["adjacency_list"] or not dual_data["adjacency_list"]:
                    continue

                if digon_zero_only and _has_digon_from_dual_adjacency(
                    dual_data["adjacency_list"]
                ):
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
            except (ValueError, RuntimeError, KeyError, IndexError):
                continue  # Skip malformed graph lines and continue streaming
    finally:
        _close_if_possible(raw_iter)

    return graphs, generated_count


def _process_graph_chunk(
    args: Tuple[List[bytes], int, bool, bool, bool]
) -> List[PlaneGraph]:
    """Process a chunk of raw lines into PlaneGraph objects."""
    lines, start_id, validate, include_primal, digon_zero_only = args
    graphs: List[PlaneGraph] = []

    for i, line in enumerate(lines):
        graph_id = start_id + i
        try:
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
    digon_zero_only: bool = False,
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
        mode = ", digon=0" if digon_zero_only else ""
        print(
            f"[Plantri] Parallel enumeration (n={dual_vertex_count}, "
            f"workers={n_workers}, chunk={effective_chunk_size}{mode})..."
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
            digon_zero_only=digon_zero_only,
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
        digon_zero_only=digon_zero_only,
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
    for i, graph in enumerate(all_graphs):
        # PlaneGraph is frozen, so we need to create new instances
        all_graphs[i] = PlaneGraph(
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
