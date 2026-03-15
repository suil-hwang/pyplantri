# src/pyplantri/enumeration.py
from __future__ import annotations

import multiprocessing
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .builder import _build_plane_graph
from .plane_graph import PlaneGraph
from .plantri import Plantri, QuadrangulationEnumerator


def _hit_max_count(current_len: int, max_count: Optional[int]) -> bool:
    """Return True once the requested output count has been reached."""
    return max_count is not None and current_len >= max_count


def _main_module_path() -> Optional[Path]:
    """Return the importable __main__ path, or None when unavailable."""
    main_module = sys.modules.get("__main__")
    if main_module is None:
        return None

    raw_path = getattr(main_module, "__file__", None)
    if not raw_path:
        return None

    try:
        main_path = Path(raw_path)
    except (TypeError, ValueError):
        return None

    if main_path.name.startswith("<") and main_path.name.endswith(">"):
        return None

    if not main_path.is_absolute():
        main_path = Path.cwd() / main_path

    return main_path


def _resolve_parallel_context(
    start_method: Optional[str],
) -> Tuple[Optional[multiprocessing.context.BaseContext], str]:
    """Resolve multiprocessing context and detect unsupported interactive entrypoints."""
    ctx = (
        multiprocessing.get_context(start_method)
        if start_method is not None
        else multiprocessing.get_context()
    )
    method = ctx.get_start_method()

    if method in {"spawn", "forkserver"}:
        main_path = _main_module_path()
        if main_path is None or not main_path.exists():
            return None, method

    return ctx, method


def enumerate_plane_graphs(
    dual_vertex_count: int,
    max_count: Optional[int] = None,
    validate: bool = True,
    verbose: bool = False,
    include_primal: bool = True,
) -> List[PlaneGraph]:
    """Enumerates all n-vertex 4-regular planar multigraphs."""
    if max_count == 0:
        return []

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

        if _hit_max_count(len(graphs), max_count):
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


def _try_build_single_graph(
    line: Union[str, bytes],
    graph_id: int,
    *,
    include_primal: bool,
    digon_zero_only: bool,
    validate: bool,
) -> Optional[PlaneGraph]:
    """Attempt to build a PlaneGraph from a raw double_code line.

    Returns None if the line is malformed, fails validation, or is
    filtered out by digon_zero_only.
    """
    try:
        primal_data, dual_data = QuadrangulationEnumerator.parse_double_code(line)
        if not primal_data.adjacency_list or not dual_data.adjacency_list:
            return None
        if digon_zero_only and _has_digon_from_dual_adjacency(
            dual_data.adjacency_list
        ):
            return None
        graph = _build_plane_graph(
            primal_data,
            dual_data,
            graph_id,
            include_primal=include_primal,
        )
        if validate:
            is_valid, _ = graph.validate()
            if not is_valid:
                return None
        return graph
    except (ValueError, RuntimeError, KeyError, IndexError):
        return None


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
    if max_count == 0:
        _close_if_possible(raw_iter)
        return graphs, generated_count

    try:
        for graph_id, line in enumerate(raw_iter):
            generated_count += 1
            graph = _try_build_single_graph(
                line,
                graph_id,
                include_primal=include_primal,
                digon_zero_only=digon_zero_only,
                validate=validate,
            )
            if graph is not None:
                graphs.append(graph)
                if _hit_max_count(len(graphs), max_count):
                    break
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
        graph = _try_build_single_graph(
            line,
            start_id + i,
            include_primal=include_primal,
            digon_zero_only=digon_zero_only,
            validate=validate,
        )
        if graph is not None:
            graphs.append(graph)

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
    if max_count == 0:
        timing = EnumerationTiming(
            plantri_s=0.0,
            parse_build_s=0.0,
            total_s=0.0,
            graph_count=0,
        )
        return FilteredEnumerationResult(
            graphs=[],
            generated_count=0,
            timing=timing,
        )

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
    num_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    include_primal: bool = True,
    digon_zero_only: bool = False,
    start_method: Optional[str] = None,
) -> FilteredEnumerationResult:
    """Parallel enumeration of plane graphs."""
    import time

    def _make_result(
        graphs: List[PlaneGraph],
        plantri_s: float,
        t_start: float,
    ) -> FilteredEnumerationResult:
        t_total = time.perf_counter() - t_start
        return FilteredEnumerationResult(
            graphs=graphs,
            generated_count=len(graphs),
            timing=EnumerationTiming(
                plantri_s=plantri_s,
                parse_build_s=t_total - plantri_s,
                total_s=t_total,
                graph_count=len(graphs),
            ),
        )

    t_start = time.perf_counter()
    if max_count == 0:
        return _make_result([], 0.0, t_start)

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

    if num_workers is None:
        num_workers = _default_parallel_workers()
    effective_chunk_size = (
        chunk_size if chunk_size is not None else _default_chunk_size()
    )

    if not prefetched:
        return _make_result([], t_plantri, t_start)

    if verbose:
        mode = ", digon=0" if digon_zero_only else ""
        print(
            f"[Plantri] Parallel enumeration (n={dual_vertex_count}, "
            f"workers={num_workers}, chunk={effective_chunk_size}{mode})..."
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
    if len(prefetched) < warmup_limit or num_workers <= 1:
        if verbose:
            print("[Plantri] Using sequential processing (small input)")
        graphs, _ = _build_graphs_from_raw_lines(
            _iter_prefixed_lines(prefetched, raw_iter),
            max_count=max_count,
            validate=validate,
            include_primal=include_primal,
            digon_zero_only=digon_zero_only,
        )
        return _make_result(graphs, t_plantri, t_start)

    # Step 2: Stream chunks directly to worker pool.
    all_graphs: List[PlaneGraph] = []
    chunk_args = _iter_chunk_args(
        _iter_prefixed_lines(prefetched, raw_iter),
        chunk_size=effective_chunk_size,
        validate=validate,
        include_primal=include_primal,
        digon_zero_only=digon_zero_only,
    )

    ctx, resolved_start_method = _resolve_parallel_context(start_method)
    if ctx is None:
        warnings.warn(
            "enumerate_plane_graphs_parallel() fell back to sequential processing "
            f"because the active '{resolved_start_method}' start method requires an "
            "importable __main__ module. Run from a script protected by "
            "\"if __name__ == '__main__':\" or use num_workers=1 in interactive "
            "sessions.",
            RuntimeWarning,
            stacklevel=2,
        )
        graphs, _ = _build_graphs_from_raw_lines(
            _iter_prefixed_lines(prefetched, raw_iter),
            max_count=max_count,
            validate=validate,
            include_primal=include_primal,
            digon_zero_only=digon_zero_only,
        )
        return _make_result(graphs, t_plantri, t_start)

    with ctx.Pool(processes=num_workers) as pool:
        try:
            for chunk_graphs in pool.imap(_process_graph_chunk, chunk_args):
                all_graphs.extend(chunk_graphs)
                if _hit_max_count(len(all_graphs), max_count):
                    all_graphs = all_graphs[:max_count]
                    break
        finally:
            _close_if_possible(chunk_args)

    if verbose:
        print(f"[Plantri] Found {len(all_graphs)} valid graphs")

    return _make_result(all_graphs, t_plantri, t_start)
