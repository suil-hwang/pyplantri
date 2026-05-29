# src/pyplantri/enumeration.py
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator

from .builder import _build_plane_graph_from_sections
from .plane_graph import PlaneGraph
from .plantri import QuadrangulationDualClass, QuadrangulationEnumerator


def _hit_max_count(current_len: int, max_count: int | None) -> bool:
    """Return True once the requested output count has been reached."""
    return max_count is not None and current_len >= max_count


def _main_module_path() -> Path | None:
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
    start_method: str | None,
) -> tuple[multiprocessing.context.BaseContext | None, str]:
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


def enumerate_simple_quadrangulation_duals(
    dual_vertex_count: int,
    max_count: int | None = None,
    validate: bool = True,
    verbose: bool = False,
    include_primal: bool = True,
) -> list[PlaneGraph]:
    """Enumerate duals of simple quadrangulations with `dual_vertex_count` vertices."""
    QuadrangulationEnumerator._validate_supported_dual_vertex_count(dual_vertex_count)
    if max_count == 0:
        return []

    if verbose:
        print(f"[Plantri] Enumerating {dual_vertex_count}-vertex quartic plane multigraphs...")

    result = enumerate_simple_quadrangulation_duals_filtered(
        dual_vertex_count,
        max_count=max_count,
        validate=validate,
        include_primal=include_primal,
        double_edge_free_only=False,
        verbose=False,
    )

    if verbose:
        print(f"[Plantri] Found {len(result.graphs)} valid graphs")

    return result.graphs


def _close_if_possible(obj: object) -> None:
    """Close an iterator/generator if it exposes a close() method."""
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _iter_raw_double_code_lines(
    raw_lines: Iterable[str | bytes],
) -> Iterator[bytes]:
    """Yield valid double_code lines from a raw line stream."""
    raw_iter = iter(raw_lines)
    try:
        for line in raw_iter:
            stripped = line.strip() if isinstance(line, bytes) else line.strip().encode("latin-1")
            if stripped and 48 <= stripped[0] <= 57:
                yield stripped
    finally:
        _close_if_possible(raw_iter)


def _iter_prefixed_lines(
    prefix: list[bytes],
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
    double_edge_free_only: bool,
) -> Iterator[tuple[list[bytes], int, bool, bool, bool]]:
    """Create chunk arguments lazily from a raw line stream."""
    raw_iter = iter(raw_lines)
    start_id = 0
    chunk: list[bytes] = []

    try:
        for line in raw_iter:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                current_chunk = chunk
                yield (current_chunk, start_id, validate, include_primal, double_edge_free_only)
                start_id += len(current_chunk)
                chunk = []
        if chunk:
            yield (chunk, start_id, validate, include_primal, double_edge_free_only)
    finally:
        _close_if_possible(raw_iter)


_DEFAULT_NUM_WORKERS = 16
_DEFAULT_CHUNK_SIZE = 5_000


def _resolve_default_num_workers(cpu_count: int) -> int:
    """Cap workers at _DEFAULT_NUM_WORKERS."""
    return max(2, min(cpu_count, _DEFAULT_NUM_WORKERS))


def _dual_class_for_filter(
    *,
    double_edge_free_only: bool,
) -> QuadrangulationDualClass:
    """Resolve the plantri generation class for the requested dual subset."""
    return QuadrangulationEnumerator._dual_class_from_filter(
        double_edge_free_only=double_edge_free_only,
    )


def _open_raw_double_code_stream(
    dual_vertex_count: int,
    *,
    dual_class: QuadrangulationDualClass,
) -> tuple[list[bytes], Iterator[bytes], float]:
    """Start plantri and return prefetched raw double_code lines plus startup latency."""
    t_start = time.perf_counter()
    enumerator = QuadrangulationEnumerator()
    raw_stream = _iter_raw_double_code_lines(
        enumerator.iter_double_code_lines(
            dual_vertex_count,
            dual_class=dual_class,
        )
    )
    raw_iter = iter(raw_stream)

    prefetched: list[bytes] = []
    try:
        prefetched.append(next(raw_iter))
    except StopIteration:
        pass

    return prefetched, raw_iter, time.perf_counter() - t_start


def _try_build_single_graph(
    line: str | bytes,
    graph_id: int,
    *,
    include_primal: bool,
    double_edge_free_only: bool,
    validate: bool,
) -> PlaneGraph | None:
    """Attempt to build a PlaneGraph from a raw double_code line."""
    try:
        primal_data, dual_data = QuadrangulationEnumerator.parse_double_code(line)
        if not primal_data.cyclic_adjacency or not dual_data.cyclic_adjacency:
            return None
        graph = _build_plane_graph_from_sections(
            primal_data,
            dual_data,
            graph_id,
            include_primal=include_primal,
        )
        if double_edge_free_only and graph.double_edges:
            return None
        if validate:
            is_valid, _ = graph.validate()
            if not is_valid:
                return None
        return graph
    except (ValueError, RuntimeError, KeyError, IndexError):
        return None


def _iter_graph_build_attempts(
    raw_lines: Iterable[str | bytes],
    *,
    start_id: int = 0,
    validate: bool,
    include_primal: bool,
    double_edge_free_only: bool,
) -> Iterator[PlaneGraph | None]:
    """Yield per-line graph build results, preserving source order and graph ids."""
    raw_iter = iter(raw_lines)
    try:
        for offset, line in enumerate(raw_iter):
            yield _try_build_single_graph(
                line,
                start_id + offset,
                include_primal=include_primal,
                double_edge_free_only=double_edge_free_only,
                validate=validate,
            )
    finally:
        _close_if_possible(raw_iter)


def _build_graphs_from_raw_lines(
    raw_lines: Iterable[str | bytes],
    *,
    max_count: int | None,
    validate: bool,
    include_primal: bool,
    double_edge_free_only: bool = False,
) -> tuple[list[PlaneGraph], int]:
    """Build PlaneGraph objects from raw double_code lines."""
    graphs: list[PlaneGraph] = []
    generated_count = 0

    if max_count == 0:
        _close_if_possible(raw_lines)
        return graphs, generated_count

    build_iter = _iter_graph_build_attempts(
        raw_lines,
        validate=validate,
        include_primal=include_primal,
        double_edge_free_only=double_edge_free_only,
    )
    try:
        for graph in build_iter:
            generated_count += 1
            if graph is not None:
                graphs.append(graph)
                if _hit_max_count(len(graphs), max_count):
                    break
    finally:
        _close_if_possible(build_iter)

    return graphs, generated_count


def _process_graph_chunk(
    args: tuple[list[bytes], int, bool, bool, bool]
) -> list[PlaneGraph | None]:
    """Process a chunk of raw lines into per-line PlaneGraph build results."""
    lines, start_id, validate, include_primal, double_edge_free_only = args
    return list(
        _iter_graph_build_attempts(
            lines,
            start_id=start_id,
            validate=validate,
            include_primal=include_primal,
            double_edge_free_only=double_edge_free_only,
        )
    )


@dataclass
class EnumerationTiming:
    """Timing breakdown for enumeration."""

    startup_s: float
    parse_build_s: float
    total_s: float
    graph_count: int


@dataclass
class FilteredEnumerationResult:
    """Enumeration result with source count and timing details."""

    graphs: list[PlaneGraph]
    generated_count: int
    timing: EnumerationTiming


def _make_filtered_result(
    graphs: list[PlaneGraph],
    *,
    generated_count: int,
    startup_s: float,
    t_start: float,
) -> FilteredEnumerationResult:
    """Create a filtered-enumeration result with consistent timing bookkeeping."""
    t_total = time.perf_counter() - t_start
    return FilteredEnumerationResult(
        graphs=graphs,
        generated_count=generated_count,
        timing=EnumerationTiming(
            startup_s=startup_s,
            parse_build_s=t_total - startup_s,
            total_s=t_total,
            graph_count=len(graphs),
        ),
    )


def enumerate_simple_quadrangulation_duals_filtered(
    dual_vertex_count: int,
    *,
    max_count: int | None = None,
    validate: bool = True,
    include_primal: bool = True,
    double_edge_free_only: bool = False,
    verbose: bool = False,
) -> FilteredEnumerationResult:
    """Enumerate simple-quadrangulation duals with optional class selection."""
    QuadrangulationEnumerator._validate_supported_dual_vertex_count(dual_vertex_count)
    dual_class = _dual_class_for_filter(double_edge_free_only=double_edge_free_only)
    t_start = time.perf_counter()
    if max_count == 0:
        return _make_filtered_result([], generated_count=0, startup_s=0.0, t_start=t_start)
    if dual_vertex_count < QuadrangulationEnumerator._min_nonempty_dual_vertices(dual_class):
        return _make_filtered_result([], generated_count=0, startup_s=0.0, t_start=t_start)

    prefetched, raw_iter, t_plantri = _open_raw_double_code_stream(dual_vertex_count, dual_class=dual_class)

    graphs, generated_count = _build_graphs_from_raw_lines(
        _iter_prefixed_lines(prefetched, raw_iter),
        max_count=max_count,
        validate=validate,
        include_primal=include_primal,
        double_edge_free_only=double_edge_free_only,
    )

    if verbose:
        mode = "double-edge-free " if double_edge_free_only else ""
        print(
            f"[Plantri] Filtered enumeration ({mode}n={dual_vertex_count}): "
            f"{len(graphs)}/{generated_count} graphs"
        )

    return _make_filtered_result(
        graphs,
        generated_count=generated_count,
        startup_s=t_plantri,
        t_start=t_start,
    )


def enumerate_simple_quadrangulation_duals_parallel(
    dual_vertex_count: int,
    max_count: int | None = None,
    validate: bool = True,
    verbose: bool = False,
    num_workers: int | None = None,
    chunk_size: int | None = None,
    include_primal: bool = True,
    double_edge_free_only: bool = False,
    start_method: str | None = None,
) -> FilteredEnumerationResult:
    """Parallel enumeration of simple-quadrangulation duals."""
    QuadrangulationEnumerator._validate_supported_dual_vertex_count(dual_vertex_count)
    dual_class = _dual_class_for_filter(double_edge_free_only=double_edge_free_only)

    t_start = time.perf_counter()
    if max_count == 0:
        return _make_filtered_result([], generated_count=0, startup_s=0.0, t_start=t_start)
    if dual_vertex_count < QuadrangulationEnumerator._min_nonempty_dual_vertices(dual_class):
        return _make_filtered_result([], generated_count=0, startup_s=0.0, t_start=t_start)

    # Step 1: Start streaming plantri output.
    prefetched, raw_iter, t_plantri = _open_raw_double_code_stream(dual_vertex_count, dual_class=dual_class)

    if num_workers is None:
        num_workers = _resolve_default_num_workers(os.cpu_count() or 4)
    effective_chunk_size = chunk_size if chunk_size is not None else _DEFAULT_CHUNK_SIZE

    if not prefetched:
        return _make_filtered_result([], generated_count=0, startup_s=t_plantri, t_start=t_start)

    if verbose:
        mode = ", double-edge-free" if double_edge_free_only else ""
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
        graphs, generated_count = _build_graphs_from_raw_lines(
            _iter_prefixed_lines(prefetched, raw_iter),
            max_count=max_count,
            validate=validate,
            include_primal=include_primal,
            double_edge_free_only=double_edge_free_only,
        )
        return _make_filtered_result(
            graphs,
            generated_count=generated_count,
            startup_s=t_plantri,
            t_start=t_start,
        )

    # Step 2: Stream chunks directly to worker pool.
    all_graphs: list[PlaneGraph] = []
    chunk_args = _iter_chunk_args(
        _iter_prefixed_lines(prefetched, raw_iter),
        chunk_size=effective_chunk_size,
        validate=validate,
        include_primal=include_primal,
        double_edge_free_only=double_edge_free_only,
    )

    ctx, resolved_start_method = _resolve_parallel_context(start_method)
    if ctx is None:
        warnings.warn(
            "enumerate_simple_quadrangulation_duals_parallel() fell back to "
            "sequential processing "
            f"because the active '{resolved_start_method}' start method requires an "
            "importable __main__ module. Run from a script protected by "
            "\"if __name__ == '__main__':\" or use num_workers=1 in interactive "
            "sessions.",
            RuntimeWarning,
            stacklevel=2,
        )
        graphs, generated_count = _build_graphs_from_raw_lines(
            _iter_prefixed_lines(prefetched, raw_iter),
            max_count=max_count,
            validate=validate,
            include_primal=include_primal,
            double_edge_free_only=double_edge_free_only,
        )
        return _make_filtered_result(
            graphs,
            generated_count=generated_count,
            startup_s=t_plantri,
            t_start=t_start,
        )

    with ctx.Pool(processes=num_workers) as pool:
        generated_count = 0
        stop_after_chunk = False
        try:
            for chunk_results in pool.imap(_process_graph_chunk, chunk_args):
                for graph in chunk_results:
                    generated_count += 1
                    if graph is not None:
                        all_graphs.append(graph)
                        if _hit_max_count(len(all_graphs), max_count):
                            stop_after_chunk = True
                            break
                if stop_after_chunk:
                    break
        finally:
            _close_if_possible(chunk_args)

    if verbose:
        print(f"[Plantri] Found {len(all_graphs)} valid graphs")

    return _make_filtered_result(
        all_graphs,
        generated_count=generated_count,
        startup_s=t_plantri,
        t_start=t_start,
    )
