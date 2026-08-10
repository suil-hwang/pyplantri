# src/pyplantri/enumeration.py
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from functools import partial
from itertools import chain, islice
from pathlib import Path

from .plane_graph import QuarticPlaneMap
from .plantri_interface import QuadrangulationDualClass, QuadrangulationEnumerator
from .types import Embedding


_MAX_AUTO_WORKERS = 16
_DEFAULT_POOL_CHUNK_SIZE = 256


@dataclass(frozen=True, slots=True)
class PlantriEnumerationResult:
    """Immutable map batch with zero time-to-first for an empty stream."""

    graphs: tuple[QuarticPlaneMap, ...]
    time_to_first_embedding_s: float
    remaining_s: float

    @property
    def total_s(self) -> float:
        """Return total enumeration time."""
        return self.time_to_first_embedding_s + self.remaining_s


@dataclass(slots=True)
class _EnumerationProgress:
    """Mutable timing state for the eager collector."""

    started_at: float
    time_to_first_embedding_s: float = 0.0


def _build_quartic_plane_map_task(
    indexed_primal_embedding: tuple[int, Embedding],
    *,
    dual_class: QuadrangulationDualClass,
) -> QuarticPlaneMap:
    """Build and audit one plantri map through the multiprocessing boundary."""
    graph_id, primal_embedding = indexed_primal_embedding
    plane_map = QuarticPlaneMap._from_plantri_embedding(
        primal_embedding,
        graph_id=graph_id,
    )
    if dual_class is QuadrangulationDualClass.SIMPLE_QUARTIC:
        support_edge_count, _ = plane_map._dual_edge_cardinality_profile()
        if support_edge_count != 2 * plane_map.dual_num_vertices:
            parallel_edges = sorted(
                (edge, multiplicity)
                for edge, multiplicity in plane_map.dual_edge_multiplicity.items()
                if multiplicity != 1
            )
            raise ValueError(f"graph_id={graph_id}: simple-quartic record contains parallel edges: {parallel_edges}")
    return plane_map


def _validate_processing_controls(
    max_count: int | None,
    num_workers: int | None,
    chunk_size: int | None,
) -> None:
    """Validate limits that precede all other public request controls."""
    if max_count is not None and (type(max_count) is not int or max_count < 0):
        raise ValueError(f"max_count must be None or a non-negative int, got {max_count!r}")
    for name, value in (("num_workers", num_workers), ("chunk_size", chunk_size)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be None or a positive int, got {value!r}")


def _resolve_enumeration_request(
    dual_vertex_count: int,
    *,
    dual_class: QuadrangulationDualClass | str,
    start_method: str | None,
) -> QuadrangulationDualClass:
    """Resolve graph-class and process controls without starting plantri."""
    if start_method is not None and (
        type(start_method) is not str
        or start_method not in multiprocessing.get_all_start_methods()
    ):
        raise ValueError(f"unsupported start_method: {start_method!r}")
    resolved_dual_class = QuadrangulationDualClass(dual_class)
    QuadrangulationEnumerator._validate_supported_dual_vertex_count(dual_vertex_count)
    return resolved_dual_class


def _iter_resolved_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    resolved_dual_class: QuadrangulationDualClass,
    max_count: int | None,
    num_workers: int | None,
    chunk_size: int | None,
    start_method: str | None,
    progress: _EnumerationProgress | None = None,
) -> Generator[QuarticPlaneMap, None, None]:
    """Yield audited maps lazily after all public controls are resolved."""
    if (
        max_count == 0
        or dual_vertex_count
        < QuadrangulationEnumerator._MIN_NONEMPTY_DUAL_VERTEX_COUNT_BY_CLASS[
            resolved_dual_class
        ]
    ):
        return

    primal_embedding_iter = iter(
        QuadrangulationEnumerator().iter_primal_embeddings(
            dual_vertex_count,
            dual_class=resolved_dual_class,
        )
    )
    selected_primal_embeddings = (
        islice(primal_embedding_iter, max_count)
        if max_count is not None
        else primal_embedding_iter
    )
    try:
        prefetched_primal_embeddings = list(islice(selected_primal_embeddings, 1))
        if progress is not None and prefetched_primal_embeddings:
            progress.time_to_first_embedding_s = (
                time.perf_counter() - progress.started_at
            )

        worker_count = (
            num_workers
            if num_workers is not None
            else max(1, min(os.cpu_count() or 4, _MAX_AUTO_WORKERS))
        )
        pool_chunk_size = chunk_size if chunk_size is not None else _DEFAULT_POOL_CHUNK_SIZE
        if max_count is not None:
            chunk_count = (max_count + pool_chunk_size - 1) // pool_chunk_size
            worker_count = min(worker_count, chunk_count)
        use_pool = bool(prefetched_primal_embeddings) and worker_count > 1

        mp_context = None
        if use_pool:
            mp_context = multiprocessing.get_context(start_method)
            resolved_start_method = mp_context.get_start_method()
            if resolved_start_method in {"spawn", "forkserver"}:
                raw_main_path = getattr(sys.modules.get("__main__"), "__file__", None)
                main_path = Path(raw_main_path) if isinstance(raw_main_path, str) and raw_main_path else None
                if (
                    main_path is None
                    or (main_path.name.startswith("<") and main_path.name.endswith(">"))
                    or not main_path.is_file()
                ):
                    warnings.warn(f"{resolved_start_method} requires importable __main__; using sequential", RuntimeWarning, stacklevel=2)
                    mp_context = None

        indexed_primal_embeddings = enumerate(
            chain(prefetched_primal_embeddings, selected_primal_embeddings)
        )
        build_plane_map = partial(
            _build_quartic_plane_map_task,
            dual_class=resolved_dual_class,
        )
        if mp_context is not None:
            pool = mp_context.Pool(processes=worker_count)
            try:
                yield from pool.imap(
                    build_plane_map,
                    indexed_primal_embeddings,
                    pool_chunk_size,
                )
            except BaseException:
                pool.terminate()
                raise
            else:
                pool.close()
            finally:
                pool.join()
        else:
            yield from map(build_plane_map, indexed_primal_embeddings)
    finally:
        close = getattr(primal_embedding_iter, "close", None)
        if callable(close):
            close()


def iter_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    max_count: int | None = None,
    num_workers: int | None = 1,
    chunk_size: int | None = None,
    start_method: str | None = None,
) -> Generator[QuarticPlaneMap, None, None]:
    """Return a lazy, source-ordered stream of audited dual maps."""
    _validate_processing_controls(max_count, num_workers, chunk_size)
    resolved_dual_class = _resolve_enumeration_request(
        dual_vertex_count,
        dual_class=dual_class,
        start_method=start_method,
    )
    return _iter_resolved_simple_quadrangulation_duals(
        dual_vertex_count,
        resolved_dual_class=resolved_dual_class,
        max_count=max_count,
        num_workers=num_workers,
        chunk_size=chunk_size,
        start_method=start_method,
    )


def enumerate_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    max_count: int | None = None,
    num_workers: int | None = 1,
    chunk_size: int | None = None,
    start_method: str | None = None,
) -> PlantriEnumerationResult:
    """Materialize the lazy dual-map stream for a bounded workload."""
    _validate_processing_controls(max_count, num_workers, chunk_size)
    resolved_dual_class = _resolve_enumeration_request(
        dual_vertex_count,
        dual_class=dual_class,
        start_method=start_method,
    )

    enumeration_started_at = time.perf_counter()
    progress = _EnumerationProgress(started_at=enumeration_started_at)
    plane_map_iter = _iter_resolved_simple_quadrangulation_duals(
        dual_vertex_count,
        resolved_dual_class=resolved_dual_class,
        max_count=max_count,
        num_workers=num_workers,
        chunk_size=chunk_size,
        start_method=start_method,
        progress=progress,
    )
    try:
        plane_maps = tuple(plane_map_iter)
    finally:
        plane_map_iter.close()

    elapsed_s = time.perf_counter() - enumeration_started_at
    return PlantriEnumerationResult(
        graphs=plane_maps,
        time_to_first_embedding_s=progress.time_to_first_embedding_s,
        remaining_s=elapsed_s - progress.time_to_first_embedding_s,
    )
