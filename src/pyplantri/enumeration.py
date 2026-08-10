# src/pyplantri/enumeration.py
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import warnings
from collections.abc import Callable, Generator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path

from .plane_graph import QuarticPlaneMap
from .plantri_interface import QuadrangulationDualClass, QuadrangulationEnumerator
from .types import Embedding


_MAX_AUTO_WORKERS = 4
_DEFAULT_POOL_CHUNKSIZE = 64


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


def _build_quartic_multigraph_task(
    item: tuple[int, Embedding],
) -> QuarticPlaneMap:
    """Build the dual of one checked arbitrary simple quadrangulation."""
    graph_id, primal_embedding = item
    return QuarticPlaneMap._from_primal_rotation_system(
        primal_embedding,
        graph_id,
        require_simple_dual=False,
    )


def _build_simple_quartic_task(item: tuple[int, Embedding]) -> QuarticPlaneMap:
    """Build one checked minimum-degree-three quadrangulation dual."""
    graph_id, primal_embedding = item
    return QuarticPlaneMap._from_primal_rotation_system(
        primal_embedding,
        graph_id,
        require_simple_dual=True,
    )


def _available_cpu_count() -> int:
    """Return the process-visible CPU count, respecting POSIX affinity."""
    get_affinity = getattr(os, "sched_getaffinity", None)
    if callable(get_affinity):
        try:
            return max(1, len(get_affinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def _validate_processing_controls(
    num_workers: int | None,
    pool_chunksize: int | None,
) -> None:
    """Validate multiprocessing controls before resolving plantri."""
    for name, value in (
        ("num_workers", num_workers),
        ("pool_chunksize", pool_chunksize),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name}: expected int > 0 or None, got {value!r}")


def _resolve_enumeration_request(
    dual_vertex_count: int,
    *,
    enumerator: QuadrangulationEnumerator,
    dual_class: QuadrangulationDualClass | str,
    start_method: str | None,
    timeout: float | None,
) -> QuadrangulationDualClass:
    """Resolve graph-class and process controls without starting plantri."""
    if start_method is not None and (
        type(start_method) is not str
        or start_method not in multiprocessing.get_all_start_methods()
    ):
        raise ValueError(f"unsupported start_method: {start_method!r}")
    resolved_dual_class = QuadrangulationDualClass(dual_class)
    enumerator._validate_supported_dual_vertex_count(dual_vertex_count)
    enumerator._validate_timeout(timeout)
    return resolved_dual_class


def _cleanup_after_error(
    error: BaseException,
    label: str,
    cleanup: Callable[[], object],
) -> None:
    """Preserve a primary error while making explicit close failures observable."""
    try:
        cleanup()
    except BaseException as cleanup_error:
        if isinstance(error, GeneratorExit):
            raise
        detail = " ".join(str(cleanup_error).split()) or type(cleanup_error).__name__
        error.add_note(f"pyplantri: {label}: {detail[:240]}")


def _close_enumerator_after(
    stream: Generator[QuarticPlaneMap, None, None],
    enumerator: QuadrangulationEnumerator,
) -> Generator[QuarticPlaneMap, None, None]:
    """Close an internally created enumerator with its lazy stream."""
    try:
        yield from stream
    except BaseException as error:
        _cleanup_after_error(error, "enumerator cleanup failed", enumerator.close)
        raise
    else:
        enumerator.close()


def _iter_resolved_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    resolved_dual_class: QuadrangulationDualClass,
    enumerator: QuadrangulationEnumerator,
    max_count: int | None,
    num_workers: int | None,
    pool_chunksize: int | None,
    start_method: str | None,
    timeout: float | None,
    progress: _EnumerationProgress | None = None,
) -> Generator[QuarticPlaneMap, None, None]:
    """Yield structurally proved maps after resolving public controls."""
    if max_count == 0:
        return

    primal_embedding_iter = iter(
        enumerator.iter_primal_embeddings(
            dual_vertex_count,
            dual_class=resolved_dual_class,
            timeout=timeout,
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
            else min(_available_cpu_count(), _MAX_AUTO_WORKERS)
        )
        resolved_pool_chunksize = (
            pool_chunksize if pool_chunksize is not None else _DEFAULT_POOL_CHUNKSIZE
        )
        if max_count is not None:
            chunk_count = (
                max_count + resolved_pool_chunksize - 1
            ) // resolved_pool_chunksize
            worker_count = min(worker_count, chunk_count)
        use_pool = bool(prefetched_primal_embeddings) and worker_count > 1

        mp_context = None
        if use_pool:
            mp_context = multiprocessing.get_context(start_method)
            method = mp_context.get_start_method()
            if method in {"spawn", "forkserver"}:
                raw_main_path = getattr(sys.modules.get("__main__"), "__file__", None)
                main_path = (
                    Path(raw_main_path)
                    if isinstance(raw_main_path, str) and raw_main_path
                    else None
                )
                if (
                    main_path is None
                    or (main_path.name.startswith("<") and main_path.name.endswith(">"))
                    or not main_path.is_file()
                ):
                    if num_workers is not None or start_method is not None:
                        raise RuntimeError(f"{method} requires an importable __main__")
                    warnings.warn(
                        f"{method} requires importable __main__; using sequential",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    mp_context = None

        indexed_primal_embeddings = enumerate(
            chain(prefetched_primal_embeddings, selected_primal_embeddings)
        )
        build_plane_map = (
            _build_simple_quartic_task
            if resolved_dual_class is QuadrangulationDualClass.SIMPLE_QUARTIC
            else _build_quartic_multigraph_task
        )
        if mp_context is not None:
            pool = mp_context.Pool(processes=worker_count)
            try:
                yield from pool.imap(
                    build_plane_map,
                    indexed_primal_embeddings,
                    resolved_pool_chunksize,
                )
            except BaseException as error:
                _cleanup_after_error(error, "pool termination failed", pool.terminate)
                _cleanup_after_error(error, "pool join failed", pool.join)
                raise
            else:
                try:
                    pool.close()
                except BaseException as error:
                    _cleanup_after_error(error, "pool join failed", pool.join)
                    raise
                pool.join()
        else:
            yield from map(build_plane_map, indexed_primal_embeddings)
    except BaseException as error:
        close = getattr(primal_embedding_iter, "close", None)
        if callable(close):
            _cleanup_after_error(error, "source cleanup failed", close)
        raise
    else:
        close = getattr(primal_embedding_iter, "close", None)
        if callable(close):
            close()


def iter_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    dual_class: QuadrangulationDualClass
    | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    max_count: int | None = None,
    num_workers: int | None = 1,
    pool_chunksize: int | None = None,
    start_method: str | None = None,
    timeout: float | None = None,
    enumerator: QuadrangulationEnumerator | None = None,
) -> Generator[QuarticPlaneMap, None, None]:
    """Yield source-ordered maps; timeout bounds plantri, not Python conversion."""
    if max_count is not None and (type(max_count) is not int or max_count < 0):
        raise ValueError(f"max_count: expected int >= 0 or None, got {max_count!r}")
    _validate_processing_controls(num_workers, pool_chunksize)
    owns_enumerator = enumerator is None
    resolved_enumerator = (
        enumerator if enumerator is not None else QuadrangulationEnumerator()
    )
    resolved_dual_class = _resolve_enumeration_request(
        dual_vertex_count,
        enumerator=resolved_enumerator,
        dual_class=dual_class,
        start_method=start_method,
        timeout=timeout,
    )
    stream = _iter_resolved_simple_quadrangulation_duals(
        dual_vertex_count,
        resolved_dual_class=resolved_dual_class,
        enumerator=resolved_enumerator,
        max_count=max_count,
        num_workers=num_workers,
        pool_chunksize=pool_chunksize,
        start_method=start_method,
        timeout=timeout,
    )
    return (
        _close_enumerator_after(stream, resolved_enumerator)
        if owns_enumerator
        else stream
    )


def enumerate_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    max_count: int,
    dual_class: QuadrangulationDualClass
    | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    num_workers: int | None = 1,
    pool_chunksize: int | None = None,
    start_method: str | None = None,
    timeout: float | None = None,
    enumerator: QuadrangulationEnumerator | None = None,
) -> PlantriEnumerationResult:
    """Materialize a bounded stream; timeout covers only active plantri generation."""
    if type(max_count) is not int or max_count < 0:
        raise ValueError(f"max_count: expected int >= 0, got {max_count!r}")
    _validate_processing_controls(num_workers, pool_chunksize)
    owns_enumerator = enumerator is None
    resolved_enumerator = (
        enumerator if enumerator is not None else QuadrangulationEnumerator()
    )
    resolved_dual_class = _resolve_enumeration_request(
        dual_vertex_count,
        enumerator=resolved_enumerator,
        dual_class=dual_class,
        start_method=start_method,
        timeout=timeout,
    )

    enumeration_started_at = time.perf_counter()
    progress = _EnumerationProgress(started_at=enumeration_started_at)
    plane_map_iter = _iter_resolved_simple_quadrangulation_duals(
        dual_vertex_count,
        resolved_dual_class=resolved_dual_class,
        enumerator=resolved_enumerator,
        max_count=max_count,
        num_workers=num_workers,
        pool_chunksize=pool_chunksize,
        start_method=start_method,
        timeout=timeout,
        progress=progress,
    )
    try:
        plane_maps = tuple(plane_map_iter)
    except BaseException as error:
        _cleanup_after_error(error, "stream cleanup failed", plane_map_iter.close)
        if owns_enumerator:
            _cleanup_after_error(
                error, "enumerator cleanup failed", resolved_enumerator.close
            )
        raise
    else:
        try:
            plane_map_iter.close()
        except BaseException as error:
            if owns_enumerator:
                _cleanup_after_error(
                    error, "enumerator cleanup failed", resolved_enumerator.close
                )
            raise
        if owns_enumerator:
            resolved_enumerator.close()

    elapsed_s = time.perf_counter() - enumeration_started_at
    return PlantriEnumerationResult(
        graphs=plane_maps,
        time_to_first_embedding_s=progress.time_to_first_embedding_s,
        remaining_s=elapsed_s - progress.time_to_first_embedding_s,
    )
