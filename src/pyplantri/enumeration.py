# src/pyplantri/enumeration.py
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import warnings
from dataclasses import dataclass
from functools import partial
from itertools import chain, islice
from pathlib import Path

from .plane_graph import QuarticPlaneMap
from .plantri import QuadrangulationDualClass, QuadrangulationEnumerator
from .types import Embedding


_MAX_AUTO_WORKERS = 16
_DEFAULT_POOL_CHUNK_SIZE = 256
_MIN_POOL_RECORDS = 1_024


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


def _build_quartic_plane_map_task(
    indexed_primal_embedding: tuple[int, Embedding],
    *,
    dual_class: QuadrangulationDualClass,
    validate: bool,
) -> QuarticPlaneMap:
    """Build one quartic map through the multiprocessing pickle boundary."""
    graph_id, primal_embedding = indexed_primal_embedding
    plane_map = QuarticPlaneMap.from_primal_embedding(
        primal_embedding,
        graph_id=graph_id,
        validate=validate,
    )
    if dual_class is QuadrangulationDualClass.SIMPLE_QUARTIC:
        support_edge_count, _, _ = plane_map.dual_topology_profile()
        if support_edge_count != 2 * plane_map.dual_num_vertices:
            parallel_edges = sorted(
                (edge, multiplicity)
                for edge, multiplicity in plane_map.dual_edge_multiplicity.items()
                if multiplicity != 1
            )
            raise ValueError(f"graph_id={graph_id}: simple-quartic record contains parallel edges: {parallel_edges}")
    return plane_map


def enumerate_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    max_count: int | None = None,
    validate: bool = True,
    verbose: bool = False,
    num_workers: int | None = 1,
    chunk_size: int | None = None,
    start_method: str | None = None,
) -> PlantriEnumerationResult:
    """Materialize dual plane maps of simple quadrangulations in source order.

    ``max_count`` bounds the source prefix; ``chunk_size`` is the pool chunksize.
    """
    if max_count is not None and (type(max_count) is not int or max_count < 0):
        raise ValueError(f"max_count must be None or a non-negative int, got {max_count!r}")
    for name, value in (("num_workers", num_workers), ("chunk_size", chunk_size)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be None or a positive int, got {value!r}")
    for name, value in (
        ("validate", validate),
        ("verbose", verbose),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be bool, got {value!r}")
    if start_method is not None and (
        type(start_method) is not str
        or start_method not in multiprocessing.get_all_start_methods()
    ):
        raise ValueError(f"unsupported start_method: {start_method!r}")
    resolved_dual_class = QuadrangulationEnumerator._normalize_dual_class(dual_class)
    QuadrangulationEnumerator._validate_supported_dual_vertex_count(dual_vertex_count)

    enumeration_started_at = time.perf_counter()
    if max_count == 0 or dual_vertex_count < QuadrangulationEnumerator._min_nonempty_dual_vertices(resolved_dual_class):
        return PlantriEnumerationResult(
            graphs=(),
            time_to_first_embedding_s=0.0,
            remaining_s=time.perf_counter() - enumeration_started_at,
        )

    primal_embedding_iter = iter(
        QuadrangulationEnumerator().iter_embeddings(
            dual_vertex_count,
            dual_class=resolved_dual_class,
        )
    )
    try:
        prefetched_primal_embeddings = list(islice(primal_embedding_iter, 1))
        time_to_first_embedding_s = (
            time.perf_counter() - enumeration_started_at
            if prefetched_primal_embeddings
            else 0.0
        )

        worker_count = num_workers
        if worker_count is None:
            worker_count = max(
                1,
                min(os.cpu_count() or 4, _MAX_AUTO_WORKERS),
            )
        pool_chunk_size = (
            chunk_size if chunk_size is not None else _DEFAULT_POOL_CHUNK_SIZE
        )
        required_pool_records = max(
            _MIN_POOL_RECORDS,
            pool_chunk_size * 2,
        )
        use_pool = bool(prefetched_primal_embeddings) and worker_count > 1 and (
            max_count is None or max_count >= required_pool_records
        )
        if use_pool:
            prefetched_primal_embeddings.extend(
                islice(
                    primal_embedding_iter,
                    required_pool_records - len(prefetched_primal_embeddings),
                )
            )
            use_pool = len(prefetched_primal_embeddings) >= required_pool_records

        mp_context = None
        if use_pool:
            mp_context = multiprocessing.get_context(start_method)
            resolved_start_method = mp_context.get_start_method()
            if resolved_start_method in {"spawn", "forkserver"}:
                raw_main_path = getattr(sys.modules.get("__main__"), "__file__", None)
                main_path = Path(raw_main_path) if isinstance(raw_main_path, str) and raw_main_path else None
                if (
                    main_path is None
                    or (
                        main_path.name.startswith("<")
                        and main_path.name.endswith(">")
                    )
                    or not main_path.is_file()
                ):
                    warnings.warn(f"{resolved_start_method} requires importable __main__; using sequential", RuntimeWarning, stacklevel=2)
                    mp_context = None

        used_parallel_workers = mp_context is not None

        selected_primal_embeddings = chain(
            prefetched_primal_embeddings,
            primal_embedding_iter,
        )
        if max_count is not None:
            selected_primal_embeddings = islice(
                selected_primal_embeddings,
                max_count,
            )
        indexed_primal_embeddings = enumerate(selected_primal_embeddings)
        build_plane_map = partial(
            _build_quartic_plane_map_task,
            validate=validate,
            dual_class=resolved_dual_class,
        )
        if mp_context is not None:
            pool = mp_context.Pool(processes=worker_count)
            try:
                plane_maps = tuple(
                    pool.imap(
                        build_plane_map,
                        indexed_primal_embeddings,
                        pool_chunk_size,
                    )
                )
            except BaseException:
                pool.terminate()
                raise
            else:
                pool.close()
            finally:
                pool.join()
        else:
            plane_maps = tuple(map(build_plane_map, indexed_primal_embeddings))
    finally:
        close = getattr(primal_embedding_iter, "close", None)
        if callable(close):
            close()

    elapsed_s = time.perf_counter() - enumeration_started_at
    result = PlantriEnumerationResult(
        graphs=plane_maps,
        time_to_first_embedding_s=time_to_first_embedding_s,
        remaining_s=elapsed_s - time_to_first_embedding_s,
    )
    if verbose:
        execution_mode = "parallel" if used_parallel_workers else "sequential"
        print(f"[Plantri] {resolved_dual_class.value} n={dual_vertex_count}: {len(plane_maps)} maps ({execution_mode} map construction)")
    return result
