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


_DEFAULT_NUM_WORKERS = 16
_DEFAULT_CHUNK_SIZE = 256
_MIN_PARALLEL_RECORDS = 1_024


def _build_graph_task(
    indexed_embedding: tuple[int, Embedding],
    *,
    dual_class: QuadrangulationDualClass,
    validate: bool,
) -> QuarticPlaneMap:
    """Build one map through the module-level multiprocessing pickle boundary."""
    graph_id, primal_embedding = indexed_embedding
    graph = QuarticPlaneMap.from_primal_embedding(
        primal_embedding,
        graph_id=graph_id,
        validate=validate,
    )
    if (
        dual_class is QuadrangulationDualClass.SIMPLE_QUARTIC
        and graph.dual_topology_profile()[0] != 2 * graph.dual_num_vertices
    ):
        parallel_edges = sorted(
            (edge, multiplicity)
            for edge, multiplicity in graph.dual_edge_multiplicity.items()
            if multiplicity != 1
        )
        raise ValueError(f"graph_id={graph_id}: simple-quartic record contains parallel edges: {parallel_edges}")
    return graph


@dataclass(frozen=True, slots=True)
class PlantriEnumerationResult:
    """Immutable enumeration result with startup-separated timing."""

    graphs: tuple[QuarticPlaneMap, ...]
    startup_s: float
    post_startup_s: float

    @property
    def total_s(self) -> float:
        """Return total enumeration time."""
        return self.startup_s + self.post_startup_s


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
    """Enumerate one plantri dual-class stream in deterministic source order."""
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
    dual_class = QuadrangulationEnumerator._normalize_dual_class(dual_class)
    QuadrangulationEnumerator._validate_supported_dual_vertex_count(dual_vertex_count)

    t_start = time.perf_counter()
    if max_count == 0 or dual_vertex_count < QuadrangulationEnumerator._min_nonempty_dual_vertices(dual_class):
        return PlantriEnumerationResult((), 0.0, time.perf_counter() - t_start)

    t_plantri_start = time.perf_counter()
    embedding_iter = iter(
        QuadrangulationEnumerator().iter_embeddings(
            dual_vertex_count,
            dual_class=dual_class,
        )
    )
    try:
        prefetched = list(islice(embedding_iter, 1))
        t_plantri = time.perf_counter() - t_plantri_start

        if num_workers is None:
            num_workers = max(
                1,
                min(os.cpu_count() or 4, _DEFAULT_NUM_WORKERS),
            )
        effective_chunk_size = (
            chunk_size if chunk_size is not None else _DEFAULT_CHUNK_SIZE
        )
        parallel_threshold = max(
            _MIN_PARALLEL_RECORDS,
            effective_chunk_size * 2,
        )
        use_parallel = bool(prefetched) and num_workers > 1 and (
            max_count is None or max_count > parallel_threshold
        )
        if use_parallel:
            prefetched.extend(
                islice(embedding_iter, parallel_threshold - len(prefetched))
            )
            use_parallel = len(prefetched) >= parallel_threshold

        ctx = None
        if use_parallel:
            ctx = multiprocessing.get_context(start_method)
            resolved_start_method = ctx.get_start_method()
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
                    ctx = None

        use_parallel = ctx is not None

        selected_embeddings = chain(prefetched, embedding_iter)
        if max_count is not None:
            selected_embeddings = islice(selected_embeddings, max_count)
        indexed_embeddings = enumerate(selected_embeddings)
        build_graph = partial(
            _build_graph_task,
            validate=validate,
            dual_class=dual_class,
        )
        if ctx is not None:
            pool = ctx.Pool(processes=num_workers)
            try:
                graphs = tuple(
                    pool.imap(build_graph, indexed_embeddings, effective_chunk_size)
                )
            except BaseException:
                pool.terminate()
                raise
            else:
                pool.close()
            finally:
                pool.join()
        else:
            graphs = tuple(map(build_graph, indexed_embeddings))
    finally:
        close = getattr(embedding_iter, "close", None)
        if callable(close):
            close()

    if verbose:
        mode = "simple-quartic " if dual_class is QuadrangulationDualClass.SIMPLE_QUARTIC else ""
        processing = "Parallel" if use_parallel else "Sequential"
        print(f"[Plantri] {processing} enumeration ({mode}n={dual_vertex_count}): {len(graphs)} graphs")

    total_s = time.perf_counter() - t_start
    return PlantriEnumerationResult(graphs, t_plantri, total_s - t_plantri)
