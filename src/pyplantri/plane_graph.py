# src/pyplantri/plane_graph.py
import gzip
import json
import logging
import multiprocessing
import os
import pickle
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple, Union

from .core import GraphConverter, QuadrangulationEnumerator, Plantri

logger = logging.getLogger(__name__)


class SecurityWarning(UserWarning):
    """Warning for security-related issues in pickle deserialization."""


@dataclass(frozen=True, slots=True)
class PlaneGraph:
    """Plane graph (embedded planar graph) with fixed combinatorial embedding.

    A plane graph is a planar graph with a specific embedding on the sphere,
    represented by clockwise cyclic edge ordering at each vertex. Generated
    by plantri (Brinkmann & McKay).

    Terminology:
        - Planar graph: A graph that CAN be embedded in the plane (abstract).
        - Plane graph: A planar graph WITH a fixed embedding (concrete).

    This class represents an immutable 4-regular plane multigraph containing
    both dual (Q*) and primal (Q) topology. All indices are 0-based.

    Dual Graph (Q*):
        - 4-regular plane multigraph (allows double edges, no loops).
        - num_vertices = n (dual vertices).
        - faces = n + 2 (dual faces = primal vertices).

    Primal Graph (Q):
        - Simple quadrangulation (no loops/multi-edges, all faces are 4-gons).
        - primal_num_vertices = n + 2 (primal vertices = dual faces).
        - primal_faces = n (primal faces = dual vertices).
    """

    num_vertices: int
    edges: Tuple[Tuple[int, int], ...]
    edge_multiplicity: Dict[Tuple[int, int], int]
    embedding: Dict[int, Tuple[int, ...]]  # CW cyclic order at each vertex
    faces: Tuple[Tuple[int, ...], ...]

    primal_num_vertices: int
    primal_embedding: Dict[int, Tuple[int, ...]]
    primal_faces: Tuple[Tuple[int, ...], ...]

    graph_id: int = 0
    _double_edges_cache: Optional[FrozenSet[Tuple[int, int]]] = field(
        default=None, init=False, repr=False, compare=False
    )
    _single_edges_cache: Optional[FrozenSet[Tuple[int, int]]] = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def num_edges(self) -> int:
        """Total edge count including multiplicity."""
        return sum(self.edge_multiplicity.values())

    @property
    def num_simple_edges(self) -> int:
        """Unique edge count (without multiplicity)."""
        return len(self.edges)

    @property
    def num_faces(self) -> int:
        """Face count (should be n + 2 by Euler's formula)."""
        return len(self.faces)

    @property
    def double_edges(self) -> FrozenSet[Tuple[int, int]]:
        """Set of double edges (digons)."""
        cache = getattr(self, "_double_edges_cache", None)
        if cache is None:
            cache = frozenset(e for e, m in self.edge_multiplicity.items() if m == 2)
            object.__setattr__(
                self,
                "_double_edges_cache",
                cache,
            )
        return cache

    @property
    def single_edges(self) -> FrozenSet[Tuple[int, int]]:
        """Set of single edges."""
        cache = getattr(self, "_single_edges_cache", None)
        if cache is None:
            cache = frozenset(e for e, m in self.edge_multiplicity.items() if m == 1)
            object.__setattr__(
                self,
                "_single_edges_cache",
                cache,
            )
        return cache

    @property
    def is_4_regular(self) -> bool:
        """Whether all vertices have degree 4."""
        return all(len(neighbors) == 4 for neighbors in self.embedding.values())

    @property
    def is_loop_free(self) -> bool:
        """Whether graph has no self-loops."""
        return all(v not in neighbors for v, neighbors in self.embedding.items())

    def get_neighbors_cw(self, vertex: int) -> Tuple[int, ...]:
        """Gets CW-ordered neighbors of a vertex."""
        return self.embedding[vertex]

    def get_neighbors_ccw(self, vertex: int) -> Tuple[int, ...]:
        """Gets CCW-ordered neighbors of a vertex."""
        return tuple(reversed(self.embedding[vertex]))

    def get_consecutive_pairs(
        self, vertex: int, ccw: bool = False
    ) -> List[Tuple[int, int]]:
        """Gets consecutive neighbor pairs for cyclic order constraints."""
        neighbors = self.get_neighbors_ccw(vertex) if ccw else self.get_neighbors_cw(vertex)
        n = len(neighbors)
        return [(neighbors[i], neighbors[(i + 1) % n]) for i in range(n)]

    def validate(self) -> Tuple[bool, List[str]]:
        """Validates graph invariants."""
        errors = []

        for v in range(self.num_vertices):
            if v not in self.embedding:
                errors.append(f"Vertex {v} missing from embedding")
                continue
            if len(self.embedding[v]) != 4:
                errors.append(
                    f"Vertex {v} has degree {len(self.embedding[v])}, expected 4"
                )

        single_count = len(self.single_edges)
        double_count = len(self.double_edges)
        expected_edge_sum = 2 * self.num_vertices
        actual_edge_sum = single_count + 2 * double_count
        if actual_edge_sum != expected_edge_sum:
            errors.append(f"Edge formula: s + 2d = {actual_edge_sum}, expected {expected_edge_sum}")

        expected_faces = self.num_vertices + 2
        if self.num_faces != expected_faces:
            errors.append(f"Face count: {self.num_faces}, expected {expected_faces}")

        for v, neighbors in self.embedding.items():
            if v in neighbors:
                errors.append(f"Self-loop at vertex {v}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict:
        """Converts to dictionary for JSON serialization."""
        return {
            "num_vertices": self.num_vertices,
            "edges": list(self.edges),
            "edge_multiplicity": {
                f"{u},{v}": m for (u, v), m in self.edge_multiplicity.items()
            },
            "embedding": {
                str(v): list(neighbors) for v, neighbors in self.embedding.items()
            },
            "faces": [list(f) for f in self.faces],
            "primal_num_vertices": self.primal_num_vertices,
            "primal_embedding": {
                str(v): list(neighbors)
                for v, neighbors in self.primal_embedding.items()
            },
            "primal_faces": [list(f) for f in self.primal_faces],
            "graph_id": self.graph_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlaneGraph":
        """Creates PlaneGraph from dictionary."""
        # Parse edges as explicit 2-tuples for type safety.
        parsed_edges: List[Tuple[int, int]] = [
            (int(e[0]), int(e[1])) for e in data["edges"]
        ]
        # Parse edge_multiplicity keys as explicit 2-tuples.
        parsed_edge_multiplicity: Dict[Tuple[int, int], int] = {
            (int(parts[0]), int(parts[1])): v
            for k, v in data["edge_multiplicity"].items()
            for parts in [k.split(",")]
        }
        return cls(
            num_vertices=data["num_vertices"],
            edges=tuple(parsed_edges),
            edge_multiplicity=parsed_edge_multiplicity,
            embedding={
                int(v): tuple(neighbors) for v, neighbors in data["embedding"].items()
            },
            faces=tuple(tuple(f) for f in data["faces"]),
            primal_num_vertices=data.get("primal_num_vertices", 0),
            primal_embedding={
                int(v): tuple(neighbors)
                for v, neighbors in data.get("primal_embedding", {}).items()
            },
            primal_faces=tuple(
                tuple(f) for f in data.get("primal_faces", [])
            ),
            graph_id=data.get("graph_id", 0),
        )


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
    # Note: vertex: 1-based → 0-based (subtract 1)
    #       position: already 0-based, keep as-is
    twin_map_0based: Dict[Tuple[int, int], Tuple[int, int]] = {
        (v - 1, i): (u - 1, j)
        for (v, i), (u, j) in twin_map_1based.items()
    }

    edge_multiplicity: Dict[Tuple[int, int], int] = {}
    unique_edges: Set[Tuple[int, int]] = set()

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

    return PlaneGraph(
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
    graphs = []

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
    graphs = []

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


# Cache format version. Increment when PlaneGraph fields change.
_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CacheMetadata:
    """Metadata for cache files."""

    format_version: int
    pyplantri_version: str
    dual_vertex_count: int
    graph_count: int
    pickle_protocol: int


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows PlaneGraph and built-in types."""

    # Whitelist of allowed modules and classes
    SAFE_MODULES: Dict[str, Set[str]] = {
        "pyplantri.plane_graph": {"PlaneGraph", "CacheMetadata"},
        "builtins": {"tuple", "dict", "list", "int", "str", "frozenset"},
        "collections": {"defaultdict"},
    }

    def __init__(self, file: Any, *, strict: bool = True):
        """Initialize SafeUnpickler."""
        super().__init__(file)
        self.strict = strict

    def find_class(self, module: str, name: str) -> Any:
        """Override to restrict loadable classes."""
        allowed = self.SAFE_MODULES.get(module, set())

        if name not in allowed:
            msg = (
                f"Attempted to unpickle forbidden class: {module}.{name}\n"
                f"Only PlaneGraph and built-in types are allowed.\n"
                f"This may indicate a malicious or corrupted file."
            )
            if self.strict:
                raise pickle.UnpicklingError(msg)
            else:
                warnings.warn(msg, SecurityWarning, stacklevel=2)

        return super().find_class(module, name)


def _get_version() -> str:
    """Get pyplantri version string."""
    try:
        from pyplantri import __version__

        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def _validate_format_version(metadata: CacheMetadata, filepath: Path) -> None:
    """Validate cache format version compatibility."""
    if metadata.format_version > _CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Cache file '{filepath}' was created with format version "
            f"{metadata.format_version}, but this pyplantri only supports "
            f"up to version {_CACHE_FORMAT_VERSION}. "
            f"Please upgrade pyplantri."
        )
    if metadata.format_version < _CACHE_FORMAT_VERSION:
        logger.warning(
            "Cache file '%s' uses older format version %d "
            "(current: %d). Consider regenerating.",
            filepath,
            metadata.format_version,
            _CACHE_FORMAT_VERSION,
        )


def _save_pickle(
    graphs: List[PlaneGraph],
    filepath: Path,
    dual_vertex_count: int,
    compress: bool,
    compress_level: int,
) -> None:
    """Pickle serialization with atomic write and optional gzip compression."""
    protocol = pickle.HIGHEST_PROTOCOL

    metadata = CacheMetadata(
        format_version=_CACHE_FORMAT_VERSION,
        pyplantri_version=_get_version(),
        dual_vertex_count=dual_vertex_count,
        graph_count=len(graphs),
        pickle_protocol=protocol,
    )

    payload = {
        "metadata": metadata,
        "graphs": graphs,
    }

    # Atomic write: write to temp file then rename.
    # Temp file must be in same directory for atomic rename.
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")

    try:
        with open(fd, "wb") as f:
            if compress:
                with gzip.GzipFile(
                    fileobj=f, mode="wb", compresslevel=compress_level
                ) as gz:
                    pickle.dump(payload, gz, protocol=protocol)
            else:
                pickle.dump(payload, f, protocol=protocol)

        # Atomic replace.
        Path(tmp_path).replace(filepath)

    except BaseException:
        # Cleanup temp file on failure.
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _save_json(
    graphs: List[PlaneGraph],
    filepath: Path,
    dual_vertex_count: int,
) -> None:
    """JSON serialization with atomic write and compact format."""
    payload = {
        "metadata": {
            "format_version": _CACHE_FORMAT_VERSION,
            "pyplantri_version": _get_version(),
            "dual_vertex_count": dual_vertex_count,
            "graph_count": len(graphs),
        },
        "graphs": [graph.to_dict() for graph in graphs],
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")

    try:
        with open(fd, "w", encoding="utf-8") as f:
            # Compact format for cache (no indent).
            json.dump(payload, f, separators=(",", ":"))
        Path(tmp_path).replace(filepath)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _load_pickle(
    filepath: Path,
    max_count: Optional[int],
    safe_mode: bool,
) -> Tuple[List[PlaneGraph], CacheMetadata]:
    """Pickle deserialization with optional safety restrictions."""
    with open(filepath, "rb") as f:
        # Auto-detect gzip by magic number (0x1f 0x8b).
        header = f.read(2)
        f.seek(0)

        if header == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                if safe_mode:
                    payload = SafeUnpickler(gz, strict=True).load()
                else:
                    payload = pickle.load(gz)
        else:
            if safe_mode:
                payload = SafeUnpickler(f, strict=True).load()
            else:
                payload = pickle.load(f)

    # Extract and validate metadata.
    metadata = payload.get("metadata")
    if metadata is None:
        # Legacy format without metadata.
        logger.warning("Legacy cache format without metadata: %s", filepath)
        graphs = payload if isinstance(payload, list) else payload.get("graphs", [])
        metadata = CacheMetadata(
            format_version=0,
            pyplantri_version="unknown",
            dual_vertex_count=0,
            graph_count=len(graphs),
            pickle_protocol=0,
        )
    else:
        if isinstance(metadata, dict):
            metadata = CacheMetadata(**metadata)
        _validate_format_version(metadata, filepath)
        graphs = payload["graphs"]

    if max_count is not None:
        graphs = graphs[:max_count]

    return graphs, metadata


def _load_json(
    filepath: Path,
    max_count: Optional[int],
) -> Tuple[List[PlaneGraph], CacheMetadata]:
    """JSON deserialization."""
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_meta = payload.get("metadata", {})
    metadata = CacheMetadata(
        format_version=raw_meta.get("format_version", 0),
        pyplantri_version=raw_meta.get("pyplantri_version", "unknown"),
        dual_vertex_count=raw_meta.get("dual_vertex_count", 0),
        graph_count=raw_meta.get("graph_count", 0),
        pickle_protocol=0,
    )
    _validate_format_version(metadata, filepath)

    raw_graphs = payload.get("graphs", payload if isinstance(payload, list) else [])
    if max_count is not None:
        raw_graphs = raw_graphs[:max_count]

    graphs = [PlaneGraph.from_dict(d) for d in raw_graphs]
    return graphs, metadata



def save_graphs_to_cache(
    graphs: List[PlaneGraph],
    filepath: Union[str, Path],
    *,
    dual_vertex_count: int = 0,
    compress: bool = True,
    compress_level: int = 6,
    use_json: bool = False,
) -> Path:
    """Save graph list to cache file with atomic write."""
    filepath = Path(filepath)

    if use_json:
        _save_json(graphs, filepath, dual_vertex_count)
    else:
        _save_pickle(graphs, filepath, dual_vertex_count, compress, compress_level)

    logger.info(
        "Saved %d graphs to %s (%.1f MB)",
        len(graphs),
        filepath,
        filepath.stat().st_size / 1e6,
    )
    return filepath


def load_graphs_from_cache(
    filepath: Union[str, Path],
    *,
    max_count: Optional[int] = None,
    use_json: bool = False,
    trusted: bool = False,
    safe_mode: bool = True,
) -> Tuple[List[PlaneGraph], CacheMetadata]:
    """Load graphs from cache file with security checks."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cache file not found: {filepath}")

    if use_json:
        return _load_json(filepath, max_count)
    else:
        if not trusted:
            raise ValueError(
                "pickle file loading requires trusted=True. "
                "pickle.load() can execute arbitrary code, so only use "
                "with files from trusted sources. "
                "Safer alternative: use_json=True"
            )
        return _load_pickle(filepath, max_count, safe_mode)


def main() -> None:
    """CLI entry point for plantri graph enumeration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enumerate 4-regular planar multigraphs via plantri",
    )
    parser.add_argument("n", type=int, help="Number of dual vertices (minimum 3)")
    parser.add_argument("--max", type=int, default=None, help="Maximum graph count")
    parser.add_argument("--export", type=str, help="Export to JSON cache file")
    parser.add_argument("--pickle", action="store_true", help="Use pickle format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--show-faces", action="store_true", help="Display face info")

    args = parser.parse_args()

    if args.n < 3:
        parser.error("n must be at least 3.")

    graphs = enumerate_plane_graphs(
        args.n,
        max_count=args.max,
        verbose=args.verbose,
    )

    print(f"\nTotal: {len(graphs)} {args.n}-vertex 4-regular planar multigraphs")

    for graph in graphs:
        print(f"\n{'='*50}")
        print(f"Graph #{graph.graph_id}")
        print(f"  Vertices: {graph.num_vertices}")
        print(f"  Edges: {graph.num_simple_edges} unique, {graph.num_edges} total")
        print(f"  Double edges: {len(graph.double_edges)}")
        print(f"  Faces: {graph.num_faces}")

        print("  Embedding (CW order, 0-based):")
        for v in range(graph.num_vertices):
            print(f"    {v}: {list(graph.embedding[v])}")

        if args.show_faces:
            print("  Faces:")
            for i, face in enumerate(graph.faces):
                face_type = "digon" if len(face) == 2 else f"{len(face)}-gon"
                print(f"    F{i}: {list(face)} ({face_type})")

    if args.export:
        save_graphs_to_cache(graphs, args.export, use_json=(not args.pickle))
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
