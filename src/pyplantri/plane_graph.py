# src/pyplantri/plane_graph.py
import gzip
import json
import logging
import pickle
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Literal, Optional, Set, Tuple, Union, overload

import numpy as np

from .core import GraphConverter, QuadrangulationEnumerator

logger = logging.getLogger(__name__)


class SecurityWarning(UserWarning):
    """Warning for security-related issues in pickle deserialization."""

    pass


@dataclass(frozen=True)
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
        return frozenset(e for e, m in self.edge_multiplicity.items() if m == 2)

    @property
    def single_edges(self) -> FrozenSet[Tuple[int, int]]:
        """Set of single edges."""
        return frozenset(e for e, m in self.edge_multiplicity.items() if m == 1)

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
    unique_edges: set = set()

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

    # Primal is a simple graph, use original extract_faces.
    primal_adj_1based = primal_data["adjacency_list"]
    primal_num_vertices = primal_data["vertex_count"]
    primal_embedding = GraphConverter.to_zero_based_embedding(primal_adj_1based)
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
) -> List[PlaneGraph]:
    """Enumerates all n-vertex 4-regular planar multigraphs."""
    if verbose:
        print(f"[Plantri] Enumerating {dual_vertex_count}-vertex 4-regular planar multigraphs...")

    enumerator = QuadrangulationEnumerator()
    graphs = []

    for graph_id, (primal_data, dual_data) in enumerate(enumerator.generate_pairs(dual_vertex_count)):
        graph = _build_plane_graph(primal_data, dual_data, graph_id)

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
) -> Iterator[PlaneGraph]:
    """Iterates over PlaneGraph objects."""
    enumerator = QuadrangulationEnumerator()

    for graph_id, (primal_data, dual_data) in enumerate(
        enumerator.generate_pairs(dual_vertex_count)
    ):
        graph = _build_plane_graph(primal_data, dual_data, graph_id)

        if validate:
            is_valid, _ = graph.validate()
            if not is_valid:
                continue

        yield graph


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

    # Extract metadata.
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


# NumPy format constants
_NUMPY_VERTEX_DTYPE = np.uint16  # Supports up to 65535 vertices (uint8 only supports 255)
_NUMPY_MULT_DTYPE = np.uint8     # Edge multiplicity (1 or 2)
_NUMPY_FACE_PADDING = 65535      # Padding value for faces array


def _embedding_to_numpy(graph: "PlaneGraph") -> np.ndarray:
    """Convert single PlaneGraph embedding to numpy array."""
    num_vertices = graph.num_vertices
    result = np.zeros((num_vertices, 4), dtype=_NUMPY_VERTEX_DTYPE)
    for v in range(num_vertices):
        result[v] = graph.embedding[v]
    return result


def _embeddings_to_numpy(graphs: List["PlaneGraph"]) -> np.ndarray:
    """Convert list of PlaneGraphs to batched numpy array."""
    if not graphs:
        return np.zeros((0, 0, 4), dtype=_NUMPY_VERTEX_DTYPE)
    num_graphs = len(graphs)
    num_vertices = graphs[0].num_vertices
    result = np.zeros((num_graphs, num_vertices, 4), dtype=_NUMPY_VERTEX_DTYPE)
    for i, graph in enumerate(graphs):
        result[i] = _embedding_to_numpy(graph)
    return result


def _edge_multiplicity_to_numpy(graph: "PlaneGraph") -> np.ndarray:
    """Convert edge_multiplicity to adjacency matrix."""
    n = graph.num_vertices
    result = np.zeros((n, n), dtype=_NUMPY_MULT_DTYPE)
    for (u, v), mult in graph.edge_multiplicity.items():
        result[u, v] = mult
        result[v, u] = mult
    return result


def _faces_to_numpy(graphs: List["PlaneGraph"]) -> tuple[np.ndarray, np.ndarray]:
    """Convert faces to numpy arrays with padding."""
    if not graphs:
        return (
            np.zeros((0, 0, 0), dtype=_NUMPY_VERTEX_DTYPE),
            np.zeros((0, 0), dtype=_NUMPY_VERTEX_DTYPE),
        )

    num_graphs = len(graphs)
    num_faces = len(graphs[0].faces)  # n + 2 by Euler's formula
    max_face_length = max(len(f) for g in graphs for f in g.faces)

    faces = np.full(
        (num_graphs, num_faces, max_face_length),
        _NUMPY_FACE_PADDING,
        dtype=_NUMPY_VERTEX_DTYPE,
    )
    face_lengths = np.zeros((num_graphs, num_faces), dtype=_NUMPY_VERTEX_DTYPE)

    for i, graph in enumerate(graphs):
        for j, face in enumerate(graph.faces):
            face_len = len(face)
            faces[i, j, :face_len] = face
            face_lengths[i, j] = face_len

    return faces, face_lengths


def _save_numpy(
    graphs: List["PlaneGraph"],
    filepath: Path,
    compressed: bool,
) -> None:
    """Save graphs to NumPy structured format (.npz)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compressed else np.savez

    if not graphs:
        save_fn(
            filepath,
            embeddings=np.zeros((0, 0, 4), dtype=_NUMPY_VERTEX_DTYPE),
            edge_multiplicity=np.zeros((0, 0, 0), dtype=_NUMPY_MULT_DTYPE),
            graph_ids=np.zeros((0,), dtype=np.uint32),
            faces=np.zeros((0, 0, 0), dtype=_NUMPY_VERTEX_DTYPE),
            face_lengths=np.zeros((0, 0), dtype=_NUMPY_VERTEX_DTYPE),
        )
        return

    num_graphs = len(graphs)
    num_vertices = graphs[0].num_vertices

    embeddings = _embeddings_to_numpy(graphs)
    graph_ids = np.array([g.graph_id for g in graphs], dtype=np.uint32)
    edge_mult = np.zeros((num_graphs, num_vertices, num_vertices), dtype=_NUMPY_MULT_DTYPE)
    for i, graph in enumerate(graphs):
        edge_mult[i] = _edge_multiplicity_to_numpy(graph)

    faces, face_lengths = _faces_to_numpy(graphs)

    save_fn(
        filepath,
        embeddings=embeddings,
        edge_multiplicity=edge_mult,
        graph_ids=graph_ids,
        faces=faces,
        face_lengths=face_lengths,
    )


def _load_numpy(filepath: Path) -> Dict[str, Any]:
    """Load graphs from NumPy structured format (.npz)."""
    with np.load(filepath) as data:
        result = {
            "embeddings": data["embeddings"],
            "edge_multiplicity": data["edge_multiplicity"],
            "graph_ids": data["graph_ids"],
        }
        # v2 format includes faces
        if "faces" in data and "face_lengths" in data:
            result["faces"] = data["faces"]
            result["face_lengths"] = data["face_lengths"]
        return result


def numpy_to_plane_graphs(data: Dict[str, Any]) -> List["PlaneGraph"]:
    """Convert numpy arrays back to PlaneGraph objects.

    Args:
        data: Dict with keys 'embeddings', 'edge_multiplicity', 'graph_ids',
              and optionally 'faces', 'face_lengths' (v2 format).

    Returns:
        List of PlaneGraph objects.
    """
    embeddings = data["embeddings"]
    edge_mult_arrays = data["edge_multiplicity"]
    graph_ids = data["graph_ids"]

    # Check for v2 format with pre-stored faces
    faces_array = data.get("faces")
    face_lengths_array = data.get("face_lengths")
    has_faces = faces_array is not None and face_lengths_array is not None

    num_graphs = embeddings.shape[0]
    if num_graphs == 0:
        return []

    num_vertices = embeddings.shape[1]
    graphs: List[PlaneGraph] = []

    for i in range(num_graphs):
        # Convert embedding array to dict
        embedding: Dict[int, Tuple[int, ...]] = {}
        for v in range(num_vertices):
            neighbors = tuple(int(x) for x in embeddings[i, v])
            embedding[v] = neighbors

        # Convert edge_multiplicity matrix to dict
        edge_multiplicity: Dict[Tuple[int, int], int] = {}
        edge_mult_matrix = edge_mult_arrays[i]
        for u in range(num_vertices):
            for v in range(u + 1, num_vertices):
                mult = int(edge_mult_matrix[u, v])
                if mult > 0:
                    edge_multiplicity[(u, v)] = mult

        # Build edges from edge_multiplicity
        edges: List[Tuple[int, int]] = []
        for (u, v), mult in edge_multiplicity.items():
            for _ in range(mult):
                edges.append((u, v))

        # Get faces - either from stored data or compute
        if has_faces:
            # Type narrowing for linter
            assert faces_array is not None and face_lengths_array is not None
            # Reconstruct faces from stored arrays (fast path)
            num_faces = face_lengths_array.shape[1]
            faces_list: List[Tuple[int, ...]] = []
            for j in range(num_faces):
                face_len = int(face_lengths_array[i, j])
                face = tuple(int(x) for x in faces_array[i, j, :face_len])
                faces_list.append(face)
            faces = tuple(faces_list)
        else:
            # Compute faces from embedding (slow path for v1 format)
            faces = GraphConverter.extract_faces(embedding, edge_multiplicity)

        # Create PlaneGraph with dummy primal data (not stored in numpy format)
        graph = PlaneGraph(
            num_vertices=num_vertices,
            edges=tuple(edges),
            edge_multiplicity=edge_multiplicity,
            embedding=embedding,
            faces=faces,
            primal_num_vertices=num_vertices + 2,
            primal_embedding={},
            primal_faces=(),
            graph_id=int(graph_ids[i]),
        )
        graphs.append(graph)

    return graphs


def save_graphs_to_cache(
    graphs: List[PlaneGraph],
    filepath: Union[str, Path],
    *,
    dual_vertex_count: int = 0,
    compress: bool = True,
    compress_level: int = 6,
    use_json: bool = False,
    use_numpy: bool = False,
) -> Path:
    """Save graph list to cache file with atomic write."""
    filepath = Path(filepath)

    if use_numpy:
        _save_numpy(graphs, filepath, compressed=compress)
    elif use_json:
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


# Type alias for numpy cache return type
NumpyCacheData = Dict[str, Any]  # {"embeddings": ndarray, "edge_multiplicity": ndarray, "graph_ids": ndarray}


@overload
def load_graphs_from_cache(
    filepath: Union[str, Path],
    *,
    max_count: Optional[int] = None,
    use_json: bool = False,
    use_numpy: Literal[True],
    trusted: bool = False,
    safe_mode: bool = True,
) -> Tuple[NumpyCacheData, CacheMetadata]: ...


@overload
def load_graphs_from_cache(
    filepath: Union[str, Path],
    *,
    max_count: Optional[int] = None,
    use_json: bool = False,
    use_numpy: Literal[False] = False,
    trusted: bool = False,
    safe_mode: bool = True,
) -> Tuple[List[PlaneGraph], CacheMetadata]: ...


def load_graphs_from_cache(
    filepath: Union[str, Path],
    *,
    max_count: Optional[int] = None,
    use_json: bool = False,
    use_numpy: bool = False,
    trusted: bool = False,
    safe_mode: bool = True,
) -> Union[Tuple[NumpyCacheData, CacheMetadata], Tuple[List[PlaneGraph], CacheMetadata]]:
    """Load graphs from cache file with security checks."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cache file not found: {filepath}")

    if use_numpy:
        data = _load_numpy(filepath)
        # Return numpy data with minimal metadata
        num_graphs = len(data["embeddings"])
        metadata = CacheMetadata(
            format_version=_CACHE_FORMAT_VERSION,
            pyplantri_version=_get_version(),
            dual_vertex_count=data["embeddings"].shape[1] if num_graphs > 0 else 0,
            graph_count=num_graphs,
            pickle_protocol=0,
        )
        return data, metadata
    elif use_json:
        return _load_json(filepath, max_count)
    else:
        if not trusted:
            raise ValueError(
                "pickle file loading requires trusted=True. "
                "pickle.load() can execute arbitrary code, so only use "
                "with files from trusted sources. "
                "Safer alternative: use_json=True or use_numpy=True"
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
