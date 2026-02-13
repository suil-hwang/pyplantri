# src/pyplantri/plane_graph.py
import gzip
import json
import logging
import pickle
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple, Union

from .plane_graph_enumeration import (
    EnumerationTiming,
    FilteredEnumerationResult,
    _build_plane_graph,
    enumerate_plane_graphs,
    enumerate_plane_graphs_filtered,
    enumerate_plane_graphs_parallel,
    iter_plane_graphs,
)

logger = logging.getLogger(__name__)

Embedding = Tuple[Tuple[int, ...], ...]


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
    embedding: Embedding  # CW cyclic order at each vertex
    faces: Tuple[Tuple[int, ...], ...]

    primal_num_vertices: int
    primal_embedding: Embedding
    primal_faces: Tuple[Tuple[int, ...], ...]

    graph_id: int = 0
    _double_edges_cache: Optional[FrozenSet[Tuple[int, int]]] = field(
        default=None, init=False, repr=False, compare=False
    )
    _single_edges_cache: Optional[FrozenSet[Tuple[int, int]]] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Normalize embedding containers to dense tuple-of-tuples."""
        normalized_embedding = self._normalize_embedding(
            self.embedding,
            expected_size=self.num_vertices,
        )
        normalized_primal_embedding = self._normalize_embedding(
            self.primal_embedding,
            expected_size=self.primal_num_vertices,
        )
        object.__setattr__(self, "embedding", normalized_embedding)
        object.__setattr__(self, "primal_embedding", normalized_primal_embedding)

    @staticmethod
    def _normalize_embedding(
        embedding: Union[Dict[int, Tuple[int, ...]], Embedding, List[Tuple[int, ...]]],
        *,
        expected_size: int = 0,
    ) -> Embedding:
        """Convert sparse/dict embedding into dense 0..n-1 tuple-of-tuples."""
        if isinstance(embedding, dict):
            size = max(expected_size, 0)
            if embedding:
                max_index = max(int(v) for v in embedding.keys()) + 1
                size = max(size, max_index)
            dense: List[Tuple[int, ...]] = [tuple() for _ in range(size)]
            for vertex, neighbors in embedding.items():
                idx = int(vertex)
                if idx < 0:
                    continue
                if idx >= len(dense):
                    dense.extend(tuple() for _ in range(idx + 1 - len(dense)))
                dense[idx] = tuple(int(u) for u in neighbors)
            return tuple(dense)

        dense_embedding: Embedding = tuple(
            tuple(int(u) for u in neighbors) for neighbors in embedding
        )
        if expected_size > len(dense_embedding):
            dense_embedding = dense_embedding + tuple(
                tuple() for _ in range(expected_size - len(dense_embedding))
            )
        return dense_embedding

    @staticmethod
    def _iter_embedding_items(embedding: Union[Dict[int, Tuple[int, ...]], Embedding]) -> Iterator[Tuple[int, Tuple[int, ...]]]:
        """Iterate (vertex, neighbors) for dict or dense embedding containers."""
        if isinstance(embedding, dict):
            for v, neighbors in embedding.items():
                yield int(v), tuple(neighbors)
        else:
            for v, neighbors in enumerate(embedding):
                yield v, neighbors

    @staticmethod
    def _iter_embedding_values(embedding: Union[Dict[int, Tuple[int, ...]], Embedding]) -> Iterator[Tuple[int, ...]]:
        """Iterate neighbor tuples for dict or dense embedding containers."""
        if isinstance(embedding, dict):
            for neighbors in embedding.values():
                yield tuple(neighbors)
        else:
            for neighbors in embedding:
                yield neighbors

    @staticmethod
    def _has_vertex(embedding: Union[Dict[int, Tuple[int, ...]], Embedding], vertex: int) -> bool:
        """Return True if vertex exists in embedding."""
        if isinstance(embedding, dict):
            return vertex in embedding
        return 0 <= vertex < len(embedding)

    @staticmethod
    def _neighbors_of(
        embedding: Union[Dict[int, Tuple[int, ...]], Embedding],
        vertex: int,
    ) -> Tuple[int, ...]:
        """Get neighbors for vertex from dict or dense embedding containers."""
        if isinstance(embedding, dict):
            neighbors = embedding.get(vertex, tuple())
            return tuple(neighbors)
        if 0 <= vertex < len(embedding):
            return embedding[vertex]
        return tuple()

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
        return all(len(neighbors) == 4 for neighbors in self._iter_embedding_values(self.embedding))

    @property
    def is_loop_free(self) -> bool:
        """Whether graph has no self-loops."""
        return all(
            v not in neighbors
            for v, neighbors in self._iter_embedding_items(self.embedding)
        )

    def get_neighbors_cw(self, vertex: int) -> Tuple[int, ...]:
        """Gets CW-ordered neighbors of a vertex."""
        return self._neighbors_of(self.embedding, vertex)

    def get_neighbors_ccw(self, vertex: int) -> Tuple[int, ...]:
        """Gets CCW-ordered neighbors of a vertex."""
        return tuple(reversed(self.get_neighbors_cw(vertex)))

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
            if not self._has_vertex(self.embedding, v):
                errors.append(f"Vertex {v} missing from embedding")
                continue
            neighbors_v = self._neighbors_of(self.embedding, v)
            if len(neighbors_v) != 4:
                errors.append(
                    f"Vertex {v} has degree {len(neighbors_v)}, expected 4"
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

        for v, neighbors in self._iter_embedding_items(self.embedding):
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
                str(v): list(neighbors)
                for v, neighbors in self._iter_embedding_items(self.embedding)
            },
            "faces": [list(f) for f in self.faces],
            "primal_num_vertices": self.primal_num_vertices,
            "primal_embedding": {
                str(v): list(neighbors)
                for v, neighbors in self._iter_embedding_items(self.primal_embedding)
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
        embedding_payload = data.get("embedding", {})
        primal_embedding_payload = data.get("primal_embedding", {})
        return cls(
            num_vertices=data["num_vertices"],
            edges=tuple(parsed_edges),
            edge_multiplicity=parsed_edge_multiplicity,
            embedding=cls._normalize_embedding(
                embedding_payload,
                expected_size=data["num_vertices"],
            ),
            faces=tuple(tuple(f) for f in data["faces"]),
            primal_num_vertices=data.get("primal_num_vertices", 0),
            primal_embedding=cls._normalize_embedding(
                primal_embedding_payload,
                expected_size=data.get("primal_num_vertices", 0),
            ),
            primal_faces=tuple(
                tuple(f) for f in data.get("primal_faces", [])
            ),
            graph_id=data.get("graph_id", 0),
        )


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
