# src/pyplantri/plane_graph.py
import gzip
import json
import logging
import pickle
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple, Union

from .core import GraphConverter, QuadrangulationEnumerator

logger = logging.getLogger(__name__)


class SecurityWarning(UserWarning):
    """Warning for security-related issues in pickle deserialization."""

    pass


@dataclass(frozen=True, slots=True)
class PlaneGraph:
    """Plane graph (embedded planar graph) with fixed combinatorial embedding.

    A plane graph is a planar graph with a specific embedding on the sphere,
    represented by clockwise cyclic edge ordering at each vertex. Generated
    by plantri (Brinkmann & McKay).

    Terminology:
        - Planar graph: A graph that CAN be embedded in the plane (abstract).
        - Plane graph: A planar graph WITH a fixed embedding (concrete).

    This class represents an immutable 4-regular plane multigraph (dual graph Q*).
    Primal graph data is computed lazily on demand. All indices are 0-based.

    Dual Graph (Q*):
        - 4-regular plane multigraph (allows double edges, no loops).
        - num_vertices = n (dual vertices).
        - faces = n + 2 (dual faces = primal vertices).

    Primal Graph (Q) - computed lazily:
        - Simple quadrangulation (no loops/multi-edges, all faces are 4-gons).
        - primal_num_vertices = n + 2 (primal vertices = dual faces).
        - primal_faces = n (primal faces = dual vertices).
    """

    num_vertices: int
    edges: Tuple[Tuple[int, int], ...]
    edge_multiplicity: Dict[Tuple[int, int], int]
    embedding: Dict[int, Tuple[int, ...]]  # CW cyclic order at each vertex
    faces: Tuple[Tuple[int, ...], ...]
    graph_id: int = 0

    @property
    def primal_num_vertices(self) -> int:
        """Number of primal vertices (= num_faces by Euler's formula)."""
        return len(self.faces)

    @property
    def primal_faces(self) -> Tuple[Tuple[int, ...], ...]:
        """Primal faces (= dual vertices, computed lazily)."""
        # Each dual vertex becomes a primal face
        # The primal face contains the dual faces (indices) adjacent to the dual vertex
        return tuple(
            tuple(sorted(set(
                face_idx
                for face_idx, face in enumerate(self.faces)
                if v in face
            )))
            for v in range(self.num_vertices)
        )

    @property
    def primal_embedding(self) -> Dict[int, Tuple[int, ...]]:
        """Primal embedding (cyclic order at each primal vertex, computed lazily)."""
        # Primal vertices = dual faces
        # For each primal vertex (face), get adjacent primal vertices in cyclic order
        embedding: Dict[int, Tuple[int, ...]] = {}
        for face_idx, face in enumerate(self.faces):
            # Adjacent primal vertices are faces sharing an edge with this face
            neighbors: List[int] = []
            for i, v in enumerate(face):
                next_v = face[(i + 1) % len(face)]
                # Find the other face sharing edge (v, next_v)
                for other_idx, other_face in enumerate(self.faces):
                    if other_idx != face_idx:
                        if v in other_face and next_v in other_face:
                            neighbors.append(other_idx)
                            break
            embedding[face_idx] = tuple(neighbors)
        return embedding

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

    def to_dict(self, include_primal: bool = False) -> Dict:
        """Converts to dictionary for JSON serialization.

        Args:
            include_primal: If True, include computed primal data (slower).
        """
        result = {
            "num_vertices": self.num_vertices,
            "edges": list(self.edges),
            "edge_multiplicity": {
                f"{u},{v}": m for (u, v), m in self.edge_multiplicity.items()
            },
            "embedding": {
                str(v): list(neighbors) for v, neighbors in self.embedding.items()
            },
            "faces": [list(f) for f in self.faces],
            "graph_id": self.graph_id,
        }
        if include_primal:
            result["primal_num_vertices"] = self.primal_num_vertices
            result["primal_embedding"] = {
                str(v): list(neighbors)
                for v, neighbors in self.primal_embedding.items()
            }
            result["primal_faces"] = [list(f) for f in self.primal_faces]
        return result

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

    # Primal data is now computed lazily via properties
    return PlaneGraph(
        num_vertices=dual_vertex_count,
        edges=edges,
        edge_multiplicity=edge_multiplicity,
        embedding=embedding,
        faces=faces,
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