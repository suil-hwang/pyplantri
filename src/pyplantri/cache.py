# src/pyplantri/cache.py
from __future__ import annotations

import gzip
import json
import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .plane_graph import FrozenEdgeMultiplicity, PlaneGraph

logger = logging.getLogger(__name__)

# Cache format version. Increment when PlaneGraph fields change.
_CACHE_FORMAT_VERSION = 5


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
    SAFE_MODULES: dict[str, set[str]] = {
        "pyplantri.plane_graph": {
            "PlaneGraph",
            "FrozenEdgeMultiplicity",
        },
        "pyplantri.cache": {
            "CacheMetadata",
        },
        "builtins": {"tuple", "dict", "list", "int", "str", "frozenset"},
        "collections": {"defaultdict"},
    }

    def __init__(self, file: Any):
        """Initialize SafeUnpickler."""
        super().__init__(file)

    def find_class(self, module: str, name: str) -> Any:
        """Override to restrict loadable classes."""
        allowed = self.SAFE_MODULES.get(module, set())

        if name not in allowed:
            raise pickle.UnpicklingError(
                f"Attempted to unpickle forbidden class: "
                f"{module}.{name}\n"
                f"Only PlaneGraph and built-in types are allowed.\n"
                "This may indicate a malicious or corrupted file."
            )

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
    if metadata.format_version != _CACHE_FORMAT_VERSION:
        raise ValueError(
            "cache: unsupported format_version "
            f"{metadata.format_version} != {_CACHE_FORMAT_VERSION} ({filepath})"
        )


def _build_cache_metadata(
    *,
    dual_vertex_count: int,
    graph_count: int,
    pickle_protocol: int,
) -> CacheMetadata:
    """Build canonical cache metadata for the current pyplantri version."""
    return CacheMetadata(
        format_version=_CACHE_FORMAT_VERSION,
        pyplantri_version=_get_version(),
        dual_vertex_count=dual_vertex_count,
        graph_count=graph_count,
        pickle_protocol=pickle_protocol,
    )


def _atomic_write(
    filepath: Path,
    *,
    mode: str,
    writer: Any,
    encoding: str | None = None,
) -> None:
    """Write a file atomically via a same-directory temp file and rename."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")

    try:
        open_kwargs: dict[str, Any] = {"mode": mode}
        if "b" not in mode:
            open_kwargs["encoding"] = encoding or "utf-8"
        with open(fd, **open_kwargs) as f:
            writer(f)
        Path(tmp_path).replace(filepath)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _coerce_pickle_metadata(raw_metadata: Any, filepath: Path) -> CacheMetadata:
    """Coerce pickle metadata payload into CacheMetadata."""
    if not isinstance(raw_metadata, CacheMetadata):
        raise ValueError(
            f"cache: invalid metadata type {type(raw_metadata).__name__}"
        )
    metadata = raw_metadata
    _validate_format_version(metadata, filepath)
    return metadata


def _coerce_json_metadata(raw_metadata: Any, filepath: Path) -> CacheMetadata:
    """Coerce canonical JSON metadata payload into CacheMetadata."""
    if not isinstance(raw_metadata, dict):
        raise ValueError(
            f"cache: JSON metadata must be object ({filepath})"
        )
    required_keys = (
        "format_version",
        "pyplantri_version",
        "dual_vertex_count",
        "graph_count",
        "pickle_protocol",
    )
    missing_keys = [key for key in required_keys if key not in raw_metadata]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"cache: JSON metadata missing keys: {missing} ({filepath})"
        )
    metadata = CacheMetadata(
        format_version=int(raw_metadata["format_version"]),
        pyplantri_version=str(raw_metadata["pyplantri_version"]),
        dual_vertex_count=int(raw_metadata["dual_vertex_count"]),
        graph_count=int(raw_metadata["graph_count"]),
        pickle_protocol=int(raw_metadata["pickle_protocol"]),
    )
    _validate_format_version(metadata, filepath)
    return metadata


def _raw_graph_dual_vertex_count(raw_graph: Any, filepath: Path) -> int | None:
    """Extract dual vertex count from a raw cache payload item."""
    if isinstance(raw_graph, PlaneGraph):
        return raw_graph.dual_num_vertices
    if isinstance(raw_graph, dict):
        if "dual_num_vertices" not in raw_graph:
            raise ValueError(
                f"cache: graph payload missing dual_num_vertices ({filepath})"
            )
        return int(raw_graph["dual_num_vertices"])
    return None


def _infer_dual_vertex_count_for_save(
    graphs: list[PlaneGraph],
    dual_vertex_count: int | None,
) -> int:
    """Infer or validate the dual vertex count when saving a cache file."""
    if graphs:
        graph_dual_counts = {graph.dual_num_vertices for graph in graphs}
        if len(graph_dual_counts) != 1:
            raise ValueError(
                "cache: graphs contain mixed dual_vertex_count values; "
                "cannot save a heterogeneous cache"
            )
        inferred_dual_vertex_count = next(iter(graph_dual_counts))
        if dual_vertex_count is None:
            return inferred_dual_vertex_count
        if dual_vertex_count != inferred_dual_vertex_count:
            raise ValueError(
                "cache: dual_vertex_count mismatch: "
                f"explicit {dual_vertex_count} != inferred {inferred_dual_vertex_count}"
            )
        return dual_vertex_count

    if dual_vertex_count is None:
        raise ValueError(
            "cache: dual_vertex_count must be provided when graphs is empty"
        )
    return dual_vertex_count


def _normalize_loaded_graphs(
    raw_graphs: Any,
    *,
    filepath: Path,
    max_count: int | None,
) -> tuple[list[PlaneGraph], int, int | None]:
    """Normalize cached graph payloads into canonical PlaneGraph instances."""
    if not isinstance(raw_graphs, (list, tuple)):
        raise ValueError(
            f"cache: graphs payload must be list/tuple ({filepath})"
        )

    raw_graph_count = len(raw_graphs)
    inferred_dual_vertex_count: int | None = None
    graph_items = list(raw_graphs[:max_count] if max_count is not None else raw_graphs)
    normalized_graphs: list[PlaneGraph] = []

    for raw_graph in raw_graphs:
        graph_dual_vertex_count = _raw_graph_dual_vertex_count(raw_graph, filepath)
        if graph_dual_vertex_count is None:
            raise ValueError(
                f"cache: unsupported graph payload type {type(raw_graph).__name__}"
            )
        if inferred_dual_vertex_count is None:
            inferred_dual_vertex_count = graph_dual_vertex_count
        elif graph_dual_vertex_count != inferred_dual_vertex_count:
            raise ValueError(
                "cache: graphs payload contains mixed dual_vertex_count values "
                f"({filepath})"
            )

    for graph in graph_items:
        if isinstance(graph, PlaneGraph):
            if isinstance(graph.dual_edge_multiplicity, FrozenEdgeMultiplicity):
                normalized_graphs.append(graph)
            else:
                normalized_graphs.append(PlaneGraph.from_dict(graph.to_dict()))
        elif isinstance(graph, dict):
            normalized_graphs.append(PlaneGraph.from_dict(graph))
        else:
            raise ValueError(
                f"cache: unsupported graph payload type {type(graph).__name__}"
            )
    return normalized_graphs, raw_graph_count, inferred_dual_vertex_count


def _validate_metadata_against_payload(
    metadata: CacheMetadata,
    *,
    filepath: Path,
    raw_graph_count: int,
    inferred_dual_vertex_count: int | None,
) -> None:
    """Validate metadata against the serialized payload."""
    if metadata.graph_count != raw_graph_count:
        raise ValueError(
            "cache: metadata.graph_count mismatch: "
            f"{metadata.graph_count} != {raw_graph_count} ({filepath})"
        )
    if (
        inferred_dual_vertex_count is not None
        and metadata.dual_vertex_count != inferred_dual_vertex_count
    ):
        raise ValueError(
            "cache: metadata.dual_vertex_count mismatch: "
            f"{metadata.dual_vertex_count} != {inferred_dual_vertex_count} ({filepath})"
        )


def _save_pickle(
    graphs: list[PlaneGraph],
    filepath: Path,
    dual_vertex_count: int | None,
    compress: bool,
    compress_level: int,
) -> None:
    """Pickle serialization with atomic write and optional gzip compression."""
    protocol = pickle.HIGHEST_PROTOCOL
    resolved_dual_vertex_count = _infer_dual_vertex_count_for_save(
        graphs,
        dual_vertex_count,
    )
    payload = {
        "metadata": _build_cache_metadata(
            dual_vertex_count=resolved_dual_vertex_count,
            graph_count=len(graphs),
            pickle_protocol=protocol,
        ),
        "graphs": graphs,
    }

    def _write_pickle(file_obj: Any) -> None:
        if compress:
            with gzip.GzipFile(
                fileobj=file_obj, mode="wb", compresslevel=compress_level
            ) as gz:
                pickle.dump(payload, gz, protocol=protocol)
        else:
            pickle.dump(payload, file_obj, protocol=protocol)

    _atomic_write(filepath, mode="wb", writer=_write_pickle)


def _save_json(
    graphs: list[PlaneGraph],
    filepath: Path,
    dual_vertex_count: int | None,
) -> None:
    """JSON serialization with atomic write and compact format."""
    resolved_dual_vertex_count = _infer_dual_vertex_count_for_save(
        graphs,
        dual_vertex_count,
    )
    payload = {
        "metadata": _build_cache_metadata(
            dual_vertex_count=resolved_dual_vertex_count,
            graph_count=len(graphs),
            pickle_protocol=0,
        ).__dict__,
        "graphs": [graph.to_dict() for graph in graphs],
    }

    def _write_json(file_obj: Any) -> None:
        # Compact format for cache (no indent).
        json.dump(payload, file_obj, separators=(",", ":"))

    _atomic_write(
        filepath,
        mode="w",
        writer=_write_json,
        encoding="utf-8",
    )


def _load_pickle(
    filepath: Path,
    max_count: int | None,
    safe_mode: bool,
) -> tuple[list[PlaneGraph], CacheMetadata]:
    """Pickle deserialization with optional safety restrictions."""
    with open(filepath, "rb") as f:
        # Auto-detect gzip by magic number (0x1f 0x8b).
        header = f.read(2)
        f.seek(0)

        if header == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                if safe_mode:
                    payload = SafeUnpickler(gz).load()
                else:
                    payload = pickle.load(gz)
        else:
            if safe_mode:
                payload = SafeUnpickler(f).load()
            else:
                payload = pickle.load(f)

    # Extract and validate metadata.
    if not isinstance(payload, dict):
        raise ValueError(
            "cache: unsupported payload type "
            f"{type(payload).__name__}; expected dict with metadata+graphs"
        )
    if "metadata" not in payload or "graphs" not in payload:
        raise ValueError(
            f"cache: missing metadata/graphs keys ({filepath})"
        )

    metadata = _coerce_pickle_metadata(payload["metadata"], filepath)
    graphs, raw_graph_count, inferred_dual_vertex_count = _normalize_loaded_graphs(
        payload["graphs"],
        filepath=filepath,
        max_count=max_count,
    )
    _validate_metadata_against_payload(
        metadata,
        filepath=filepath,
        raw_graph_count=raw_graph_count,
        inferred_dual_vertex_count=inferred_dual_vertex_count,
    )
    return graphs, metadata


def _load_json(
    filepath: Path,
    max_count: int | None,
) -> tuple[list[PlaneGraph], CacheMetadata]:
    """JSON deserialization."""
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(
            f"cache: JSON payload must be object with metadata+graphs ({filepath})"
        )
    if "metadata" not in payload or "graphs" not in payload:
        raise ValueError(
            f"cache: JSON payload missing metadata/graphs ({filepath})"
        )

    metadata = _coerce_json_metadata(payload["metadata"], filepath)
    graphs, raw_graph_count, inferred_dual_vertex_count = _normalize_loaded_graphs(
        payload["graphs"],
        filepath=filepath,
        max_count=max_count,
    )
    _validate_metadata_against_payload(
        metadata,
        filepath=filepath,
        raw_graph_count=raw_graph_count,
        inferred_dual_vertex_count=inferred_dual_vertex_count,
    )
    return graphs, metadata


def save_graphs_to_cache(
    graphs: list[PlaneGraph],
    filepath: str | Path,
    *,
    dual_vertex_count: int | None = None,
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
    filepath: str | Path,
    *,
    max_count: int | None = None,
    use_json: bool = False,
    trusted: bool = False,
    safe_mode: bool = True,
) -> tuple[list[PlaneGraph], CacheMetadata]:
    """Load graphs from cache file with security checks."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cache file not found: {filepath}")

    if use_json:
        return _load_json(filepath, max_count)
    else:
        if not trusted:
            raise ValueError(
                "cache: pickle loading requires trusted=True; "
                "use use_json=True for untrusted files"
            )
        return _load_pickle(filepath, max_count, safe_mode)
