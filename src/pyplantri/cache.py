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
_CACHE_FORMAT_VERSION = 4


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

    # Redirect classes that moved between modules during refactoring.
    _MODULE_REDIRECTS: dict[tuple[str, str], tuple[str, str]] = {
        ("pyplantri.plane_graph", "CacheMetadata"): ("pyplantri.cache", "CacheMetadata"),
    }

    def __init__(self, file: Any):
        """Initialize SafeUnpickler."""
        super().__init__(file)

    def find_class(self, module: str, name: str) -> Any:
        """Override to restrict loadable classes."""
        original_module, original_name = module, name

        # Redirect classes that moved to a different module before whitelist checks.
        module, name = self._MODULE_REDIRECTS.get((module, name), (module, name))
        allowed = self.SAFE_MODULES.get(module, set())

        if name not in allowed:
            raise pickle.UnpicklingError(
                f"Attempted to unpickle forbidden class: "
                f"{original_module}.{original_name}\n"
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
    if metadata.format_version > _CACHE_FORMAT_VERSION:
        raise ValueError(
            "cache: unsupported format_version "
            f"{metadata.format_version} > {_CACHE_FORMAT_VERSION} ({filepath})"
        )
    if metadata.format_version < _CACHE_FORMAT_VERSION:
        logger.warning(
            "Cache file '%s' uses older format version %d "
            "(current: %d). Consider regenerating.",
            filepath,
            metadata.format_version,
            _CACHE_FORMAT_VERSION,
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
    if isinstance(raw_metadata, dict):
        metadata = CacheMetadata(**raw_metadata)
    elif isinstance(raw_metadata, CacheMetadata):
        metadata = raw_metadata
    else:
        raise ValueError(
            f"cache: invalid metadata type {type(raw_metadata).__name__}"
        )
    _validate_format_version(metadata, filepath)
    return metadata


def _coerce_json_metadata(raw_metadata: Any, filepath: Path) -> CacheMetadata:
    """Coerce JSON metadata payload into CacheMetadata with legacy-safe defaults."""
    if not isinstance(raw_metadata, dict):
        raise ValueError(
            f"cache: JSON metadata must be object ({filepath})"
        )
    metadata = CacheMetadata(
        format_version=raw_metadata.get("format_version", 0),
        pyplantri_version=raw_metadata.get("pyplantri_version", "unknown"),
        dual_vertex_count=raw_metadata.get("dual_vertex_count", 0),
        graph_count=raw_metadata.get("graph_count", 0),
        pickle_protocol=0,
    )
    _validate_format_version(metadata, filepath)
    return metadata


def _normalize_loaded_graphs(
    raw_graphs: Any,
    *,
    filepath: Path,
    max_count: int | None,
) -> list[PlaneGraph]:
    """Normalize cached graph payloads into canonical PlaneGraph instances."""
    if not isinstance(raw_graphs, (list, tuple)):
        raise ValueError(
            f"cache: graphs payload must be list/tuple ({filepath})"
        )

    graph_items = list(raw_graphs[:max_count] if max_count is not None else raw_graphs)
    normalized_graphs: list[PlaneGraph] = []
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
    return normalized_graphs


def _save_pickle(
    graphs: list[PlaneGraph],
    filepath: Path,
    dual_vertex_count: int,
    compress: bool,
    compress_level: int,
) -> None:
    """Pickle serialization with atomic write and optional gzip compression."""
    protocol = pickle.HIGHEST_PROTOCOL
    payload = {
        "metadata": _build_cache_metadata(
            dual_vertex_count=dual_vertex_count,
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
    dual_vertex_count: int,
) -> None:
    """JSON serialization with atomic write and compact format."""
    payload = {
        "metadata": _build_cache_metadata(
            dual_vertex_count=dual_vertex_count,
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
    graphs = _normalize_loaded_graphs(
        payload["graphs"],
        filepath=filepath,
        max_count=max_count,
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
    graphs = _normalize_loaded_graphs(
        payload["graphs"],
        filepath=filepath,
        max_count=max_count,
    )
    return graphs, metadata


def save_graphs_to_cache(
    graphs: list[PlaneGraph],
    filepath: str | Path,
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
