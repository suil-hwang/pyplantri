# src/pyplantri/cache.py
from __future__ import annotations

import gzip
import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .plane_graph import PlaneGraph

logger = logging.getLogger(__name__)

# v5 remains readable; v6 records graph-ID provenance and primal inclusion.
CACHE_FORMAT_VERSION = 6
LEGACY_CACHE_FORMAT_VERSION = 5
SUPPORTED_CACHE_FORMAT_VERSIONS = frozenset(
    {LEGACY_CACHE_FORMAT_VERSION, CACHE_FORMAT_VERSION}
)
CacheGraphClass = Literal["quartic_multigraph", "simple_quartic"]
_SUPPORTED_GRAPH_CLASSES = frozenset({"quartic_multigraph", "simple_quartic"})


def _require_int_at_least(
    value: Any,
    *,
    field_name: str,
    minimum: int,
    filepath: Path | None = None,
) -> None:
    """Validate a non-boolean integer cache field."""
    if type(value) is not int or value < minimum:
        location = f" ({filepath})" if filepath is not None else ""
        raise ValueError(
            f"cache: {field_name} must be an integer >= {minimum}; got {value!r}{location}"
        )


@dataclass(frozen=True)
class CacheMetadata:
    """Versioned cache identity, size, and generation provenance."""

    format_version: int
    pyplantri_version: str
    dual_vertex_count: int
    graph_count: int
    pickle_protocol: int
    graph_class: str | None = None
    include_primal: bool | None = None

    def __setstate__(self, state: Any) -> None:
        """Restore current metadata and legacy v5 state."""
        if not isinstance(state, dict):
            raise TypeError(
                f"CacheMetadata pickle state must be dict; got {type(state).__name__}"
            )

        required_keys = (
            "format_version",
            "pyplantri_version",
            "dual_vertex_count",
            "graph_count",
            "pickle_protocol",
        )
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise KeyError(
                f"CacheMetadata pickle state missing keys: {', '.join(missing_keys)}"
            )

        for name in required_keys:
            object.__setattr__(self, name, state[name])
        object.__setattr__(self, "graph_class", state.get("graph_class"))
        object.__setattr__(self, "include_primal", state.get("include_primal"))


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler for pyplantri cache payloads."""

    # Current cache classes plus compatibility types used by legacy payloads.
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

    def find_class(self, module: str, name: str) -> Any:
        """Override to restrict loadable classes."""
        allowed = self.SAFE_MODULES.get(module, set())

        if name not in allowed:
            raise pickle.UnpicklingError(f"Forbidden pickle class {module}.{name}; cache may be malicious or corrupted")

        return super().find_class(module, name)


def _get_version() -> str:
    """Get pyplantri version string."""
    try:
        from pyplantri import __version__

        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def _validate_format_version(
    metadata: CacheMetadata,
    filepath: Path | None,
) -> None:
    """Validate cache format version compatibility."""
    if (
        type(metadata.format_version) is int
        and metadata.format_version in SUPPORTED_CACHE_FORMAT_VERSIONS
    ):
        return
    location = f" ({filepath})" if filepath is not None else ""
    supported = sorted(SUPPORTED_CACHE_FORMAT_VERSIONS)
    raise ValueError(
        f"cache: unsupported format_version {metadata.format_version!r}; supported={supported}{location}"
    )


def _validate_metadata_fields(
    metadata: CacheMetadata,
    *,
    filepath: Path | None = None,
) -> None:
    """Validate cache metadata fields without coercion."""
    _validate_format_version(metadata, filepath)
    location = f" ({filepath})" if filepath is not None else ""
    if type(metadata.pyplantri_version) is not str:
        raise ValueError(
            f"cache: metadata.pyplantri_version must be str; got {metadata.pyplantri_version!r}{location}"
        )
    for field_name, value, minimum in (
        ("metadata.dual_vertex_count", metadata.dual_vertex_count, 3),
        ("metadata.graph_count", metadata.graph_count, 0),
        ("metadata.pickle_protocol", metadata.pickle_protocol, 0),
    ):
        _require_int_at_least(
            value,
            field_name=field_name,
            minimum=minimum,
            filepath=filepath,
        )

    if metadata.format_version == LEGACY_CACHE_FORMAT_VERSION:
        if metadata.graph_class is not None or metadata.include_primal is not None:
            raise ValueError(
                f"cache: v5 metadata cannot declare graph_class/include_primal{location}"
            )
        return

    if (
        type(metadata.graph_class) is not str
        or metadata.graph_class not in _SUPPORTED_GRAPH_CLASSES
    ):
        raise ValueError(
            f"cache: metadata.graph_class must be one of {sorted(_SUPPORTED_GRAPH_CLASSES)}; got {metadata.graph_class!r}{location}"
        )
    if type(metadata.include_primal) is not bool:
        raise ValueError(
            f"cache: metadata.include_primal must be bool; got {metadata.include_primal!r}{location}"
        )


def validate_cache_metadata(
    metadata: CacheMetadata,
    *,
    expected_dual_vertex_count: int | None = None,
    expected_graph_class: CacheGraphClass | None = None,
    expected_include_primal: bool | None = None,
    filepath: str | Path | None = None,
) -> None:
    """Validate metadata and optional caller-owned cache expectations."""
    if not isinstance(metadata, CacheMetadata):
        raise ValueError(f"cache: invalid metadata type {type(metadata).__name__}")
    resolved_path = Path(filepath) if filepath is not None else None
    _validate_metadata_fields(metadata, filepath=resolved_path)
    location = f" ({resolved_path})" if resolved_path is not None else ""

    if (
        expected_dual_vertex_count is not None
        and metadata.dual_vertex_count != expected_dual_vertex_count
    ):
        raise ValueError(
            f"cache: metadata.dual_vertex_count {metadata.dual_vertex_count} != expected {expected_dual_vertex_count}{location}"
        )
    if expected_graph_class is not None:
        if (
            type(expected_graph_class) is not str
            or expected_graph_class not in _SUPPORTED_GRAPH_CLASSES
        ):
            raise ValueError(
                f"cache: unsupported expected graph_class {expected_graph_class!r}"
            )
        if (
            metadata.graph_class is not None
            and metadata.graph_class != expected_graph_class
        ):
            raise ValueError(
                f"cache: metadata.graph_class {metadata.graph_class!r} != expected {expected_graph_class!r}{location}"
            )
    if expected_include_primal is not None:
        if type(expected_include_primal) is not bool:
            raise ValueError(
                f"cache: expected_include_primal must be bool; got {expected_include_primal!r}"
            )
        if (
            metadata.include_primal is not None
            and metadata.include_primal is not expected_include_primal
        ):
            raise ValueError(
                f"cache: metadata.include_primal {metadata.include_primal!r} != expected {expected_include_primal!r}{location}"
            )


def _build_cache_metadata(
    *,
    dual_vertex_count: int,
    graph_count: int,
    pickle_protocol: int,
    graph_class: CacheGraphClass,
    include_primal: bool,
) -> CacheMetadata:
    """Build canonical cache metadata for the current pyplantri version."""
    metadata = CacheMetadata(
        format_version=CACHE_FORMAT_VERSION,
        pyplantri_version=_get_version(),
        dual_vertex_count=dual_vertex_count,
        graph_count=graph_count,
        pickle_protocol=pickle_protocol,
        graph_class=graph_class,
        include_primal=include_primal,
    )
    _validate_metadata_fields(metadata)
    return metadata


def _atomic_write(
    filepath: Path,
    *,
    writer: Any,
) -> None:
    """Write a binary file atomically via a same-directory temp file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")

    try:
        with open(fd, "wb") as f:
            writer(f)
        Path(tmp_path).replace(filepath)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def _validate_pickle_metadata(raw_metadata: Any, filepath: Path) -> CacheMetadata:
    """Validate and return pickle cache metadata."""
    if not isinstance(raw_metadata, CacheMetadata):
        raise ValueError(f"cache: invalid metadata type {type(raw_metadata).__name__}")
    validate_cache_metadata(raw_metadata, filepath=filepath)
    return raw_metadata


def _validate_graph_metadata(
    graph: PlaneGraph,
    metadata: CacheMetadata,
    *,
    graph_index: int,
    filepath: Path,
) -> None:
    """Validate graph payload against declared v6 provenance."""
    if metadata.graph_class == "simple_quartic" and graph.double_edges:
        raise ValueError(
            f"cache: graph {graph_index} has double edges but graph_class is 'simple_quartic' ({filepath})"
        )
    if metadata.include_primal is None:
        return
    has_primal = graph.primal_num_vertices > 0
    if has_primal is not metadata.include_primal:
        raise ValueError(
            f"cache: graph {graph_index} include_primal={has_primal} != metadata {metadata.include_primal} ({filepath})"
        )


def _infer_dual_vertex_count_for_save(
    graphs: list[PlaneGraph],
    dual_vertex_count: int | None,
) -> int:
    """Infer or validate the dual vertex count when saving a cache file."""
    if dual_vertex_count is not None:
        _require_int_at_least(
            dual_vertex_count,
            field_name="dual_vertex_count",
            minimum=3,
        )

    if graphs:
        graph_dual_counts = {graph.dual_num_vertices for graph in graphs}
        if len(graph_dual_counts) != 1:
            raise ValueError(
                "cache: graphs contain mixed dual_vertex_count values; cannot save a heterogeneous cache"
            )
        inferred_dual_vertex_count = next(iter(graph_dual_counts))
        _require_int_at_least(
            inferred_dual_vertex_count,
            field_name="dual_vertex_count",
            minimum=3,
        )
        if dual_vertex_count is None:
            return inferred_dual_vertex_count
        if dual_vertex_count != inferred_dual_vertex_count:
            raise ValueError(
                f"cache: dual_vertex_count mismatch: explicit {dual_vertex_count} != inferred {inferred_dual_vertex_count}"
            )
        return dual_vertex_count

    if dual_vertex_count is None:
        raise ValueError(
            "cache: dual_vertex_count must be provided when graphs is empty"
        )
    return dual_vertex_count


def _validate_graph_semantics(graphs: list[PlaneGraph]) -> None:
    """Reject the first graph whose domain invariants do not validate."""
    for graph_index, graph in enumerate(graphs):
        try:
            is_valid, errors = graph.validate()
        except Exception as exc:
            raise ValueError(
                f"cache: invalid graph {graph_index}: validation raised {type(exc).__name__}: {exc}"
            ) from exc
        if is_valid:
            continue

        shown_errors = errors[:3]
        summary = "; ".join(shown_errors) or "validation failed without details"
        if len(errors) > len(shown_errors):
            summary += f"; ... ({len(errors)} errors total)"
        raise ValueError(f"cache: invalid graph {graph_index}: {summary}")


def _select_pickle_graphs(
    raw_graphs: Any,
    *,
    metadata: CacheMetadata,
    filepath: Path,
    max_count: int | None,
) -> tuple[list[PlaneGraph], int, int | None]:
    """Validate cached PlaneGraph objects and select the requested prefix."""
    if not isinstance(raw_graphs, (list, tuple)):
        raise ValueError(f"cache: graphs payload must be list/tuple ({filepath})")

    raw_graph_count = len(raw_graphs)
    inferred_dual_vertex_count: int | None = None
    selected_graphs: list[PlaneGraph] = []
    # Scan the full payload for consistency even when max_count returns only a prefix.
    for graph_index, raw_graph in enumerate(raw_graphs):
        if not isinstance(raw_graph, PlaneGraph):
            raise ValueError(
                f"cache: unsupported graph payload type {type(raw_graph).__name__}"
            )
        _validate_graph_metadata(
            raw_graph,
            metadata,
            graph_index=graph_index,
            filepath=filepath,
        )
        graph_dual_vertex_count = raw_graph.dual_num_vertices
        if inferred_dual_vertex_count is None:
            inferred_dual_vertex_count = graph_dual_vertex_count
        elif graph_dual_vertex_count != inferred_dual_vertex_count:
            raise ValueError(
                f"cache: graphs payload contains mixed dual_vertex_count values ({filepath})"
            )

        if max_count is None or graph_index < max_count:
            selected_graphs.append(raw_graph)
    return selected_graphs, raw_graph_count, inferred_dual_vertex_count


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
            f"cache: metadata.graph_count mismatch: {metadata.graph_count} != {raw_graph_count} ({filepath})"
        )
    if (
        inferred_dual_vertex_count is not None
        and metadata.dual_vertex_count != inferred_dual_vertex_count
    ):
        raise ValueError(
            f"cache: metadata.dual_vertex_count mismatch: {metadata.dual_vertex_count} != {inferred_dual_vertex_count} ({filepath})"
        )


def _save_pickle(
    graphs: list[PlaneGraph],
    filepath: Path,
    dual_vertex_count: int | None,
    graph_class: CacheGraphClass,
    include_primal: bool,
    compress: bool,
    compress_level: int,
) -> None:
    """Pickle serialization with atomic write and optional gzip compression."""
    protocol = pickle.HIGHEST_PROTOCOL
    resolved_dual_vertex_count = _infer_dual_vertex_count_for_save(
        graphs,
        dual_vertex_count,
    )
    metadata = _build_cache_metadata(
        dual_vertex_count=resolved_dual_vertex_count,
        graph_count=len(graphs),
        pickle_protocol=protocol,
        graph_class=graph_class,
        include_primal=include_primal,
    )
    for graph_index, graph in enumerate(graphs):
        _validate_graph_metadata(
            graph,
            metadata,
            graph_index=graph_index,
            filepath=filepath,
        )
    payload = {
        "metadata": metadata,
        "graphs": graphs,
    }

    def _write_pickle(file_obj: Any) -> None:
        if compress:
            # Omit variable gzip metadata to keep identical cache payloads reproducible.
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=compress_level,
                fileobj=file_obj,
                mtime=0,
            ) as gz:
                pickle.dump(payload, gz, protocol=protocol)
        else:
            pickle.dump(payload, file_obj, protocol=protocol)

    _atomic_write(filepath, writer=_write_pickle)


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
                payload = SafeUnpickler(gz).load() if safe_mode else pickle.load(gz)
        else:
            payload = SafeUnpickler(f).load() if safe_mode else pickle.load(f)

    # Extract and validate metadata.
    if not isinstance(payload, dict):
        raise ValueError(
            f"cache: unsupported payload type {type(payload).__name__}; expected dict with metadata+graphs"
        )
    if "metadata" not in payload or "graphs" not in payload:
        raise ValueError(f"cache: missing metadata/graphs keys ({filepath})")

    metadata = _validate_pickle_metadata(payload["metadata"], filepath)
    graphs, raw_graph_count, inferred_dual_vertex_count = _select_pickle_graphs(
        payload["graphs"],
        metadata=metadata,
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
    graph_class: CacheGraphClass,
    include_primal: bool,
    compress: bool = True,
    compress_level: int = 6,
) -> Path:
    """Save graphs atomically with explicit generation provenance."""
    filepath = Path(filepath)
    _save_pickle(
        graphs,
        filepath,
        dual_vertex_count,
        graph_class,
        include_primal,
        compress,
        compress_level,
    )

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
    trusted: bool = False,
    safe_mode: bool = True,
    validate_graphs: bool = False,
) -> tuple[list[PlaneGraph], CacheMetadata]:
    """Load graphs from a trusted pickle cache and validate them on request."""
    if max_count is not None:
        _require_int_at_least(
            max_count,
            field_name="max_count",
            minimum=0,
        )
    if not isinstance(validate_graphs, bool):
        raise ValueError(
            f"cache: validate_graphs must be bool; got {validate_graphs!r}"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cache file not found: {filepath}")

    if not trusted:
        raise ValueError("cache: pickle loading requires trusted=True")
    graphs, metadata = _load_pickle(filepath, max_count, safe_mode)

    if validate_graphs:
        _validate_graph_semantics(graphs)
    return graphs, metadata
