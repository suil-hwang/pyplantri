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
from .plantri import QuadrangulationDualClass

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 6
CACHE_PICKLE_PROTOCOL = 5
CacheGraphClass = Literal["quartic_multigraph", "simple_quartic"]
_CacheGraphClassInput = CacheGraphClass | QuadrangulationDualClass
_SUPPORTED_GRAPH_CLASSES = frozenset({"quartic_multigraph", "simple_quartic"})


def _one_line(value: object) -> str:
    """Collapse nested diagnostic text to one line."""
    return " ".join(str(value).splitlines())


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
            f"cache: invalid {field_name}={value!r}{location}"
        )


def _require_bool(value: Any, *, field_name: str) -> None:
    """Validate an exact boolean control value."""
    if type(value) is not bool:
        raise ValueError(
            f"cache: invalid {field_name}={value!r}"
        )


def _normalize_graph_class(
    value: _CacheGraphClassInput,
    *,
    field_name: str,
) -> CacheGraphClass:
    """Normalize public graph-class inputs to canonical cache strings."""
    raw_value = value.value if isinstance(value, QuadrangulationDualClass) else value
    if type(raw_value) is str:
        if raw_value == "quartic_multigraph":
            return "quartic_multigraph"
        if raw_value == "simple_quartic":
            return "simple_quartic"
    raise ValueError(f"cache: invalid {field_name}={value!r}")


@dataclass(frozen=True)
class CacheMetadata:
    """Versioned cache identity, size, and generation provenance."""

    format_version: int
    pyplantri_version: str
    dual_vertex_count: int
    graph_count: int
    pickle_protocol: int
    graph_class: CacheGraphClass
    include_primal: bool

    def __setstate__(self, state: Any) -> None:
        """Restore only the canonical current cache metadata state."""
        if not isinstance(state, dict):
            raise TypeError(
                f"cache: invalid metadata state type {type(state).__name__}"
            )

        required_keys = (
            "format_version",
            "pyplantri_version",
            "dual_vertex_count",
            "graph_count",
            "pickle_protocol",
            "graph_class",
            "include_primal",
        )
        actual_keys = set(state)
        expected_keys = set(required_keys)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"cache: metadata keys mismatch: missing={missing}, extra={extra}"
            )  # fmt: skip

        for name in required_keys:
            object.__setattr__(self, name, state[name])


class SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler for pyplantri cache payloads."""

    # Only globals emitted by the canonical current cache schema.
    SAFE_MODULES: dict[str, set[str]] = {
        "pyplantri.plane_graph": {
            "PlaneGraph",
            "FrozenEdgeMultiplicity",
        },
        "pyplantri.cache": {
            "CacheMetadata",
        },
    }

    def find_class(self, module: str, name: str) -> Any:
        """Override to restrict loadable classes."""
        allowed = self.SAFE_MODULES.get(module, set())

        if name not in allowed:
            raise pickle.UnpicklingError(
                f"cache: forbidden pickle class {module}.{name}"
            )

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
        and metadata.format_version == CACHE_FORMAT_VERSION
    ):
        return
    location = f" ({filepath})" if filepath is not None else ""
    raise ValueError(
        f"cache: format_version {metadata.format_version!r}!={CACHE_FORMAT_VERSION}{location}"
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
            f"cache: invalid pyplantri_version={metadata.pyplantri_version!r}{location}"
        )
    for field_name, value, minimum in (
        ("metadata.dual_vertex_count", metadata.dual_vertex_count, 3),
        ("metadata.graph_count", metadata.graph_count, 0),
    ):
        _require_int_at_least(
            value,
            field_name=field_name,
            minimum=minimum,
            filepath=filepath,
        )
    if (
        type(metadata.pickle_protocol) is not int
        or metadata.pickle_protocol != CACHE_PICKLE_PROTOCOL
    ):
        raise ValueError(
            f"cache: pickle_protocol {metadata.pickle_protocol!r}!={CACHE_PICKLE_PROTOCOL}{location}"
        )

    if (
        type(metadata.graph_class) is not str
        or metadata.graph_class not in _SUPPORTED_GRAPH_CLASSES
    ):
        raise ValueError(
            f"cache: invalid graph_class={metadata.graph_class!r}{location}"
        )
    if type(metadata.include_primal) is not bool:
        raise ValueError(
            f"cache: invalid include_primal={metadata.include_primal!r}{location}"
        )


def validate_cache_metadata(
    metadata: CacheMetadata,
    *,
    expected_dual_vertex_count: int | None = None,
    expected_graph_class: _CacheGraphClassInput | None = None,
    expected_include_primal: bool | None = None,
    filepath: str | Path | None = None,
) -> None:
    """Validate metadata and optional caller-owned cache expectations."""
    if type(metadata) is not CacheMetadata:
        raise ValueError(
            f"cache: invalid metadata type {type(metadata).__name__}"
        )
    resolved_path = Path(filepath) if filepath is not None else None
    _validate_metadata_fields(metadata, filepath=resolved_path)
    location = f" ({resolved_path})" if resolved_path is not None else ""

    if expected_dual_vertex_count is not None:
        _require_int_at_least(
            expected_dual_vertex_count,
            field_name="expected_dual_vertex_count",
            minimum=3,
        )
        if metadata.dual_vertex_count != expected_dual_vertex_count:
            raise ValueError(
                f"cache: dual_vertex_count {metadata.dual_vertex_count}!={expected_dual_vertex_count}{location}"
            )
    if expected_graph_class is not None:
        resolved_graph_class = _normalize_graph_class(
            expected_graph_class,
            field_name="expected_graph_class",
        )
        if metadata.graph_class != resolved_graph_class:
            raise ValueError(
                f"cache: graph_class {metadata.graph_class!r}!={resolved_graph_class!r}{location}"
            )
    if expected_include_primal is not None:
        _require_bool(
            expected_include_primal,
            field_name="expected_include_primal",
        )
        if metadata.include_primal is not expected_include_primal:
            raise ValueError(
                f"cache: include_primal {metadata.include_primal!r}!={expected_include_primal!r}{location}"
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
    if type(raw_metadata) is not CacheMetadata:
        raise ValueError(
            f"cache: invalid metadata type {type(raw_metadata).__name__}"
        )
    validate_cache_metadata(raw_metadata, filepath=filepath)
    return raw_metadata


def _has_primal_payload(graph: PlaneGraph) -> bool:
    """Return whether any serialized primal field is populated."""
    return (
        graph.primal_num_vertices != 0
        or bool(graph.primal_embedding)
        or bool(graph.primal_faces)
        or bool(graph.dual_vertex_to_primal_face)
        or bool(graph.primal_vertex_to_dual_face)
        or bool(graph.primal_edge_label_pairs)
    )


def _validate_graph_metadata(
    graph: PlaneGraph,
    metadata: CacheMetadata,
    *,
    graph_index: int,
    filepath: Path,
) -> None:
    """Validate one graph against cheap current-format payload invariants."""
    n = metadata.dual_vertex_count
    if graph.dual_num_vertices != n:
        raise ValueError(
            f"cache: graph {graph_index} n={graph.dual_num_vertices}!={n} ({filepath})"
        )
    if (
        len(graph.dual_embedding) != n
        or any(len(neighbors) != 4 for neighbors in graph.dual_embedding)
    ):
        raise ValueError(
            f"cache: graph {graph_index} is not {n}-vertex 4-regular ({filepath})"
        )
    if len(graph.dual_faces) != n + 2:
        raise ValueError(
            f"cache: graph {graph_index} faces={len(graph.dual_faces)}!={n + 2} ({filepath})"
        )

    edge_items = tuple(graph.dual_edge_multiplicity.items())
    if graph.dual_support_edges != tuple(edge for edge, _ in edge_items):
        raise ValueError(
            f"cache: graph {graph_index} support-edge mismatch ({filepath})"
        )
    for edge, multiplicity in edge_items:
        u, v = edge
        if not (0 <= u < v < n) or multiplicity not in (1, 2):
            raise ValueError(
                f"cache: graph {graph_index} invalid edge entry {edge!r}:{multiplicity!r} ({filepath})"
            )
    dual_edge_count = sum(multiplicity for _, multiplicity in edge_items)
    if dual_edge_count != 2 * n:
        raise ValueError(
            f"cache: graph {graph_index} |E*|={dual_edge_count}!={2 * n} ({filepath})"
        )

    if metadata.graph_class == "simple_quartic" and any(
        multiplicity != 1 for _, multiplicity in edge_items
    ):
        raise ValueError(
            f"cache: graph {graph_index} is not simple_quartic ({filepath})"
        )

    has_primal = _has_primal_payload(graph)
    if has_primal is not metadata.include_primal:
        raise ValueError(
            f"cache: graph {graph_index} include_primal={has_primal}!={metadata.include_primal} ({filepath})"
        )
    if not has_primal:
        return
    if (
        graph.primal_num_vertices != n + 2
        or len(graph.primal_embedding) != n + 2
        or len(graph.primal_faces) != n
        or any(len(face) != 4 for face in graph.primal_faces)
        or len(graph.dual_vertex_to_primal_face) != n
        or len(graph.primal_vertex_to_dual_face) != n + 2
    ):
        raise ValueError(
            f"cache: graph {graph_index} primal cardinality mismatch ({filepath})"
        )


def _infer_dual_vertex_count_for_save(
    graphs: list[PlaneGraph],
    dual_vertex_count: int | None,
) -> int:
    """Infer or validate the dual vertex count when saving a cache file."""
    if type(graphs) is not list:
        raise ValueError(
            f"cache: invalid graphs type {type(graphs).__name__}"
        )
    if dual_vertex_count is not None:
        _require_int_at_least(
            dual_vertex_count,
            field_name="dual_vertex_count",
            minimum=3,
        )

    if graphs:
        for graph_index, graph in enumerate(graphs):
            if type(graph) is not PlaneGraph:
                raise ValueError(
                    f"cache: invalid graph {graph_index} type {type(graph).__name__}"
                )
        graph_dual_counts = {graph.dual_num_vertices for graph in graphs}
        if len(graph_dual_counts) != 1:
            raise ValueError(
                f"cache: mixed dual_vertex_count={sorted(graph_dual_counts)}"
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
                f"cache: dual_vertex_count={dual_vertex_count}; inferred {inferred_dual_vertex_count}"
            )
        return dual_vertex_count

    if dual_vertex_count is None:
        raise ValueError("cache: dual_vertex_count required for empty graphs")
    return dual_vertex_count


def _validate_graph_semantics(graphs: list[PlaneGraph]) -> None:
    """Reject the first graph whose domain invariants do not validate."""
    for graph_index, graph in enumerate(graphs):
        try:
            is_valid, errors = graph.validate()
        except Exception as exc:
            raise ValueError(
                f"cache: invalid graph {graph_index}: {type(exc).__name__}: {_one_line(exc)}"
            ) from exc
        if is_valid:
            continue

        summary = _one_line(errors[0]) if errors else "validation failed"
        if len(errors) > 1:
            summary += f" (+{len(errors) - 1})"
        raise ValueError(
            f"cache: invalid graph {graph_index}: {summary}"
        )


def _validate_graph_payload(
    raw_graphs: Any,
    *,
    metadata: CacheMetadata,
    filepath: Path,
) -> list[PlaneGraph]:
    """Validate the complete canonical graph payload."""
    if type(raw_graphs) is not list:
        raise ValueError(
            f"cache: invalid graphs payload type {type(raw_graphs).__name__} ({filepath})"
        )
    if metadata.graph_count != len(raw_graphs):
        raise ValueError(
            f"cache: graph_count={metadata.graph_count}; actual {len(raw_graphs)} ({filepath})"
        )
    if (
        metadata.graph_class == "simple_quartic"
        and raw_graphs
        and metadata.dual_vertex_count < 6
    ):
        raise ValueError(
            f"cache: nonempty simple_quartic n={metadata.dual_vertex_count}<6 ({filepath})"
        )

    graph_id_to_index: dict[int, int] = {}
    for graph_index, raw_graph in enumerate(raw_graphs):
        if type(raw_graph) is not PlaneGraph:
            raise ValueError(
                f"cache: invalid graph {graph_index} type {type(raw_graph).__name__}"
            )
        graph_id = raw_graph.graph_id
        if type(graph_id) is not int or graph_id < 0:
            raise ValueError(
                f"cache: graph {graph_index} invalid graph_id={graph_id!r} ({filepath})"
            )
        prior_index = graph_id_to_index.setdefault(graph_id, graph_index)
        if prior_index != graph_index:
            raise ValueError(
                f"cache: duplicate graph_id={graph_id} at {prior_index},{graph_index} ({filepath})"
            )
        _validate_graph_metadata(
            raw_graph,
            metadata,
            graph_index=graph_index,
            filepath=filepath,
        )
    return raw_graphs


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
    protocol = CACHE_PICKLE_PROTOCOL
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
    _validate_graph_payload(graphs, metadata=metadata, filepath=filepath)
    _validate_graph_semantics(graphs)
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
) -> tuple[list[PlaneGraph], CacheMetadata]:
    """Restricted deserialization of one exact current-format payload."""
    with open(filepath, "rb") as f:
        # Auto-detect gzip by magic number (0x1f 0x8b).
        header = f.read(2)
        f.seek(0)

        if header == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                payload = SafeUnpickler(gz).load()
                try:
                    trailing = gz.read(1)
                except OSError as exc:
                    raise ValueError(
                        f"cache: invalid gzip trailer ({filepath})"
                    ) from exc
        else:
            payload = SafeUnpickler(f).load()
            trailing = f.read(1)
    if trailing:
        raise ValueError(
            f"cache: trailing pickle data ({filepath})"
        )

    # Extract and validate metadata.
    if type(payload) is not dict:
        raise ValueError(
            f"cache: invalid payload type {type(payload).__name__}"
        )
    expected_keys = {"metadata", "graphs"}
    if set(payload) != expected_keys:
        raise ValueError(
            f"cache: invalid payload keys={sorted(map(repr, payload))} ({filepath})"
        )

    metadata = _validate_pickle_metadata(payload["metadata"], filepath)
    graphs = _validate_graph_payload(
        payload["graphs"],
        metadata=metadata,
        filepath=filepath,
    )
    return graphs, metadata


def save_graphs_to_cache(
    graphs: list[PlaneGraph],
    filepath: str | Path,
    *,
    dual_vertex_count: int | None = None,
    graph_class: _CacheGraphClassInput,
    include_primal: bool,
    compress: bool = True,
    compress_level: int = 6,
) -> Path:
    """Save graphs atomically with explicit generation provenance."""
    _require_bool(include_primal, field_name="include_primal")
    _require_bool(compress, field_name="compress")
    if type(compress_level) is not int or not 0 <= compress_level <= 9:
        raise ValueError(
            f"cache: invalid compress_level={compress_level!r}"
        )
    resolved_graph_class = _normalize_graph_class(
        graph_class,
        field_name="graph_class",
    )
    filepath = Path(filepath)
    _save_pickle(
        graphs,
        filepath,
        dual_vertex_count,
        resolved_graph_class,
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
    validate_graphs: bool = False,
) -> tuple[list[PlaneGraph], CacheMetadata]:
    """Load graphs from a trusted pickle cache and validate them on request."""
    if max_count is not None:
        _require_int_at_least(
            max_count,
            field_name="max_count",
            minimum=0,
        )
    _require_bool(trusted, field_name="trusted")
    _require_bool(validate_graphs, field_name="validate_graphs")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"cache: not found {filepath}"
        )

    if not trusted:
        raise ValueError("cache: trusted=True required")
    graphs, metadata = _load_pickle(filepath)

    if validate_graphs:
        _validate_graph_semantics(graphs)
    if max_count is not None:
        return graphs[:max_count], metadata
    return graphs, metadata
