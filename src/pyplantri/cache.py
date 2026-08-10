# src/pyplantri/cache.py
from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import pickle
import struct
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

from .plane_graph import QuarticPlaneMap
from .plantri_interface import QuadrangulationDualClass

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 11
CACHE_DEFAULT_CHUNK_SIZE = 512
_CACHE_PICKLE_PROTOCOL = 5
_CACHE_FOOTER_STRUCT = struct.Struct(">Q32s")
_MAX_MANIFEST_SIZE = 64 * 1024 * 1024
_UINT32_LIMIT = 1 << 32

CacheGraphClass = Literal["quartic_multigraph", "simple_quartic"]
_CacheGraphClassInput = CacheGraphClass | QuadrangulationDualClass
_Compression = Literal["gzip", "none"]
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
        raise ValueError(f"cache: invalid {field_name}={value!r}{location}")


def _require_bool(value: Any, *, field_name: str) -> None:
    """Validate an exact boolean control value."""
    if type(value) is not bool:
        raise ValueError(f"cache: invalid {field_name}={value!r}")


def _normalize_graph_class(
    value: _CacheGraphClassInput,
    *,
    field_name: str,
) -> CacheGraphClass:
    """Normalize public graph-class inputs to canonical cache strings."""
    raw_value = (
        value.value
        if isinstance(value, QuadrangulationDualClass)
        else value
    )
    if type(raw_value) is str:
        if raw_value == "quartic_multigraph":
            return "quartic_multigraph"
        if raw_value == "simple_quartic":
            return "simple_quartic"
    raise ValueError(f"cache: invalid {field_name}={value!r}")


def _require_order_name(
    value: Any,
    *,
    field_name: str,
    filepath: Path | None = None,
) -> None:
    """Validate one canonical storage-order identity string."""
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        location = f" ({filepath})" if filepath is not None else ""
        raise ValueError(f"cache: invalid {field_name}={value!r}{location}")


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    """Versioned cache identity, size, and generation provenance."""

    format_version: int
    dual_vertex_count: int
    graph_count: int
    graph_class: CacheGraphClass
    storage_order_name: str = "source"
    storage_order_version: int = 1


@dataclass(frozen=True, slots=True)
class _CacheChunk:
    """One independently serialized graph chunk."""

    first_graph_index: int
    graph_count: int
    offset: int
    size: int
    digest: bytes


@dataclass(frozen=True, slots=True)
class _GraphIdIndex:
    """Dense source-Graph-ID to physical stored-index mapping."""

    offset: int
    size: int
    width: Literal[4, 8]
    digest: bytes


@dataclass(frozen=True, slots=True)
class _CacheManifest:
    """Validated internal index for the footer-manifest container."""

    metadata: CacheMetadata
    chunk_size: int
    compression: _Compression
    graph_id_index: _GraphIdIndex
    chunks: tuple[_CacheChunk, ...]


class _SafeUnpickler(pickle.Unpickler):
    """Restricted unpickler for graph-chunk payloads."""

    SAFE_MODULES: dict[str, set[str]] = {
        "pyplantri.plane_graph": {"QuarticPlaneMap"},
    }

    def find_class(self, module: str, name: str) -> Any:
        """Restrict cache chunks to current canonical graph classes."""
        if name not in self.SAFE_MODULES.get(module, set()):
            raise pickle.UnpicklingError(f"cache: forbidden pickle class {module}.{name}")
        return super().find_class(module, name)


def _validate_metadata_fields(
    metadata: CacheMetadata,
    *,
    filepath: Path | None = None,
) -> None:
    """Validate all current cache metadata fields without coercion."""
    location = f" ({filepath})" if filepath is not None else ""
    if (
        type(metadata.format_version) is not int
        or metadata.format_version != CACHE_FORMAT_VERSION
    ):
        raise ValueError(f"cache: format_version {metadata.format_version!r}!={CACHE_FORMAT_VERSION}{location}")
    for field_name, value, minimum in (
        ("metadata.dual_vertex_count", metadata.dual_vertex_count, 3),
        ("metadata.graph_count", metadata.graph_count, 0),
        (
            "metadata.storage_order_version",
            metadata.storage_order_version,
            1,
        ),
    ):
        _require_int_at_least(
            value,
            field_name=field_name,
            minimum=minimum,
            filepath=filepath,
        )
    if (
        type(metadata.graph_class) is not str
        or metadata.graph_class not in _SUPPORTED_GRAPH_CLASSES
    ):
        raise ValueError(f"cache: invalid graph_class={metadata.graph_class!r}{location}")
    _require_order_name(
        metadata.storage_order_name,
        field_name="metadata.storage_order_name",
        filepath=filepath,
    )


def validate_cache_metadata(
    metadata: CacheMetadata,
    *,
    expected_dual_vertex_count: int | None = None,
    expected_graph_class: _CacheGraphClassInput | None = None,
    expected_storage_order_name: str | None = None,
    expected_storage_order_version: int | None = None,
    filepath: str | Path | None = None,
) -> None:
    """Validate metadata and optional caller-owned cache expectations."""
    if type(metadata) is not CacheMetadata:
        raise ValueError(f"cache: invalid metadata type {type(metadata).__name__}")
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
            raise ValueError(f"cache: dual_vertex_count {metadata.dual_vertex_count}!={expected_dual_vertex_count}{location}")
    if expected_graph_class is not None:
        resolved_graph_class = _normalize_graph_class(
            expected_graph_class,
            field_name="expected_graph_class",
        )
        if metadata.graph_class != resolved_graph_class:
            raise ValueError(f"cache: graph_class {metadata.graph_class!r}!={resolved_graph_class!r}{location}")
    if expected_storage_order_name is not None:
        _require_order_name(
            expected_storage_order_name,
            field_name="expected_storage_order_name",
        )
        if metadata.storage_order_name != expected_storage_order_name:
            raise ValueError(f"cache: storage_order_name {metadata.storage_order_name!r}!={expected_storage_order_name!r}{location}")
    if expected_storage_order_version is not None:
        _require_int_at_least(
            expected_storage_order_version,
            field_name="expected_storage_order_version",
            minimum=1,
        )
        if (
            metadata.storage_order_version
            != expected_storage_order_version
        ):
            raise ValueError(f"cache: storage_order_version {metadata.storage_order_version}!={expected_storage_order_version}{location}")


def _parse_sha256(
    value: Any,
    *,
    field_name: str,
    filepath: Path,
) -> bytes:
    """Parse one canonical lowercase SHA-256 hex digest."""
    message = f"cache: invalid {field_name}={value!r} ({filepath})"
    if type(value) is not str or len(value) != 64:
        raise ValueError(message)
    try:
        digest = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(message) from exc
    if digest.hex() != value:
        raise ValueError(message)
    return digest


def _manifest_from_bytes(
    manifest_bytes: bytes,
    filepath: Path,
    *,
    manifest_offset: int,
) -> _CacheManifest:
    """Parse and validate the manifest and exact physical partition."""
    try:
        manifest_data = json.loads(manifest_bytes)
    except ValueError as exc:
        raise ValueError(f"cache: invalid manifest JSON ({filepath})") from exc
    if type(manifest_data) is not dict:
        raise ValueError(f"cache: invalid manifest type {type(manifest_data).__name__} ({filepath})")
    manifest_keys = {
        "metadata",
        "chunk_size",
        "compression",
        "graph_id_index_sha256",
        "chunks",
    }
    if set(manifest_data) != manifest_keys:
        raise ValueError(f"cache: invalid manifest keys ({filepath})")

    metadata_data = manifest_data["metadata"]
    if type(metadata_data) is not dict:
        raise ValueError(f"cache: invalid metadata type {type(metadata_data).__name__} ({filepath})")
    metadata_keys = {
        "format_version",
        "dual_vertex_count",
        "graph_count",
        "graph_class",
        "storage_order_name",
        "storage_order_version",
    }
    actual_metadata_keys = set(metadata_data)
    if actual_metadata_keys != metadata_keys:
        missing = sorted(metadata_keys - actual_metadata_keys)
        extra = sorted(actual_metadata_keys - metadata_keys)
        raise ValueError(f"cache: metadata keys mismatch: missing={missing}, extra={extra}")
    metadata = CacheMetadata(**metadata_data)
    _validate_metadata_fields(metadata, filepath=filepath)
    chunk_size = manifest_data["chunk_size"]
    _require_int_at_least(
        chunk_size,
        field_name="manifest.chunk_size",
        minimum=1,
        filepath=filepath,
    )
    compression = manifest_data["compression"]
    if type(compression) is not str or compression not in ("gzip", "none"):
        raise ValueError(f"cache: invalid compression={compression!r} ({filepath})")

    chunk_descriptors = manifest_data["chunks"]
    if type(chunk_descriptors) is not list:
        raise ValueError(f"cache: invalid chunks type {type(chunk_descriptors).__name__} ({filepath})")
    expected_chunk_count = (metadata.graph_count + chunk_size - 1) // chunk_size
    if len(chunk_descriptors) != expected_chunk_count:
        raise ValueError(f"cache: chunk count {len(chunk_descriptors)}!={expected_chunk_count} ({filepath})")

    chunks: list[_CacheChunk] = []
    next_chunk_offset = 0
    for chunk_index, chunk_descriptor in enumerate(chunk_descriptors):
        if type(chunk_descriptor) is not list or len(chunk_descriptor) != 2:
            raise ValueError(f"cache: invalid chunk {chunk_index} descriptor ({filepath})")
        chunk_byte_count, chunk_digest_hex = chunk_descriptor
        _require_int_at_least(
            chunk_byte_count,
            field_name=f"chunk[{chunk_index}].size",
            minimum=1,
            filepath=filepath,
        )
        first_graph_index = chunk_index * chunk_size
        chunk_graph_count = min(chunk_size, metadata.graph_count - first_graph_index)
        chunks.append(
            _CacheChunk(
                first_graph_index=first_graph_index,
                graph_count=chunk_graph_count,
                offset=next_chunk_offset,
                size=chunk_byte_count,
                digest=_parse_sha256(
                    chunk_digest_hex,
                    field_name=f"chunk[{chunk_index}].sha256",
                    filepath=filepath,
                ),
            )
        )
        next_chunk_offset += chunk_byte_count

    index_width: Literal[4, 8] = 4 if metadata.graph_count <= _UINT32_LIMIT else 8
    index_size = metadata.graph_count * index_width
    graph_id_index = _GraphIdIndex(
        offset=next_chunk_offset,
        size=index_size,
        width=index_width,
        digest=_parse_sha256(
            manifest_data["graph_id_index_sha256"],
            field_name="graph_id_index_sha256",
            filepath=filepath,
        ),
    )
    if graph_id_index.offset + graph_id_index.size != manifest_offset:
        raise ValueError(f"cache: graph_id_index does not end at manifest ({filepath})")
    return _CacheManifest(
        metadata=metadata,
        chunk_size=chunk_size,
        compression=compression,
        graph_id_index=graph_id_index,
        chunks=tuple(chunks),
    )


def _validate_graph_envelope(
    graph: QuarticPlaneMap,
    metadata: CacheMetadata,
    *,
    graph_index: int,
    filepath: Path,
) -> int:
    """Validate cache identity and cheap graph cardinalities."""
    if type(graph) is not QuarticPlaneMap:
        raise ValueError(f"cache: invalid graph {graph_index} type {type(graph).__name__}")
    graph_id = graph.graph_id
    if (
        type(graph_id) is not int
        or not 0 <= graph_id < metadata.graph_count
    ):
        raise ValueError(f"cache: graph {graph_index} noncanonical graph_id={graph_id!r}; expected 0..{metadata.graph_count - 1} ({filepath})")
    if (
        metadata.storage_order_name == "source"
        and graph_id != graph_index
    ):
        raise ValueError(f"cache: source-order graph {graph_index} has graph_id={graph_id} ({filepath})")

    n = metadata.dual_vertex_count
    if len(graph.twin) != 4 * n:
        raise ValueError(f"cache: graph {graph_index} dart count {len(graph.twin)}!={4 * n} ({filepath})")

    num_support_edges, double_edge_count = graph._dual_edge_cardinality_profile()
    if not n <= num_support_edges <= 2 * n:
        raise ValueError(f"cache: graph {graph_index} support-edge cardinality mismatch ({filepath})")
    if (
        metadata.graph_class == "simple_quartic"
        and (double_edge_count or num_support_edges != 2 * n)
    ):
        raise ValueError(f"cache: graph {graph_index} is not simple_quartic ({filepath})")
    return graph_id


def _resolve_dual_vertex_count(
    graphs: list[QuarticPlaneMap],
    dual_vertex_count: int | None,
) -> int:
    """Infer or validate the dual vertex count when saving."""
    if type(graphs) is not list:
        raise ValueError(f"cache: invalid graphs type {type(graphs).__name__}")
    if dual_vertex_count is None:
        if not graphs:
            raise ValueError("cache: dual_vertex_count required for empty graphs")
        first_graph = graphs[0]
        if type(first_graph) is not QuarticPlaneMap:
            raise ValueError(f"cache: invalid graph 0 type {type(first_graph).__name__}")
        dual_vertex_count = first_graph.dual_num_vertices
    _require_int_at_least(
        dual_vertex_count,
        field_name="dual_vertex_count",
        minimum=3,
    )
    return dual_vertex_count


def _validate_graph_semantics_one(
    graph: QuarticPlaneMap,
    graph_index: int,
) -> None:
    """Reject one graph whose complete domain invariants fail."""
    is_valid, errors = graph.validate()
    if is_valid:
        return
    summary = " ".join(str(errors[0]).splitlines()) if errors else "validation failed"
    if len(errors) > 1:
        summary += f" (+{len(errors) - 1})"
    raise ValueError(f"cache: invalid graph {graph_index}: {summary}")


def _prepare_graph_payload(
    graphs: list[QuarticPlaneMap],
    *,
    metadata: CacheMetadata,
    filepath: Path,
    validate_graphs: bool,
) -> bytearray:
    """Validate the save envelope and build graph-ID index data."""
    if (
        metadata.graph_class == "simple_quartic"
        and graphs
        and metadata.dual_vertex_count < 6
    ):
        raise ValueError(f"cache: nonempty simple_quartic n={metadata.dual_vertex_count}<6 ({filepath})")

    width: Literal[4, 8] = (
        4 if metadata.graph_count <= _UINT32_LIMIT else 8
    )
    pack_index = struct.Struct(
        "<I" if width == 4 else "<Q"
    ).pack_into
    seen_graph_ids = bytearray(metadata.graph_count)
    index_payload = bytearray(metadata.graph_count * width)
    for stored_index, graph in enumerate(graphs):
        graph_id = _validate_graph_envelope(
            graph,
            metadata,
            graph_index=stored_index,
            filepath=filepath,
        )
        if seen_graph_ids[graph_id]:
            raise ValueError(f"cache: duplicate graph_id={graph_id} ({filepath})")
        seen_graph_ids[graph_id] = 1
        if validate_graphs:
            _validate_graph_semantics_one(graph, stored_index)
        pack_index(
            index_payload,
            graph_id * width,
            stored_index,
        )
    return index_payload


def _open_manifest(
    filepath: Path,
) -> _CacheManifest:
    """Open and validate only the footer, manifest, and physical ranges."""
    with filepath.open("rb") as stream:
        file_size = stream.seek(0, io.SEEK_END)
        minimum_size = 1 + _CACHE_FOOTER_STRUCT.size
        if file_size < minimum_size:
            raise ValueError(f"cache: truncated v{CACHE_FORMAT_VERSION} container ({filepath})")

        footer_offset = file_size - _CACHE_FOOTER_STRUCT.size
        stream.seek(footer_offset)
        raw_footer = stream.read(_CACHE_FOOTER_STRUCT.size)
        if len(raw_footer) != _CACHE_FOOTER_STRUCT.size:
            raise ValueError(f"cache: truncated footer ({filepath})")
        (
            manifest_size,
            expected_manifest_digest,
        ) = _CACHE_FOOTER_STRUCT.unpack(raw_footer)
        if (
            not 1 <= manifest_size <= _MAX_MANIFEST_SIZE
            or manifest_size > footer_offset
        ):
            raise ValueError(f"cache: invalid manifest range ({filepath})")
        manifest_offset = footer_offset - manifest_size

        stream.seek(manifest_offset)
        manifest_bytes = stream.read(manifest_size)
        if len(manifest_bytes) != manifest_size:
            raise ValueError(f"cache: truncated manifest ({filepath})")
        if (
            hashlib.sha256(manifest_bytes).digest()
            != expected_manifest_digest
        ):
            raise ValueError(f"cache: manifest hash mismatch ({filepath})")

    return _manifest_from_bytes(
        manifest_bytes,
        filepath,
        manifest_offset=manifest_offset,
    )


class QuarticPlaneMapCatalog(Sequence[QuarticPlaneMap]):
    """Path-backed lazy sequence of QuarticPlaneMap records in physical storage order."""

    __slots__ = (
        "_filepath",
        "_manifest",
        "_cached_chunks",
        "_cached_chunk_limit",
    )

    def __init__(
        self,
        filepath: Path,
        manifest: _CacheManifest,
        *,
        cached_chunks: int = 1,
    ) -> None:
        """Initialize the lazy catalog without reading graph payloads."""
        _require_int_at_least(
            cached_chunks,
            field_name="cached_chunks",
            minimum=1,
        )
        self._filepath = filepath
        self._manifest = manifest
        self._cached_chunks: OrderedDict[int, tuple[QuarticPlaneMap, ...]] = OrderedDict()
        self._cached_chunk_limit = cached_chunks

    @property
    def metadata(self) -> CacheMetadata:
        """Return the validated public cache metadata."""
        return self._manifest.metadata

    def __len__(self) -> int:
        """Return the manifest-declared graph count."""
        return self.metadata.graph_count

    def _read_range(self, offset: int, size: int) -> bytes:
        """Read one exact byte range from the cache file."""
        with self._filepath.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(size)
        if len(payload) != size:
            raise ValueError(f"cache: truncated range at {offset}+{size} ({self._filepath})")
        return payload

    def _load_chunk(
        self,
        chunk_index: int,
    ) -> tuple[QuarticPlaneMap, ...]:
        """Return one decoded graph chunk, loading and caching it on demand."""
        graphs = self._cached_chunks.get(chunk_index)
        if graphs is not None:
            self._cached_chunks.move_to_end(chunk_index)
            return graphs

        chunk = self._manifest.chunks[chunk_index]
        payload = self._read_range(chunk.offset, chunk.size)
        graphs = self._decode_chunk(chunk_index, payload)
        self._cache_chunk(chunk_index, graphs)
        return graphs

    def _cache_chunk(
        self,
        chunk_index: int,
        graphs: tuple[QuarticPlaneMap, ...],
    ) -> None:
        """Insert one decoded chunk and evict the least-recently used chunk."""
        self._cached_chunks[chunk_index] = graphs
        self._cached_chunks.move_to_end(chunk_index)
        if len(self._cached_chunks) > self._cached_chunk_limit:
            self._cached_chunks.popitem(last=False)

    def _decode_chunk(
        self,
        chunk_index: int,
        payload: bytes,
    ) -> tuple[QuarticPlaneMap, ...]:
        """Hash-check, restricted-unpickle, and envelope-validate one chunk."""
        chunk = self._manifest.chunks[chunk_index]
        if hashlib.sha256(payload).digest() != chunk.digest:
            raise ValueError(f"cache: chunk {chunk_index} hash mismatch ({self._filepath})")
        buffer = io.BytesIO(payload)
        try:
            if self._manifest.compression == "gzip":
                with gzip.GzipFile(fileobj=buffer, mode="rb") as stream:
                    raw_graphs = _SafeUnpickler(stream).load()
                    trailing = stream.read(1)
            else:
                raw_graphs = _SafeUnpickler(buffer).load()
                trailing = buffer.read(1)
        except (OSError, EOFError, pickle.UnpicklingError, TypeError, ValueError, AttributeError, IndexError) as exc:
            raise ValueError(f"cache: invalid chunk {chunk_index} payload ({self._filepath})") from exc
        if trailing:
            raise ValueError(f"cache: trailing pickle data in chunk {chunk_index} ({self._filepath})")
        if type(raw_graphs) is not list:
            raise ValueError(f"cache: invalid graph chunk type {type(raw_graphs).__name__} ({self._filepath})")
        if len(raw_graphs) != chunk.graph_count:
            raise ValueError(f"cache: chunk graph count {len(raw_graphs)}!={chunk.graph_count} ({self._filepath})")

        seen_graph_ids: set[int] = set()
        for local_index, graph in enumerate(raw_graphs):
            graph_index = chunk.first_graph_index + local_index
            graph_id = _validate_graph_envelope(
                graph,
                self.metadata,
                graph_index=graph_index,
                filepath=self._filepath,
            )
            if graph_id in seen_graph_ids:
                raise ValueError(f"cache: duplicate graph_id={graph_id} in chunk ({self._filepath})")
            seen_graph_ids.add(graph_id)
        return tuple(raw_graphs)

    def _get_stored(self, stored_index: int) -> QuarticPlaneMap:
        """Return the graph at an already bounds-checked physical index."""
        chunk_index = stored_index // self._manifest.chunk_size
        chunk = self._manifest.chunks[chunk_index]
        return self._load_chunk(chunk_index)[
            stored_index - chunk.first_graph_index
        ]

    @overload
    def __getitem__(self, index: int) -> QuarticPlaneMap: ...

    @overload
    def __getitem__(self, index: slice) -> list[QuarticPlaneMap]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> QuarticPlaneMap | list[QuarticPlaneMap]:
        """Return one physical-order graph or a materialized slice."""
        if isinstance(index, slice):
            return [
                self[position]
                for position in range(*index.indices(len(self)))
            ]
        if type(index) is not int:
            raise TypeError(f"catalog index must be int or slice, got {type(index).__name__}")
        position = index
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("QuarticPlaneMapCatalog index out of range")
        return self._get_stored(position)

    def __iter__(self) -> Iterator[QuarticPlaneMap]:
        """Yield graphs in physical storage order through the bounded chunk cache."""
        with self._filepath.open("rb") as stream:
            for chunk_index, chunk in enumerate(self._manifest.chunks):
                graphs = self._cached_chunks.get(chunk_index)
                if graphs is None:
                    stream.seek(chunk.offset)
                    payload = stream.read(chunk.size)
                    if len(payload) != chunk.size:
                        raise ValueError(f"cache: truncated chunk {chunk_index} ({self._filepath})")
                    graphs = self._decode_chunk(chunk_index, payload)
                    self._cache_chunk(chunk_index, graphs)
                else:
                    self._cached_chunks.move_to_end(chunk_index)
                yield from graphs

    def get_by_graph_id(self, graph_id: int) -> QuarticPlaneMap:
        """Return the graph with one dense source-stream Graph ID."""
        if type(graph_id) is not int:
            raise TypeError(f"graph_id must be int, got {type(graph_id).__name__}")
        if not 0 <= graph_id < len(self):
            raise KeyError(graph_id)

        if self.metadata.storage_order_name == "source":
            stored_index = graph_id
        else:
            descriptor = self._manifest.graph_id_index
            stored_index = int.from_bytes(
                self._read_range(
                    descriptor.offset + graph_id * descriptor.width,
                    descriptor.width,
                ),
                byteorder="little",
            )
            if stored_index >= len(self):
                raise ValueError(f"cache: invalid stored index {stored_index} for graph_id={graph_id} ({self._filepath})")
        graph = self._get_stored(stored_index)
        if graph.graph_id != graph_id:
            raise ValueError(f"cache: graph_id_index maps {graph_id} to graph {graph.graph_id} ({self._filepath})")
        return graph

    def audit_all_graphs(self) -> None:
        """Audit every stored chunk, dense ID/index mapping, and graph invariant."""
        descriptor = self._manifest.graph_id_index
        index_payload = self._read_range(descriptor.offset, descriptor.size)
        if hashlib.sha256(index_payload).digest() != descriptor.digest:
            raise ValueError(f"cache: graph_id_index hash mismatch ({self._filepath})")
        index_struct = struct.Struct(
            "<I" if descriptor.width == 4 else "<Q"
        )
        seen_graph_ids = bytearray(len(self))
        for stored_index, graph in enumerate(self):
            _validate_graph_semantics_one(graph, stored_index)
            graph_id = graph.graph_id
            if seen_graph_ids[graph_id]:
                raise ValueError(f"cache: duplicate graph_id={graph_id} ({self._filepath})")
            seen_graph_ids[graph_id] = 1
            indexed_position = index_struct.unpack_from(
                index_payload,
                graph_id * descriptor.width,
            )[0]
            if indexed_position != stored_index:
                raise ValueError(f"cache: graph_id_index mismatch for graph_id={graph_id} ({self._filepath})")


def load_graphs_from_cache(
    filepath: str | Path,
    *,
    trusted: bool = False,
    cached_chunks: int = 1,
) -> QuarticPlaneMapCatalog:
    """Open a lazy catalog with a bounded decoded-chunk LRU cache."""
    _require_bool(trusted, field_name="trusted")
    _require_int_at_least(
        cached_chunks,
        field_name="cached_chunks",
        minimum=1,
    )
    resolved_path = Path(filepath)
    if not trusted:
        raise ValueError("cache: trusted=True required")

    return QuarticPlaneMapCatalog(
        resolved_path,
        _open_manifest(resolved_path),
        cached_chunks=cached_chunks,
    )


def save_graphs_to_cache(
    graphs: list[QuarticPlaneMap],
    filepath: str | Path,
    *,
    dual_vertex_count: int | None = None,
    graph_class: _CacheGraphClassInput,
    compress: bool = True,
    compress_level: int = 1,
    chunk_size: int = CACHE_DEFAULT_CHUNK_SIZE,
    storage_order_name: str = "source",
    storage_order_version: int = 1,
    validate_graphs: bool = True,
) -> Path:
    """Save a dense-ID footer-manifest cache atomically."""
    _require_bool(compress, field_name="compress")
    if type(compress_level) is not int or not 0 <= compress_level <= 9:
        raise ValueError(f"cache: invalid compress_level={compress_level!r}")
    _require_int_at_least(
        chunk_size,
        field_name="chunk_size",
        minimum=1,
    )
    _require_order_name(
        storage_order_name,
        field_name="storage_order_name",
    )
    _require_int_at_least(
        storage_order_version,
        field_name="storage_order_version",
        minimum=1,
    )
    _require_bool(validate_graphs, field_name="validate_graphs")
    resolved_graph_class = _normalize_graph_class(
        graph_class,
        field_name="graph_class",
    )
    resolved_path = Path(filepath)
    compression: _Compression = "gzip" if compress else "none"
    resolved_dual_vertex_count = _resolve_dual_vertex_count(
        graphs,
        dual_vertex_count,
    )
    metadata = CacheMetadata(
        format_version=CACHE_FORMAT_VERSION,
        dual_vertex_count=resolved_dual_vertex_count,
        graph_count=len(graphs),
        graph_class=resolved_graph_class,
        storage_order_name=storage_order_name,
        storage_order_version=storage_order_version,
    )
    index_payload = _prepare_graph_payload(
        graphs,
        metadata=metadata,
        filepath=resolved_path,
        validate_graphs=validate_graphs,
    )

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=resolved_path.parent, suffix=".tmp")
    temporary_path = Path(temporary_name)
    try:
        with open(fd, "wb") as stream:
            chunks: list[tuple[int, str]] = []
            for first_graph_index in range(0, len(graphs), chunk_size):
                graph_chunk = graphs[
                    first_graph_index : first_graph_index + chunk_size
                ]
                buffer = io.BytesIO()
                if compression == "gzip":
                    with gzip.GzipFile(
                        filename="",
                        mode="wb",
                        compresslevel=compress_level,
                        fileobj=buffer,
                        mtime=0,
                    ) as chunk_stream:
                        pickle.dump(
                            graph_chunk,
                            chunk_stream,
                            protocol=_CACHE_PICKLE_PROTOCOL,
                        )
                else:
                    pickle.dump(
                        graph_chunk,
                        buffer,
                        protocol=_CACHE_PICKLE_PROTOCOL,
                    )
                payload = buffer.getvalue()
                stream.write(payload)
                chunks.append(
                    (len(payload), hashlib.sha256(payload).hexdigest())
                )

            stream.write(index_payload)
            manifest_bytes = json.dumps(
                {
                    "metadata": {
                        "format_version": metadata.format_version,
                        "dual_vertex_count": metadata.dual_vertex_count,
                        "graph_count": metadata.graph_count,
                        "graph_class": metadata.graph_class,
                        "storage_order_name": metadata.storage_order_name,
                        "storage_order_version": metadata.storage_order_version,
                    },
                    "chunk_size": chunk_size,
                    "compression": compression,
                    "graph_id_index_sha256": hashlib.sha256(index_payload).hexdigest(),
                    "chunks": chunks,
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            manifest_size = len(manifest_bytes)
            if not 1 <= manifest_size <= _MAX_MANIFEST_SIZE:
                raise ValueError(f"cache: invalid manifest size {manifest_size} ({resolved_path})")
            stream.write(manifest_bytes)
            stream.write(
                _CACHE_FOOTER_STRUCT.pack(
                    manifest_size,
                    hashlib.sha256(manifest_bytes).digest(),
                )
            )
        temporary_path.replace(resolved_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    logger.info(
        "Saved %d graphs to %s (%.1f MB)",
        len(graphs),
        resolved_path,
        resolved_path.stat().st_size / 1e6,
    )
    return resolved_path
