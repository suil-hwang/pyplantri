# src/pyplantri/cache.py
from __future__ import annotations

import gzip
import hashlib
import io
import json
import mmap
import os
import struct
import tempfile
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast, overload

from .plane_graph import (
    MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT,
    QuarticPlaneMap,
)
from .plantri_interface import QuadrangulationDualClass

CACHE_FORMAT_VERSION = 12
CACHE_DEFAULT_CHUNK_SIZE = 512
_CACHE_MAGIC = b"PYPLANTRI"
_CACHE_FOOTER_STRUCT = struct.Struct(">9sIQ32s")
_MAX_MANIFEST_SIZE = 64 * 1024 * 1024
# Bound both physical reads and decompressed chunks under hostile manifests.
_MAX_RAW_CHUNK_SIZE = 64 * 1024 * 1024
_MAX_ENCODED_CHUNK_SIZE = _MAX_RAW_CHUNK_SIZE + 1024 * 1024
_MAX_GRAPH_COUNT = (1 << 64) - 1
_UINT32_LIMIT = 1 << 32
_INDEX_ENCODING = "stored-index-plus-one-le"
_IMPLICIT_RECORD_ENCODING = "twin-u8-implicit-id"
_MAX_RECORDS_PER_CHUNK = _MAX_RAW_CHUNK_SIZE // (
    4 * MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT + 8
)

CacheGraphClass = Literal["quartic_multigraph", "simple_quartic"]
CacheIndexMode = Literal["implicit", "explicit"]
CacheValidation = Literal["envelope", "topology"]
_CacheGraphClassInput = CacheGraphClass | QuadrangulationDualClass
_Compression = Literal["gzip", "none"]
_SUPPORTED_GRAPH_CLASSES = frozenset({"quartic_multigraph", "simple_quartic"})


def _one_line(value: object, *, limit: int) -> str:
    """Bound one diagnostic fragment without trusting its string conversion."""
    try:
        text = str(value)
    except BaseException:
        text = type(value).__name__
    text = " ".join(text.split()) or type(value).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _brief(value: object) -> str:
    """Return a bounded, single-line representation for an invalid value."""
    try:
        representation = repr(value)
    except BaseException:
        representation = type(value).__name__
    return _one_line(representation, limit=120)


def _location(filepath: Path | None) -> str:
    """Format an optional path suffix for compact errors."""
    return f" ({_one_line(filepath, limit=160)})" if filepath is not None else ""


def _cache_error(detail: str, filepath: Path | None = None) -> ValueError:
    """Build one bounded, single-line cache error."""
    return ValueError(f"cache: {_one_line(detail, limit=320)}{_location(filepath)}")


def _invalid(
    field: str,
    actual: object,
    expected: str | None = None,
    filepath: Path | None = None,
) -> ValueError:
    """Build a compact invalid-field error with an optional expected contract."""
    detail = f"invalid {field}={_brief(actual)}"
    if expected is not None:
        detail += f" expected={expected}"
    return _cache_error(detail, filepath)


def _mismatch(
    field: str,
    actual: object,
    expected: object,
    filepath: Path | None = None,
) -> ValueError:
    """Build a compact actual-versus-expected invariant error."""
    return _cache_error(f"{field} {_brief(actual)}!={_brief(expected)}", filepath)


def _note_cleanup_failure(
    error: BaseException,
    message: str,
    cleanup: Callable[[], object],
) -> None:
    """Run cleanup without replacing the primary failure."""
    try:
        cleanup()
    except BaseException as cleanup_error:
        detail = " ".join(str(cleanup_error).split()) or type(cleanup_error).__name__
        error.add_note(f"cache: {message}: {detail[:240]}")


def _require_int_between(
    value: Any,
    *,
    field_name: str,
    minimum: int,
    maximum: int,
    filepath: Path | None = None,
) -> None:
    """Require an exact integer in one closed interval."""
    if type(value) is not int or not minimum <= value <= maximum:
        expected = f"int[{minimum},{maximum}]"
        raise _invalid(field_name, value, expected, filepath)


def _require_order_name(
    value: Any,
    *,
    field_name: str,
    filepath: Path | None = None,
) -> None:
    """Require one nonempty, single-line storage-order name."""
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise _invalid(field_name, value, "nonempty single-line str", filepath)


def _normalize_graph_class(
    value: _CacheGraphClassInput,
    *,
    field_name: str,
) -> CacheGraphClass:
    """Normalize the public enum or string graph class."""
    raw_value = value.value if isinstance(value, QuadrangulationDualClass) else value
    if raw_value == "quartic_multigraph":
        return "quartic_multigraph"
    if raw_value == "simple_quartic":
        return "simple_quartic"
    raise _invalid(field_name, value, "supported graph class")


def _parse_sha256(
    value: Any,
    *,
    field_name: str,
    filepath: Path | None = None,
) -> bytes:
    """Parse one canonical lowercase SHA-256 digest."""
    if type(value) is str and len(value) == 64:
        try:
            digest = bytes.fromhex(value)
        except ValueError:
            pass
        else:
            if digest.hex() == value:
                return digest
    raise _invalid(field_name, value, "lowercase sha256", filepath)


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    """Exact record-sequence identity and caller-owned cache provenance."""

    format_version: int
    dual_vertex_count: int
    graph_count: int
    graph_class: CacheGraphClass
    record_sequence_sha256: str
    storage_order_name: str = "source"
    storage_order_version: int = 1


@dataclass(frozen=True, slots=True)
class _CacheChunk:
    """Describe one independently hashed physical chunk."""

    first_graph_index: int
    graph_count: int
    offset: int
    size: int
    digest: bytes


@dataclass(frozen=True, slots=True)
class _GraphIdIndex:
    """Describe the explicit Graph-ID inverse index."""

    offset: int
    size: int
    width: Literal[4, 8]
    digest: bytes


@dataclass(frozen=True, slots=True)
class _CacheManifest:
    """Hold the strictly validated v12 physical layout."""

    metadata: CacheMetadata
    chunk_size: int
    compression: _Compression
    record_encoding: str
    record_size: int
    graph_id_index: _GraphIdIndex | None
    chunks: tuple[_CacheChunk, ...]

    @property
    def index_mode(self) -> CacheIndexMode:
        """Derive physical ID encoding from index presence."""
        return "implicit" if self.graph_id_index is None else "explicit"


def _validate_metadata_fields(
    metadata: CacheMetadata,
    *,
    filepath: Path | None = None,
) -> None:
    """Validate the complete public metadata schema."""
    if (
        type(metadata.format_version) is not int
        or metadata.format_version != CACHE_FORMAT_VERSION
    ):
        error = _mismatch(
            "format_version", metadata.format_version, CACHE_FORMAT_VERSION, filepath
        )
        raise error
    _require_int_between(
        metadata.dual_vertex_count,
        field_name="metadata.dual_vertex_count",
        minimum=3,
        maximum=MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT,
        filepath=filepath,
    )
    _require_int_between(
        metadata.graph_count,
        field_name="metadata.graph_count",
        minimum=0,
        maximum=_MAX_GRAPH_COUNT,
        filepath=filepath,
    )
    _require_int_between(
        metadata.storage_order_version,
        field_name="metadata.storage_order_version",
        minimum=1,
        maximum=_MAX_GRAPH_COUNT,
        filepath=filepath,
    )
    if (
        type(metadata.graph_class) is not str
        or metadata.graph_class not in _SUPPORTED_GRAPH_CLASSES
    ):
        error = _invalid(
            "graph_class", metadata.graph_class, "supported class", filepath
        )
        raise error
    _require_order_name(
        metadata.storage_order_name,
        field_name="metadata.storage_order_name",
        filepath=filepath,
    )
    _parse_sha256(
        metadata.record_sequence_sha256,
        field_name="metadata.record_sequence_sha256",
        filepath=filepath,
    )
    if (
        metadata.graph_class == "simple_quartic"
        and metadata.graph_count
        and metadata.dual_vertex_count < 6
    ):
        detail = f"nonempty simple_quartic n={metadata.dual_vertex_count}<6"
        raise _cache_error(detail, filepath)


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
        detail = f"invalid metadata type {type(metadata).__name__}"
        raise _cache_error(detail)
    resolved_path = Path(filepath) if filepath is not None else None
    _validate_metadata_fields(metadata, filepath=resolved_path)
    if expected_dual_vertex_count is not None:
        _require_int_between(
            expected_dual_vertex_count,
            field_name="expected_dual_vertex_count",
            minimum=3,
            maximum=MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT,
        )
        if metadata.dual_vertex_count != expected_dual_vertex_count:
            error = _mismatch(
                "dual_vertex_count",
                metadata.dual_vertex_count,
                expected_dual_vertex_count,
                resolved_path,
            )
            raise error
    if expected_graph_class is not None:
        graph_class = _normalize_graph_class(
            expected_graph_class,
            field_name="expected_graph_class",
        )
        if metadata.graph_class != graph_class:
            error = _mismatch(
                "graph_class", metadata.graph_class, graph_class, resolved_path
            )
            raise error
    if expected_storage_order_name is not None:
        _require_order_name(
            expected_storage_order_name,
            field_name="expected_storage_order_name",
        )
        if metadata.storage_order_name != expected_storage_order_name:
            error = _mismatch(
                "storage_order_name",
                metadata.storage_order_name,
                expected_storage_order_name,
                resolved_path,
            )
            raise error
    if expected_storage_order_version is not None:
        _require_int_between(
            expected_storage_order_version,
            field_name="expected_storage_order_version",
            minimum=1,
            maximum=_MAX_GRAPH_COUNT,
        )
        if metadata.storage_order_version != expected_storage_order_version:
            error = _mismatch(
                "storage_order_version",
                metadata.storage_order_version,
                expected_storage_order_version,
                resolved_path,
            )
            raise error


def _manifest_from_bytes(
    manifest_bytes: bytes,
    filepath: Path,
    *,
    manifest_offset: int,
) -> _CacheManifest:
    """Parse the manifest and prove its exact file partition."""
    try:
        manifest_data = json.loads(manifest_bytes)
    except ValueError as exc:
        error = _cache_error("invalid manifest JSON", filepath)
        raise error from exc
    if type(manifest_data) is not dict:
        detail = f"invalid manifest type {type(manifest_data).__name__}"
        raise _cache_error(detail, filepath)
    expected_manifest_keys = {
        "metadata",
        "chunk_size",
        "compression",
        "record_encoding",
        "record_size",
        "graph_id_index",
        "chunks",
    }
    actual_manifest_keys = set(manifest_data)
    if actual_manifest_keys != expected_manifest_keys:
        missing = _brief(sorted(expected_manifest_keys - actual_manifest_keys))
        extra = _brief(sorted(actual_manifest_keys - expected_manifest_keys))
        detail = f"invalid manifest keys missing={missing} extra={extra}"
        raise _cache_error(detail, filepath)

    metadata_data = manifest_data["metadata"]
    if type(metadata_data) is not dict:
        detail = f"invalid metadata type {type(metadata_data).__name__}"
        raise _cache_error(detail, filepath)
    expected_metadata_keys = {
        "format_version",
        "dual_vertex_count",
        "graph_count",
        "graph_class",
        "record_sequence_sha256",
        "storage_order_name",
        "storage_order_version",
    }
    actual_metadata_keys = set(metadata_data)
    if actual_metadata_keys != expected_metadata_keys:
        missing = _brief(sorted(expected_metadata_keys - actual_metadata_keys))
        extra = _brief(sorted(actual_metadata_keys - expected_metadata_keys))
        detail = f"metadata keys mismatch missing={missing} extra={extra}"
        raise _cache_error(detail, filepath)
    metadata = CacheMetadata(**metadata_data)
    _validate_metadata_fields(metadata, filepath=filepath)

    chunk_size = manifest_data["chunk_size"]
    _require_int_between(
        chunk_size,
        field_name="manifest.chunk_size",
        minimum=1,
        maximum=_MAX_RECORDS_PER_CHUNK,
        filepath=filepath,
    )
    compression = manifest_data["compression"]
    if type(compression) is not str or compression not in ("gzip", "none"):
        raise _invalid("compression", compression, "'gzip'|'none'", filepath)

    index_data = manifest_data["graph_id_index"]
    graph_id_index_width: Literal[4, 8] | None
    graph_id_index_digest: bytes | None
    if index_data is None:
        graph_id_index_width = None
        graph_id_index_digest = None
        expected_record_encoding = _IMPLICIT_RECORD_ENCODING
    else:
        if type(index_data) is not dict or set(index_data) != {
            "width",
            "encoding",
            "sha256",
        }:
            raise _cache_error("invalid graph_id_index descriptor", filepath)
        graph_id_index_width = index_data["width"]
        if graph_id_index_width not in (4, 8) or type(graph_id_index_width) is not int:
            error = _invalid(
                "graph_id_index.width", graph_id_index_width, "4|8", filepath
            )
            raise error
        # The +1 sentinel needs eight bytes exactly when N reaches 2^32.
        expected_width = 4 if metadata.graph_count < _UINT32_LIMIT else 8
        if graph_id_index_width != expected_width:
            error = _mismatch(
                "graph_id_index.width", graph_id_index_width, expected_width, filepath
            )
            raise error
        if index_data["encoding"] != _INDEX_ENCODING:
            error = _invalid(
                "graph_id_index.encoding",
                index_data["encoding"],
                _INDEX_ENCODING,
                filepath,
            )
            raise error
        graph_id_index_digest = _parse_sha256(
            index_data["sha256"],
            field_name="graph_id_index.sha256",
            filepath=filepath,
        )
        expected_record_encoding = f"graph-id-u{8 * graph_id_index_width}-le+twin-u8"

    record_encoding = manifest_data["record_encoding"]
    if record_encoding != expected_record_encoding:
        error = _invalid(
            "record_encoding", record_encoding, expected_record_encoding, filepath
        )
        raise error
    expected_record_size = 4 * metadata.dual_vertex_count + (graph_id_index_width or 0)
    record_size = manifest_data["record_size"]
    if type(record_size) is not int or record_size != expected_record_size:
        raise _mismatch("record_size", record_size, expected_record_size, filepath)

    chunk_descriptors = manifest_data["chunks"]
    if type(chunk_descriptors) is not list:
        detail = f"invalid chunks type {type(chunk_descriptors).__name__}"
        raise _cache_error(detail, filepath)
    expected_chunk_count = (metadata.graph_count + chunk_size - 1) // chunk_size
    if len(chunk_descriptors) != expected_chunk_count:
        error = _mismatch(
            "chunk count", len(chunk_descriptors), expected_chunk_count, filepath
        )
        raise error

    chunks: list[_CacheChunk] = []
    next_offset = 0
    for chunk_index, descriptor in enumerate(chunk_descriptors):
        if type(descriptor) is not list or len(descriptor) != 2:
            detail = f"invalid chunk {chunk_index} descriptor"
            raise _cache_error(detail, filepath)
        size, digest_hex = descriptor
        first_graph_index = chunk_index * chunk_size
        graph_count = min(
            chunk_size,
            metadata.graph_count - first_graph_index,
        )
        expected_raw_size = graph_count * record_size
        minimum_size = expected_raw_size if compression == "none" else 1
        maximum_size = (
            expected_raw_size if compression == "none" else _MAX_ENCODED_CHUNK_SIZE
        )
        _require_int_between(
            size,
            field_name=f"chunk[{chunk_index}].size",
            minimum=minimum_size,
            maximum=maximum_size,
            filepath=filepath,
        )
        chunks.append(
            _CacheChunk(
                first_graph_index=first_graph_index,
                graph_count=graph_count,
                offset=next_offset,
                size=size,
                digest=_parse_sha256(
                    digest_hex,
                    field_name=f"chunk[{chunk_index}].sha256",
                    filepath=filepath,
                ),
            )
        )
        next_offset += size

    graph_id_index: _GraphIdIndex | None = None
    if graph_id_index_width is not None and graph_id_index_digest is not None:
        index_size = metadata.graph_count * graph_id_index_width
        graph_id_index = _GraphIdIndex(
            offset=next_offset,
            size=index_size,
            width=graph_id_index_width,
            digest=graph_id_index_digest,
        )
        next_offset += index_size
    if next_offset != manifest_offset:
        raise _mismatch("payload_end", next_offset, manifest_offset, filepath)
    return _CacheManifest(
        metadata=metadata,
        chunk_size=chunk_size,
        compression=compression,
        record_encoding=record_encoding,
        record_size=record_size,
        graph_id_index=graph_id_index,
        chunks=tuple(chunks),
    )


def _validate_graph_envelope(
    graph: QuarticPlaneMap,
    *,
    graph_index: int,
    dual_vertex_count: int,
    graph_class: CacheGraphClass,
    graph_count: int | None,
    index_mode: CacheIndexMode,
    filepath: Path,
) -> int:
    """Validate cache-specific ID, size, and declared edge-class invariants."""
    # QuarticPlaneMap already owns core type, twin, and nonnegative-ID checks.
    graph_id = graph.graph_id
    if graph_id >= _MAX_GRAPH_COUNT:
        detail = f"graph {graph_index} graph_id={graph_id} expected<={_MAX_GRAPH_COUNT - 1}"
        raise _cache_error(detail, filepath)
    if graph_count is not None and graph_id >= graph_count:
        detail = f"graph {graph_index} graph_id={graph_id} expected=0..{graph_count - 1}"
        raise _cache_error(detail, filepath)
    if index_mode == "implicit" and graph_id != graph_index:
        detail = f"implicit graph {graph_index} graph_id={graph_id} expected={graph_index}"
        raise _cache_error(detail, filepath)
    expected_dart_count = 4 * dual_vertex_count
    if len(graph.twin) != expected_dart_count:
        detail = f"graph {graph_index} dart count {len(graph.twin)}!={expected_dart_count}"
        raise _cache_error(detail, filepath)
    maximum_multiplicity, has_loop = graph._dual_edge_envelope()
    if has_loop:
        raise _cache_error(f"graph {graph_index} contains a dual loop", filepath)
    allowed_multiplicity = 1 if graph_class == "simple_quartic" else 2
    if maximum_multiplicity > allowed_multiplicity:
        detail = f"graph {graph_index} edge multiplicity {maximum_multiplicity}>{allowed_multiplicity}"
        raise _cache_error(detail, filepath)
    return graph_id


def _validate_graph_semantics_one(
    graph: QuarticPlaneMap,
    graph_index: int,
) -> None:
    """Run the complete topology audit and compact its first error."""
    is_valid, errors = graph.audit_sqs_topology()
    if is_valid:
        return
    summary = _one_line(errors[0], limit=240)
    if len(errors) > 1:
        summary += f" (+{len(errors) - 1})"
    raise _cache_error(f"invalid graph {graph_index}: {summary}")


def _write_chunk(
    stream: BinaryIO,
    raw: bytes | bytearray,
    *,
    compression: _Compression,
    compress_level: int,
) -> tuple[int, str]:
    """Write one chunk and return its size and digest."""
    if compression == "none":
        stream.write(raw)
        return len(raw), hashlib.sha256(raw).hexdigest()
    buffer = io.BytesIO()
    # mtime=0 makes equal raw chunks byte-for-byte reproducible.
    with gzip.GzipFile(
        filename="", mode="wb", compresslevel=compress_level, fileobj=buffer, mtime=0
    ) as gzip_stream:
        gzip_stream.write(raw)
    payload = buffer.getvalue()
    stream.write(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _open_manifest(filepath: Path) -> _CacheManifest:
    """Read only the footer and manifest of one v12 cache."""
    # The footer commits the complete manifest without touching any chunk.
    with filepath.open("rb") as stream:
        file_size = stream.seek(0, io.SEEK_END)
        if file_size < _CACHE_FOOTER_STRUCT.size + 2:
            minimum_size = _CACHE_FOOTER_STRUCT.size + 2
            detail = f"truncated v12 container size={file_size} expected>={minimum_size}"
            raise _cache_error(detail, filepath)
        footer_offset = file_size - _CACHE_FOOTER_STRUCT.size
        stream.seek(footer_offset)
        raw_footer = stream.read(_CACHE_FOOTER_STRUCT.size)
        if len(raw_footer) != _CACHE_FOOTER_STRUCT.size:
            detail = f"truncated footer bytes={len(raw_footer)} expected={_CACHE_FOOTER_STRUCT.size}"
            raise _cache_error(detail, filepath)
        magic, version, manifest_size, expected_digest = _CACHE_FOOTER_STRUCT.unpack(raw_footer)
        if magic != _CACHE_MAGIC:
            detail = f"unsupported cache magic actual={_brief(magic)} expected={_brief(_CACHE_MAGIC)}"
            raise _cache_error(detail, filepath)
        if version != CACHE_FORMAT_VERSION:
            raise _mismatch("format_version", version, CACHE_FORMAT_VERSION, filepath)
        if not 2 <= manifest_size <= min(_MAX_MANIFEST_SIZE, footer_offset):
            maximum_size = min(_MAX_MANIFEST_SIZE, footer_offset)
            detail = f"invalid manifest range size={manifest_size} expected=2..{maximum_size}"
            raise _cache_error(detail, filepath)
        manifest_offset = footer_offset - manifest_size
        stream.seek(manifest_offset)
        manifest_bytes = stream.read(manifest_size)
        if len(manifest_bytes) != manifest_size:
            detail = f"truncated manifest bytes={len(manifest_bytes)} expected={manifest_size}"
            raise _cache_error(detail, filepath)
        if hashlib.sha256(manifest_bytes).digest() != expected_digest:
            raise _cache_error("manifest hash mismatch", filepath)
    return _manifest_from_bytes(
        manifest_bytes,
        filepath,
        manifest_offset=manifest_offset,
    )


class QuarticPlaneMapCache(Sequence[QuarticPlaneMap]):
    """Path-backed lazy cache sequence in physical storage order."""

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
        """Initialize from factory-validated layout and cache capacity."""
        self._filepath = filepath
        self._manifest = manifest
        self._cached_chunks: OrderedDict[int, tuple[QuarticPlaneMap, ...]] = (
            OrderedDict()
        )
        self._cached_chunk_limit = cached_chunks

    @property
    def metadata(self) -> CacheMetadata:
        """Return validated public cache metadata."""
        return self._manifest.metadata

    def __len__(self) -> int:
        """Return the manifest-declared record count."""
        return self.metadata.graph_count

    def _read_range(self, offset: int, size: int) -> bytes:
        """Read one exact byte range from the backing file."""
        with self._filepath.open("rb") as stream:
            stream.seek(offset)
            payload = stream.read(size)
        if len(payload) != size:
            detail = f"truncated range at {offset}+{size} actual={len(payload)}"
            raise _cache_error(detail, self._filepath)
        return payload

    def _decode_chunk(
        self,
        chunk_index: int,
        payload: bytes,
    ) -> tuple[QuarticPlaneMap, ...]:
        """Hash, bounded-decode, construct, and envelope-check one chunk."""
        chunk = self._manifest.chunks[chunk_index]
        if hashlib.sha256(payload).digest() != chunk.digest:
            raise _cache_error(f"chunk {chunk_index} hash mismatch", self._filepath)
        expected_size = chunk.graph_count * self._manifest.record_size
        if self._manifest.compression == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
                    # One extra byte detects decompression bombs and overlong records.
                    raw = stream.read(expected_size + 1)
            except (OSError, EOFError, zlib.error) as exc:
                error = _cache_error(f"invalid chunk {chunk_index} payload", self._filepath)
                raise error from exc
            if len(raw) != expected_size:
                detail = f"chunk {chunk_index} raw size {len(raw)}!={expected_size}"
                raise _cache_error(detail, self._filepath)
        else:
            # Manifest partitioning plus an exact range read already proves this size.
            raw = payload

        descriptor = self._manifest.graph_id_index
        width = descriptor.width if descriptor is not None else 0
        graphs: list[QuarticPlaneMap] = []
        seen_ids: set[int] | None = set() if width else None
        for local_index in range(chunk.graph_count):
            graph_index = chunk.first_graph_index + local_index
            offset = local_index * self._manifest.record_size
            if width:
                graph_id = int.from_bytes(raw[offset : offset + width], "little")
                offset += width
            else:
                graph_id = graph_index
            graph = QuarticPlaneMap(
                bytes(raw[offset : offset + 4 * self.metadata.dual_vertex_count]),
                graph_id,
            )
            graph_id = _validate_graph_envelope(
                graph,
                graph_index=graph_index,
                dual_vertex_count=self.metadata.dual_vertex_count,
                graph_class=self.metadata.graph_class,
                graph_count=len(self),
                index_mode=self._manifest.index_mode,
                filepath=self._filepath,
            )
            if seen_ids is not None:
                if graph_id in seen_ids:
                    detail = f"duplicate graph_id={graph_id} in chunk {chunk_index}"
                    raise _cache_error(detail, self._filepath)
                seen_ids.add(graph_id)
            graphs.append(graph)
        return tuple(graphs)

    def _cache_chunk(
        self,
        chunk_index: int,
        graphs: tuple[QuarticPlaneMap, ...],
    ) -> None:
        """Retain one decoded chunk under the configured LRU bound."""
        if self._cached_chunk_limit == 0:
            return
        self._cached_chunks[chunk_index] = graphs
        self._cached_chunks.move_to_end(chunk_index)
        if len(self._cached_chunks) > self._cached_chunk_limit:
            self._cached_chunks.popitem(last=False)

    def _load_chunk(self, chunk_index: int) -> tuple[QuarticPlaneMap, ...]:
        """Return one cached or freshly decoded chunk."""
        graphs = self._cached_chunks.get(chunk_index)
        if graphs is not None:
            self._cached_chunks.move_to_end(chunk_index)
            return graphs
        chunk = self._manifest.chunks[chunk_index]
        graphs = self._decode_chunk(
            chunk_index,
            self._read_range(chunk.offset, chunk.size),
        )
        self._cache_chunk(chunk_index, graphs)
        return graphs

    def _get_stored(self, stored_index: int) -> QuarticPlaneMap:
        """Return one already bounds-checked physical record."""
        chunk_index = stored_index // self._manifest.chunk_size
        chunk = self._manifest.chunks[chunk_index]
        return self._load_chunk(chunk_index)[stored_index - chunk.first_graph_index]

    @overload
    def __getitem__(self, index: int) -> QuarticPlaneMap: ...

    @overload
    def __getitem__(self, index: slice) -> list[QuarticPlaneMap]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> QuarticPlaneMap | list[QuarticPlaneMap]:
        """Return one physical record or materialized slice."""
        if isinstance(index, slice):
            result: list[QuarticPlaneMap] = []
            loaded_chunk_index = -1
            loaded_graphs: tuple[QuarticPlaneMap, ...] = ()
            for position in range(*index.indices(len(self))):
                chunk_index = position // self._manifest.chunk_size
                if chunk_index != loaded_chunk_index:
                    loaded_graphs = self._load_chunk(chunk_index)
                    loaded_chunk_index = chunk_index
                chunk = self._manifest.chunks[chunk_index]
                result.append(loaded_graphs[position - chunk.first_graph_index])
            return result
        if type(index) is not int:
            actual_type = _one_line(type(index).__name__, limit=120)
            detail = f"cache index must be int or slice, got {actual_type}"
            raise TypeError(detail)
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError("QuarticPlaneMapCache index out of range")
        return self._get_stored(position)

    def _read_chunk_payload(
        self,
        stream: BinaryIO,
        chunk_index: int,
        chunk: _CacheChunk,
    ) -> bytes:
        """Read one chunk's exact payload from an already-open stream."""
        stream.seek(chunk.offset)
        payload = stream.read(chunk.size)
        if len(payload) != chunk.size:
            detail = f"truncated chunk {chunk_index} bytes={len(payload)} expected={chunk.size}"
            raise _cache_error(detail, self._filepath)
        return payload

    def __iter__(self) -> Iterator[QuarticPlaneMap]:
        """Yield physical-order records through the bounded LRU."""
        with self._filepath.open("rb") as stream:
            for chunk_index, chunk in enumerate(self._manifest.chunks):
                graphs = self._cached_chunks.get(chunk_index)
                if graphs is None:
                    payload = self._read_chunk_payload(stream, chunk_index, chunk)
                    graphs = self._decode_chunk(chunk_index, payload)
                    self._cache_chunk(chunk_index, graphs)
                else:
                    self._cached_chunks.move_to_end(chunk_index)
                yield from graphs

    def _iter_fresh(self) -> Iterator[QuarticPlaneMap]:
        """Yield records while rereading every chunk for whole-file audits."""
        # Whole-file evidence must not trust data retained by an earlier access.
        with self._filepath.open("rb") as stream:
            for chunk_index, chunk in enumerate(self._manifest.chunks):
                payload = self._read_chunk_payload(stream, chunk_index, chunk)
                yield from self._decode_chunk(chunk_index, payload)

    def get_by_graph_id(self, graph_id: int) -> QuarticPlaneMap:
        """Resolve one dense source Graph ID to its physical record."""
        if type(graph_id) is not int:
            actual_type = _one_line(type(graph_id).__name__, limit=120)
            detail = f"graph_id must be int, got {actual_type}"
            raise TypeError(detail)
        if not 0 <= graph_id < len(self):
            raise KeyError(graph_id)
        descriptor = self._manifest.graph_id_index
        if descriptor is None:
            stored_index = graph_id
        else:
            encoded = int.from_bytes(
                self._read_range(
                    descriptor.offset + graph_id * descriptor.width,
                    descriptor.width,
                ),
                "little",
            )
            if encoded == 0 or encoded > len(self):
                detail = f"invalid stored-index sentinel {encoded} for graph_id={graph_id}"
                raise _cache_error(detail, self._filepath)
            stored_index = encoded - 1
        graph = self._get_stored(stored_index)
        if descriptor is not None and graph.graph_id != graph_id:
            detail = f"graph_id_index maps {graph_id} to graph {graph.graph_id}"
            raise _cache_error(detail, self._filepath)
        return graph

    def _scan_all(self, *, audit_semantics: bool) -> None:
        """Share one fresh integrity scan with optional topology audits."""
        descriptor = self._manifest.graph_id_index
        with ExitStack() as resources:
            index_view: memoryview | None = None
            if descriptor is not None:
                index_stream = resources.enter_context(self._filepath.open("rb"))
                if descriptor.size:
                    map_offset = (
                        descriptor.offset
                        - descriptor.offset % mmap.ALLOCATIONGRANULARITY
                    )
                    delta = descriptor.offset - map_offset
                    mapped_index = resources.enter_context(
                        mmap.mmap(
                            index_stream.fileno(),
                            delta + descriptor.size,
                            access=mmap.ACCESS_READ,
                            offset=map_offset,
                        )
                    )
                    index_view = memoryview(mapped_index)[
                        delta : delta + descriptor.size
                    ]
                    resources.callback(index_view.release)
                if (
                    hashlib.sha256(
                        index_view if index_view is not None else b""
                    ).digest()
                    != descriptor.digest
                ):
                    raise _cache_error("graph_id_index hash mismatch", self._filepath)

            sequence_hasher = hashlib.sha256()
            index_struct = (
                struct.Struct("<I" if descriptor.width == 4 else "<Q")
                if descriptor is not None
                else None
            )
            for stored_index, graph in enumerate(self._iter_fresh()):
                if audit_semantics:
                    _validate_graph_semantics_one(graph, stored_index)
                if descriptor is not None:
                    # A nonempty explicit cache necessarily mapped its nonempty index.
                    encoded = cast(struct.Struct, index_struct).unpack_from(
                        cast(memoryview, index_view),
                        graph.graph_id * descriptor.width,
                    )[0]
                    if encoded != stored_index + 1:
                        detail = f"graph_id_index mismatch for graph_id={graph.graph_id}"
                        raise _cache_error(detail, self._filepath)
                # This identifies exact physical records/order, not graph isomorphism.
                sequence_hasher.update(graph.graph_id.to_bytes(8, "little"))
                sequence_hasher.update(graph.twin)
            if sequence_hasher.hexdigest() != self.metadata.record_sequence_sha256:
                raise _cache_error("record sequence hash mismatch", self._filepath)

    def verify_integrity(self) -> None:
        """Verify every chunk, raw record, inverse index, and sequence digest."""
        self._scan_all(audit_semantics=False)

    def audit_all_graphs(self) -> None:
        """Verify integrity and every complete SQS topology invariant."""
        self._scan_all(audit_semantics=True)


def open_graph_cache(
    filepath: str | Path,
    *,
    cached_chunks: int = 1,
) -> QuarticPlaneMapCache:
    """Open the v12 raw-record cache without reading graph chunks."""
    _require_int_between(
        cached_chunks,
        field_name="cached_chunks",
        minimum=0,
        maximum=_MAX_GRAPH_COUNT,
    )
    resolved_path = Path(filepath)
    return QuarticPlaneMapCache(
        resolved_path,
        _open_manifest(resolved_path),
        cached_chunks=cached_chunks,
    )


def write_graph_cache(
    graphs: Iterable[QuarticPlaneMap],
    filepath: str | Path,
    *,
    graph_count: int | None = None,
    dual_vertex_count: int | None = None,
    graph_class: _CacheGraphClassInput,
    compression: _Compression = "gzip",
    compress_level: int = 1,
    chunk_size: int = CACHE_DEFAULT_CHUNK_SIZE,
    storage_order_name: str = "source",
    storage_order_version: int = 1,
    index_mode: CacheIndexMode = "implicit",
    validation: CacheValidation = "topology",
    durable: bool = False,
) -> Path:
    """Write v12 records atomically with envelope or full topology validation."""
    if compression not in ("gzip", "none") or type(compression) is not str:
        raise _invalid("compression", compression, "'gzip'|'none'")
    if compression == "gzip":
        _require_int_between(
            compress_level,
            field_name="compress_level",
            minimum=0,
            maximum=9,
        )
    _require_int_between(
        chunk_size,
        field_name="chunk_size",
        minimum=1,
        maximum=_MAX_RECORDS_PER_CHUNK,
    )
    _require_order_name(storage_order_name, field_name="storage_order_name")
    _require_int_between(
        storage_order_version,
        field_name="storage_order_version",
        minimum=1,
        maximum=_MAX_GRAPH_COUNT,
    )
    if index_mode not in ("implicit", "explicit") or type(index_mode) is not str:
        raise _invalid("index_mode", index_mode, "'implicit'|'explicit'")
    if validation not in ("envelope", "topology") or type(validation) is not str:
        raise _invalid("validation", validation, "'envelope'|'topology'")
    if type(durable) is not bool:
        raise _invalid("durable", durable, "bool")
    resolved_graph_class = _normalize_graph_class(
        graph_class,
        field_name="graph_class",
    )
    if dual_vertex_count is not None:
        _require_int_between(
            dual_vertex_count,
            field_name="dual_vertex_count",
            minimum=3,
            maximum=MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT,
        )
    if graph_count is not None:
        _require_int_between(
            graph_count,
            field_name="graph_count",
            minimum=0,
            maximum=_MAX_GRAPH_COUNT,
        )

    resolved_path = Path(filepath)
    expected_count = graph_count
    if expected_count is None and isinstance(graphs, Sized):
        expected_count = len(graphs)
    if index_mode == "explicit" and expected_count is None:
        raise _cache_error("graph_count required for unsized explicit-index stream")

    # stored_index + 1 reserves zero for duplicate detection in the mmap.
    width: Literal[4, 8] | None = (
        (4 if expected_count < _UINT32_LIMIT else 8)
        if index_mode == "explicit" and expected_count is not None
        else None
    )
    try:
        graph_iterator = iter(graphs)
    except TypeError as exc:
        detail = f"invalid graphs type {type(graphs).__name__}"
        error = _cache_error(detail)
        raise error from exc
    close_iterator = getattr(graph_iterator, "close", None)
    resources = ExitStack()
    temporary_path: Path | None = None
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        # A same-directory temporary file keeps the final replace atomic.
        # The wrapper proxies every binary file method but does not subclass BinaryIO.
        destination_stream = cast(
            BinaryIO,
            resources.enter_context(
                tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=resolved_path.parent,
                    suffix=".tmp",
                    delete=False,
                )
            ),
        )
        temporary_path = Path(destination_stream.name)
        index_file: BinaryIO | None = None
        writable_index: mmap.mmap | None = None
        if width is not None and expected_count is not None:
            # Disk-backed random writes avoid an O(N) in-memory inverse index.
            index_file = cast(
                BinaryIO,
                resources.enter_context(
                    tempfile.TemporaryFile(dir=resolved_path.parent)
                ),
            )
            index_file.truncate(expected_count * width)
            if expected_count:
                writable_index = resources.enter_context(
                    mmap.mmap(index_file.fileno(), 0, access=mmap.ACCESS_WRITE)
                )

        chunks: list[tuple[int, str]] = []
        raw_chunk = bytearray()
        records_in_chunk = 0
        stored_count = 0
        resolved_n = dual_vertex_count
        sequence_hasher = hashlib.sha256()
        index_struct = (
            struct.Struct("<I" if width == 4 else "<Q") if width is not None else None
        )

        for graph in graph_iterator:
            if expected_count is not None and stored_count >= expected_count:
                detail = f"graph count exceeds expected actual>{expected_count}"
                raise _cache_error(detail, resolved_path)
            if type(graph) is not QuarticPlaneMap:
                detail = f"invalid graph {stored_count} type {type(graph).__name__}"
                raise _cache_error(detail)
            if resolved_n is None:
                resolved_n = graph.dual_num_vertices
                _require_int_between(
                    resolved_n,
                    field_name="dual_vertex_count",
                    minimum=3,
                    maximum=MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT,
                )
            if (
                stored_count == 0
                and resolved_graph_class == "simple_quartic"
                and resolved_n < 6
            ):
                detail = f"nonempty simple_quartic n={resolved_n}<6"
                raise _cache_error(detail, resolved_path)
            graph_id = _validate_graph_envelope(
                graph,
                graph_index=stored_count,
                dual_vertex_count=resolved_n,
                graph_class=resolved_graph_class,
                graph_count=expected_count,
                index_mode=index_mode,
                filepath=resolved_path,
            )
            if validation == "topology":
                _validate_graph_semantics_one(graph, stored_count)
            if (
                writable_index is not None
                and index_struct is not None
                and width is not None
            ):
                index_offset = graph_id * width
                if index_struct.unpack_from(writable_index, index_offset)[0]:
                    raise _cache_error(f"duplicate graph_id={graph_id}", resolved_path)
                index_struct.pack_into(writable_index, index_offset, stored_count + 1)
                raw_chunk.extend(graph_id.to_bytes(width, "little"))
            raw_chunk.extend(graph.twin)
            sequence_hasher.update(graph_id.to_bytes(8, "little"))
            sequence_hasher.update(graph.twin)
            stored_count += 1
            records_in_chunk += 1
            if records_in_chunk == chunk_size:
                chunks.append(
                    _write_chunk(
                        destination_stream,
                        raw_chunk,
                        compression=compression,
                        compress_level=compress_level,
                    )
                )
                raw_chunk.clear()
                records_in_chunk = 0

        if expected_count is not None and stored_count != expected_count:
            raise _mismatch("graph count", stored_count, expected_count, resolved_path)
        if resolved_n is None:
            raise _cache_error("dual_vertex_count required for empty graph stream")
        if raw_chunk:
            chunks.append(
                _write_chunk(
                    destination_stream,
                    raw_chunk,
                    compression=compression,
                    compress_level=compress_level,
                )
            )

        if writable_index is not None:
            writable_index.flush()
        graph_id_index: dict[str, object] | None = None
        if index_file is not None and width is not None:
            # Append and hash the completed index without materializing it.
            index_file.seek(0)
            index_hasher = hashlib.sha256()
            while block := index_file.read(1024 * 1024):
                destination_stream.write(block)
                index_hasher.update(block)
            graph_id_index = {
                "width": width,
                "encoding": _INDEX_ENCODING,
                "sha256": index_hasher.hexdigest(),
            }

        metadata = CacheMetadata(
            format_version=CACHE_FORMAT_VERSION,
            dual_vertex_count=resolved_n,
            graph_count=stored_count,
            graph_class=resolved_graph_class,
            record_sequence_sha256=sequence_hasher.hexdigest(),
            storage_order_name=storage_order_name,
            storage_order_version=storage_order_version,
        )
        record_size = 4 * resolved_n + (width or 0)
        record_encoding = (
            _IMPLICIT_RECORD_ENCODING
            if width is None
            else f"graph-id-u{8 * width}-le+twin-u8"
        )
        manifest_bytes = json.dumps(
            {
                "metadata": {
                    "format_version": metadata.format_version,
                    "dual_vertex_count": metadata.dual_vertex_count,
                    "graph_count": metadata.graph_count,
                    "graph_class": metadata.graph_class,
                    "record_sequence_sha256": metadata.record_sequence_sha256,
                    "storage_order_name": metadata.storage_order_name,
                    "storage_order_version": metadata.storage_order_version,
                },
                "chunk_size": chunk_size,
                "compression": compression,
                "record_encoding": record_encoding,
                "record_size": record_size,
                "graph_id_index": graph_id_index,
                "chunks": chunks,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(manifest_bytes) > _MAX_MANIFEST_SIZE:
            detail = f"manifest size={len(manifest_bytes)} expected<={_MAX_MANIFEST_SIZE}"
            raise _cache_error(detail, resolved_path)
        destination_stream.write(manifest_bytes)
        destination_stream.write(
            _CACHE_FOOTER_STRUCT.pack(
                _CACHE_MAGIC,
                CACHE_FORMAT_VERSION,
                len(manifest_bytes),
                hashlib.sha256(manifest_bytes).digest(),
            )
        )
        if durable:
            destination_stream.flush()
            os.fsync(destination_stream.fileno())

        # Close every file mapping and stream before iterator finalization/replace.
        resources.close()
        if callable(close_iterator):
            close = close_iterator
            close_iterator = None
            close()
        temporary_path.replace(resolved_path)
        if durable and os.name == "posix":
            # POSIX directory fsync persists the rename itself after a crash.
            try:
                directory_fd = os.open(resolved_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                except BaseException as error:
                    _note_cleanup_failure(
                        error,
                        "failed to close directory fd",
                        lambda: os.close(directory_fd),
                    )
                    raise
                os.close(directory_fd)
            except BaseException as error:
                error.add_note("cache: replace committed; durable finalization failed")
                raise
    except BaseException as error:
        _note_cleanup_failure(
            error, "failed to close temporary resources", resources.close
        )
        if temporary_path is not None:
            path_to_remove = temporary_path
            _note_cleanup_failure(
                error,
                "failed to remove temporary file",
                lambda: path_to_remove.unlink(missing_ok=True),
            )
        if callable(close_iterator):
            _note_cleanup_failure(
                error, "iterator close failed during error cleanup", close_iterator
            )
        raise

    return resolved_path
