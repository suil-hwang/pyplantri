# plantri.py
from __future__ import annotations

import math
import os
import queue
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import as_file, files
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, cast

# Endpoint-sorted undirected support-edge key.
SupportEdge = tuple[int, int]

# Vertex-index face-boundary sequence.
FaceCycle = tuple[int, ...]

# Derived vertex-indexed exterior-view CW neighbor cycles (0-based).
Embedding = tuple[tuple[int, ...], ...]

MIN_SUPPORTED_DUAL_VERTEX_COUNT = 3
_PLANTRI_MAXN = 64
BUNDLED_MAX_DUAL_VERTEX_COUNT = _PLANTRI_MAXN - 2

_CLEANUP_TIMEOUT_S = 5.0


class PrimalMinimumDegree(Enum):
    """Minimum-degree policy for plantri's primal quadrangulation ``G``.

    - ``AT_LEAST_2``: explicit ``-q -c2 -m2``.
    - ``AT_LEAST_3``: ``-q -c2``; default minimum degree 3; ``-m3`` omitted.
    """

    AT_LEAST_2 = 2
    AT_LEAST_3 = 3

    @property
    def _plantri_switches(self) -> tuple[str, ...]:
        return ("-q", "-c2", "-m2") if self is self.AT_LEAST_2 else ("-q", "-c2")

    @property
    def _minimum_nonempty_dual_vertex_count(self) -> int:
        return MIN_SUPPORTED_DUAL_VERTEX_COUNT if self is self.AT_LEAST_2 else 6


@dataclass(frozen=True, slots=True, init=False)
class SimpleQuadrangulation:
    """Candidate primal quadrangulation ``G`` paired with ``G*``."""

    _dual: QuarticPlaneMap = field(repr=False)
    _embedding: Embedding = field(repr=False)
    _faces: tuple[FaceCycle, ...] = field(repr=False)

    def __init__(self, dual: QuarticPlaneMap) -> None:
        """Derive the candidate-primal rotation system paired with ``dual``."""
        if type(dual) is not QuarticPlaneMap:
            raise TypeError("dual must be QuarticPlaneMap")
        twin = dual.twin
        right_face = [-1] * len(twin)
        face_orbits: list[list[int]] = []
        # Label dual right-face orbits with their primal-vertex indices.
        for start_dart in range(len(twin)):
            if right_face[start_dart] >= 0:
                continue
            face_index = len(face_orbits)
            face_darts: list[int] = []
            dart = start_dart
            while right_face[dart] < 0:
                right_face[dart] = face_index
                face_darts.append(dart)
                dart = dual.right_face_next(dart)
            face_orbits.append(face_darts)
        embedding = tuple(
            tuple(right_face[twin[dart]] for dart in face_darts)
            for face_darts in face_orbits
        )
        faces = tuple(
            tuple(right_face[twin[dart]] for dart in range(base, base + 4))
            for base in range(0, len(twin), 4)
        )
        object.__setattr__(self, "_dual", dual)
        object.__setattr__(self, "_embedding", embedding)
        object.__setattr__(self, "_faces", faces)

    @property
    def dual(self) -> QuarticPlaneMap:
        """Return the owning candidate dual ``G*``."""
        return self._dual

    @property
    def num_vertices(self) -> int:
        """Return the number of candidate-primal vertices."""
        return len(self._embedding)

    @property
    def embedding(self) -> Embedding:
        """Return vertex-indexed exterior-view-CW candidate-primal rotations."""
        return self._embedding

    @property
    def faces(self) -> tuple[FaceCycle, ...]:
        """Return candidate-primal faces indexed by candidate-dual vertices."""
        return self._faces

    @property
    def vertex_to_dual_face(self) -> range:
        """Return the identity map from candidate-primal vertices to dual faces."""
        return range(self.num_vertices)


@dataclass(frozen=True, slots=True)
class QuarticPlaneMap:
    """Candidate dual plane graph ``G*`` paired with its primal quadrangulation ``G``."""

    twin: bytes
    graph_id: int = field(default=0, compare=False)
    _faces: tuple[FaceCycle, ...] | None = field(default=None, init=False, repr=False, compare=False)
    _edge_multiplicity: Mapping[SupportEdge, int] | None = field(default=None, init=False, repr=False, compare=False)
    _face_size_sequence: tuple[int, ...] | None = field(default=None, init=False, repr=False, compare=False)
    _primal: SimpleQuadrangulation | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate compact record fields and the twin encoding."""
        if type(self.twin) is not bytes:
            raise TypeError("twin must be bytes")
        if type(self.graph_id) is not int:
            raise TypeError("graph_id must be int")
        if self.graph_id < 0:
            raise ValueError(f"graph_id must be non-negative, got {self.graph_id}")

        dart_count = len(self.twin)
        if dart_count == 0 or dart_count % 4:
            raise ValueError("dart count must be a positive multiple of 4")
        if dart_count > 256:
            raise ValueError("byte-valued twin supports at most 256 darts")
        for dart, opposite in enumerate(self.twin):
            if opposite >= dart_count:
                raise ValueError(f"twin out of range: {dart}->{opposite}")
            if opposite == dart:
                raise ValueError(f"self-twin dart: {dart}")
            if self.twin[opposite] != dart:
                raise ValueError(f"twin is not involutive: {dart}->{opposite}")

    def __reduce__(self) -> tuple[type[QuarticPlaneMap], tuple[bytes, int]]:
        """Serialize only the persistent twin and namespace-local Graph ID."""
        return type(self), (self.twin, self.graph_id)

    @property
    def num_vertices(self) -> int:
        """Return the number of candidate-dual vertices."""
        return len(self.twin) // 4

    @property
    def embedding(self) -> Embedding:
        """Return vertex-indexed exterior-view-CW candidate-dual rotations."""
        twin = self.twin
        return tuple(
            tuple(opposite // 4 for opposite in twin[base : base + 4])
            for base in range(0, len(twin), 4)
        )

    @property
    def faces(self) -> tuple[FaceCycle, ...]:
        """Return candidate-dual right-face vertex cycles."""
        faces = self._faces
        if faces is None:
            visited = bytearray(len(self.twin))
            face_cycles: list[FaceCycle] = []
            for start_dart in range(len(self.twin)):
                if visited[start_dart]:
                    continue
                face_cycle: list[int] = []
                dart = start_dart
                while not visited[dart]:
                    visited[dart] = 1
                    face_cycle.append(self.vertex(dart))
                    dart = self.right_face_next(dart)
                face_cycles.append(tuple(face_cycle))
            faces = tuple(face_cycles)
            object.__setattr__(self, "_faces", faces)
        return faces

    @property
    def num_faces(self) -> int:
        """Return the number of candidate-dual face orbits."""
        return len(self.faces)

    @property
    def edge_multiplicity(self) -> Mapping[SupportEdge, int]:
        """Return multiplicities of normalized candidate-dual support edges."""
        multiplicities = self._edge_multiplicity
        if multiplicities is None:
            counts: dict[SupportEdge, int] = {}
            for dart, twin_dart in enumerate(self.twin):
                # The lower dart counts each twin pair once and orders its endpoints.
                if dart > twin_dart:
                    continue
                edge = (dart // 4, twin_dart // 4)
                counts[edge] = counts.get(edge, 0) + 1
            multiplicities = MappingProxyType(dict(sorted(counts.items())))
            object.__setattr__(self, "_edge_multiplicity", multiplicities)
        return multiplicities

    @property
    def support_edges(self) -> tuple[SupportEdge, ...]:
        """Return normalized candidate-dual support edges in lexicographic order."""
        return tuple(self.edge_multiplicity)

    @property
    def double_edges(self) -> frozenset[SupportEdge]:
        """Return dual support edges of multiplicity two."""
        return frozenset(
            edge for edge, multiplicity in self.edge_multiplicity.items()
            if multiplicity == 2
        )

    @property
    def face_size_sequence(self) -> tuple[int, ...]:
        """Return candidate-dual face sizes in nonincreasing order."""
        sequence = self._face_size_sequence
        if sequence is None:
            sequence = tuple(sorted(map(len, self.faces), reverse=True))
            object.__setattr__(self, "_face_size_sequence", sequence)
        return sequence

    @property
    def vertex_to_primal_face(self) -> range:
        """Return the identity map from candidate-dual vertices to primal faces."""
        return range(self.num_vertices)

    @property
    def primal(self) -> SimpleQuadrangulation:
        """Return the candidate-primal rotation view paired with this ``G*``."""
        primal = self._primal
        if primal is None:
            primal = SimpleQuadrangulation(self)
            object.__setattr__(self, "_primal", primal)
        return primal

    @staticmethod
    def vertex(dart: int) -> int:
        """Return the source vertex of a valid dart."""
        return dart // 4

    @staticmethod
    def next_at_vertex(dart: int) -> int:
        """Return the next valid dart in its exterior-view-CW vertex rotation."""
        return 4 * (dart // 4) + (dart + 1) % 4

    @staticmethod
    def prev_at_vertex(dart: int) -> int:
        """Return the previous valid dart in its exterior-view-CW vertex rotation."""
        return 4 * (dart // 4) + (dart - 1) % 4

    def right_face_next(self, dart: int) -> int:
        """Return the next valid dart along the face on a dart's right."""
        return self.prev_at_vertex(self.twin[dart])

    def neighbor(self, dart: int) -> int:
        """Return the target vertex of a valid dart."""
        return self.vertex(self.twin[dart])


class PlantriError(Exception):
    """Failure while invoking plantri_sqs or decoding its fixed records."""


@dataclass(frozen=True, slots=True)
class PlantriEnumerationResult:
    """Immutable map batch with zero time-to-first-record for an empty stream.

    The first-record interval includes resource resolution, process startup,
    FILTER conversion, pipe transfer, and Python decoding.
    ``remaining_s`` measures from the first decoded record through cleanup; for
    an empty batch, it equals ``total_s``.
    """

    graphs: tuple[QuarticPlaneMap, ...]
    time_to_first_record_s: float
    remaining_s: float

    @property
    def total_s(self) -> float:
        """Return total enumeration time."""
        return self.time_to_first_record_s + self.remaining_s


class _PipeReader:
    """Read plantri stdout in a background thread under one shared deadline."""

    def __init__(self, stream: BinaryIO, timeout: float | None) -> None:
        """Initialize the reader and its shared deadline."""
        self._stream = stream
        self._timeout = timeout
        self._deadline = None if timeout is None else time.monotonic() + timeout
        self._queue: queue.Queue[bytes | BaseException | None] = queue.Queue(maxsize=2)
        self._cancelled = threading.Event()
        self._eof_received = False
        self._thread = threading.Thread(target=self._read, name="pyplantri-stdout", daemon=True)

    def start(self) -> None:
        """Start the stdout reader thread."""
        self._thread.start()

    def cancel(self) -> None:
        """Request producer cancellation."""
        self._cancelled.set()

    def join(self) -> None:
        """Wait for bounded reader cleanup."""
        self._thread.join(_CLEANUP_TIMEOUT_S)
        if self._thread.is_alive():
            raise PlantriError("plantri_sqs: stdout reader did not stop")

    def remaining_timeout(self) -> float | None:
        """Return the remaining shared wait or raise the timeout error."""
        if self._deadline is None:
            return None
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise self.timeout_error()
        return remaining

    def timeout_error(self) -> PlantriError:
        """Build the shared timeout error."""
        return PlantriError(f"plantri_sqs: timed out after {self._timeout}s")

    def chunks(self) -> Iterator[bytes]:
        """Yield stdout chunks until EOF or failure."""
        while True:
            try:
                item = self._queue.get(timeout=self.remaining_timeout())
            except queue.Empty:
                raise self.timeout_error() from None
            if item is None:
                self._eof_received = True
                return
            if isinstance(item, BaseException):
                detail = _error_detail(str(item))
                raise PlantriError(f"plantri_sqs: failed to read binary stdout: {detail}") from item
            yield item

    @property
    def eof_received(self) -> bool:
        """Return whether the consumer received the stdout EOF marker."""
        return self._eof_received

    def _publish(self, item: bytes | BaseException | None) -> None:
        """Publish one item unless cancellation is requested."""
        while not self._cancelled.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                pass

    def _read(self) -> None:
        """Publish stdout chunks followed by one terminal item."""
        try:
            read = getattr(self._stream, "read1", self._stream.read)
            while not self._cancelled.is_set() and (chunk := read(64 * 1024)):
                self._publish(chunk)
        except BaseException as error:
            self._publish(error)
        else:
            self._publish(None)


def _error_detail(text: str | bytes, *, limit: int = 4000) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def _add_cleanup_note(
    error: BaseException,
    cleanup_error: BaseException,
    *,
    resource: str = "process",
) -> None:
    detail = _error_detail(str(cleanup_error), limit=240)
    error.add_note(f"pyplantri: {resource} cleanup failed: {detail}")


def _validate_enumeration_request(
    dual_vertex_count: int,
    primal_minimum_degree: PrimalMinimumDegree,
    max_count: int,
    timeout: float | None,
) -> None:
    if type(dual_vertex_count) is not int:
        raise ValueError(f"dual_vertex_count must be int; got {dual_vertex_count!r}")

    minimum = MIN_SUPPORTED_DUAL_VERTEX_COUNT
    maximum = BUNDLED_MAX_DUAL_VERTEX_COUNT
    if not minimum <= dual_vertex_count <= maximum:
        raise ValueError(f"dual_vertex_count unsupported: {dual_vertex_count}; expected [{minimum},{maximum}]")
    if type(max_count) is not int or max_count < 0:
        raise ValueError(f"max_count: expected int >= 0, got {max_count!r}")
    if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError(f"plantri_sqs: timeout must be None or finite positive; got {timeout!r}")
    if not isinstance(primal_minimum_degree, PrimalMinimumDegree):
        raise TypeError(f"primal_minimum_degree: expected PrimalMinimumDegree, got {primal_minimum_degree!r}")


@contextmanager
def _resolved_executable() -> Generator[Path, None, None]:
    exe_name = "plantri_sqs.exe" if os.name == "nt" else "plantri_sqs"
    resource = files("pyplantri").joinpath("bin", exe_name)
    with ExitStack() as stack:
        try:
            candidate = stack.enter_context(as_file(resource))
        except FileNotFoundError as error:
            raise PlantriError(f"plantri_sqs: bundled executable {exe_name} was not found") from error
        executable = candidate.resolve()
        if not executable.is_file():
            raise PlantriError(f"plantri_sqs: executable not found {executable}")
        if os.name == "posix" and not os.access(executable, os.X_OK):
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        yield executable


def _start_process(
    executable: Path,
    dual_vertex_count: int,
    primal_minimum_degree: PrimalMinimumDegree,
    stderr_file: BinaryIO,
) -> subprocess.Popen[bytes]:
    primal_vertex_count = dual_vertex_count + 2
    command = [
        str(executable),
        *primal_minimum_degree._plantri_switches,
        str(primal_vertex_count),
    ]
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
    except FileNotFoundError as error:
        raise PlantriError(f"plantri_sqs: executable not found {executable}") from error
    except OSError as error:
        detail = _error_detail(str(error))
        raise PlantriError(f"plantri_sqs: executable is not runnable {executable}: {detail}") from error


def _build_execution_error(
    process: subprocess.Popen[bytes],
    stderr_file: BinaryIO,
) -> PlantriError:
    stderr_file.flush()
    stderr_file.seek(0)
    detail = _error_detail(stderr_file.read()) or "no output"
    return PlantriError(f"plantri_sqs: execution failed (exit {process.returncode}); {detail}")


def _stop_process(process: subprocess.Popen[bytes]) -> int | None:
    """Stop if running; return a naturally observed exit code, otherwise ``None``."""
    observed_return_code = process.poll()
    if observed_return_code is not None:
        return observed_return_code
    try:
        process.terminate()
    except OSError:
        observed_return_code = process.poll()
        if observed_return_code is None:
            raise
        return observed_return_code
    try:
        process.wait(timeout=_CLEANUP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=_CLEANUP_TIMEOUT_S)
        except subprocess.TimeoutExpired as error:
            raise PlantriError("plantri_sqs: process did not exit after kill") from error
    return None


def _finalize_filter_run(
    process: subprocess.Popen[bytes],
    reader: _PipeReader,
) -> bool:
    """Release one FILTER run and report an observed natural nonzero exit."""
    reader.cancel()
    try:
        if not reader.eof_received:
            # None means this caller requested termination; an observed natural
            # nonzero exit remains a failure even when the prefix limit was met.
            observed_return_code = _stop_process(process)
            return observed_return_code not in (None, 0)
        try:
            return process.wait(timeout=reader.remaining_timeout()) != 0
        except (PlantriError, subprocess.TimeoutExpired):
            _stop_process(process)
            raise reader.timeout_error() from None
    finally:
        try:
            reader.join()
        finally:
            if process.stdout is not None:
                process.stdout.close()


def _iter_fixed_records(
    chunks: Iterator[bytes],
    dual_vertex_count: int,
) -> Iterator[bytes]:
    dual_dart_count = 4 * dual_vertex_count
    primal_vertex_count = dual_vertex_count + 2
    record_size = dual_dart_count + primal_vertex_count
    carry = b""
    record_index = 0
    for chunk in chunks:
        data = carry + chunk
        complete_size = len(data) - len(data) % record_size
        for start in range(0, complete_size, record_size):
            yield data[start : start + record_size]
            record_index += 1
        carry = data[complete_size:]
    if carry:
        raise PlantriError(f"plantri_sqs: truncated fixed record {record_index} ({len(carry)}/{record_size} bytes)")


def _decode_filter_record(
    record: bytes,
    dual_vertex_count: int,
    primal_minimum_degree: PrimalMinimumDegree,
    graph_id: int,
) -> QuarticPlaneMap:
    dual_dart_count = 4 * dual_vertex_count
    twin = record[:dual_dart_count]
    primal_degree_profile = record[dual_dart_count:]
    try:
        dual = QuarticPlaneMap(twin=twin, graph_id=graph_id)
        primal_vertex_count = dual.num_vertices + 2
        if len(primal_degree_profile) != primal_vertex_count:
            raise ValueError(f"primal degree profile length {len(primal_degree_profile)}!={primal_vertex_count}")
        if any(left < right for left, right in pairwise(primal_degree_profile)):
            raise ValueError("primal degree profile is not descending")
        if primal_degree_profile[-1] < primal_minimum_degree.value:
            raise PlantriError(f"plantri_sqs: record {graph_id} primal minimum degree below {primal_minimum_degree.value}")
        if primal_degree_profile[0] >= primal_vertex_count:
            raise ValueError("primal degree profile exceeds the simple-primal maximum")
        degree_sum = sum(primal_degree_profile)
        if degree_sum != len(twin):
            raise ValueError(f"primal degree profile sum {degree_sum}!={len(twin)}")
        object.__setattr__(dual, "_face_size_sequence", tuple(primal_degree_profile))
        return dual
    except (TypeError, ValueError) as error:
        raise PlantriError(f"plantri_sqs: malformed record {graph_id}: {error}") from error


@contextmanager
def _filter_session(
    dual_vertex_count: int,
    primal_minimum_degree: PrimalMinimumDegree,
    timeout: float | None,
) -> Generator[_PipeReader, None, None]:
    """Own one FILTER process and yield its bounded stdout reader."""
    with _resolved_executable() as executable, tempfile.TemporaryFile() as stderr_file:
        stderr = cast(BinaryIO, stderr_file)
        process = _start_process(
            executable,
            dual_vertex_count,
            primal_minimum_degree,
            stderr,
        )
        stdout = cast(BinaryIO, process.stdout)
        try:
            reader = _PipeReader(stdout, timeout)
            reader.start()
        except BaseException as setup_error:
            try:
                _stop_process(process)
            except BaseException as cleanup_error:
                _add_cleanup_note(setup_error, cleanup_error)
            try:
                stdout.close()
            except BaseException as cleanup_error:
                _add_cleanup_note(setup_error, cleanup_error, resource="stdout")
            raise
        body_error: BaseException | None = None
        try:
            yield reader
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                natural_nonzero_exit_observed = _finalize_filter_run(process, reader)
                execution_error = (
                    _build_execution_error(process, stderr)
                    if natural_nonzero_exit_observed
                    else None
                )
            except BaseException as finalization_error:
                if body_error is None:
                    raise
                _add_cleanup_note(body_error, finalization_error)
            else:
                if execution_error is not None:
                    raise execution_error from body_error


def enumerate_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    max_count: int,
    primal_minimum_degree: PrimalMinimumDegree = PrimalMinimumDegree.AT_LEAST_2,
    timeout: float | None = None,
) -> PlantriEnumerationResult:
    """Materialize candidate duals; each graph exposes its primal via ``.primal``.

    ``timeout`` starts with stdout-reader construction and bounds stream and
    child-exit waits, not resource resolution, process startup, or Python
    decoding already in progress.
    """
    _validate_enumeration_request(dual_vertex_count, primal_minimum_degree, max_count, timeout)
    enumeration_started_at = time.perf_counter()
    time_to_first_record_s = 0.0
    duals: list[QuarticPlaneMap] = []

    if max_count and dual_vertex_count >= primal_minimum_degree._minimum_nonempty_dual_vertex_count:
        with _filter_session(dual_vertex_count, primal_minimum_degree, timeout) as reader:
            for graph_id, fixed_record in enumerate(_iter_fixed_records(reader.chunks(), dual_vertex_count)):
                dual = _decode_filter_record(fixed_record, dual_vertex_count, primal_minimum_degree, graph_id)

                if not duals:
                    time_to_first_record_s = time.perf_counter() - enumeration_started_at

                duals.append(dual)
                if len(duals) >= max_count:
                    break

    graphs = tuple(duals)
    total_elapsed_s = time.perf_counter() - enumeration_started_at

    return PlantriEnumerationResult(
        graphs=graphs,
        time_to_first_record_s=time_to_first_record_s,
        remaining_s=total_elapsed_s - time_to_first_record_s,
    )
