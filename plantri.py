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


class PrimalMinimumDegree(Enum):
    """Minimum-degree policy for plantri's primal quadrangulation ``G``.

    - ``AT_LEAST_2``: explicit ``-q -c2 -m2``.
    - ``AT_LEAST_3``: ``-q -c2``; default minimum degree 3; ``-m3`` omitted.
    """

    AT_LEAST_2 = 2
    AT_LEAST_3 = 3


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
        face_dart_orbits: list[tuple[int, ...]] = []
        dual_face_index_by_dart = [-1] * len(twin)
        for start_dart in range(len(twin)):
            if dual_face_index_by_dart[start_dart] != -1:
                continue
            face_index = len(face_dart_orbits)
            face_darts: list[int] = []
            dart = start_dart
            while dual_face_index_by_dart[dart] == -1:
                face_darts.append(dart)
                dual_face_index_by_dart[dart] = face_index
                dart = dual.right_face_next(dart)
            face_dart_orbits.append(tuple(face_darts))
        embedding: Embedding = tuple(
            tuple(dual_face_index_by_dart[twin[dart]] for dart in face_darts)
            for face_darts in face_dart_orbits
        )
        faces = tuple(
            tuple(
                dual_face_index_by_dart[twin[dart]]
                for dart in range(base, base + 4)
            )
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
            edge
            for edge, multiplicity in self.edge_multiplicity.items()
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

MIN_SUPPORTED_DUAL_VERTEX_COUNT = 3
_BUNDLED_PLANTRI_MAX_PRIMAL_VERTEX_COUNT = 64
BUNDLED_MAX_DUAL_VERTEX_COUNT = _BUNDLED_PLANTRI_MAX_PRIMAL_VERTEX_COUNT - 2

_IO_CHUNK_SIZE = 1 << 16
_PROCESS_TERMINATE_TIMEOUT_S = 5.0
_PIPE_QUEUE_CAPACITY = 2

@dataclass(frozen=True, slots=True)
class _QuadrangulationSpec:
    primal_switches: tuple[str, ...]
    primal_minimum_degree: int
    minimum_nonempty_dual_vertex_count: int


_PRIMAL_MINIMUM_DEGREE_SPECS: dict[
    PrimalMinimumDegree, _QuadrangulationSpec
] = {
    PrimalMinimumDegree.AT_LEAST_2: _QuadrangulationSpec(
        primal_switches=("-q", "-c2", "-m2"),
        primal_minimum_degree=2,
        minimum_nonempty_dual_vertex_count=MIN_SUPPORTED_DUAL_VERTEX_COUNT,
    ),
    PrimalMinimumDegree.AT_LEAST_3: _QuadrangulationSpec(
        primal_switches=("-q", "-c2"),
        primal_minimum_degree=3,
        minimum_nonempty_dual_vertex_count=6,
    ),
}


class PlantriError(Exception):
    """Base class for invoking plantri_sqs and decoding its fixed records."""


class PlantriExecutableNotFoundError(PlantriError, FileNotFoundError):
    """The bundled plantri_sqs executable could not be resolved."""


class PlantriTimeoutError(PlantriError, TimeoutError):
    """plantri_sqs exceeded a caller-supplied wall-clock deadline."""


class _ExecutionError(PlantriError):
    """A nonzero child exit, preferred over downstream stream corruption."""


@dataclass(frozen=True, slots=True)
class PlantriEnumerationResult:
    """Immutable map batch with zero time-to-first-record for an empty stream.

    The first-record interval includes process startup, C FILTER conversion,
    pipe transfer, and Python decoding of the first fixed record.
    """

    graphs: tuple[QuarticPlaneMap, ...]
    time_to_first_record_s: float
    remaining_s: float

    @property
    def total_s(self) -> float:
        """Return total enumeration time."""
        return self.time_to_first_record_s + self.remaining_s


@dataclass(slots=True)
class _EnumerationProgress:
    started_at: float
    time_to_first_record_s: float = 0.0


@dataclass(frozen=True, slots=True)
class _StreamDeadline:
    deadline: float | None
    timeout: float | None

    @classmethod
    def from_timeout(cls, timeout: float | None) -> _StreamDeadline:
        return cls(None if timeout is None else time.monotonic() + timeout, timeout)

    def wait_timeout(self) -> float | None:
        """Return the remaining queue wait or raise the shared timeout error."""
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise self.expired()
        return remaining

    def expired(self) -> PlantriTimeoutError:
        return PlantriTimeoutError(f"plantri_sqs: timed out after {self.timeout}s")


@dataclass(frozen=True, slots=True)
class _ReadFailure:
    error: BaseException


class _PipeReader:
    """Bounded background reader so Windows pipe reads remain interruptible."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._queue: queue.Queue[bytes | _ReadFailure | None] = queue.Queue(
            _PIPE_QUEUE_CAPACITY
        )
        self._cancelled = threading.Event()
        self._eof_received = False
        self._thread = threading.Thread(
            target=self._read,
            name="pyplantri-stdout",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancelled.set()

    def join(self) -> None:
        self._thread.join(_PROCESS_TERMINATE_TIMEOUT_S)
        if self._thread.is_alive():
            raise PlantriError("plantri_sqs: stdout reader did not stop")

    def chunks(self, deadline: _StreamDeadline) -> Iterator[bytes]:
        while True:
            try:
                item = self._queue.get(timeout=deadline.wait_timeout())
            except queue.Empty:
                raise deadline.expired() from None
            if item is None:
                self._eof_received = True
                return
            if isinstance(item, _ReadFailure):
                detail = _summarize_process_text(str(item.error))
                raise PlantriError(
                    f"plantri_sqs: failed to read binary stdout: {detail}"
                ) from item.error
            yield item

    @property
    def exhausted(self) -> bool:
        return self._eof_received

    def _publish(self, item: bytes | _ReadFailure | None) -> None:
        while not self._cancelled.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                pass

    def _read(self) -> None:
        read = getattr(self._stream, "read1", self._stream.read)
        try:
            while chunk := read(_IO_CHUNK_SIZE):
                self._publish(chunk)
                if self._cancelled.is_set():
                    return
        except BaseException as error:
            self._publish(_ReadFailure(error))
        else:
            self._publish(None)


def _summarize_process_text(text: str | bytes, *, limit: int = 4000) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def _validate_timeout(timeout: float | None) -> None:
    if timeout is not None and (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            f"plantri_sqs: timeout must be None or finite positive; got {timeout!r}"
        )


def _validate_request(
    dual_vertex_count: int,
    primal_minimum_degree: PrimalMinimumDegree,
    max_count: int | None,
    timeout: float | None,
) -> _QuadrangulationSpec:
    if type(dual_vertex_count) is not int:
        raise ValueError(f"dual_vertex_count must be int; got {dual_vertex_count!r}")
    if not (
        MIN_SUPPORTED_DUAL_VERTEX_COUNT
        <= dual_vertex_count
        <= BUNDLED_MAX_DUAL_VERTEX_COUNT
    ):
        expected = (
            f"[{MIN_SUPPORTED_DUAL_VERTEX_COUNT},"
            f"{BUNDLED_MAX_DUAL_VERTEX_COUNT}]"
        )
        raise ValueError(
            f"dual_vertex_count unsupported: {dual_vertex_count}; expected {expected}"
        )
    if max_count is not None and (type(max_count) is not int or max_count < 0):
        raise ValueError(f"max_count: expected int >= 0 or None, got {max_count!r}")
    _validate_timeout(timeout)
    if not isinstance(primal_minimum_degree, PrimalMinimumDegree):
        raise TypeError(
            "primal_minimum_degree: expected PrimalMinimumDegree, "
            f"got {primal_minimum_degree!r}"
        )
    return _PRIMAL_MINIMUM_DEGREE_SPECS[primal_minimum_degree]


@contextmanager
def _resolved_executable() -> Generator[Path, None, None]:
    exe_name = "plantri_sqs.exe" if os.name == "nt" else "plantri_sqs"
    resource = files("pyplantri").joinpath("bin", exe_name)
    with ExitStack() as stack:
        try:
            candidate = stack.enter_context(as_file(resource))
        except FileNotFoundError as error:
            raise PlantriExecutableNotFoundError(
                f"plantri_sqs: bundled executable {exe_name} was not found"
            ) from error
        executable = candidate.resolve()
        if not executable.is_file():
            raise PlantriExecutableNotFoundError(
                f"plantri_sqs: executable not found {executable}"
            )
        if os.name == "posix" and not os.access(executable, os.X_OK):
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        yield executable


def _start_process(
    executable: Path,
    dual_vertex_count: int,
    spec: _QuadrangulationSpec,
    stderr_file: BinaryIO,
) -> subprocess.Popen[bytes]:
    primal_vertex_count = dual_vertex_count + 2
    command = [
        str(executable),
        *spec.primal_switches,
        str(primal_vertex_count),
    ]
    try:
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
    except FileNotFoundError as error:
        raise PlantriExecutableNotFoundError(
            f"plantri_sqs: executable not found {executable}"
        ) from error
    except OSError as error:
        detail = _summarize_process_text(str(error))
        raise PlantriError(
            f"plantri_sqs: executable is not runnable {executable}: {detail}"
        ) from error


def _process_error(
    process: subprocess.Popen[bytes],
    stderr_file: BinaryIO,
) -> _ExecutionError:
    stderr_file.flush()
    stderr_file.seek(0)
    detail = _summarize_process_text(stderr_file.read()) or "no output"
    return _ExecutionError(
        f"plantri_sqs: execution failed (exit {process.returncode}); {detail}"
    )


def _finalize_process(
    process: subprocess.Popen[bytes],
    reader: _PipeReader,
    stderr_file: BinaryIO,
    deadline: _StreamDeadline,
    *,
    stream_exhausted: bool,
    intentional_stop: bool,
) -> None:
    """Check natural completion or stop an intentionally shortened stream."""
    reader.cancel()
    initial_return_code = process.poll()
    natural_exit = stream_exhausted or reader.exhausted

    def stop() -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            if process.poll() is None:
                raise
        try:
            process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    try:
        if natural_exit:
            try:
                return_code = process.wait(timeout=deadline.wait_timeout())
            except (PlantriTimeoutError, subprocess.TimeoutExpired):
                stop()
                raise deadline.expired() from None
            if return_code != 0:
                raise _process_error(process, stderr_file)
        else:
            stop()
            if (
                not intentional_stop
                and initial_return_code is not None
                and initial_return_code != 0
            ):
                raise _process_error(process, stderr_file)
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
        raise PlantriError(
            "plantri_sqs: truncated fixed record "
            f"{record_index} ({len(carry)}/{record_size} bytes)"
        )


def _decode_map(
    record: bytes,
    dual_vertex_count: int,
    spec: _QuadrangulationSpec,
    graph_id: int,
) -> QuarticPlaneMap:
    dual_dart_count = 4 * dual_vertex_count
    twin = record[:dual_dart_count]
    primal_degree_profile = record[dual_dart_count:]
    if (
        not primal_degree_profile
        or primal_degree_profile[-1] < spec.primal_minimum_degree
    ):
        raise PlantriError(
            f"plantri_sqs: record {graph_id} primal minimum degree below "
            f"{spec.primal_minimum_degree}"
        )
    try:
        graph = QuarticPlaneMap(twin=twin, graph_id=graph_id)
        primal_vertex_count = graph.num_vertices + 2
        if len(primal_degree_profile) != primal_vertex_count:
            raise ValueError(
                "primal degree profile length "
                f"{len(primal_degree_profile)}!={primal_vertex_count}"
            )
        if any(left < right for left, right in pairwise(primal_degree_profile)):
            raise ValueError("primal degree profile is not descending")
        if primal_degree_profile[0] >= primal_vertex_count:
            raise ValueError(
                "primal degree profile exceeds the simple-primal maximum"
            )
        if sum(primal_degree_profile) != len(twin):
            raise ValueError(
                "primal degree profile sum "
                f"{sum(primal_degree_profile)}!={len(twin)}"
            )
        object.__setattr__(graph, "_face_size_sequence", tuple(primal_degree_profile))
        return graph
    except (TypeError, ValueError) as error:
        raise PlantriError(
            f"plantri_sqs: malformed record {graph_id}: {error}"
        ) from error


def _iter_maps(
    dual_vertex_count: int,
    spec: _QuadrangulationSpec,
    max_count: int | None,
    timeout: float | None,
    progress: _EnumerationProgress | None = None,
) -> Generator[QuarticPlaneMap, None, None]:
    with _resolved_executable() as executable, tempfile.TemporaryFile() as stderr_file:
        process = _start_process(
            executable,
            dual_vertex_count,
            spec,
            cast(BinaryIO, stderr_file),
        )
        stdout = cast(BinaryIO, process.stdout)
        reader = _PipeReader(stdout)
        reader.start()
        deadline = _StreamDeadline.from_timeout(timeout)
        stream_exhausted = False
        intentional_stop = False
        primary_error: BaseException | None = None
        try:
            records = _iter_fixed_records(reader.chunks(deadline), dual_vertex_count)
            for graph_id, record in enumerate(records):
                graph = _decode_map(
                    record,
                    dual_vertex_count,
                    spec,
                    graph_id,
                )
                if progress is not None and graph_id == 0:
                    progress.time_to_first_record_s = (
                        time.perf_counter() - progress.started_at
                    )
                yield graph
                if max_count is not None and graph_id + 1 >= max_count:
                    intentional_stop = True
                    return
            stream_exhausted = True
        except GeneratorExit:
            intentional_stop = True
            raise
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                _finalize_process(
                    process,
                    reader,
                    cast(BinaryIO, stderr_file),
                    deadline,
                    stream_exhausted=stream_exhausted,
                    intentional_stop=intentional_stop,
                )
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                if isinstance(cleanup_error, _ExecutionError):
                    raise cleanup_error from primary_error
                detail = _summarize_process_text(str(cleanup_error), limit=240)
                primary_error.add_note(
                    f"pyplantri: process cleanup failed: {detail}"
                )


def _empty_stream() -> Generator[QuarticPlaneMap, None, None]:
    yield from ()


def iter_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    primal_minimum_degree: PrimalMinimumDegree = PrimalMinimumDegree.AT_LEAST_2,
    max_count: int | None = None,
    timeout: float | None = None,
) -> Generator[QuarticPlaneMap, None, None]:
    """Yield source-ordered candidate duals ``G*`` from the FILTER executable."""
    spec = _validate_request(
        dual_vertex_count, primal_minimum_degree, max_count, timeout
    )
    if (
        max_count == 0
        or dual_vertex_count
        < spec.minimum_nonempty_dual_vertex_count
    ):
        return _empty_stream()
    return _iter_maps(
        dual_vertex_count,
        spec,
        max_count,
        timeout,
    )


def enumerate_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    max_count: int,
    primal_minimum_degree: PrimalMinimumDegree = PrimalMinimumDegree.AT_LEAST_2,
    timeout: float | None = None,
) -> PlantriEnumerationResult:
    """Materialize a bounded source-order prefix and preserve timing fields."""
    spec = _validate_request(
        dual_vertex_count, primal_minimum_degree, max_count, timeout
    )
    started_at = time.perf_counter()
    progress = _EnumerationProgress(started_at)
    if (
        max_count == 0
        or dual_vertex_count
        < spec.minimum_nonempty_dual_vertex_count
    ):
        graphs: tuple[QuarticPlaneMap, ...] = ()
    else:
        graphs = tuple(
            _iter_maps(
                dual_vertex_count,
                spec,
                max_count,
                timeout,
                progress,
            )
        )
    elapsed_s = time.perf_counter() - started_at
    return PlantriEnumerationResult(
        graphs=graphs,
        time_to_first_record_s=progress.time_to_first_record_s,
        remaining_s=elapsed_s - progress.time_to_first_record_s,
    )
