# plantri.py
from __future__ import annotations

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
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO, cast

_PLANTRI_MAXN = 64  # plantri.c MAXN

# plantri_sqs.c requires 5 <= maxnv < MAXN primal vertices; |V(G*)| = |V(G)| - 2.
MIN_DUAL_VERTEX_COUNT = 3
MAX_DUAL_VERTEX_COUNT = (_PLANTRI_MAXN - 1) - 2


@dataclass(frozen=True, slots=True)
class DualPlaneGraph:
    """Candidate dual plane graph ``G*`` paired with its primal quadrangulation ``G``."""

    twin: bytes
    graph_id: int = field(default=0, compare=False)
    _right_faces: bytes | None = field(default=None, init=False, repr=False, compare=False)
    _faces: tuple[tuple[int, ...], ...] | None = field(default=None, init=False, repr=False, compare=False)
    _edge_multiplicity: Mapping[tuple[int, int], int] | None = field(default=None, init=False, repr=False, compare=False)
    _face_size_sequence: tuple[int, ...] | bytes | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate persistent fields and common-family topology membership."""
        twin = self.twin
        if type(twin) is not bytes or type(self.graph_id) is not int:
            raise TypeError("twin must be bytes and graph_id must be int")

        dart_count = len(twin)
        vertex_count = dart_count // 4
        valid = self.graph_id >= 0 and dart_count % 4 == 0 and 4 * MIN_DUAL_VERTEX_COUNT <= dart_count <= 256
        if valid:
            # translate composes twin with itself, 0xff marking out-of-range; a zero XOR byte is a self-twin.
            identity = bytes(range(dart_count))
            self_twins = (int.from_bytes(twin, "big") ^ int.from_bytes(identity, "big")).to_bytes(dart_count, "big")
            valid = twin.translate(twin.ljust(256, b"\xff")) == identity and b"\x00" not in self_twins
        if valid:
            reached = {0}
            pending = [0]
            while pending:
                vertex = pending.pop()
                for opposite in twin[4 * vertex : 4 * vertex + 4]:
                    neighbor = opposite // 4
                    if neighbor not in reached:
                        reached.add(neighbor)
                        pending.append(neighbor)

            # Dual right-face orbits are the vertices of the paired primal map.
            face_by_dart = [-1] * dart_count
            face_count = 0
            for start_dart in range(dart_count):
                if face_by_dart[start_dart] < 0:
                    dart = start_dart
                    while face_by_dart[dart] < 0:
                        face_by_dart[dart] = face_count
                        opposite = twin[dart]
                        dart = 4 * (opposite // 4) + (opposite % 4 - 1) % 4
                    face_count += 1

            # With n >= 3, connected + spherical + simple already forces primal degree >= 2 and 4-cycle faces.
            primal_edges = {
                frozenset((face_by_dart[dart], face_by_dart[opposite]))
                for dart, opposite in enumerate(twin)
                if dart < opposite and face_by_dart[dart] != face_by_dart[opposite]
            }
            valid = len(reached) == vertex_count and face_count == vertex_count + 2 and len(primal_edges) == dart_count // 2
        if not valid:
            raise ValueError("graph_id must be non-negative and twin must encode a spherical dual of a simple quadrangulation")

    @classmethod
    def _from_filter(
        cls,
        twin: bytes,
        graph_id: int,
        face_size_profile: bytes,
        _new=object.__new__,
        _set=object.__setattr__,
    ) -> DualPlaneGraph:
        """Construct from one trusted record emitted by the bundled C FILTER."""
        dual = _new(cls)
        _set(dual, "twin", twin)
        _set(dual, "graph_id", graph_id)
        _set(dual, "_right_faces", None)
        _set(dual, "_faces", None)
        _set(dual, "_edge_multiplicity", None)
        _set(dual, "_face_size_sequence", face_size_profile)
        return dual

    def __reduce__(self) -> tuple[type[DualPlaneGraph], tuple[bytes, int]]:
        """Serialize only the persistent twin and namespace-local Graph ID."""
        return type(self), (self.twin, self.graph_id)

    @property
    def num_vertices(self) -> int:
        """Return the number of candidate-dual vertices."""
        return len(self.twin) // 4

    @property
    def embedding(self) -> tuple[tuple[int, ...], ...]:
        """Return vertex-indexed exterior-view-CW candidate-dual rotations."""
        twin = self.twin
        return tuple(
            tuple(opposite // 4 for opposite in twin[base : base + 4])
            for base in range(0, len(twin), 4)
        )

    @property
    def right_faces(self) -> bytes:
        """Return the right face of every dart, labelled in ``faces`` order."""
        if self._right_faces is None:
            self.faces  # The single orbit walk records the dart labels as well.
        return cast(bytes, self._right_faces)

    @property
    def faces(self) -> tuple[tuple[int, ...], ...]:
        """Return candidate-dual right-face vertex cycles in first-dart order."""
        faces = self._faces
        if faces is None:
            labels = [-1] * len(self.twin)
            cycles: list[tuple[int, ...]] = []
            for start_dart in range(len(self.twin)):
                if labels[start_dart] < 0:
                    cycle: list[int] = []
                    dart = start_dart
                    while labels[dart] < 0:
                        labels[dart] = len(cycles)
                        cycle.append(self.vertex(dart))
                        dart = self.right_face_next(dart)
                    cycles.append(tuple(cycle))
            faces = tuple(cycles)
            object.__setattr__(self, "_faces", faces)
            object.__setattr__(self, "_right_faces", bytes(labels))
        return faces

    @property
    def num_faces(self) -> int:
        """Return the number of candidate-dual faces."""
        return self.num_vertices + 2

    @property
    def edge_multiplicity(self) -> Mapping[tuple[int, int], int]:
        """Return multiplicities of normalized candidate-dual support edges."""
        multiplicities = self._edge_multiplicity
        if multiplicities is None:
            counts: dict[tuple[int, int], int] = {}
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
    def support_edges(self) -> tuple[tuple[int, int], ...]:
        """Return normalized candidate-dual support edges in lexicographic order."""
        return tuple(self.edge_multiplicity)

    @property
    def double_edges(self) -> frozenset[tuple[int, int]]:
        """Return dual support edges of multiplicity two."""
        return frozenset(
            edge for edge, multiplicity in self.edge_multiplicity.items()
            if multiplicity == 2
        )

    @property
    def face_size_sequence(self) -> tuple[int, ...]:
        """Return candidate-dual face sizes in nonincreasing order."""
        sequence = self._face_size_sequence
        if type(sequence) is not tuple:
            sequence = (
                tuple(sorted(map(len, self.faces), reverse=True))
                if sequence is None
                else tuple(sequence)
            )
            object.__setattr__(self, "_face_size_sequence", sequence)
        return sequence

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

    def right_face(self, dart: int) -> int:
        """Return the face on the right of a valid dart."""
        return self.right_faces[dart]

    def neighbor(self, dart: int) -> int:
        """Return the target vertex of a valid dart."""
        return self.vertex(self.twin[dart])


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
        return MIN_DUAL_VERTEX_COUNT if self is self.AT_LEAST_2 else 6


@dataclass(frozen=True, slots=True, init=False)
class SimpleQuadrangulation:
    """Candidate primal quadrangulation ``G`` viewed over the darts of its dual ``G*``."""

    _dual: DualPlaneGraph = field(repr=False)
    _embedding: tuple[tuple[int, ...], ...] | None = field(default=None, init=False, repr=False, compare=False)
    _faces: tuple[tuple[int, ...], ...] | None = field(default=None, init=False, repr=False, compare=False)

    def __init__(self, dual: DualPlaneGraph) -> None:
        """Pair the candidate-primal view with ``dual``."""
        if type(dual) is not DualPlaneGraph:
            raise TypeError("dual must be DualPlaneGraph")
        object.__setattr__(self, "_dual", dual)
        object.__setattr__(self, "_embedding", None)
        object.__setattr__(self, "_faces", None)

    @property
    def dual(self) -> DualPlaneGraph:
        """Return the paired candidate dual ``G*``."""
        return self._dual

    @property
    def twin(self) -> bytes:
        """Return the dart involution shared with ``G*``."""
        return self._dual.twin

    @property
    def graph_id(self) -> int:
        """Return the Graph ID shared with ``G*``."""
        return self._dual.graph_id

    @property
    def num_vertices(self) -> int:
        """Return the number of candidate-primal vertices, ``|V(G*)| + 2``."""
        return self._dual.num_vertices + 2

    @property
    def num_edges(self) -> int:
        """Return the number of candidate-primal edges, one per dual edge."""
        return len(self._dual.twin) // 2

    @property
    def num_faces(self) -> int:
        """Return the number of candidate-primal faces, one per dual vertex."""
        return self._dual.num_vertices

    @property
    def embedding(self) -> tuple[tuple[int, ...], ...]:
        """Return vertex-indexed exterior-view-CW candidate-primal rotations."""
        embedding = self._embedding
        if embedding is None:
            # Each incident 4-cycle contributes the neighbour preceding the vertex on it;
            # the vertex's dual face lists those 4-cycles in rotation order.
            faces = self.faces
            embedding = tuple(
                tuple(faces[face][faces[face].index(vertex) - 1] for face in incident_faces)
                for vertex, incident_faces in enumerate(self._dual.faces)
            )
            object.__setattr__(self, "_embedding", embedding)
        return embedding

    @property
    def faces(self) -> tuple[tuple[int, ...], ...]:
        """Return candidate-primal 4-cycles indexed by candidate-dual vertices."""
        faces = self._faces
        if faces is None:
            twin = self.twin
            faces = tuple(
                tuple(self.vertex(twin[dart]) for dart in range(base, base + 4))
                for base in range(0, len(twin), 4)
            )
            object.__setattr__(self, "_faces", faces)
        return faces

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return candidate-primal edges as endpoint-sorted pairs in lexicographic order."""
        right_faces = self._dual.right_faces
        return tuple(sorted(
            (min(right_faces[dart], right_faces[opposite]), max(right_faces[dart], right_faces[opposite]))
            for dart, opposite in enumerate(self.twin)
            if dart < opposite
        ))

    @property
    def degrees(self) -> tuple[int, ...]:
        """Return vertex-indexed candidate-primal degrees."""
        return tuple(map(self._dual.right_faces.count, range(self.num_vertices)))

    @property
    def degree_sequence(self) -> tuple[int, ...]:
        """Return candidate-primal degrees in nonincreasing order."""
        return self._dual.face_size_sequence

    @property
    def min_degree(self) -> int:
        """Return the minimum candidate-primal degree; 3 or more means ``G*`` is digon-free."""
        return min(self.degrees)

    def vertex(self, dart: int) -> int:
        """Return the source vertex of a valid dart."""
        return self._dual.right_faces[dart]

    def next_at_vertex(self, dart: int) -> int:
        """Return the next valid dart in its exterior-view-CW vertex rotation."""
        return self._dual.right_face_next(dart)

    def prev_at_vertex(self, dart: int) -> int:
        """Return the previous valid dart in its exterior-view-CW vertex rotation."""
        return self.twin[self._dual.next_at_vertex(dart)]

    def right_face_next(self, dart: int) -> int:
        """Return the next valid dart along the face on the right of a dart."""
        twin = self.twin
        return twin[self._dual.next_at_vertex(twin[dart])]

    def right_face(self, dart: int) -> int:
        """Return the face on the right of a valid dart."""
        return self._dual.vertex(self.twin[dart])

    def neighbor(self, dart: int) -> int:
        """Return the target vertex of a valid dart."""
        return self.vertex(self.twin[dart])


class PlantriError(Exception):
    """Failure while invoking plantri_sqs or decoding its fixed records."""


@dataclass(frozen=True, slots=True)
class PlantriEnumerationResult:
    """Immutable map batch with zero time-to-first-record for an empty stream."""

    graphs: tuple[DualPlaneGraph, ...]
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
        self._thread.join(5.0)
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

    minimum = MIN_DUAL_VERTEX_COUNT
    maximum = MAX_DUAL_VERTEX_COUNT
    if not minimum <= dual_vertex_count <= maximum:
        raise ValueError(f"dual_vertex_count unsupported: {dual_vertex_count}; expected [{minimum},{maximum}]")
    if type(max_count) is not int or max_count < 0:
        raise ValueError(f"max_count: expected int >= 0, got {max_count!r}")
    if timeout is not None and (
        type(timeout) not in (int, float)
        or not 0 < timeout <= threading.TIMEOUT_MAX
    ):
        raise ValueError(f"plantri_sqs: timeout must be None or 0 < timeout <= {threading.TIMEOUT_MAX}")
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
    command = [
        str(executable),
        *primal_minimum_degree._plantri_switches,
        str(dual_vertex_count + 2),
    ]
    try:
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)
    except OSError as error:
        detail = _error_detail(str(error)) or type(error).__name__
        raise PlantriError(f"plantri_sqs: failed to start executable {executable}: {detail}") from error


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
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5.0)
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


def _iter_filter_records(
    chunks: Iterator[bytes],
    dual_vertex_count: int,
) -> Iterator[tuple[bytes, bytes]]:
    twin_size = 4 * dual_vertex_count
    record_size = twin_size + dual_vertex_count + 2
    carry = b""
    record_count = 0
    for chunk in chunks:
        data = carry + chunk
        complete_count = len(data) // record_size
        complete_size = complete_count * record_size
        for start in range(0, complete_size, record_size):
            split = start + twin_size
            yield data[start:split], data[split : start + record_size]
        record_count += complete_count
        carry = data[complete_size:]
    if carry:
        raise PlantriError(f"plantri_sqs: truncated fixed record {record_count} ({len(carry)}/{record_size} bytes)")


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
    """Materialize candidate duals from one bundled FILTER run."""
    _validate_enumeration_request(dual_vertex_count, primal_minimum_degree, max_count, timeout)
    enumeration_started_at = time.perf_counter()
    time_to_first_record_s = 0.0
    duals: list[DualPlaneGraph] = []

    if max_count and dual_vertex_count >= primal_minimum_degree._minimum_nonempty_dual_vertex_count:
        with _filter_session(dual_vertex_count, primal_minimum_degree, timeout) as reader:
            records = _iter_filter_records(reader.chunks(), dual_vertex_count)
            for graph_id, (twin, profile) in enumerate(islice(records, max_count)):
                dual = DualPlaneGraph._from_filter(twin, graph_id, profile)

                if graph_id == 0:
                    time_to_first_record_s = time.perf_counter() - enumeration_started_at

                duals.append(dual)

    graphs = tuple(duals)
    total_elapsed_s = time.perf_counter() - enumeration_started_at

    return PlantriEnumerationResult(
        graphs=graphs,
        time_to_first_record_s=time_to_first_record_s,
        remaining_s=total_elapsed_s - time_to_first_record_s,
    )
