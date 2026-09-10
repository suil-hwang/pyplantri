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
        return tuple(map(len, self._dual.faces))

    @property
    def degree_sequence(self) -> tuple[int, ...]:
        """Return candidate-primal degrees in nonincreasing order."""
        return self._dual.face_size_sequence

    @property
    def min_degree(self) -> int:
        """Return the minimum candidate-primal degree; 3 or more means ``G*`` is digon-free."""
        return self.degree_sequence[-1]

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
class PlantriEnumeration:
    """Completed immutable map batch; reader resources exist only during its run."""

    graphs: tuple[DualPlaneGraph, ...]
    time_to_first_record_s: float
    remaining_s: float

    @property
    def total_s(self) -> float:
        """Return total enumeration time."""
        return self.time_to_first_record_s + self.remaining_s

    @classmethod
    def enumerate(
        cls,
        dual_vertex_count: int,
        *,
        max_count: int,
        primal_minimum_degree: PrimalMinimumDegree = PrimalMinimumDegree.AT_LEAST_2,
        timeout: float | None = None,
    ) -> PlantriEnumeration:
        """Collect a source-order prefix and return its completed immutable result."""
        cls._validate_enumeration_request(dual_vertex_count, primal_minimum_degree, max_count, timeout)
        started_at = time.perf_counter()
        time_to_first_record_s = 0.0
        graphs: list[DualPlaneGraph] = []

        if max_count and dual_vertex_count >= primal_minimum_degree._minimum_nonempty_dual_vertex_count:
            with cls._session(dual_vertex_count, primal_minimum_degree, timeout) as records:
                for graph_id, (twin, profile) in enumerate(records):
                    graph = DualPlaneGraph._from_filter(twin, graph_id, profile)
                    if graph_id == 0:
                        time_to_first_record_s = time.perf_counter() - started_at
                    graphs.append(graph)
                    if len(graphs) == max_count:
                        break

        completed_graphs = tuple(graphs)
        total_elapsed_s = time.perf_counter() - started_at
        return cls(completed_graphs, time_to_first_record_s, total_elapsed_s - time_to_first_record_s)

    @staticmethod
    def _remaining_timeout(deadline: float | None, timeout: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PlantriError(f"plantri_sqs: timed out after {timeout}s")
        return remaining

    @staticmethod
    def _publish(
        pending: queue.Queue[bytes | BaseException | None],
        cancelled: threading.Event,
        item: bytes | BaseException | None,
    ) -> None:
        """Publish one item unless cancellation is requested."""
        while not cancelled.is_set():
            try:
                pending.put(item, timeout=0.05)
                return
            except queue.Full:
                pass

    @classmethod
    def _read_stdout(
        cls,
        stream: BinaryIO,
        pending: queue.Queue[bytes | BaseException | None],
        cancelled: threading.Event,
    ) -> None:
        """Publish stdout chunks followed by one terminal item."""
        try:
            read = getattr(stream, "read1", stream.read)
            while not cancelled.is_set() and (chunk := read(64 * 1024)):
                cls._publish(pending, cancelled, chunk)
        except BaseException as error:
            cls._publish(pending, cancelled, error)
        else:
            cls._publish(pending, cancelled, None)

    @classmethod
    def _chunks(
        cls,
        pending: queue.Queue[bytes | BaseException | None],
        eof_received: threading.Event,
        deadline: float | None,
        timeout: float | None,
    ) -> Iterator[bytes]:
        """Consume queued chunks under the same deadline as process completion."""
        while True:
            try:
                chunk = pending.get(timeout=cls._remaining_timeout(deadline, timeout))
            except queue.Empty:
                raise PlantriError(f"plantri_sqs: timed out after {timeout}s") from None
            if chunk is None:
                eof_received.set()
                return
            if isinstance(chunk, BaseException):
                detail = cls._error_detail(str(chunk))
                raise PlantriError(f"plantri_sqs: failed to read binary stdout: {detail}") from chunk
            yield chunk

    @staticmethod
    def _records(chunks: Iterator[bytes], dual_vertex_count: int) -> Iterator[tuple[bytes, bytes]]:
        """Split arbitrary stdout chunks into complete fixed-size FILTER records."""
        twin_size = 4 * dual_vertex_count
        record_size = twin_size + dual_vertex_count + 2
        carry = b""
        record_count = 0
        for chunk in chunks:
            data = carry + chunk
            complete_size = len(data) // record_size * record_size
            for start in range(0, complete_size, record_size):
                split = start + twin_size
                yield data[start:split], data[split : start + record_size]
                record_count += 1
            carry = data[complete_size:]
        if carry:
            raise PlantriError(f"plantri_sqs: truncated fixed record {record_count} ({len(carry)}/{record_size} bytes)")

    @staticmethod
    def _join_reader(reader: threading.Thread) -> None:
        reader.join(5.0)
        if reader.is_alive():
            raise PlantriError("plantri_sqs: stdout reader did not stop")

    @classmethod
    def _finish(
        cls,
        process: subprocess.Popen[bytes],
        reader: threading.Thread | None,
        eof_received: bool,
        deadline: float | None,
        timeout: float | None,
    ) -> bool:
        """Finish the child, join a started reader, then close stdout in every case."""
        with ExitStack() as cleanup:
            cleanup.callback(cast(BinaryIO, process.stdout).close)
            if reader is not None:
                cleanup.callback(cls._join_reader, reader)
            if not eof_received:
                return cls._stop_process(process) not in (None, 0)
            try:
                return process.wait(timeout=cls._remaining_timeout(deadline, timeout)) != 0
            except (PlantriError, subprocess.TimeoutExpired):
                cls._stop_process(process)
                raise PlantriError(f"plantri_sqs: timed out after {timeout}s") from None

    @classmethod
    @contextmanager
    def _session(
        cls,
        dual_vertex_count: int,
        primal_minimum_degree: PrimalMinimumDegree,
        timeout: float | None,
    ) -> Generator[Iterator[tuple[bytes, bytes]], None, None]:
        """Own one reader/process lifetime, preserving failures in the consuming code."""
        cancelled, eof_received = threading.Event(), threading.Event()
        with cls._resolved_executable() as executable, tempfile.TemporaryFile() as stderr_file:
            stderr = cast(BinaryIO, stderr_file)
            process = cls._start_process(executable, dual_vertex_count, primal_minimum_degree, stderr)
            reader = None
            deadline = None
            body_error = None
            try:
                deadline = None if timeout is None else time.monotonic() + timeout
                pending: queue.Queue[bytes | BaseException | None] = queue.Queue(maxsize=2)
                worker = threading.Thread(
                    target=cls._read_stdout,
                    args=(cast(BinaryIO, process.stdout), pending, cancelled),
                    name="pyplantri-stdout",
                    daemon=True,
                )
                worker.start()
                reader = worker
                yield cls._records(cls._chunks(pending, eof_received, deadline, timeout), dual_vertex_count)
            except BaseException as error:
                body_error = error
                raise
            finally:
                cancelled.set()
                try:
                    natural_nonzero_exit = cls._finish(process, reader, eof_received.is_set(), deadline, timeout)
                    execution_error = (
                        cls._build_execution_error(process, stderr)
                        if reader is not None and natural_nonzero_exit
                        else None
                    )
                except BaseException as cleanup_error:
                    if body_error is None:
                        raise
                    detail = cls._error_detail(str(cleanup_error), limit=240)
                    body_error.add_note(f"pyplantri: process cleanup failed: {detail}")
                else:
                    if execution_error is not None:
                        raise execution_error from body_error

    @staticmethod
    def _error_detail(text: str | bytes, *, limit: int = 4000) -> str:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        normalized = " ".join(text.split())
        return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."

    @staticmethod
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

    @staticmethod
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

    @classmethod
    def _start_process(
        cls,
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
            detail = cls._error_detail(str(error)) or type(error).__name__
            raise PlantriError(f"plantri_sqs: failed to start executable {executable}: {detail}") from error

    @classmethod
    def _build_execution_error(
        cls,
        process: subprocess.Popen[bytes],
        stderr_file: BinaryIO,
    ) -> PlantriError:
        stderr_file.flush()
        stderr_file.seek(0)
        detail = cls._error_detail(stderr_file.read()) or "no output"
        return PlantriError(f"plantri_sqs: execution failed (exit {process.returncode}); {detail}")

    @staticmethod
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
