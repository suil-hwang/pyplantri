# src/pyplantri/plantri_interface.py
from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from importlib.resources import as_file, files
from itertools import chain, islice
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, cast

from .plane_graph import MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT, QuarticPlaneMap
from .types import Embedding, QuadrangulationDualClass

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext
    from multiprocessing.pool import Pool


# Public request and capability surface.
PlantriOutput = Literal[
    "planar_code",
    "ascii",
    "graph6",
    "sparse6",
    "edge_code",
    "none",
]
# Public bundled SQS bounds; n=N-2 for a quadrangulation with N primal vertices.
MIN_DUAL_VERTEX_COUNT = 3
BUNDLED_PLANTRI_MAX_VERTEX_COUNT = 64
BUNDLED_MAX_DUAL_VERTEX_COUNT = BUNDLED_PLANTRI_MAX_VERTEX_COUNT - 2


# Command-line switch tables.
_OUTPUT_SWITCH_BY_FORMAT: dict[PlantriOutput, str | None] = {
    "planar_code": None,
    "ascii": "-a",
    "graph6": "-g",
    "sparse6": "-s",
    "edge_code": "-E",
    "none": "-u",
}
_TEXT_OUTPUT_FORMATS = frozenset(("ascii", "graph6", "sparse6"))
_OUTPUT_SWITCH_CHARS = frozenset("agsETu")
_C_INT_MAX = 2_147_483_647


# Process and stream polling budgets.
_IO_CHUNK_SIZE = 1 << 16
_FILE_POLL_INTERVAL_S = 0.01
_PROCESS_TERMINATE_TIMEOUT_S = 5.0
_VERSION_PROBE_TIMEOUT_S = 5.0


# planar_code record encoding.
# Translate separators to 255 and labels 1..255 to 0..254 in one C-level pass.
_ZERO_BASE_TRANSLATION = bytes((255, *range(255)))
_TRANSLATED_ROW_SEPARATOR = b"\xff"
# One-byte planar_code supports 1..255 labels; bundled input is smaller.
_ONE_BYTE_PLANAR_CODE_MAX_N = 255


class PlantriError(Exception):
    """Base class for running plantri and decoding its output."""


class PlantriExecutableNotFoundError(PlantriError, FileNotFoundError):
    """Plantri executable could not be found."""


class PlanarCodeError(PlantriError, ValueError):
    """Malformed or unsupported planar_code input."""


class PlantriTimeoutError(PlantriError, TimeoutError):
    """Plantri exceeded a caller-supplied wall-clock deadline."""


def iter_planar_code(
    stream: BinaryIO,
    *,
    expected_vertex_count: int | None = None,
    expected_edge_count: int | None = None,
    chunk_size: int = 65_536,
) -> Iterator[Embedding]:
    """Decode headerless one-byte planar_code, with an optional fixed-size fast path."""
    _validate_planar_code_expectations(expected_vertex_count, expected_edge_count)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")

    yield from _decode_planar_code_records(
        iter(lambda: stream.read(chunk_size), b""),
        expected_vertex_count=expected_vertex_count,
        expected_edge_count=expected_edge_count,
    )


def _validate_planar_code_expectations(
    expected_vertex_count: int | None,
    expected_edge_count: int | None,
) -> None:
    """Validate the paired planar_code record-size expectations."""
    if expected_vertex_count is not None and (
        type(expected_vertex_count) is not int
        or not 1 <= expected_vertex_count <= _ONE_BYTE_PLANAR_CODE_MAX_N
    ):
        expected = f"None or int in [1,{_ONE_BYTE_PLANAR_CODE_MAX_N}]"
        message = f"expected_vertex_count={expected_vertex_count!r}; expected {expected}"
        raise ValueError(message)
    if expected_edge_count is not None and (
        type(expected_edge_count) is not int or expected_edge_count < 0
    ):
        message = f"expected_edge_count={expected_edge_count!r}; expected None or int >= 0"
        raise ValueError(message)
    if expected_edge_count is not None and expected_vertex_count is None:
        raise ValueError("expected_edge_count requires expected_vertex_count")


def _decode_planar_code_records(
    chunks: Iterable[bytes],
    *,
    expected_vertex_count: int | None,
    expected_edge_count: int | None,
) -> Iterator[Embedding]:
    """Dispatch to the generic or fixed-layout headerless decoder."""
    if expected_vertex_count is not None and expected_edge_count is not None:
        yield from _decode_fixed_planar_code_chunks(
            chunks,
            vertex_count=expected_vertex_count,
            edge_count=expected_edge_count,
        )
        return
    yield from _decode_planar_code_chunks(
        chunks,
        expected_vertex_count=expected_vertex_count,
    )


def _decode_planar_code_chunks(
    chunks: Iterable[bytes],
    *,
    expected_vertex_count: int | None,
) -> Iterator[Embedding]:
    """Decode complete records across arbitrary binary chunk boundaries."""

    record_index = 0
    vertex_count: int | None = None
    adjacency_rows: list[tuple[int, ...]] = []
    neighbors: list[int] = []

    for chunk in chunks:
        for value in chunk:
            if vertex_count is None:
                if value == 0:
                    message = f"record {record_index}: extended planar_code is unsupported"
                    raise PlanarCodeError(message)
                if expected_vertex_count is not None and value != expected_vertex_count:
                    message = f"record {record_index}: vertex count {value} != expected {expected_vertex_count}"
                    raise PlanarCodeError(message)
                vertex_count = value
            elif value:
                if value > vertex_count:
                    vertex = len(adjacency_rows)
                    message = f"record {record_index}, vertex {vertex}: neighbor {value} outside [1, {vertex_count}]"
                    raise PlanarCodeError(message)
                neighbors.append(value - 1)
            else:
                adjacency_rows.append(tuple(neighbors))
                neighbors.clear()
                if len(adjacency_rows) == vertex_count:
                    yield tuple(adjacency_rows)
                    record_index += 1
                    vertex_count = None
                    adjacency_rows.clear()

    if vertex_count is not None:
        vertex = len(adjacency_rows)
        raise PlanarCodeError(f"record {record_index}: truncated at vertex {vertex}")


def _decode_fixed_planar_code_chunks(
    chunks: Iterable[bytes],
    *,
    vertex_count: int,
    edge_count: int,
) -> Iterator[Embedding]:
    """Decode fixed-size headerless records after the graph class fixes |V| and |E|."""
    record_size = 1 + vertex_count + 2 * edge_count
    carry = b""
    record_index = 0

    for chunk in chunks:
        data = carry + chunk
        complete_size = len(data) - len(data) % record_size
        for start in range(0, complete_size, record_size):
            record = data[start : start + record_size]
            if record[0] != vertex_count:
                actual = record[0]
                message = f"record {record_index}: vertex count {actual} != expected {vertex_count}"
                raise PlanarCodeError(message)

            body = record[1:]
            rows = body.translate(_ZERO_BASE_TRANSLATION).split(
                _TRANSLATED_ROW_SEPARATOR
            )
            if len(rows) != vertex_count + 1 or rows[-1]:
                message = f"record {record_index}: invalid adjacency separators"
                raise PlanarCodeError(message)
            rows.pop()
            if body and max(body) > vertex_count:
                for vertex, row in enumerate(body.split(b"\0")):
                    if row and (neighbor := max(row)) > vertex_count:
                        message = f"record {record_index}, vertex {vertex}: neighbor {neighbor} outside [1, {vertex_count}]"
                        raise PlanarCodeError(message)
            yield tuple(map(tuple, rows))
            record_index += 1
        carry = data[complete_size:]

    if carry:
        message = f"record {record_index}: truncated fixed-size record ({len(carry)}/{record_size} bytes)"
        raise PlanarCodeError(message)


def _summarize_process_text(text: str | bytes, *, limit: int = 400) -> str:
    """Decode and collapse process output into a bounded single-line excerpt."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def _switch_chars(switches: Sequence[str]) -> set[str]:
    """Return every flag character selected by a switch sequence."""
    return {char for option in switches for char in option[1:]}


def _normalize_switches(switches: Sequence[str]) -> tuple[str, ...]:
    """Validate graph-selection switches and reject positional/output tokens."""
    if isinstance(switches, (str, bytes)) or not isinstance(switches, Sequence):
        actual = type(switches).__name__
        raise ValueError(f"plantri: switches must be a sequence, got {actual}")
    normalized = tuple(switches)
    for index, option in enumerate(normalized):
        if type(option) is not str:
            actual = type(option).__name__
            raise ValueError(f"plantri: switches[{index}] must be str, got {actual}")
    positional = [
        option for option in normalized if option == "-" or not option.startswith("-")
    ]
    if positional:
        message = f"plantri: switches contain positional values: {positional}"
        raise ValueError(message)
    selected = _switch_chars(normalized)
    if "T" in selected:
        raise ValueError("plantri: -T output is unsupported")
    if selected_output_flags := selected & _OUTPUT_SWITCH_CHARS:
        formatted = ", ".join(f"-{flag}" for flag in sorted(selected_output_flags))
        message = f"plantri: output switches must use output_format; found {formatted}"
        raise ValueError(message)
    if "h" in selected:
        raise ValueError("plantri: -h is controlled by the selected output API")
    return normalized


def _validate_split(split: tuple[int, int] | None) -> None:
    """Validate an optional plantri residue/modulus partition."""
    if split is None:
        return
    if (
        type(split) is not tuple
        or len(split) != 2
        or type(split[0]) is not int
        or type(split[1]) is not int
        or split[1] <= 0
        or split[1] > _C_INT_MAX
        or not 0 <= split[0] < split[1]
    ):
        expected = f"0 <= residue < modulus <= {_C_INT_MAX}"
        raise ValueError(f"plantri: invalid split {split!r}; expected {expected}")


def _validate_timeout(timeout: float | None) -> None:
    """Require a finite positive timeout when one is supplied."""
    if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        message = f"plantri: timeout must be None or finite positive; got {timeout!r}"
        raise ValueError(message)


def _cleanup_after_error(
    error: BaseException,
    label: str,
    cleanup: Callable[[], object],
) -> None:
    """Preserve a primary error while making explicit close failures observable."""
    try:
        cleanup()
    except BaseException as cleanup_error:
        if isinstance(error, GeneratorExit):
            raise
        detail = " ".join(str(cleanup_error).split()) or type(cleanup_error).__name__
        error.add_note(f"pyplantri: {label}: {detail[:240]}")


@contextmanager
def _closing_on_error(label: str, close: Callable[[], object]) -> Generator[None, None, None]:
    """Close one owned resource; a close failure never replaces the primary error."""
    try:
        yield
    except BaseException as error:
        _cleanup_after_error(error, label, close)
        raise
    close()


@contextmanager
def _translating_launch_errors(executable: Path) -> Generator[None, None, None]:
    """Normalize the two ways one executable can fail to launch."""
    try:
        yield
    except FileNotFoundError as e:
        message = f"plantri: executable not found {executable}"
        raise PlantriExecutableNotFoundError(message) from e
    except OSError as e:
        detail = _summarize_process_text(str(e))
        message = f"plantri: executable is not runnable {executable}: {detail}"
        raise PlantriError(message) from e


@dataclass(frozen=True, slots=True)
class _StreamDeadline:
    """One streaming wall-clock budget, or an unbounded one when no timeout is set."""

    deadline: float | None
    timeout: float | None

    @classmethod
    def from_timeout(cls, timeout: float | None) -> _StreamDeadline:
        """Start one budget from an already validated optional timeout."""
        return cls(None if timeout is None else time.monotonic() + timeout, timeout)

    def _expired(self) -> PlantriTimeoutError:
        """Build the shared expiry error naming the caller's original budget."""
        return PlantriTimeoutError(f"plantri: timed out after {self.timeout}s")

    def check(self) -> None:
        """Raise when an active streaming process has reached the deadline."""
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise self._expired()

    def wait(self) -> None:
        """Sleep for one polling interval without crossing the deadline."""
        if self.deadline is None:
            time.sleep(_FILE_POLL_INTERVAL_S)
            return
        # One clock reading: re-checking would let the sleep length go negative.
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise self._expired()
        time.sleep(min(_FILE_POLL_INTERVAL_S, remaining))


def _validate_plantri_construction(
    *,
    bundled: bool,
    max_vertex_count: int | None,
    expected_version: str | None,
) -> None:
    """Validate wrapper construction arguments before any resource is acquired."""
    if max_vertex_count is not None and (
        type(max_vertex_count) is not int or max_vertex_count <= 0
    ):
        message = f"plantri: max_vertex_count must be positive int; got {max_vertex_count!r}"
        raise ValueError(message)
    if expected_version is not None and (
        type(expected_version) is not str
        or not re.fullmatch(r"\d+\.\d+", expected_version)
    ):
        raise ValueError(f"plantri: invalid expected_version={expected_version!r}")
    if bundled and max_vertex_count not in (None, BUNDLED_PLANTRI_MAX_VERTEX_COUNT):
        maximum = BUNDLED_PLANTRI_MAX_VERTEX_COUNT
        message = f"plantri: bundled max_vertex_count is fixed at {maximum}"
        raise ValueError(message)


def _read_planar_code_chunk(output_file: BinaryIO) -> bytes:
    """Read one planar-code chunk and normalize storage failures."""
    try:
        return output_file.read(_IO_CHUNK_SIZE)
    except OSError as e:
        detail = _summarize_process_text(str(e))
        message = f"plantri: failed to read planar_code output: {detail}"
        raise PlantriError(message) from e


def _open_output_when_created(
    output_path: Path,
    process: subprocess.Popen[bytes],
    deadline: _StreamDeadline,
) -> BinaryIO:
    """Wait for a live writer to create its output file, then open it for reading."""
    while True:
        writer_exited = process.poll() is not None
        if not writer_exited:
            deadline.check()
        try:
            return cast(BinaryIO, output_path.open("rb"))
        except FileNotFoundError:
            if writer_exited:
                raise PlantriError("plantri: planar_code output was not created")
        except PermissionError as e:
            if writer_exited:
                detail = _summarize_process_text(str(e))
                message = f"plantri: failed to open planar_code output: {detail}"
                raise PlantriError(message) from e
        except OSError as e:
            detail = _summarize_process_text(str(e))
            message = f"plantri: failed to open planar_code output: {detail}"
            raise PlantriError(message) from e
        if not writer_exited:
            deadline.wait()


def _iter_live_output_chunks(
    output_file: BinaryIO,
    process: subprocess.Popen[bytes],
    deadline: _StreamDeadline,
) -> Iterator[bytes]:
    """Iterate chunks while the writer may still append, then drain the remainder."""

    def read_chunk() -> bytes:
        """Read available bytes or wait while the writer is active."""
        while True:
            if process.poll() is None:
                deadline.check()
            if chunk := _read_planar_code_chunk(output_file):
                return chunk
            if process.poll() is not None:
                return _read_planar_code_chunk(output_file)
            deadline.wait()

    return iter(read_chunk, b"")


def _iter_lines_guarding_read_errors(stream: BinaryIO) -> Iterator[bytes]:
    """Yield raw lines, translating read failures but never consumer failures.

    The manual ``next()`` is load-bearing: a plain ``for`` under one ``try`` would
    also translate an OSError raised by whoever consumes a yielded record.
    """
    lines = iter(stream)
    while True:
        try:
            raw_line = next(lines)
        except StopIteration:
            return
        except OSError as e:
            detail = _summarize_process_text(str(e))
            message = f"plantri: failed to read text output: {detail}"
            raise PlantriError(message) from e
        yield raw_line


class Plantri:
    """Wrapper for the plantri executable."""

    def __init__(
        self,
        executable: str | Path | None = None,
        *,
        max_vertex_count: int | None = None,
        expected_version: str | None = None,
    ) -> None:
        """Resolve one binary and its caller-declared compile-time capacity."""
        bundled = executable is None
        _validate_plantri_construction(
            bundled=bundled,
            max_vertex_count=max_vertex_count,
            expected_version=expected_version,
        )
        self._resources = ExitStack()
        self._closed = False
        self._version: str | None = None
        self._executable_sha256: str | None = None
        self._expected_version = (expected_version or "5.5") if bundled else expected_version
        self.max_vertex_count = (
            BUNDLED_PLANTRI_MAX_VERTEX_COUNT if bundled else max_vertex_count
        )
        # Only the resource stack needs unwinding, so the guard starts at its first entry.
        try:
            self.executable = self._resolve_executable(executable)
        except BaseException as error:
            _cleanup_after_error(error, "resource cleanup failed", self._resources.close)
            raise

    def _resolve_executable(self, executable: str | Path | None) -> Path:
        """Extract or locate the owned binary, and make it runnable."""
        if executable is None:
            exe_name = "plantri.exe" if os.name == "nt" else "plantri"
            resource = files("pyplantri").joinpath("bin", exe_name)
            candidate = self._resources.enter_context(as_file(resource))
        else:
            candidate = Path(executable).expanduser()
        resolved = candidate.resolve()
        if not resolved.is_file():
            message = f"plantri: executable not found {resolved}"
            raise PlantriExecutableNotFoundError(message)
        if (
            executable is None
            and os.name == "posix"
            and not os.access(resolved, os.X_OK)
        ):
            resolved.chmod(resolved.stat().st_mode | stat.S_IXUSR)
        return resolved

    def __enter__(self) -> Plantri:
        """Return this wrapper while its executable resource is available."""
        self._require_open()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        """Release executable-resource ownership."""
        self.close()

    def close(self) -> None:
        """Release any temporary package-resource extraction."""
        if not self._closed:
            self._closed = True
            self._resources.close()

    def _require_open(self) -> None:
        """Reject operations after executable-resource ownership ends."""
        if getattr(self, "_closed", False):
            raise RuntimeError("plantri: closed")

    @property
    def version(self) -> str:
        """Return the binary-reported major.minor plantri version."""
        self._require_open()
        if self._version is None:
            result = self._run_checked(
                [str(self.executable), "--help"], timeout=_VERSION_PROBE_TIMEOUT_S
            )
            match = re.search(
                rb"Plantri version (\d+\.\d+)", result.stderr + b"\n" + result.stdout
            )
            if match is None:
                raise PlantriError("plantri: version probe failed")
            version = match.group(1).decode("ascii")
            if self._expected_version is not None and version != self._expected_version:
                expected = self._expected_version
                raise PlantriError(f"plantri: version {version}!={expected}")
            self._version = version
        return self._version

    @property
    def executable_sha256(self) -> str:
        """Return the exact executable-byte SHA-256 digest."""
        self._require_open()
        if self._executable_sha256 is None:
            self._executable_sha256 = hashlib.sha256(
                self.executable.read_bytes()
            ).hexdigest()
        return self._executable_sha256

    def run(
        self,
        n_vertices: int,
        switches: Sequence[str] = (),
        output_format: PlantriOutput = "planar_code",
        *,
        split: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> bytes:
        """Materialize bounded output; planar_code retains its standard header."""
        _validate_timeout(timeout)
        if output_format == "none":
            cmd = self._build_command(
                n_vertices,
                switches=switches,
                output_format=output_format,
                split=split,
            )
            self._run_checked(cmd, timeout=timeout)
            return b""

        with tempfile.TemporaryDirectory(prefix="pyplantri-") as temp_dir:
            output_path = Path(temp_dir) / "output"
            cmd = self._build_command(
                n_vertices,
                switches=switches,
                output_format=output_format,
                split=split,
                output_path=output_path,
            )
            self._run_checked(cmd, timeout=timeout)
            try:
                return output_path.read_bytes()
            except FileNotFoundError as e:
                message = f"plantri: {output_format} output was not created"
                raise PlantriError(message) from e
            except OSError as e:
                detail = _summarize_process_text(str(e))
                message = f"plantri: failed to read {output_format} output: {detail}"
                raise PlantriError(message) from e

    def count(
        self,
        n_vertices: int,
        switches: Sequence[str] = (),
        timeout: float | None = None,
        *,
        split: tuple[int, int] | None = None,
    ) -> int:
        """Count generated objects selected by switches and one optional split."""
        _validate_timeout(timeout)
        cmd = self._build_command(
            n_vertices,
            switches=switches,
            output_format="none",
            split=split,
        )
        result = self._run_checked(cmd, timeout=timeout)

        stderr_text = result.stderr.decode("utf-8", errors="replace")
        for line in reversed(stderr_text.splitlines()):
            fields = line.split()
            if (
                fields
                and fields[0].isdecimal()
                and any(
                    field.lower().rstrip(";") == "generated" for field in fields[1:]
                )
            ):
                return int(fields[0])

        detail = _summarize_process_text(result.stderr) or "no output"
        raise PlantriError(f"plantri: count parse failed: {detail}")

    def iter_planar_code(
        self,
        n_vertices: int,
        switches: Sequence[str] = (),
        *,
        expected_vertex_count: int | None = None,
        expected_edge_count: int | None = None,
        split: tuple[int, int] | None = None,
        timeout: float | None = None,
    ) -> Iterator[Embedding]:
        """Stream headerless records; timeout bounds the active writer process."""
        _validate_planar_code_expectations(expected_vertex_count, expected_edge_count)
        _validate_timeout(timeout)
        with (
            tempfile.TemporaryDirectory(prefix="pyplantri-") as temp_dir,
            tempfile.TemporaryFile() as stderr_file,
        ):
            output_path = Path(temp_dir) / "output.planar_code"
            cmd = self._build_command(
                n_vertices,
                switches=switches,
                output_format="planar_code",
                headerless=True,
                split=split,
                output_path=output_path,
            )
            process = self._start_stream_process(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=cast(BinaryIO, stderr_file),
            )
            deadline = _StreamDeadline.from_timeout(timeout)

            # Opening stays inside the guard: a failure there still owns the writer.
            stream_exhausted = False
            try:
                with _open_output_when_created(
                    output_path, process, deadline
                ) as output_file:
                    for embedding in _decode_planar_code_records(
                        _iter_live_output_chunks(output_file, process, deadline),
                        expected_vertex_count=expected_vertex_count,
                        expected_edge_count=expected_edge_count,
                    ):
                        if process.poll() is None:
                            deadline.check()
                        yield embedding
                    stream_exhausted = True
            finally:
                self._finalize_stream_process(
                    process,
                    cast(BinaryIO, stderr_file),
                    stream_exhausted=stream_exhausted,
                )

    def iter_text_records(
        self,
        n_vertices: int,
        switches: Sequence[str] = (),
        output_format: Literal["ascii", "graph6", "sparse6"] = "ascii",
        *,
        split: tuple[int, int] | None = None,
    ) -> Iterator[bytes]:
        """Stream non-empty text records without altering payload whitespace."""
        if output_format not in _TEXT_OUTPUT_FORMATS:
            message = f"plantri: line output requires ascii/graph6/sparse6; got {output_format!r}"
            raise ValueError(message)
        cmd = self._build_command(
            n_vertices,
            switches=switches,
            output_format=output_format,
            split=split,
        )
        yield from self._iter_text_records(cmd)

    def _build_command(
        self,
        n_vertices: int,
        *,
        switches: Sequence[str],
        output_format: PlantriOutput,
        headerless: bool = False,
        split: tuple[int, int] | None = None,
        output_path: Path | None = None,
    ) -> list[str]:
        """Build a command whose output format and positional grammar are explicit."""
        self._require_open()
        if output_format not in _OUTPUT_SWITCH_BY_FORMAT:
            raise ValueError(f"plantri: unsupported output_format {output_format!r}")
        if type(n_vertices) is not int or n_vertices <= 0:
            message = f"plantri: n_vertices must be a positive int; got {n_vertices!r}"
            raise ValueError(message)
        max_vertex_count = getattr(self, "max_vertex_count", None)
        if max_vertex_count is not None and n_vertices > max_vertex_count:
            message = f"plantri: n_vertices {n_vertices}>{max_vertex_count} declared maximum"
            raise ValueError(message)
        normalized_switches = _normalize_switches(switches)
        _validate_split(split)
        if output_path is not None and output_format == "none":
            message = "plantri: output_path is incompatible with output_format='none'"
            raise ValueError(message)

        if getattr(self, "_expected_version", None) is not None:
            # A declared binary contract must be checked before generation.
            _ = self.version
        self._reject_broken_quadrangulation_split(normalized_switches, split)

        cmd = [str(self.executable), *normalized_switches]
        if output_switch := _OUTPUT_SWITCH_BY_FORMAT[output_format]:
            cmd.append(output_switch)
        if headerless:
            cmd.append("-h")
        cmd.append(str(n_vertices))
        if split is not None:
            cmd.append(f"{split[0]}/{split[1]}")
        if output_path is not None:
            cmd.append(str(output_path))
        return cmd

    def _reject_broken_quadrangulation_split(
        self,
        switches: Sequence[str],
        split: tuple[int, int] | None,
    ) -> None:
        """Reject plantri 5.2's duplicate-producing -q and -pb splits."""
        selected = _switch_chars(switches)
        if (
            split is not None
            and ("q" in selected or {"p", "b"} <= selected)
            and self.version == "5.2"
        ):
            raise PlantriError("plantri: version 5.2 has a broken -q/-pb split")

    def _run_checked(
        self,
        cmd: list[str],
        *,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one bounded command without a shell; the caller owns validation."""
        # TimeoutExpired is a SubprocessError, so it bypasses the launch ladder.
        try:
            with _translating_launch_errors(self.executable):
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired as e:
            raise PlantriTimeoutError(f"plantri: timed out after {timeout}s") from e
        if result.returncode != 0:
            detail = _summarize_process_text(result.stderr) or _summarize_process_text(result.stdout) or "no output"
            message = f"plantri: execution failed (exit {result.returncode}); {detail}"
            raise PlantriError(message)
        return result

    def _start_stream_process(
        self,
        cmd: list[str],
        *,
        stdout: int,
        stderr: BinaryIO,
    ) -> subprocess.Popen[bytes]:
        """Start one streaming process and normalize executable failures."""
        with _translating_launch_errors(self.executable):
            return subprocess.Popen(cmd, stdout=stdout, stderr=stderr)

    def _iter_text_records(
        self,
        cmd: list[str],
    ) -> Iterator[bytes]:
        """Yield newline-delimited text records and own the process lifecycle."""
        # Spool stderr to disk so long streams cannot deadlock on a full pipe.
        with tempfile.TemporaryFile() as stderr_file:
            process = self._start_stream_process(
                cmd,
                stdout=subprocess.PIPE,
                stderr=cast(BinaryIO, stderr_file),
            )

            with cast(BinaryIO, process.stdout) as stdout:
                stream_exhausted = False
                try:
                    for raw_line in _iter_lines_guarding_read_errors(stdout):
                        if line := raw_line.rstrip(b"\r\n"):
                            yield line
                    stream_exhausted = True
                finally:
                    self._finalize_stream_process(
                        process,
                        cast(BinaryIO, stderr_file),
                        stream_exhausted=stream_exhausted,
                    )

    @staticmethod
    def _finalize_stream_process(
        process: subprocess.Popen[bytes],
        stderr_file: BinaryIO,
        *,
        stream_exhausted: bool,
    ) -> None:
        """Check a natural exit or stop an intentionally shortened stream."""
        return_code = process.poll()
        if stream_exhausted or return_code is not None:
            return_code = process.wait()
            if return_code != 0:
                stderr_file.seek(0)
                stderr_excerpt = (
                    _summarize_process_text(stderr_file.read(), limit=4000)
                    or "no output"
                )
                message = f"plantri: execution failed (exit {return_code}); {stderr_excerpt}"
                raise PlantriError(message) from None
            return

        process.terminate()
        try:
            process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


# Enumeration: plantri primal embeddings converted to proved quartic plane maps.
_MAX_AUTO_WORKERS = 4
_DEFAULT_POOL_CHUNKSIZE = 64


@dataclass(frozen=True, slots=True)
class PlantriEnumerationResult:
    """Immutable map batch with zero time-to-first for an empty stream."""

    graphs: tuple[QuarticPlaneMap, ...]
    time_to_first_embedding_s: float
    remaining_s: float

    @property
    def total_s(self) -> float:
        """Return total enumeration time."""
        return self.time_to_first_embedding_s + self.remaining_s


@dataclass(frozen=True, slots=True)
class PlantriProvenance:
    """Identity of one binary and quadrangulation graph-ID namespace."""

    version: str
    executable_sha256: str
    declared_max_vertex_count: int
    primal_vertex_count: int
    switches: tuple[str, ...]
    split: tuple[int, int] | None
    mirror_images_identified: bool = True


@dataclass(slots=True)
class _EnumerationProgress:
    """Mutable timing state for the eager collector."""

    started_at: float
    time_to_first_embedding_s: float = 0.0


# Pool tasks stay module-level: spawn resolves them by qualified name.
def _build_quartic_multigraph_task(
    item: tuple[int, Embedding],
) -> QuarticPlaneMap:
    """Build the dual of one checked arbitrary simple quadrangulation."""
    graph_id, primal_embedding = item
    return QuarticPlaneMap._from_primal_rotation_system(
        primal_embedding,
        graph_id,
        require_simple_dual=False,
    )


def _build_simple_quartic_task(item: tuple[int, Embedding]) -> QuarticPlaneMap:
    """Build one checked minimum-degree-three quadrangulation dual."""
    graph_id, primal_embedding = item
    return QuarticPlaneMap._from_primal_rotation_system(
        primal_embedding,
        graph_id,
        require_simple_dual=True,
    )


def _available_cpu_count() -> int:
    """Return the process-visible CPU count, respecting POSIX affinity."""
    # Probed by attribute: sched_getaffinity is absent on both Windows and macOS.
    if hasattr(os, "sched_getaffinity"):
        with suppress(OSError):
            return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def _validate_processing_controls(
    num_workers: int | None,
    pool_chunksize: int | None,
) -> None:
    """Validate multiprocessing controls before resolving plantri."""
    for name, value in (
        ("num_workers", num_workers),
        ("pool_chunksize", pool_chunksize),
    ):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name}: expected int > 0 or None, got {value!r}")


def _resolve_worker_count(
    num_workers: int | None,
    *,
    max_count: int | None,
    pool_chunksize: int,
) -> int:
    """Resolve effective workers, capped by the chunks a bounded run can fill."""
    worker_count = (
        num_workers
        if num_workers is not None
        else min(_available_cpu_count(), _MAX_AUTO_WORKERS)
    )
    if max_count is None:
        return worker_count
    return min(worker_count, (max_count + pool_chunksize - 1) // pool_chunksize)


def _has_importable_main() -> bool:
    """Report whether __main__ is a real file a spawned child can re-import."""
    raw_main_path = getattr(sys.modules.get("__main__"), "__file__", None)
    if not isinstance(raw_main_path, str) or not raw_main_path:
        return False
    main_path = Path(raw_main_path)
    # A pseudo-path such as <stdin> is unimportable even where the name is legal.
    if main_path.name.startswith("<") and main_path.name.endswith(">"):
        return False
    return main_path.is_file()


def _resolve_pool_context(
    start_method: str | None,
    *,
    controls_are_explicit: bool,
) -> BaseContext | None:
    """Return a usable pool context, or None when spawn has no importable __main__."""
    mp_context = multiprocessing.get_context(start_method)
    method = mp_context.get_start_method()
    if method not in {"spawn", "forkserver"} or _has_importable_main():
        return mp_context
    if controls_are_explicit:
        raise RuntimeError(f"{method} requires an importable __main__")
    # stacklevel 3 keeps the caller-facing attribution now that the warn sits
    # one frame below the generator it used to live in.
    warnings.warn(
        f"{method} requires importable __main__; using sequential",
        RuntimeWarning,
        stacklevel=3,
    )
    return None


def _imap_with_pool(
    pool: Pool,
    build_plane_map: Callable[[tuple[int, Embedding]], QuarticPlaneMap],
    items: Iterable[tuple[int, Embedding]],
    chunksize: int,
) -> Generator[QuarticPlaneMap, None, None]:
    """Stream one pool's results, terminating it on any early exit."""
    try:
        yield from pool.imap(build_plane_map, items, chunksize)
    except BaseException as error:
        _cleanup_after_error(error, "pool termination failed", pool.terminate)
        _cleanup_after_error(error, "pool join failed", pool.join)
        raise
    else:
        # _closing_on_error runs its callback on the success path too, so this
        # joins whether or not close() raises; a second join() would be a no-op.
        with _closing_on_error("pool join failed", pool.join):
            pool.close()


def _close_enumerator_after(
    stream: Generator[QuarticPlaneMap, None, None],
    enumerator: QuadrangulationEnumerator,
) -> Generator[QuarticPlaneMap, None, None]:
    """Close an internally created enumerator with its lazy stream."""
    with _closing_on_error("enumerator cleanup failed", enumerator.close):
        yield from stream


class QuadrangulationEnumerator:
    """Enumerate supported simple quadrangulations by plane-dual family.

    ``iter_primal_embeddings()`` yields zero-based exterior-view-CW primal rotation
    systems. A ``dual_vertex_count`` of ``k`` corresponds to ``k + 2`` primal
    vertices.

    - ``QUARTIC_MULTIGRAPH``: ``-q -c2 -m2``
    - ``SIMPLE_QUARTIC``: ``-q -c2``
    """

    _PLANTRI_SWITCHES_BY_DUAL_CLASS: dict[QuadrangulationDualClass, tuple[str, ...]] = {
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: ("-q", "-c2", "-m2"),
        QuadrangulationDualClass.SIMPLE_QUARTIC: ("-q", "-c2"),
    }
    _MIN_NONEMPTY_DUAL_VERTEX_COUNT_BY_CLASS: dict[QuadrangulationDualClass, int] = {
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: MIN_DUAL_VERTEX_COUNT,
        QuadrangulationDualClass.SIMPLE_QUARTIC: 6,
    }

    def __init__(self, plantri: Plantri | None = None) -> None:
        """Initialize the SQS enumerator."""
        # Delay executable resolution so known-empty requests do not require plantri.
        self._plantri = plantri
        self._owns_plantri = plantri is None
        self._closed = False
        declared_max_vertex_count = (
            BUNDLED_PLANTRI_MAX_VERTEX_COUNT
            if plantri is None
            else plantri.max_vertex_count
        )
        if declared_max_vertex_count is None:
            raise ValueError("plantri: custom binary requires max_vertex_count")
        if declared_max_vertex_count < MIN_DUAL_VERTEX_COUNT + 2:
            maximum = declared_max_vertex_count
            message = f"plantri: max_vertex_count {maximum} cannot generate a supported quadrangulation"
            raise ValueError(message)
        # Stored only once narrowed, so every reader sees the checked capacity as int.
        self._max_vertex_count: int = declared_max_vertex_count

    def __enter__(self) -> QuadrangulationEnumerator:
        """Return this enumerator while its internal resource is owned."""
        self._require_open()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        """Close any internally created plantri wrapper."""
        self.close()

    def close(self) -> None:
        """Close an internally owned plantri resource."""
        if not self._closed:
            self._closed = True
            if self._owns_plantri and self._plantri is not None:
                try:
                    self._plantri.close()
                finally:
                    self._plantri = None

    def _require_open(self) -> None:
        """Reject operations after this enumerator is closed."""
        if self._closed:
            raise RuntimeError("plantri: enumerator closed")

    @property
    def max_dual_vertex_count(self) -> int:
        """Return the joint plantri-capability and byte-encoding bound."""
        return min(self._max_vertex_count - 2, MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT)

    def iter_primal_embeddings(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass
        | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
        timeout: float | None = None,
    ) -> Generator[Embedding, None, None]:
        """Yield zero-based exterior-view-CW simple primal embeddings."""
        _validate_timeout(timeout)
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        resolved_dual_class = QuadrangulationDualClass(dual_class)
        minimum = self._MIN_NONEMPTY_DUAL_VERTEX_COUNT_BY_CLASS[resolved_dual_class]
        if dual_vertex_count < minimum:
            return
        # Euler's formula and quadrilateral faces give V_primal = V_dual + 2.
        primal_vertex_count = dual_vertex_count + 2
        yield from self._get_plantri().iter_planar_code(
            primal_vertex_count,
            self._PLANTRI_SWITCHES_BY_DUAL_CLASS[resolved_dual_class],
            expected_vertex_count=primal_vertex_count,
            expected_edge_count=2 * primal_vertex_count - 4,
            timeout=timeout,
        )

    def count(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass
        | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
        timeout: float | None = None,
        split: tuple[int, int] | None = None,
    ) -> int:
        """Count plane-map isomorphism classes, identifying mirror images."""
        _validate_timeout(timeout)
        _validate_split(split)
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        resolved_dual_class = QuadrangulationDualClass(dual_class)
        minimum = self._MIN_NONEMPTY_DUAL_VERTEX_COUNT_BY_CLASS[resolved_dual_class]
        if dual_vertex_count < minimum:
            return 0
        primal_vertex_count = dual_vertex_count + 2
        return self._get_plantri().count(
            primal_vertex_count,
            switches=self._PLANTRI_SWITCHES_BY_DUAL_CLASS[resolved_dual_class],
            timeout=timeout,
            split=split,
        )

    def provenance(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass
        | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
        split: tuple[int, int] | None = None,
    ) -> PlantriProvenance:
        """Return exact binary/request provenance for one graph-ID namespace."""
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        resolved_dual_class = QuadrangulationDualClass(dual_class)
        _validate_split(split)
        plantri = self._get_plantri()
        switches = self._PLANTRI_SWITCHES_BY_DUAL_CLASS[resolved_dual_class]
        plantri._reject_broken_quadrangulation_split(switches, split)
        return PlantriProvenance(
            version=plantri.version,
            executable_sha256=plantri.executable_sha256,
            declared_max_vertex_count=self._max_vertex_count,
            primal_vertex_count=dual_vertex_count + 2,
            switches=switches,
            split=split,
        )

    def iter_duals(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass
        | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
        max_count: int | None = None,
        num_workers: int | None = 1,
        pool_chunksize: int | None = None,
        start_method: str | None = None,
        timeout: float | None = None,
    ) -> Generator[QuarticPlaneMap, None, None]:
        """Yield source-ordered maps; timeout bounds plantri, not Python conversion."""
        if max_count is not None and (type(max_count) is not int or max_count < 0):
            raise ValueError(f"max_count: expected int >= 0 or None, got {max_count!r}")
        resolved_dual_class = self._resolve_enumeration_request(
            dual_vertex_count,
            dual_class=dual_class,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            start_method=start_method,
            timeout=timeout,
        )
        return self._iter_resolved_duals(
            dual_vertex_count,
            resolved_dual_class=resolved_dual_class,
            max_count=max_count,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            start_method=start_method,
            timeout=timeout,
        )

    def enumerate_duals(
        self,
        dual_vertex_count: int,
        *,
        max_count: int,
        dual_class: QuadrangulationDualClass
        | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
        num_workers: int | None = 1,
        pool_chunksize: int | None = None,
        start_method: str | None = None,
        timeout: float | None = None,
    ) -> PlantriEnumerationResult:
        """Materialize a bounded stream; timeout covers only active plantri generation."""
        if type(max_count) is not int or max_count < 0:
            raise ValueError(f"max_count: expected int >= 0, got {max_count!r}")
        resolved_dual_class = self._resolve_enumeration_request(
            dual_vertex_count,
            dual_class=dual_class,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            start_method=start_method,
            timeout=timeout,
        )

        enumeration_started_at = time.perf_counter()
        progress = _EnumerationProgress(started_at=enumeration_started_at)
        plane_map_iter = self._iter_resolved_duals(
            dual_vertex_count,
            resolved_dual_class=resolved_dual_class,
            max_count=max_count,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            start_method=start_method,
            timeout=timeout,
            progress=progress,
        )
        with _closing_on_error("stream cleanup failed", plane_map_iter.close):
            plane_maps = tuple(plane_map_iter)

        elapsed_s = time.perf_counter() - enumeration_started_at
        return PlantriEnumerationResult(
            graphs=plane_maps,
            time_to_first_embedding_s=progress.time_to_first_embedding_s,
            remaining_s=elapsed_s - progress.time_to_first_embedding_s,
        )

    def _get_plantri(self) -> Plantri:
        """Return the shared lazily initialized Plantri wrapper."""
        self._require_open()
        if self._plantri is None:
            self._plantri = Plantri()
        return self._plantri

    def _validate_supported_dual_vertex_count(self, dual_vertex_count: int) -> None:
        """Reject dual sizes outside this enumerator's effective capability."""
        self._require_open()
        if type(dual_vertex_count) is not int:
            message = f"dual_vertex_count must be int; got {dual_vertex_count!r}"
            raise ValueError(message)
        if dual_vertex_count < MIN_DUAL_VERTEX_COUNT:
            minimum = MIN_DUAL_VERTEX_COUNT
            message = f"dual_vertex_count unsupported: {dual_vertex_count} < {minimum}"
            raise ValueError(message)
        if dual_vertex_count > self.max_dual_vertex_count:
            maximum = self.max_dual_vertex_count
            declared = self._max_vertex_count
            twin_bound = MAX_BYTE_ENCODED_DUAL_VERTEX_COUNT
            message = f"dual_vertex_count unsupported: {dual_vertex_count} > {maximum} (declared={declared}; twin bound={twin_bound})"
            raise ValueError(message)

    def _resolve_enumeration_request(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass | str,
        num_workers: int | None,
        pool_chunksize: int | None,
        start_method: str | None,
        timeout: float | None,
    ) -> QuadrangulationDualClass:
        """Validate every enumeration control once, before resolving plantri."""
        _validate_processing_controls(num_workers, pool_chunksize)
        if start_method is not None and (
            type(start_method) is not str
            or start_method not in multiprocessing.get_all_start_methods()
        ):
            raise ValueError(f"unsupported start_method: {start_method!r}")
        resolved_dual_class = QuadrangulationDualClass(dual_class)
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        _validate_timeout(timeout)
        return resolved_dual_class

    def _iter_resolved_duals(
        self,
        dual_vertex_count: int,
        *,
        resolved_dual_class: QuadrangulationDualClass,
        max_count: int | None,
        num_workers: int | None,
        pool_chunksize: int | None,
        start_method: str | None,
        timeout: float | None,
        progress: _EnumerationProgress | None = None,
    ) -> Generator[QuarticPlaneMap, None, None]:
        """Yield structurally proved maps after resolving public controls."""
        if max_count == 0:
            return

        primal_embedding_iter = self.iter_primal_embeddings(
            dual_vertex_count,
            dual_class=resolved_dual_class,
            timeout=timeout,
        )
        selected_primal_embeddings = (
            islice(primal_embedding_iter, max_count)
            if max_count is not None
            else primal_embedding_iter
        )
        with _closing_on_error("source cleanup failed", primal_embedding_iter.close):
            prefetched_primal_embeddings = list(islice(selected_primal_embeddings, 1))
            if progress is not None and prefetched_primal_embeddings:
                progress.time_to_first_embedding_s = (
                    time.perf_counter() - progress.started_at
                )

            resolved_pool_chunksize = (
                pool_chunksize if pool_chunksize is not None else _DEFAULT_POOL_CHUNKSIZE
            )
            worker_count = _resolve_worker_count(
                num_workers,
                max_count=max_count,
                pool_chunksize=resolved_pool_chunksize,
            )
            # A context is resolved only where one could actually be used.
            mp_context = (
                _resolve_pool_context(
                    start_method,
                    controls_are_explicit=num_workers is not None
                    or start_method is not None,
                )
                if prefetched_primal_embeddings and worker_count > 1
                else None
            )

            build_plane_map = (
                _build_simple_quartic_task
                if resolved_dual_class is QuadrangulationDualClass.SIMPLE_QUARTIC
                else _build_quartic_multigraph_task
            )
            indexed_primal_embeddings = enumerate(
                chain(prefetched_primal_embeddings, selected_primal_embeddings)
            )
            if mp_context is None:
                yield from map(build_plane_map, indexed_primal_embeddings)
                return
            yield from _imap_with_pool(
                mp_context.Pool(processes=worker_count),
                build_plane_map,
                indexed_primal_embeddings,
                resolved_pool_chunksize,
            )


def iter_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    dual_class: QuadrangulationDualClass
    | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    max_count: int | None = None,
    num_workers: int | None = 1,
    pool_chunksize: int | None = None,
    start_method: str | None = None,
    timeout: float | None = None,
    enumerator: QuadrangulationEnumerator | None = None,
) -> Generator[QuarticPlaneMap, None, None]:
    """Yield source-ordered maps, closing an internally created enumerator."""
    if enumerator is not None:
        return enumerator.iter_duals(
            dual_vertex_count,
            dual_class=dual_class,
            max_count=max_count,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            start_method=start_method,
            timeout=timeout,
        )
    owned = QuadrangulationEnumerator()
    return _close_enumerator_after(
        owned.iter_duals(
            dual_vertex_count,
            dual_class=dual_class,
            max_count=max_count,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            start_method=start_method,
            timeout=timeout,
        ),
        owned,
    )


def enumerate_simple_quadrangulation_duals(
    dual_vertex_count: int,
    *,
    max_count: int,
    dual_class: QuadrangulationDualClass
    | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    num_workers: int | None = 1,
    pool_chunksize: int | None = None,
    start_method: str | None = None,
    timeout: float | None = None,
    enumerator: QuadrangulationEnumerator | None = None,
) -> PlantriEnumerationResult:
    """Materialize a bounded batch, closing an internally created enumerator."""
    if enumerator is not None:
        return enumerator.enumerate_duals(
            dual_vertex_count,
            max_count=max_count,
            dual_class=dual_class,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            start_method=start_method,
            timeout=timeout,
        )
    owned = QuadrangulationEnumerator()
    with _closing_on_error("enumerator cleanup failed", owned.close):
        result = owned.enumerate_duals(
            dual_vertex_count,
            max_count=max_count,
            dual_class=dual_class,
            num_workers=num_workers,
            pool_chunksize=pool_chunksize,
            start_method=start_method,
            timeout=timeout,
        )
    return result
