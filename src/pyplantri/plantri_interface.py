# src/pyplantri/plantri_interface.py
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import BinaryIO, Literal, cast

from .types import Embedding


PlantriOutput = Literal[
    "planar_code",
    "ascii",
    "graph6",
    "sparse6",
    "edge_code",
    "none",
]
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
# Translate separators to 255 and labels 1..255 to 0..254 in one C-level pass.
_ZERO_BASE_TRANSLATION = bytes((255, *range(255)))
_TRANSLATED_ROW_SEPARATOR = b"\xff"
_IO_CHUNK_SIZE = 1 << 16
_FILE_POLL_INTERVAL_S = 0.01
_PROCESS_TERMINATE_TIMEOUT_S = 5.0
_C_INT_MAX = 2_147_483_647

# One-byte planar_code supports 1..255 labels; bundled input is smaller.
_ONE_BYTE_PLANAR_CODE_MAX_N = 255
_BUNDLED_PLANTRI_MAX_N = 64

# Public SQS dual bounds; n=N-2 for a quadrangulation with N primal vertices.
MIN_DUAL_VERTEX_COUNT = 3
MAX_DUAL_VERTEX_COUNT = _BUNDLED_PLANTRI_MAX_N - 2

def _summarize_process_text(text: str | bytes, *, limit: int = 400) -> str:
    """Decode and collapse process output into a bounded single-line excerpt."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


def _normalize_switches(switches: Sequence[str]) -> tuple[str, ...]:
    """Validate graph-selection switches and reject positional/output tokens."""
    if isinstance(switches, (str, bytes)):
        raise ValueError(f"plantri: switches must be a sequence; got {switches!r}")
    normalized = tuple(switches)
    if any(type(option) is not str for option in normalized):
        raise ValueError(f"plantri: switches must contain only strings; got {switches!r}")
    positional = [option for option in normalized if option == "-" or not option.startswith("-")]
    if positional:
        raise ValueError(f"plantri: switches must not contain positional values; got {positional}")
    selected_output_flags = {
        char
        for option in normalized
        for char in option[1:]
        if char in _OUTPUT_SWITCH_CHARS
    }
    if "T" in selected_output_flags:
        raise ValueError("plantri: -T output is unsupported")
    if selected_output_flags:
        formatted = ", ".join(f"-{flag}" for flag in sorted(selected_output_flags))
        raise ValueError(f"plantri: output switches must use output_format; found {formatted}")
    if any("h" in option[1:] for option in normalized):
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
        raise ValueError(f"plantri: invalid split {split!r}; expected 0 <= residue < modulus <= {_C_INT_MAX}")


def _validate_n_vertices(n_vertices: int) -> None:
    """Require a positive integer vertex count for public plantri commands."""
    if type(n_vertices) is not int or n_vertices <= 0:
        raise ValueError(f"plantri: n_vertices must be a positive int; got {n_vertices!r}")


def _resolve_bundled_plantri_executable() -> Path:
    """Resolve the bundled executable candidate for the active installation."""
    exe_name = "plantri.exe" if os.name == "nt" else "plantri"
    package_dir = Path(__file__).parent
    fallback_executable = package_dir / "bin" / exe_name

    # Editable installs may map native resources outside the Python source tree.
    resource_executable = files("pyplantri").joinpath("bin").joinpath(exe_name)
    if isinstance(resource_executable, Path) and resource_executable.is_file():
        return resource_executable
    return fallback_executable


class PlantriError(Exception):
    """Plantri execution failure."""


class PlantriExecutableNotFoundError(PlantriError, FileNotFoundError):
    """Plantri executable could not be found."""


class PlanarCodeError(ValueError):
    """Malformed or unsupported planar_code input."""


class QuadrangulationDualClass(str, Enum):
    """Plane-dual families selected by supported simple-quadrangulation modes.

    ``QUARTIC_MULTIGRAPH`` uses primal flags ``-q -c2 -m2`` and denotes
    loop-free, 4-regular, 4-edge-connected plane multigraphs. Parallel edges
    are permitted, and simple members are included.

    ``SIMPLE_QUARTIC`` uses primal flags ``-q -c2`` and restricts the primal
    quadrangulation to minimum degree at least 3; its duals are simple,
    4-regular, 4-edge-connected plane graphs.
    """

    QUARTIC_MULTIGRAPH = "quartic_multigraph"
    SIMPLE_QUARTIC = "simple_quartic"


def _validate_expected_vertex_count(expected_vertex_count: int | None) -> None:
    """Validate an optional one-byte planar_code vertex count."""
    if expected_vertex_count is not None and (
        type(expected_vertex_count) is not int
        or not 1 <= expected_vertex_count <= _ONE_BYTE_PLANAR_CODE_MAX_N
    ):
        raise ValueError(f"expected_vertex_count must be None or 1..{_ONE_BYTE_PLANAR_CODE_MAX_N} int: {expected_vertex_count!r}")


def _validate_expected_edge_count(expected_edge_count: int | None) -> None:
    """Validate an optional fixed planar_code edge count."""
    if expected_edge_count is not None and (
        type(expected_edge_count) is not int or expected_edge_count < 0
    ):
        raise ValueError(f"expected_edge_count must be None or a non-negative int: {expected_edge_count!r}")


def iter_planar_code(
    stream: BinaryIO,
    *,
    expected_vertex_count: int | None = None,
    expected_edge_count: int | None = None,
    chunk_size: int = 65_536,
) -> Iterator[Embedding]:
    """Decode headerless one-byte planar_code, with an optional fixed-size fast path."""
    _validate_expected_vertex_count(expected_vertex_count)
    _validate_expected_edge_count(expected_edge_count)
    if expected_edge_count is not None and expected_vertex_count is None:
        raise ValueError("expected_edge_count requires expected_vertex_count")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")

    yield from _decode_planar_code_records(
        iter(lambda: stream.read(chunk_size), b""),
        expected_vertex_count=expected_vertex_count,
        expected_edge_count=expected_edge_count,
    )


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
                    raise PlanarCodeError(f"record {record_index}: extended planar_code is unsupported")
                if expected_vertex_count is not None and value != expected_vertex_count:
                    raise PlanarCodeError(f"record {record_index}: vertex count {value} != expected {expected_vertex_count}")
                vertex_count = value
            elif value:
                if value > vertex_count:
                    raise PlanarCodeError(f"record {record_index}, vertex {len(adjacency_rows)}: neighbor {value} outside [1, {vertex_count}]")
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
        raise PlanarCodeError(f"record {record_index}: truncated at vertex {len(adjacency_rows)}")


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
                raise PlanarCodeError(f"record {record_index}: vertex count {record[0]} != expected {vertex_count}")

            body = record[1:]
            rows = body.translate(_ZERO_BASE_TRANSLATION).split(
                _TRANSLATED_ROW_SEPARATOR
            )
            if len(rows) != vertex_count + 1 or rows[-1]:
                raise PlanarCodeError(f"record {record_index}: invalid adjacency separators for fixed-size record")
            rows.pop()
            if body and max(body) > vertex_count:
                for vertex, row in enumerate(body.split(b"\0")):
                    if row and (neighbor := max(row)) > vertex_count:
                        raise PlanarCodeError(f"record {record_index}, vertex {vertex}: neighbor {neighbor} outside [1, {vertex_count}]")
            yield tuple(map(tuple, rows))
            record_index += 1
        carry = data[complete_size:]

    if carry:
        raise PlanarCodeError(f"record {record_index}: truncated fixed-size record ({len(carry)}/{record_size} bytes)")


class Plantri:
    """Wrapper for the plantri executable."""

    def __init__(self, executable: str | Path | None = None) -> None:
        """Initializes Plantri with the executable path."""
        candidate = (
            Path(executable).expanduser().resolve()
            if executable is not None
            else _resolve_bundled_plantri_executable()
        )
        self.executable = candidate.resolve()
        if not self.executable.is_file():
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}")

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
                raise PlantriError(f"plantri: {output_format} output was not created") from e
            except OSError as e:
                raise PlantriError(f"plantri: failed to read {output_format} output: {_summarize_process_text(str(e))}") from e

    def _run_checked(
        self,
        cmd: list[str],
        *,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run one bounded command without a shell and translate process failures."""
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise PlantriError(f"plantri: timed out after {timeout}s") from e
        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
        except OSError as e:
            raise PlantriError(f"plantri: executable is not runnable {self.executable}: {_summarize_process_text(str(e))}") from e
        if result.returncode != 0:
            detail = (_summarize_process_text(result.stderr) or _summarize_process_text(result.stdout) or "no output")
            raise PlantriError(f"plantri: execution failed (exit {result.returncode}); {detail}")
        return result

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
        if output_format not in _OUTPUT_SWITCH_BY_FORMAT:
            raise ValueError(f"plantri: unsupported output_format {output_format!r}")
        _validate_n_vertices(n_vertices)
        normalized_switches = _normalize_switches(switches)
        _validate_split(split)
        if type(headerless) is not bool:
            raise ValueError(f"plantri: headerless must be bool; got {headerless!r}")
        if headerless and output_format != "planar_code":
            raise ValueError("plantri: headerless output is supported only for planar_code")
        if output_path is not None and output_format == "none":
            raise ValueError("plantri: output_path is incompatible with output_format='none'")

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

    def _start_stream_process(
        self,
        cmd: list[str],
        *,
        stdout: int,
        stderr: BinaryIO,
    ) -> subprocess.Popen[bytes]:
        """Start one streaming process and normalize executable failures."""
        try:
            return subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
        except OSError as e:
            raise PlantriError(f"plantri: executable is not runnable {self.executable}: {_summarize_process_text(str(e))}") from e

    @staticmethod
    def _read_planar_code_chunk(output_file: BinaryIO) -> bytes:
        """Read one planar-code chunk and normalize storage failures."""
        try:
            return output_file.read(_IO_CHUNK_SIZE)
        except OSError as e:
            raise PlantriError(f"plantri: failed to read planar_code output: {_summarize_process_text(str(e))}") from e

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
                    records = iter(stdout)
                    while True:
                        try:
                            raw_line = next(records)
                        except StopIteration:
                            stream_exhausted = True
                            break
                        except OSError as e:
                            raise PlantriError(f"plantri: failed to read text output: {_summarize_process_text(str(e))}") from e
                        line = raw_line.rstrip(b"\r\n")
                        if line:
                            yield line
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
                stderr_excerpt = _summarize_process_text(stderr_file.read(), limit=4000)
                raise PlantriError(f"plantri: execution failed (exit {return_code}); {stderr_excerpt}") from None
            return

        process.terminate()
        try:
            process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def iter_planar_code(
        self,
        n_vertices: int,
        switches: Sequence[str] = (),
        *,
        expected_vertex_count: int | None = None,
        expected_edge_count: int | None = None,
        split: tuple[int, int] | None = None,
    ) -> Iterator[Embedding]:
        """Stream headerless records, using fixed-size decoding when |V| and |E| are known."""
        _validate_expected_vertex_count(expected_vertex_count)
        _validate_expected_edge_count(expected_edge_count)
        if expected_edge_count is not None and expected_vertex_count is None:
            raise ValueError("expected_edge_count requires expected_vertex_count")

        with tempfile.TemporaryDirectory(prefix="pyplantri-") as temp_dir, tempfile.TemporaryFile() as stderr_file:
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

            stream_exhausted = False
            writer_exited = False
            try:
                while True:
                    try:
                        output_file = output_path.open("rb")
                        break
                    except FileNotFoundError:
                        if writer_exited:
                            raise PlantriError("plantri: planar_code output was not created")
                        writer_exited = process.poll() is not None
                    except PermissionError as e:
                        if writer_exited:
                            raise PlantriError(f"plantri: failed to open planar_code output: {_summarize_process_text(str(e))}") from e
                        writer_exited = process.poll() is not None
                    except OSError as e:
                        raise PlantriError(f"plantri: failed to open planar_code output: {_summarize_process_text(str(e))}") from e
                    if not writer_exited:
                        time.sleep(_FILE_POLL_INTERVAL_S)

                with output_file:
                    def read_chunk() -> bytes:
                        while True:
                            if chunk := self._read_planar_code_chunk(output_file):
                                return chunk
                            if process.poll() is not None:
                                return self._read_planar_code_chunk(output_file)
                            time.sleep(_FILE_POLL_INTERVAL_S)

                    yield from _decode_planar_code_records(
                        iter(read_chunk, b""),
                        expected_vertex_count=expected_vertex_count,
                        expected_edge_count=expected_edge_count,
                    )
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
            raise ValueError(f"plantri: line output requires ascii, graph6, or sparse6; got {output_format!r}")
        cmd = self._build_command(
            n_vertices,
            switches=switches,
            output_format=output_format,
            split=split,
        )
        yield from self._iter_text_records(cmd)

    def count(
        self,
        n_vertices: int,
        switches: Sequence[str] = (),
        timeout: float | None = None,
        *,
        split: tuple[int, int] | None = None,
    ) -> int:
        """Count generated objects selected by switches and one optional split."""
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
                and any(field.lower().rstrip(";") == "generated" for field in fields[1:])
            ):
                return int(fields[0])

        detail = _summarize_process_text(result.stderr) or "no output"
        raise PlantriError(f"plantri: count parse failed: {detail}")


class QuadrangulationEnumerator:
    """Enumerate supported simple quadrangulations by plane-dual family.

    ``iter_primal_embeddings()`` yields zero-based exterior-view-CW primal rotation
    systems. A ``dual_vertex_count`` of ``k`` corresponds to ``k + 2`` primal
    vertices.

    - ``QUARTIC_MULTIGRAPH``: ``-q -c2 -m2``
    - ``SIMPLE_QUARTIC``: ``-q -c2``
    """

    _PLANTRI_SWITCHES_BY_DUAL_CLASS: dict[
        QuadrangulationDualClass, tuple[str, ...]
    ] = {
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

    def _get_plantri(self) -> Plantri:
        """Return the shared lazily initialized Plantri wrapper."""
        if self._plantri is None:
            self._plantri = Plantri()
        return self._plantri

    @staticmethod
    def _validate_supported_dual_vertex_count(dual_vertex_count: int) -> None:
        """Reject dual sizes outside the bundled plantri count range."""
        if type(dual_vertex_count) is not int:
            raise ValueError(f"dual_vertex_count must be an integer; got {dual_vertex_count!r}")
        if dual_vertex_count < MIN_DUAL_VERTEX_COUNT:
            raise ValueError(f"dual_vertex_count unsupported: {dual_vertex_count} < {MIN_DUAL_VERTEX_COUNT}")
        if dual_vertex_count > MAX_DUAL_VERTEX_COUNT:
            raise ValueError(f"dual_vertex_count unsupported: {dual_vertex_count} > {MAX_DUAL_VERTEX_COUNT} (bundled plantri MAXN={_BUNDLED_PLANTRI_MAX_N})")

    def iter_primal_embeddings(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> Iterator[Embedding]:
        """Yield zero-based exterior-view-CW simple primal embeddings."""
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
        )

    def count(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
        timeout: float | None = None,
    ) -> int:
        """Count plane-map isomorphism classes, identifying mirror images."""
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
        )
