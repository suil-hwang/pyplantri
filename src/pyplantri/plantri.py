# src/pyplantri/plantri.py
from __future__ import annotations

import os
import re
import subprocess
import sysconfig
import tempfile
from contextlib import ExitStack, suppress
from enum import Enum
from importlib.resources import files
from pathlib import Path
from shutil import which
from collections.abc import Iterable, Iterator
from typing import BinaryIO, Literal, cast

from .types import Embedding


_LINE_ORIENTED_OUTPUT_FLAGS = frozenset("ags")
_OUTPUT_FLAGS = frozenset("agsETu")
# Sorted because frozenset iteration order varies per process; the error message must not.
_LINE_ORIENTED_FLAG_HINT = "/".join(f"-{flag}" for flag in sorted(_LINE_ORIENTED_OUTPUT_FLAGS))

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


def _selected_plantri_flags(
    options: list[str] | None,
    flags: frozenset[str],
) -> set[str]:
    """Return selected boolean plantri flags, including combined options."""
    return {
        char
        for option in (options or [])
        if option.startswith("-")
        for char in option[1:]
        if char in flags
    }


def _without_plantri_flags(
    options: list[str] | None,
    flags: frozenset[str],
) -> list[str]:
    """Remove boolean plantri flags while preserving combined option tokens."""
    normalized: list[str] = []
    for option in options or []:
        if not option.startswith("-") or option == "-":
            normalized.append(option)
            continue
        remaining = "".join(char for char in option[1:] if char not in flags)
        if remaining:
            normalized.append(f"-{remaining}")
    return normalized


def _validate_n_vertices(n_vertices: int) -> None:
    """Require a positive integer vertex count for public plantri commands."""
    if type(n_vertices) is not int or n_vertices <= 0:
        raise ValueError(f"plantri: n_vertices must be a positive int; got {n_vertices!r}")


def _is_executable_candidate(path: Path) -> bool:
    """Return whether path is an executable candidate on this platform."""
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def _find_plantri_exe() -> Path:
    """Find the active package, build, or PATH plantri executable."""
    exe_name = "plantri.exe" if os.name == "nt" else "plantri"
    package_dir = Path(__file__).parent
    fallback_executable = package_dir / "bin" / exe_name

    # The active editable or wheel resource must precede stale local build tags.
    resource_executable = files("pyplantri").joinpath("bin").joinpath(exe_name)
    if isinstance(resource_executable, Path) and _is_executable_candidate(resource_executable):
        return resource_executable
    if _is_executable_candidate(fallback_executable):
        return fallback_executable

    # scikit-build-core build folder (editable/dev): plantri.py -> pyplantri -> src -> project_root.
    project_root = package_dir.parent.parent
    build_dir = project_root / "build"
    if build_dir.is_dir():
        platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        tag_dirs = (
            path
            for path in build_dir.iterdir()
            if path.is_dir() and path.name.endswith(f"-{platform_tag}")
        )
        for tag_dir in sorted(tag_dirs, key=lambda path: path.name):
            for candidate in (tag_dir / "Release" / exe_name, tag_dir / exe_name):
                if _is_executable_candidate(candidate):
                    return candidate

    path_exe = which(exe_name)
    if path_exe is not None:
        resolved_path_exe = Path(path_exe)
        if _is_executable_candidate(resolved_path_exe):
            return resolved_path_exe

    return fallback_executable


class PlantriError(Exception):
    """Plantri execution failure."""


class PlantriExecutableNotFoundError(PlantriError, FileNotFoundError):
    """Plantri executable could not be found."""


class PlanarCodeError(ValueError):
    """Malformed or unsupported planar_code input."""


class QuadrangulationDualClass(str, Enum):
    """Dual graph classes available from plantri quadrangulation modes."""

    QUARTIC_MULTIGRAPH = "quartic_multigraph"
    SIMPLE_QUARTIC = "simple_quartic"


def _validate_expected_vertex_count(expected_vertex_count: int | None) -> None:
    """Validate an optional one-byte planar_code record size."""
    if expected_vertex_count is not None and (
        type(expected_vertex_count) is not int
        or not 1 <= expected_vertex_count <= _ONE_BYTE_PLANAR_CODE_MAX_N
    ):
        raise ValueError(f"expected_vertex_count must be None or 1..{_ONE_BYTE_PLANAR_CODE_MAX_N} int: {expected_vertex_count!r}")


def iter_planar_code(
    stream: BinaryIO,
    *,
    expected_vertex_count: int | None = None,
    chunk_size: int = 65_536,
) -> Iterator[Embedding]:
    """Decode headerless planar_code records with one-byte sizes in 1..255."""
    _validate_expected_vertex_count(expected_vertex_count)
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")

    yield from _decode_planar_code_chunks(
        iter(lambda: stream.read(chunk_size), b""),
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


class Plantri:
    """Wrapper for the plantri executable."""

    def __init__(self, executable: Path | None = None) -> None:
        """Initializes Plantri with the executable path."""
        candidate = (
            Path(executable).expanduser().resolve()
            if executable is not None
            else _find_plantri_exe()
        )
        self.executable = candidate.resolve()
        if not self.executable.is_file():
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}; run 'pip install -e .' or add plantri to PATH")
        if not _is_executable_candidate(self.executable):
            raise PlantriError(f"plantri: executable is not runnable {self.executable}")

    def run(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> bytes:
        """Runs plantri with the given parameters."""
        cmd = self._build_command(n_vertices, options=options, output_format=output_format)

        with ExitStack() as stack:
            output_path: Path | None = None
            if output_format == "planar_code":
                temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="pyplantri-"))
                output_path = Path(temp_dir) / "output.planar_code"

            try:
                result = subprocess.run(
                    [*cmd, str(output_path)] if output_path is not None else cmd,
                    stdout=subprocess.DEVNULL if output_path is not None else subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise PlantriError(f"plantri: execution failed (exit {e.returncode}); {_summarize_process_text(e.stderr or str(e))}") from e
            except FileNotFoundError as e:
                raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
            except OSError as e:
                raise PlantriError(f"plantri: executable is not runnable {self.executable}: {_summarize_process_text(str(e))}") from e

            if output_path is None:
                return cast(bytes, result.stdout)
            try:
                return output_path.read_bytes()
            except FileNotFoundError as e:
                raise PlantriError("plantri: planar_code output was not created") from e
            except OSError as e:
                raise PlantriError(f"plantri: failed to read planar_code output: {_summarize_process_text(str(e))}") from e

    def _build_command(
        self,
        n_vertices: int,
        *,
        options: list[str] | None,
        output_format: Literal["planar_code", "ascii"],
    ) -> list[str]:
        """Builds a plantri command line for the given options."""
        if output_format not in ("planar_code", "ascii"):
            raise ValueError(f"plantri: unsupported output_format {output_format!r}")
        _validate_n_vertices(n_vertices)
        requested_output_flags = _selected_plantri_flags(options, _OUTPUT_FLAGS)
        if "T" in requested_output_flags:
            raise ValueError("plantri: -T output is unsupported")
        effective_output_flags = requested_output_flags | (
            {"a"} if output_format == "ascii" else set()
        )
        if len(effective_output_flags) > 1:
            formatted_flags = ", ".join(f"-{flag}" for flag in sorted(effective_output_flags))
            raise ValueError(f"plantri: conflicting output flags: {formatted_flags}")

        cmd = [str(self.executable)]
        if options:
            cmd.extend(options)

        if output_format == "ascii" and "a" not in requested_output_flags:
            cmd.append("-a")

        cmd.append(str(n_vertices))
        return cmd

    def _iter_nonempty_stdout_lines(
        self,
        cmd: list[str],
    ) -> Iterator[bytes]:
        """Yield nonempty output lines and own the process lifecycle."""
        # Spool stderr to disk so long streams cannot deadlock on a full pipe.
        with tempfile.TemporaryFile() as stderr_file:
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                )
            except FileNotFoundError as e:
                raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
            except OSError as e:
                raise PlantriError(f"plantri: executable is not runnable {self.executable}: {_summarize_process_text(str(e))}") from e

            if process.stdout is None:
                process.kill()
                process.wait()
                raise PlantriError("plantri: failed to capture stdout")

            with cast(BinaryIO, process.stdout) as stdout:
                stream_exhausted = False
                try:
                    for raw_line in stdout:
                        if line := raw_line.strip():
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
                stderr_excerpt = _summarize_process_text(stderr_file.read(), limit=4000)
                raise PlantriError(f"plantri: execution failed (exit {return_code}); {stderr_excerpt}") from None
            return

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def iter_planar_code(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        *,
        expected_vertex_count: int | None = None,
    ) -> Iterator[Embedding]:
        """Stream records, optionally asserting their output vertex count."""
        _validate_expected_vertex_count(expected_vertex_count)
        selected_output_flags = _selected_plantri_flags(options, _OUTPUT_FLAGS)
        if selected_output_flags:
            formatted_flags = ", ".join(f"-{flag}" for flag in sorted(selected_output_flags))
            raise ValueError(f"plantri: iter_planar_code conflicts with output flags: {formatted_flags}")

        cmd = self._build_command(
            n_vertices,
            options=options,
            output_format="planar_code",
        )
        if not _selected_plantri_flags(options, frozenset("h")):
            cmd.insert(-1, "-h")

        with (
            tempfile.TemporaryDirectory(prefix="pyplantri-") as temp_dir,
            tempfile.TemporaryFile() as stderr_file,
        ):
            output_path = Path(temp_dir) / "output.planar_code"
            cmd.append(str(output_path))
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                )
            except FileNotFoundError as e:
                raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
            except OSError as e:
                raise PlantriError(f"plantri: executable is not runnable {self.executable}: {_summarize_process_text(str(e))}") from e

            stream_exhausted = False
            writer_exited = False
            try:
                while True:
                    try:
                        output_file = output_path.open("rb")
                        break
                    except (FileNotFoundError, PermissionError):
                        if writer_exited:
                            raise PlantriError("plantri: planar_code output was not created")
                        with suppress(subprocess.TimeoutExpired):
                            process.wait(timeout=0.01)
                            writer_exited = True

                with output_file:
                    def read_chunk() -> bytes:
                        while not (chunk := output_file.read(65_536)):
                            with suppress(subprocess.TimeoutExpired):
                                process.wait(timeout=0.01)
                                return output_file.read(65_536)
                        return chunk

                    yield from _decode_planar_code_chunks(iter(read_chunk, b""), expected_vertex_count=expected_vertex_count)
                    stream_exhausted = True
            finally:
                self._finalize_stream_process(
                    process,
                    cast(BinaryIO, stderr_file),
                    stream_exhausted=stream_exhausted,
                )

    def iter_stdout_lines(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> Iterator[bytes]:
        """Stream non-empty stdout lines for line-oriented plantri output."""
        cmd = self._build_command(n_vertices, options=options, output_format=output_format)
        # Binary planar_code may contain newline bytes, so streaming requires a line format.
        if output_format != "ascii" and not _selected_plantri_flags(options, _LINE_ORIENTED_OUTPUT_FLAGS):
            raise ValueError(f"plantri: line output requires output_format='ascii' or {_LINE_ORIENTED_FLAG_HINT}")
        yield from self._iter_nonempty_stdout_lines(cmd)

    def count_from_options(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        timeout: float = 3600.0,
    ) -> int:
        """Count graphs selected by plantri switch options via ``-u``."""
        _validate_n_vertices(n_vertices)
        count_options = _without_plantri_flags(
            options,
            _OUTPUT_FLAGS,
        )

        try:
            result = subprocess.run(
                [str(self.executable), *count_options, "-u", str(n_vertices)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise PlantriError(f"plantri: timed out after {timeout}s for n={n_vertices}, options={count_options}") from e
        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
        except OSError as e:
            raise PlantriError(f"plantri: executable is not runnable {self.executable}: {_summarize_process_text(str(e))}") from e

        if result.returncode != 0:
            detail = _summarize_process_text(result.stderr) or _summarize_process_text(result.stdout) or "no output"
            raise PlantriError(f"plantri: count failed (exit {result.returncode}): {detail}")

        # With -u, the final summary is always "<count> <class> generated".
        if generated_counts := re.findall(r"(?im)^\s*(\d+)\s+.*\bgenerated\b", result.stderr):
            return int(generated_counts[-1])

        detail = _summarize_process_text(result.stderr) or _summarize_process_text(result.stdout) or "no output"
        raise PlantriError(f"plantri: count parse failed: {detail}")


class QuadrangulationEnumerator:
    """Enumerates dual quartic plane multigraphs of simple quadrangulations.

    Streams the simple primal rotation system in headerless planar_code.

    - `QUARTIC_MULTIGRAPH`: `-q -c2 -m2 -h`
    - `SIMPLE_QUARTIC`: `-q -c2 -h`
    """

    _PLANTRI_OPTIONS_BY_DUAL_CLASS: dict[
        QuadrangulationDualClass, tuple[str, ...]
    ] = {
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: ("-q", "-c2", "-m2"),
        QuadrangulationDualClass.SIMPLE_QUARTIC: ("-q", "-c2"),
    }
    _MIN_NONEMPTY_DUAL_VERTEX_COUNT_BY_CLASS: dict[QuadrangulationDualClass, int] = {
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: MIN_DUAL_VERTEX_COUNT,
        QuadrangulationDualClass.SIMPLE_QUARTIC: 6,
    }

    def __init__(self) -> None:
        """Initialize the SQS enumerator."""
        # Delay executable resolution so known-empty requests do not require plantri.
        self._plantri: Plantri | None = None

    def _get_plantri(self) -> Plantri:
        """Return the shared lazily initialized Plantri wrapper."""
        if self._plantri is None:
            self._plantri = Plantri()
        return self._plantri

    @staticmethod
    def _normalize_dual_class(
        dual_class: QuadrangulationDualClass | str,
    ) -> QuadrangulationDualClass:
        if isinstance(dual_class, QuadrangulationDualClass):
            return dual_class
        try:
            return QuadrangulationDualClass(dual_class)
        except ValueError as exc:
            raise ValueError(f"unsupported quadrangulation dual_class: {dual_class!r}") from exc

    @classmethod
    def _min_nonempty_dual_vertex_count(
        cls,
        dual_class: QuadrangulationDualClass,
    ) -> int:
        return cls._MIN_NONEMPTY_DUAL_VERTEX_COUNT_BY_CLASS[dual_class]

    @staticmethod
    def _validate_supported_dual_vertex_count(dual_vertex_count: int) -> None:
        """Reject dual sizes outside the bundled plantri count range."""
        if type(dual_vertex_count) is not int:
            raise ValueError(f"dual_vertex_count must be an integer; got {dual_vertex_count!r}")
        if dual_vertex_count < MIN_DUAL_VERTEX_COUNT:
            raise ValueError(f"dual_vertex_count unsupported: {dual_vertex_count} < {MIN_DUAL_VERTEX_COUNT}")
        if dual_vertex_count > MAX_DUAL_VERTEX_COUNT:
            raise ValueError(f"dual_vertex_count unsupported: {dual_vertex_count} > {MAX_DUAL_VERTEX_COUNT} (bundled plantri MAXN={_BUNDLED_PLANTRI_MAX_N})")

    def iter_embeddings(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> Iterator[Embedding]:
        """Yield zero-based exterior-view-CW simple primal embeddings."""
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        resolved_dual_class = self._normalize_dual_class(dual_class)
        if dual_vertex_count < self._min_nonempty_dual_vertex_count(resolved_dual_class):
            return
        # Euler's formula and quadrilateral faces give V_primal = V_dual + 2.
        primal_vertex_count = dual_vertex_count + 2
        yield from self._get_plantri().iter_planar_code(
            primal_vertex_count,
            list(self._PLANTRI_OPTIONS_BY_DUAL_CLASS[resolved_dual_class]),
            expected_vertex_count=primal_vertex_count,
        )

    def count(
        self,
        dual_vertex_count: int,
        *,
        dual_class: QuadrangulationDualClass | str = QuadrangulationDualClass.QUARTIC_MULTIGRAPH,
    ) -> int:
        """Count plane-map isomorphism classes, identifying mirror images."""
        self._validate_supported_dual_vertex_count(dual_vertex_count)
        resolved_dual_class = self._normalize_dual_class(dual_class)
        if dual_vertex_count < self._min_nonempty_dual_vertex_count(resolved_dual_class):
            return 0
        primal_vertex_count = dual_vertex_count + 2
        return self._get_plantri().count_from_options(
            primal_vertex_count,
            options=list(self._PLANTRI_OPTIONS_BY_DUAL_CLASS[resolved_dual_class]),
        )
