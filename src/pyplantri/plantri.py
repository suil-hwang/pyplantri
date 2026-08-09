# src/pyplantri/plantri.py
from __future__ import annotations

import os
import re
import subprocess
import sysconfig
import tempfile
from enum import Enum
from pathlib import Path
from shutil import which
from collections.abc import Iterable, Iterator
from typing import BinaryIO, Callable, Literal, NoReturn, TypeVar, cast

from .types import Embedding


def _summarize_process_text(text: str, *, limit: int = 400) -> str:
    """Collapse process output into a short single-line excerpt."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


_LINE_ORIENTED_OUTPUT_FLAGS = frozenset("ags")
_OUTPUT_FLAGS = frozenset("agsETu")
# Sorted because frozenset iteration order varies per process; the error message must not.
_LINE_ORIENTED_FLAG_HINT = "/".join(f"-{flag}" for flag in sorted(_LINE_ORIENTED_OUTPUT_FLAGS))


def _has_plantri_flag(options: list[str] | None, flags: frozenset[str]) -> bool:
    """Return whether separate or combined plantri options contain a flag."""
    return any(
        option.startswith("-") and any(char in flags for char in option[1:])
        for option in (options or [])
    )


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


def _build_tag_sort_key(tag_dir: Path) -> tuple[bool, str]:
    """Prefer build directories for the current platform, then sort by name."""
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return not tag_dir.name.endswith(platform_tag), tag_dir.name


def _is_executable(path: Path) -> bool:
    """Return whether path names a runnable executable on this platform."""
    return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))


def _find_plantri_exe() -> Path:
    """Finds the plantri executable path."""
    exe_name = "plantri.exe" if os.name == "nt" else "plantri"

    # Package bin folder (installed).
    pkg_bin = Path(__file__).parent / "bin" / exe_name
    if _is_executable(pkg_bin):
        return pkg_bin

    # scikit-build-core build folder (editable/dev): plantri.py -> pyplantri -> src -> project_root.
    project_root = Path(__file__).parent.parent.parent
    build_dir = project_root / "build"
    if build_dir.is_dir():
        for tag_dir in sorted(build_dir.iterdir(), key=_build_tag_sort_key):
            if tag_dir.is_dir():
                # Release folder (Visual Studio build).
                release_exe = tag_dir / "Release" / exe_name
                if _is_executable(release_exe):
                    return release_exe
                # MinGW/Unix build.
                direct_exe = tag_dir / exe_name
                if _is_executable(direct_exe):
                    return direct_exe

    path_exe = which(exe_name)
    if path_exe is not None:
        resolved_path_exe = Path(path_exe)
        if _is_executable(resolved_path_exe):
            return resolved_path_exe

    # Default path for error messages.
    return Path(__file__).parent / "bin" / exe_name


# Bundled plantri limit for one-byte planar_code records.
_BUNDLED_PLANTRI_MAX_N = 64

# Public SQS dual bounds; n=N-2 for a quadrangulation with N primal vertices.
MIN_DUAL_VERTEX_COUNT = 3
MAX_DUAL_VERTEX_COUNT = _BUNDLED_PLANTRI_MAX_N - 2


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


def iter_planar_code(
    stream: BinaryIO,
    *,
    expected_vertex_count: int | None = None,
    chunk_size: int = 65_536,
) -> Iterator[Embedding]:
    """Decode headerless one-byte planar_code records from ``stream``."""
    if expected_vertex_count is not None and (
        type(expected_vertex_count) is not int
        or not 1 <= expected_vertex_count <= _BUNDLED_PLANTRI_MAX_N
    ):
        raise ValueError(
            "expected_vertex_count must be None or an int in "
            f"[1, {_BUNDLED_PLANTRI_MAX_N}], got {expected_vertex_count!r}"
        )
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
    cyclic_adjacency: list[tuple[int, ...]] = []
    neighbors: list[int] = []

    for chunk in chunks:
        for value in chunk:
            if vertex_count is None:
                if value == 0:
                    raise PlanarCodeError(
                        f"record {record_index}: extended planar_code is unsupported"
                    )
                if value > _BUNDLED_PLANTRI_MAX_N:
                    raise PlanarCodeError(
                        f"record {record_index}: vertex count {value} exceeds "
                        f"bundled MAXN={_BUNDLED_PLANTRI_MAX_N}"
                    )
                if expected_vertex_count is not None and value != expected_vertex_count:
                    raise PlanarCodeError(
                        f"record {record_index}: vertex count {value} != "
                        f"expected {expected_vertex_count}"
                    )
                vertex_count = value
                continue

            if value:
                if value > vertex_count:
                    raise PlanarCodeError(
                        f"record {record_index}, vertex {len(cyclic_adjacency)}: "
                        f"neighbor {value} outside [1, {vertex_count}]"
                    )
                neighbors.append(value - 1)
                continue

            cyclic_adjacency.append(tuple(neighbors))
            neighbors = []
            if len(cyclic_adjacency) == vertex_count:
                yield tuple(cyclic_adjacency)
                record_index += 1
                vertex_count = None
                cyclic_adjacency = []

    if vertex_count is not None:
        raise PlanarCodeError(
            f"record {record_index}: truncated at vertex {len(cyclic_adjacency)}"
        )


_StreamItem = TypeVar("_StreamItem")


def _iter_nonempty_lines(stream: BinaryIO) -> Iterator[bytes]:
    """Yield stripped nonempty binary lines."""
    for raw_line in stream:
        if line := raw_line.strip():
            yield line


def _iter_growing_file_chunks(
    stream: BinaryIO,
    process: subprocess.Popen[bytes],
) -> Iterator[bytes]:
    """Follow a regular file until its writer exits, then drain it."""
    while True:
        if chunk := stream.read(65_536):
            yield chunk
        elif process.poll() is not None:
            if chunk := stream.read(65_536):
                yield chunk
            else:
                return
        else:
            try:
                process.wait(timeout=0.01)
            except subprocess.TimeoutExpired:
                pass


def _raise_executable_not_found(executable: Path) -> NoReturn:
    raise PlantriExecutableNotFoundError(f"plantri: executable not found {executable}; run 'pip install -e .' or add plantri to PATH")


def _raise_executable_not_runnable(
    executable: Path,
    error: OSError | None = None,
) -> NoReturn:
    message = f"plantri: executable is not runnable {executable}"
    if error is not None:
        message = f"{message}: {error}"
    raise PlantriError(message) from error


class Plantri:
    """Wrapper for the plantri executable."""

    def __init__(self, executable: Path | None = None) -> None:
        """Initializes Plantri with the executable path."""
        self.executable = (
            Path(executable).expanduser().resolve()
            if executable is not None
            else _find_plantri_exe()
        )
        if not self.executable.is_file():
            _raise_executable_not_found(self.executable)
        if not _is_executable(self.executable):
            _raise_executable_not_runnable(self.executable)

    def run(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        output_format: Literal["planar_code", "ascii"] = "planar_code",
    ) -> bytes:
        """Runs plantri with the given parameters."""
        cmd = self._build_command(n_vertices, options=options, output_format=output_format)

        try:
            if output_format == "planar_code":
                with tempfile.TemporaryDirectory(prefix="pyplantri-") as temp_dir:
                    output_path = Path(temp_dir) / "output.planar_code"
                    result = subprocess.run(
                        [*cmd, str(output_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        check=True,
                    )
                    try:
                        return output_path.read_bytes()
                    except OSError as exc:
                        raise PlantriError(
                            "plantri: planar_code output was not created"
                        ) from exc

            result = subprocess.run(cmd, capture_output=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode(errors="replace") if e.stderr else str(e)
            raise PlantriError(f"plantri: execution failed (exit {e.returncode}); {_summarize_process_text(stderr_text)}") from e
        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
        except OSError as e:
            _raise_executable_not_runnable(self.executable, e)

    def _build_command(
        self,
        n_vertices: int,
        *,
        options: list[str] | None,
        output_format: Literal["planar_code", "ascii"],
    ) -> list[str]:
        """Builds a plantri command line for the given options."""
        if output_format not in ("planar_code", "ascii"):
            raise ValueError(f"plantri: unsupported output_format {output_format!r}; use 'planar_code' or 'ascii'")
        _validate_n_vertices(n_vertices)
        selected_output_flags = _selected_plantri_flags(options, _OUTPUT_FLAGS)
        if "T" in selected_output_flags:
            raise ValueError("plantri: -T output is unsupported; use planar_code")
        if output_format == "ascii":
            selected_output_flags.add("a")
        if len(selected_output_flags) > 1:
            formatted_flags = ", ".join(
                f"-{flag}" for flag in sorted(selected_output_flags)
            )
            raise ValueError(f"plantri: conflicting output flags: {formatted_flags}")

        cmd = [str(self.executable)]
        if options:
            cmd.extend(options)

        # Set output format flag.
        if output_format == "ascii" and not _has_plantri_flag(options, frozenset("a")):
            cmd.append("-a")

        cmd.append(str(n_vertices))
        return cmd

    def _iter_process_stdout(
        self,
        cmd: list[str],
        decoder: Callable[[BinaryIO], Iterator[_StreamItem]],
    ) -> Iterator[_StreamItem]:
        """Decode one plantri stdout stream and own its process lifecycle."""
        # Spool stderr to disk so long streams cannot deadlock on a full pipe.
        with tempfile.TemporaryFile() as stderr_file:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                )
            except FileNotFoundError as e:
                raise PlantriExecutableNotFoundError(
                    f"plantri: executable not found {self.executable}"
                ) from e
            except OSError as e:
                _raise_executable_not_runnable(self.executable, e)

            if proc.stdout is None:
                proc.kill()
                proc.wait()
                raise PlantriError("plantri: failed to capture stdout")

            fully_consumed = False
            try:
                yield from decoder(cast(BinaryIO, proc.stdout))
                fully_consumed = True
            finally:
                proc.stdout.close()
                self._finish_process(
                    proc,
                    cast(BinaryIO, stderr_file),
                    fully_consumed=fully_consumed,
                )

    @staticmethod
    def _finish_process(
        process: subprocess.Popen[bytes],
        stderr_file: BinaryIO,
        *,
        fully_consumed: bool,
    ) -> None:
        """Check a natural exit or stop an intentionally shortened stream."""
        if fully_consumed:
            return_code = process.wait()
            if return_code != 0:
                stderr_file.seek(0)
                stderr_excerpt = _summarize_process_text(
                    stderr_file.read().decode("utf-8", errors="replace"),
                    limit=4000,
                )
                raise PlantriError(
                    f"plantri: execution failed (exit {return_code}); {stderr_excerpt}"
                ) from None
        elif process.poll() is None:
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
    ) -> Iterator[Embedding]:
        """Stream headerless one-byte planar_code as zero-based embeddings."""
        selected_output_flags = _selected_plantri_flags(options, _OUTPUT_FLAGS)
        if selected_output_flags:
            formatted_flags = ", ".join(
                f"-{flag}" for flag in sorted(selected_output_flags)
            )
            raise ValueError(
                "plantri: iter_planar_code conflicts with output flags: "
                f"{formatted_flags}"
            )

        cmd = self._build_command(
            n_vertices,
            options=options,
            output_format="planar_code",
        )
        if not _has_plantri_flag(options, frozenset("h")):
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
                raise PlantriExecutableNotFoundError(
                    f"plantri: executable not found {self.executable}"
                ) from e
            except OSError as e:
                _raise_executable_not_runnable(self.executable, e)

            output_file: BinaryIO | None = None
            fully_consumed = False
            try:
                while output_file is None:
                    try:
                        output_file = output_path.open("rb")
                    except (FileNotFoundError, PermissionError):
                        if process.poll() is not None:
                            try:
                                output_file = output_path.open("rb")
                            except (FileNotFoundError, PermissionError):
                                fully_consumed = True
                                raise PlantriError(
                                    "plantri: planar_code output was not created"
                                )
                        else:
                            try:
                                process.wait(timeout=0.01)
                            except subprocess.TimeoutExpired:
                                pass

                yield from _decode_planar_code_chunks(
                    _iter_growing_file_chunks(output_file, process),
                    expected_vertex_count=n_vertices,
                )
                fully_consumed = True
            finally:
                if output_file is not None:
                    output_file.close()
                self._finish_process(
                    process,
                    cast(BinaryIO, stderr_file),
                    fully_consumed=fully_consumed,
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
        if output_format != "ascii" and not _has_plantri_flag(options, _LINE_ORIENTED_OUTPUT_FLAGS):
            raise ValueError(f"plantri: iter_stdout_lines requires line-oriented output; use output_format='ascii' or a {_LINE_ORIENTED_FLAG_HINT} flag")
        yield from self._iter_process_stdout(cmd, _iter_nonempty_lines)

    def count_from_options(
        self,
        n_vertices: int,
        options: list[str] | None = None,
        timeout: float = 3600.0,
    ) -> int:
        """Counts graphs with arbitrary generation options via plantri ``-u``."""
        _validate_n_vertices(n_vertices)
        normalized_options = _without_plantri_flags(
            options,
            _OUTPUT_FLAGS,
        )

        try:
            result = subprocess.run(
                [str(self.executable)] + normalized_options + ["-u", str(n_vertices)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise PlantriError(f"plantri: timed out after {timeout}s for n={n_vertices}, options={normalized_options}") from e
        except FileNotFoundError as e:
            raise PlantriExecutableNotFoundError(f"plantri: executable not found {self.executable}") from e
        except OSError as e:
            _raise_executable_not_runnable(self.executable, e)

        if result.returncode != 0:
            stderr_excerpt = _summarize_process_text(result.stderr)
            stdout_excerpt = _summarize_process_text(result.stdout)
            raise PlantriError(f"plantri: count failed (exit {result.returncode}); stderr={stderr_excerpt}; stdout={stdout_excerpt}")

        # The output class name varies by mode; only the trailing verb is stable.
        for line in reversed(result.stderr.splitlines()):
            match = re.match(
                r"^\s*(\d+)\s+.*\b(?:generated|written)\b",
                line,
                flags=re.IGNORECASE,
            )
            if match:
                return int(match.group(1))

        stderr_excerpt = _summarize_process_text(result.stderr)
        stdout_excerpt = _summarize_process_text(result.stdout)
        raise PlantriError(f"plantri: count parse failed; stderr={stderr_excerpt}; stdout={stdout_excerpt}")


class QuadrangulationEnumerator:
    """Enumerates dual quartic plane multigraphs of simple quadrangulations.

    Streams the simple primal rotation system in headerless planar_code.

    - `QUARTIC_MULTIGRAPH`: `-q -c2 -m2 -h`
    - `SIMPLE_QUARTIC`: `-q -c2 -h`
    """

    _FLAGS_BY_DUAL_CLASS: dict[QuadrangulationDualClass, list[str]] = {
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: ["-q", "-c2", "-m2"],
        QuadrangulationDualClass.SIMPLE_QUARTIC: ["-q", "-c2"],
    }
    _MIN_NONEMPTY_DUAL_VERTICES: dict[QuadrangulationDualClass, int] = {
        QuadrangulationDualClass.QUARTIC_MULTIGRAPH: MIN_DUAL_VERTEX_COUNT,
        QuadrangulationDualClass.SIMPLE_QUARTIC: 6,
    }

    def __init__(self) -> None:
        """Initialize the SQS enumerator."""
        # Delay executable resolution so known-empty requests do not require plantri.
        self._plantri: Plantri | None = None

    def _get_plantri(self) -> Plantri:
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
    def _flags_for_dual_class(
        cls,
        dual_class: QuadrangulationDualClass,
    ) -> list[str]:
        return list(cls._FLAGS_BY_DUAL_CLASS[dual_class])

    @classmethod
    def _min_nonempty_dual_vertices(
        cls,
        dual_class: QuadrangulationDualClass,
    ) -> int:
        return cls._MIN_NONEMPTY_DUAL_VERTICES[dual_class]

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
        if dual_vertex_count < self._min_nonempty_dual_vertices(resolved_dual_class):
            return
        # Euler's formula and quadrilateral faces give V_primal = V_dual + 2.
        primal_vertex_count = dual_vertex_count + 2
        yield from self._get_plantri().iter_planar_code(
            primal_vertex_count,
            self._flags_for_dual_class(resolved_dual_class),
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
        if dual_vertex_count < self._min_nonempty_dual_vertices(resolved_dual_class):
            return 0
        primal_vertex_count = dual_vertex_count + 2
        return self._get_plantri().count_from_options(
            primal_vertex_count,
            options=self._flags_for_dual_class(resolved_dual_class),
        )
